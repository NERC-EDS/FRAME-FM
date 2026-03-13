#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from framefm_core import Store, Batch, fourier


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FoundationDataset(Dataset):
    def __init__(
            self,
            store: Store,
            tokens_per_sample: int = 256,
            samples_per_epoch: int = 4000,
            mask_ratio: float = 0.20,
            min_masked_tokens: int = 8,
            static_fraction: float = 0.25,
            max_dt_days: float = 14.0,
            max_dist_deg: float = 2.0,
            seed: int = 0,
    ):
        self.s = store
        self.T = int(tokens_per_sample)
        self.samples = int(samples_per_epoch)
        self.mask_ratio = float(mask_ratio)
        self.min_masked_tokens = int(min_masked_tokens)
        self.static_fraction = float(static_fraction)
        self.max_dt_days = float(max_dt_days)
        self.max_dist_deg = float(max_dist_deg)
        self.rng = np.random.default_rng(seed)

        time_days = self.s.coord[self.s.pretrain_idx, 2]
        self.dynamic_idx = self.s.pretrain_idx[time_days >= 0]
        self.static_idx = self.s.pretrain_idx[time_days < 0]
        self.categorical_idx = self.s.pretrain_idx[self.s.is_categorical[self.s.pretrain_idx] == 1]
        self.numeric_idx = self.s.pretrain_idx[self.s.is_categorical[self.s.pretrain_idx] == 0]

        if len(self.dynamic_idx) == 0:
            raise RuntimeError("No dynamic pretraining rows available.")

    def __len__(self):
        return self.samples

    def _rand_from(self, pool: np.ndarray, n: int, replace: bool = True) -> np.ndarray:
        if n <= 0 or len(pool) == 0:
            return np.empty((0,), dtype=np.int64)
        if (not replace) and (n <= len(pool)):
            sel = self.rng.choice(len(pool), size=n, replace=False)
            return pool[sel]
        return pool[self.rng.integers(0, len(pool), size=n)]

    def _choose_anchor(self) -> int:
        # Occasionally anchor on a categorical token if available.
        if len(self.categorical_idx) > 0 and self.rng.random() < 0.15:
            return int(self.categorical_idx[self.rng.integers(0, len(self.categorical_idx))])
        return int(self.dynamic_idx[self.rng.integers(0, len(self.dynamic_idx))])

    def _local_dynamic_candidates(self, anchor_idx: int, probe_n: int = 6000) -> np.ndarray:
        if len(self.dynamic_idx) == 0:
            return np.empty((0,), dtype=np.int64)

        a = self.s.coord[anchor_idx]
        lat0, lon0, t0 = float(a[0]), float(a[1]), float(a[2])

        probe_n = min(probe_n, len(self.dynamic_idx))
        cand = self.dynamic_idx[self.rng.integers(0, len(self.dynamic_idx), size=probe_n)]
        cc = self.s.coord[cand]

        spatial = np.abs(cc[:, 0] - lat0) + np.abs(cc[:, 1] - lon0)
        temporal = np.abs(cc[:, 2] - t0)
        keep = (spatial <= self.max_dist_deg) & (temporal <= self.max_dt_days)
        out = cand[keep]
        out = out[out != anchor_idx]
        return out

    def _sample_ids(self, anchor_idx: int) -> np.ndarray:
        k_total = self.T
        k_ctx = max(0, k_total - 1)

        local_dyn = self._local_dynamic_candidates(anchor_idx)
        n_static = min(len(self.static_idx), max(0, int(round(k_ctx * self.static_fraction))))
        n_dyn = max(0, k_ctx - n_static)

        if len(local_dyn) >= n_dyn and n_dyn > 0:
            sel = self.rng.choice(len(local_dyn), size=n_dyn, replace=False)
            picked_dyn = local_dyn[sel]
        else:
            picked_dyn = local_dyn
            need = n_dyn - len(picked_dyn)
            if need > 0:
                picked_dyn = np.concatenate([picked_dyn, self._rand_from(self.dynamic_idx, need)], axis=0)

        picked_static = self._rand_from(self.static_idx, n_static)
        ids = np.concatenate([[anchor_idx], picked_dyn, picked_static], axis=0)

        if len(ids) < k_total:
            need = k_total - len(ids)
            ids = np.concatenate([ids, self._rand_from(self.s.pretrain_idx, need)], axis=0)

        ids = ids[:k_total]

        # Light dedupe while keeping anchor.
        keep = np.ones(len(ids), dtype=bool)
        seen = {int(ids[0])}
        for i in range(1, len(ids)):
            x = int(ids[i])
            if x in seen:
                keep[i] = False
            else:
                seen.add(x)
        ids = ids[keep]

        if len(ids) < k_total:
            need = k_total - len(ids)
            ids = np.concatenate([ids, self._rand_from(self.s.pretrain_idx, need)], axis=0)[:k_total]

        return ids.astype(np.int64, copy=False)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        anchor = self._choose_anchor()
        ids = self._sample_ids(anchor)
        T = len(ids)

        coord = np.asarray(self.s.coord[ids], dtype=np.float32)
        value = np.asarray(self.s.value_norm[ids], dtype=np.float32)
        var_id = np.asarray(self.s.var_id[ids], dtype=np.int64)
        source_id = np.asarray(self.s.source_id[ids], dtype=np.int64)
        modality_id = np.asarray(self.s.modality_id[ids], dtype=np.int64)
        support_id = np.asarray(self.s.support_id[ids], dtype=np.int64)
        agg_id = np.asarray(self.s.agg_id[ids], dtype=np.int64)
        category_id = np.asarray(self.s.category_id[ids], dtype=np.int64)
        is_categorical = np.asarray(self.s.is_categorical[ids], dtype=np.float32)

        visible = np.ones((T, 1), dtype=np.float32)

        masked = self.rng.random(T) < self.mask_ratio
        masked[0] = True  # always mask anchor

        n_min = min(self.min_masked_tokens, T)
        if int(masked.sum()) < n_min:
            extra = np.where(~masked)[0]
            take = min(n_min - int(masked.sum()), len(extra))
            if take > 0:
                chosen = self.rng.choice(extra, size=take, replace=False)
                masked[chosen] = True

        visible[masked, 0] = 0.0

        return {
            "coord": torch.from_numpy(coord),
            "value": torch.from_numpy(value[:, None]),
            "visible": torch.from_numpy(visible),
            "var_id": torch.from_numpy(var_id),
            "source_id": torch.from_numpy(source_id),
            "modality_id": torch.from_numpy(modality_id),
            "support_id": torch.from_numpy(support_id),
            "agg_id": torch.from_numpy(agg_id),
            "category_id": torch.from_numpy(category_id),
            "is_categorical": torch.from_numpy(is_categorical[:, None]),
            "masked_target": torch.from_numpy(masked.astype(bool)),
            "pad_mask": torch.from_numpy(np.zeros(T, dtype=bool)),
            "target_value": torch.from_numpy(value),
            "target_category": torch.from_numpy(category_id),
            "target_var_id": torch.from_numpy(var_id),
            "target_source_id": torch.from_numpy(source_id),
        }


def collate(rows):
    keys = rows[0].keys()
    return Batch(**{k: torch.stack([r[k] for r in rows], dim=0) for k in keys})


class MultiTaskFoundationModel(nn.Module):
    def __init__(
            self,
            coord_dim: int,
            n_var: int,
            n_source: int,
            n_mod: int,
            n_support: int,
            n_agg: int,
            n_cat: int,
            dim: int = 256,
            depth: int = 6,
            heads: int = 8,
            dropout: float = 0.1,
    ):
        super().__init__()

        ff_dim = coord_dim * 12

        self.coord_proj = nn.Sequential(
            nn.Linear(coord_dim + ff_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.value_proj = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.mask_value = nn.Parameter(torch.zeros(dim))

        self.var_emb = nn.Embedding(n_var, dim)
        self.source_emb = nn.Embedding(n_source, dim)
        self.mod_emb = nn.Embedding(n_mod, dim)
        self.support_emb = nn.Embedding(n_support, dim)
        self.agg_emb = nn.Embedding(n_agg, dim)
        self.cat_emb = nn.Embedding(max(2, n_cat), dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

        self.num_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.var_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, n_var),
        )
        self.source_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, n_source),
        )
        self.cat_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, max(2, n_cat)),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.mask_value, std=0.02)

    def forward(self, b: Batch) -> Dict[str, torch.Tensor]:
        ff = fourier(b.coord)
        tok = self.coord_proj(torch.cat([b.coord, ff], dim=-1))

        v = self.value_proj(b.value)
        v = v * b.visible + self.mask_value.view(1, 1, -1) * (1.0 - b.visible)

        tok = tok + v
        tok = tok + self.var_emb(b.var_id)
        tok = tok + self.source_emb(b.source_id)
        tok = tok + self.mod_emb(b.modality_id)
        tok = tok + self.support_emb(b.support_id)
        tok = tok + self.agg_emb(b.agg_id)
        tok = tok + self.cat_emb(b.category_id.clamp_min(0)) * b.is_categorical

        h = self.encoder(tok, src_key_padding_mask=b.pad_mask)
        h = self.norm(h)

        return {
            "pred_num": self.num_head(h).squeeze(-1),
            "pred_var": self.var_head(h),
            "pred_source": self.source_head(h),
            "pred_cat": self.cat_head(h),
        }


def move_batch_to_device(batch: Batch, device: str) -> Batch:
    return Batch(**{
        k: getattr(batch, k).to(device, non_blocking=True)
        for k in batch.__dataclass_fields__.keys()
    })


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    return (x * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def compute_losses(out, batch, w_num, w_var, w_source, w_cat):
    masked = batch.masked_target & (~batch.pad_mask)
    is_num = batch.is_categorical.squeeze(-1) < 0.5
    is_cat = ~is_num

    num_mask = masked & is_num
    cat_mask = masked & is_cat & (batch.target_category > 0)

    if num_mask.any():
        num_raw = F.smooth_l1_loss(out["pred_num"], batch.target_value, reduction="none", beta=1.0)
        num_loss = masked_mean(num_raw, num_mask)
    else:
        num_loss = torch.zeros((), device=batch.coord.device)

    if masked.any():
        var_loss = F.cross_entropy(out["pred_var"][masked], batch.target_var_id[masked])
        source_loss = F.cross_entropy(out["pred_source"][masked], batch.target_source_id[masked])
    else:
        var_loss = torch.zeros((), device=batch.coord.device)
        source_loss = torch.zeros((), device=batch.coord.device)

    if cat_mask.any():
        cat_loss = F.cross_entropy(out["pred_cat"][cat_mask], batch.target_category[cat_mask])
    else:
        cat_loss = torch.zeros((), device=batch.coord.device)

    loss = (
            w_num * num_loss +
            w_var * var_loss +
            w_source * source_loss +
            w_cat * cat_loss
    )

    stats = {
        "num_loss": float(num_loss.detach().item()),
        "var_loss": float(var_loss.detach().item()),
        "source_loss": float(source_loss.detach().item()),
        "cat_loss": float(cat_loss.detach().item()),
        "masked_tokens": int(masked.sum().item()),
        "masked_cat_tokens": int(cat_mask.sum().item()),
    }
    return loss, stats


def train(args):
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    amp_enabled = (device == "cuda") and (not args.no_amp)

    print("Loading store")
    store = Store(args.store_dir)

    ds = FoundationDataset(
        store=store,
        tokens_per_sample=args.tokens_per_sample,
        samples_per_epoch=args.samples_per_epoch,
        mask_ratio=args.mask_ratio,
        min_masked_tokens=args.min_masked_tokens,
        static_fraction=args.static_fraction,
        max_dt_days=args.max_dt_days,
        max_dist_deg=args.max_dist_deg,
        seed=args.seed,
    )

    dl_kwargs = dict(
        dataset=ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=(device == "cuda"),
        shuffle=False,
        drop_last=False,
    )
    if args.num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 2

    dl = DataLoader(**dl_kwargs)

    model = MultiTaskFoundationModel(
        coord_dim=store.coord_dim,
        n_var=int(np.max(store.var_id)) + 1,
        n_source=int(np.max(store.source_id)) + 1,
        n_mod=int(np.max(store.modality_id)) + 1,
        n_support=int(np.max(store.support_id)) + 1,
        n_agg=int(np.max(store.agg_id)) + 1,
        n_cat=max(2, int(np.max(store.category_id)) + 1),
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    os.makedirs(args.out_dir, exist_ok=True)

    history = []

    for epoch in range(args.epochs):
        model.train()

        running = {
            "loss": 0.0,
            "num_loss": 0.0,
            "var_loss": 0.0,
            "source_loss": 0.0,
            "cat_loss": 0.0,
            "masked_tokens": 0,
            "masked_cat_tokens": 0,
        }

        for batch_idx, batch in enumerate(dl, start=1):
            batch = move_batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                out = model(batch)
                loss, stats = compute_losses(
                    out, batch,
                    w_num=args.w_num,
                    w_var=args.w_var,
                    w_source=args.w_source,
                    w_cat=args.w_cat,
                )

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            running["loss"] += float(loss.item())
            running["num_loss"] += stats["num_loss"]
            running["var_loss"] += stats["var_loss"]
            running["source_loss"] += stats["source_loss"]
            running["cat_loss"] += stats["cat_loss"]
            running["masked_tokens"] += stats["masked_tokens"]
            running["masked_cat_tokens"] += stats["masked_cat_tokens"]

        denom = max(len(dl), 1)
        epoch_stats = {
            "epoch": epoch,
            "loss": running["loss"] / denom,
            "num_loss": running["num_loss"] / denom,
            "var_loss": running["var_loss"] / denom,
            "source_loss": running["source_loss"] / denom,
            "cat_loss": running["cat_loss"] / denom,
            "masked_tokens_per_batch": running["masked_tokens"] / denom,
            "masked_cat_tokens_per_batch": running["masked_cat_tokens"] / denom,
        }
        history.append(epoch_stats)

        print(
            f"epoch {epoch} | "
            f"loss={epoch_stats['loss']:.4f} "
            f"num={epoch_stats['num_loss']:.4f} "
            f"var={epoch_stats['var_loss']:.4f} "
            f"src={epoch_stats['source_loss']:.4f} "
            f"cat={epoch_stats['cat_loss']:.4f} "
            f"masked/batch={epoch_stats['masked_tokens_per_batch']:.1f}"
        )

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": vars(args),
            "history": history,
        },
        os.path.join(args.out_dir, "mini_foundation_final.pt"),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--store_dir", required=True)
    ap.add_argument("--out_dir", default="runs/run1")

    ap.add_argument("--tokens_per_sample", type=int, default=256)
    ap.add_argument("--samples_per_epoch", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=max(1, os.cpu_count() - 1))

    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=30)

    ap.add_argument("--mask_ratio", type=float, default=0.20)
    ap.add_argument("--min_masked_tokens", type=int, default=8)
    ap.add_argument("--static_fraction", type=float, default=0.25)
    ap.add_argument("--max_dt_days", type=float, default=14.0)
    ap.add_argument("--max_dist_deg", type=float, default=2.0)

    ap.add_argument("--w_num", type=float, default=1.0)
    ap.add_argument("--w_var", type=float, default=0.2)
    ap.add_argument("--w_source", type=float, default=0.2)
    ap.add_argument("--w_cat", type=float, default=1.0)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no_amp", action="store_true")

    train(ap.parse_args())
