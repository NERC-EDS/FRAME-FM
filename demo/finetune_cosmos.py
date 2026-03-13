#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from framefm_core import Batch, Store, fourier


# ------------------------------------------------------------
# Minimal demo:
# - one 5-fold split, run one fold via --fold_index
# - three models only:
#     1) pooled_context_baseline
#     2) pretrained_frozen
#     3) pretrained_full
# - query keeps location/time/variable metadata
# - context is local: same site + same var first, then same var
# - pretrained models predict residual over context mean
# ------------------------------------------------------------


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    denom = np.sqrt(np.sum(yt ** 2) * np.sum(yp ** 2))
    r = float(np.sum(yt * yp) / denom) if denom > 0 else float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r": r, "r2": r2}


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


@dataclass
class NormalizationMap:
    mu: float
    sigma: float

    def denorm(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) * self.sigma + self.mu


def fit_affine_denormalizer(store: Store, row_ids: np.ndarray) -> NormalizationMap | None:
    vals_raw = np.asarray(store.value_num[row_ids], dtype=np.float64)
    vals_norm = np.asarray(store.value_norm[row_ids], dtype=np.float64)
    good = np.isfinite(vals_raw) & np.isfinite(vals_norm)
    vals_raw = vals_raw[good]
    vals_norm = vals_norm[good]
    if len(vals_raw) < 4:
        return None
    A = np.stack([vals_norm, np.ones_like(vals_norm)], axis=1)
    sol, *_ = np.linalg.lstsq(A, vals_raw, rcond=None)
    sigma, mu = sol
    if not np.isfinite(mu) or not np.isfinite(sigma) or abs(sigma) < 1e-12:
        return None
    return NormalizationMap(mu=float(mu), sigma=float(sigma))


def get_name_to_id_maps(store: Store):
    var_name_to_id = {v: k for k, v in store.var_id_to_name.items()}
    source_name_to_id = {v: k for k, v in store.source_id_to_name.items()}
    site_name_to_id = {v: k for k, v in store.site_id_to_name.items()}
    return var_name_to_id, source_name_to_id, site_name_to_id


def find_cosmos_rows(store: Store) -> np.ndarray:
    _, source_name_to_id, _ = get_name_to_id_maps(store)
    cosmos_sid = source_name_to_id.get("cosmos")
    if cosmos_sid is None:
        raise RuntimeError("Could not find source_name='cosmos'.")
    rows = np.where(np.asarray(store.source_id[:], dtype=np.int64) == int(cosmos_sid))[0]
    if len(rows) == 0:
        raise RuntimeError("No COSMOS rows found.")
    return rows.astype(np.int64)


def choose_target_var(store: Store, cosmos_rows: np.ndarray, target_var_name: str | None) -> tuple[int, str]:
    var_ids = np.asarray(store.var_id[cosmos_rows], dtype=np.int64)
    is_num = np.asarray(store.is_categorical[cosmos_rows], dtype=np.uint8) == 0
    if target_var_name is not None:
        var_name_to_id, _, _ = get_name_to_id_maps(store)
        return int(var_name_to_id[target_var_name]), target_var_name
    uniq, cnt = np.unique(var_ids[is_num], return_counts=True)
    best = int(uniq[np.argmax(cnt)])
    return best, store.var_id_to_name[best]


def make_5fold_site_split(store: Store, fold_index: int, seed: int, target_var_name: str | None,
                          min_rows_per_site: int):
    cosmos_rows = find_cosmos_rows(store)
    target_var_id, target_var_name = choose_target_var(store, cosmos_rows, target_var_name)
    _, source_name_to_id, site_name_to_id = get_name_to_id_maps(store)

    rows = cosmos_rows[np.asarray(store.var_id[cosmos_rows] == target_var_id)]
    rows = rows[np.asarray(store.is_categorical[rows] == 0)]

    site_ids = np.asarray(store.site_id[rows], dtype=np.int64)
    eligible_site_ids = []
    for sid in np.unique(site_ids):
        if int(np.sum(site_ids == sid)) >= min_rows_per_site:
            eligible_site_ids.append(int(sid))
    site_names = sorted([store.site_id_to_name[sid] for sid in eligible_site_ids])
    if len(site_names) < 5:
        raise RuntimeError(f"Need at least 5 eligible sites, found {len(site_names)}")

    rng = np.random.default_rng(seed)
    site_names = list(site_names)
    rng.shuffle(site_names)
    folds = [[] for _ in range(5)]
    for i, s in enumerate(site_names):
        folds[i % 5].append(s)

    test_sites = sorted(folds[fold_index])
    trainval_sites = sorted([s for i, fold in enumerate(folds) if i != fold_index for s in fold])
    rng = np.random.default_rng(seed + 17)
    rng.shuffle(trainval_sites)
    n_val = max(1, int(round(0.2 * len(trainval_sites))))
    val_sites = sorted(trainval_sites[:n_val])
    train_sites = sorted(trainval_sites[n_val:])

    train_site_ids = {site_name_to_id[s] for s in train_sites}
    val_site_ids = {site_name_to_id[s] for s in val_sites}
    test_site_ids = {site_name_to_id[s] for s in test_sites}

    row_site_ids = np.asarray(store.site_id[rows], dtype=np.int64)
    train_targets = rows[np.isin(row_site_ids, list(train_site_ids))]
    val_targets = rows[np.isin(row_site_ids, list(val_site_ids))]
    test_targets = rows[np.isin(row_site_ids, list(test_site_ids))]

    cosmos_sid = source_name_to_id["cosmos"]
    heldout_site_ids = val_site_ids | test_site_ids
    not_heldout = ~np.isin(np.asarray(store.site_id[:], dtype=np.int64), list(heldout_site_ids))
    finite_num = np.isfinite(np.asarray(store.value_norm[:], dtype=np.float32))
    context_mask = ((store.pretrain_mask[:] > 0) | (store.source_id[:] == cosmos_sid)) & not_heldout & finite_num
    context_pool = np.where(context_mask)[0].astype(np.int64)

    return {
        "target_var_name": target_var_name,
        "train_targets": train_targets,
        "val_targets": val_targets,
        "test_targets": test_targets,
        "context_pool": context_pool,
        "train_sites": train_sites,
        "val_sites": val_sites,
        "test_sites": test_sites,
    }


class SimpleCosmosDataset(Dataset):
    def __init__(self, store: Store, target_indices: np.ndarray, context_pool: np.ndarray, tokens_per_sample: int = 32,
                 seed: int = 0):
        self.s = store
        self.targets = np.asarray(target_indices, dtype=np.int64)
        self.context_pool = np.asarray(context_pool, dtype=np.int64)
        self.T = int(tokens_per_sample)
        self.rng = np.random.default_rng(seed)

        self.context_site = np.asarray(self.s.site_id[self.context_pool], dtype=np.int64)
        self.context_var = np.asarray(self.s.var_id[self.context_pool], dtype=np.int64)

        if hasattr(self.s, "time_center_sec"):
            self.context_time = np.asarray(self.s.time_center_sec[self.context_pool], dtype=np.int64)
        elif hasattr(self.s, "time_start_sec"):
            self.context_time = np.asarray(self.s.time_start_sec[self.context_pool], dtype=np.int64)
        else:
            raise AttributeError("Store has neither time_center_sec nor time_start_sec")

    def __len__(self):
        return len(self.targets)

    def _sample_local_context(self, target: int) -> np.ndarray:
        n_ctx = self.T - 1
        target_site = int(self.s.site_id[target])
        target_var = int(self.s.var_id[target])

        if hasattr(self.s, "time_center_sec"):
            target_time = int(self.s.time_center_sec[target])
        elif hasattr(self.s, "time_start_sec"):
            target_time = int(self.s.time_start_sec[target])
        else:
            raise AttributeError("Store has neither time_center_sec nor time_start_sec")

        same_site_same_var = np.where((self.context_site == target_site) & (self.context_var == target_var))[0]
        same_var = np.where(self.context_var == target_var)[0]

        chosen_pool_idx = None
        if len(same_site_same_var) >= 8:
            delta = np.abs(self.context_time[same_site_same_var] - target_time)
            order = np.argsort(delta)
            chosen_pool_idx = same_site_same_var[order[: min(len(order), max(n_ctx * 3, 16))]]
        elif len(same_var) >= 8:
            delta = np.abs(self.context_time[same_var] - target_time)
            order = np.argsort(delta)
            chosen_pool_idx = same_var[order[: min(len(order), max(n_ctx * 4, 32))]]
        else:
            chosen_pool_idx = np.arange(len(self.context_pool))

        pool = self.context_pool[chosen_pool_idx]
        pool = pool[pool != target]
        if len(pool) == 0:
            raise RuntimeError("Context pool became empty after removing target row")

        replace = len(pool) < n_ctx
        ctx = self.rng.choice(pool, size=n_ctx, replace=replace)
        return np.asarray(ctx, dtype=np.int64)

    def __getitem__(self, idx: int):
        target = int(self.targets[idx])
        ctx = self._sample_local_context(target)
        ids = np.concatenate([[target], ctx], axis=0)

        coord = np.asarray(self.s.coord[ids], dtype=np.float32)
        var_id = np.asarray(self.s.var_id[ids], dtype=np.int64)
        source_id = np.asarray(self.s.source_id[ids], dtype=np.int64)
        modality_id = np.asarray(self.s.modality_id[ids], dtype=np.int64)
        support_id = np.asarray(self.s.support_id[ids], dtype=np.int64)
        agg_id = np.asarray(self.s.agg_id[ids], dtype=np.int64)
        category_id = np.asarray(self.s.category_id[ids], dtype=np.int64)
        value_norm = np.asarray(self.s.value_norm[ids], dtype=np.float32)
        visible = np.ones((self.T, 1), dtype=np.float32)
        visible[0, 0] = 0.0

        return {
            "batch": Batch(
                coord=torch.from_numpy(coord),
                value=torch.from_numpy(value_norm[:, None]),
                visible=torch.from_numpy(visible),
                var_id=torch.from_numpy(var_id),
                source_id=torch.from_numpy(source_id),
                modality_id=torch.from_numpy(modality_id),
                support_id=torch.from_numpy(support_id),
                agg_id=torch.from_numpy(agg_id),
                category_id=torch.from_numpy(category_id),
                is_categorical=torch.from_numpy(np.asarray(self.s.is_categorical[ids], dtype=np.float32)[:, None]),
                masked_target=torch.tensor([True] + [False] * (self.T - 1), dtype=torch.bool),
                pad_mask=torch.zeros(self.T, dtype=torch.bool),
                target_value=torch.from_numpy(value_norm),
                target_category=torch.from_numpy(np.asarray(self.s.category_id[ids], dtype=np.int64)),
                target_var_id=torch.from_numpy(np.asarray(self.s.var_id[ids], dtype=np.int64)),
                target_source_id=torch.from_numpy(np.asarray(self.s.source_id[ids], dtype=np.int64)),
            ),
            "target_reg": torch.tensor(float(value_norm[0]), dtype=torch.float32),
            "target_row_id": torch.tensor(target, dtype=torch.int64),
        }


def collate_rows(rows):
    batch_rows = [r["batch"] for r in rows]
    keys = batch_rows[0].__dataclass_fields__.keys()
    batch = Batch(**{k: torch.stack([getattr(r, k) for r in batch_rows], dim=0) for k in keys})
    return {
        "batch": batch,
        "target_reg": torch.stack([r["target_reg"] for r in rows], dim=0),
        "target_row_id": torch.stack([r["target_row_id"] for r in rows], dim=0),
    }


def make_loader(ds, batch_size: int, num_workers: int, shuffle: bool):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_rows,
                      drop_last=False)


def move_batch(batch: Batch, device: str) -> Batch:
    return Batch(**{k: getattr(batch, k).to(device) for k in batch.__dataclass_fields__.keys()})


class FoundationBackbone(nn.Module):
    def __init__(self, coord_dim, n_var, n_source, n_mod, n_support, n_agg, n_cat, dim=256, depth=4, heads=4,
                 dropout=0.1):
        super().__init__()
        ff_dim = coord_dim * 12
        self.coord_proj = nn.Sequential(nn.Linear(coord_dim + ff_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.value_proj = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.mask_value = nn.Parameter(torch.zeros(dim))
        self.var_emb = nn.Embedding(n_var, dim)
        self.source_emb = nn.Embedding(n_source, dim)
        self.mod_emb = nn.Embedding(n_mod, dim)
        self.support_emb = nn.Embedding(n_support, dim)
        self.agg_emb = nn.Embedding(n_agg, dim)
        self.cat_emb = nn.Embedding(max(2, n_cat), dim)
        enc = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def encode(self, b: Batch) -> torch.Tensor:
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
        return self.norm(h)


class FoundationRegressor(nn.Module):
    def __init__(self, backbone: FoundationBackbone, dim: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, b: Batch) -> torch.Tensor:
        ctx_val = b.value[:, 1:, 0]
        valid = b.visible[:, 1:, 0]
        same_var = (b.var_id[:, 1:] == b.var_id[:, :1]).float()
        use = valid * same_var
        ctx_mean = (ctx_val * use).sum(dim=1) / use.sum(dim=1).clamp_min(1.0)
        h = self.backbone.encode(b)
        delta = self.head(h[:, 0]).squeeze(-1)
        return ctx_mean + delta


class ContextSummaryMLP(nn.Module):
    def __init__(self, coord_dim: int, n_var: int, n_source: int, n_mod: int, n_support: int, n_agg: int,
                 hidden: int = 192):
        super().__init__()
        self.var_emb = nn.Embedding(n_var, 16)
        self.source_emb = nn.Embedding(n_source, 8)
        self.mod_emb = nn.Embedding(n_mod, 8)
        self.support_emb = nn.Embedding(n_support, 8)
        self.agg_emb = nn.Embedding(n_agg, 8)
        feat_dim = coord_dim * 3 + 2 + 16 + 8 + 8 + 8 + 8
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, b: Batch) -> torch.Tensor:
        ctx_coord = b.coord[:, 1:, :]
        ctx_val = b.value[:, 1:, 0]
        valid = b.visible[:, 1:, 0]
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        coord_mean = (ctx_coord * valid.unsqueeze(-1)).sum(dim=1) / denom
        coord_std = torch.sqrt(
            (((ctx_coord - coord_mean.unsqueeze(1)) ** 2) * valid.unsqueeze(-1)).sum(dim=1) / denom + 1e-6)
        query_coord = b.coord[:, 0, :]
        val_mean = (ctx_val * valid).sum(dim=1, keepdim=True) / denom
        val_std = torch.sqrt((((ctx_val - val_mean) ** 2) * valid).sum(dim=1, keepdim=True) / denom + 1e-6)
        query_meta = torch.cat([
            self.var_emb(b.var_id[:, 0]),
            self.source_emb(b.source_id[:, 0]),
            self.mod_emb(b.modality_id[:, 0]),
            self.support_emb(b.support_id[:, 0]),
            self.agg_emb(b.agg_id[:, 0]),
        ], dim=1)
        x = torch.cat([query_coord, coord_mean, coord_std, val_mean, val_std, query_meta], dim=1)
        return self.net(x).squeeze(-1)


def infer_backbone_dim_from_ckpt(ckpt_path: str | None) -> int | None:
    if ckpt_path is None:
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    if "mask_value" in state and getattr(state["mask_value"], "ndim", None) == 1:
        return int(state["mask_value"].shape[0])
    return None


def build_backbone(store: Store, dim: int, depth: int, heads: int, dropout: float) -> FoundationBackbone:
    return FoundationBackbone(
        coord_dim=store.coord_dim,
        n_var=max(2, int(np.max(store.var_id)) + 1),
        n_source=max(2, int(np.max(store.source_id)) + 1),
        n_mod=max(2, int(np.max(store.modality_id)) + 1),
        n_support=max(2, int(np.max(store.support_id)) + 1),
        n_agg=max(2, int(np.max(store.agg_id)) + 1),
        n_cat=max(2, int(np.max(store.category_id)) + 1),
        dim=dim,
        depth=depth,
        heads=heads,
        dropout=dropout,
    )


def build_summary_baseline(store: Store, hidden: int) -> ContextSummaryMLP:
    return ContextSummaryMLP(
        coord_dim=store.coord_dim,
        n_var=max(2, int(np.max(store.var_id)) + 1),
        n_source=max(2, int(np.max(store.source_id)) + 1),
        n_mod=max(2, int(np.max(store.modality_id)) + 1),
        n_support=max(2, int(np.max(store.support_id)) + 1),
        n_agg=max(2, int(np.max(store.agg_id)) + 1),
        hidden=hidden,
    )


def maybe_load_pretrained(backbone: FoundationBackbone, ckpt_path: str | None) -> None:
    if ckpt_path is None:
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)

    current = backbone.state_dict()
    filtered = {}
    skipped = []
    for k, v in state.items():
        if k not in current:
            continue
        if tuple(current[k].shape) != tuple(v.shape):
            skipped.append((k, tuple(v.shape), tuple(current[k].shape)))
            continue
        filtered[k] = v

    backbone.load_state_dict(filtered, strict=False)

    if skipped:
        print("[pretrained] skipped incompatible checkpoint tensors:")
        for name, src_shape, dst_shape in skipped:
            print(f"  - {name}: checkpoint{src_shape} != model{dst_shape}")


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def freeze_backbone(model: FoundationRegressor) -> None:
    set_requires_grad(model.backbone, False)
    set_requires_grad(model.head, True)


def unfreeze_all(model: FoundationRegressor) -> None:
    set_requires_grad(model, True)


def make_optimizer(model, lr: float, backbone_lr_mult: float = 0.5):
    if isinstance(model, FoundationRegressor):
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        head_params = [p for p in model.head.parameters() if p.requires_grad]
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr * backbone_lr_mult})
        if head_params:
            groups.append({"params": head_params, "lr": lr})
        return torch.optim.AdamW(groups)
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)


def run_epoch(model, dl, device, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    ys, ps, losses, row_ids = [], [], [], []
    ctx = torch.enable_grad() if train_mode else torch.inference_mode()
    with ctx:
        for item in dl:
            batch = move_batch(item["batch"], device)
            target = item["target_reg"].to(device)
            pred = model(batch)
            loss = F.smooth_l1_loss(pred, target, beta=1.0)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            ys.append(target.detach().cpu().numpy())
            ps.append(pred.detach().cpu().numpy())
            losses.append(float(loss.detach().item()))
            row_ids.append(item["target_row_id"].cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    rid = np.concatenate(row_ids)
    metrics = regression_metrics(y, p)
    metrics["loss"] = float(np.mean(losses))
    metrics["y_true"] = y
    metrics["y_pred"] = p
    metrics["row_ids"] = rid
    return metrics


def fit_model(model, train_dl, val_dl, test_dl, device, epochs, lr, name, freeze_backbone_epochs=0,
              backbone_lr_mult=0.5, patience=4):
    best_val = float("inf")
    best = None
    best_epoch = -1
    stale = 0
    history = []
    current_mode = None
    opt = None
    started = time.time()

    for epoch in range(1, epochs + 1):
        if isinstance(model, FoundationRegressor):
            if epoch <= freeze_backbone_epochs:
                freeze_backbone(model)
                mode = "head_only"
            else:
                unfreeze_all(model)
                mode = "full"
        else:
            mode = "full"

        if opt is None or mode != current_mode:
            opt = make_optimizer(model, lr=lr, backbone_lr_mult=backbone_lr_mult)
            current_mode = mode

        train_m = run_epoch(model, train_dl, device, optimizer=opt)
        val_m = run_epoch(model, val_dl, device)
        test_m = run_epoch(model, test_dl, device)
        history.append(
            {"epoch": epoch, "train_rmse": train_m["rmse"], "val_rmse": val_m["rmse"], "test_rmse": test_m["rmse"]})
        print(
            f"[{name}] epoch {epoch:02d} mode={mode} train={train_m['rmse']:.4f} val={val_m['rmse']:.4f} test={test_m['rmse']:.4f} pred_std={np.std(test_m['y_pred']):.4f}",
            flush=True,
        )

        if val_m["rmse"] < best_val:
            best_val = val_m["rmse"]
            best_epoch = epoch
            best = {"test": test_m, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best["state"])
    return {
        "name": name,
        "best_epoch": best_epoch,
        "elapsed_sec": time.time() - started,
        "history": history,
        "test": best["test"],
    }


def save_bar_chart(path: str, rows: list[dict], metric_key: str, title: str, ylabel: str):
    order = ["pooled_context_baseline", "pretrained_frozen", "pretrained_full"]
    labels = ["baseline", "pretrained\nfrozen", "pretrained\nfull"]
    vals = [next(r[metric_key] for r in rows if r["model"] == m) for m in order]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, vals)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_scatter(path: str, y_true: np.ndarray, y_pred: np.ndarray, title: str):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    pad = 0.05 * max(1e-6, hi - lo)
    lo -= pad
    hi += pad
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=12, alpha=0.45)
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def subsample_rows(rows: np.ndarray, budget: int, rng: np.random.Generator) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if budget <= 0 or budget >= len(rows):
        return rows
    idx = rng.choice(len(rows), size=budget, replace=False)
    return rows[idx]


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    device = "cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda"

    inferred_dim = infer_backbone_dim_from_ckpt(args.pretrained_ckpt)
    if inferred_dim is not None and inferred_dim != args.dim:
        print(f"[pretrained] overriding --dim from {args.dim} to checkpoint dim {inferred_dim}")
        args.dim = inferred_dim

    store = Store(args.store_dir)

    split = make_5fold_site_split(
        store=store,
        fold_index=args.fold_index,
        seed=args.seed,
        target_var_name=args.target_var,
        min_rows_per_site=args.min_rows_per_site,
    )

    rng = np.random.default_rng(args.seed + 99)
    train_targets = subsample_rows(split["train_targets"], args.train_budget, rng)
    val_targets = subsample_rows(split["val_targets"], args.val_budget, rng)
    test_targets = subsample_rows(split["test_targets"], args.test_budget, rng)

    print(
        f"Split built | target={split['target_var_name']} train={len(train_targets)} val={len(val_targets)} test={len(test_targets)} tokens={args.tokens_per_sample}",
        flush=True,
    )

    train_ds = SimpleCosmosDataset(store, train_targets, split["context_pool"],
                                   tokens_per_sample=args.tokens_per_sample, seed=args.seed + 1)
    val_ds = SimpleCosmosDataset(store, val_targets, split["context_pool"], tokens_per_sample=args.tokens_per_sample,
                                 seed=args.seed + 2)
    test_ds = SimpleCosmosDataset(store, test_targets, split["context_pool"], tokens_per_sample=args.tokens_per_sample,
                                  seed=args.seed + 3)

    train_dl = make_loader(train_ds, args.batch_size, args.num_workers, shuffle=True)
    val_dl = make_loader(val_ds, args.batch_size, args.num_workers, shuffle=False)
    test_dl = make_loader(test_ds, args.batch_size, args.num_workers, shuffle=False)

    baseline = build_summary_baseline(store, hidden=args.hidden).to(device)
    baseline_res = fit_model(
        baseline,
        train_dl,
        val_dl,
        test_dl,
        device,
        epochs=args.epochs,
        lr=args.baseline_lr,
        name="pooled_context_baseline",
        patience=args.patience,
    )

    frozen_backbone = build_backbone(store, args.dim, args.depth, args.heads, args.dropout)
    maybe_load_pretrained(frozen_backbone, args.pretrained_ckpt)
    frozen_model = FoundationRegressor(frozen_backbone, dim=args.dim).to(device)
    frozen_res = fit_model(
        frozen_model,
        train_dl,
        val_dl,
        test_dl,
        device,
        epochs=args.epochs,
        lr=args.lr,
        name="pretrained_frozen",
        freeze_backbone_epochs=args.epochs,
        backbone_lr_mult=args.backbone_lr_mult,
        patience=args.patience,
    )

    full_backbone = build_backbone(store, args.dim, args.depth, args.heads, args.dropout)
    maybe_load_pretrained(full_backbone, args.pretrained_ckpt)
    full_model = FoundationRegressor(full_backbone, dim=args.dim).to(device)
    full_res = fit_model(
        full_model,
        train_dl,
        val_dl,
        test_dl,
        device,
        epochs=args.epochs,
        lr=args.lr,
        name="pretrained_full",
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        backbone_lr_mult=args.backbone_lr_mult,
        patience=args.patience,
    )

    denorm = fit_affine_denormalizer(store, train_targets)

    rows = []
    for res in [baseline_res, frozen_res, full_res]:
        y_true = res["test"]["y_true"]
        y_pred = res["test"]["y_pred"]
        metrics = regression_metrics(y_true, y_pred)
        row = {
            "model": res["name"],
            "best_epoch": res["best_epoch"],
            "elapsed_sec": round(res["elapsed_sec"], 2),
            "rmse_norm": metrics["rmse"],
            "mae_norm": metrics["mae"],
            "r_norm": metrics["r"],
            "r2_norm": metrics["r2"],
        }
        if denorm is not None:
            y_true_raw = denorm.denorm(y_true)
            y_pred_raw = denorm.denorm(y_pred)
            raw = regression_metrics(y_true_raw, y_pred_raw)
            row.update({
                "rmse_raw": raw["rmse"],
                "mae_raw": raw["mae"],
                "r_raw": raw["r"],
                "r2_raw": raw["r2"],
            })
        rows.append(row)

    write_csv(os.path.join(args.out_dir, "demo_results.csv"), rows)

    by_name = {r["model"]: r for r in rows}
    baseline_rmse = by_name["pooled_context_baseline"].get("rmse_raw", by_name["pooled_context_baseline"]["rmse_norm"])
    frozen_rmse = by_name["pretrained_frozen"].get("rmse_raw", by_name["pretrained_frozen"]["rmse_norm"])
    full_rmse = by_name["pretrained_full"].get("rmse_raw", by_name["pretrained_full"]["rmse_norm"])

    summary = {
        "target_var": split["target_var_name"],
        "fold_index": args.fold_index,
        "train_budget": int(len(train_targets)),
        "val_budget": int(len(val_targets)),
        "test_budget": int(len(test_targets)),
        "test_sites": split["test_sites"],
        "why_foundation_model_is_good": {
            "baseline_rmse": baseline_rmse,
            "pretrained_frozen_rmse": frozen_rmse,
            "relative_improvement_percent": 100.0 * (baseline_rmse - frozen_rmse) / max(1e-12, baseline_rmse),
        },
        "why_finetuning_is_better": {
            "pretrained_frozen_rmse": frozen_rmse,
            "pretrained_full_rmse": full_rmse,
            "relative_improvement_percent": 100.0 * (frozen_rmse - full_rmse) / max(1e-12, frozen_rmse),
        },
        "results": rows,
    }

    with open(os.path.join(args.out_dir, "demo_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    metric_key = "rmse_raw" if "rmse_raw" in rows[0] else "rmse_norm"
    ylabel = "RMSE (raw units)" if metric_key == "rmse_raw" else "RMSE (normalised)"
    save_bar_chart(os.path.join(args.out_dir, "rmse_comparison.png"), rows, metric_key=metric_key,
                   title="5-fold demo: one split, three models", ylabel=ylabel)

    best_scatter = full_res["test"]
    if denorm is not None:
        y_true = denorm.denorm(best_scatter["y_true"])
        y_pred = denorm.denorm(best_scatter["y_pred"])
    else:
        y_true = best_scatter["y_true"]
        y_pred = best_scatter["y_pred"]
    save_scatter(os.path.join(args.out_dir, "pretrained_full_scatter.png"), y_true, y_pred,
                 "Fine-tuned foundation model")

    metric_for_summary = "rmse_raw" if "rmse_raw" in rows[0] else "rmse_norm"
    baseline_val = by_name["pooled_context_baseline"][metric_for_summary]
    frozen_val = by_name["pretrained_frozen"][metric_for_summary]
    full_val = by_name["pretrained_full"][metric_for_summary]
    pretrain_gain = 100.0 * (baseline_val - frozen_val) / max(1e-12, baseline_val)
    finetune_gain = 100.0 * (frozen_val - full_val) / max(1e-12, frozen_val)
    unit_label = "raw-unit RMSE" if metric_for_summary == "rmse_raw" else "normalised RMSE"

    print("\nDemo headline summary")
    print("-" * 72)
    print(f"Target variable: {split['target_var_name']}")
    print(f"Held-out test sites: {', '.join(split['test_sites'])}")
    print(f"Training samples used: {len(train_targets)}")
    print(f"Validation samples used: {len(val_targets)}")
    print(f"Test samples used: {len(test_targets)}")
    print()
    print(
        f"Foundation model benefit: pretrained_frozen improves over pooled_context_baseline by {pretrain_gain:.1f}% in {unit_label} ({baseline_val:.4f} -> {frozen_val:.4f}).")
    print(
        f"Fine-tuning benefit: pretrained_full improves over pretrained_frozen by {finetune_gain:.1f}% in {unit_label} ({frozen_val:.4f} -> {full_val:.4f}).")
    print("Done.")
    print(f"Results CSV: {os.path.join(args.out_dir, 'demo_results.csv')}")
    print(f"Summary JSON: {os.path.join(args.out_dir, 'demo_summary.json')}")
    print(f"RMSE plot: {os.path.join(args.out_dir, 'rmse_comparison.png')}")
    print(f"Scatter: {os.path.join(args.out_dir, 'pretrained_full_scatter.png')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Minimal 5-fold demo comparing baseline vs foundation model variants.")
    ap.add_argument("--store_dir", required=True)
    ap.add_argument("--pretrained_ckpt", required=True)
    ap.add_argument("--out_dir", default="runs/simple_foundation_demo")
    ap.add_argument("--target_var", default=None)
    ap.add_argument("--fold_index", type=int, default=0, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--train_budget", type=int, default=128)
    ap.add_argument("--val_budget", type=int, default=128)
    ap.add_argument("--test_budget", type=int, default=256)
    ap.add_argument("--min_rows_per_site", type=int, default=8)
    ap.add_argument("--tokens_per_sample", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--baseline_lr", type=float, default=1e-4)
    ap.add_argument("--backbone_lr_mult", type=float, default=0.5)
    ap.add_argument("--freeze_backbone_epochs", type=int, default=1)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    main(ap.parse_args())
