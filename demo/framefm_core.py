from __future__ import annotations
import json
import os
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict


@dataclass
class Batch:
    coord: torch.Tensor
    value: torch.Tensor
    visible: torch.Tensor
    var_id: torch.Tensor
    source_id: torch.Tensor
    modality_id: torch.Tensor
    support_id: torch.Tensor
    agg_id: torch.Tensor
    category_id: torch.Tensor
    is_categorical: torch.Tensor
    masked_target: torch.Tensor
    pad_mask: torch.Tensor
    target_value: torch.Tensor
    target_category: torch.Tensor
    target_var_id: torch.Tensor
    target_source_id: torch.Tensor


class Store:
    """
    Memory-mapped dataset store used by all scripts.
    """

    def __init__(self, base: str):
        meta_path = os.path.join(base, "meta.json")
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.N = int(self.meta["total_rows"])
        self.coord_dim = int(self.meta["coord_dim"])

        def mm(name, dtype, shape):
            return np.memmap(os.path.join(base, name), dtype=dtype, mode="r", shape=shape)

        self.coord = mm("coord.dat", np.float32, (self.N, self.coord_dim))
        self.value_norm = mm("value_norm.dat", np.float32, (self.N,))
        self.value_num = mm("value_num.dat", np.float32, (self.N,))
        self.var_id = mm("var_id.dat", np.int32, (self.N,))
        self.source_id = mm("source_id.dat", np.int32, (self.N,))
        self.modality_id = mm("modality_id.dat", np.int32, (self.N,))
        self.support_id = mm("support_type_id.dat", np.int32, (self.N,))
        self.agg_id = mm("agg_type_id.dat", np.int32, (self.N,))
        self.category_id = mm("category_id.dat", np.int32, (self.N,))
        self.is_categorical = mm("is_categorical.dat", np.uint8, (self.N,))
        self.pretrain_mask = mm("pretrain_mask.dat", np.uint8, (self.N,))
        self.finetune_train_mask = mm("finetune_train_mask.dat", np.uint8, (self.N,))
        self.finetune_test_mask = mm("finetune_test_mask.dat", np.uint8, (self.N,))
        self.time_center_sec = mm("time_center_sec.dat", np.int64, (self.N,))
        self.lat = mm("lat.dat", np.float32, (self.N,))
        self.lon = mm("lon.dat", np.float32, (self.N,))
        self.site_id = mm("site_id.dat", np.int32, (self.N,))

        regs = self.meta["registries"]
        self.var_id_to_name = {int(v): k for k, v in regs["var_name"].items()}
        self.source_id_to_name = {int(v): k for k, v in regs["source_name"].items()}
        self.site_id_to_name = {int(v): k for k, v in regs["site_id"].items()}

        # Pretraining rows
        self.pretrain_idx = np.where(self.pretrain_mask[:] > 0)[0]

        # COSMOS rows
        self.cosmos_train_idx = np.where(self.finetune_train_mask[:] > 0)[0]
        self.cosmos_test_idx = np.where(self.finetune_test_mask[:] > 0)[0]

        # Fallback split if test mask empty
        if len(self.cosmos_test_idx) == 0 and len(self.cosmos_train_idx) > 10:
            print("WARNING: no COSMOS test split found — creating fallback split")
            base_idx = self.cosmos_train_idx.copy()
            order = np.argsort(self.time_center_sec[base_idx])
            split = int(len(order) * 0.9)
            self.cosmos_train_idx = base_idx[order[:split]]
            self.cosmos_test_idx = base_idx[order[split:]]

        self.context_pool = np.where(
            (self.pretrain_mask[:] > 0) | (self.finetune_train_mask[:] > 0)
        )[0]

        print(
            f"Store loaded | total={self.N:,} "
            f"pretrain={len(self.pretrain_idx):,} "
            f"cosmos_train={len(self.cosmos_train_idx):,} "
            f"cosmos_test={len(self.cosmos_test_idx):,}"
        )


def fourier(x: torch.Tensor, bands: int = 6) -> torch.Tensor:
    freqs = torch.logspace(0, 1.5, bands, device=x.device, dtype=x.dtype)
    z = x.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(z), torch.cos(z)], dim=-1).flatten(-2)


class MiniFoundationModel(nn.Module):

    def __init__(
            self,
            coord_dim,
            n_var,
            n_source,
            n_mod,
            n_support,
            n_agg,
            n_cat,
            dim=256,
            depth=6,
            heads=8,
            dropout=0.1,
    ):
        super().__init__()

        ff_dim = coord_dim * 12

        self.coord_proj = nn.Sequential(
            nn.Linear(coord_dim + ff_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self.value_proj = nn.Linear(1, dim)
        self.mask_value = nn.Parameter(torch.zeros(dim))

        self.var_emb = nn.Embedding(n_var, dim)
        self.source_emb = nn.Embedding(n_source, dim)
        self.mod_emb = nn.Embedding(n_mod, dim)
        self.support_emb = nn.Embedding(n_support, dim)
        self.agg_emb = nn.Embedding(n_agg, dim)
        self.cat_emb = nn.Embedding(max(2, n_cat), dim)

        enc_layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            dim * 4,
            dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(enc_layer, depth)
        self.norm = nn.LayerNorm(dim)

        self.num_head = nn.Linear(dim, 1)

    def forward(self, b: Batch):
        ff = fourier(b.coord)
        tok = self.coord_proj(torch.cat([b.coord, ff], dim=-1))

        v = self.value_proj(b.value)
        v = v * b.visible + self.mask_value.view(1, 1, -1) * (1 - b.visible)

        tok = tok + v
        tok = tok + self.var_emb(b.var_id)
        tok = tok + self.source_emb(b.source_id)
        tok = tok + self.mod_emb(b.modality_id)
        tok = tok + self.support_emb(b.support_id)
        tok = tok + self.agg_emb(b.agg_id)
        tok = tok + self.cat_emb(b.category_id.clamp_min(0)) * b.is_categorical

        h = self.encoder(tok, src_key_padding_mask=b.pad_mask)
        h = self.norm(h)

        return {"pred_num": self.num_head(h).squeeze(-1)}
