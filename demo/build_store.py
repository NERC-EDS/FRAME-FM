#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PASS1_COLS = [
    "var_name",
    "source_name",
    "modality",
    "support_type",
    "agg_type",
    "task_group",
    "site_id",
    "is_categorical",
    "category_value",
    "value_num",
]

PASS2_COLS = [
    "lat",
    "lon",
    "time_start_sec",
    "time_end_sec",
    "dt_hours",
    "dx_m",
    "dy_m",
    "var_name",
    "source_name",
    "modality",
    "support_type",
    "agg_type",
    "task_group",
    "site_id",
    "is_categorical",
    "category_value",
    "value_num",
]


def write_memmap(path, dtype, shape):
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape)


def finalize_numeric_stats(stats):
    out = {}
    for k, v in stats.items():
        n = max(int(v["count"]), 1)
        mean = v["sum"] / n
        var = max(v["sumsq"] / n - mean * mean, 0.0)
        std = math.sqrt(var)
        if std <= 1e-8:
            std = 1.0
        out[k] = {"mean": float(mean), "std": float(std)}
    return out


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.tokens_dir, "*.parquet")))
    if not paths:
        raise RuntimeError("No parquet shards found")

    print(f"Found {len(paths)} shards")

    row_counts = []
    total_rows = 0
    for p in paths:
        n = pq.ParquetFile(p).metadata.num_rows
        row_counts.append(n)
        total_rows += n

    print("Total rows:", total_rows)

    # pass 1: registries + numeric stats
    var_vals = set()
    source_vals = set()
    modality_vals = set()
    support_vals = set()
    agg_vals = set()
    task_vals = set()
    site_vals = set()
    cat_vals = set()
    raw_stats = {}

    for i, p in enumerate(paths, start=1):
        df = pq.read_table(p, columns=PASS1_COLS).to_pandas()

        var_vals.update(df["var_name"].fillna("").astype(str).unique().tolist())
        source_vals.update(df["source_name"].fillna("").astype(str).unique().tolist())
        modality_vals.update(df["modality"].fillna("").astype(str).unique().tolist())
        support_vals.update(df["support_type"].fillna("").astype(str).unique().tolist())
        agg_vals.update(df["agg_type"].fillna("").astype(str).unique().tolist())
        task_vals.update(df["task_group"].fillna("").astype(str).unique().tolist())
        site_vals.update(df["site_id"].fillna("").astype(str).unique().tolist())

        cat_mask = df["is_categorical"].fillna(0).astype(np.uint8).to_numpy() == 1
        if np.any(cat_mask):
            cats = pd.to_numeric(df.loc[cat_mask, "category_value"], errors="coerce").dropna().astype(np.int64)
            if len(cats):
                cat_vals.update(cats.tolist())

        num_mask = (df["is_categorical"].fillna(0).astype(np.uint8).to_numpy() == 0)
        vals = pd.to_numeric(df["value_num"], errors="coerce").to_numpy(dtype=np.float64)
        var_names = df["var_name"].fillna("").astype(str).to_numpy()

        good = num_mask & np.isfinite(vals)
        if np.any(good):
            for vname in np.unique(var_names[good]):
                m = good & (var_names == vname)
                vv = vals[m]
                if len(vv) == 0:
                    continue
                cur = raw_stats.setdefault(vname, {"count": 0, "sum": 0.0, "sumsq": 0.0})
                cur["count"] += int(len(vv))
                cur["sum"] += float(vv.sum())
                cur["sumsq"] += float((vv * vv).sum())

        print(f"pass1 {i}/{len(paths)}")

    var_categories = sorted(var_vals)
    source_categories = sorted(source_vals)
    modality_categories = sorted(modality_vals)
    support_categories = sorted(support_vals)
    agg_categories = sorted(agg_vals)
    task_categories = sorted(task_vals)
    site_categories = sorted(site_vals)
    cat_categories = sorted(cat_vals)

    var_map = {v: i for i, v in enumerate(var_categories)}
    source_map = {v: i for i, v in enumerate(source_categories)}
    modality_map = {v: i for i, v in enumerate(modality_categories)}
    support_map = {v: i for i, v in enumerate(support_categories)}
    agg_map = {v: i for i, v in enumerate(agg_categories)}
    task_map = {v: i for i, v in enumerate(task_categories)}
    site_map = {v: i for i, v in enumerate(site_categories)}
    cat_map_int = {int(v): i + 1 for i, v in enumerate(cat_categories)}
    cat_map_meta = {str(v): i + 1 for i, v in enumerate(cat_categories)}

    stats = finalize_numeric_stats(raw_stats)

    n_vars = len(var_categories)
    var_mean = np.zeros(n_vars, dtype=np.float32)
    var_std = np.ones(n_vars, dtype=np.float32)
    for name, idx in var_map.items():
        st = stats.get(name)
        if st is not None:
            var_mean[idx] = st["mean"]
            var_std[idx] = st["std"]

    # allocate
    coord = write_memmap(f"{args.out_dir}/coord.dat", np.float32, (total_rows, 11))
    value_norm = write_memmap(f"{args.out_dir}/value_norm.dat", np.float32, (total_rows,))
    value_num = write_memmap(f"{args.out_dir}/value_num.dat", np.float32, (total_rows,))
    lat = write_memmap(f"{args.out_dir}/lat.dat", np.float32, (total_rows,))
    lon = write_memmap(f"{args.out_dir}/lon.dat", np.float32, (total_rows,))
    time_center = write_memmap(f"{args.out_dir}/time_center_sec.dat", np.int64, (total_rows,))
    time_bin = write_memmap(f"{args.out_dir}/time_bin.dat", np.int32, (total_rows,))
    var_id = write_memmap(f"{args.out_dir}/var_id.dat", np.int32, (total_rows,))
    source_id = write_memmap(f"{args.out_dir}/source_id.dat", np.int32, (total_rows,))
    modality_id = write_memmap(f"{args.out_dir}/modality_id.dat", np.int32, (total_rows,))
    support_id = write_memmap(f"{args.out_dir}/support_type_id.dat", np.int32, (total_rows,))
    agg_id = write_memmap(f"{args.out_dir}/agg_type_id.dat", np.int32, (total_rows,))
    task_id = write_memmap(f"{args.out_dir}/task_group_id.dat", np.int32, (total_rows,))
    site_id = write_memmap(f"{args.out_dir}/site_id.dat", np.int32, (total_rows,))
    category_id = write_memmap(f"{args.out_dir}/category_id.dat", np.int32, (total_rows,))
    is_categorical = write_memmap(f"{args.out_dir}/is_categorical.dat", np.uint8, (total_rows,))
    pretrain_mask = write_memmap(f"{args.out_dir}/pretrain_mask.dat", np.uint8, (total_rows,))
    finetune_train = write_memmap(f"{args.out_dir}/finetune_train_mask.dat", np.uint8, (total_rows,))
    finetune_test = write_memmap(f"{args.out_dir}/finetune_test_mask.dat", np.uint8, (total_rows,))

    pre_cut = int(pd.Timestamp(args.pretrain_cutoff, tz="UTC").timestamp())
    fin_cut = int(pd.Timestamp(args.finetune_cutoff, tz="UTC").timestamp())
    cosmos_source_id = source_map.get("cosmos", -1)
    foundation_task_id = task_map.get("foundation", -1)

    offset = 0
    cosmos_rows = []

    for i, p in enumerate(paths, start=1):
        df = pq.read_table(p, columns=PASS2_COLS).to_pandas()
        n = len(df)
        sl = slice(offset, offset + n)

        lat_arr = pd.to_numeric(df["lat"], errors="coerce").to_numpy(dtype=np.float32)
        lon_arr = pd.to_numeric(df["lon"], errors="coerce").to_numpy(dtype=np.float32)

        ts = pd.to_numeric(df["time_start_sec"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
        te = pd.to_numeric(df["time_end_sec"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
        tc = np.where(ts >= 0, (ts + te) // 2, -1).astype(np.int64)
        tb = np.where(tc >= 0, tc // 86400, -1).astype(np.int32)

        var_names = df["var_name"].fillna("").astype(str).to_numpy()
        source_names = df["source_name"].fillna("").astype(str).to_numpy()
        modality_names = df["modality"].fillna("").astype(str).to_numpy()
        support_names = df["support_type"].fillna("").astype(str).to_numpy()
        agg_names = df["agg_type"].fillna("").astype(str).to_numpy()
        task_names = df["task_group"].fillna("").astype(str).to_numpy()
        site_names = df["site_id"].fillna("").astype(str).to_numpy()

        var_ids = np.array([var_map[x] for x in var_names], dtype=np.int32)
        source_ids = np.array([source_map[x] for x in source_names], dtype=np.int32)
        modality_ids = np.array([modality_map[x] for x in modality_names], dtype=np.int32)
        support_ids = np.array([support_map[x] for x in support_names], dtype=np.int32)
        agg_ids = np.array([agg_map[x] for x in agg_names], dtype=np.int32)
        task_ids = np.array([task_map[x] for x in task_names], dtype=np.int32)
        site_ids = np.array([site_map[x] for x in site_names], dtype=np.int32)

        iscat = df["is_categorical"].fillna(0).astype(np.uint8).to_numpy()
        vals = pd.to_numeric(df["value_num"], errors="coerce").to_numpy(dtype=np.float32)

        cat_ids = np.zeros(n, dtype=np.int32)
        if np.any(iscat == 1):
            raw_cat = pd.to_numeric(df["category_value"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
            m = iscat == 1
            cat_ids[m] = np.array([cat_map_int.get(int(x), 0) for x in raw_cat[m]], dtype=np.int32)

        mu = var_mean[var_ids]
        sd = var_std[var_ids]
        vnorm = np.where(
            iscat == 0,
            (vals - mu) / np.maximum(sd, 1e-6),
            0.0,
        ).astype(np.float32)

        doy = np.where(tc >= 0, (tb % 365).astype(np.float32), -1.0)
        season_sin = np.zeros(n, dtype=np.float32)
        season_cos = np.zeros(n, dtype=np.float32)
        m = doy >= 0
        season_sin[m] = np.sin(2 * np.pi * doy[m] / 365.25)
        season_cos[m] = np.cos(2 * np.pi * doy[m] / 365.25)

        hour = np.where(tc >= 0, ((tc % 86400) / 3600.0).astype(np.float32), -1.0)
        diurnal_sin = np.zeros(n, dtype=np.float32)
        diurnal_cos = np.zeros(n, dtype=np.float32)
        m = hour >= 0
        diurnal_sin[m] = np.sin(2 * np.pi * hour[m] / 24.0)
        diurnal_cos[m] = np.cos(2 * np.pi * hour[m] / 24.0)

        dt_hours_arr = pd.to_numeric(df["dt_hours"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        dx_m_arr = pd.to_numeric(df["dx_m"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        dy_m_arr = pd.to_numeric(df["dy_m"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)

        coord_arr = np.stack([
            lat_arr,
            lon_arr,
            np.where(tc >= 0, tc.astype(np.float32) / 86400.0, -1.0),
            dt_hours_arr,
            np.log1p(np.maximum(dx_m_arr, 0.0)),
            np.log1p(np.maximum(dy_m_arr, 0.0)),
            season_sin,
            season_cos,
            diurnal_sin,
            diurnal_cos,
            iscat.astype(np.float32),
        ], axis=1).astype(np.float32)

        is_cosmos = source_ids == cosmos_source_id
        is_foundation = task_ids == foundation_task_id
        is_static = tc < 0
        is_temporal = tc >= 0

        pre_mask = (
            is_foundation &
            (is_static | (is_temporal & (tc < pre_cut)))
        ).astype(np.uint8)

        ft_train = (
            is_cosmos &
            is_temporal &
            (tc < fin_cut)
        ).astype(np.uint8)

        ft_test = (
            is_cosmos &
            is_temporal &
            (tc >= fin_cut)
        ).astype(np.uint8)

        coord[sl] = coord_arr
        value_norm[sl] = vnorm
        value_num[sl] = vals
        lat[sl] = lat_arr
        lon[sl] = lon_arr
        time_center[sl] = tc
        time_bin[sl] = tb
        var_id[sl] = var_ids
        source_id[sl] = source_ids
        modality_id[sl] = modality_ids
        support_id[sl] = support_ids
        agg_id[sl] = agg_ids
        task_id[sl] = task_ids
        site_id[sl] = site_ids
        category_id[sl] = cat_ids
        is_categorical[sl] = iscat
        pretrain_mask[sl] = pre_mask
        finetune_train[sl] = ft_train
        finetune_test[sl] = ft_test

        cosmos_rows.extend((offset + np.where(is_cosmos & is_temporal)[0]).tolist())

        offset += n
        print(f"pass2 {i}/{len(paths)}")

    # fallback split if train or test empty
    ft_train_count = int(np.sum(finetune_train[:] > 0))
    ft_test_count = int(np.sum(finetune_test[:] > 0))
    if (ft_train_count == 0 or ft_test_count == 0) and len(cosmos_rows) >= 20:
        print("Applying fallback time-based COSMOS split")
        cosmos_rows = np.asarray(cosmos_rows, dtype=np.int64)
        order = np.argsort(time_center[cosmos_rows])
        cosmos_sorted = cosmos_rows[order]
        split = max(1, min(len(cosmos_sorted) - 1, int(round(0.9 * len(cosmos_sorted)))))
        finetune_train[:] = 0
        finetune_test[:] = 0
        finetune_train[cosmos_sorted[:split]] = 1
        finetune_test[cosmos_sorted[split:]] = 1

    for mm in [
        coord, value_norm, value_num, lat, lon, time_center, time_bin,
        var_id, source_id, modality_id, support_id, agg_id, task_id, site_id,
        category_id, is_categorical, pretrain_mask, finetune_train, finetune_test
    ]:
        mm.flush()

    meta = {
        "total_rows": int(total_rows),
        "coord_dim": 11,
        "var_stats": stats,
        "registries": {
            "var_name": var_map,
            "source_name": source_map,
            "modality": modality_map,
            "support_type": support_map,
            "agg_type": agg_map,
            "task_group": task_map,
            "site_id": site_map,
            "category_value": cat_map_meta,
        },
        "cutoffs": {
            "pretrain_cutoff": args.pretrain_cutoff,
            "finetune_cutoff": args.finetune_cutoff,
        },
    }

    with open(f"{args.out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Store written")
    print("pretrain rows:", int(np.sum(pretrain_mask[:] > 0)))
    print("finetune train rows:", int(np.sum(finetune_train[:] > 0)))
    print("finetune test rows:", int(np.sum(finetune_test[:] > 0)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens_dir", required=True)
    ap.add_argument("--out_dir", default="foundation_store")
    ap.add_argument("--pretrain_cutoff", default="2020-01-01")
    ap.add_argument("--finetune_cutoff", default="2021-01-01")
    main(ap.parse_args())