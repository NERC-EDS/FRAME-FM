#!/usr/bin/env python3
from __future__ import annotations

"""
Ingest environmental datasets into parquet token shards.

Designed for the final experiment:
- foundation sources can cover a broad temporal window
- COSMOS can use a separate, longer window for downstream training/testing
- static layers are emitted once with time_start_sec=time_end_sec=-1
- output schema is consistent across all sources
"""

import argparse
import glob
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
import xarray as xr
from rasterio.transform import xy
from rasterio.warp import transform as rio_transform
from tqdm.auto import tqdm

try:
    import geopandas as gpd
except ImportError:
    gpd = None

TOKEN_COLUMNS = [
    "source_name",
    "var_name",
    "modality",
    "support_type",
    "agg_type",
    "task_group",
    "lat",
    "lon",
    "time_start_sec",
    "time_end_sec",
    "dt_hours",
    "dx_m",
    "dy_m",
    "value_num",
    "category_value",
    "is_categorical",
    "site_id",
    "units",
]

TOKEN_SCHEMA = pa.schema([
    ("source_name", pa.string()),
    ("var_name", pa.string()),
    ("modality", pa.string()),
    ("support_type", pa.string()),
    ("agg_type", pa.string()),
    ("task_group", pa.string()),
    ("lat", pa.float32()),
    ("lon", pa.float32()),
    ("time_start_sec", pa.int64()),
    ("time_end_sec", pa.int64()),
    ("dt_hours", pa.float32()),
    ("dx_m", pa.float32()),
    ("dy_m", pa.float32()),
    ("value_num", pa.float32()),
    ("category_value", pa.int32()),
    ("is_categorical", pa.uint8()),
    ("site_id", pa.string()),
    ("units", pa.string()),
])


@dataclass
class IngestConfig:
    out_dir: str
    shard_rows: int
    max_rows_per_static: int
    seed: int
    foundation_group_name: str = "foundation"
    downstream_group_name: str = "cosmos_downstream"
    chess_workers: int = 4
    chess_time_batch: int = 16
    chess_engine: str = "h5netcdf"
    foundation_start_date: str = "2015-01-01"
    foundation_end_date: str = "2020-01-01"
    cosmos_start_date: str = "2014-01-01"
    cosmos_end_date: str = "2024-12-31"
    require_cosmos: bool = True


class BufferedParquetWriter:
    def __init__(self, out_dir: str, prefix: str, shard_rows: int):
        self.out_dir = out_dir
        self.prefix = prefix
        self.shard_rows = int(shard_rows)
        self.tables: list[pa.Table] = []
        self.n_rows = 0
        self.part = 0
        self.total_rows_written = 0
        self.files_written: list[str] = []
        os.makedirs(out_dir, exist_ok=True)

    def write_dict(self, arrays: dict[str, np.ndarray]) -> None:
        if not arrays:
            return
        n = len(next(iter(arrays.values())))
        if n == 0:
            return
        table = pa.Table.from_pydict(arrays, schema=TOKEN_SCHEMA)
        self.tables.append(table)
        self.n_rows += table.num_rows
        self.total_rows_written += table.num_rows
        if self.n_rows >= self.shard_rows:
            self.flush()

    def flush(self) -> None:
        if not self.tables:
            return
        table = pa.concat_tables(self.tables, promote_options="default")
        path = os.path.join(self.out_dir, f"{self.prefix}_part_{self.part:05d}.parquet")
        pq.write_table(table, path, compression="zstd")
        self.files_written.append(path)
        self.tables.clear()
        self.n_rows = 0
        self.part += 1

    def close(self) -> None:
        self.flush()


def parse_window(start_date: str, end_date: str) -> tuple[int, int]:
    start_sec = int(pd.Timestamp(start_date, tz="UTC").timestamp())
    end_sec = int(pd.Timestamp(end_date, tz="UTC").timestamp())
    if end_sec <= start_sec:
        raise ValueError("end_date must be after start_date")
    return start_sec, end_sec


def overlap_mask(start_sec: np.ndarray, end_sec: np.ndarray, window_start: int, window_end: int) -> np.ndarray:
    return (end_sec > window_start) & (start_sec < window_end)


def to_epoch_seconds(values) -> np.ndarray:
    arr = np.asarray(values)
    flat = arr.ravel()
    out = np.full(flat.shape, -1, dtype=np.int64)
    for i, v in enumerate(flat):
        try:
            ts = pd.to_datetime(v, utc=True, errors="coerce")
            if not pd.isna(ts):
                out[i] = int(ts.timestamp())
        except Exception:
            pass
    return out.reshape(arr.shape)


def infer_interval_bounds_from_starts(start_sec: np.ndarray, default_dt_sec: int) -> tuple[np.ndarray, np.ndarray]:
    start_sec = np.asarray(start_sec, dtype=np.int64)
    n = len(start_sec)
    if n == 0:
        return start_sec.copy(), start_sec.copy()
    if n == 1:
        return start_sec.copy(), start_sec + default_dt_sec

    diffs = np.diff(start_sec)
    pos = diffs[diffs > 0]
    step = int(np.median(pos)) if len(pos) else int(default_dt_sec)
    step = max(step, 1)

    end_sec = np.empty_like(start_sec)
    end_sec[:-1] = np.where(diffs > 0, start_sec[:-1] + diffs, start_sec[:-1] + step)
    end_sec[-1] = start_sec[-1] + step
    return start_sec, end_sec


def parse_dates_from_filename(path: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    m = re.search(r"(\d{8})-(\d{8})", os.path.basename(path))
    if not m:
        return None
    s, e = m.groups()
    return pd.Timestamp(s, tz="UTC"), pd.Timestamp(e, tz="UTC")


def chess_file_in_timerange(path: str, window_start: int, window_end: int) -> bool:
    dates = parse_dates_from_filename(path)
    if dates is None:
        return True
    file_start_ts, file_end_ts = dates
    file_start = int(file_start_ts.timestamp())
    file_end = int((file_end_ts + pd.Timedelta(days=1)).timestamp())
    return (file_end > window_start) and (file_start < window_end)


def _constant_str_array(value: str, n: int) -> np.ndarray:
    return np.full(n, value, dtype=object)


def _infer_xy_names(da: xr.DataArray) -> tuple[str, str]:
    dims = list(da.dims)
    x_name = next((d for d in dims if d.lower() in {"x", "lon", "longitude", "rlon"}), dims[-1])
    y_name = next((d for d in dims if d.lower() in {"y", "lat", "latitude", "rlat"}), dims[-2])
    return y_name, x_name


def _infer_time_name(da: xr.DataArray) -> Optional[str]:
    for d in da.dims:
        if d.lower() in {"time", "t", "date", "datetime"}:
            return d
    return None


def build_grid_cache(y_native: np.ndarray, x_native: np.ndarray):
    xx_native, yy_native = np.meshgrid(x_native, y_native)
    lon_flat, lat_flat = rio_transform(
        "EPSG:27700",
        "EPSG:4326",
        xx_native.ravel().tolist(),
        yy_native.ravel().tolist(),
    )
    dx_m = float(np.nanmedian(np.abs(np.diff(x_native)))) if len(x_native) > 1 else np.nan
    dy_m = float(np.nanmedian(np.abs(np.diff(y_native)))) if len(y_native) > 1 else np.nan
    return (
        np.asarray(lat_flat, dtype=np.float32),
        np.asarray(lon_flat, dtype=np.float32),
        np.float32(dx_m),
        np.float32(dy_m),
    )


def process_chess_file(
        path: str,
        out_dir: str,
        prefix: str,
        source_name: str,
        var_names: list[str],
        task_group: str,
        shard_rows: int,
        time_batch: int,
        engine: str,
        window_start: int,
        window_end: int,
) -> tuple[str, int]:
    pid = os.getpid()
    writer = BufferedParquetWriter(
        out_dir=out_dir,
        prefix=f"{prefix}_pid{pid}_{Path(path).stem}",
        shard_rows=shard_rows,
    )
    rows_written = 0
    ds = xr.open_dataset(path, decode_times=True, engine=engine)

    try:
        sample_da = None
        for raw_var in var_names:
            if raw_var in ds:
                sample_da = ds[raw_var]
                break
        if sample_da is None:
            return path, 0

        y_name, x_name = _infer_xy_names(sample_da)
        y_native = np.asarray(ds[y_name].values, dtype=np.float64)
        x_native = np.asarray(ds[x_name].values, dtype=np.float64)
        lat_flat, lon_flat, dx_m, dy_m = build_grid_cache(y_native, x_native)

        for raw_var in var_names:
            if raw_var not in ds:
                continue

            da = ds[raw_var]
            time_name = _infer_time_name(da)
            if time_name is None:
                continue

            y_name2, x_name2 = _infer_xy_names(da)
            da = da.transpose(time_name, y_name2, x_name2)

            time_values = to_epoch_seconds(da[time_name].values)
            keep_idx = np.where(time_values >= 0)[0]
            if keep_idx.size == 0:
                continue

            time_values = time_values[keep_idx]
            time_start_all, time_end_all = infer_interval_bounds_from_starts(time_values, 86400)
            da = da.isel({time_name: keep_idx})

            keep_time = overlap_mask(time_start_all, time_end_all, window_start, window_end)
            if not np.any(keep_time):
                continue

            da = da.isel({time_name: np.where(keep_time)[0]})
            time_start = time_start_all[keep_time]
            time_end = time_end_all[keep_time]
            dt_hours = ((time_end - time_start) / 3600.0).astype(np.float32)

            nt = da.sizes[time_name]
            if nt == 0:
                continue

            probe = np.asarray(da.values, dtype=np.float32).reshape(nt, -1)
            valid_cell_idx = np.flatnonzero(np.any(np.isfinite(probe), axis=0))
            if valid_cell_idx.size == 0:
                continue

            lat_valid = lat_flat[valid_cell_idx]
            lon_valid = lon_flat[valid_cell_idx]
            units = str(da.attrs.get("units", ""))
            agg_type = "sum" if raw_var == "precip" else "mean"

            for start in range(0, nt, time_batch):
                stop = min(start + time_batch, nt)
                block = np.asarray(da.isel({time_name: slice(start, stop)}).values, dtype=np.float32)
                tb = block.shape[0]
                vals2 = block.reshape(tb, -1)[:, valid_cell_idx]
                mask = np.isfinite(vals2)
                if not mask.any():
                    continue

                t_idx, j_idx = np.where(mask)
                vals = vals2[t_idx, j_idx]
                ts = time_start[start:stop][t_idx]
                te = time_end[start:stop][t_idx]
                dth = dt_hours[start:stop][t_idx]
                n = vals.shape[0]

                writer.write_dict({
                    "source_name": np.full(n, source_name, dtype=object),
                    "var_name": np.full(n, raw_var, dtype=object),
                    "modality": np.full(n, "grid_numeric", dtype=object),
                    "support_type": np.full(n, "gridcell", dtype=object),
                    "agg_type": np.full(n, agg_type, dtype=object),
                    "task_group": np.full(n, task_group, dtype=object),
                    "lat": lat_valid[j_idx],
                    "lon": lon_valid[j_idx],
                    "time_start_sec": ts.astype(np.int64, copy=False),
                    "time_end_sec": te.astype(np.int64, copy=False),
                    "dt_hours": dth.astype(np.float32, copy=False),
                    "dx_m": np.full(n, float(dx_m), dtype=np.float32),
                    "dy_m": np.full(n, float(dy_m), dtype=np.float32),
                    "value_num": vals.astype(np.float32, copy=False),
                    "category_value": np.full(n, -1, dtype=np.int32),
                    "is_categorical": np.zeros(n, dtype=np.uint8),
                    "site_id": np.full(n, "", dtype=object),
                    "units": np.full(n, units, dtype=object),
                })
                rows_written += n
    finally:
        ds.close()
        writer.close()

    return path, rows_written


def ingest_chess_dataset_parallel(
        out_dir: str,
        nc_glob: str,
        source_name: str,
        var_names: list[str],
        task_group: str,
        shard_rows: int,
        time_batch: int,
        workers: int,
        engine: str,
        window_start: int,
        window_end: int,
) -> int:
    files_all = sorted(f for f in glob.glob(nc_glob) if f.lower().endswith(".nc"))
    if not files_all:
        raise FileNotFoundError(f"No .nc files matched: {nc_glob}")

    files = [f for f in files_all if chess_file_in_timerange(f, window_start, window_end)]
    if not files:
        raise FileNotFoundError(f"No CHESS files overlapped the requested window")

    total_rows = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(
                process_chess_file,
                path=f,
                out_dir=out_dir,
                prefix=source_name,
                source_name=source_name,
                var_names=var_names,
                task_group=task_group,
                shard_rows=shard_rows,
                time_batch=time_batch,
                engine=engine,
                window_start=window_start,
                window_end=window_end,
            )
            for f in files
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{source_name} files", unit="file"):
            _, rows = fut.result()
            total_rows += rows
    return total_rows


def ingest_static_multiband_raster(
        writer: BufferedParquetWriter,
        raster_path: str,
        source_name: str,
        var_names: list[str],
        task_group: str,
        max_rows: int,
        seed: int,
) -> int:
    before = writer.total_rows_written
    rng = np.random.default_rng(seed)

    with rasterio.open(raster_path) as src:
        if src.count != len(var_names):
            raise ValueError(f"{raster_path} has {src.count} bands but got {len(var_names)} names")

        dx_m = float(abs(src.transform.a)) if src.transform else np.nan
        dy_m = float(abs(src.transform.e)) if src.transform else np.nan

        for band_idx in range(1, src.count + 1):
            arr = src.read(band_idx)
            nodata = src.nodata
            mask = np.isfinite(arr)
            if nodata is not None:
                mask &= arr != nodata

            ys, xs = np.where(mask)
            if ys.size == 0:
                continue
            if ys.size > max_rows:
                keep = rng.choice(ys.size, size=max_rows, replace=False)
                ys = ys[keep]
                xs = xs[keep]

            xcoords, ycoords = xy(src.transform, ys, xs, offset="center")
            xcoords = np.asarray(xcoords, dtype=np.float64)
            ycoords = np.asarray(ycoords, dtype=np.float64)

            if src.crs is not None and str(src.crs).upper() not in ("EPSG:4326", "OGC:CRS84"):
                lon, lat = rio_transform(src.crs, "EPSG:4326", xcoords.tolist(), ycoords.tolist())
                lon = np.asarray(lon, dtype=np.float64)
                lat = np.asarray(lat, dtype=np.float64)
            else:
                lon = xcoords
                lat = ycoords

            vals = arr[ys, xs].astype(np.float32, copy=False)
            n = vals.shape[0]

            writer.write_dict({
                "source_name": _constant_str_array(source_name, n),
                "var_name": _constant_str_array(var_names[band_idx - 1], n),
                "modality": _constant_str_array("grid_numeric", n),
                "support_type": _constant_str_array("gridcell", n),
                "agg_type": _constant_str_array("static", n),
                "task_group": _constant_str_array(task_group, n),
                "lat": lat.astype(np.float32, copy=False),
                "lon": lon.astype(np.float32, copy=False),
                "time_start_sec": np.full(n, -1, dtype=np.int64),
                "time_end_sec": np.full(n, -1, dtype=np.int64),
                "dt_hours": np.zeros(n, dtype=np.float32),
                "dx_m": np.full(n, dx_m, dtype=np.float32),
                "dy_m": np.full(n, dy_m, dtype=np.float32),
                "value_num": vals,
                "category_value": np.full(n, -1, dtype=np.int32),
                "is_categorical": np.zeros(n, dtype=np.uint8),
                "site_id": np.full(n, "", dtype=object),
                "units": np.full(n, "percent_cover", dtype=object),
            })

    return writer.total_rows_written - before


def ingest_static_vector(
        writer: BufferedParquetWriter,
        vector_path: str,
        source_name: str,
        var_name: str,
        field_name: str,
        task_group: str,
        max_rows: int,
        seed: int,
) -> int:
    if gpd is None:
        raise ImportError("geopandas is required for vector static layers")

    before = writer.total_rows_written
    rng = np.random.default_rng(seed)

    gdf = gpd.read_file(vector_path)
    if gdf.empty:
        return 0
    if gdf.crs is None:
        raise ValueError(f"Vector layer has no CRS: {vector_path}")
    if field_name not in gdf.columns:
        raise ValueError(f"Field '{field_name}' not found in {vector_path}")

    gdf = gdf[[field_name, gdf.geometry.name]].copy()
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[gdf[field_name].notnull()].copy()
    if gdf.empty:
        return 0

    if len(gdf) > max_rows:
        keep = rng.choice(len(gdf), size=max_rows, replace=False)
        gdf = gdf.iloc[keep].copy()

    gdf_ll = gdf.to_crs("EPSG:4326")
    reps = gdf_ll.geometry.representative_point()

    vals = pd.to_numeric(gdf_ll[field_name], errors="coerce").to_numpy()
    valid = np.isfinite(vals)
    if not np.any(valid):
        return 0

    vals = vals[valid].astype(np.float32, copy=False)
    lon = reps.x.to_numpy(dtype=np.float64)[valid]
    lat = reps.y.to_numpy(dtype=np.float64)[valid]
    n = len(vals)

    writer.write_dict({
        "source_name": _constant_str_array(source_name, n),
        "var_name": _constant_str_array(var_name, n),
        "modality": _constant_str_array("vector_numeric", n),
        "support_type": _constant_str_array("polygon", n),
        "agg_type": _constant_str_array("static", n),
        "task_group": _constant_str_array(task_group, n),
        "lat": lat.astype(np.float32, copy=False),
        "lon": lon.astype(np.float32, copy=False),
        "time_start_sec": np.full(n, -1, dtype=np.int64),
        "time_end_sec": np.full(n, -1, dtype=np.int64),
        "dt_hours": np.zeros(n, dtype=np.float32),
        "dx_m": np.full(n, np.nan, dtype=np.float32),
        "dy_m": np.full(n, np.nan, dtype=np.float32),
        "value_num": vals,
        "category_value": np.full(n, -1, dtype=np.int32),
        "is_categorical": np.zeros(n, dtype=np.uint8),
        "site_id": np.full(n, "", dtype=object),
        "units": np.full(n, "", dtype=object),
    })
    return writer.total_rows_written - before


def ingest_cosmos_folder(
        writer: BufferedParquetWriter,
        cosmos_dir: str,
        task_group: str,
        window_start: int,
        window_end: int,
        value_column: str,
        site_meta_csv: str,
) -> dict:
    meta = pd.read_csv(site_meta_csv)
    cols_upper = {c.upper(): c for c in meta.columns}
    site_col = cols_upper.get("SITE_ID") or cols_upper.get("SITE") or cols_upper.get("ID")
    lat_col = cols_upper.get("LAT") or cols_upper.get("LATITUDE")
    lon_col = cols_upper.get("LON") or cols_upper.get("LONGITUDE") or cols_upper.get("LONG")
    if site_col is None or lat_col is None or lon_col is None:
        raise ValueError("site_meta_csv must contain site id + lat + lon columns")

    site_lookup = {}
    for _, row in meta[[site_col, lat_col, lon_col]].dropna().iterrows():
        site_lookup[str(row[site_col]).upper()] = (float(row[lat_col]), float(row[lon_col]))

    selected_files = sorted(
        f for f in glob.glob(os.path.join(cosmos_dir, "*.csv"))
        if "_daily_" in os.path.basename(f).lower() and not os.path.basename(f).lower().endswith("_flags.csv")
    )
    if not selected_files:
        raise ValueError("No matching COSMOS daily files found")

    window_start_ts = pd.Timestamp(window_start, unit="s", tz="UTC")
    window_end_ts = pd.Timestamp(window_end, unit="s", tz="UTC")

    rows_total = 0
    files_with_rows = 0
    files_no_overlap = 0

    for csv_path in selected_files:
        df = pd.read_csv(csv_path, low_memory=False)
        cols_upper = {c.upper(): c for c in df.columns}
        dt_col = cols_upper.get("DATE_TIME")
        site_col = cols_upper.get("SITE_ID")
        val_col = cols_upper.get(value_column.upper())

        if dt_col is None or site_col is None or val_col is None:
            continue

        ts = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
        vals = pd.to_numeric(df[val_col], errors="coerce").replace(-9999, np.nan)
        site_ids = df[site_col].astype(str).str.upper()

        valid = ts.notnull() & vals.notnull() & site_ids.notnull()
        if not valid.any():
            continue

        ts = ts.loc[valid]
        vals = vals.loc[valid].to_numpy(dtype=np.float32)
        site_ids = site_ids.loc[valid].to_numpy(dtype=object)

        coords = pd.Series(site_ids).map(pd.Series(site_lookup))
        good = coords.notnull().to_numpy()
        if not np.any(good):
            continue

        ts = ts.iloc[good]
        vals = vals[good]
        site_ids = site_ids[good]
        coords = coords.iloc[good]
        lat = np.array([xy[0] for xy in coords], dtype=np.float32)
        lon = np.array([xy[1] for xy in coords], dtype=np.float32)

        start_dt = ts.dt.floor("D")
        end_dt = start_dt + pd.Timedelta(days=1)
        keep = (end_dt > window_start_ts) & (start_dt < window_end_ts)
        keep_n = int(keep.sum())

        if keep_n == 0:
            files_no_overlap += 1
            continue

        keep_np = keep.to_numpy()
        start_dt = start_dt.loc[keep]
        end_dt = end_dt.loc[keep]
        vals = vals[keep_np]
        site_ids = site_ids[keep_np]
        lat = lat[keep_np]
        lon = lon[keep_np]

        time_start = to_epoch_seconds(start_dt.to_numpy())
        time_end = to_epoch_seconds(end_dt.to_numpy())
        n = len(vals)

        writer.write_dict({
            "source_name": _constant_str_array("cosmos", n),
            "var_name": _constant_str_array(value_column.lower(), n),
            "modality": _constant_str_array("point_numeric", n),
            "support_type": _constant_str_array("point", n),
            "agg_type": _constant_str_array("daily_mean", n),
            "task_group": _constant_str_array(task_group, n),
            "lat": lat,
            "lon": lon,
            "time_start_sec": time_start,
            "time_end_sec": time_end,
            "dt_hours": np.full(n, 24.0, dtype=np.float32),
            "dx_m": np.full(n, np.nan, dtype=np.float32),
            "dy_m": np.full(n, np.nan, dtype=np.float32),
            "value_num": vals.astype(np.float32, copy=False),
            "category_value": np.full(n, -1, dtype=np.int32),
            "is_categorical": np.zeros(n, dtype=np.uint8),
            "site_id": site_ids.astype(object, copy=False),
            "units": _constant_str_array("vwc", n),
        })

        rows_total += n
        files_with_rows += 1

    writer.close()

    return {
        "rows_total": rows_total,
        "files_selected": len(selected_files),
        "files_with_rows": files_with_rows,
        "files_no_overlap": files_no_overlap,
        "files_written": list(writer.files_written),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/tokens_out")

    p.add_argument("--chess_met_glob", required=True)
    p.add_argument("--chess_met_vars", default="precip,huss,rsds,rlds,sfcWind,tas")

    p.add_argument("--land_cover_raster", required=True)
    p.add_argument("--soil_carbon_vector", required=True)
    p.add_argument("--soil_ph_vector", required=True)
    p.add_argument("--bulk_density_vector", required=True)

    p.add_argument("--cosmos_dir", required=True)
    p.add_argument("--cosmos_site_meta_csv", required=True)

    p.add_argument("--foundation_start_date", default="2015-01-01")
    p.add_argument("--foundation_end_date", default="2020-01-01")
    p.add_argument("--cosmos_start_date", default="2014-01-01")
    p.add_argument("--cosmos_end_date", default="2024-12-31")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard_rows", type=int, default=500_000)
    p.add_argument("--max_rows_per_static", type=int, default=1_000_000)
    p.add_argument("--chess_workers", type=int, default=max(1, min(4, (os.cpu_count() or 4) - 1)))
    p.add_argument("--chess_time_batch", type=int, default=16)
    p.add_argument("--chess_engine", default="h5netcdf", choices=["h5netcdf", "netcdf4"])
    p.add_argument("--allow_zero_cosmos", action="store_true")
    return p.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()

    cfg = IngestConfig(
        out_dir=args.out_dir,
        shard_rows=args.shard_rows,
        max_rows_per_static=args.max_rows_per_static,
        seed=args.seed,
        chess_workers=args.chess_workers,
        chess_time_batch=args.chess_time_batch,
        chess_engine=args.chess_engine,
        foundation_start_date=args.foundation_start_date,
        foundation_end_date=args.foundation_end_date,
        cosmos_start_date=args.cosmos_start_date,
        cosmos_end_date=args.cosmos_end_date,
        require_cosmos=(not args.allow_zero_cosmos),
    )

    os.makedirs(cfg.out_dir, exist_ok=True)

    foundation_start, foundation_end = parse_window(cfg.foundation_start_date, cfg.foundation_end_date)
    cosmos_start, cosmos_end = parse_window(cfg.cosmos_start_date, cfg.cosmos_end_date)

    rows_by_source = {}
    artifacts = {}

    met_vars = [v.strip() for v in args.chess_met_vars.split(",") if v.strip()]
    rows_by_source["chess_met"] = ingest_chess_dataset_parallel(
        out_dir=cfg.out_dir,
        nc_glob=args.chess_met_glob,
        source_name="chess_met",
        var_names=met_vars,
        task_group=cfg.foundation_group_name,
        shard_rows=cfg.shard_rows,
        time_batch=cfg.chess_time_batch,
        workers=cfg.chess_workers,
        engine=cfg.chess_engine,
        window_start=foundation_start,
        window_end=foundation_end,
    )
    artifacts["chess_met"] = sorted(glob.glob(os.path.join(cfg.out_dir, "chess_met*.parquet")))

    static_writer = BufferedParquetWriter(cfg.out_dir, prefix="static", shard_rows=cfg.shard_rows)

    rows_by_source["land_cover"] = ingest_static_multiband_raster(
        static_writer,
        args.land_cover_raster,
        "land_cover",
        [f"lcm_band_{i}" for i in range(1, 11)],
        cfg.foundation_group_name,
        cfg.max_rows_per_static,
        cfg.seed + 1,
    )

    rows_by_source["soil_carbon"] = ingest_static_vector(
        static_writer,
        args.soil_carbon_vector,
        "soil_carbon",
        "topsoil_carbon",
        "CCONC_07",
        cfg.foundation_group_name,
        cfg.max_rows_per_static,
        cfg.seed + 2,
    )

    rows_by_source["soil_ph"] = ingest_static_vector(
        static_writer,
        args.soil_ph_vector,
        "soil_ph",
        "soil_ph",
        "PH_07",
        cfg.foundation_group_name,
        cfg.max_rows_per_static,
        cfg.seed + 3,
    )

    rows_by_source["bulk_density"] = ingest_static_vector(
        static_writer,
        args.bulk_density_vector,
        "bulk_density",
        "bulk_density",
        "BULKD_07",
        cfg.foundation_group_name,
        cfg.max_rows_per_static,
        cfg.seed + 4,
    )

    static_writer.close()
    artifacts["static"] = list(static_writer.files_written)

    cosmos_writer = BufferedParquetWriter(cfg.out_dir, prefix="cosmos", shard_rows=cfg.shard_rows)
    cosmos_info = ingest_cosmos_folder(
        cosmos_writer,
        cosmos_dir=args.cosmos_dir,
        task_group=cfg.downstream_group_name,
        window_start=cosmos_start,
        window_end=cosmos_end,
        value_column="COSMOS_VWC",
        site_meta_csv=args.cosmos_site_meta_csv,
    )
    rows_by_source["cosmos"] = cosmos_info["rows_total"]
    artifacts["cosmos"] = cosmos_info["files_written"]

    if cfg.require_cosmos and rows_by_source["cosmos"] == 0:
        raise RuntimeError("COSMOS ingest wrote 0 rows")

    manifest = {
        "schema": TOKEN_COLUMNS,
        "rows_by_source": rows_by_source,
        "artifacts": artifacts,
        "config": {
            "foundation_start_date": cfg.foundation_start_date,
            "foundation_end_date": cfg.foundation_end_date,
            "cosmos_start_date": cfg.cosmos_start_date,
            "cosmos_end_date": cfg.cosmos_end_date,
            "shard_rows": cfg.shard_rows,
            "max_rows_per_static": cfg.max_rows_per_static,
        },
        "cosmos_summary": cosmos_info,
    }

    with open(os.path.join(cfg.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nSummary")
    for k, v in rows_by_source.items():
        print(f"  {k}: {v:,} rows")
    print(f"\nWrote parquet token shards to {cfg.out_dir}")


if __name__ == "__main__":
    main()
