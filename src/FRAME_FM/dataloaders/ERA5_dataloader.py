# src/FRAME_FM/dataloaders/ERA5_dataloader.py
#
# A time-aware ERA5 dataloader for FRAME-FM.
#
# Big picture:
# - Open ERA5 through a kerchunk reference JSON.
# - Pick a few variables such as d2m / t2m / u10.
# - Optionally crop to a time window.
# - Cut the global latitude-longitude grid into smaller tiles.
# - Optionally group consecutive timesteps together so one sample is a short
#   spatiotemporal cube instead of a single 2D map.

# ERA5SpatialPixelsDataModule
#    Returns values + explicit per-pixel coordinates.
#    Each sample is:
#       values    -> (C, T, H, W)
#       times     -> (T,)
#       positions -> (3, T, H, W) for (time, lat, lon)

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import TensorDataset
import xarray as xr

from FRAME_FM.utils.LightningDataModuleWrapper import BaseDataModule
from FRAME_FM.datasets.InputTimeCoords_Dataset import TransformedInputTimeCoordsDataset
from FRAME_FM.transforms.transforms import (
    TilerTransform,
    tiled_to_pixel_coordinates,
    tiled_to_coordinate_bounds,
    TiledIndexMapper,
)


class ERA5BaseDataModule(BaseDataModule):
    """Shared ERA5 loading and tiling utilities."""

    train_dataset: torch.utils.data.Dataset
    val_dataset: torch.utils.data.Dataset
    test_dataset: torch.utils.data.Dataset | None

    def __init__(
        self,
        data_root: str,
        # What to load from ERA5.
        variables: Sequence[str],
        time_min: Optional[str] = None,
        time_max: Optional[str] = None,
        # How to tile the global grid.
        tile_size_lat: int = 64,
        tile_size_lon: int = 64,
        # How many consecutive timesteps to group into one sample.
        time_slice_size: int = 1,
        # Convert ERA5 longitudes from [0, 360) to [-180, 180) if wanted.
        convert_longitude_to_180: bool = True,
        # Standard Lightning / DataLoader arguments.
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        train_split: float = 0.85,
        val_split: float = 0.15,
        test_split: float = 0.0,
        split_strategy: str = "fraction",
        train_transforms: Callable | None = None,
        val_transforms: Callable | None = None,
        test_transforms: Callable | None = None,
        # Optional explicit split indices.
        train_indices: Optional[Sequence[int]] = None,
        val_indices: Optional[Sequence[int]] = None,
        test_indices: Optional[Sequence[int]] = None,
        # Optional xarray / dask chunking when opening the dataset.
        chunks: Optional[dict] = None,
        # Debug prints are useful while building, but noisy in training.
        debug: bool = True,
    ) -> None:
        super().__init__(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            split_strategy=split_strategy,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
        )

        if time_slice_size < 1:
            raise ValueError("time_slice_size must be at least 1.")

        self.variables = list(variables)
        self.time_min = time_min
        self.time_max = time_max
        self.tile_size_lat = tile_size_lat
        self.tile_size_lon = tile_size_lon
        self.time_slice_size = time_slice_size
        self.convert_longitude_to_180 = convert_longitude_to_180
        self.chunks = chunks
        self.debug = debug

        self._global_lat: np.ndarray | None = None
        self._global_lon: np.ndarray | None = None

    def _log(self, *args) -> None:
        if self.debug:
            print(*args)

    def _load_raw_data(self) -> xr.DataArray:
        open_kwargs = {"engine": "kerchunk"}
        if self.chunks is not None:
            open_kwargs["chunks"] = self.chunks

        ds = xr.open_dataset(self.data_root, **open_kwargs)

        missing = [v for v in self.variables if v not in ds.data_vars]
        if missing:
            raise ValueError(
                f"Some requested variables are not present in dataset: {missing}. "
                f"Available variables include: {list(ds.data_vars)[:20]}..."
            )
        ds = ds[self.variables]

        # Optionally crop the time range before tiling.
        if self.time_min is not None or self.time_max is not None:
            t0 = np.datetime64(self.time_min) if self.time_min is not None else ds["time"].min().values
            t1 = np.datetime64(self.time_max) if self.time_max is not None else ds["time"].max().values
            ds = ds.sel(time=slice(t0, t1))

        # ERA5 longitude is often stored as 0..360. For humans, -180..180 is usually easier to think about.
        if self.convert_longitude_to_180 and "longitude" in ds.coords:
            lon_180 = ((ds["longitude"] + 180) % 360) - 180
            ds = ds.assign_coords(longitude=lon_180).sortby("longitude")

        arr = ds.to_array(dim="channel")

        # Keep the original global coordinate axes so subclasses can later build
        # real lat/lon/time metadata for each tile.
        self._global_lat = arr["latitude"].values
        self._global_lon = arr["longitude"].values
        self._global_time = arr["time"].values

        # Keep the loader robust: fail early if ERA5-style coordinates are missing.
        for req in ["time", "latitude", "longitude"]:
            if req not in arr.coords and req not in arr.dims:
                raise ValueError(
                    f"Expected ERA5-style coord/dim '{req}' not found. "
                    f"Found dims={arr.dims}, coords={list(arr.coords)}"
                )

        # Put channel first because downstream model code is written with that in mind.
        arr = arr.transpose("channel", "time", "latitude", "longitude")
        self._log("Raw ERA5 array shape:", arr.shape)
        self._log("Raw ERA5 array dims :", arr.dims)
        return arr

    def _tile_array(self, arr: xr.DataArray) -> xr.DataArray:
        if "latitude" not in arr.dims or "longitude" not in arr.dims:
            raise ValueError(f"Expected latitude/longitude dims, got {arr.dims}.")

        n_lat = arr.sizes["latitude"]
        n_lon = arr.sizes["longitude"]
        if n_lat < self.tile_size_lat or n_lon < self.tile_size_lon:
            raise ValueError(
                "ERA5 grid is smaller than the requested tile size: "
                f"{n_lat}x{n_lon} < {self.tile_size_lat}x{self.tile_size_lon}"
            )

        # Use the shared transform-layer tiler so ERA5 follows the same tiling
        # contract as other geospatial dataloaders.
        tiles = TilerTransform(
            boundary="pad",
            time=self.time_slice_size,
            latitude=self.tile_size_lat,
            longitude=self.tile_size_lon,
        )(arr)

        # Final per-sample layout expected by model code.
        tiles = tiles.transpose("batch_dim", "channel", "time_fine", "latitude_fine", "longitude_fine")

        # Padding introduced by xarray becomes NaN; fill with zeros for tensors.
        tiles = tiles.fillna(0)

        self._log(
            "Each sample shape (C, T, H, W) =",
            (
                tiles.sizes["channel"],
                tiles.sizes["time_fine"],
                tiles.sizes["latitude_fine"],
                tiles.sizes["longitude_fine"],
            ),
        )
        self._log("Tiled array shape:", tiles.shape)
        self._log("Tiled array dims :", tiles.dims)
        return tiles

    def _extract_times(self, tiles: xr.DataArray) -> torch.Tensor:
        """
        Extract the real timestamps for each tiled sample.

        Returns:
            times of shape (N, T)
        where:
            N = number of tiled samples
            T = time_slice_size

        Values are integer seconds since Unix epoch.
        This is convenient because it is easy to store, compare, and feed into
        coordinate logic later.
        """

        # After xarray construct+stack, the remaining "time" coordinate is aligned
        # with (batch_dim, time_inner). We convert it into Unix seconds.
        time_np = np.asarray(tiles["time"].values).astype("datetime64[s]").astype("int64")

        # If T=1, xarray may hand back shape (N,) instead of (N, 1).
        # We normalise to always return (N, T).
        if time_np.ndim == 1:
            time_np = time_np[:, None]
        return torch.tensor(time_np, dtype=torch.int64)

    def _extract_time_bounds(self, times: torch.Tensor) -> torch.Tensor:
        """
        Convert a per-sample time window into min/max bounds.

        Input:
            times -> (N, T)

        Output:
            time_bounds -> (N, 1, 2)
        where each item is [t_min, t_max].
        """

        if times.ndim != 2:
            raise ValueError(f"Expected times with shape (N, T), got {tuple(times.shape)}")

        time_bounds = torch.stack(
            [times.amin(dim=1), times.amax(dim=1)],
            dim=-1,
        )
        return time_bounds.unsqueeze(1).to(torch.float32)

    def _pad_axis_slice(self, axis_slice: np.ndarray, target_len: int) -> np.ndarray:
        """
        Pad a 1D coordinate slice by repeating the final coordinate.

        This mirrors what tiling with boundary="pad" does for the data itself.
        It matters only for edge tiles where the global grid is not perfectly
        divisible by tile size.
        """

        axis_slice = np.asarray(axis_slice)
        if axis_slice.size == 0:
            raise RuntimeError("Encountered an empty coordinate slice while building metadata.")
        if axis_slice.size >= target_len:
            return axis_slice[:target_len]
        pad = np.full(target_len - axis_slice.size, axis_slice[-1], dtype=axis_slice.dtype)
        return np.concatenate([axis_slice, pad], axis=0)


class ERA5SpatialPixelsDataModule(ERA5TiledBaseDataModule):
    """
    ERA5 loader that returns full per-pixel spatiotemporal coordinates.

    Each sample is:
        values    -> (C, T, H, W)
        times     -> (T,)
        positions -> (3, T, H, W)

    Position channels are ordered as:
        0 -> time
        1 -> latitude
        2 -> longitude

    This matches STPatchEmbed / inputs_positioned="pixels" for time-aware
    spatiotemporal inputs.
    """

    def _extract_pixel_positions(self, tiles: xr.DataArray) -> torch.Tensor:
        """
        Build a real coordinate grid for every sample.

        Output shape:
            (N, 3, T, H, W)

        Intuition:
        - values tell the model "what happened"
        - positions tell the model "where and when it happened"
        """

        if self._global_lat is None or self._global_lon is None:
            raise RuntimeError("Global coordinates were not stored before position extraction.")

        batch_tuples = tiles["batch_dim"].values
        n_samples = tiles.sizes["batch_dim"]
        t_size = tiles.sizes["time_inner"]
        h_size = tiles.sizes["tile_lat"]
        w_size = tiles.sizes["tile_lon"]

        # Use the tile-aligned time coordinate created by xarray, not the batch index.
        time_np = np.asarray(tiles["time"].values).astype("datetime64[s]").astype("int64")
        if time_np.ndim == 1:
            time_np = time_np[:, None]

        pos = torch.empty((n_samples, 3, t_size, h_size, w_size), dtype=torch.float32)

        for i, (_, tile_lat_id, tile_lon_id) in enumerate(batch_tuples):
            lat_start = int(tile_lat_id) * self.tile_size_lat
            lon_start = int(tile_lon_id) * self.tile_size_lon

            lat_slice = self._pad_axis_slice(self._global_lat[lat_start: lat_start + h_size], h_size)
            lon_slice = self._pad_axis_slice(self._global_lon[lon_start: lon_start + w_size], w_size)

            lat_2d = torch.tensor(lat_slice, dtype=torch.float32).view(h_size, 1).repeat(1, w_size)
            lon_2d = torch.tensor(lon_slice, dtype=torch.float32).view(1, w_size).repeat(h_size, 1)
            t_1d = torch.tensor(time_np[i], dtype=torch.float32)

            pos[i, 0] = t_1d.view(t_size, 1, 1).repeat(1, h_size, w_size)
            pos[i, 1] = lat_2d.unsqueeze(0).repeat(t_size, 1, 1)
            pos[i, 2] = lon_2d.unsqueeze(0).repeat(t_size, 1, 1)

        return pos

    def _create_datasets(self, stage: str | None = None) -> None:
        """
        Create datasets for the explicit per-pixel coordinate case.

        Returned per sample:
            (values, times, positions)

        Note:
        - times are kept separately because they are convenient for debugging,
          plotting, or later bookkeeping
        - the model itself usually consumes (values, positions)
        """

        tiles = self._tile_array(self._raw_data)
        values = torch.tensor(tiles.values, dtype=torch.float32)
        times = self._extract_times(tiles)
        positions = self._extract_pixel_positions(tiles)

        base = TensorDataset(values, times, positions)
        train_base, val_base, test_base = self._split_dataset(base)

        self.train_dataset = TransformedInputTimeCoordsDataset(train_base, self.train_transforms)
        self.val_dataset = TransformedInputTimeCoordsDataset(val_base, self.val_transforms)
        self.test_dataset = (
            None
            if test_base is None
            else TransformedInputTimeCoordsDataset(test_base, self.test_transforms)
        )


if __name__ == "__main__":
    from FRAME_FM.models.mmmae import MultimodalMaskedAutoencoder

    print("Starting ERA5 dataloader demo...")

    # Demo using the pixel-coordinate version because it shows the full time/lat/lon grid.
    data_module = ERA5SpatialPixelsDataModule(
        data_root=(
            "https://gws-access.jasmin.ac.uk/public/eds_ai/era5_repack/aggregations/data/"
            "ecmwf-era5X_oper_an_sfc_2000_2020_2d_repack.kr1.0.json"
        ),
        variables=["d2m"],
        time_min="2005-01-01T00",
        time_max="2005-01-01T03",
        time_slice_size=2,
        tile_size_lat=64,
        tile_size_lon=64,
        batch_size=1,
        chunks={"time": 1},
        num_workers=0,
        debug=True,
    )

    data_module.setup()

    train_values, train_times, train_pos = next(iter(data_module.train_dataloader()))
    val_values, val_times, val_pos = next(iter(data_module.val_dataloader()))

    print("train:", train_values.shape, train_times.shape, train_pos.shape)
    print("val  :", val_values.shape, val_times.shape, val_pos.shape)

    _, c_dim, t_dim, h_dim, w_dim = val_values.shape
    t_min = float(val_pos[:, 0].min().item())
    t_max = float(val_pos[:, 0].max().item())

    model = MultimodalMaskedAutoencoder(
        input_shapes=[(t_dim, h_dim, w_dim)],
        n_channels=[c_dim],
        patch_shapes=[(1, 16, 16)],
        inputs_positioned="pixels",
        position_space=((t_min, t_max), (-90.0, 90.0), (-180.0, 180.0)),
        pos_embed_ratio=(1.0, 1.0, 1.0),
        encoder_embed_dim=256,
        encoder_depth=4,
        encoder_num_heads=8,
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=4,
    )

    model.eval()
    with torch.no_grad():
        latent, pos_embed, mask, ids_restore = model.forward_encoder(
            inputs=[(val_values, val_pos)],
            mask_ratio=0.5,
        )

    print("latent:", latent.shape)
    print("pos_embed:", pos_embed.shape)
    print("mask:", mask.shape)
    print("ids_restore:", ids_restore.shape)
