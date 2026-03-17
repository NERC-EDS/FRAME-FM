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
from FRAME_FM.datasets.InputTimeCoords_Dataset import MMMAEInputFromTimeCoordsDataset
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
        tile_boundary: str = "trim",
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
        # Return model-ready batch samples for MMMAE training.
        model_ready_inputs: bool = False,
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
        self.tile_boundary = tile_boundary
        self.time_slice_size = time_slice_size
        self.convert_longitude_to_180 = convert_longitude_to_180
        self.chunks = chunks
        self.model_ready_inputs = model_ready_inputs
        self.debug = debug

        if self.tile_boundary not in {"pad", "trim"}:
            raise ValueError(
                f"Unsupported tile_boundary='{self.tile_boundary}'. Expected one of: ['pad', 'trim']"
            )

        self._global_lat: np.ndarray | None = None
        self._global_lon: np.ndarray | None = None
        self._tile_index_mapper: TiledIndexMapper | None = None

    def _log(self, *args) -> None:
        if self.debug:
            print(*args)

    def _load_raw_data(self) -> xr.DataArray:
        open_kwargs = {"engine": "kerchunk"}
        if self.chunks is not None:
            # Hydra may pass a DictConfig; xarray expects a plain dict/int/auto/None.
            open_kwargs["chunks"] = dict(self.chunks)

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
            boundary=self.tile_boundary,
            time=self.time_slice_size,
            latitude=self.tile_size_lat,
            longitude=self.tile_size_lon,
        )(arr)

        # Final per-sample layout expected by model code.
        tiles = tiles.transpose("batch_dim", "channel", "time_fine", "latitude_fine", "longitude_fine")

        # Only pad mode can introduce NaNs at edge tiles.
        if self.tile_boundary == "pad":
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

        pos = tiled_to_pixel_coordinates(tiles, coord_dims=["time", "latitude", "longitude"])
        times = pos[:, 0, :, 0, 0].to(torch.int64)
        if times.ndim == 1:
            times = times[:, None]
        return times

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


class ERA5SpatialPixelsDataModule(ERA5BaseDataModule):
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

        return tiled_to_pixel_coordinates(tiles, coord_dims=["time", "latitude", "longitude"])

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
        self._tile_index_mapper = TiledIndexMapper.from_tiled_array(tiles)
        values = torch.tensor(tiles.values, dtype=torch.float32)
        times = self._extract_times(tiles)
        positions = self._extract_pixel_positions(tiles)

        base = TensorDataset(values, times, positions)
        train_base, val_base, test_base = self._split_dataset(base)

        wrapper_cls = (
            MMMAEInputFromTimeCoordsDataset if self.model_ready_inputs else TransformedInputTimeCoordsDataset
        )

        self.train_dataset = wrapper_cls(train_base, self.train_transforms)
        self.val_dataset = wrapper_cls(val_base, self.val_transforms)
        self.test_dataset = None if test_base is None else wrapper_cls(test_base, self.test_transforms)

    def get_tile_index_mapper(self) -> TiledIndexMapper:
        if self._tile_index_mapper is None:
            raise RuntimeError("Tile index mapper is not available before setup()/_create_datasets().")
        return self._tile_index_mapper


class ERA5SpatialBoundsDataModule(ERA5BaseDataModule):
    """
    ERA5 loader that returns per-tile coordinate bounds.

    Each sample is:
        values    -> (C, T, H, W)
        times     -> (T,)
        positions -> (3, 2) for bounds in (time, latitude, longitude)

    This matches MMMAE / inputs_positioned="bounds" for spatiotemporal inputs.
    """

    def _extract_tile_bounds(self, tiles: xr.DataArray) -> torch.Tensor:
        return tiled_to_coordinate_bounds(tiles, coord_dims=["time", "latitude", "longitude"])

    def _create_datasets(self, stage: str | None = None) -> None:
        tiles = self._tile_array(self._raw_data)
        self._tile_index_mapper = TiledIndexMapper.from_tiled_array(tiles)
        values = torch.tensor(tiles.values, dtype=torch.float32)
        times = self._extract_times(tiles)
        bounds = self._extract_tile_bounds(tiles)

        base = TensorDataset(values, times, bounds)
        train_base, val_base, test_base = self._split_dataset(base)

        wrapper_cls = (
            MMMAEInputFromTimeCoordsDataset if self.model_ready_inputs else TransformedInputTimeCoordsDataset
        )

        self.train_dataset = wrapper_cls(train_base, self.train_transforms)
        self.val_dataset = wrapper_cls(val_base, self.val_transforms)
        self.test_dataset = None if test_base is None else wrapper_cls(test_base, self.test_transforms)

    def get_tile_index_mapper(self) -> TiledIndexMapper:
        if self._tile_index_mapper is None:
            raise RuntimeError("Tile index mapper is not available before setup()/_create_datasets().")
        return self._tile_index_mapper


if __name__ == "__main__":
    raise RuntimeError(
        "ERA5_dataloader.py no longer hosts the demo entrypoint. "
        "Run: python src/FRAME_FM/training/era5_mmmae_demo.py "
        "data=era5_spatial_pixels_demo model=era5_demo_mmmae"
    )