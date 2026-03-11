# src/FRAME_FM/dataloaders/era5_kerchunk_dataloader.py
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
#
# This file provides 3 flavours of the same ERA5 loader:
# 1) ERA5TiledBaseDataModule
#    Returns values only.
#    Each sample is shaped (C, T, H, W).
#
# 2) ERA5SpatialPixelsDataModule
#    Returns values + explicit per-pixel coordinates.
#    Each sample is:
#       values    -> (C, T, H, W)
#       times     -> (T,)
#       positions -> (3, T, H, W) for (time, lat, lon)
#
# 3) ERA5SpatialBoundsDataModule
#    Returns values + tile bounds.
#    Each sample is:
#       values -> (C, T, H, W)
#       times  -> (T,)
#       bounds -> (3, 2) for ([t_min, t_max], [lat_min, lat_max], [lon_min, lon_max])
#
# The comments intentionally stay simple and intuitive so that users can
# quickly re-read the file and remember why each step exists.

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
import xarray as xr

from FRAME_FM.utils.LightningDataModuleWrapper import BaseDataModule
from FRAME_FM.datasets.InputOnly_Dataset import (
    TransformedInputCoordsDataset,
    TransformedInputDataset,
)
from FRAME_FM.datasets.InputTimeCoords_Dataset import TransformedInputTimeCoordsDataset


class _SingleTensorDataset(Dataset):
    """
    Tiny helper dataset for the "values only" case.

    Why this exists:
    TensorDataset(x) returns samples like (x_i,), i.e. a 1-element tuple.
    For the plain base dataloader we usually want each sample to be just the
    tensor itself, not a tuple wrapping it.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tensor[idx]


class ERA5TiledBaseDataModule(BaseDataModule):
    """
    Base ERA5 datamodule using a kerchunk reference JSON.

    Simple mental picture:
    - The raw ERA5 data is a huge movie over the whole globe.
    - We do not feed the whole globe to the model at once.
    - Instead, we cut each frame into smaller map tiles.
    - We can also group neighbouring timesteps together.

    Final sample shape from this base class:
        (C, T, H, W)
    where:
        C = number of requested ERA5 variables
        T = number of timesteps grouped together (time_slice_size)
        H = tile_size_lat
        W = tile_size_lon

    This base class returns values only.
    If you also want coordinates, use one of the subclasses below.
    """

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

        # These are filled once raw data is loaded.
        self._global_lat: np.ndarray | None = None
        self._global_lon: np.ndarray | None = None
        self._global_time: np.ndarray | None = None

    def _log(self, *args) -> None:
        """Print only when debug=True."""
        if self.debug:
            print(*args)

    def _load_raw_data(self) -> xr.DataArray:
        """
        Open ERA5 through kerchunk and return one xarray DataArray.

        Returned dimensions are ordered as:
            (channel, time, latitude, longitude)

        Why convert Dataset -> DataArray here?
        A Dataset stores each weather variable separately.
        A model usually wants one tensor with a channel axis.
        So:
            d2m(time, lat, lon)
            t2m(time, lat, lon)
        becomes:
            arr(channel, time, lat, lon)
        where channel 0 might be d2m and channel 1 might be t2m.
        """

        open_kwargs = {"engine": "kerchunk"}
        if self.chunks is not None:
            open_kwargs["chunks"] = self.chunks

        # ds = xr.open_dataset(self.data_root, **open_kwargs)
        ds = xr.open_dataset(self.data_root, engine="kerchunk")
        

        # Keep only the requested variables.
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

        # ERA5 longitude is often stored as 0..360.
        # For humans, -180..180 is usually easier to think about.
        if self.convert_longitude_to_180 and "longitude" in ds.coords:
            lon = ds["longitude"]
            lon_180 = ((lon + 180) % 360) - 180
            ds = ds.assign_coords(longitude=lon_180).sortby("longitude")

        # Collapse multiple weather variables into one channel axis.
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
        """
        Turn one large global ERA5 array into many smaller samples.

        Input shape:
            (channel, time, latitude, longitude)

        Output shape:
            (batch_dim, channel, time_inner, tile_lat, tile_lon)

        Intuition:
        - First, group time into blocks of length time_slice_size.
        - Then, cut latitude and longitude into tiles.
        - Finally, stack (time block, tile row, tile column) into one sample index.

        So one dataset sample means:
            "this tile, at this place on Earth, over this short time window"
        """

        if "latitude" not in arr.dims or "longitude" not in arr.dims:
            raise ValueError(f"Expected latitude/longitude dims, got {arr.dims}.")

        n_lat = arr.sizes["latitude"]
        n_lon = arr.sizes["longitude"]

        if n_lat < self.tile_size_lat or n_lon < self.tile_size_lon:
            raise ValueError(
                "ERA5 grid is smaller than the requested tile size: "
                f"{n_lat}x{n_lon} < {self.tile_size_lat}x{self.tile_size_lon}"
            )

        # coarsen(..., boundary="pad") means edge tiles are padded if the global
        # grid size is not perfectly divisible by tile size.
        tiles = arr.coarsen(
            time=self.time_slice_size,
            latitude=self.tile_size_lat,
            longitude=self.tile_size_lon,
            boundary="pad",
        )

        # construct(...) splits each coarse axis into:
        # - an outer index saying "which block is this?"
        # - an inner index saying "where am I inside that block?"
        #
        # Example for time:
        #   original time axis -> time_coarse x time_inner
        # where time_inner has length time_slice_size.
        tiles = tiles.construct(
            time=("time_coarse", "time_inner"),
            latitude=("tile_lat_id", "tile_lat"),
            longitude=("tile_lon_id", "tile_lon"),
        )

        # A single sample is identified by:
        #   (time_coarse, tile_lat_id, tile_lon_id)
        tiles = tiles.stack(batch_dim=("time_coarse", "tile_lat_id", "tile_lon_id"))

        # Final per-sample layout expected by the model side.
        tiles = tiles.transpose("batch_dim", "channel", "time_inner", "tile_lat", "tile_lon")

        # Padding introduced by xarray becomes NaN; fill with zeros for tensors.
        tiles = tiles.fillna(0)

        self._log(
            "Each sample shape (C, T, H, W) =",
            (
                tiles.sizes["channel"],
                tiles.sizes["time_inner"],
                tiles.sizes["tile_lat"],
                tiles.sizes["tile_lon"],
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

        pad_value = axis_slice[-1]
        pad = np.full(target_len - axis_slice.size, pad_value, dtype=axis_slice.dtype)
        return np.concatenate([axis_slice, pad], axis=0)

    def _create_datasets(self, stage: str | None = None) -> None:
        """
        Create train/val/test datasets for the plain values-only case.

        Important shape reminder:
            each item is (C, T, H, W)

        We keep the base class "simple" on purpose:
        - it returns values only
        - it still fully supports time_slice_size internally
        - if you need explicit times or coordinates, use a subclass below
        """

        tiles = self._tile_array(self._raw_data)
        tile_tensor = torch.tensor(tiles.values, dtype=torch.float32)

        # Use the tiny helper dataset so each sample is a tensor, not (tensor,).
        base = _SingleTensorDataset(tile_tensor)
        train_base, val_base, test_base = self._split_dataset(base)

        self.train_dataset = TransformedInputDataset(train_base, self.train_transforms)
        self.val_dataset = TransformedInputDataset(val_base, self.val_transforms)
        self.test_dataset = (
            None
            if test_base is None
            else TransformedInputDataset(test_base, self.test_transforms)
        )


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

        N = tiles.sizes["batch_dim"]
        T = tiles.sizes["time_inner"]
        H = tiles.sizes["tile_lat"]
        W = tiles.sizes["tile_lon"]

        # Use the tile-aligned time coordinate created by xarray, not the batch index.
        time_np = np.asarray(tiles["time"].values).astype("datetime64[s]").astype("int64")
        if time_np.ndim == 1:
            time_np = time_np[:, None]

        pos = torch.empty((N, 3, T, H, W), dtype=torch.float32)

        for i, (_, tile_lat_id, tile_lon_id) in enumerate(batch_tuples):
            lat_start = int(tile_lat_id) * self.tile_size_lat
            lon_start = int(tile_lon_id) * self.tile_size_lon

            lat_slice = self._pad_axis_slice(self._global_lat[lat_start: lat_start + H], H)
            lon_slice = self._pad_axis_slice(self._global_lon[lon_start: lon_start + W], W)

            # Build one latitude-longitude grid for the tile.
            lat_2d = torch.tensor(lat_slice, dtype=torch.float32).view(H, 1).repeat(1, W)
            lon_2d = torch.tensor(lon_slice, dtype=torch.float32).view(1, W).repeat(H, 1)

            # Build the time axis for this sample.
            t_1d = torch.tensor(time_np[i], dtype=torch.float32)

            # Expand all three axes to a common shape (T, H, W).
            time_3d = t_1d.view(T, 1, 1).repeat(1, H, W)
            lat_3d = lat_2d.unsqueeze(0).repeat(T, 1, 1)
            lon_3d = lon_2d.unsqueeze(0).repeat(T, 1, 1)

            pos[i, 0] = time_3d
            pos[i, 1] = lat_3d
            pos[i, 2] = lon_3d

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

        self.train_dataset = TransformedInputTimeCoordsDataset(
            train_base, self.train_transforms
        )
        self.val_dataset = TransformedInputTimeCoordsDataset(
            val_base, self.val_transforms
        )
        self.test_dataset = (
            None
            if test_base is None
            else TransformedInputTimeCoordsDataset(test_base, self.test_transforms)
        )


class ERA5SpatialBoundsDataModule(ERA5TiledBaseDataModule):
    """
    ERA5 loader that returns tile bounds instead of per-pixel coordinate grids.

    Each sample is:
        values -> (C, T, H, W)
        times  -> (T,)
        bounds -> (3, 2)

    Bounds are ordered as:
        bounds[0] = [t_min,  t_max]
        bounds[1] = [lat_min, lat_max]
        bounds[2] = [lon_min, lon_max]

    This is the natural time-aware extension of the old spatial-only bounds case.
    If your position_space is (time, lat, lon), then bounds must also have
    3 coordinate dimensions.
    """

    def _extract_bounds(self, tiles: xr.DataArray, times: torch.Tensor) -> torch.Tensor:
        """
        Build real bounds for every tile using the stored global ERA5 axes.

        Output shape:
            (N, 3, 2)

        Why include time bounds too?
        Because once each sample is a short time window, it is no longer enough to
        say only where the tile is. The sample also occupies a time interval.
        """

        if self._global_lat is None or self._global_lon is None:
            raise RuntimeError("Global coordinates were not stored before bounds extraction.")

        batch_tuples = tiles["batch_dim"].values

        N = tiles.sizes["batch_dim"]
        H = tiles.sizes["tile_lat"]
        W = tiles.sizes["tile_lon"]

        # Start with time bounds of shape (N, 1, 2).
        time_bounds = self._extract_time_bounds(times)

        # Then add spatial bounds of shape (N, 2, 2).
        spatial_bounds = torch.empty((N, 2, 2), dtype=torch.float32)

        for i, (_, tile_lat_id, tile_lon_id) in enumerate(batch_tuples):
            lat_start = int(tile_lat_id) * self.tile_size_lat
            lon_start = int(tile_lon_id) * self.tile_size_lon

            lat_slice = self._pad_axis_slice(self._global_lat[lat_start: lat_start + H], H)
            lon_slice = self._pad_axis_slice(self._global_lon[lon_start: lon_start + W], W)

            spatial_bounds[i, 0, 0] = float(np.min(lat_slice))
            spatial_bounds[i, 0, 1] = float(np.max(lat_slice))
            spatial_bounds[i, 1, 0] = float(np.min(lon_slice))
            spatial_bounds[i, 1, 1] = float(np.max(lon_slice))

        # Concatenate so the final coordinate order is:
        #   time, latitude, longitude
        bounds = torch.cat([time_bounds, spatial_bounds], dim=1)
        return bounds

    def _create_datasets(self, stage: str | None = None) -> None:
        """
        Create datasets for the bounds-based coordinate case.

        Returned per sample:
            (values, times, bounds)

        In model code, you will normally pass:
            (values, bounds)
        to the bounds embedder, while keeping times separately for inspection.
        """

        tiles = self._tile_array(self._raw_data)
        values = torch.tensor(tiles.values, dtype=torch.float32)
        times = self._extract_times(tiles)
        bounds = self._extract_bounds(tiles, times)

        base = TensorDataset(values, times, bounds)
        train_base, val_base, test_base = self._split_dataset(base)

        self.train_dataset = TransformedInputTimeCoordsDataset(
            train_base, self.train_transforms
        )
        self.val_dataset = TransformedInputTimeCoordsDataset(
            val_base, self.val_transforms
        )
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

    print("Setting up DataModule...")
    data_module.setup()

    print("\nDataset sizes:")
    print("Train:", len(data_module.train_dataset))
    print("Val  :", len(data_module.val_dataset))
    print("Test :", 0 if data_module.test_dataset is None else len(data_module.test_dataset))

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    train_values, train_times, train_pos = train_batch
    val_values, val_times, val_pos = val_batch

    print("\nTrain batch contents:")
    print("Train batch length:", len(train_batch), "Expected: 3 (values, times, positions)")
    print("values:", train_values.shape, "Expected: (B, C, T, H, W)")
    print("times :", train_times.shape, "Expected: (B, T)")
    print("pos   :", train_pos.shape, "Expected: (B, 3, T, H, W)")

    print("\nSingle sample shapes:")
    print("values sample:", train_values[0].shape)
    print("times sample :", train_times[0].shape)
    print("pos sample   :", train_pos[0].shape)

    print("\nVal batch shapes:")
    print("values:", val_values.shape)
    print("times :", val_times.shape)
    print("pos   :", val_pos.shape)

    # Here the actual weather values live in val_values.
    # val_times tells us which timestamps this short sequence covers.
    # val_pos gives a full (time, lat, lon) grid so the model knows where each
    # value sits in space-time.

    print("\nTesting forward pass through MMMAE encoder...")

    B, C, T, H, W = val_values.shape
    print(f"Input batch shape: {val_values.shape} (B, C, T, H, W)")
    print("val_pos.shape:", val_pos.shape)

    # Choose the time range directly from the batch. This is safer than hardcoding.
    t_min = float(val_pos[:, 0].min().item())
    t_max = float(val_pos[:, 0].max().item())

    model = MultimodalMaskedAutoencoder(
        input_shapes=[(T, H, W)],
        n_channels=[C],
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

    embed = model.input_embedders[0]
    print("embed.input_shape:", embed.input_shape)
    print("embed.patch_shape:", embed.patch_shape)
    print("len(position_space):", len(embed.position_space))
    print("pos_conv_kernel shape:", embed.pos_conv_kernel.shape)

    # Inspect the raw position-convolution output.
    conv_fn = torch.nn.functional.conv3d
    pconv = conv_fn(
        val_pos,
        embed.pos_conv_kernel,
        stride=embed.patch_shape,
        groups=len(embed.position_space),
    )
    print("raw position conv shape:", pconv.shape)

    with torch.no_grad():
        x_tokens = embed.proj(val_values).flatten(start_dim=2).transpose(1, 2)
        p_tokens = embed.pos_embed(val_pos)
        print("value token shape:", x_tokens.shape)
        print("position token shape:", p_tokens.shape)

    with torch.no_grad():
        latent, pos_embed, mask, ids_restore = model.forward_encoder(
            inputs=[(val_values, val_pos)],
            mask_ratio=0.5,
        )

    print("\n Encoder forward pass successful.")
    print("Latent shape     :", latent.shape)
    print("pos_embed shape  :", pos_embed.shape)
    print("Mask shape       :", mask.shape)
    print("ids_restore shape:", ids_restore.shape)
