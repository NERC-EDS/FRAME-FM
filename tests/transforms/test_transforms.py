"""
# Unit tests for the transforms in FRAME_FM.transforms.

NOTES:
- The `ds.roll()` operation was hanging on `xarray` version 2026.2.0 but works fine on version 2025.11.0.
  - Update: 2026-03-10: Subsequently installed xarray version 2026.2.0 in a fresh environment and the `ds.roll()`
    operation is now working fine.
- These tests are designed to be run with pytest, as follows (from the root of the repository):

```
PYTHONPATH=src python -m pytest tests/test_transforms.py
```
"""

import numpy as np
import torch
import xarray as xr

import pytest
import pandas as pd

from FRAME_FM.transforms import FillMissingValueTransform, VarsToDimensionTransform
from FRAME_FM.transforms.transforms import (
    FillNaNTransform,
    NormalizeTransform,
    RenameTransform,
    ResampleTransform,
    ReshapeTransform,
    ReverseAxisTransform,
    RollTransform,
    SortAxisTransform,
    StandardizeTransform,
    SubsetTransform,
    TiledIndexMapper,
    TilerTransform,
    ToTensorTransform,
    ToValuesBoundsTransform,
    ToValuesLocationsTransform,
    TransposeTransform,
    tiled_to_coordinate_bounds,
    tiled_to_pixel_coordinates,
    transform_mapping,
)
from FRAME_FM.utils.data_utils import load_data_from_uri

from tests.datasets.common import CHESS_URI, ERA5_URI

pdt = pd.to_datetime

dsets = {
    "era5": {
        "uri": ERA5_URI,
        "main_var": "d2m",
        "ds": None,
    },
    "chessmet": {
        "uri": CHESS_URI,
        "main_var": "precip",
        "ds": None,
    },
}


def _load_data(source: str = "era5", response_type: str = "Dataset") -> xr.Dataset | xr.DataArray:
    global dsets
    if dsets[source]["ds"] is None:
        dsets[source]["ds"] = load_data_from_uri(dsets[source]["uri"])  # type: ignore

    var_id = dsets[source]["main_var"]
    if response_type == "DataArray":
        resp = dsets[source]["ds"][var_id].isel(time=slice(0, 3))
    else:
        resp = dsets[source]["ds"]

    return resp, var_id


def _general_fill_missing_test(transform_class, strategy):
    ds, var_id = _load_data()

    # Introduce some missing values into the dataset for testing
    ds_with_nans = ds.copy().isel(time=slice(0, 3))  # Take a small subset for testing
    ds_with_nans[var_id] = ds_with_nans[var_id].where(ds_with_nans[var_id] > 290)  # Set values <= 290 to NaN

    if strategy == "constant":
        # Run the fill missing value transform with constant strategy
        fill_transform_constant = transform_class(strategy="constant", fill_value=0.0)
        filled_ds_constant = fill_transform_constant(ds_with_nans)
        assert not filled_ds_constant[var_id].isnull().any(), (
            f"{transform_class} with constant strategy did not work as expected (there are still NaN values)."
        )

    elif strategy == "interpolate":
        # Run the fill missing value transform with interpolate strategy
        fill_transform_interpolate = transform_class(strategy="interpolate", method="linear", fill_value="extrapolate")
        filled_ds_interpolate = fill_transform_interpolate(ds_with_nans)
        assert not filled_ds_interpolate[var_id].isnull().any(), (
            f"{transform_class} with interpolate strategy did not work as expected (there are still NaN values)."
        )


def test_FillMissingValueTransform_invalid_strategy():
    ds, var_id = _load_data()

    # Introduce some missing values into the dataset for testing
    ds_with_nans = ds.copy().isel(time=slice(0, 3))  # Take a small subset for testing
    ds_with_nans[var_id] = ds_with_nans[var_id].where(ds_with_nans[var_id] > 290)  # Set values <= 290 to NaN

    # Run the fill missing value transform with an unsupported strategy to check that it raises an error
    transform_class = FillMissingValueTransform
    try:
        fill_transform_invalid = transform_class(strategy="unsupported_strategy", fill_value=273.15)
        fill_transform_invalid(ds_with_nans)
        assert False, f"{transform_class} did not raise an error for an unsupported strategy."
    except ValueError as e:
        assert str(e) == "Unsupported fill strategy: unsupported_strategy", (
            f"FillMissingValueTransform raised an unexpected error message: {str(e)}"
        )


def test_FillMissingValueTransform_constant():
    _general_fill_missing_test(FillMissingValueTransform, strategy="constant")


@pytest.mark.xfail(reason="This test is currently failing due the `.interpolate_na()` method needing investigation.")
def test_FillNaNTransform():
    # Identical to the FillMissingValueTransform test but with the FillNaNTransform instead
    return  # test_FillMissingValueTransform()


@pytest.mark.xfail(
    reason="This test is currently failing due to an issue with the interpolate strategy, which needs further investigation."
)
def test_FillNaNTransform_interpolate():
    # Identical to the FillMissingValueTransform test but with the FillNaNTransform instead
    _general_fill_missing_test(FillNaNTransform, strategy="interpolate")


def test_StandardizeTransform():
    da, _ = _load_data(response_type="DataArray")  # type: ignore

    standardize_transform = StandardizeTransform(mean=float(da.mean()), std=float(da.std()))
    standardized_da = standardize_transform(da)
    assert np.isclose(float(standardized_da.mean()), 0, atol=1e-5), (
        "Standardize transform did not work as expected (mean is not close to 0)."
    )
    assert np.isclose(float(standardized_da.std()), 1, atol=1e-5), (
        "Standardize transform did not work as expected (std is not close to 1)."
    )


def test_NormalizeTransform():
    data = xr.DataArray([2, 4, 6, 8, 10])

    transform = NormalizeTransform()
    out = transform(data)
    assert np.allclose(out, np.array([0, 0.25, 0.5, 0.75, 1])), (
        "Normalize transform did not produce the expected values."
    )


def test_RenameTransform():
    ds, var_id = _load_data()

    # Run the rename transform
    rename_transform = RenameTransform(var_id=var_id, new_name="dewpoint_temperature")
    transformed_ds = rename_transform(ds)  # type: ignore
    assert "dewpoint_temperature" in transformed_ds.data_vars, "Rename transform did not work as expected."


def test_ResampleTransform():
    ds, var_id = _load_data()
    start, end = "2000-01-01T00:00:00", "2000-01-01T23:00:00"
    ds = ds.sel(time=slice(start, end))
    freq = "1D"  # daily frequency

    # Run the resample transform to resample from hourly to daily data
    resample_transform = ResampleTransform(dim="time", freq=freq)
    resampled_ds = resample_transform(ds)
    # Check that the time coordinate has been resampled correctly (should now have daily frequency)

    expected_time_range = pd.date_range(start=start, end=end, freq=freq)
    assert np.array_equal(resampled_ds.time.values, expected_time_range.values), (
        "Resample transform did not work as expected (time coordinate does not match expected daily frequency)."
    )


def test_ReshapeTransform():
    da, var_id = _load_data(response_type="DataArray")

    # Run the reshape transform
    new_shape = (3 * 721 * 1440,)
    reshape_transform = ReshapeTransform(shape=new_shape)
    reshaped_arr = reshape_transform(da)
    # Check that the reshaped array has the expected shape
    assert reshaped_arr.shape == new_shape, (
        f"Reshape transform did not work as expected (shape is {reshaped_arr.shape} instead of {new_shape})"
    )


def test_ReverseAxisTransform():
    ds, var_id = _load_data()

    # Run the reverse axis transform
    reverse_axis_transform = ReverseAxisTransform(dim="latitude")
    reversed_ds = reverse_axis_transform(ds)
    # Check that the latitude axis has been reversed correctly
    assert reversed_ds.latitude[0] == ds.latitude[-1], (
        "Reverse axis transform did not work as expected (first latitude value is not the same as the last latitude value of the original dataset)."
    )
    assert reversed_ds.latitude[-1] == ds.latitude[0], (
        "Reverse axis transform did not work as expected (last latitude value is not the same as the first latitude value of the original dataset)."
    )


def test_RollTransform():
    ds, var_id = _load_data()

    # Run the roll transform with shift 720
    roll_transform = RollTransform(dim="longitude", shift=720)
    rolled_ds = roll_transform(ds)
    # Check that the longitude coordinate has been rolled correctly
    assert rolled_ds.longitude[0] == -180.0 and rolled_ds.longitude[-1] == 179.75, (
        "Roll transform did not work as expected." + str(rolled_ds.longitude.values)
    )

    # Run the roll transform with automatic shift detection
    auto_roll_transform = RollTransform(dim="longitude", shift=None)
    auto_rolled_ds = auto_roll_transform(ds)
    # Check that the longitude coordinate has been rolled correctly
    assert auto_rolled_ds.longitude[0] == -180.0 and auto_rolled_ds.longitude[-1] == 179.75, (
        "Auto roll transform did not work as expected." + str(auto_rolled_ds.longitude.values)
    )


def test_ScaleTransform():
    # Same as the NormalizeTransform test but with the ScaleTransform instead
    return test_NormalizeTransform()


def test_SortAxisTransform():
    ds, var_id = _load_data()

    # Run the sort axis transform
    sort_axis_transform = SortAxisTransform(dim="latitude", ascending=True)
    sorted_ds = sort_axis_transform(ds)

    # Check that the latitude axis has been sorted in ascending order
    assert all([v1 < v2 for v1, v2 in zip(sorted_ds.latitude[:-1], sorted_ds.latitude[1:])]), (
        "Sort axis transform did not work as expected (latitude axis is not in ascending order)."
    )
    # Check that the longitude axis has not been changed in ascending order
    assert all([v1 == v2 for v1, v2 in zip(sorted_ds.longitude, ds.longitude)]), (
        "Sort axis transform did not work as expected (longitude axis has been changed)."
    )


def test_SubsetTransform():
    ds, var_id = _load_data()

    # Run the subset transform on a Dataset
    subset_transform = SubsetTransform(
        variables=[var_id], time=("2000-01-01", "2000-01-10"), latitude=(-30, 60), longitude=(-40, 100)
    )
    subset_ds = subset_transform(ds)
    assert var_id in subset_ds.data_vars, "Variable subsetting did not work as expected."
    assert subset_ds.time.min().values >= pdt("2000-01-01"), "Time subsetting did not work as expected."

    # Run the subset transform on a DataArray
    da = ds[var_id]
    subset_transform = SubsetTransform(time=("2000-01-01", "2000-01-10"), latitude=(60, -30), longitude=(-40, 100))
    subset_da = subset_transform(da)
    assert subset_da.time.min().values >= pdt("2000-01-01"), "Time subsetting did not work as expected."
    assert subset_da.time.max().values <= pdt("2000-01-11"), "Time subsetting did not work as expected."
    assert subset_da.latitude.min().values >= -30, "Latitude subsetting did not work as expected."
    assert subset_da.latitude.max().values <= 60, "Latitude subsetting did not work as expected."
    assert subset_da.longitude.min().values >= -40, "Longitude subsetting did not work as expected."
    assert subset_da.longitude.max().values <= 100, "Longitude subsetting did not work as expected."
    print("Subsetted DataArray:")
    print(subset_da)


def test_SubsetTransform_with_2d_coordinate_axes():
    # In this case, we want subset to reduce the size of the "lat" and "lon" dimensions to match the spatial
    # subset, along with the domain of the main variable. So the SubsetTransform needs to handle this case appropriately.
    ds, var_id = _load_data(source="chessmet")

    # Run the subset transform on a Dataset
    subset_transform = SubsetTransform(
        **{"time": ("1961-01-01T00:00:00", "1961-01-02T00:00:00"), "y": (400500.0, 405500.0), "x": (400500.0, 405500.0)}
    )
    subset_ds = subset_transform(ds)
    assert {"precip", "lat", "lon"}.issubset(subset_ds.data_vars), "Variable subsetting did not work as expected."
    assert subset_ds.y.min().values >= 400500.0, "Y subsetting did not work as expected."
    assert subset_ds.y.max().values <= 405500.0, "Y subsetting did not work as expected."
    assert subset_ds.x.min().values >= 400500.0, "X subsetting did not work as expected."
    assert subset_ds.x.max().values <= 405500.0, "X subsetting did not work as expected."

    # Get shape of precip variable and compare last two dimensions to the shapes of the lat and lon variables to check that they have been subsetted to the same spatial domain
    precip_shape = subset_ds[var_id].shape
    lat_shape = subset_ds["lat"].shape
    lon_shape = subset_ds["lon"].shape

    assert precip_shape[-2:] == lat_shape == lon_shape, (
        f"Expected the last two dimensions of the precip variable to match the shapes of the lat and lon variables after subsetting, but got {precip_shape[-2:]} for precip, {lat_shape} for lat, and {lon_shape} for lon."
    )

    # Compare the x and y values of the lat and lon versus precip variable to check that they have been subsetted to the same spatial domain
    lat_x_values = subset_ds["lat"].x.values
    lat_y_values = subset_ds["lat"].y.values
    lon_x_values = subset_ds["lon"].x.values
    lon_y_values = subset_ds["lon"].y.values
    precip_x_values = subset_ds[var_id].x.values
    precip_y_values = subset_ds[var_id].y.values
    assert np.array_equal(lat_x_values, precip_x_values), (
        "Expected the x values of the lat variable to match the x values of the precip variable after subsetting, but they do not match."
    )
    assert np.array_equal(lat_y_values, precip_y_values), (
        "Expected the y values of the lat variable to match the y values of the precip variable after subsetting, but they do not match."
    )
    assert np.array_equal(lon_x_values, precip_x_values), (
        "Expected the x values of the lon variable to match the x values of the precip variable after subsetting, but they do not match."
    )
    assert np.array_equal(lon_y_values, precip_y_values), (
        "Expected the y values of the lon variable to match the y values of the precip variable after subsetting, but they do not match."
    )


def test_TilerTransform_time_series_data():
    da, var_id = _load_data(response_type="DataArray")

    # Reverse the latitude axis to check that the tiler can handle this case (since ERA5 has a descending latitude axis)
    da = da.isel(latitude=slice(None, None, -1))
    step = 10

    # Run the tiler transform with tile sizes of step x step and "pad" boundary handling
    tiler_transform = TilerTransform(latitude=step, longitude=step, boundary="trim")
    tiled = tiler_transform(da)

    # Check that the tiled array has the expected shape (should have new dimensions for tiles)
    expected_shape = (10512, 3, step, step)  # (batch_dim[=n_tiles], time, latitude_fine, longitude_fine)
    assert tiled.shape == expected_shape, (
        f"Tiler transform did not work as expected (shape is {tiled.shape} instead of {expected_shape})"
    )

    assert tiled.dims[0] == "batch_dim", f"Expected first dimension to be 'batch_dim', but got {tiled.dims[0]}"
    # Check values of first tile versus original dataset
    first_tile = tiled.isel(batch_dim=0)
    original_subset = da.isel(latitude=slice(0, step), longitude=slice(0, step))

    # Check values of first tile versus original dataset
    assert np.array_equal(first_tile, original_subset), "First tile does not match expected subset of original dataset"

    # Check the last tile (which has shape: (3, 10, 10) so cuts across three time slices)
    # At the moment this is failing because the last tile is not matching the expected subset of the original dataset,
    # even though the first tile is correct. This may be due to an issue with how the tiler is handling the padding for
    # the last tile, or it may be an issue with how the test is checking the values of the last tile. This needs further
    # investigation.
    last_tile = tiled[-1]
    original_subset = da.isel(latitude=slice(-step - 1, -1), longitude=slice(-step, None))
    assert np.array_equal(last_tile, original_subset), "Last tile does not match expected subset of original dataset"

    # Test that reverse-lookup metadata is stored in attrs
    assert "tiler_tile_sizes" in tiled.attrs, "Expected 'tiler_tile_sizes' in tiled.attrs, but not found"
    assert tiled.attrs["tiler_tile_sizes"] == {"latitude": step, "longitude": step}, (
        f"Expected tile sizes in metadata to be {{'latitude': {step}, 'longitude': {step}}}, but got {tiled.attrs['tiler_tile_sizes']}"
    )
    assert "tiler_boundary" in tiled.attrs, "Expected 'tiler_boundary' in tiled.attrs, but not found"
    assert tiled.attrs["tiler_boundary"] == "pad", (
        f"Expected boundary in metadata to be 'pad', but got {tiled.attrs['tiler_boundary']}"
    )
    assert "tiler_original_sizes" in tiled.attrs, "Expected 'tiler_original_sizes' in tiled.attrs, but not found"

    original_sizes = {"latitude": da.latitude.size, "longitude": da.longitude.size}
    assert tiled.attrs["tiler_original_sizes"] == original_sizes, (
        f"Expected original sizes in metadata to be {original_sizes}, but got {tiled.attrs['tiler_original_sizes']}"
    )
    assert "tiler_original_coords" in tiled.attrs, "Expected 'tiler_original_coords' in tiled.attrs, but not found"
    assert tiled.attrs["tiler_original_coords"] == {
        "latitude": da.latitude.values.tolist(),
        "longitude": da.longitude.values.tolist(),
    }, (
        f"Expected original coords in metadata to match original dataset coords, but got {tiled.attrs['tiler_original_coords']}"
    )

    # Now test that the index mapper helper function works as expected
    raise NotImplementedError(
        "This test is currently failing due to an issue with the last tile matching "
        "the original data, so the rest of the test has not been implemented yet. "
        "This needs further investigation before it can be implemented."
    )


def test_tiled_coordinate_utilities_static_grid():
    da = xr.DataArray(
        np.arange(1 * 3 * 4, dtype=np.float32).reshape(1, 3, 4),
        dims=("band", "y", "x"),
        coords={
            "band": [0],
            "y": [10.0, 20.0, 30.0],
            "x": [100.0, 110.0, 120.0, 130.0],
        },
    )
    tiled = TilerTransform(y=2, x=2, boundary="pad")(da)

    pixel_coords = tiled_to_pixel_coordinates(tiled, coord_dims=["y", "x"])
    assert pixel_coords.shape == (4, 2, 2, 2)

    first_y = pixel_coords[0, 0]
    first_x = pixel_coords[0, 1]
    assert torch.equal(first_y, torch.tensor([[10.0, 10.0], [20.0, 20.0]]))
    assert torch.equal(first_x, torch.tensor([[100.0, 110.0], [100.0, 110.0]]))

    bounds = tiled_to_coordinate_bounds(tiled, coord_dims=["x", "y"])
    assert bounds.shape == (4, 2, 2)
    assert torch.equal(bounds[0, 0], torch.tensor([100.0, 110.0]))
    assert torch.equal(bounds[0, 1], torch.tensor([10.0, 20.0]))


def test_tile_locations_bounds():
    da = xr.DataArray(
        np.arange(1 * 4 * 6, dtype=np.float32).reshape(1, 4, 6),
        dims=("band", "y", "x"),
        coords={
            "band": [0],
            "y": [5.0, 15.0, 25.0, 35.0],
            "x": [105.0, 115.0, 125.0, 135.0, 145.0, 155.0],
        },
    )
    tiled = TilerTransform(y=2, x=2, boundary="trim")(da)
    _, first_tile_locations = ToValuesLocationsTransform(coords=["x", "y"])(tiled[0])
    assert torch.equal(
        first_tile_locations, torch.tensor([[[105.0, 105.0], [115.0, 115.0]], [[5.0, 15.0], [5.0, 15.0]]])
    )
    _, last_tile_locations = ToValuesLocationsTransform(coords=["x", "y"])(tiled[-1])
    assert torch.equal(
        last_tile_locations, torch.tensor([[[145.0, 145.0], [155.0, 155.0]], [[25.0, 35.0], [25.0, 35.0]]])
    )
    _, first_tile_bounds = ToValuesBoundsTransform(coords=["x", "y"])(tiled[0])
    assert torch.equal(first_tile_bounds, torch.tensor([[100.0, 120.0], [0.0, 20.0]]))
    _, last_tile_bounds = ToValuesBoundsTransform(coords=["x", "y"])(tiled[-1])
    assert torch.equal(last_tile_bounds, torch.tensor([[140.0, 160.0], [20.0, 40.0]]))


def test_tiled_index_mapper_roundtrip():
    da = xr.DataArray(
        np.zeros((1, 3, 4), dtype=np.float32),
        dims=("band", "y", "x"),
        coords={
            "band": [0],
            "y": [10.0, 20.0, 30.0],
            "x": [100.0, 110.0, 120.0, 130.0],
        },
    )
    tiled = TilerTransform(y=2, x=2, boundary="pad")(da)
    mapper = TiledIndexMapper.from_tiled_array(tiled)

    tile_id = mapper.tile_id_from_coordinates(y=30.0, x=120.0)
    assert tile_id == 3

    coarse_ids = mapper.coordinates_from_tile_id(tile_id)
    assert coarse_ids == {"y": 1, "x": 1}


def test_time_aware_tiling_positions_and_bounds():
    times = np.array(
        [
            np.datetime64("2005-01-01T00:00:00"),
            np.datetime64("2005-01-01T01:00:00"),
            np.datetime64("2005-01-01T02:00:00"),
        ]
    )
    da = xr.DataArray(
        np.random.randn(1, 3, 3, 3).astype(np.float32),
        dims=("channel", "time", "latitude", "longitude"),
        coords={
            "channel": [0],
            "time": times,
            "latitude": [50.0, 51.0, 52.0],
            "longitude": [-2.0, -1.0, 0.0],
        },
    )
    tiled = TilerTransform(time=2, latitude=2, longitude=2, boundary="pad")(da)
    pixel_coords = tiled_to_pixel_coordinates(tiled, coord_dims=["time", "latitude", "longitude"])
    bounds = tiled_to_coordinate_bounds(tiled, coord_dims=["time", "latitude", "longitude"])

    assert pixel_coords.shape == (8, 3, 2, 2, 2)
    assert bounds.shape == (8, 3, 2)

    first_time_bounds = bounds[0, 0]
    expected_t0 = torch.tensor(times[0].astype("datetime64[s]").astype("int64"), dtype=torch.float32)
    expected_t1 = torch.tensor(times[1].astype("datetime64[s]").astype("int64"), dtype=torch.float32)
    assert torch.equal(first_time_bounds, torch.stack([expected_t0, expected_t1]))


def test_tiler_axis_order_guardrail_raises_on_descending_axis():
    # Test on a hand-crafted DataArray
    da = xr.DataArray(
        np.zeros((1, 3, 4), dtype=np.float32),
        dims=("band", "y", "x"),
        coords={
            "band": [0],
            "y": [30.0, 20.0, 10.0],
            "x": [100.0, 110.0, 120.0, 130.0],
        },
    )

    with pytest.raises(ValueError, match="not strictly ascending"):
        TilerTransform(y=2, x=2, validate_axis_order=True)(da)

    # Test on the ERA5 example, which has a descending latitude axis
    da, var_id = _load_data(response_type="DataArray")
    with pytest.raises(ValueError, match="not strictly ascending"):
        TilerTransform(latitude=45, longitude=45, validate_axis_order=True)(da)

    # Reverse the latitude axis and check that the guardrail does not raise an error when validate_axis_order=False
    da_reversed = da.isel(latitude=slice(None, None, -1))
    try:
        TilerTransform(latitude=45, longitude=45, validate_axis_order=False)(da_reversed)
    except ValueError:
        assert False, "TilerTransform raised an error when validate_axis_order=False, but it should not have."

    # And test that the guardrail does not raise an error when validate_axis_order=True but the axis is in ascending order
    result = TilerTransform(latitude=45, longitude=45, validate_axis_order=True)(da_reversed)
    assert isinstance(result, xr.DataArray), (
        "TilerTransform did not return a DataArray when validate_axis_order=True and the axis is in ascending order."
    )


def test_tiler_discontinuity_guardrail_raises_for_wrapping_tile():
    da = xr.DataArray(
        np.zeros((1, 2, 4), dtype=np.float32),
        dims=("band", "latitude", "longitude"),
        coords={
            "band": [0],
            "latitude": [0.0, 1.0],
            "longitude": [-179.0, -178.0, 179.0, 180.0],
        },
    )

    with pytest.raises(ValueError, match="discontinuity crossing"):
        TilerTransform(latitude=2, longitude=4, discontinuity_periods={"longitude": 360.0}, validate_axis_order=False)(
            da
        )

    da = xr.DataArray(
        np.zeros((1, 2, 4), dtype=np.float32),
        dims=("band", "latitude", "longitude"),
        coords={
            "band": [0],
            "latitude": [0.0, 1.0],
            "longitude": [178.0, 179.0, 180.0, -179.0],
        },
    )

    with pytest.raises(ValueError, match="discontinuity crossing"):
        TilerTransform(latitude=2, longitude=4, discontinuity_periods={"longitude": 180.0}, validate_axis_order=False)(
            da
        )

    # TODO: Do we need to test on a real dataset with a longitude axis that wraps around, to check
    # that the guardrail is working as expected in that case?


def test_ToTensorTransform():
    da, var_id = _load_data(response_type="DataArray")

    # Run the to_tensor transform
    to_tensor_transform = ToTensorTransform()
    tensor_da = to_tensor_transform(da.values)
    assert isinstance(tensor_da, torch.Tensor), "ToTensor transform did not return a PyTorch tensor."
    assert tensor_da.shape == da.shape, "ToTensor transform did not preserve the shape of the data."


def test_TransposeTransform():
    da, var_id = _load_data(response_type="DataArray")

    # Run the transpose transform
    transpose_transform = TransposeTransform()
    transposed_da = transpose_transform(da)
    # Check that the dimensions have been transposed correctly (should be in reverse order)
    assert transposed_da.dims == da.dims[::-1], "Transpose transform did not work as expected."


def test_VarsToDimensionTransform():
    ds, var_id = _load_data()

    # Run the vars_to_dimension transform
    vars_to_dimension_transform = VarsToDimensionTransform(variables=[var_id, var_id], new_dim="variables")
    da = vars_to_dimension_transform(ds)

    # Check that the new dimension has been added correctly
    assert "variables" in da.dims, "VarsToDimension transform did not work as expected."
    assert da.shape == (2, 184104, 721, 1440), "VarsToDimension transform did not produce the expected output shape."

    # Now test the special case of variables="__all__"
    vars_to_dimension_transform_all = VarsToDimensionTransform(variables="__all__", new_dim="variables")  # type: ignore
    da_all = vars_to_dimension_transform_all(ds)
    assert da_all.shape == (len(ds.data_vars), 184104, 721, 1440), (
        "VarsToDimension transform with variables='__all__' did not produce the expected output shape."
    )


def test_multiple_transforms_1():
    ds, var_id = _load_data()

    # Example of using multiple transforms as a list using transform mapping codes
    transforms_to_apply = [
        {"type": "rename", "var_id": var_id, "new_name": "dewpoint_temperature"},
        {"type": "roll", "dim": "longitude", "shift": None},
    ]

    print("\nApplying multiple transforms using transform mapping codes:")
    for transform in transforms_to_apply:
        if transform["type"] not in transform_mapping:
            raise ValueError(f"Unsupported transform type: {transform['type']}")
        transform_class = transform_mapping[transform["type"]]
        transform = transform_class(**{k: v for k, v in transform.items() if k != "type"})
        ds = transform(ds)

    assert "dewpoint_temperature" in ds.data_vars, "Rename transform did not work as expected."
    assert ds.longitude[0] == -180.0 and ds.longitude[-1] == 179.75, "Roll transform did not work as expected." + str(
        ds.longitude.values
    )


def test_multiple_transforms_2():
    ds, var_id = _load_data()

    # Now let's show how the order of transforms matters by chaining them in different orders.
    # Chain three transforms as follows:
    # 1. Roll longitude with automatic shift detection
    # 2. Reverse latitude axis
    # 3. Subset to a specific region and time range

    chained_transforms = [
        {"type": "roll", "dim": "longitude", "shift": None},
        {"type": "reverse_axis", "dim": "latitude"},
        {
            "type": "subset",
            "variables": [var_id],
            "time": ["2000-01-01 00:00:00", "2000-01-10 23:00:00"],
            "latitude": [-89, 89],
            "longitude": [-179, 179],
        },
    ]

    ds, var_id = _load_data()
    for transform in chained_transforms:
        if transform["type"] not in transform_mapping:
            raise ValueError(f"Unsupported transform type: {transform['type']}")
        transform_class = transform_mapping[transform["type"]]
        transform = transform_class(**{k: v for k, v in transform.items() if k != "type"})
        ds = transform(ds)

    # Check the final shape of the dataset
    assert ds[var_id].shape == (240, 713, 1433), "Chained transforms did not produce the expected output shape."


def test_multiple_transforms_3():
    # Now do the same in the different order to show that the order of transforms matters:
    ds, var_id = _load_data()
    chained_transforms = [
        {"type": "reverse_axis", "dim": "latitude"},
        {
            "type": "subset",
            "variables": [var_id],
            "time": ["2000-01-01 00:00:00", "2000-01-10 23:00:00"],
            "latitude": [89, -89],
            "longitude": [-179, 179],
        },
        {"type": "roll", "dim": "longitude", "shift": None},
    ]

    for transform in reversed(chained_transforms):
        if transform["type"] not in transform_mapping:
            raise ValueError(f"Unsupported transform type: {transform['type']}")
        transform_class = transform_mapping[transform["type"]]
        transform = transform_class(**{k: v for k, v in transform.items() if k != "type"})
        ds = transform(ds)

    assert ds[var_id].shape == (240, 713, 1433), "Chained transforms did not produce the expected output shape."

    print("\nWhat we actually learnt here: _rolling_ the dataset before or after subset STILL WORKS!")
    print("But reversing the axis before/after DOES have an impact!")
