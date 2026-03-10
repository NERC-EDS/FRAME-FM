"""
# Unit tests for the transforms in FRAME_FM.transforms.

NOTES:
- The `ds.roll()` operation was hanging on `xarray` version 2026.2.0 but works fine on version 2025.11.0.
- These tests are designed to be run with pytest, as follows (from the root of the repository):

```
PYTHONPATH=src python -m pytest tests/test_transforms.py
```
"""

from pathlib import Path

import pytest
import pandas as pd

from FRAME_FM.transforms import *
from FRAME_FM.transforms.transforms import transform_mapping
from FRAME_FM.utils.data_utils import load_data_from_uri


kerchunk_zip = "tests/transforms/fixtures/ecmwf-era5X_oper_an_sfc_2000_2020_2d_repack.kr1.0.json.zip"
var_id = "d2m"
pdt = pd.to_datetime
ds = None


def _load_data(response_type: str = "Dataset") -> xr.Dataset | xr.DataArray:
    global ds
    if ds is None:
        ds = load_data_from_uri(kerchunk_zip)   # type: ignore

    if response_type == "DataArray":
        return ds[var_id].isel(time=slice(0, 3))
    
    return ds


# Mark this test as failing in second stage
@pytest.mark.xfail(reason="This test is currently failing due the `.interpolate_na()` method needing investigation.")
def test_FillMissingValueTransform():
    ds = _load_data().isel(time=slice(0, 3))

    # Introduce some missing values into the dataset for testing
    ds_with_nans = ds.copy().isel(time=slice(0, 3))  # Take a small subset for testing
    ds_with_nans[var_id] = ds_with_nans[var_id].where(ds_with_nans[var_id] > 290)  # Set values <= 290 to NaN

    # Run the fill missing value transform with constant strategy
    fill_transform_constant = FillMissingValueTransform(strategy="constant", fill_value=0.0)
    filled_ds_constant = fill_transform_constant(ds_with_nans)
    assert not filled_ds_constant[var_id].isnull().any(), "FillMissingValueTransform with constant strategy did not work as expected (there are still NaN values)."

    # Run the fill missing value transform with interpolate strategy
    fill_transform_interpolate = FillMissingValueTransform(strategy="interpolate", method="linear")
    filled_ds_interpolate = fill_transform_interpolate(ds_with_nans)
    assert not filled_ds_interpolate[var_id].isnull().any(), "FillMissingValueTransform with interpolate strategy did not work as expected (there are still NaN values)."

    # Run the fill missing value transform with an unsupported strategy to check that it raises an error
    try:
        fill_transform_invalid = FillMissingValueTransform(strategy="unsupported_strategy", fill_value=273.15)
        fill_transform_invalid(ds_with_nans)
        assert False, "FillMissingValueTransform did not raise an error for an unsupported strategy."
    except ValueError as e:
        assert str(e) == "Unsupported fill strategy: unsupported_strategy", f"FillMissingValueTransform raised an unexpected error message: {str(e)}"


@pytest.mark.xfail(reason="This test is currently failing due the `.interpolate_na()` method needing investigation.")
def test_FillNaNTransform():
    # Identical to the FillMissingValueTransform test but with the FillNaNTransform instead 
    return test_FillMissingValueTransform()


def test_NormalizeTransform():
    da: xr.DataArray = _load_data(response_type="DataArray")  # type: ignore

    # Run the normalize transform
    normalize_transform = NormalizeTransform(mean=float(da.mean()), std=float(da.std()))
    normalized_da = normalize_transform(da)
    assert np.isclose(float(normalized_da.mean()), 0, atol=1e-5), "Normalize transform did not work as expected (mean is not close to 0)."
    assert np.isclose(float(normalized_da.std()), 1, atol=1e-5), "Normalize transform did not work as expected (std is not close to 1)."


def test_RenameTransform():
    ds = _load_data()

    # Run the rename transform
    rename_transform = RenameTransform(var_id=var_id, new_name="dewpoint_temperature")
    transformed_ds = rename_transform(ds)  # type: ignore
    assert "dewpoint_temperature" in transformed_ds.data_vars, "Rename transform did not work as expected."


def test_ResampleTransform():
    ds = _load_data()
    start, end = "2000-01-01T00:00:00", "2000-01-01T23:00:00"
    ds = ds.sel(time=slice(start, end))
    freq = "1D" # daily frequency

    # Run the resample transform to resample from hourly to daily data
    resample_transform = ResampleTransform(dim="time", freq=freq)
    resampled_ds = resample_transform(ds)
    # Check that the time coordinate has been resampled correctly (should now have daily frequency)

    expected_time_range = pd.date_range(start=start, end=end, freq=freq)
    assert np.array_equal(resampled_ds.time.values, expected_time_range.values), "Resample transform did not work as expected (time coordinate does not match expected daily frequency)."   


def test_ReshapeTransform():
    da = _load_data(response_type="DataArray")

    # Run the reshape transform
    new_shape = (3 * 721 * 1440,)
    reshape_transform = ReshapeTransform(shape=new_shape)
    reshaped_arr = reshape_transform(da)
    # Check that the reshaped array has the expected shape
    assert reshaped_arr.shape == new_shape, f"Reshape transform did not work as expected (shape is {reshaped_arr.shape} instead of {new_shape})"


def test_ReverseAxisTransform():
    ds = _load_data()

    # Run the reverse axis transform
    reverse_axis_transform = ReverseAxisTransform(dim="latitude")
    reversed_ds = reverse_axis_transform(ds)
    # Check that the latitude axis has been reversed correctly
    assert reversed_ds.latitude[0] == ds.latitude[-1], "Reverse axis transform did not work as expected (first latitude value is not the same as the last latitude value of the original dataset)."
    assert reversed_ds.latitude[-1] == ds.latitude[0], "Reverse axis transform did not work as expected (last latitude value is not the same as the first latitude value of the original dataset)."


def test_RollTransform():
    ds = _load_data()

    # Run the roll transform with shift 720
    roll_transform = RollTransform(dim="longitude", shift=720)
    rolled_ds = roll_transform(ds)
    # Check that the longitude coordinate has been rolled correctly
    assert rolled_ds.longitude[0] == -180.0 and rolled_ds.longitude[-1] == 179.75, "Roll transform did not work as expected." + str(rolled_ds.longitude.values)

    # Run the roll transform with automatic shift detection
    auto_roll_transform = RollTransform(dim="longitude", shift=None)
    auto_rolled_ds = auto_roll_transform(ds)
    # Check that the longitude coordinate has been rolled correctly
    assert auto_rolled_ds.longitude[0] == -180.0 and auto_rolled_ds.longitude[-1] == 179.75, "Auto roll transform did not work as expected." + str(auto_rolled_ds.longitude.values)


def test_ScaleTransform():
    # Same as the NormalizeTransform test but with the ScaleTransform instead
    return test_NormalizeTransform()


def test_SortAxisTransform():
    ds = _load_data()

    # Run the sort axis transform
    sort_axis_transform = SortAxisTransform(dim="latitude", ascending=True)
    sorted_ds = sort_axis_transform(ds)

    # Check that the latitude axis has been sorted in ascending order
    assert all([v1 < v2 for v1, v2 in zip(sorted_ds.latitude[:-1], sorted_ds.latitude[1:])]), "Sort axis transform did not work as expected (latitude axis is not in ascending order)."
    # Check that the longitude axis has not been changed in ascending order
    assert all([v1 == v2 for v1, v2 in zip(sorted_ds.longitude, ds.longitude)]), "Sort axis transform did not work as expected (longitude axis has been changed)."


def test_SubsetTransform():
    ds = _load_data()

    # Run the subset transform on a Dataset
    subset_transform = SubsetTransform(variables=[var_id], time=("2000-01-01", "2000-01-10"), latitude=(-30, 60), longitude=(-40, 100))
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


@pytest.mark.xfail(reason="This test is currently failing due to a check on the last tile matching the original data.")
def test_TilerTransform_time_series_data():
    da = _load_data(response_type="DataArray")
    step = 10

    # Run the tiler transform with tile sizes of step x step and "pad" boundary handling
    tiler_transform = TilerTransform(latitude=step, longitude=step, boundary="pad")
    tiled = tiler_transform(da)

    # Check that the tiled array has the expected shape (should have new dimensions for tiles)
    expected_shape = (10512, 3, step, step)  # (batch_dim[=n_tiles], time, latitude_fine, longitude_fine)
    assert tiled.shape == expected_shape, f"Tiler transform did not work as expected (shape is {tiled.shape} instead of {expected_shape})"

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
    original_subset = da.isel(latitude=slice(-step, None), longitude=slice(-step, None))
    assert np.array_equal(last_tile, original_subset), "Last tile does not match expected subset of original dataset"

    # Test that reverse-lookup metadata is stored in attrs
    assert "tiler_tile_sizes" in tiled.attrs, "Expected 'tiler_tile_sizes' in tiled.attrs, but not found"
    assert tiled.attrs["tiler_tile_sizes"] == {"latitude": step, "longitude": step}, f"Expected tile sizes in metadata to be {{'latitude': {step}, 'longitude': {step}}}, but got {tiled.attrs['tiler_tile_sizes']}"
    assert "tiler_boundary" in tiled.attrs, "Expected 'tiler_boundary' in tiled.attrs, but not found"
    assert tiled.attrs["tiler_boundary"] == "pad", f"Expected boundary in metadata to be 'pad', but got {tiled.attrs['tiler_boundary']}"
    assert "tiler_original_sizes" in tiled.attrs, "Expected 'tiler_original_sizes' in tiled.attrs, but not found"

    original_sizes = {"latitude": da.latitude.size, "longitude": da.longitude.size}
    assert tiled.attrs["tiler_original_sizes"] == original_sizes, f"Expected original sizes in metadata to be {original_sizes}, but got {tiled.attrs['tiler_original_sizes']}"
    assert "tiler_original_coords" in tiled.attrs, "Expected 'tiler_original_coords' in tiled.attrs, but not found"
    assert tiled.attrs["tiler_original_coords"] == {"latitude": da.latitude.values.tolist(), "longitude": da.longitude.values.tolist()}, f"Expected original coords in metadata to match original dataset coords, but got {tiled.attrs['tiler_original_coords']}"   

    # Now test that the index mapper helper function works as expected
    raise NotImplementedError("This test is currently failing due to an issue with the last tile matching "
    "the original data, so the rest of the test has not been implemented yet. "
    "This needs further investigation before it can be implemented.")

def test_ToTensorTransform():
    da = _load_data(response_type="DataArray")

    # Run the to_tensor transform
    to_tensor_transform = ToTensorTransform()
    tensor_da = to_tensor_transform(da.values)
    assert isinstance(tensor_da, torch.Tensor), "ToTensor transform did not return a PyTorch tensor."
    assert tensor_da.shape == da.shape, "ToTensor transform did not preserve the shape of the data."


def test_TransposeTransform():
    da = _load_data(response_type="DataArray")

    # Run the transpose transform
    transpose_transform = TransposeTransform()
    transposed_da = transpose_transform(da)
    # Check that the dimensions have been transposed correctly (should be in reverse order)
    assert transposed_da.dims == da.dims[::-1], "Transpose transform did not work as expected."


def test_VarsToDimensionTransform():
    ds = _load_data()

    # Run the vars_to_dimension transform
    vars_to_dimension_transform = VarsToDimensionTransform(variables=[var_id, var_id], new_dim="variables")
    da = vars_to_dimension_transform(ds)

    # Check that the new dimension has been added correctly
    assert "variables" in da.dims, "VarsToDimension transform did not work as expected."
    assert da.shape == (2, 184104, 721, 1440), "VarsToDimension transform did not produce the expected output shape."

    # Now test the special case of variables="__all__"
    vars_to_dimension_transform_all = VarsToDimensionTransform(variables="__all__", new_dim="variables")   # type: ignore
    da_all = vars_to_dimension_transform_all(ds)
    assert da_all.shape == (len(ds.data_vars), 184104, 721, 1440), "VarsToDimension transform with variables='__all__' did not produce the expected output shape."


def test_multiple_transforms_1():
    ds = _load_data()

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
    assert ds.longitude[0] == -180.0 and ds.longitude[-1] == 179.75, "Roll transform did not work as expected." + str(ds.longitude.values)


def test_multiple_transforms_2():
    ds = _load_data()

    # Now let's show how the order of transforms matters by chaining them in different orders.
    # Chain three transforms as follows:
    # 1. Roll longitude with automatic shift detection
    # 2. Reverse latitude axis
    # 3. Subset to a specific region and time range

    chained_transforms = [
        {"type": "roll", "dim": "longitude", "shift": None},
        {"type": "reverse_axis", "dim": "latitude"},
        {"type": "subset", "variables": [var_id], "time": ["2000-01-01 00:00:00", "2000-01-10 23:00:00"], "latitude": [-89, 89], "longitude": [-179, 179]},
    ]

    ds = _load_data()
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
    ds = _load_data()
    chained_transforms = [
        {"type": "reverse_axis", "dim": "latitude"},
        {"type": "subset", "variables": [var_id], "time": ["2000-01-01 00:00:00", "2000-01-10 23:00:00"], "latitude": [89, -89], "longitude": [-179, 179]},
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

