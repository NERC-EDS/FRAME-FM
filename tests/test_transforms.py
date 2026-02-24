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
from io import BytesIO
import zipfile
import xarray as xr
import json
import pandas as pd

from FRAME_FM.transforms import *


KERCHUNK_ZIP = Path("tests/fixtures/ecmwf-era5X_oper_an_sfc_2000_2020_2d_repack.kr1.0.json.zip")
KERCHUNK_FILE = BytesIO()
pdt = pd.to_datetime

# Unzip the file into an in-memory BytesIO object
with KERCHUNK_ZIP.open("rb") as f:
    with zipfile.ZipFile(f) as z:
        with z.open(z.namelist()[0]) as kerchunk_file:
            KERCHUNK_FILE.write(kerchunk_file.read())


def _load_kerchunk_dataset(kerchunk_file: str | Path | BytesIO = KERCHUNK_FILE,
                           response_type: str = "Dataset") -> xr.Dataset | xr.DataArray:
    # Reset the file pointer to the beginning
    kerchunk_file.seek(0)             # type: ignore
    refs = json.load(kerchunk_file)   # type: ignore
    ds = xr.open_dataset(refs, engine="kerchunk")
    if response_type == "DataArray":
        return ds["d2m"].isel(time=slice(0, 2))
    return ds


def test_SubsetTransform():
    ds = _load_kerchunk_dataset()

    # Run the subset transform on a Dataset
    subset_transform = SubsetTransform(variables=["d2m"], time=("2000-01-01", "2000-01-10"), latitude=(-30, 60), longitude=(-40, 100))
    subset_ds = subset_transform(ds)
    assert "d2m" in subset_ds.data_vars, "Variable subsetting did not work as expected."
    assert subset_ds.time.min().values >= pdt("2000-01-01"), "Time subsetting did not work as expected."

    # Run the subset transform on a DataArray
    da = ds["d2m"]
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


def test_RenameTransform():
    ds = _load_kerchunk_dataset()

    # Run the rename transform
    rename_transform = RenameTransform(var_id="d2m", new_name="dewpoint_temperature")
    transformed_ds = rename_transform(ds)
    assert "dewpoint_temperature" in transformed_ds.data_vars, "Rename transform did not work as expected."


def test_NormalizeTransform():
    ds = _load_kerchunk_dataset()

    # Run the normalize transform
    da = ds["d2m"].isel(time=0)
    normalize_transform = NormalizeTransform()
    normalized_da = normalize_transform(da, mean=float(da.mean()), std=float(da.std()))
    assert np.isclose(float(normalized_da.mean()), 0, atol=1e-5), "Normalize transform did not work as expected (mean is not close to 0)."
    assert np.isclose(float(normalized_da.std()), 1, atol=1e-5), "Normalize transform did not work as expected (std is not close to 1)."


def test_ResizeTransform():
    da = _load_kerchunk_dataset(response_type="DataArray")

    # Run the resize transform
    new_size = (2*721*1440,)
    resize_transform = ResizeTransform(size=new_size)
    resized_arr = resize_transform(da)
    # Check that the resized array has the expected shape
    assert resized_arr.shape == new_size, f"Resize transform did not work as expected (shape is {resized_arr.shape} instead of {new_size})"


def test_RollTransform():
    ds = _load_kerchunk_dataset()

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
    da = _load_kerchunk_dataset(response_type="DataArray")

    # Run the scale transform (which is the same as normalize in this case)
    scale_transform = ScaleTransform()
    scaled_da = scale_transform(da, mean=float(da.mean()), std=float(da.std()))   # type: ignore
    assert np.isclose(float(scaled_da.mean()), 0, atol=1e-5), "Scale transform did not work as expected (mean is not close to 0)."  # type: ignore
    assert np.isclose(float(scaled_da.std()), 1, atol=1e-5), "Scale transform did not work as expected (std is not close to 1)."    # type: ignore


def test_ToTensorTransform():
    da = _load_kerchunk_dataset(response_type="DataArray")

    # Run the to_tensor transform
    to_tensor_transform = ToTensorTransform()
    tensor_da = to_tensor_transform(da.values)
    assert isinstance(tensor_da, torch.Tensor), "ToTensor transform did not return a PyTorch tensor."
    assert tensor_da.shape == da.shape, "ToTensor transform did not preserve the shape of the data."


def test_ReverseAxisTransform():
    ds = _load_kerchunk_dataset()

    # Run the reverse axis transform
    reverse_axis_transform = ReverseAxisTransform(dim="latitude")
    reversed_ds = reverse_axis_transform(ds)
    # Check that the latitude axis has been reversed correctly
    assert reversed_ds.latitude[0] == ds.latitude[-1], "Reverse axis transform did not work as expected (first latitude value is not the same as the last latitude value of the original dataset)."
    assert reversed_ds.latitude[-1] == ds.latitude[0], "Reverse axis transform did not work as expected (last latitude value is not the same as the first latitude value of the original dataset)."


def test_multiple_transforms_1():
    ds = _load_kerchunk_dataset()

    # Example of using multiple transforms as a list using transform mapping codes
    transforms_to_apply = [
        {"type": "rename", "var_id": "d2m", "new_name": "dewpoint_temperature"},
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
    ds = _load_kerchunk_dataset()

    # Now let's show how the order of transforms matters by chaining them in different orders.
    # Chain three transforms as follows:
    # 1. Roll longitude with automatic shift detection
    # 2. Reverse latitude axis
    # 3. Subset to a specific region and time range

    chained_transforms = [
        {"type": "roll", "dim": "longitude", "shift": None},
        {"type": "reverse_axis", "dim": "latitude"},
        {"type": "subset", "variables": ["d2m"], "time": ["2000-01-01 00:00:00", "2000-01-10 23:00:00"], "latitude": [-89, 89], "longitude": [-179, 179]},
    ]

    ds = _load_kerchunk_dataset()
    for transform in chained_transforms:
        if transform["type"] not in transform_mapping:
            raise ValueError(f"Unsupported transform type: {transform['type']}")
        transform_class = transform_mapping[transform["type"]]
        transform = transform_class(**{k: v for k, v in transform.items() if k != "type"})
        ds = transform(ds)

    # Check the final shape of the dataset
    assert ds.d2m.shape == (240, 713, 1433), "Chained transforms did not produce the expected output shape."


def test_multiple_transforms_3():
    # Now do the same in the different order to show that the order of transforms matters:
    ds = _load_kerchunk_dataset()
    chained_transforms = [
        {"type": "reverse_axis", "dim": "latitude"},
        {"type": "subset", "variables": ["d2m"], "time": ["2000-01-01 00:00:00", "2000-01-10 23:00:00"], "latitude": [89, -89], "longitude": [-179, 179]},
        {"type": "roll", "dim": "longitude", "shift": None},
    ]

    for transform in reversed(chained_transforms):
        if transform["type"] not in transform_mapping:
            raise ValueError(f"Unsupported transform type: {transform['type']}")
        transform_class = transform_mapping[transform["type"]]
        transform = transform_class(**{k: v for k, v in transform.items() if k != "type"})
        ds = transform(ds)

    assert ds.d2m.shape == (240, 713, 1433), "Chained transforms did not produce the expected output shape."

    print("\nWhat we actually learnt here: _rolling_ the dataset before or after subset STILL WORKS!")
    print("But reversing the axis before/after DOES have an impact!")

