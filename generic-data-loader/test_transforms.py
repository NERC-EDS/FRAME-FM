from big_geo_loader.transforms import *


def main():
    # Example usage
    kerchunk_url = "https://gws-access.jasmin.ac.uk/public/eds_ai/era5_repack/aggregations/data/ecmwf-era5X_oper_an_sfc_2000_2020_2d_repack.kr1.0.json"
    ds = xr.open_dataset(kerchunk_url, engine="kerchunk")

    # Run the subset transform on a DataArray
    da = ds["d2m"]
    subset_transform = SubsetTransform(time=("2000-01-01", "2000-01-10"), latitude=(-30, 60), longitude=(-40, 100))
    subset_da = subset_transform(da)
    print("Subsetted DataArray:")
    print(subset_da)
    
    # Run the subset transform
    subset_transform = SubsetTransform(variables=["d2m"], time=("2000-01-01", "2000-01-10"), latitude=(-30, 60), longitude=(-40, 100))
    subset_ds = subset_transform(ds)
    print("\nSubsetted Dataset:")
    print(subset_ds)

    # Run the rename transform
    rename_transform = RenameTransform(var_id="d2m", new_name="dewpoint_temperature")
    transformed_ds = rename_transform(ds)
    print("\nTransformed Dataset:")
    print(transformed_ds)

    # Run the normalize transform
    da = ds["d2m"].isel(time=0)
    normalize_transform = NormalizeTransform()
    normalized_da = normalize_transform(da, mean=float(da.mean()), std=float(da.std()))
    print("\nNormalized DataArray:")
    print(normalized_da)

    # Run the resize transform
    resize_transform = ResizeTransform(size=(721*1440,))
    resized_arr = resize_transform(da)
    print("\nResized DataArray:")
    print(resized_arr.shape)

    # Run the roll transform
    roll_transform = RollTransform(dim="longitude", shift=720)
    rolled_ds = roll_transform(ds)
    print("\nRolled Dataset:")
    print(rolled_ds.longitude)

    # Run the scale transform (which is the same as normalize in this case)
    scale_transform = ScaleTransform()
    scaled_da = scale_transform(da, mean=float(da.mean()), std=float(da.std()))
    print("\nScaled DataArray:")
    print(scaled_da)

    # Run the to_tensor transform
    to_tensor_transform = ToTensor()
    tensor_da = to_tensor_transform(da.values)
    print("\nTensor DataArray:")
    print(tensor_da.shape)

    # Run the reverse axis transform
    reverse_axis_transform = ReverseAxisTransform(dim="latitude")
    reversed_ds = reverse_axis_transform(ds)
    print("\nReversed Dataset:")
    print(ds.latitude.values[:5], reversed_ds.latitude.values[:5])

    # Run the roll transform with automatic shift detection
    auto_roll_transform = RollTransform(dim="longitude", shift=None)
    auto_rolled_ds = auto_roll_transform(ds)
    print("\nAuto-Rolled Dataset:")
    print(auto_rolled_ds.longitude)

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

    print("\nTransformed Dataset:")
    assert "dewpoint_temperature" in ds.data_vars, "Rename transform did not work as expected."
    assert ds.longitude[0] == -180.0 and ds.longitude[-1] == 179.75, "Roll transform did not work as expected." + str(ds.longitude.values)
    print(ds)

    print("\nNow let's show how the order of transforms matters by chaining them in different orders...\n\n")
    # Chain three transforms as follows:
    # 1. Roll longitude with automatic shift detection
    # 2. Reverse latitude axis
    # 3. Subset to a specific region and time range
    chained_transforms = [
        {"type": "roll", "dim": "longitude", "shift": None},
        {"type": "reverse_axis", "dim": "latitude"},
        {"type": "subset", "variables": ["d2m"], "time": ["2000-01-01 00:00:00", "2000-01-10 23:00:00"], "latitude": [-30, 60], "longitude": [-40, 100]},
    ]

    ds = xr.open_dataset(kerchunk_url, engine="kerchunk")
    for transform in chained_transforms:
        if transform["type"] not in transform_mapping:
            raise ValueError(f"Unsupported transform type: {transform['type']}")
        transform_class = transform_mapping[transform["type"]]
        transform = transform_class(**{k: v for k, v in transform.items() if k != "type"})
        ds = transform(ds)

    print("\nTransformed Dataset:")
    print(ds)
    print(ds.d2m.shape)
    print("\n---------------------\n")

    # Now do the same in the different order to show that the order of transforms matters:
    ds = xr.open_dataset(kerchunk_url, engine="kerchunk")
    chained_transforms = [
        {"type": "reverse_axis", "dim": "latitude"},
        {"type": "subset", "variables": ["d2m"], "time": ["2000-01-01 00:00:00", "2000-01-10 23:00:00"], "latitude": [60, -30], "longitude": [-40, 100]},
        {"type": "roll", "dim": "longitude", "shift": None},
    ]

    for transform in reversed(chained_transforms):
        if transform["type"] not in transform_mapping:
            raise ValueError(f"Unsupported transform type: {transform['type']}")
        transform_class = transform_mapping[transform["type"]]
        transform = transform_class(**{k: v for k, v in transform.items() if k != "type"})
        ds = transform(ds)

    print("\nTransformed Dataset:")
    print(ds)
    print(ds.d2m.shape)

    print("\nWhat we actually learnt here: _rolling_ the dataset before or after subset STILL WORKS!")
    print("But reversing the axis before/after DOES have an impact!")

if __name__ == "__main__":
    main()
