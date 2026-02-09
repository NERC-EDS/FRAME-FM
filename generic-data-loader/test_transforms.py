from big_geo_loader.transforms import *


def main():
    # Example usage
    kerchunk_url = "https://gws-access.jasmin.ac.uk/public/eds_ai/era5_repack/aggregations/data/ecmwf-era5X_oper_an_sfc_2000_2020_2d_repack.kr1.0.json"
    ds = xr.open_dataset(kerchunk_url, engine="kerchunk")
    
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


if __name__ == "__main__":
    main()
