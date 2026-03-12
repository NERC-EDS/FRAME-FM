# transforms

A location for all our transformation classes and relevant utilities to 
glue them to our framework.

## Overview of transforms

PyTorch `Dataset` classes typically employ a sequence of _transform_ objects that 
are constructed to modify the input data into a form ready for model training/inference.

This usually happens in the `__getitem__()` method, but it may also be appropriate 
to use transforms elsewhere, such as in the `__init__()` method when initial modifications
are required.

Within the `FRAME_FM` package, transforms are all children of the `FRAME_FM.transforms.BaseTransform` 
class. They are all typically run as follows:

```python
from FRAME_FM.transforms import NormalizeTransform

ds = xr.load_dataset(<some_dataset>)
da = ds["d2m"].isel(time=0)

normalize_transform = NormalizeTransform()
normalized_da = normalize_transform(da, mean=float(da.mean()), std=float(da.std()))
```

## Current error (Feb 2026) with `ds.roll` operation on `Xarray.Dataset`

The `ds.roll()` operation on an `Xarray.Dataset` object can cause the system to hang on 
some installations. So far, we have diagnosed that:
- works fine on `xarray==2025.11.0`
- fails (hangs) on `xarray==2026.2.0`


## Supporting pre-existing transforms

In future, we may support transforms from other tools/systems, e.g.:

- `anemoi-transform`: https://anemoi.readthedocs.io/projects/transform/en/latest/: 
- `torchvision.transformsi`: https://docs.pytorch.org/vision/main/transforms.html

At present these are not supported.