# SPDX-FileCopyrightText: 2026 2026 FRAME-FM Contributors
#
# SPDX-License-Identifier: Apache-2.0

# datasets

A location for the FRAME-FM datasets, which are all sub-classes of `torch.utils.data.Dataset`.
They all use the standard model of exposing a _dataset_ as a generator that can be directly 
used by a `DataLoader` object. The common interface is:
- contruction: `__init__()`
- length: `__len__()`
- get item by index: `__getitem__(idx)`

## Class Hierarchy

The class hierarchy is built as follows:

```
torch.utils.data.Dataset
    - BaseDataset
        - BaseGriddedDataset
        - BaseGeoTIFFDataset
            - BaseASCIIGridDataset
            - LandCoverMapGriddedDataset
        - BaseShapefileDataset
            - TopsoilDataset
        - BaseGriddedTimeSeriesDataset
            - CHESSMetGriddedTimeSeriesDataset
            - ERA5GriddedTimeSeriesDataset
            - SoilWaterIndexGriddedTimeSeriesDataset
        - COSMOSUKSiteTimeSeriesDataset
```

## Changing datasets using `preprocessors` and `transforms`

Each `Dataset` class can have two types of _operations_ defined by arguments 
sent to the contstructor:
- `preprocessors`:
    - A list of operations that get run when the Dataset instance is created.
    - These get run once only.
    - They operate sequentially, with the first taking in an `xr.Dataset`
    - The final object is saved in `self.data`
    - The resulting output should be ready for use by the standard methods:
        - `def __len__(self):`
        - `def __getitem__(self, idx):`
- `transforms`:
    - A list of operations that get run at training time, within the `__getitem__(idx)`
      call.
    - These are run whenever a `DataLoader` needs to access single items or batches
      of items with a `Dataset` object.
    - These are typically run like this:
      ```python
      for transform in transforms:
          sample = transform(sample)
      ```

Note that the `FRAME_FM.transforms.transforms.py` module contains all the transform 
classes that can be in either/both of the `preprocessors` and `transforms` lists.

See the examples in the unit tests: `tests/transforms/test_transforms.py`

See the `Dataset` unit tests for examples: `tests/datasets/test_*.py`