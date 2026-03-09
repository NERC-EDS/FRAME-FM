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
    BaseDataset
        BaseGriddedDataset

        BaseGeoTIFFDataset
            BaseASCIIGridDataset
            LandCoverMapGriddedDataset

        BaseShapefileDataset
            TopsoilDataset

        BaseGriddedTimeSeriesDataset

            CHESSMetGriddedTimeSeriesDataset
            ERA5GriddedTimeSeriesDataset
            SoilWaterIndexGriddedTimeSeriesDataset

        COSMOSUKSiteTimeSeriesDataset
```

## Changing datasets using `preprocessors` and `transforms`

