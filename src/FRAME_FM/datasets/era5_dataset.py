from pathlib import Path

from FRAME_FM.datasets.base_gridded_dataset import BaseGriddedTimeSeriesDataset


class ERA5GriddedTimeSeriesDataset(BaseGriddedTimeSeriesDataset):
    """
    A dataset class for loading ERA5 reanalysis data for gridded time series forecasting tasks. 
    This class inherits from the BaseGriddedTimeSeriesDataset and can be extended with 
    ERA5-specific loading or preprocessing logic if needed.
    """

    def __init__(self, 
                 data_uri: str | Path | list | tuple,
                 transforms: list | None = None,
                 time_range: tuple | None = None,
                 time_stride: int = 16,
                 chunks: dict | None = None,
                 override_transforms: bool = False
                 ):
        super().__init__(
            data_uri=data_uri,
            time_range=time_range,
            transforms=transforms,
            time_stride=time_stride,
            chunks=chunks,
            override_transforms=override_transforms
        )


if __name__ == "__main__":

    SAMPLE_DATA_PATH = "/gws/ssde/j25b/eds_ai/public/era5_repack/aggregations/data/ecmwf-era5X_oper_an_sfc_2000_2020_*_repack.kr1.0.json"

    dataset = ERA5GriddedTimeSeriesDataset(
        data_uri=SAMPLE_DATA_PATH,
        time_range=("2010-05-01", "2012-10-05"),
        time_stride=4
    )

    ds = dataset.data
    print(ds)
    assert "time" in ds.coords, "Dataset must have a time coordinate"
    assert set(["u10", "v10", "t2m", "d2m"]).issubset(set(ds.data_vars)), "Dataset must contain the required variables" 

    # Parent class will avoid applying the same transforms twice, so we can apply some additional ones here if we want to test them out.
    transforms = [
        {"type": "subset", "latitude": (60, -30), "longitude": (-40, 100)},
        {"type": "vars_to_dimension", "variables": ["u10", "v10", "t2m", "d2m"], "new_dim": "variable"},
        {"type": "to_tensor"},
    ]

    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(sample.shape)
