from pathlib import Path

from FRAME_FM.datasets.base_gridded_dataset import BaseGriddedTimeSeriesDataset

SAMPLE_DATA_PATH = "tests/transforms/fixtures/ecmwf-era5X_oper_an_sfc_2000_2020_2d_repack.kr1.0.json.zip"


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

    dataset = ERA5GriddedTimeSeriesDataset(
        data_uri=SAMPLE_DATA_PATH,
        time_range=("2010-05-01", "2010-10-01"),
        time_stride=1
    )

    # Parent class will avoid applying the same transforms twice, so we can apply some additional ones here if we want to test them out.
    transforms = [
        {"type": "subset", "variables": ["d2m"], "latitude": (60, -30), "longitude": (-40, 100)},
        {"type": "vars_to_dimension", "variables": "__all__", "new_dim": "variable"},
        {"type": "to_tensor"},
    ]

    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(sample.shape)
