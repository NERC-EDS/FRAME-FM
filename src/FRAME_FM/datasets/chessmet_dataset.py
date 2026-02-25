from pathlib import Path

from FRAME_FM.datasets.base_gridded_dataset import BaseGriddedTimeSeriesDataset

SAMPLE_DATA_PATH = "tests/fixtures/ecmwf-era5X_oper_an_sfc_2000_2020_2d_repack.kr1.0.json.zip"


class ERA5GriddedTimeSeriesDataset(BaseGriddedTimeSeriesDataset):
    """
    A dataset class for loading ERA5 reanalysis data for gridded time series forecasting tasks. 
    This class inherits from the BaseGriddedTimeSeriesDataset and can be extended with 
    ERA5-specific loading or preprocessing logic if needed.
    """
    transforms = [
        {"type": "subset", "variables": ["d2m"], "latitude": (60, -30), "longitude": (-40, 100)},
        {"type": "vars_to_dimension", "variables": "__all__", "new_dim": "variable"},
        {"type": "to_tensor"}
    ]


    def __init__(self, 
                    data_uri: str | Path = SAMPLE_DATA_PATH,
                    time_range: tuple | None = None,
                    transforms: list | None = None,
                    time_stride: int = 16
                 ):
        super().__init__(
            data_uri=data_uri,
            time_range=time_range,
            transforms=transforms or self.transforms,
            time_stride=time_stride
        )


if __name__ == "__main__":
    dataset = ERA5GriddedTimeSeriesDataset(
        time_range=("2010-05-01", "2010-10-01"),
        time_stride=1
    )

    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(sample.shape)
