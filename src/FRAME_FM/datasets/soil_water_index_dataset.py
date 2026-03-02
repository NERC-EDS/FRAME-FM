from pathlib import Path

from FRAME_FM.datasets.base_gridded_dataset import BaseGriddedTimeSeriesDataset

SAMPLE_DATA_PATH = "/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/soil_water_index_europe_1km_daily_v1/aggregations/soil_water_index_europe_1km_daily_v1_2015-2025.nca"
SAMPLE_DATA_PATH = "/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/soil_water_index_europe_1km_daily_v1/aggregations/yearly/soil_water_index_europe_1km_daily_v1_*.nca"
#SAMPLE_DATA_PATH = "/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/soil_water_index_europe_1km_daily_v1/data/2024/01/0*/*/*.nc"


class SoilWaterIndexGriddedTimeSeriesDataset(BaseGriddedTimeSeriesDataset):
    """
    A dataset class for loading Soil Water Index data for gridded time series forecasting tasks. 
    This class inherits from the BaseGriddedTimeSeriesDataset and can be extended with 
    Soil Water Index-specific loading or preprocessing logic if needed.
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

    filepaths = glob.glob(SAMPLE_DATA_PATH) if isinstance(SAMPLE_DATA_PATH, str) else SAMPLE_DATA_PATH
    print(f"Testing with filepaths: {filepaths}")

    dataset = SoilWaterIndexGriddedTimeSeriesDataset(
        data_uri=filepaths,
        time_range=("2024-02-01", "2024-02-05"),
        time_stride=1,
        chunks={"time": 1}
    )

    # Parent class will avoid applying the same transforms twice, so we can apply some additional ones here if we want to test them out.
    transforms = [
        {"type": "subset", "lat": (60, 40), "lon": (-5, 5), "time": ("2024-02-02", "2024-02-04")},
        {"type": "vars_to_dimension", "variables": "__all__", "new_dim": "variable"},
        {"type": "to_tensor"},
    ]

    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(sample.shape)

    print("""
          NOTE: Data error:
>>> for a, b in zip(ds.time.values[:-1], ds.time.values[1:]):
...  if a >= b:
...   print(a, b)
... 
2025-07-12T12:00:00.000000000 2021-01-01T12:00:00.000000000
          """)