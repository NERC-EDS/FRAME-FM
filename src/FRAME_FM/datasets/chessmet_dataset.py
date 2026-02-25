from pathlib import Path

from FRAME_FM.datasets.base_gridded_dataset import BaseGriddedTimeSeriesDataset

SAMPLE_DATA_URI = "/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/chess-met/aggregations/chess-met_precip*.nca"


class CHESSMetGriddedTimeSeriesDataset(BaseGriddedTimeSeriesDataset):
    """
    A dataset class for loading CHESS-met data for gridded time series forecasting tasks. 
    This class inherits from the BaseGriddedTimeSeriesDataset and can be extended with 
    CHESS-met-specific loading or preprocessing logic if needed.
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

    dataset = CHESSMetGriddedTimeSeriesDataset(
        data_uri=SAMPLE_DATA_URI,
        time_range=("2016-01-01", "2018-10-20"),
        time_stride=1,
        chunks={"time": 64}
    )

    # Parent class will avoid applying the same transforms twice, so we can apply some additional ones here if we want to test them out.
    transforms = [
        {"type": "subset", "variables": ["precip"], "y": (100500., 257500.0), "x": (200500.0, 156500.0), "time": ("2016-01-27", "2016-01-02")},
        {"type": "vars_to_dimension", "variables": "__all__", "new_dim": "variable"},
        {"type": "to_tensor"},
    ]

    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(sample.shape)
