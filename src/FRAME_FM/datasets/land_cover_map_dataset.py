from pathlib import Path

from FRAME_FM.datasets.base_gridded_dataset import BaseGeoTIFFDataset

SAMPLE_DATA_URI = "/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/land_cover_map_2015/data/LCM2015_GB_1km_percent_cover_aggregate_class.tif"


class LandCoverMapGriddedDataset(BaseGeoTIFFDataset):
    """
    A dataset class for loading the Land Cover Map data for gridded time series forecasting tasks. 
    This class inherits from the BaseGeoTIFFDataset and can be extended with 
    Land Cover Map-specific loading or preprocessing logic if needed.
    """

    def __init__(self, 
                 data_uri: str | Path | list | tuple,
                 transforms: list | None = None,
                 chunks: dict | None = None,
                 override_transforms: bool = False
                 ):
        super().__init__(
            data_uri=data_uri,
            transforms=transforms,
            chunks=chunks,
            override_transforms=override_transforms
        )


if __name__ == "__main__":

    dataset = LandCoverMapGriddedDataset(
        data_uri=SAMPLE_DATA_URI
    )

    # Parent class will avoid applying the same transforms twice, so we can apply some additional ones here if we want to test them out.
    transforms = [
        {"type": "subset", "variables": ["band_data"], "y": (10000, 500), "x": (1000, 5000)},
        {"type": "vars_to_dimension", "variables": ["band_data"], "new_dim": "variable"},
        {"type": "to_tensor"},
    ]

    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(sample.shape)
