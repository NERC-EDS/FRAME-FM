from pathlib import Path

from FRAME_FM.datasets.base_shapefile_dataset import BaseShapefileDataset

SAMPLE_TOPSOIL_CARBON_DATA_URI = "/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/model_estimates_of_topsoil_carbon/data"
SAMPLE_TOPSOIL_PROPERTIES_DATA_URI = "/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/model_estimates_of_topsoil_pH_and_bulk_density/properties"


class TopsoilDataset(BaseShapefileDataset):
    """
    A dataset class for loading the Topsoil datasets for gridded time series forecasting tasks. 
    This class inherits from the BaseShapefileDataset and can be extended with 
    Topsoil dataset-specific loading or preprocessing logic if needed.
    """

    def __init__(self, 
                 data_uri: str | Path | list | tuple,
                 transforms: list | None = None,
                 override_transforms: bool = False
                 ):
        super().__init__(
            data_uri=data_uri,
            transforms=transforms,
            override_transforms=override_transforms
        )


if __name__ == "__main__":

    dataset = TopsoilDataset(
        data_uri=SAMPLE_TOPSOIL_CARBON_DATA_URI
    )

    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(sample.shape)

    print("----------------------------------")
    dataset = TopsoilDataset(
        data_uri=SAMPLE_TOPSOIL_PROPERTIES_DATA_URI
    )

    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(sample.shape)
    