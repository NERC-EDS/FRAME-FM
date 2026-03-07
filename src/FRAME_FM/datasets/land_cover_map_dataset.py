from pathlib import Path

from FRAME_FM.datasets.base_gridded_dataset import BaseGeoTIFFDataset


class LandCoverMapGriddedDataset(BaseGeoTIFFDataset):
    """
    A dataset class for loading the Land Cover Map data for gridded time series forecasting tasks. 
    This class inherits from the BaseGeoTIFFDataset and can be extended with 
    Land Cover Map-specific loading or preprocessing logic if needed.
    """

