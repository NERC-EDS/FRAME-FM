from pathlib import Path

from FRAME_FM.datasets.base_gridded_dataset import BaseGriddedTimeSeriesDataset


class CHESSMetGriddedTimeSeriesDataset(BaseGriddedTimeSeriesDataset):
    """
    A dataset class for loading CHESS-met data for gridded time series forecasting tasks. 
    This class inherits from the BaseGriddedTimeSeriesDataset and can be extended with 
    CHESS-met-specific loading or preprocessing logic if needed.
    """

