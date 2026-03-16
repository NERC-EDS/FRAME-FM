from pathlib import Path

from FRAME_FM.datasets.base_gridded_dataset import BaseGriddedTimeSeriesDataset



class SoilWaterIndexGriddedTimeSeriesDataset(BaseGriddedTimeSeriesDataset):
    """
    A dataset class for loading Soil Water Index data for gridded time series forecasting tasks. 
    This class inherits from the BaseGriddedTimeSeriesDataset and can be extended with 
    Soil Water Index-specific loading or preprocessing logic if needed.
    """

