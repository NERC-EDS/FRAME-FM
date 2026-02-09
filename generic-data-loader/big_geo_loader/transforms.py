# Define transforms
import xarray as xr
import numpy as np
DA = xr.DataArray
DS = xr.Dataset

from .utils import _check_xarray_object
from torchvision.transforms import ToTensor



class BaseTransform:
    def __call__(self, sample, *args, **kwargs):
        raise NotImplementedError("Transform must implement the __call__ method.")

class NormalizeTransform(BaseTransform):
    def __call__(self, sample, mean: float, std: float):
        # Implement normalization logic here
        _check_xarray_object(sample, expected_type=DA)
        return (sample - mean) / std

class ResizeTransform(BaseTransform):
    def __init__(self, size: tuple):
        self.size = size

    def __call__(self, sample):
        # Implement resizing logic here
        _check_xarray_object(sample, expected_type=DA)
        return sample.to_numpy().reshape(self.size)

class RenameTransform(BaseTransform):
    def __init__(self, var_id: str, new_name: str):
        self.var_id = var_id
        self.new_name = new_name

    def __call__(self, sample):
        # Implement renaming logic here
        _check_xarray_object(sample, expected_type=DS)
        sample = sample.rename_vars({self.var_id: self.new_name})
        return sample
    
class RollTransform(BaseTransform):
    def __init__(self, dim: str, shift: None|int):
        self.dim = dim
        self.shift = shift

    def __call__(self, sample):
        # Implement rolling logic here
        _check_xarray_object(sample, expected_type=DS)
        shift = self.shift
        
        if shift is None:
            # Check if we need to roll
            if float(sample[self.dim].max()) > 350 and float(sample[self.dim].min()) < 10:
                shift = sample.sizes[self.dim] // 2
            else:
                shift = 0

        print(f"Rolling {self.dim} by {shift} positions.")
        rolled = sample.roll({self.dim: shift}, roll_coords=True)

        # Adjust the coordinate values after rolling
        coord_vals = rolled.coords[self.dim].values
        rolled.coords[self.dim] = np.where(coord_vals >= 180., coord_vals - 360., coord_vals)

        print("NEED TESTING HERE TO CHECK THIS ROLLING LOGIC WORKS AS EXPECTED...")
        return rolled
    
class ReverseAxisTransform(BaseTransform):
    def __init__(self, dim: str):
        self.dim = dim

    def __call__(self, sample):
        # Implement axis reversal logic here
        _check_xarray_object(sample, expected_type=DS)
        ds_rev = sample.isel(**{self.dim: slice(None, None, -1)})
        return ds_rev

class ScaleTransform(NormalizeTransform): 
    pass



transform_mapping = {
    "normalize": NormalizeTransform,
    "resize": ResizeTransform,
    "rename": RenameTransform,
    "roll": RollTransform,
    "scale": ScaleTransform,
    "to_tensor": ToTensor,
    "reverse_axis": ReverseAxisTransform,
}


def resolve_transform(transform_config: dict) -> BaseTransform:
    """
    If a transform is a dictionary with a "type" key, resolve it to the corresponding transform class instance.
    If it is already an instance of a transform class, return it as is.
    Args:
    - transform_config (dict or BaseTransform): The transform configuration to resolve.
    Returns:
    - BaseTransform: An instance of a transform class.
    """
    if isinstance(transform_config, BaseTransform):
        return transform_config
    
    transform_type = transform_config.get("type")
    if transform_type not in transform_mapping:
        raise ValueError(f"Unsupported transform type: {transform_type}")
    
    transform_class = transform_mapping[transform_type]
    return transform_class(**{k: v for k, v in transform_config.items() if k != "type"})
