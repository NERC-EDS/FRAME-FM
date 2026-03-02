# Define transforms
import xarray as xr
import cf_xarray  # noqa: F401 - We just need to register the accessor for CF-compliant operations on xarray objects
import numpy as np
DA = xr.DataArray
DS = xr.Dataset

import torch

from FRAME_FM.utils.transform_utils import check_object_type
from FRAME_FM.utils.data_utils import convert_subset_selectors_to_slices


class BaseTransform:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, sample):
        raise NotImplementedError("Transform must implement the __call__ method.")


class FillMissingValueTransform(BaseTransform):
    def __init__(self, strategy: str = "constant", fill_value: None | float = None, 
                 method: None |str = "constant"):
        self.strategy = strategy
        self.fill_value = fill_value
        self.method = method

    def __call__(self, sample: DS | DA) -> DS | DA:
        # Implement missing value filling logic here
        check_object_type(sample, allowed_types=(DS, DA), caller=self.__class__.__name__)

        # Depending on the method, implement infilling strategy
        if self.strategy == "constant":
            if self.fill_value is None:
                raise ValueError("fill_value must be provided for 'constant' method.")
            filled = sample.fillna(self.fill_value)

        elif self.strategy == "interpolate":
            filled = sample.interpolate_na(dim=None, method=self.method)  # type: ignore

        else:
            raise ValueError(f"Unsupported fill strategy: {self.strategy}")

        return filled


class FillNaNTransform(FillMissingValueTransform):
    pass


class NormalizeTransform(BaseTransform):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(self, sample: DA) -> DA:
        # Implement normalization logic here
        check_object_type(sample, allowed_types=DA, caller=self.__class__.__name__)
        return (sample - self.mean) / self.std


class ScaleTransform(NormalizeTransform): 
    pass


class RenameTransform(BaseTransform):
    def __init__(self, var_id: str, new_name: str):
        self.var_id = var_id
        self.new_name = new_name

    def __call__(self, sample: DS) -> DS:
        # Implement renaming logic here
        check_object_type(sample, allowed_types=DS, caller=self.__class__.__name__)
        sample = sample.rename_vars({self.var_id: self.new_name})
        return sample


class ResampleTransform(BaseTransform):
    def __init__(self, dim: str, freq: str, method: str = "mean"):
        self.dim = dim
        self.freq = freq
        self.method = method

    def __call__(self, sample):
        # Implement resampling logic here
        check_object_type(sample, allowed_types=(DS, DA), caller=self.__class__.__name__)
        if self.method not in ["mean", "sum", "max", "min", "median"]:
            raise ValueError(f"Unsupported resampling method: {self.method}")
        
        # Select the appropriate resampling method from xarray's ResampleGroupBy based on the provided method string
        resampled = sample.resample({self.dim: self.freq})
        
        if not hasattr(resampled, self.method):
            raise ValueError(f"Invalid resample method: {self.method}")

        result = getattr(resampled, self.method)()
        return result


class ResizeTransform(BaseTransform):
    def __init__(self, size: tuple):
        self.size = size

    def __call__(self, sample):
        # Implement resizing logic here
        check_object_type(sample, allowed_types=DA, caller=self.__class__.__name__)
        return sample.to_numpy().reshape(self.size)


class RollTransform(BaseTransform):
    def __init__(self, dim: str, shift: None|int):
        self.dim = dim
        self.shift = shift

    def __call__(self, sample):
        # Implement rolling logic here
        check_object_type(sample, allowed_types=DS, caller=self.__class__.__name__)
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

        return rolled


class ReverseAxisTransform(BaseTransform):
    def __init__(self, dim: str):
        self.dim = dim

    def __call__(self, sample):
        # Implement axis reversal logic here
        check_object_type(sample, allowed_types=DS, caller=self.__class__.__name__)
        ds_rev = sample.isel(**{self.dim: slice(None, None, -1)})
        return ds_rev


class SubsetTransform(BaseTransform):
    def __init__(self, **subset_selectors):
        if "variables" in subset_selectors:
            variables = subset_selectors.pop("variables")
            self.variables = variables if isinstance(variables, (list, tuple)) else [variables]
        else:
            self.variables = None
        self.subset_selectors = convert_subset_selectors_to_slices(subset_selectors)

    def __call__(self, sample):
        # Implement subsetting logic here
        check_object_type(sample, allowed_types=(DS, DA), caller=self.__class__.__name__)

        if self.variables is None:
            # If no specific variables are provided, apply the subset to all variables in 
            # the Dataset or the single DataArray
            return sample.sel(**self.subset_selectors)
        
        # If we have variables then we need to create a new Dataset with only those 
        # variables and apply the subset selectors to each variable
        ds = xr.Dataset()
        ds.attrs.update(sample.attrs)

        for var_id in self.variables:
            # Use common subset selectors unless overridden by variable-specific selectors
            if self.subset_selectors:
                ds[var_id] = sample[var_id].sel(**self.subset_selectors)
            else:
                ds[var_id] = sample[var_id]

        return ds


class ToTensorTransform(BaseTransform):
    def __call__(self, sample):
        # Implement conversion to PyTorch tensor here
        check_object_type(sample, allowed_types=(DA, np.ndarray), caller=self.__class__.__name__)
        if isinstance(sample, DA):
            sample = sample.values
        return torch.from_numpy(sample)


class VarsToDimensionTransform(BaseTransform):
    """
    A transform that takes a list of variables from a Dataset and stacks them into a 
    new dimension, effectively converting the variable dimension into a coordinate 
    dimension. This is useful for models that expect a single multi-channel input 
    rather than separate variables.

    Since the purpose is to prepare the data for conversion to a Tensor, we assume 
    that ancillary variables that are not genuine coordinates can be dropped.
    """
    exclusion_vars = ["time_bounds", "lat_bounds", "lon_bounds", 
                       "time_bnds", "lat_bnds", "lon_bnds",
                       "crs", "spatial_ref", "bounds", "bnds"]
    
    def __init__(self, variables: list, new_dim: str):
        self.variables = variables
        self.new_dim = new_dim

    def __call__(self, sample):
        # Implement logic to convert variables to a new dimension here
        check_object_type(sample, allowed_types=DS, caller=self.__class__.__name__)
        
        # Check special case of variables = "__all__", take all variables and filter out those not needed/suitable
        if self.variables == "__all__":

            # Exclude variables relate to bounds and coordinates
            bounds_vars = set([b_list[0] for b_list in sample.cf.bounds.values()])
            vars_without_time = set([var_id for var_id in sample.data_vars 
                                     if not hasattr(sample[var_id], "time")])
            exclusion_vars = set([var_id for var_id in self.exclusion_vars if var_id in sample.data_vars])

            # Combine all exclusion criteria into a single set of variables to drop
            all_exclusion_vars = bounds_vars | vars_without_time | exclusion_vars

            # Drop the variables from the sample.
            sample.drop_vars(all_exclusion_vars)
            # Remove those variables from the wish list
            variables = set(sample.data_vars) - all_exclusion_vars

        else:
            variables = self.variables

        # Create a set of arrays to concatenate together
        arrays = [sample[var_id] for var_id in variables]

        stacked = xr.concat(arrays, dim=self.new_dim)
        return stacked


transform_mapping = {
    "fill_missing": FillMissingValueTransform,
    "fill_nan": FillNaNTransform,
    "normalize": NormalizeTransform,
    "rename": RenameTransform,
    "resize": ResizeTransform,
    "resample": ResampleTransform,
    "reverse_axis": ReverseAxisTransform,
    "roll": RollTransform,
    "scale": ScaleTransform,
    "subset": SubsetTransform,
    "to_tensor": ToTensorTransform,
    "vars_to_dimension": VarsToDimensionTransform
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
