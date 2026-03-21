# Define transforms
import xarray as xr
import cf_xarray  # noqa: F401 - We just need to register the accessor for CF-compliant operations on xarray objects
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
import math

from FRAME_FM.utils.common_utils import convert_subset_selectors_to_slices, check_object_type
from FRAME_FM.utils.transform_utils import CRS_conversion_spec, CRS_convertor

DA = xr.DataArray
DS = xr.Dataset
TT = torch.Tensor


class BaseTransform:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, sample):
        raise NotImplementedError("Transform must implement the __call__ method.")


def _parse_coordinate(coord):
    return pd.to_datetime(coord) if isinstance(coord, str) else coord


class AddFixedCoordinates(BaseTransform):
    def __init__(self, coords: dict[str, float]):
        self.coords = {dim: _parse_coordinate(coord) for dim, coord in coords.items()}

    def __call__(self, sample: DS | DA):
        check_object_type(sample, allowed_types=(DS, DA), caller=self.__class__.__name__)
        return sample.assign_coords(self.coords)


class FillMissingValueTransform(BaseTransform):
    def __init__(self, strategy: str = "constant", fill_value: None | float = None,
                 method: None | str = "linear"):
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
    def __init__(self, dim: str, freq: str | int, method: str = "mean"):
        self.dim = dim
        self.freq = freq
        self.method = method

    def __call__(self, sample):
        # Implement resampling logic here
        check_object_type(sample, allowed_types=(DS, DA), caller=self.__class__.__name__)
        if self.method not in ["mean", "sum", "max", "min", "median"]:
            raise ValueError(f"Unsupported resampling method: {self.method}")

        # Choose resample if we have a time dimension, otherwise use coarsen for spatial dimensions
        if self.dim == "time":
            resampled = sample.resample({self.dim: self.freq})
        else:
            resampled = sample.coarsen({self.dim: self.freq}, boundary="trim")

        if not hasattr(resampled, self.method):
            raise ValueError(f"Invalid resample method: {self.method}")

        result = getattr(resampled, self.method)()
        return result


class ReshapeTransform(BaseTransform):
    def __init__(self, shape: tuple):
        self.shape = shape

    def __call__(self, sample):
        # Implement reshaping logic here
        check_object_type(sample, allowed_types=DA, caller=self.__class__.__name__)
        return sample.to_numpy().reshape(self.shape)


class RollTransform(BaseTransform):
    def __init__(self, dim: str, shift: None | int):
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


class SortAxisTransform(BaseTransform):
    def __init__(self, dim: str, ascending: bool = True):
        self.dim = dim
        self.ascending = ascending

    def __call__(self, sample):
        # Implement axis sorting logic here
        check_object_type(sample, allowed_types=DS, caller=self.__class__.__name__)
        sorted_sample = sample.sortby(self.dim, ascending=self.ascending)
        return sorted_sample


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
            # If subset selectors exist, then apply, but accept subsetting over some
            # variables that may have reduced dimensions (e.g. no time dimension).
            # E.g. "lon" and "lat" may be 2D coordinate variables that do not have a time dimension,
            # so we should allow for subsetting over the time dimension for the main variable(s) of
            # interest, but still subset in space for time-invariant variables.
            if self.subset_selectors:
                subset_selectors = self.subset_selectors.copy()

                # Prepare a subset dictionary per variable by checking available dimensions.
                var_dims = sample[var_id].dims
                for dim in self.subset_selectors:
                    if dim not in var_dims:
                        subset_selectors.pop(dim)

                ds[var_id] = sample[var_id].sel(**subset_selectors)
            else:
                # If no subset selectors, just copy the variable over as is.
                ds[var_id] = sample[var_id]

        return ds


class SqueezeTransform(BaseTransform):
    def __call__(self, sample):
        # Implement squeezing logic here
        check_object_type(sample, allowed_types=(DS, DA, TT), caller=self.__class__.__name__)
        return sample.squeeze()


class TilerTransform(BaseTransform):
    """
    A transform that takes a Dataset or DataArray and breaks it into smaller tiles along specified dimensions.
    This uses the xarray `coarsen` + `construct` pattern to create non-overlapping tiles of the data, which can
    be useful for training models on large spatial datasets by reducing memory usage and allowing for batch
    processing of smaller chunks of data.
    """
    def __init__(
        self,
        boundary: str = "pad",
        validate_axis_order: bool = False,
        discontinuity_periods: dict[str, float] | None = None,
        **dim_tile_sizes,
    ):
        self.boundary = boundary
        self.validate_axis_order = validate_axis_order
        self.discontinuity_periods = discontinuity_periods or {"longitude": 360.0, "lon": 360.0}
        self.tile_sizes = dim_tile_sizes

    def _validate_axis_order(self, sample: DA) -> None:
        for dim in self.tile_sizes:
            if dim not in sample.coords:
                continue
            coords = np.asarray(sample.coords[dim].values)
            if coords.ndim != 1:
                continue
            if coords.size < 2:
                continue

            diffs = np.diff(coords)
            if np.issubdtype(diffs.dtype, np.timedelta64):
                ok = np.all(diffs > np.timedelta64(0, "ns"))
            else:
                ok = np.all(diffs > 0)

            if not ok:
                raise ValueError(
                    f"Axis '{dim}' is not strictly ascending. "
                    "Either sort/reverse this axis before tiling, or set "
                    "validate_axis_order=False to bypass this guardrail."
                )

    def _validate_no_discontinuity_crossing(self, coarsened: DA, tile_dims: dict[str, tuple[str, str]]) -> None:
        for dim, period in self.discontinuity_periods.items():
            if dim not in tile_dims:
                continue
            coarse_dim, fine_dim = tile_dims[dim]
            if dim in coarsened.coords:
                coord = coarsened[dim]
            elif fine_dim in coarsened.coords:
                coord = coarsened[fine_dim]
            else:
                continue

            if coarse_dim not in coord.dims or fine_dim not in coord.dims:
                continue

            values = np.asarray(coord.transpose(coarse_dim, fine_dim).values)
            if values.ndim != 2 or values.shape[1] < 2:
                continue
            if np.issubdtype(values.dtype, np.datetime64):
                continue

            diffs = np.abs(np.diff(values.astype(np.float64), axis=1))
            crossing = diffs > (period / 2.0)
            if np.any(crossing):
                bad_tiles = np.where(crossing.any(axis=1))[0].tolist()
                raise ValueError(
                    f"Detected tiler discontinuity crossing on axis '{dim}' "
                    f"(period={period}). Affected coarse tile ids: {bad_tiles[:10]}"
                )

    def __call__(self, sample: DA) -> DA:
        check_object_type(sample, allowed_types=DA, caller=self.__class__.__name__)

        if self.validate_axis_order:
            self._validate_axis_order(sample)

        # Create the dictionary to send to the ".construct()" method, using a naming convention of
        # ("{dim}_coarse", "{dim}") for the new dimensions created by the tiling process.
        tile_dims = {dim: (f"{dim}_coarse", f"{dim}") for dim in self.tile_sizes}
        coarse_dims = {dim: f"{dim}_coarse" for dim in self.tile_sizes}
        fine_dims = {dim: f"{dim}" for dim in self.tile_sizes}
        coarsened = sample.coarsen(**self.tile_sizes, boundary=self.boundary).construct(**tile_dims)  # type: ignore

        self._validate_no_discontinuity_crossing(coarsened, tile_dims)

        # Prepare a stacking regrouping of the original dimensions and the new dimensions
        batch_dims = []
        target_dims = []
        for dim in sample.dims:
            if dim in self.tile_sizes:
                batch_dims.append(f"{dim}_coarse")
                target_dims.append(dim)
            else:
                target_dims.append(dim)

        stacked = coarsened.stack(batch_dim=batch_dims)
        # Reorder to have batch_dim first, followed by the original dimensions and then the fine tile dimensions
        tiled = stacked.transpose("batch_dim", *target_dims)

        # Store reverse-lookup metadata in attrs
        tiled.attrs.update({
            "tiler_tile_sizes": self.tile_sizes,
            "tiler_boundary": self.boundary,
            "tiler_validate_axis_order": self.validate_axis_order,
            "tiler_discontinuity_periods": self.discontinuity_periods,
            "tiler_original_sizes": {dim: sample.sizes[dim] for dim in self.tile_sizes},
            "tiler_original_coords": {
                dim: sample.coords[dim].values.tolist()
                for dim in self.tile_sizes if dim in sample.coords
                },
            "tiler_coarse_dims": coarse_dims,
            "tiler_fine_dims": fine_dims,
            "tiler_batch_dims": [coarse_dims[dim] for dim in sample.dims if dim in self.tile_sizes],
        })
        return tiled


def _as_tiler_dict(value: dict | None, field_name: str) -> dict:
    if value is None:
        raise ValueError(f"Missing required tiler metadata field: '{field_name}'.")
    if not isinstance(value, dict):
        raise TypeError(f"Expected '{field_name}' metadata to be a dictionary, got: {type(value)}")
    return value


def _resolve_coord_dims(tiles: DA, coord_dims: list[str] | None = None) -> list[str]:
    tile_sizes = _as_tiler_dict(tiles.attrs.get("tiler_tile_sizes"), "tiler_tile_sizes")
    if coord_dims is None:
        return list(tile_sizes.keys())
    for dim in coord_dims:
        if dim not in tile_sizes:
            raise ValueError(f"Requested coordinate dim '{dim}' is not tiled. Available: {list(tile_sizes.keys())}")
    return coord_dims


def tiled_to_pixel_coordinates(
    tiles: DA,
    coord_dims: list[str] | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert a tiled DataArray into a per-pixel coordinate tensor.

    Returns:
        torch.Tensor with shape (N, D, *fine_dims)
        where:
          N = number of tiles (batch_dim)
          D = number of coordinate channels
    """
    check_object_type(tiles, allowed_types=DA, caller="tiled_to_pixel_coordinates")
    dims = _resolve_coord_dims(tiles, coord_dims)
    fine_dims = _as_tiler_dict(tiles.attrs.get("tiler_fine_dims"), "tiler_fine_dims")

    coord_arrays: list[xr.DataArray] = []
    for dim in dims:
        if dim in tiles.coords:
            coord = tiles[dim]
        else:
            fine_dim = fine_dims[dim]
            if fine_dim not in tiles.coords:
                raise ValueError(
                    f"Could not find coordinate '{dim}' or fallback coordinate '{fine_dim}' in tiled array."
                )
            coord = tiles[fine_dim]

        if "batch_dim" not in coord.dims:
            coord = coord.expand_dims(batch_dim=tiles.sizes["batch_dim"])
        coord_arrays.append(coord)

    broadcasted = xr.broadcast(*coord_arrays)
    ordered_fine_dims = [fine_dims[d] for d in dims]

    tensors: list[torch.Tensor] = []
    for arr in broadcasted:
        arr_t = arr.transpose("batch_dim", *ordered_fine_dims)
        values = np.asarray(arr_t.values)
        if np.issubdtype(values.dtype, np.datetime64):
            values = values.astype("datetime64[s]").astype("int64")
        tensors.append(torch.tensor(values, dtype=dtype))

    return torch.stack(tensors, dim=1)


def tiled_to_coordinate_bounds(
    tiles: DA,
    coord_dims: list[str] | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert tiled coordinates into per-tile coordinate bounds.

    Returns:
        torch.Tensor with shape (N, D, 2), where each bound is [min, max].
    """
    pixel_coords = tiled_to_pixel_coordinates(tiles=tiles, coord_dims=coord_dims, dtype=dtype)
    reduce_dims = tuple(range(2, pixel_coords.ndim))
    mins = pixel_coords.amin(dim=reduce_dims)
    maxs = pixel_coords.amax(dim=reduce_dims)
    return torch.stack([mins, maxs], dim=-1)


@dataclass
class TiledIndexMapper:
    """
    Reverse-lookup helper for tiled DataArray outputs created by TilerTransform.

    It maps physical coordinates (e.g. time/lat/lon values) to a tile id in batch_dim,
    and supports the inverse mapping from tile id to coarse tile indices.
    """

    tile_sizes: dict[str, int]
    original_sizes: dict[str, int]
    original_coords: dict[str, list]
    batch_dims: list[str]

    @classmethod
    def from_tiled_array(cls, tiles: DA) -> "TiledIndexMapper":
        check_object_type(tiles, allowed_types=DA, caller="TiledIndexMapper.from_tiled_array")
        tile_sizes = _as_tiler_dict(tiles.attrs.get("tiler_tile_sizes"), "tiler_tile_sizes")
        original_sizes = _as_tiler_dict(tiles.attrs.get("tiler_original_sizes"), "tiler_original_sizes")
        original_coords = _as_tiler_dict(tiles.attrs.get("tiler_original_coords"), "tiler_original_coords")
        batch_dims = tiles.attrs.get("tiler_batch_dims")
        if not isinstance(batch_dims, list):
            raise ValueError("Missing required tiler metadata field: 'tiler_batch_dims'.")
        return cls(
            tile_sizes={str(k): int(v) for k, v in tile_sizes.items()},
            original_sizes={str(k): int(v) for k, v in original_sizes.items()},
            original_coords={str(k): list(v) for k, v in original_coords.items()},
            batch_dims=[str(d) for d in batch_dims],
        )

    def _original_dim_from_batch_dim(self, batch_dim: str) -> str:
        if batch_dim.endswith("_coarse"):
            return batch_dim[:-7]
        raise ValueError(f"Unexpected batch dim '{batch_dim}'. Expected suffix '_coarse'.")

    def _n_coarse_for_dim(self, dim: str) -> int:
        return int(math.ceil(self.original_sizes[dim] / self.tile_sizes[dim]))

    def _coarse_index_from_coord(self, dim: str, coord_value: float | int | np.datetime64) -> int:
        coords = np.asarray(self.original_coords[dim])
        if np.issubdtype(coords.dtype, np.datetime64):
            target = np.datetime64(coord_value)
            abs_diff = np.abs(coords - target)
            idx = int(abs_diff.argmin())
        else:
            target = float(coord_value)
            abs_diff = np.abs(coords.astype(np.float64) - target)
            idx = int(abs_diff.argmin())

        coarse_idx = idx // self.tile_sizes[dim]
        max_idx = self._n_coarse_for_dim(dim) - 1
        return int(min(max(coarse_idx, 0), max_idx))

    def tile_id_from_coordinates(self, **coords: float | int | np.datetime64) -> int:
        """
        Map real-world coordinates to a tile id in batch_dim.

        Example:
            mapper.tile_id_from_coordinates(time=np.datetime64("2005-01-01T01"), latitude=48.0, longitude=-3.0)
        """
        coarse_indices: list[int] = []
        n_coarses: list[int] = []

        for batch_dim in self.batch_dims:
            dim = self._original_dim_from_batch_dim(batch_dim)
            if dim not in coords:
                raise ValueError(f"Missing coordinate for '{dim}'. Provided keys: {list(coords)}")
            coarse_indices.append(self._coarse_index_from_coord(dim, coords[dim]))
            n_coarses.append(self._n_coarse_for_dim(dim))

        tile_id = 0
        for coarse_idx, n_coarse in zip(coarse_indices, n_coarses):
            tile_id = tile_id * n_coarse + coarse_idx
        return int(tile_id)

    def coordinates_from_tile_id(self, tile_id: int) -> dict[str, int]:
        """
        Inverse mapping from tile id to coarse tile indices by original dimension.
        """
        n_total = 1
        n_coarses: list[int] = []
        dims: list[str] = []
        for batch_dim in self.batch_dims:
            dim = self._original_dim_from_batch_dim(batch_dim)
            dims.append(dim)
            n_coarse = self._n_coarse_for_dim(dim)
            n_coarses.append(n_coarse)
            n_total *= n_coarse

        if tile_id < 0 or tile_id >= n_total:
            raise ValueError(f"tile_id {tile_id} is out of range [0, {n_total - 1}]")

        indices_reversed: list[int] = []
        remaining = int(tile_id)
        for n_coarse in reversed(n_coarses):
            indices_reversed.append(remaining % n_coarse)
            remaining //= n_coarse

        coarse_indices = list(reversed(indices_reversed))
        return {dim: idx for dim, idx in zip(dims, coarse_indices)}


class ToDataArray(BaseTransform):
    def __init__(self, var_id: str):
        self.var_id = var_id

    def __call__(self, sample: DS | DA) -> DA:
        # Implement conversion to xarray DataArray here
        check_object_type(sample, allowed_types=(DS, DA), caller=self.__class__.__name__)

        if isinstance(sample, DS):
            if len(sample.data_vars) != 1:
                raise ValueError("ToDataArrayTransform can only be applied to Datasets with a single variable.")
            return sample[self.var_id]
        return sample


class ToTensorTransform(BaseTransform):
    def __call__(self, sample: DA | np.ndarray) -> torch.Tensor:
        # Implement conversion to PyTorch tensor here
        check_object_type(sample, allowed_types=(DA, np.ndarray), caller=self.__class__.__name__)
        if isinstance(sample, DA):
            sample = sample.values

        return torch.from_numpy(sample)


def datetime_coords_to_float(da: DA) -> DA:
    datetime_coords = {
        name: (coord.dims, coord.values.astype("datetime64[ns]").astype("float64"))
        for name, coord in da.coords.items()
        if pd.api.types.is_datetime64_any_dtype(coord)
    }
    return da.assign_coords(datetime_coords)


class ToValuesBoundsTransform(BaseTransform):
    def __init__(self, dims):
        self.dims = dims

    def __call__(self, sample: DA) -> tuple[TT, TT]:
        sample = datetime_coords_to_float(sample)
        pixel_halfwidths = [
            (sample[dim][1].values - sample[dim][0].values) / 2 if sample[dim].size > 1 else None
            for dim in self.dims
            ]
        bounds = np.array([
            [sample[dim][0].values - halfwidth, sample[dim][-1].values + halfwidth]
            if sample[dim].ndim > 0 else [sample[dim].values, sample[dim].values]
            for dim, halfwidth in zip(self.dims, pixel_halfwidths)
            ])
        return torch.from_numpy(sample.values), torch.from_numpy(bounds)


class ToValuesLocationsTransform(BaseTransform):
    def __init__(self,
                 dims: list[str],
                 crs_conversion_spec: CRS_conversion_spec | tuple | list | None = None):
        self.dims = dims
        self.crs_conversion = (
            None if crs_conversion_spec is None else CRS_convertor(crs_conversion_spec)
            )

    def __call__(self, sample: DA) -> tuple[TT, TT]:
        if self.crs_conversion is not None:
            sample = self.crs_conversion.add_converted_coords(sample, self.dims)
        sample = datetime_coords_to_float(sample)
        coord_array = xr.broadcast(*[sample[dim] for dim in self.dims])
        locations = torch.stack(
            [torch.tensor(coords.values, dtype=torch.float32) for coords in coord_array],
            dim=0
            )
        return torch.from_numpy(sample.values), locations


class TransposeTransform(BaseTransform):
    def __call__(self, sample):
        # Implement transposing logic here
        check_object_type(sample, allowed_types=(DA, TT), caller=self.__class__.__name__)
        return sample.transpose()


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

    def __init__(self, variables: list, new_dim: str, only_vars_with_time: bool = True):
        self.variables = variables
        self.new_dim = new_dim
        self.only_vars_with_time = only_vars_with_time

    def __call__(self, sample):
        # Implement logic to convert variables to a new dimension here
        check_object_type(sample, allowed_types=DS, caller=self.__class__.__name__)

        # Check special case of variables = "__all__", take all variables and filter out those not needed/suitable
        if self.variables == "__all__":

            # Exclude variables relate to bounds and coordinates
            bounds_vars = set([b_list[0] for b_list in sample.cf.bounds.values()])

            if self.only_vars_with_time:
                vars_without_time = set([var_id for var_id in sample.data_vars
                                        if not hasattr(sample[var_id], "time")])
            else:
                vars_without_time = set()

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
    "add_fixed_coordinates": AddFixedCoordinates,
    "fill_missing": FillMissingValueTransform,
    "fill_nan": FillNaNTransform,
    "normalize": NormalizeTransform,
    "rename": RenameTransform,
    "resample": ResampleTransform,
    "reshape": ReshapeTransform,
    "reverse_axis": ReverseAxisTransform,
    "roll": RollTransform,
    "scale": ScaleTransform,
    "sort_axis": SortAxisTransform,
    "squeeze": SqueezeTransform,
    "subset": SubsetTransform,
    "tiler": TilerTransform,
    "to_dataarray": ToDataArray,
    "to_tensor": ToTensorTransform,
    "to_values_bounds_tensors": ToValuesBoundsTransform,
    "to_values_locations_tensors": ToValuesLocationsTransform,
    "transpose": TransposeTransform,
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


def apply_transforms(data: xr.Dataset | xr.DataArray, preprocessors: list) -> xr.Dataset | xr.DataArray:
    """
    Apply a list of preprocessing transforms to a data sample.
    Args:
        sample (xr.Dataset | xr.DataArray): The input data sample to be transformed.
        preprocessors (list): A list of transform configurations to apply to the sample.
    Returns:
        xr.Dataset | xr.DataArray: The transformed data sample after applying all preprocessors.
    """
    for preprocessor in preprocessors:
        if not isinstance(preprocessor, dict) or "type" not in preprocessor:
            raise ValueError(f"Each preprocessor must be a dictionary with a 'type' key. Invalid preprocessor: {preprocessor}")
        data = resolve_transform(preprocessor)(data)

    return data


# Create `apply_preprocessors` as an alias for `apply_transforms` to allow for more intuitive naming when used
# in the context of preprocessing steps.
apply_preprocessors = apply_transforms
