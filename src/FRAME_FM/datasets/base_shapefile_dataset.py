from pathlib import Path
import geopandas as gpd
import numpy as np
import xarray as xr
from rasterio.features import rasterize
from affine import Affine
from sklearn.preprocessing import OrdinalEncoder
import yaml
import torch
from torch.utils.data import Dataset
from FRAME_FM.utils.data_utils import unify_transforms
from FRAME_FM.transforms import resolve_transform


class BaseShapefileDataset(Dataset):
    _transforms = [{"type": "to_tensor"}]

    def __init__(
        self,
        data_uri: str | Path | list | tuple,
        transforms: list | None = None,
        override_transforms: bool = False,
    ):
        self.data_uri = data_uri
        self.transforms = unify_transforms(
            transforms, self._transforms, override_transforms
        )

        self.category_mappings = {}  # Stores category→integer mappings for each shapefile/column

        # Initialise from confing.
        cfg_in = self.load_yaml_ordered(self.data_uri)
        self.build_inputs_from_config(cfg_in)

        # Set up the class and build the dataset.
        self.build_dataset()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Returns the data sample at the specified index"""
        sample = self.data.isel(band=idx)

        # Apply runtime transforms if any
        for transform in self.transforms:
            sample = resolve_transform(transform)(sample)

        return sample  # type: ignore

    def proc_shapefiles(
        self,
        file_list: list[str],
        parent_grd: str,
        categorical_columns: dict[str, list],
    ):
        """Load a list of shapefiles into GeoDataFrames.
        file_list - the list of shapefiles to process
        parent_grd - the file to use as the parent grid who's boundaries will be used for all other files
        categorical_columns - a dict of columns to process
        """
        self.gdfs = {}

        # Loop over each file and convert caterogical columns as needed.
        for file_path in file_list:
            gdf = gpd.read_file(file_path)

            if file_path in categorical_columns:
                cols_convert = categorical_columns[file_path]
                if cols_convert:
                    gdf = self.encode_categories(gdf, cols_convert, file_path)

            # self.gdfs.append(gdf)
            self.gdfs[file_path] = gdf

            # Define bounds of parent grid based on chosen shapefile.
            if file_path == parent_grd:
                self.parent_bounds = gdf.total_bounds

    def build_parent_grid(self):
        """Builds a common parent grid and scale it to the target resolution"""

        xmin, ymin, xmax, ymax = self.parent_bounds
        res = self.resolution

        self.nx = int(np.ceil((xmax - xmin) / res)) + 1
        self.ny = int(np.ceil((ymax - ymin) / res)) + 1

        # Affine transform for rasterisation
        self.transform = Affine.translation(xmin, ymax) * Affine.scale(res, -res)

        # Coordinates
        self.x = np.arange(xmin, xmin + (self.nx * res), res)
        self.y = np.flip(
            np.arange(ymin, ymin + (self.ny * res), res)
        )  # flipped for raster orientation
        # self.y = np.arange(ymin, ymin + (self.ny * res), res)

    def encode_categories(self, gdf, categorical_columns: list[str], file_path: str):
        """Encodes the categorical columns
        gdf - a geopandas data file to encode
        categorical_columns - the list of categorical columns from the data file
        file_path - the path to the file we are processing
        returns a geopandas data file with the columns overwritten to the encoded values
        """

        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

        integer_values = enc.fit_transform(gdf[categorical_columns])
        mapping = {
            col: {cat: i for i, cat in enumerate(enc.categories_[j])}
            for j, col in enumerate(categorical_columns)
        }

        # Store mapping
        self.category_mappings[file_path] = mapping

        # Overwrite columns with encoded values
        for idx, col in enumerate(categorical_columns):
            gdf[col] = integer_values[:, idx]

        return gdf

    def rasterise(self, gdf, column: str):
        """Rasterises a single variable from a geopandas data file
        gdf - a geopandas data file
        column - a string specifying the column name
        returns a rasterised array of the shapefile for the specified column
        """
        shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf[column])]

        arr = rasterize(
            shapes,
            out_shape=(self.ny, self.nx),
            transform=self.transform,
            fill=0,
            dtype="float32",
        )
        return arr

    def to_xarray(self, variable_map):
        """Create the final xarray dataset"""
        data_vars = {}

        for curr_gdf in variable_map:
            # Print statement to show what is currently being processed.
            print(f"Processing file: {curr_gdf}")

            # Check if we are writing output for the current file.
            if variable_map[curr_gdf]:
                for col in variable_map[curr_gdf]:
                    # Print statement to show which variable in the current file is being processed.
                    print(f"Processing variable: {col}")

                    arr = self.rasterise(self.gdfs[curr_gdf], col)

                    data_vars[col] = (("y", "x"), arr)

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={"x": self.x, "y": self.y},
            attrs={"resolution": self.resolution, "crs": self.target_crs},
        )

        self.dataset_out = ds

    def load_yaml_ordered(self, path):
        """read in and decode the config yaml file."""
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    def build_inputs_from_config(self, cfg: str):
        """Extract the correct file lists from the config file.
        cfg - a string with the path to the configuration file
        """

        # --- Required fields ---
        if "resolution" not in cfg:
            raise ValueError("config must define top-level 'resolution'.")
        if "sources" not in cfg or not cfg["sources"]:
            raise ValueError("config must define 'sources' with at least one entry.")
        if "target_crs" not in cfg:
            raise ValueError("config must define top-level 'target_crs'.")

        # Get the sources and resolution from the config.
        self.resolution = cfg["resolution"]
        self.target_crs = cfg["target_crs"]
        sources = cfg["sources"]

        # Populate the file_list, categorical columns and the variable map.
        self.file_list = []
        self.cat_cols_map = {}
        self.var_out_map = {}
        self.parent_grd = []

        # Internal only variable to store list of parent grids to check that only one is defined.
        parent_grd_list = []

        for src_name, s in sources.items():
            # Get the files.
            file_path = s.get("file")
            if not file_path:
                raise ValueError(f"Source '{src_name}' is missing 'file'.")

            self.file_list.append(file_path)

            # Also extract the parent grid.
            par_grd = s.get("parent_grid")
            if not par_grd:
                raise ValueError(f"Source '{src_name}' is missing 'parent_grid'.")

            parent_grd_list.append(par_grd)

            # Now do the categorical columns.
            cat_cols = s.get("categorical_columns", None)
            # normalize empty list to [] and null to None
            if cat_cols is None:
                self.cat_cols_map[file_path] = None
            elif isinstance(cat_cols, list):
                self.cat_cols_map[file_path] = cat_cols
            else:
                raise ValueError(
                    f"'categorical_columns' for '{src_name}' must be a list or null."
                )

            # Finally build the variable map.
            var_cols = s.get("variables", None)
            # normalize empty list to [] and null to None
            if var_cols is None:
                self.var_out_map[file_path] = None
            elif isinstance(var_cols, list):
                self.var_out_map[file_path] = var_cols
            else:
                raise ValueError(
                    f"'variables' for '{src_name}' must be a list or null."
                )

        # Final check to ensure only one parent grid is defined and define that grid.
        if all(x == "NO" for x in parent_grd_list):
            raise ValueError(f"No parent grid defined. Please correct config.")
        elif parent_grd_list.count("YES") > 1:
            raise ValueError(f"More the one parent grid defined. Please correct config")
        else:
            self.parent_grd = self.file_list[parent_grd_list.index("YES")]

    def build_dataset(self):
        """Final wrapper function to run the whole process.
        Builds the xarray dataset from a set of shapefiles
        """

        # Execute the stpes to build the dataset from the shapefiles.
        # Read the shapefiles.
        self.proc_shapefiles(
            self.file_list,
            parent_grd=self.parent_grd,
            categorical_columns=self.cat_cols_map,
        )

        # Build the parent grid.
        self.build_parent_grid()

        # Create the xarray dataset.
        self.data = self.to_xarray(self.var_out_map)
