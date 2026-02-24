import geopandas as gpd
import numpy as np
import xarray as xr
from rasterio.features import rasterize
from affine import Affine
from sklearn.preprocessing import OrdinalEncoder
import yaml

class Shapefiletoxarray:
    def __init__(self, config_path):
        """
        Parameters
        ----------
        resolution : float
            Target grid resolution in coordinate units.
        """
        self.category_mappings = {}   # Stores category→integer mappings for each shapefile/column

        #Initialise from confing.
        cfg_in = self.load_yaml_ordered(config_path)
        self.build_inputs_from_config(cfg_in)


    # ----------------------------------------------------------
    # 1. LOAD SHAPEFILES
    # ----------------------------------------------------------
    def proc_shapefiles(self, file_list, parent_grd, categorical_columns):
        # """Load a list of shapefiles into GeoDataFrames."""
        # self.gdfs = [gpd.read_file(f) for f in file_list]
        self.gdfs = {}
  
        # Loop over each file and convert caterogical columns as needed.
        for file_path  in file_list:
            gdf = gpd.read_file(file_path)

            if file_path in categorical_columns:
                cols_convert = categorical_columns[file_path]
                if cols_convert:
                    gdf = self.encode_categories(gdf, cols_convert, file_path)

            #self.gdfs.append(gdf)
            self.gdfs[file_path] = gdf

            # Define bounds of parent grid based on chosen shapefile.
            if file_path == parent_grd:
                self.parent_bounds = gdf.total_bounds

    # ----------------------------------------------------------
    # 2. BUILD A COMMON PARENT GRID
    # ----------------------------------------------------------
    def build_parent_grid(self):
        xmin, ymin, xmax, ymax = self.parent_bounds
        res = self.resolution

        self.nx = int(np.ceil((xmax - xmin) / res)) + 1
        self.ny = int(np.ceil((ymax - ymin) / res)) + 1

        # Affine transform for rasterisation
        self.transform = Affine.translation(xmin, ymax) * Affine.scale(res, -res)

        # Coordinates
        self.x = np.arange(xmin, xmin + (self.nx * res), res)
        self.y = np.flip(np.arange(ymin, ymin + (self.ny * res), res))  # flipped for raster orientation


    # ----------------------------------------------------------
    # 3. ENCODE CATEGORICAL COLUMNS - INTERNAL ONLY.
    # ----------------------------------------------------------
    def encode_categories(self, gdf, categorical_columns, file_path):
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


    # ----------------------------------------------------------
    # 4. RASTERISE A SINGLE VARIABLE FROM A GDF
    # ----------------------------------------------------------
    def rasterise(self, gdf, column):
        shapes = [(geom, value) 
                  for geom, value in zip(gdf.geometry, gdf[column])]
        
        arr = rasterize(
            shapes,
            out_shape=(self.ny, self.nx),
            transform=self.transform,
            fill=0,
            dtype="float32"
        )
        return arr


    # ----------------------------------------------------------
    # 5. CREATE THE FINAL XARRAY DATASET
    # ----------------------------------------------------------
    def to_xarray(self, variable_map):
        
        data_vars = {}

        for curr_gdf in variable_map:

            # Print statement to show what is currently being processed.
            print(f"Processing file: {curr_gdf}")

            #Check if we are writing output for the current file.
            if variable_map[curr_gdf]:

                for col in variable_map[curr_gdf]:

                   #Print statement to show which variable in the current file is being processed.
                   print(f"Processing variable: {col}")

                   arr = self.rasterise(self.gdfs[curr_gdf], col)

                   data_vars[col] = (("y", "x"), arr)

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={"x": self.x, "y": self.y},
            attrs={"resolution": self.resolution}
        )

        self.dataset_out = ds

    # Code to read in and decode the config yaml file.
    def load_yaml_ordered(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    # Extract the correct file lists from the config file.
    def build_inputs_from_config(self, cfg):

        
        # --- Required fields ---
        if "resolution" not in cfg:
            raise ValueError("config must define top-level 'resolution'.")
        if "sources" not in cfg or not cfg["sources"]:
            raise ValueError("config must define 'sources' with at least one entry.")

        # Get the sources and resolution from the config.
        self.resolution = cfg["resolution"]
        sources = cfg["sources"]

        # Populate the file_list, categorical columns and the variable map.
        self.file_list    = []
        self.cat_cols_map = {}
        self.var_out_map  = {}
        self.parent_grd   = []

        #Internal only variable to store list of parent grids to check that only one is defined.
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
                raise ValueError(f"'categorical_columns' for '{src_name}' must be a list or null.")

            # Finally build the variable map.
            var_cols = s.get("variables", None) 
            # normalize empty list to [] and null to None
            if var_cols is None:
                self.var_out_map[file_path] = None
            elif isinstance(var_cols, list):
                self.var_out_map[file_path] = var_cols
            else:
                raise ValueError(f"'variables' for '{src_name}' must be a list or null.")

        # Final check to ensure only one parent grid is defined and define that grid.
        if all(x == "NO" for x in parent_grd_list):
            raise ValueError(f"No parent grid defined. Please correct config.")
        elif  parent_grd_list.count("YES") > 1:
            raise ValueError(f"More the one parent grid defined. Please correct config")
        else:
            self.parent_grd = self.file_list[parent_grd_list.index("YES")]

    # Final wrapper function to run the whole process.
    def build_dataset(self):

        # Execute the stpes to build the dataset from the shapefiles.
        # Read the shapefiles.
        self.proc_shapefiles(self.file_list, parent_grd=self.parent_grd, categorical_columns=self.cat_cols_map)

        # Build the parent grid.
        self.build_parent_grid()

        #Create the xarray dataset.
        ds = self.to_xarray(self.var_out_map)



#Set the short names for files for reference.
# shp_sname = ['Soil_parent','Soil_Carbon','Soil_prop']

# #Set a file list.
# shp_files = ['/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/soil_parent_material_1km/data/SPMM_1km/SoilParentMateriall_V1_portal1km.shp',
#              '/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/model_estimates_of_topsoil_carbon/data/CS_topsoil_carbon.shp',
#              '/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/model_estimates_of_topsoil_pH_and_bulk_density/data/CS_topsoil_pH_bulkDensity.shp']


# #Set up some test categorical columns to convery to integers.
# all_cat_cols = { '/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/soil_parent_material_1km/data/SPMM_1km/SoilParentMateriall_V1_portal1km.shp': ['ESB_DESC','CARB_CNTNT','PMM_GRAIN','SOIL_GROUP','SOIL_TEX','SOIL_DEPTH'],
#                  '/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/model_estimates_of_topsoil_carbon/data/CS_topsoil_carbon.shp': None,
#                  '/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/model_estimates_of_topsoil_pH_and_bulk_density/data/CS_topsoil_pH_bulkDensity.shp': None
# }

# #Create the variable map to produce the final output.
# var_out_map = { '/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/soil_parent_material_1km/data/SPMM_1km/SoilParentMateriall_V1_portal1km.shp': ['ESB_DESC','CARB_CNTNT','PMM_GRAIN','SOIL_GROUP','SOIL_TEX','SOIL_DEPTH'],
#                 '/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/model_estimates_of_topsoil_carbon/data/CS_topsoil_carbon.shp': ['CCONC_07'],
#                 '/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/model_estimates_of_topsoil_pH_and_bulk_density/data/CS_topsoil_pH_bulkDensity.shp': ['BULKD_07']
# }


# #Move the following lines into one final wrapper function that will run the whole process.
# #Ideally try and get the above dictionaries to be read in as a yaml/config file?

# #Create the class.
# r = Shapefiletoxarray(resolution=1000.0)

# #Read the shapefiles.
# r.proc_shapefiles(shp_files, parent_grd=shp_files[0], categorical_columns=all_cat_cols)

# #Build the parent grid.
# r.build_parent_grid()

# #Create the xarray dataset.
# ds = r.to_xarray(var_out_map)

# print(ds)

#print(r.gdfs['/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/soil_parent_material_1km/data/SPMM_1km/SoilParentMateriall_V1_portal1km.shp'].head())
# for test in var_out_map:
#     print(var_out_map[test])

r = Shapefiletoxarray('config.yaml')

# print(r.resolution)
# print(r.file_list)
# print(r.cat_cols_map)
# print(r.var_out_map)
# print(r.parent_grd)

# Build the dataset.
r.build_dataset()


print(r.dataset_out)