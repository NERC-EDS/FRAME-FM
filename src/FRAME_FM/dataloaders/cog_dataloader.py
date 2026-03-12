import rioxarray
import glob

class CogHandler(self):

    def __init__(self):
        self.glob_string = '/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/soil_water_index_europe_1km_daily_v1/data/**/*.cog'

    def list_files(self, glob_string):
        file_list = glob.glob(self.glob_string,recursive=True)

    def load_files(self, file_list):

        for file in file_list:
            rds = rioxarray.open_rasterio(file, masked=True, overview_level=4)
            pause = 1

    def main(self):
        file_list = list(files)

        lemon = load_files(file_list)

    if __name__ == "__main__":
        main()