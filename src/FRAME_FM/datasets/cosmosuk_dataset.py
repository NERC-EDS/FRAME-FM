# SPDX-FileCopyrightText: 2026 FRAME-FM Contributors
#
# SPDX-License-Identifier: Apache-2.0

import xarray as xr
import pandas as pd
import glob
import os
import numpy as np
from pathlib import Path
from FRAME_FM.datasets.base_dataset import BaseDataset


class CosmosUKDataset(BaseDataset):
    """
    Loads the CosmosUK Dataset

    This expects a directory with a series of files named cosmos-uk_?????_hydrosoil_sh_????-????.csv
    (where ????? is a 5 characeter site ID, the two ???? are years) plus
    QC files cosmos-uk_?????_hydrosoil_sh_????-????_qc_flags.csv and cosmos-uk_?????_hydrosoil_sh_????-????_flags.csv.

    In the parent directory to these there should be a file cosmos-uk_sitemetadata_2013-2024.csv which lists
    the sites and their precise locations.

    Descriptions of the data format are in cosmos-uk_supportinginformation_2013-2024.docx
    from https://data-package.ceh.ac.uk/sd/2dce161d-2fab-47bb-9fe6-38e7ed1ae18a.zip
    """

    _transforms = []

    def __init__(
        self,
        data_uri: str | Path | list | tuple,
        preprocessors: list | None = None,
        transforms: list | None = None,
        chunks: dict | None = None,
        override_transforms: bool = False,
        cache_dir: None | Path | str = None,
        generate_stats: bool = True,
        force_recache: bool = False,
        qc_bitmask: int = 0b11111111111,
        drop_qc_flags: list = ["M", "U", "I", "E"],
    ):
        # Save the QC content so it can be used in the _setup_dataset method to filter the data as it's loaded.
        # This is necessary because the QC flags are stored in separate files and need to be applied at load time.
        self.qc_bitmask = qc_bitmask
        self.drop_qc_flags = drop_qc_flags

        super().__init__(
            data_uri=data_uri,
            preprocessors=preprocessors,
            transforms=transforms,
            chunks=chunks,
            override_transforms=override_transforms,
            cache_dir=cache_dir,
            generate_stats=generate_stats,
            force_recache=force_recache,
        )

    def _setup_dataset(self):
        self.data = self._csv_to_xarray(self.data_uri, self.qc_bitmask, self.drop_qc_flags)

    def __len__(self):
        return len(self.data)

    def _process_bitmask(self, df, qc_df, qc_bitmask: int):
        """
        Drops data based on the QC flags and specfied bitmask.
        Any data failing QC will be converted to a NaN.

        df is a pandas dataframe containing the data
        qc_df is a pandas dataframe containing the QC flags (0-1024)
        qc_bitmask is a 11-bit bit mask that selects which flags to drop, a bit set to 1 means drop, 0 means keep,

        Returns a pandas dataframe with all remaining data. Any dropped data is turned into a NaN.
        """
        # handle the bitmask flags
        # first build a boolean mask
        # skip first two columns (data and site name) as they are non integer/float and not subject to QC flags
        for column in qc_df.columns[2:]:
            # ensure any NaNs are zero or they'll fail the integer conversion
            qc_df[column] = qc_df[column].fillna(0)
            qc_df[column] = qc_df[column].astype(int)
            # print("QC mask ", column)
            # print(qc_df[column])
            # applies a bitwise AND to every row in the column against our mask of acceptable values
            qc_df[column] = np.bitwise_and(qc_df[column].astype("int16"), qc_bitmask)
            # print("QC mask anded")
            # print(qc_df[column])
            # convert all values to booleans, by default anything non-zero becomes true, but non-zero means we want to drop it
            # so invert the output, qc_df now contains true for all entries which passed QC and false for entries which failed it
            qc_df[column] = ~qc_df[column].astype("bool")
            # print("QC inverted and turned into a boolean")
            # print(qc_df[column])

        # apply the boolean mask to the data
        # anything with a true in the mask will stay, anything with a false turns into a NaN
        for column in df.columns[2:]:
            df[column] = df[column].where(qc_df[column + "_QCFLAG"])
            # print("Final result for column", column)
            # print(df[column])
        return df

    def _process_flags(self, df, flags_df, drop_qc_flags: list):
        """
        converts items which have QC flags set to NaNs
        df is a pandas dataframe containing the data
        flags_df is a pandas dataframe containing the QC flags
        drop_qc_flags is a list of flags to drop, possible QC flags: M=missing, U=unchecked, I=infilled, E=estimated
        returns a pandas dataframe with the dropped data converted to NaNs
        """

        # convert QC array to a mask
        # turn all entries to a true where there are no flags/no flags we want to drop and a false where there's a flag we want to drop
        for column in flags_df.columns[2:]:
            # print("Processing flags for column", column, "before masking:")
            # print(flags_df[column])
            for flag in drop_qc_flags:
                # flags which were empty are already NaNs, these will turn into False when we run notna()
                # make flags into Trues, notna() will keep them as true, but if invert it's response we'll get what we want
                # e.g. flags = false, no flag = true
                flags_df[column] = flags_df[column].where(flags_df[column] != flag, "False")
            # print("after masking:\n",flags_df[column])
            # columns are either NaN (no flags), false (flagged to drop) or a value (flag which we are ignoring)
            # convert anything that's not false to true
            flags_df[column] = flags_df[column].where(flags_df[column] == "False", "True")
            # print("before inversion:\n",flags_df[column])
            flags_df[column] = flags_df[column].where(flags_df[column] == "True", 0).astype("bool")
            # flags_df[column] = flags_df[column].astype("bool")
            # print("after inversion")
            # print(flags_df[column])
        # flags_df should now be a boolean mask

        # convert flagged columns to NaNs
        for column in df.columns[2:]:
            # print("masking column",column)
            # print("value before mask",df[column])
            df[column] = df[column].where(flags_df[column + "_FLAG"])
            # print("value after mask", df[column])
        return df

    def _csv_to_xarray(self, data_path: str, qc_bitmask: int, drop_qc_flags: list):
        """
        Loads the cosmos UK CSV data and converts it into an Xarray dataset

        data_path is the directory inside base path where the data is stored. Do not specify file names, the code automatically picks up files with the correct names (cosmos-uk_*_hydrosoil_sh_????-????.csv). It is assumed that a metadata file called cosmos-uk_sitemetadata_2013-2024.csv exists in the parent directory
        of this. This will contain the station locations.

        qc_bitmask is the 11-bit mask of which QC bit files we *WANT* to allow through, set to zero to mask nothing (e.g. accept all data)
        set to 0b111111111111 to drop all data with a QC flag
        possible QC bit fields: 0=passed, 1=missing, 2=zero data, 4=too few samples
        8=low power, 16=sensor fault, 32=diagnostic fault, 64=out of range,
        128=secondary variable, 256=midnight soil heat flux calibration, 512=spike
        1024=error code stored as value

        drop_qc_flags is a list of which flags to drop, possible values are M=missing, U=unchecked, I=infilled, E=estimated

        Returns a list containing xarray datasets with a single site each. These will have QC'ed values filtered out. Latitude/longitude (WGS84) and OS easting/northings (OSGB36) are added to each dataset as attributes.
        """

        files = glob.glob(data_path + "/cosmos-uk_?????_hydrosoil_sh_????-????.csv")

        metadata_df = pd.read_csv(data_path + "../cosmos-uk_sitemetadata_2013-2024.csv", index_col="SITE_ID")

        all_data = []

        # load each CSV into the all_data array
        for file in files:
            print("Loading file", file)

            # check QC files exist
            qc_file = file[:-4] + "_qc_flags.csv"
            flags_file = file[:-4] + "_flags.csv"

            if not os.path.isfile(qc_file):
                raise FileNotFoundError("QC file " + qc_file + " not found")

            if not os.path.isfile(flags_file):
                raise FileNotFoundError("QC Flags file " + flags_file + " not found")

            # missing values should be -9999 anyway and we turn them to NaNs at load time
            data_df = pd.read_csv(file, delimiter=",", parse_dates=["DATE_TIME"], na_values=[-9999])
            qc_df = pd.read_csv(qc_file, delimiter=",", parse_dates=["DATE_TIME"])
            flags_df = pd.read_csv(flags_file, delimiter=",", parse_dates=["DATE_TIME"], low_memory=False)

            # check the data and QC files match in shape
            assert data_df.shape == qc_df.shape == flags_df.shape, "Shapes of Data, QC and Flags files are not same."

            # check column names are the same and are in the same order
            qc_columns = [c.replace("_QCFLAG", "") for c in qc_df.columns]
            flags_columns = [c.replace("_FLAG", "") for c in flags_df.columns]
            assert data_df.columns.tolist() == qc_columns == flags_columns, (
                "Data, QC and Flags column names are not same."
            )

            # drop the data failing QC
            data_df = self._process_bitmask(data_df, qc_df, qc_bitmask)
            data_df = self._process_flags(data_df, flags_df, drop_qc_flags)

            # add lat/long
            station_id = data_df.SITE_ID[0]
            # ensure there's only one station ID used in the whole file
            assert data_df.SITE_ID.nunique() == 1
            latitude = metadata_df.LATITUDE[station_id]
            longitude = metadata_df.LONGITUDE[station_id]
            # data_df['LATITUDE'] = latitude
            # data_df['LONGITUDE'] = longitude

            # remove timezone from the datetime as xarray's to_netcdf doesn't like it
            data_df["DATE_TIME"] = pd.to_datetime(data_df.DATE_TIME).dt.tz_localize(None)
            # make DATE_TIME the index instead of using a index number
            data_df.set_index(["DATE_TIME"]).sort_index().to_xarray()

            # add to an xarray dataset
            # we're only interested in soil moisture here (TDTx_VWC)
            ds = xr.Dataset(
                {
                    "TDT1_VWC": (("TIME_DATE"), data_df["TDT1_VWC"]),
                    "TDT2_VWC": (("TIME_DATE"), data_df["TDT2_VWC"]),
                    "TDT3_VWC": (("TIME_DATE"), data_df["TDT3_VWC"]),
                    "TDT4_VWC": (("TIME_DATE"), data_df["TDT4_VWC"]),
                    "TDT5_VWC": (("TIME_DATE"), data_df["TDT5_VWC"]),
                    "TDT6_VWC": (("TIME_DATE"), data_df["TDT6_VWC"]),
                    "TDT7_VWC": (("TIME_DATE"), data_df["TDT7_VWC"]),
                    "TDT8_VWC": (("TIME_DATE"), data_df["TDT8_VWC"]),
                },
                attrs={
                    "easting": metadata_df.EASTING[station_id],
                    "northing": metadata_df.NORTHING[station_id],
                    "latitude": latitude,
                    "longitude": longitude,
                    "site_id": station_id,
                },
                coords={"time": data_df["DATE_TIME"]},
            )

            all_data.append(ds)
            # should site ID form part of the index?
            # sorts all the entries by date_time as not doing can break selecting on dates

        # convert to xarray
        # note that there can be multiple entries with the same timestamp so the index_col entries are not unique
        # should we make station ID part of the index?
        # ds = pd.concat(all_data).sort_index().to_xarray()

        # what should the final result look like?
        # soil moisture was important, how do we handle multiple soil moisiture variables

        return all_data
