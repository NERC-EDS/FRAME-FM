import xarray as xr
import pandas as pd
import glob
import os
import numpy as np

# Descriptions of metadata are in cosmos-uk_supportinginformation_2013-2024.docx from https://data-package.ceh.ac.uk/sd/2dce161d-2fab-47bb-9fe6-38e7ed1ae18a.zip

def process_bitmask(df, qc_df, qc_bitmask: int):
    '''
    Drops data based on the QC flags and specfied bitmask.
    Any data failing QC will be converted to a NaN.
    
    df is a pandas dataframe containing the data
    qc_df is a pandas dataframe containing the QC flags (0-1024)
    qc_bitmask is a 11-bit bit mask that selects which flags to drop, a bit set to 1 means drop, 0 means keep, 

    Returns a pandas dataframe with all remaining data. Any dropped data is turned into a NaN.
    '''
    # handle the bitmask flags
    # first build a boolean mask
    # skip first two columns (data and site name) as they are non integer/float and not subject to QC flags
    for column in qc_df.columns[2:]:
        #ensure any NaNs are zero or they'll fail the integer conversion
        qc_df[column] = qc_df[column].fillna(0)
        qc_df[column] = qc_df[column].astype(int)
        #applies a bitwise AND to every row in the column against our mask of acceptable values
        qc_df[column] = np.bitwise_and(qc_df[column].astype("int16"), qc_bitmask)
        
        #convert all values to booleans, by default anything non-zero becomes true, but non-zero means we want to drop it
        #so invert the output, qc_df now contains true for all entries which passed QC and false for entries which failed it
        qc_df[column] = ~qc_df[column].astype("bool")

    #apply the boolean mask to the data
    #anything with a true in the mask will stay, anything with a false turns into a NaN
    for column in df.columns[2:]:
        df[column] = df[column].where(qc_df[column + "_QCFLAG"])

    return df

def process_flags(df, flags_df, drop_qc_flags: list):
    '''
    converts items which have QC flags set to NaNs
    df is a pandas dataframe containing the data
    flags_df is a pandas dataframe containing the QC flags
    drop_qc_flags is a list of flags to drop, possible QC flags: M=missing, U=unchecked, I=infilled, E=estimated
    returns a pandas dataframe with the dropped data converted to NaNs
    '''
    
    # convert QC array to a mask
    # turn all entries to a true where there are no flags/no flags we want to drop and a false where there's a flag we want to drop
    for column in flags_df.columns[2:]:
        
        for flag in drop_qc_flags:
            # flags which were empty are already NaNs, these will turn into False when we run notna()
            # make flags into Trues, notna() will keep them as true, but if invert it's response we'll get what we want
            # e.g. flags = false, no flag = true
            flags_df[column] = flags_df[column].where(flags_df[column] != flag, "False")
        flags_df[column] = ~flags_df[column].notna()
    
    # flags_df should now be a boolean mask
    
    # convert flagged columns to NaNs
    for column in df.columns[2:]:
        df[column] = df[column].where(flags_df[column + "_FLAG"])

    return df


def csv_to_xarray(data_path: str, qc_bitmask: int, drop_qc_flags: list):
    '''
    Loads the CSV 

    data_path is the directory inside base path where the data is stored. Do not specify file names, the code automatically picks up files with the correct names (cosmos-uk_*_hydrosoil_sh_????-????.csv). It is assumed that a metadata file called cosmos-uk_sitemetadata_2013-2024.csv exists in the parent directory 
    of this. This will contain the station locations. 
    
    qc_bitmask is the 11-bit mask of which QC bit files we *WANT* to allow through, set to zero to mask nothing (e.g. accept all data)
    set to 0b111111111111 to drop all data with a QC flag
    possible QC bit fields: 0=passed, 1=missing, 2=zero data, 4=too few samples
    8=low power, 16=sensor fault, 32=diagnostic fault, 64=out of range, 
    128=secondary variable, 256=midnight soil heat flux calibration, 512=spike
    1024=error code stored as value

    drop_qc_flags is a list of which flags to drop, possible values are M=missing, U=unchecked, I=infilled, E=estimated

    Returns an xarray dataset containing all the data with QC'ed values filtered out
    '''
    
    daily = xr.Dataset()
    files = glob.glob(data_path + "/cosmos-uk_*_hydrosoil_sh_????-????.csv")

    metadata_df = pd.read_csv(data_path + "../cosmos-uk_sitemetadata_2013-2024.csv", index_col="SITE_ID")

    all_data = []

    # load each CSV into the dfs array
    for file in files:
        print(file)

        # check QC files exist
        qc_file=file[:-4] + "_qc_flags.csv"
        flags_file=file[:-4] + "_flags.csv"
        
        if not os.path.isfile(qc_file):
            raise FileNotFoundError("QC file " + qc_file + " not found")
        
        if not os.path.isfile(flags_file):
            raise FileNotFoundError("QC Flags file " + flags_file + " not found")
        
        # missing values should be -9999 anyway and we turn them to NaNs at load time
        data_df = pd.read_csv(file, delimiter=",",parse_dates=["DATE_TIME"],na_values=[-9999])
        qc_df = pd.read_csv(qc_file, delimiter=",",parse_dates=["DATE_TIME"])
        flags_df = pd.read_csv(flags_file, delimiter=",",parse_dates=["DATE_TIME"], low_memory=False)
        
        # check the data and QC files match in shape
        assert data_df.shape == qc_df.shape == flags_df.shape, "Shapes of Data, QC and Flags files are not same."
        
        # check column names are the same and are in the same order
        qc_columns = [c.replace('_QCFLAG', '') for c in qc_df.columns]
        flags_columns = [c.replace('_FLAG', '') for c in flags_df.columns]
        assert data_df.columns.tolist() == qc_columns == flags_columns, "Data, QC and Flags column names are not same."

        # drop the data failing QC
        data_df = process_bitmask(data_df, qc_df, qc_bitmask)
        data_df = process_flags(data_df, flags_df, drop_qc_flags)

        # add lat/long
        station_id = data_df.SITE_ID[0]
        # ensure there's only one station ID used in the whole file
        assert data_df.SITE_ID.nunique() == 1
        latitude = metadata_df.LATITUDE[station_id]
        longitude = metadata_df.LONGITUDE[station_id]
        data_df['LATITUDE'] = latitude
        data_df['LONGITUDE'] = longitude
        
        #remove timezone from the datetime as xarray's to_netcdf doesn't like it
        data_df["DATE_TIME"] = pd.to_datetime(data_df.DATE_TIME).dt.tz_localize(None)
        #make DATE_TIME the index instead of using a index number
        all_data.append(data_df.set_index(["DATE_TIME"]))
        #should site ID form part of the index?
        #sorts all the entries by date_time as not doing can break selecting on dates
    #note that there can be multiple entries with the same timestamp so the index_col entries are not unique
    #should we make station ID part of the index?
    ds = pd.concat(all_data).sort_index().to_xarray() 

    return ds
