import pytest
import csv_loader
#tests to write

import math

# set to 0b111111111111 to drop all data with a QC flag
# possible QC bit fields: 0=passed, 1=missing, 2=zero data, 4=too few samples
# 8=low power, 16=sensor fault, 32=diagnostic fault, 64=out of range, 
# 128=secondary variable, 256=midnight soil heat flux calibration, 512=spike
# 1024=error code stored as value

# test QC masks

def test_qc_mask_all_flags():
    ds = csv_loader.csv_to_xarray("FRAME-FM/tests/", 0b11111111111, [])
    assert ds['LWIN'].values[0] == 339.4
    # annoyingly python won't let us assert against a list of nans properly
    # so we've got to do it one by one
    for i in range(1,9):
        assert math.isnan(ds['LWIN'].values[i])

def test_qc_mask_missing():
    ds = csv_loader.csv_to_xarray("FRAME-FM/tests/", 0b0000000001, [])
    assert ds['LWIN'].values == [339.4, math.nan, 316.6,  316.5, 313.0, 316.8, 323.6, math.nan, math.nan]

def test_qc_mask_no_flags():
    ds = csv_loader.csv_to_xarray("FRAME-FM/tests/", 0b0000000000, [])
    assert ds['LWIN'].values == [339.4, 329.4, 316.6,  316.5, 313.0, 316.8, 323.6, 336.2, 324.0]
    
# test flags
def test_flags_none():
    ds = csv_loader.csv_to_xarray("FRAME-FM/tests/", 0b0000000000, [])
    assert ds['LWIN'].values == [339.4, 329.4, 316.6,  316.5, 313.0, 316.8, 323.6, 336.2, 324.0]

def test_flags_all():
    ds = csv_loader.csv_to_xarray("FRAME-FM/tests/", 0b0000000000, ["I", "M", "J", "U"])
    assert ds['LWIN'].values == [math.nan, math.nan, math.nan, math.nan, 313.0, 316.8, 323.6, 336.2, 324.0]

# test flags

def test_flags_u():
    ds = csv_loader.csv_to_xarray("FRAME-FM/tests/", 0b0000000000, ["U"])
    assert ds['LWIN'].values == [math.nan, 329.4, 316.6,  316.5, 313.0, 316.8, 323.6, 336.2, 324.0]


def test_flags_i():
    ds = csv_loader.csv_to_xarray("FRAME-FM/tests/", 0b0000000000, ["I"])
    assert ds['LWIN'].values == [339.4, math.nan, 316.6,  316.5, 313.0, 316.8, 323.6, 336.2, 324.0]

def test_flags_e():
    ds = csv_loader.csv_to_xarray("FRAME-FM/tests/", 0b0000000000, ["E"])
    assert ds['LWIN'].values == [339.4, 329.4, math.nan, 316.5, 313.0, 316.8, 323.6, 336.2, 324.0]

def test_flags_m():
    ds = csv_loader.csv_to_xarray("FRAME-FM/tests/", 0b0000000000, ["M"])
    assert ds['LWIN'].values == [339.4, 329.4, 316.6, math.nan, 313.0, 316.8, 323.6, 336.2, 324.0]

def test_flags_invalid():
    # specify a flag we don't use, all data should be present
    ds = csv_loader.csv_to_xarray("FRAME-FM/tests/", 0b0000000000, ["A"])
    assert ds['LWIN'].values == [339.4, 329.4, 316.6, 316.5, 313.0, 316.8, 323.6, 336.2, 324.0]

