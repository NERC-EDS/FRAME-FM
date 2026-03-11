import pytest
import FRAME_FM.datasets.cosmosuk_dataset as cosmosuk

import math
import numpy as np

# set to 0b111111111111 to drop all data with a QC flag
# possible QC bit fields: 0=passed, 1=missing, 2=zero data, 4=too few samples
# 8=low power, 16=sensor fault, 32=diagnostic fault, 64=out of range, 
# 128=secondary variable, 256=midnight soil heat flux calibration, 512=spike
# 1024=error code stored as value

# test QC masks

def test_qc_mask_all_flags():
    ds = cosmosuk.CosmosUKDataset(data_uri="FRAME-FM/tests/datasets/fixtures/cosmos-uk/", qc_bitmask = 0b11111111111,  drop_qc_flags = []).data
    assert np.array_equal(ds[0]['TDT1_TSOIL'].values,[339.4, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],equal_nan=True)

def test_qc_mask_missing():
    ds = cosmosuk.CosmosUKDataset(data_uri="FRAME-FM/tests/datasets/fixtures/cosmos-uk/", qc_bitmask = 0b0000000001, drop_qc_flags = []).data
    assert np.array_equal(ds[0]['TDT1_TSOIL'].values,[339.4, math.nan, 316.6,  316.5, 313.0, 316.8, 323.6, math.nan, math.nan],equal_nan=True)

def test_qc_mask_no_flags():
    ds = cosmosuk.CosmosUKDataset(data_uri="FRAME-FM/tests/datasets/fixtures/cosmos-uk/", qc_bitmask = 0b0000000000, drop_qc_flags = []).data
    assert np.array_equal(ds[0]['TDT1_TSOIL'].values,[339.4, 329.4, 316.6,  316.5, 313.0, 316.8, 323.6, 336.2, 324.0],equal_nan=True)
    
# test flags
def test_flags_none():
    ds = cosmosuk.CosmosUKDataset(data_uri="FRAME-FM/tests/datasets/fixtures/cosmos-uk/", qc_bitmask = 0b0000000000, drop_qc_flags = []).data
    assert np.array_equal(ds[0]['TDT1_TSOIL'].values, [339.4, 329.4, 316.6,  316.5, 313.0, 316.8, 323.6, 336.2, 324.0],equal_nan=True)

def test_flags_all():
    ds = cosmosuk.CosmosUKDataset(data_uri="FRAME-FM/tests/datasets/fixtures/cosmos-uk/", qc_bitmask = 0b0000000000, drop_qc_flags =  ["I", "M", "E", "U"]).data
    assert np.array_equal(ds[0]['TDT1_TSOIL'].values, [math.nan, math.nan, math.nan, math.nan, 313.0, 316.8, 323.6, 336.2, 324.0],equal_nan=True)

def test_flags_u():
    ds = cosmosuk.CosmosUKDataset(data_uri="FRAME-FM/tests/datasets/fixtures/cosmos-uk/", qc_bitmask = 0b0000000000, drop_qc_flags =  ["U"]).data
    assert np.array_equal(ds[0]['TDT1_TSOIL'].values,[math.nan, 329.4, 316.6,  316.5, 313.0, 316.8, 323.6, 336.2, 324.0],equal_nan=True)

def test_flags_i():
    ds = cosmosuk.CosmosUKDataset(data_uri="FRAME-FM/tests/datasets/fixtures/cosmos-uk/", qc_bitmask = 0b0000000000, drop_qc_flags =  ["I"]).data
    assert np.array_equal(ds[0]['TDT1_TSOIL'].values,[339.4, math.nan, 316.6,  316.5, 313.0, 316.8, 323.6, 336.2, 324.0],equal_nan=True)

def test_flags_e():
    ds = cosmosuk.CosmosUKDataset(data_uri="FRAME-FM/tests/datasets/fixtures/cosmos-uk/", qc_bitmask = 0b0000000000, drop_qc_flags =  ["E"]).data
    assert np.array_equal(ds[0]['TDT1_TSOIL'].values,[339.4, 329.4, math.nan, 316.5, 313.0, 316.8, 323.6, 336.2, 324.0],equal_nan=True)

def test_flags_m():
    ds = cosmosuk.CosmosUKDataset(data_uri="FRAME-FM/tests/datasets/fixtures/cosmos-uk/", qc_bitmask = 0b0000000000, drop_qc_flags =  ["M"]).data
    assert np.array_equal(ds[0]['TDT1_TSOIL'].values,[339.4, 329.4, 316.6, math.nan, 313.0, 316.8, 323.6, 336.2, 324.0],equal_nan=True)

def test_flags_invalid():
    # specify a flag we don't use, all data should be present
    ds = cosmosuk.CosmosUKDataset(data_uri="FRAME-FM/tests/datasets/fixtures/cosmos-uk/", qc_bitmask = 0b0000000000, drop_qc_flags =  ["A"]).data
    assert np.array_equal(ds[0]['TDT1_TSOIL'].values,[339.4, 329.4, 316.6, 316.5, 313.0, 316.8, 323.6, 336.2, 324.0],equal_nan=True)
