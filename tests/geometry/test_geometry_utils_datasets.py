from pathlib import Path

import pytest
import pandas as pd
import numpy as np
import pyproj

from FRAME_FM.transforms import *
from FRAME_FM.transforms.transforms import transform_mapping
from FRAME_FM.utils.data_utils import load_data_from_uri

#from tests.datasets.common import CHESS_URI, ERA5_URI

from FRAME_FM.utils.geometry_utils import get_centroids, get_bounds
import FRAME_FM.utils.geometry_utils.exceptions as geo_exceptions

CHESS_URI = "/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/chess-met/aggregations/chess-met_precip_gb_1km_daily_19610101-20191231.nca" # 12M
ERA5_URI = "tests/transforms/fixtures/ecmwf-era5X_oper_an_sfc_2000_2020_2d_repack.kr1.0.json.zip" # 1.3M

pdt = pd.to_datetime

dsets = {
    "era5": {
        "uri": ERA5_URI,
        "main_var": "d2m",
        "ds": None,
    },
    "chessmet": {
        "uri": CHESS_URI,
        "main_var": "precip",
        "ds": None,
    },
}


def _convert_point(point: tuple[float, float], from_crs: str = "EPSG:4326", to_crs: str = "EPSG:4326") -> tuple[float, float]:
    transformer = pyproj.Transformer.from_crs(from_crs, to_crs, always_xy=True)
    x, y = transformer.transform(point[0], point[1])
    return x, y


def _load_data(source: str = "era5", response_type: str = "Dataset") -> xr.Dataset | xr.DataArray:
    global dsets
    if dsets[source]["ds"] is None:
        dsets[source]["ds"] = load_data_from_uri(dsets[source]["uri"], chunks="auto")   # type: ignore

    var_id = dsets[source]["main_var"]
    if response_type == "DataArray":
        resp = dsets[source]["ds"][var_id].isel(time=slice(0, 3))
    else:
        resp = dsets[source]["ds"]

    return resp, var_id


def _load_era5_data(response_type: str = "Dataset") -> xr.Dataset | xr.DataArray:
    da, var_id = _load_data(source="era5", response_type="DataArray")

    # Reduce the size of the array to 300 x 300 lat and lons and then tile it to 9 x 9 tiles of 100 x 100 lat and lons
    da = da.isel(latitude=slice(0, 300), longitude=slice(0, 300))
    return da


def _load_chessmet_data(response_type: str = "Dataset") -> xr.Dataset | xr.DataArray:
    da, var_id = _load_data(source="chessmet", response_type="DataArray")

    # Reduce the size of the array to 300 x 300 y and x and then tile it to 9 x 9 tiles of 100 x 100 y and x
    da = da.isel(x=slice(0, 300), y=slice(0, 300))
    return da


def test_get_centroids_era5_tile_by_tile():
    da = _load_era5_data(response_type="DataArray")
    tiled = TilerTransform(time=2, latitude=100, longitude=100, boundary="pad")(da)

    tile = tiled.isel(batch_dim=0)
    with pytest.raises(geo_exceptions.ExpectedDimsMismatchError, match="Object does not match `expected` dims 'tzyx'.*"):
        get_centroids(tile, expected="tzyx", chosen="yx", by="tile")

    result = get_centroids(tile, expected="tyx", chosen="yx", by="tile")
    assert result == {'centroid': {'y': 77.625, 'x': 12.375}, 'crs': 'EPSG:4326'}, f"Centroid calculation for tiled ERA5 lat/lon failed, response was: {result}."
    # Get only the y (latitude) centroid
    result_y = get_centroids(tile, expected="tyx", chosen="y", by="tile")
    assert result_y == {'centroid': {'y': 77.625}, 'crs': 'EPSG:4326'}, f"Centroid calculation for tiled ERA5 latitude failed, response was: {result_y}."

    # Get back time only, as a datetime object
    result_t = get_centroids(tile, expected="tyx", chosen="t", time_as_datetime=True, by="tile")
    assert result_t == {'centroid': {'t': np.datetime64('2000-01-01T00:30:00.000000000')}, 'crs': 'EPSG:4326'}, f"Centroid calculation for tiled ERA5 time failed, response was: {result_t}."


def test_get_centroids_era5_tile_by_pixel():
    da = _load_era5_data(response_type="DataArray")
    tiled = TilerTransform(time=2, latitude=5, longitude=5, boundary="pad")(da)

    tile = tiled.isel(batch_dim=0)
    with pytest.raises(geo_exceptions.ExpectedDimsMismatchError, match="Object does not match `expected` dims 'tzyx'.*"):
        get_centroids(tile, expected="tzyx", chosen="yx", by="pixel")

    result = get_centroids(tile, expected="tyx", chosen="yx", by="pixel")
    # Check y and x centroids are the same in both results, and that the CRS is correct
    assert np.array_equal(result["centroid_y"], tile["latitude"].values), f"Centroid calculation for tiled ERA5 latitude by pixel failed, response was: {result}."
    assert np.array_equal(result["centroid_x"], tile["longitude"].values), f"Centroid calculation for tiled ERA5 longitude by pixel failed, response was: {result}."

    # Get only the y (latitude) centroid
    result_y = get_centroids(tile, expected="tyx", chosen="y", by="pixel")
    # Check y centroids are the same in both results, and that the CRS is correct
    assert np.array_equal(result_y["centroid_y"], tile["latitude"].values), f"Centroid calculation for tiled ERA5 latitude by pixel failed, response was: {result_y}."

    # Get back time only, as a datetime object
    result_t = get_centroids(tile, expected="tyx", chosen="t", time_as_datetime=True, by="tile")
    # Check time centroid is the same in both results, and that the CRS is correct
    assert np.array_equal(result_t["centroid"]["t"], np.datetime64('2000-01-01T00:30:00.000000000')), f"Centroid calculation for tiled ERA5 time failed, response was: {result_t}."


def test_get_bounds_era5_tile_by_pixel():
    da = _load_era5_data(response_type="DataArray")
    tiled = TilerTransform(time=2, latitude=5, longitude=5, boundary="pad")(da)

    tile = tiled.isel(batch_dim=0)
    with pytest.raises(geo_exceptions.ExpectedDimsMismatchError, match="Object does not match `expected` dims 'tzyx'.*"):
        get_bounds(tile, expected="tzyx", chosen="yx", by="pixel")

    result = get_bounds(tile, expected="tyx", chosen="yx", by="pixel")

    # Calculate the bounds for latitude and longitude by pixel, which should be the same
    bounds_y = []
    tile_lats = tile["latitude"].values
    y_interval = (tile_lats[1] - tile_lats[0]) / 2
    for i in range(len(tile["latitude"].values)):
        v = tile_lats[i]
        bounds_y.append((v - y_interval, v + y_interval))

    bounds_x = []
    tile_lons = tile["longitude"].values    
    x_interval = (tile_lons[1] - tile_lons[0]) / 2
    for i in range(len(tile["longitude"].values)):
        v = tile_lons[i]
        bounds_x.append((v - x_interval, v + x_interval))

    bounds_y_min = np.array(bounds_y)[:, 0]
    bounds_y_max = np.array(bounds_y)[:, 1]
    bounds_x_min = np.array(bounds_x)[:, 0]
    bounds_x_max = np.array(bounds_x)[:, 1]

    assert np.array_equal(result["bounds_y_min"], bounds_y_min), f"Bounds y min does not match expected values for tiled ERA5 latitude by pixel, response was: {result}."
    assert np.array_equal(result["bounds_y_max"], bounds_y_max), f"Bounds y max does not match expected values for tiled ERA5 latitude by pixel, response was: {result}."
    assert np.array_equal(result["bounds_x_min"], bounds_x_min), f"Bounds x min does not match expected values for tiled ERA5 longitude by pixel, response was: {result}."
    assert np.array_equal(result["bounds_x_max"], bounds_x_max), f"Bounds x max does not match expected values for tiled ERA5 longitude by pixel, response was: {result}."

    # Get only the y (latitude) bounds
    result_y = get_bounds(tile, expected="tyx", chosen="y", by="pixel")
    # Check y bounds are the same in both results, and that the CRS is correct
    assert np.array_equal(result["bounds_y_min"], bounds_y_min), f"Bounds y min does not match expected values for tiled ERA5 latitude by pixel, response was: {result}."
    assert np.array_equal(result["bounds_y_max"], bounds_y_max), f"Bounds y max does not match expected values for tiled ERA5 latitude by pixel, response was: {result}."

    # Get back time only, as a datetime object
    result_t = get_bounds(tile, expected="tyx", chosen="t", time_as_datetime=True, by="tile")
    # Check time bounds is the same in both results, and that the CRS is correct
    assert np.array_equal(result_t["bounds"]["t"], np.array(
        (np.datetime64('1999-12-31T23:30:00.000000000'), np.datetime64('2000-01-01T01:30:00.000000000')),
        )), f"Bounds calculation for tiled ERA5 time failed, response was: {result_t}."


def test_get_centroids_chessmet_tile_by_pixel():
    da = _load_chessmet_data(response_type="DataArray")
    tiled = TilerTransform(time=2, y=5, x=5, boundary="pad")(da)

    tile = tiled.isel(batch_dim=0)
    with pytest.raises(geo_exceptions.ExpectedDimsMismatchError, match="Object does not match `expected` dims 'tzyx'.*"):
        get_centroids(tile, expected="tzyx", chosen="yx", by="pixel")

    result = get_centroids(tile, expected="tyx", chosen="yx", by="pixel")
    # Check y and x centroids are the same in both results, and that the CRS is correct
    assert np.array_equal(result["centroid_y"], tile["y"].values), f"Centroid calculation for tiled CHESSMet y by pixel failed, response was: {result}."
    assert np.array_equal(result["centroid_x"], tile["x"].values), f"Centroid calculation for tiled CHESSMet x by pixel failed, response was: {result}."

    # Get only the y (latitude) centroid
    result_y = get_centroids(tile, expected="tyx", chosen="y", by="pixel")
    # Check y centroids are the same in both results, and that the CRS is correct
    assert np.array_equal(result_y["centroid_y"], tile["y"].values), f"Centroid calculation for tiled CHESSMet y by pixel failed, response was: {result_y}."

    # Get back time only, as a datetime object
    result_t = get_centroids(tile, expected="tyx", chosen="t", time_as_datetime=True, by="tile")
    # Check time centroid is the same in both results, and that the CRS is correct
    assert np.array_equal(result_t["centroid"]["t"], np.datetime64('1961-01-01T12:00:00.000000000')), f"Centroid calculation for tiled CHESSMet time failed, response was: {result_t}."


def test_get_bounds_chessmet_tile_by_pixel():
    da = _load_chessmet_data(response_type="DataArray")
    tiled = TilerTransform(time=2, y=5, x=5, boundary="pad")(da)

    tile = tiled.isel(batch_dim=0)
    with pytest.raises(geo_exceptions.ExpectedDimsMismatchError, match="Object does not match `expected` dims 'tzyx'.*"):
        get_bounds(tile, expected="tzyx", chosen="yx", by="pixel")

    result = get_bounds(tile, expected="tyx", chosen="yx", by="pixel")

    # Calculate the bounds for latitude and longitude by pixel, which should be the same
    bounds_y = []
    tile_lats = tile["y"].values
    y_interval = (tile_lats[1] - tile_lats[0]) / 2
    for i in range(len(tile["y"].values)):
        v = tile_lats[i]
        bounds_y.append((v - y_interval, v + y_interval))

    bounds_x = []
    tile_lons = tile["x"].values
    x_interval = (tile_lons[1] - tile_lons[0]) / 2
    for i in range(len(tile["x"].values)):
        v = tile_lons[i]
        bounds_x.append((v - x_interval, v + x_interval))

    bounds_y_min = np.array(bounds_y)[:, 0]
    bounds_y_max = np.array(bounds_y)[:, 1]
    bounds_x_min = np.array(bounds_x)[:, 0]
    bounds_x_max = np.array(bounds_x)[:, 1]

    assert np.array_equal(result["bounds_y_min"], bounds_y_min), f"Bounds y min does not match expected values for tiled CHESSMet y by pixel, response was: {result}."
    assert np.array_equal(result["bounds_y_max"], bounds_y_max), f"Bounds y max does not match expected values for tiled CHESSMet y by pixel, response was: {result}."
    assert np.array_equal(result["bounds_x_min"], bounds_x_min), f"Bounds x min does not match expected values for tiled CHESSMet x by pixel, response was: {result}."
    assert np.array_equal(result["bounds_x_max"], bounds_x_max), f"Bounds x max does not match expected values for tiled CHESSMet x by pixel, response was: {result}."

    # Check CRS is correct
    assert result.crs == "crsOSGB", f"CRS does not match expected value for tiled CHESSMet by pixel, response was: {result}."

    # Get only the y (latitude) bounds
    result_y = get_bounds(tile, expected="tyx", chosen="y", by="pixel")
    # Check y bounds are the same in both results, and that the CRS is correct
    assert np.array_equal(result["bounds_y_min"], bounds_y_min), f"Bounds y min does not match expected values for tiled CHESSMet y by pixel, response was: {result}."
    assert np.array_equal(result["bounds_y_max"], bounds_y_max), f"Bounds y max does not match expected values for tiled CHESSMet y by pixel, response was: {result}."

    # Get back time only, as a datetime object
    result_t = get_bounds(tile, expected="tyx", chosen="t", time_as_datetime=True, by="tile")
    # Check time bounds is the same in both results, and that the CRS is correct
    assert np.array_equal(result_t["bounds"]["t"], np.array(
        (np.datetime64('1960-12-31T12:00:00.000000000'), np.datetime64('1961-01-02T12:00:00.000000000')),
        )), f"Bounds calculation for tiled CHESSMet time failed, response was: {result_t}."


def test_get_centroids_chessmet_tile_by_pixel_as_epsg_4326():
    da = _load_chessmet_data(response_type="DataArray")
    tiled = TilerTransform(time=2, y=5, x=5, boundary="pad")(da)
    tile = tiled.isel(batch_dim=0)

    result_lat_lon = get_centroids(tile, expected="tyx", chosen="yx", by="pixel", crs="EPSG:27700", target_crs="EPSG:4326")
    result = get_centroids(tile, expected="tyx", chosen="yx", by="pixel")

    # First centroid should be the same as the original, but transformed to EPSG:4326
    expected_first_centroid = _convert_point((result["centroid_x"][0], result["centroid_y"][0]), from_crs="EPSG:27700", to_crs="EPSG:4326")
    assert np.allclose(result_lat_lon["centroid_x"][0], expected_first_centroid[0]), f"Centroid x coordinate does not match expected value for tiled CHESSMet x by pixel with CRS transformation, response was: {result_lat_lon}."
    assert np.allclose(result_lat_lon["centroid_y"][0], expected_first_centroid[1]), f"Centroid y coordinate does not match expected value for tiled CHESSMet y by pixel with CRS transformation, response was: {result_lat_lon}."

    # Final centroid should be the same as the original, but transformed to EPSG:4326
    expected_final_centroid = _convert_point((result["centroid_x"][-1], result["centroid_y"][-1]), from_crs="EPSG:27700", to_crs="EPSG:4326")
    assert np.allclose(result_lat_lon["centroid_x"][-1], expected_final_centroid[0]), f"Centroid x coordinate does not match expected value for tiled CHESSMet x by pixel with CRS transformation, response was: {result_lat_lon}."
    assert np.allclose(result_lat_lon["centroid_y"][-1], expected_final_centroid[1]), f"Centroid y coordinate does not match expected value for tiled CHESSMet y by pixel with CRS transformation, response was: {result_lat_lon}."

    