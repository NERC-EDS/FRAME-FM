from torch.testing import assert_close

from .common import GEOTIFF_URI, NC_URI
from FRAME_FM.datasets.base_dataset import BaseDataset
from FRAME_FM.datasets.combined_dataset import CorrespondingTilesDataset, ZipDataset


def test_zip_dataset():
    pre_dataset = BaseDataset(
        NC_URI,
        preprocessors=[
            {"type": "vars_to_dimension", "variables": ["pre"], "new_dim": "pre"},
            {"type": "subset", "latitude": [48, 63], "longitude": [-10, 2]},
            {"type": "tiler", "boundary": "trim", "latitude": 4, "longitude": 2, "time": 6}
            ],
        transforms=[{"type": "to_tensor"}],
        )
    geotiff_dataset = BaseDataset(
        GEOTIFF_URI,
        preprocessors=[
            {"type": "sort_axis", "dim": "y"},
            {"type": "to_dataarray", "var_id": "band_data"},
            {"type": "tiler", "boundary": "trim", "x": 64, "y": 64}
            ],
        transforms=[{"type": "to_tensor"}],
        )
    min_len = min(len(pre_dataset), len(geotiff_dataset))
    zip_dataset = ZipDataset(pre_dataset, geotiff_dataset)
    assert len(zip_dataset) == min_len, \
        f"ZipDataset failure: length not min. of input lengths ({len(zip_dataset)} != {min_len})."
    for index in range(0, len(zip_dataset), len(zip_dataset) // 4):
        sample = zip_dataset[index]
        assert len(sample) == 2, \
            f"ZipDataset failure: # of items #{index} not number of datasets ({len(sample)} != 2)."
        assert_close(sample[0], pre_dataset[index], equal_nan=True, msg=(
            f"ZipDataset failure: first of items #{index} not #{index} of first dataset"
            f" ({sample[0]} cf {pre_dataset[index]})."
            ))
        assert_close(sample[1], geotiff_dataset[index], equal_nan=True, msg=(
            f"ZipDataset failure: second of items #{index} not #{index} of second dataset"
            f" ({sample[1]} cf {geotiff_dataset[index]})."
            ))


def test_corresponding_tiles_dataset():
    wgs_osng_conv_spec = ((
        (4326, {'Lat': 'latitude', 'Lon': 'longitude'}),
        (27700, {'E': 'x', 'N': 'y'}),
        ))
    geotiff_dataset = BaseDataset(
        GEOTIFF_URI,
        preprocessors=[
            {"type": "sort_axis", "dim": "y"},
            {"type": "to_dataarray", "var_id": "band_data"},
            {"type": "tiler", "boundary": "trim", "x": 64, "y": 64}
            ],
        transforms=[{"type": "to_values_bounds_tensors", "dims": ["x", "y"]}],
        )
    pre_dataset = BaseDataset(
        NC_URI,
        preprocessors=[
            {"type": "vars_to_dimension", "variables": ["pre"], "new_dim": "pre"},
            {"type": "tiler", "boundary": "trim", "latitude": 4, "longitude": 2, "time": 6}
            ],
        transforms=[{
            "type": "to_values_locations_tensors",
            "dims": ["x", "y", "time"],
            "crs_conversion_spec": wgs_osng_conv_spec,
            }],
        )
    corresponding_dataset = CorrespondingTilesDataset(
        datasets=[pre_dataset, geotiff_dataset], crs_specs=[wgs_osng_conv_spec, None]
        )
    assert len(corresponding_dataset) == len(pre_dataset), (
        f"CorrespondingTilesDataset failure: length not that of first dataset"
        f" ({len(corresponding_dataset)} != {len(pre_dataset)})."
        )
    for index in range(0, len(corresponding_dataset), len(corresponding_dataset) // 4):
        sample = corresponding_dataset[index]
        assert len(sample) == 2, (
            "CorrespondingTilesDataset failure: # of items not # of datasets"
            f" ({len(sample)} != 2)."
            )
        (first_sample_vals, first_sample_locs), (_, gt_xy_b) = sample
        pre_sample_vals, _ = pre_dataset[index]
        assert_close(first_sample_vals, pre_sample_vals, equal_nan=True, msg=(
            f"CorrespondingTilesDataset failure: first of item #{index} values not"
            f" #{index} of first dataset."
            ))
        pre_xy_c = first_sample_locs[:2].mean(dim=[1, 2, 3])
        if ((pre_xy_c[0].numpy() < geotiff_dataset.data['x'].min())
                or (pre_xy_c[0].numpy() > geotiff_dataset.data['x'].max())
                or (pre_xy_c[1].numpy() < geotiff_dataset.data['y'].min())
                or (pre_xy_c[1].numpy() > geotiff_dataset.data['y'].max())):
            continue
        assert (gt_xy_b[0, 0] <= pre_xy_c[0]) and (pre_xy_c[0] <= gt_xy_b[0, 1]), (
            "CorrespondingTilesDataset failure: first tile centroid outside x-bounds"
            f" of second tile ({pre_xy_c[0]} not in [{gt_xy_b[0, 0]}, {gt_xy_b[0, 1]}])."
            )
        assert (gt_xy_b[1, 0] <= pre_xy_c[1]) and (pre_xy_c[1] <= gt_xy_b[1, 1]), (
            "CorrespondingTilesDataset failure: first tile centroid outside y-bounds"
            f" of second tile ({pre_xy_c[1]} not in [{gt_xy_b[1, 0]}, {gt_xy_b[1, 1]}])."
            )
