from big_geo_loader.datasets import BigGeoDataset
from big_geo_loader.utils import dump_selectors_to_yaml, load_selectors_from_yaml


def main():

    selectors = [
            {   
                "uri": "https://gws-access.jasmin.ac.uk/public/eds_ai/era5_repack/aggregations/data/ecmwf-era5X_oper_an_sfc_2000_2020_2d_repack.kr1.0.json",
                "common": {
                    "subset": {
                        "time": ["2000-01-01 00:00:00", "2000-01-10 23:00:00"],
                        "latitude": [80, -80],
                        "longitude": [-140, 150.75]
                    },
                    "pre_transforms": [
                        {"type": "rename", "var_id": "d2m", "new_name": "dewpoint_temperature"},
                        {"type": "roll", "dim": "longitude", "shift": None},
                        {"type": "reverse_axis", "dim": "latitude"}
                    ],
                    "pre_transform_rule": "append",  # or "override" to replace common pre-transforms with variable-specific ones
                    "chunks": {"time": 48}  # Example chunking strategy, can be adjusted as needed
                },
                "variables": {
                    "d2m": {},
                }
            },
            {
                "uri": "/gws/ssde/j25b/eds_ai/frame-fm/data/inputs/chess-met/aggregations/chess-met_precip_gb_1km_daily_19610101-20191231.nca",
                "common": {
                    "subset": {
                        "time": ["2000-01-01 00:00:00", "2000-01-06 00:00:00"],
                        "y": [912500, 1041500],
                        "x": [535500, 655500],
                    },
                    "pre_transforms": [
                        {"type": "rename", "var_id": "precip", "new_name": "precipitation"},
                    ],
                    "pre_transform_rule": "append",
                    "chunks": {"time": 3, "y": 64, "x": 64}  # Example chunking strategy, can be adjusted as needed
                },
                "variables": {
                    "precip": {},
                 }
            },
        ]
    
    # Show how we can save and load selectors from YAML
    yaml_path = "selectors.yaml"
    dump_selectors_to_yaml(selectors, yaml_path)
    loaded_selectors = load_selectors_from_yaml(yaml_path)

    print(loaded_selectors)

    # Example usage
    dataset = BigGeoDataset(
        selectors=loaded_selectors,
        cache_dir="DATASET_CACHE",
        pre_transforms=None,
        runtime_transforms=["to_tensor"],
        generate_stats=True
    )
    try:
        print(f"Dataset length: {len(dataset)}")
    except Exception as e:
        print(f"Expected error getting dataset length: {e}")

    print("Pre-caching data...")
    dataset.precache_data()

    print(f"Dataset length: {len(dataset)}")

    sample = dataset[0]
    print(f"First sample in dataset: {sample.shape = }, {sample.dtype = }, {sample.min() = }, {sample.max() = }")  


if __name__ == "__main__":
    main()
