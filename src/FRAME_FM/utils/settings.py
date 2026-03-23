from pathlib import Path
DEBUG = False


class DatasetSettings:
    chunks: dict[str, int] = {"time": 24}
    precache: bool = True
    cache_dir: Path | str = Path("./.cache")
    caching_backend: str = "basic" # Or "series", "dask_distributed" or "slurm" - which will use the `zarr-parallel` library.
    preprocessor_hash_key: str = "_preprocessor_cache_hash"
    zarr_format: int = 2

