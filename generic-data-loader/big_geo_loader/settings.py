from dataclasses import dataclass
from pathlib import Path

DEBUG = False


class DatasetSettings:
    chunks: dict[str, int] = {"time": 24}
    precache: bool = True
    cache_dir: Path | str = Path("./.cache")
    pre_transforms: list | None = None
    runtime_transforms: list | None = None
    generate_stats: bool = True


class DefaultSettings:
    chunks: dict[str, int] = {"time": 24}
    zarr_version: int = 2
