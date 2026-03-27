# SPDX-FileCopyrightText: 2026 2026 FRAME-FM Contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
DEBUG = False


class DatasetSettings:
    chunks: dict[str, int] = {"time": 24}
    precache: bool = True
    cache_dir: Path | str = Path("./.cache")
    preprocessor_hash_key: str = "_preprocessor_cache_hash"


class DefaultSettings:
    chunks: dict[str, int] = {"time": 24}
    zarr_format: int = 2