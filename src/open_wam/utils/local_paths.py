"""Compatibility exports for configuration path resolution.

New code should import these contracts from :mod:`open_wam.configs`.
"""

from open_wam.configs.local_paths import (
    LOCAL_PATHS_ENV_VAR,
    LOCAL_PATHS_PATH,
    LOCAL_PATHS_SAMPLE_PATH,
    load_local_path_registry,
    read_yaml_with_local_paths,
)

__all__ = [
    "LOCAL_PATHS_ENV_VAR",
    "LOCAL_PATHS_PATH",
    "LOCAL_PATHS_SAMPLE_PATH",
    "load_local_path_registry",
    "read_yaml_with_local_paths",
]
