"""Compatibility exports for dependency-free project path contracts.

New code should import these helpers from :mod:`open_wam.contracts`.
"""

from open_wam.contracts.paths import REPO_ROOT, find_repo_root, resolve_repo_path

__all__ = ["REPO_ROOT", "find_repo_root", "resolve_repo_path"]
