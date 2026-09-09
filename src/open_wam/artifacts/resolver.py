"""Resolve explicitly configured catalog objects, otherwise local artifacts."""

import json
import os
import weakref
from contextlib import ExitStack, contextmanager
from pathlib import Path

from open_wam.configs.asset_cache import ArtifactCacheConfig

from .object_cache import Catalog, ObjectCache


class ArtifactResolver:
    """One workflow's immutable catalog and process-local, bounded object cache.

    Catalog leases last for this resolver's lifetime. DataLoader workers reopen
    their own connections after fork or pickle; no process-global source
    selection can redirect an unrelated dataset or model.
    """

    def __init__(self, config: ArtifactCacheConfig | None = None) -> None:
        self.config = config
        self._state = None
        self._finalizer = None

    def __getstate__(self):
        return {"config": self.config, "_state": None, "_finalizer": None}

    def close(self) -> None:
        if self._finalizer is not None:
            self._finalizer()
        self._state = self._finalizer = None

    def _configured(self):
        if self.config is None:
            return None
        if self._state is None or self._state[0] != os.getpid():
            self.close()
            cfg = self.config
            spec = json.loads(Path(cfg.catalog_spec_path).expanduser().read_bytes())
            cache = ObjectCache(
                Path(cfg.cache_dir).expanduser(), cfg.max_bytes, cfg.min_free_bytes
            )
            leases = ExitStack()
            try:
                catalog = Catalog(leases.enter_context(cache.acquire(spec)))
                leases.callback(catalog.close)
            except BaseException:
                leases.close()
                raise
            self._finalizer = weakref.finalize(self, leases.close)
            self._state = os.getpid(), cache, catalog
        return self._state

    def contains(self, path: Path) -> bool:
        state = self._configured()
        return state is not None and state[2].get(str(path)) is not None

    @contextmanager
    def materialize(self, path: Path):
        state = self._configured()
        spec = None if state is None else state[2].get(str(path))
        if spec is None:
            # A catalog describes its own objects, not every local file in a run.
            yield path
        else:
            with state[1].acquire(spec) as cached:
                yield cached
