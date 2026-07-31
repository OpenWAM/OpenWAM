from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from io import BytesIO
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Iterable
import warnings

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import Dataset

from open_wam.configs import (
    ConsortiumCacheMode,
    ConsortiumChannelSelectionMode,
    ConsortiumFramePackingOrder,
    ConsortiumMissingChannelPolicy,
    ConsortiumSplitMode,
    ConsortiumViewPackingMode,
    DataConfig,
    LeRobotConsortiumDataConfig,
)
from open_wam.configs.enums import serialize_enum_values

from .contracts import WAMSample
from .distributed_sampling import UnpaddedEpochOrderDistributedSampler
from .lerobot_consortium_contracts import (
    build_lerobot_consortium_contract_catalog_from_inventory_rows,
    write_lerobot_consortium_contract_catalog,
)
from .lerobot_consortium_index import (
    LeRobotConsortiumInventoryRow,
    LeRobotConsortiumRepoTarget,
    build_lerobot_consortium_inventory,
    infer_lerobot_consortium_source_group,
    load_lerobot_consortium_inventory_rows,
    load_lerobot_consortium_repo_targets,
    write_lerobot_consortium_inventory_csv,
    write_lerobot_consortium_inventory_markdown,
    write_lerobot_consortium_repo_targets,
)
from .lerobot_consortium_sampling import ConsortiumEpochOrderPlan
from .row_action_targets import build_row_action_targets, resolve_row_key
from .sequence_packing import pack_temporal_sequence


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONSORTIUM_INDEX_REPO_IDS_PATH = _REPO_ROOT / "notes" / "index" / "lerobot_consortium_hf_repo_ids.txt"
_CONSORTIUM_INDEX_INVENTORY_CSV_PATH = _REPO_ROOT / "notes" / "index" / "lerobot_consortium_hf_dataset_inventory.csv"
_CONSORTIUM_INDEX_INVENTORY_MD_PATH = _REPO_ROOT / "notes" / "index" / "lerobot_consortium_hf_dataset_inventory.md"
_CONSORTIUM_INDEX_CONTRACTS_JSON_PATH = _REPO_ROOT / "notes" / "index" / "lerobot_consortium_hf_dataset_contracts.json"
_CONSORTIUM_INDEX_SANITY_CACHE: set[tuple[str, ...]] = set()


def _parse_feature_dim(feature: dict[str, Any] | None) -> int | None:
    if not isinstance(feature, dict):
        return None
    shape = feature.get("shape")
    if isinstance(shape, int):
        return int(shape)
    if isinstance(shape, (list, tuple)):
        if len(shape) == 1:
            return int(shape[0])
        return int(shape[-1]) if shape else None
    return None


def _parse_visual_shape(shape: Any) -> tuple[int | None, int | None, int | None, str]:
    if not isinstance(shape, (list, tuple)):
        return None, None, None, "unknown"
    dims = [int(value) for value in shape]
    if len(dims) == 2:
        return dims[0], dims[1], None, "hw"
    if len(dims) != 3:
        return None, None, None, "unknown"
    a, b, c = dims
    if a <= 4 and b > 16 and c > 16:
        return b, c, a, "chw"
    if c <= 4 and a > 16 and b > 16:
        return a, b, c, "hwc"
    return a, b, c, "unknown"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _strip_file_uri(value: str | None) -> str | None:
    if value is None:
        return None
    if value.startswith("file://"):
        return value[len("file://") :]
    return value


@dataclass(frozen=True)
class ConsortiumSourceSpec:
    member_id: str
    repo_id: str | None
    local_root: str | None


@dataclass(frozen=True)
class ConsortiumEpisodeRecord:
    episode_index: int
    length: int
    tasks: tuple[str, ...]


@dataclass(frozen=True)
class ConsortiumVisualChannelContract:
    source_name: str
    dtype: str
    height: int | None
    width: int | None
    channels: int | None
    channel_order: str


@dataclass(frozen=True)
class ConsortiumMemberContract:
    member_id: str
    repo_id: str | None
    local_root: str | None
    source_group: str | None
    fps: float | None
    observation_fps: float | None
    action_fps: float | None
    chunk_size: int
    total_episodes: int
    total_frames: int | None
    data_path_template: str
    visual_channels: tuple[ConsortiumVisualChannelContract, ...]
    action_dim: int | None
    state_dim: int | None
    episodes: tuple[ConsortiumEpisodeRecord, ...]
    tasks_by_index: dict[int, str]


@dataclass(frozen=True)
class ConsortiumCatalog:
    members: tuple[ConsortiumMemberContract, ...]


@dataclass(frozen=True)
class ConsortiumEpisodeKey:
    member_id: str
    repo_id: str | None
    episode_index: int


@dataclass(frozen=True)
class ConsortiumChannelSelection:
    target_slot: str
    source_name: str | None


@dataclass(frozen=True)
class ConsortiumWindowRecord:
    member_id: str
    repo_id: str | None
    episode_index: int
    observation_start: int
    source_camera_name: str | None
    channel_selections: tuple[ConsortiumChannelSelection, ...]


@dataclass(frozen=True)
class ConsortiumResolvedSplit:
    train_episodes: tuple[ConsortiumEpisodeKey, ...]
    val_episodes: tuple[ConsortiumEpisodeKey, ...]
    audit_payload: dict[str, Any]


class NoopConsortiumCache:
    def resolve(self, *, source: ConsortiumSourceSpec, relative_path: str, cache_dir: str | None) -> Path:
        if source.local_root is not None:
            return Path(source.local_root).expanduser().resolve() / relative_path
        if source.repo_id is None:
            raise ValueError(f"Cannot resolve consortium source for member '{source.member_id}' without repo_id.")
        return Path(
            hf_hub_download(
                repo_id=source.repo_id,
                filename=relative_path,
                repo_type="dataset",
                cache_dir=cache_dir,
            )
        )


class LocalConsortiumCache:
    def __init__(self, root: str) -> None:
        self.root = Path(root).expanduser().resolve()

    def path_for(self, *, source: ConsortiumSourceSpec, relative_path: str) -> Path:
        return self.root / source.member_id / relative_path

    def has(self, *, source: ConsortiumSourceSpec, relative_path: str) -> bool:
        return self.path_for(source=source, relative_path=relative_path).exists()

    def resolve(self, *, source: ConsortiumSourceSpec, relative_path: str) -> Path:
        return self.path_for(source=source, relative_path=relative_path)

    def store(self, *, source: ConsortiumSourceSpec, relative_path: str, source_path: Path) -> Path:
        target = self.path_for(source=source, relative_path=relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source_path.resolve() != target.resolve():
            shutil.copy2(source_path, target)
        return target


class CloudConsortiumCache:
    """Filesystem-backed stand-in for optional cloud cache roots.

    The cache is named "cloud" at the contract level, but intentionally stays
    backend-agnostic here: a mounted filesystem path is enough to exercise the
    interface and keeps the default disabled path simple.
    """

    def __init__(self, root: str) -> None:
        self.root = Path(_strip_file_uri(root) or root).expanduser().resolve()

    def path_for(self, *, source: ConsortiumSourceSpec, relative_path: str) -> Path:
        return self.root / source.member_id / relative_path

    def has(self, *, source: ConsortiumSourceSpec, relative_path: str) -> bool:
        return self.path_for(source=source, relative_path=relative_path).exists()

    def resolve(self, *, source: ConsortiumSourceSpec, relative_path: str) -> Path:
        return self.path_for(source=source, relative_path=relative_path)

    def store(self, *, source: ConsortiumSourceSpec, relative_path: str, source_path: Path) -> Path:
        target = self.path_for(source=source, relative_path=relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source_path.resolve() != target.resolve():
            shutil.copy2(source_path, target)
        return target


class ConsortiumSourceResolver:
    """Resolve consortium source files with optional local/cloud caches."""

    def __init__(self, data_config: LeRobotConsortiumDataConfig) -> None:
        self.data_config = data_config
        self.noop = NoopConsortiumCache()
        self.local_cache = (
            LocalConsortiumCache(data_config.local_cache.root)
            if data_config.local_cache.mode != ConsortiumCacheMode.DISABLED and data_config.local_cache.root
            else None
        )
        self.cloud_cache = (
            CloudConsortiumCache(data_config.cloud_cache.root)
            if data_config.cloud_cache.mode != ConsortiumCacheMode.DISABLED and data_config.cloud_cache.root
            else None
        )

    def resolve(self, *, source: ConsortiumSourceSpec, relative_path: str) -> Path:
        if self.local_cache is not None and self.local_cache.has(source=source, relative_path=relative_path):
            return self.local_cache.resolve(source=source, relative_path=relative_path)
        if self.cloud_cache is not None and self.cloud_cache.has(source=source, relative_path=relative_path):
            resolved = self.cloud_cache.resolve(source=source, relative_path=relative_path)
            if self.local_cache is not None and self.data_config.local_cache.mode == ConsortiumCacheMode.WRITE_THROUGH:
                return self.local_cache.store(source=source, relative_path=relative_path, source_path=resolved)
            return resolved

        source_path = self.noop.resolve(source=source, relative_path=relative_path, cache_dir=self.data_config.cache_dir)

        if self.cloud_cache is not None and self.data_config.cloud_cache.mode == ConsortiumCacheMode.WRITE_THROUGH:
            cached = self.cloud_cache.store(source=source, relative_path=relative_path, source_path=source_path)
            if self.local_cache is not None and self.data_config.local_cache.mode == ConsortiumCacheMode.WRITE_THROUGH:
                return self.local_cache.store(source=source, relative_path=relative_path, source_path=cached)
            return cached
        if self.local_cache is not None and self.data_config.local_cache.mode == ConsortiumCacheMode.WRITE_THROUGH:
            return self.local_cache.store(source=source, relative_path=relative_path, source_path=source_path)
        return source_path

    def read_json(self, *, source: ConsortiumSourceSpec, relative_path: str) -> dict[str, Any]:
        return _read_json(self.resolve(source=source, relative_path=relative_path))

    def read_jsonl(self, *, source: ConsortiumSourceSpec, relative_path: str) -> list[dict[str, Any]]:
        return _read_jsonl(self.resolve(source=source, relative_path=relative_path))

    def read_parquet_rows(self, *, source: ConsortiumSourceSpec, relative_path: str) -> list[dict[str, Any]]:
        return pq.read_table(self.resolve(source=source, relative_path=relative_path)).to_pylist()


def discover_local_lerobot_consortium_members(local_root: str | None) -> tuple[ConsortiumSourceSpec, ...]:
    if local_root is None:
        return ()
    root = Path(local_root).expanduser().resolve()
    if not root.exists():
        return ()
    candidates: list[Path] = []
    if (root / "meta" / "info.json").exists():
        candidates.append(root)
    else:
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if (child / "meta" / "info.json").exists() and (child / "meta" / "episodes.jsonl").exists():
                candidates.append(child)
    return tuple(
        ConsortiumSourceSpec(member_id=candidate.name, repo_id=None, local_root=str(candidate))
        for candidate in candidates
    )


def _resolve_member_sources(data_config: LeRobotConsortiumDataConfig) -> tuple[ConsortiumSourceSpec, ...]:
    explicit = [
        ConsortiumSourceSpec(
            member_id=_resolve_member_id(
                explicit_member_id=member.member_id,
                repo_id=member.repo_id,
                local_root=member.local_root,
            ),
            repo_id=member.repo_id,
            local_root=member.local_root,
        )
        for member in data_config.consortium_members
        if member.enabled
    ]
    if explicit:
        return tuple(explicit)
    discovered = discover_local_lerobot_consortium_members(data_config.local_root)
    if discovered:
        return discovered
    if data_config.repo_id is not None:
        return (
            ConsortiumSourceSpec(
                member_id=_resolve_member_id(
                    explicit_member_id=None,
                    repo_id=data_config.repo_id,
                    local_root=None,
                ),
                repo_id=data_config.repo_id,
                local_root=None,
            ),
        )
    raise ValueError(
        "LeRobot consortium loader requires either `data.consortium_members`, "
        "`data.local_root` with discoverable repo bundles, or `data.repo_id`."
    )


def _resolve_member_id(
    *,
    explicit_member_id: str | None,
    repo_id: str | None,
    local_root: str | None,
) -> str:
    if explicit_member_id:
        return explicit_member_id
    if repo_id:
        return repo_id
    if local_root:
        return Path(local_root).expanduser().resolve().name
    raise ValueError("Cannot resolve consortium member id without explicit id, repo_id, or local_root.")


def _resolve_source_group(
    data_config: LeRobotConsortiumDataConfig,
    *,
    member_id: str,
) -> str | None:
    for member in data_config.consortium_members:
        candidate_id = _resolve_member_id(
            explicit_member_id=member.member_id,
            repo_id=member.repo_id,
            local_root=member.local_root,
        )
        if candidate_id == member_id:
            return member.source_group
    return None


def _configured_remote_repo_ids(data_config: LeRobotConsortiumDataConfig) -> tuple[str, ...]:
    repo_ids = sorted({source.repo_id for source in _resolve_member_sources(data_config) if source.repo_id is not None})
    return tuple(repo_ids)


def _configured_remote_repo_targets(data_config: LeRobotConsortiumDataConfig) -> tuple[LeRobotConsortiumRepoTarget, ...]:
    deduped: dict[str, LeRobotConsortiumRepoTarget] = {}
    for source in _resolve_member_sources(data_config):
        if source.repo_id is None:
            continue
        source_group = _resolve_source_group(data_config, member_id=source.member_id) or infer_lerobot_consortium_source_group(
            source.repo_id,
            default_source_group="manual",
        )
        deduped.setdefault(
            source.repo_id,
            LeRobotConsortiumRepoTarget(
                repo_id=source.repo_id,
                source_group=source_group,
            ),
        )
    return tuple(sorted(deduped.values(), key=lambda target: (target.source_group, target.repo_id)))


def _consortium_index_prompt_available() -> bool:
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except Exception:  # pragma: no cover - defensive tty guard
        return False


def _refresh_lerobot_consortium_index_snapshots(
    data_config: LeRobotConsortiumDataConfig,
) -> None:
    configured_targets = _configured_remote_repo_targets(data_config)

    target_by_repo_id: dict[str, LeRobotConsortiumRepoTarget] = {}
    if _CONSORTIUM_INDEX_REPO_IDS_PATH.exists():
        for target in load_lerobot_consortium_repo_targets(
            _CONSORTIUM_INDEX_REPO_IDS_PATH,
            default_source_group="manual",
        ):
            target_by_repo_id[target.repo_id] = target

    existing_inventory_rows: list[LeRobotConsortiumInventoryRow] = []
    if _CONSORTIUM_INDEX_INVENTORY_CSV_PATH.exists():
        existing_inventory_rows = load_lerobot_consortium_inventory_rows(_CONSORTIUM_INDEX_INVENTORY_CSV_PATH)

    for target in configured_targets:
        target_by_repo_id[target.repo_id] = target

    inventory_by_repo_id = {row.repo_id: row for row in existing_inventory_rows}
    repo_targets_to_refresh = [
        target
        for repo_id, target in sorted(target_by_repo_id.items())
        if repo_id not in inventory_by_repo_id
    ]
    if repo_targets_to_refresh:
        refreshed_rows = build_lerobot_consortium_inventory(repo_targets_to_refresh)
        for row in refreshed_rows:
            inventory_by_repo_id[row.repo_id] = row

    retained_repo_ids = set(target_by_repo_id)
    merged_inventory_rows = sorted(
        (row for repo_id, row in inventory_by_repo_id.items() if repo_id in retained_repo_ids),
        key=lambda row: (row.source_group, row.repo_id),
    )
    merged_repo_targets = tuple(sorted(target_by_repo_id.values(), key=lambda target: (target.source_group, target.repo_id)))
    contracts = build_lerobot_consortium_contract_catalog_from_inventory_rows(merged_inventory_rows)

    write_lerobot_consortium_repo_targets(_CONSORTIUM_INDEX_REPO_IDS_PATH, merged_repo_targets)
    write_lerobot_consortium_inventory_csv(_CONSORTIUM_INDEX_INVENTORY_CSV_PATH, merged_inventory_rows)
    write_lerobot_consortium_inventory_markdown(_CONSORTIUM_INDEX_INVENTORY_MD_PATH, merged_inventory_rows)
    write_lerobot_consortium_contract_catalog(_CONSORTIUM_INDEX_CONTRACTS_JSON_PATH, contracts)


def validate_lerobot_consortium_index_snapshot(data_config: LeRobotConsortiumDataConfig) -> None:
    configured_repo_ids = _configured_remote_repo_ids(data_config)
    if not configured_repo_ids:
        return
    if configured_repo_ids in _CONSORTIUM_INDEX_SANITY_CACHE:
        return

    issues: list[str] = []
    repo_list_ids: tuple[str, ...] = ()
    inventory_repo_ids: tuple[str, ...] = ()
    contract_repo_ids: tuple[str, ...] = ()
    contract_count: int | None = None

    if not _CONSORTIUM_INDEX_REPO_IDS_PATH.exists():
        issues.append(f"missing repo-id list: {_CONSORTIUM_INDEX_REPO_IDS_PATH}")
    else:
        repo_list_ids = tuple(target.repo_id for target in load_lerobot_consortium_repo_targets(_CONSORTIUM_INDEX_REPO_IDS_PATH))

    if not _CONSORTIUM_INDEX_INVENTORY_CSV_PATH.exists():
        issues.append(f"missing inventory CSV: {_CONSORTIUM_INDEX_INVENTORY_CSV_PATH}")
    else:
        inventory_rows = load_lerobot_consortium_inventory_rows(_CONSORTIUM_INDEX_INVENTORY_CSV_PATH)
        inventory_repo_ids = tuple(row.repo_id for row in inventory_rows)

    if not _CONSORTIUM_INDEX_CONTRACTS_JSON_PATH.exists():
        issues.append(f"missing contracts JSON: {_CONSORTIUM_INDEX_CONTRACTS_JSON_PATH}")
    else:
        contracts_payload = json.loads(_CONSORTIUM_INDEX_CONTRACTS_JSON_PATH.read_text(encoding="utf-8"))
        contract_repo_ids = tuple(dataset["repo_id"] for dataset in contracts_payload.get("datasets", ()))
        contract_count = int(contracts_payload.get("dataset_count", len(contract_repo_ids)))

    repo_list_set = set(repo_list_ids)
    inventory_set = set(inventory_repo_ids)
    contract_set = set(contract_repo_ids)

    if repo_list_ids and inventory_repo_ids and len(repo_list_ids) != len(inventory_repo_ids):
        issues.append(
            "repo-id list and inventory CSV row count differ: "
            f"{len(repo_list_ids)} vs {len(inventory_repo_ids)}"
        )
    if inventory_repo_ids and contract_repo_ids and len(inventory_repo_ids) != len(contract_repo_ids):
        issues.append(
            "inventory CSV and contracts dataset count differ: "
            f"{len(inventory_repo_ids)} vs {len(contract_repo_ids)}"
        )
    if contract_count is not None and contract_count != len(contract_repo_ids):
        issues.append(
            "contracts JSON dataset_count does not match contained dataset rows: "
            f"{contract_count} vs {len(contract_repo_ids)}"
        )
    if repo_list_ids and inventory_repo_ids and repo_list_set != inventory_set:
        missing_from_inventory = sorted(repo_list_set - inventory_set)
        missing_from_repo_list = sorted(inventory_set - repo_list_set)
        issues.append(
            "repo-id list and inventory CSV repo sets differ"
            + (f"; missing_from_inventory={missing_from_inventory}" if missing_from_inventory else "")
            + (f"; missing_from_repo_list={missing_from_repo_list}" if missing_from_repo_list else "")
        )
    if inventory_repo_ids and contract_repo_ids and inventory_set != contract_set:
        missing_from_contracts = sorted(inventory_set - contract_set)
        missing_from_inventory = sorted(contract_set - inventory_set)
        issues.append(
            "inventory CSV and contracts JSON repo sets differ"
            + (f"; missing_from_contracts={missing_from_contracts}" if missing_from_contracts else "")
            + (f"; missing_from_inventory={missing_from_inventory}" if missing_from_inventory else "")
        )

    missing_for_current_loader = sorted(
        repo_id
        for repo_id in configured_repo_ids
        if repo_id not in repo_list_set or repo_id not in inventory_set or repo_id not in contract_set
    )
    if missing_for_current_loader:
        issues.append(
            "current consortium config uses repo ids not fully represented in the local snapshots: "
            f"{missing_for_current_loader}"
        )

    if not issues:
        _CONSORTIUM_INDEX_SANITY_CACHE.add(configured_repo_ids)
        return

    message = (
        "Detected discrepancy between the configured LeRobot consortium repo ids and the local parsed inventory/contracts. "
        "This usually means the repo-id list, inventory CSV, and contract JSON are out of sync.\n"
        + "\n".join(f"- {issue}" for issue in issues)
        + "\nRefresh command: "
        + "PYTHONPATH=src python scripts/build_lerobot_consortium_index.py "
          "--repo-list notes/index/lerobot_consortium_hf_repo_ids.txt"
    )

    if _consortium_index_prompt_available():
        prompt = (
            f"{message}\n"
            "Refresh the local consortium inventory/contracts now? "
            "(HF metadata only; no dataset data/video download) [y/N]: "
        )
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            answer = ""
        if answer in {"y", "yes"}:
            try:
                _refresh_lerobot_consortium_index_snapshots(data_config)
            except Exception as exc:  # pragma: no cover - defensive refresh guard
                warnings.warn(f"{message}\nAutomatic refresh failed: {exc}", stacklevel=2)
            else:
                _CONSORTIUM_INDEX_SANITY_CACHE.add(configured_repo_ids)
            return

    warnings.warn(message, stacklevel=2)
    _CONSORTIUM_INDEX_SANITY_CACHE.add(configured_repo_ids)


def build_lerobot_consortium_catalog(data_config: LeRobotConsortiumDataConfig) -> ConsortiumCatalog:
    validate_lerobot_consortium_index_snapshot(data_config)
    resolver = ConsortiumSourceResolver(data_config)
    members: list[ConsortiumMemberContract] = []
    for source in _resolve_member_sources(data_config):
        info = resolver.read_json(source=source, relative_path="meta/info.json")
        episodes = resolver.read_jsonl(source=source, relative_path="meta/episodes.jsonl")
        tasks = resolver.read_jsonl(source=source, relative_path="meta/tasks.jsonl")
        features = info.get("features", {})
        observation_fps_raw = info.get("observation_fps", info.get("fps"))
        action_fps_raw = info.get("action_fps", info.get("fps"))
        visual_channels: list[ConsortiumVisualChannelContract] = []
        for feature_name, feature in features.items():
            if not isinstance(feature, dict):
                continue
            dtype = str(feature.get("dtype") or "").lower()
            if dtype not in {"image", "video"}:
                continue
            height, width, channels, channel_order = _parse_visual_shape(feature.get("shape"))
            visual_channels.append(
                ConsortiumVisualChannelContract(
                    source_name=str(feature_name),
                    dtype=dtype,
                    height=height,
                    width=width,
                    channels=channels,
                    channel_order=channel_order,
                )
            )
        episodes_payload = tuple(
            ConsortiumEpisodeRecord(
                episode_index=int(record["episode_index"]),
                length=int(record["length"]),
                tasks=tuple(record.get("tasks", ())),
            )
            for record in episodes
        )
        action_feature = features.get(data_config.action_target.source_key)
        if action_feature is None:
            action_feature = features.get(f"{data_config.action_target.source_key}s")
        state_feature = features.get(data_config.action_target.pose_source_key)
        if state_feature is None:
            state_feature = features.get(f"{data_config.action_target.pose_source_key}s")
        members.append(
            ConsortiumMemberContract(
                member_id=source.member_id,
                repo_id=source.repo_id,
                local_root=source.local_root,
                source_group=_resolve_source_group(data_config, member_id=source.member_id),
                fps=float(info["fps"]) if "fps" in info else None,
                observation_fps=float(observation_fps_raw) if observation_fps_raw is not None else None,
                action_fps=float(action_fps_raw) if action_fps_raw is not None else None,
                chunk_size=int(info.get("chunks_size", info.get("chunk_size", 1))),
                total_episodes=int(info.get("total_episodes", len(episodes_payload))),
                total_frames=int(info["total_frames"]) if info.get("total_frames") is not None else None,
                data_path_template=str(info["data_path"]),
                visual_channels=tuple(visual_channels),
                action_dim=_parse_feature_dim(action_feature),
                state_dim=_parse_feature_dim(state_feature),
                episodes=episodes_payload,
                tasks_by_index={
                    int(record["task_index"]): str(record["task"])
                    for record in tasks
                },
            )
        )
    return ConsortiumCatalog(members=tuple(sorted(members, key=lambda item: item.member_id)))


def _resolve_channel_selections(
    data_config: LeRobotConsortiumDataConfig,
    member_contract: ConsortiumMemberContract,
) -> tuple[ConsortiumChannelSelection, ...]:
    available_names = [channel.source_name for channel in member_contract.visual_channels]
    member_cfg = next(
        (
            member
            for member in data_config.consortium_members
            if _resolve_member_id(
                explicit_member_id=member.member_id,
                repo_id=member.repo_id,
                local_root=member.local_root,
            )
            == member_contract.member_id
        ),
        None,
    )

    if member_cfg is not None and member_cfg.channel_mappings:
        mapping_items = member_cfg.channel_mappings
    else:
        mapping_items = data_config.channel_mappings

    if data_config.view_packing_mode == ConsortiumViewPackingMode.MULTICAM_AS_FRAMES:
        return _resolve_frame_packed_channel_selections(
            data_config=data_config,
            member_contract=member_contract,
            available_names=available_names,
            member_cfg=member_cfg,
            mapping_items=mapping_items,
        )
    return _resolve_slot_packed_channel_selections(
        data_config=data_config,
        member_contract=member_contract,
        available_names=available_names,
        member_cfg=member_cfg,
        mapping_items=mapping_items,
    )


def _resolve_slot_packed_channel_selections(
    *,
    data_config: LeRobotConsortiumDataConfig,
    member_contract: ConsortiumMemberContract,
    available_names: list[str],
    member_cfg: Any,
    mapping_items: tuple[Any, ...],
) -> tuple[ConsortiumChannelSelection, ...]:
    selections: dict[str, str | None] = {slot: None for slot in data_config.camera_names}

    if data_config.channel_selection_mode == ConsortiumChannelSelectionMode.EXPLICIT_MAPPING:
        for mapping in mapping_items:
            if mapping.target_slot not in selections:
                raise ValueError(
                    f"Consortium channel mapping for member '{member_contract.member_id}' targets unknown slot "
                    f"'{mapping.target_slot}'. Known slots: {data_config.camera_names}."
                )
            if mapping.source_name in available_names:
                selections[mapping.target_slot] = mapping.source_name
    elif data_config.channel_selection_mode == ConsortiumChannelSelectionMode.REQUIRED_SUBSET:
        required_channels = member_cfg.include_channels if (member_cfg and member_cfg.include_channels) else data_config.required_channels
        if not required_channels:
            raise ValueError("`required_subset` channel selection requires non-empty `required_channels`.")
        for source_name in required_channels:
            if source_name not in available_names:
                if data_config.missing_channel_policy == ConsortiumMissingChannelPolicy.ERROR:
                    raise ValueError(
                        f"Member '{member_contract.member_id}' is missing required visual channel '{source_name}'."
                    )
                continue
            if source_name in selections:
                selections[source_name] = source_name
                continue
            raise ValueError(
                "Required consortium channels must either match configured camera_names or use explicit_mapping mode."
            )
    else:
        # All-available mode maps discovered source streams onto the declared
        # canonical slots in order. Missing slots are handled by the policy below.
        for target_slot, source_name in zip(data_config.camera_names, available_names):
            selections[target_slot] = source_name

    resolved = tuple(
        ConsortiumChannelSelection(target_slot=target_slot, source_name=source_name)
        for target_slot, source_name in selections.items()
    )
    if data_config.missing_channel_policy == ConsortiumMissingChannelPolicy.ERROR:
        missing = [item.target_slot for item in resolved if item.source_name is None]
        if missing:
            raise ValueError(
                f"Member '{member_contract.member_id}' is missing required consortium slots {missing}. "
                f"Available channels: {available_names}."
            )
    return resolved


def _dedupe_preserve_order(values: Iterable[str | None]) -> tuple[str | None, ...]:
    seen: set[str | None] = set()
    resolved: list[str | None] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        resolved.append(value)
    return tuple(resolved)


def _resolve_frame_packed_channel_selections(
    *,
    data_config: LeRobotConsortiumDataConfig,
    member_contract: ConsortiumMemberContract,
    available_names: list[str],
    member_cfg: Any,
    mapping_items: tuple[Any, ...],
) -> tuple[ConsortiumChannelSelection, ...]:
    target_slot = data_config.camera_names[0]
    source_names: tuple[str | None, ...]

    if data_config.frame_packing_order != ConsortiumFramePackingOrder.CAMERA_MAJOR:
        raise ValueError(
            f"Unsupported consortium frame packing order: {data_config.frame_packing_order}."
        )

    if data_config.channel_selection_mode == ConsortiumChannelSelectionMode.EXPLICIT_MAPPING:
        resolved_sources: list[str | None] = []
        for mapping in mapping_items:
            if mapping.target_slot != target_slot:
                raise ValueError(
                    "`view_packing_mode=multicam_as_frames` requires all explicit mappings to target "
                    f"the single configured slot '{target_slot}', got '{mapping.target_slot}'."
                )
            if mapping.source_name in available_names:
                resolved_sources.append(mapping.source_name)
        source_names = _dedupe_preserve_order(resolved_sources)
    elif data_config.channel_selection_mode == ConsortiumChannelSelectionMode.REQUIRED_SUBSET:
        required_channels = member_cfg.include_channels if (member_cfg and member_cfg.include_channels) else data_config.required_channels
        if not required_channels:
            raise ValueError("`required_subset` channel selection requires non-empty `required_channels`.")
        resolved_sources = []
        for source_name in required_channels:
            if source_name not in available_names:
                if data_config.missing_channel_policy == ConsortiumMissingChannelPolicy.ERROR:
                    raise ValueError(
                        f"Member '{member_contract.member_id}' is missing required visual channel '{source_name}'."
                    )
                continue
            resolved_sources.append(source_name)
        source_names = _dedupe_preserve_order(resolved_sources)
    else:
        source_names = _dedupe_preserve_order(available_names)

    if not source_names:
        if data_config.missing_channel_policy == ConsortiumMissingChannelPolicy.ERROR:
            raise ValueError(
                f"Member '{member_contract.member_id}' exposes no usable channels for single-slot frame packing. "
                f"Available channels: {available_names}."
            )
        source_names = (None,)

    return tuple(
        ConsortiumChannelSelection(target_slot=target_slot, source_name=source_name)
        for source_name in source_names
    )


def _episode_membership_from_manifest(
    manifest_rows: tuple,
    *,
    allowed_member_ids: set[str],
) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for item in manifest_rows:
        if item.member_id not in allowed_member_ids:
            continue
        for episode_index in item.episode_indices:
            keys.add((item.member_id, int(episode_index)))
    return keys


def resolve_lerobot_consortium_train_val_split(
    data_config: LeRobotConsortiumDataConfig,
    catalog: ConsortiumCatalog,
) -> ConsortiumResolvedSplit:
    member_lookup = {member.member_id: member for member in catalog.members}
    allowed_member_ids = set(member_lookup)
    all_episode_keys = [
        ConsortiumEpisodeKey(member_id=member.member_id, repo_id=member.repo_id, episode_index=episode.episode_index)
        for member in catalog.members
        for episode in member.episodes
    ]

    if data_config.split_mode == ConsortiumSplitMode.EXPLICIT_MANIFEST:
        train_membership = _episode_membership_from_manifest(
            data_config.explicit_train_episodes,
            allowed_member_ids=allowed_member_ids,
        )
        val_membership = _episode_membership_from_manifest(
            data_config.explicit_val_episodes,
            allowed_member_ids=allowed_member_ids,
        )
        train_keys = [key for key in all_episode_keys if (key.member_id, key.episode_index) in train_membership]
        val_keys = [key for key in all_episode_keys if (key.member_id, key.episode_index) in val_membership]
    elif data_config.split_mode == ConsortiumSplitMode.HASH_BY_EPISODE:
        train_keys = []
        val_keys = []
        for key in all_episode_keys:
            token = f"{data_config.split_seed}:{key.member_id}:{key.episode_index}".encode("utf-8")
            score = int(hashlib.sha256(token).hexdigest()[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
            if score < data_config.train_fraction:
                train_keys.append(key)
            else:
                val_keys.append(key)
    else:
        shuffled = list(all_episode_keys)
        rng = random.Random(data_config.split_seed)
        rng.shuffle(shuffled)
        train_count = int(len(shuffled) * data_config.train_fraction)
        train_count = min(max(train_count, 1), len(shuffled)) if shuffled else 0
        train_keys = shuffled[:train_count]
        val_keys = shuffled[train_count:]

    train_keys = sorted(train_keys, key=lambda item: (item.member_id, item.episode_index))
    val_keys = sorted(val_keys, key=lambda item: (item.member_id, item.episode_index))

    if data_config.max_train_episodes is not None:
        train_keys = train_keys[: data_config.max_train_episodes]
    if data_config.max_val_episodes is not None:
        val_keys = val_keys[: data_config.max_val_episodes]
    if not val_keys and train_keys:
        val_keys = train_keys[:1]

    audit_payload = {
        "split_mode": data_config.split_mode,
        "split_seed": data_config.split_seed,
        "train_fraction": data_config.train_fraction,
        "train_episode_keys": [
            {"member_id": key.member_id, "repo_id": key.repo_id, "episode_index": key.episode_index}
            for key in train_keys
        ],
        "val_episode_keys": [
            {"member_id": key.member_id, "repo_id": key.repo_id, "episode_index": key.episode_index}
            for key in val_keys
        ],
    }
    return ConsortiumResolvedSplit(
        train_episodes=tuple(train_keys),
        val_episodes=tuple(val_keys),
        audit_payload=audit_payload,
    )


def build_lerobot_consortium_window_index(
    data_config: LeRobotConsortiumDataConfig,
    catalog: ConsortiumCatalog,
    episode_keys: Iterable[ConsortiumEpisodeKey],
) -> tuple[ConsortiumWindowRecord, ...]:
    member_lookup = {member.member_id: member for member in catalog.members}
    window_records: list[ConsortiumWindowRecord] = []
    required_span = (data_config.num_frames - 1) * data_config.frame_stride + data_config.action_schema.action_horizon
    for episode_key in episode_keys:
        member = member_lookup[episode_key.member_id]
        episode_record = next(
            episode for episode in member.episodes if episode.episode_index == episode_key.episode_index
        )
        max_start = episode_record.length - required_span
        if max_start < 0:
            continue
        channel_selections = _resolve_channel_selections(data_config, member)
        if data_config.view_packing_mode == ConsortiumViewPackingMode.MULTICAM_AS_FRAMES:
            for selection in channel_selections:
                for start in range(0, max_start + 1, data_config.sample_stride):
                    window_records.append(
                        ConsortiumWindowRecord(
                            member_id=member.member_id,
                            repo_id=member.repo_id,
                            episode_index=episode_key.episode_index,
                            observation_start=start,
                            source_camera_name=selection.source_name,
                            channel_selections=(selection,),
                        )
                    )
        else:
            for start in range(0, max_start + 1, data_config.sample_stride):
                window_records.append(
                    ConsortiumWindowRecord(
                        member_id=member.member_id,
                        repo_id=member.repo_id,
                        episode_index=episode_key.episode_index,
                        observation_start=start,
                        source_camera_name=None,
                        channel_selections=channel_selections,
                    )
                )
    return tuple(window_records)


class ConsortiumTrainSampler(UnpaddedEpochOrderDistributedSampler):
    """Deterministic train sampler for consortium datasets."""

    def __init__(
        self,
        dataset: "LeRobotConsortiumWindowDataset",
        *,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        super().__init__(
            dataset,
            world_size=world_size,
            rank=rank,
            empty_dataset_message=None,
        )


class LeRobotConsortiumWindowDataset(Dataset[WAMSample]):
    """Windowed reader for a configurable consortium of LeRobot-format datasets."""

    def __init__(
        self,
        *,
        data_config: LeRobotConsortiumDataConfig,
        catalog: ConsortiumCatalog,
        window_index: tuple[ConsortiumWindowRecord, ...],
        split_name: str,
        split_audit_payload: dict[str, Any],
    ) -> None:
        self.data_config = data_config
        self.catalog = catalog
        self.sample_index = list(window_index)
        self.split_name = split_name
        self.split_audit_payload = split_audit_payload
        self._resolver = ConsortiumSourceResolver(data_config)
        self._member_lookup = {member.member_id: member for member in catalog.members}
        self._episode_cache: OrderedDict[tuple[str, int], list[dict[str, Any]]] = OrderedDict()
        self._member_sample_indices: dict[str, tuple[int, ...]] = defaultdict(tuple)
        grouped_indices: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(self.sample_index):
            grouped_indices[record.member_id].append(index)
        self._member_sample_indices = {
            member_id: tuple(indices)
            for member_id, indices in grouped_indices.items()
        }
        self._epoch_order_plan = ConsortiumEpochOrderPlan.from_member_indices(
            member_indices=self._member_sample_indices,
            member_weights={
                _resolve_member_id(
                    explicit_member_id=member.member_id,
                    repo_id=member.repo_id,
                    local_root=member.local_root,
                ): (
                    member.sampling_weight
                    if member.sampling_weight is not None
                    else 1.0
                )
                for member in self.data_config.consortium_members
                if member.enabled
            },
            random_mode=self.data_config.random_mode,
            weight_mode=self.data_config.weight_mode,
            sampling_seed=self.data_config.sampling_seed,
        )
        self.audit_payload = self._build_audit_payload()

    def __len__(self) -> int:
        return len(self.sample_index)

    def build_train_sampler(self, *, world_size: int = 1, rank: int = 0) -> ConsortiumTrainSampler:
        return ConsortiumTrainSampler(self, world_size=world_size, rank=rank)

    @property
    def epoch_order_plan(self) -> ConsortiumEpochOrderPlan:
        """Return the immutable source-balancing plan for this split."""

        return self._epoch_order_plan

    def build_epoch_index_order(self, *, epoch: int) -> list[int]:
        return list(self._epoch_order_plan.build_epoch_index_order(epoch=epoch))

    def write_audit_artifacts(self, output_dir: str | Path) -> None:
        output_root = Path(output_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / f"{self.split_name}_audit.json").write_text(
            json.dumps(self.audit_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _build_audit_payload(self) -> dict[str, Any]:
        member_lookup = {member.member_id: member for member in self.catalog.members}
        channel_mappings: dict[str, list[dict[str, Any]]] = {}
        for record in self.sample_index:
            if record.member_id in channel_mappings:
                continue
            channel_mappings[record.member_id] = [asdict(item) for item in record.channel_selections]
        return {
            "dataset_type": self.data_config.dataset_type,
            "split": self.split_name,
            "config": serialize_enum_values(self.data_config),
            "members": [
                {
                    "member_id": member.member_id,
                    "repo_id": member.repo_id,
                    "local_root": member.local_root,
                    "source_group": member.source_group,
                    "observation_fps": member.observation_fps,
                    "action_fps": member.action_fps,
                    "visual_channels": [asdict(channel) for channel in member.visual_channels],
                    "window_count": len(self._member_sample_indices.get(member.member_id, ())),
                    "resolved_channel_mappings": channel_mappings.get(member.member_id, []),
                }
                for member in member_lookup.values()
            ],
            "split_resolution": self.split_audit_payload,
            "sampling": {
                "random_mode": self.data_config.random_mode,
                "weight_mode": self.data_config.weight_mode,
                "sampling_seed": self.data_config.sampling_seed,
                "epoch0_order": self.build_epoch_index_order(epoch=0),
            },
        }

    def __getitem__(self, index: int) -> WAMSample:
        window = self.sample_index[index]
        member = self._member_lookup[window.member_id]
        rows = self._load_episode_rows(member, episode_index=window.episode_index)

        observation_rows = [
            rows[window.observation_start + offset * self.data_config.frame_stride]
            for offset in range(self.data_config.num_frames)
        ]
        anchor_frame_index = window.observation_start + (self.data_config.num_frames - 1) * self.data_config.frame_stride
        action_rows = rows[anchor_frame_index : anchor_frame_index + self.data_config.action_schema.action_horizon]
        target_state_rows = rows[anchor_frame_index : anchor_frame_index + self.data_config.action_schema.action_horizon]
        state_start = max(0, anchor_frame_index - self.data_config.action_schema.state_horizon + 1)
        state_rows = rows[state_start : anchor_frame_index + 1]

        views = {
            selection.target_slot: self._build_view_sequence(
                observation_rows=observation_rows,
                selection=selection,
            )
            for selection in window.channel_selections
        }
        actions, action_mask, action_metadata = self._build_action_targets(
            action_rows=action_rows,
            target_state_rows=target_state_rows,
        )
        state_source_key = self.data_config.action_target.pose_source_key
        state, state_mask = self._extract_sequence(
            rows=state_rows,
            key=state_source_key,
            target_dim=self.data_config.action_schema.state_dim,
            target_length=self.data_config.action_schema.state_horizon,
            left_pad=True,
        )

        task_index = int(observation_rows[-1].get("task_index", 0))
        task_text = member.tasks_by_index.get(task_index)
        return WAMSample(
            views=views,
            actions=actions,
            action_mask=action_mask,
            state=state,
            state_mask=state_mask,
            task_text=task_text,
            metadata={
                "dataset_type": self.data_config.dataset_type,
                "member_id": member.member_id,
                "repo_id": member.repo_id,
                "local_root": member.local_root,
                "source_group": member.source_group,
                "episode_index": window.episode_index,
                "observation_start": window.observation_start,
                "source_camera_name": window.source_camera_name,
                "anchor_frame_index": anchor_frame_index,
                "observation_frame_indices": [int(row["frame_index"]) for row in observation_rows],
                "action_frame_indices": [int(row["frame_index"]) for row in action_rows],
                "target_state_frame_indices": [int(row["frame_index"]) for row in target_state_rows],
                "observation_fps": member.observation_fps,
                "action_fps": member.action_fps,
                "view_packing_mode": self.data_config.view_packing_mode,
                "frame_packing_order": self.data_config.frame_packing_order,
                "resolved_channel_slots": {
                    selection.target_slot: selection.source_name
                    for selection in window.channel_selections
                },
                "state_source_key": state_source_key,
                "action_representation": self.data_config.action_target.representation,
                **action_metadata,
            },
        )

    def _load_episode_rows(self, member: ConsortiumMemberContract, *, episode_index: int) -> list[dict[str, Any]]:
        cache_key = (member.member_id, episode_index)
        if cache_key in self._episode_cache:
            self._episode_cache.move_to_end(cache_key)
            return self._episode_cache[cache_key]
        source = ConsortiumSourceSpec(member_id=member.member_id, repo_id=member.repo_id, local_root=member.local_root)
        relative_path = member.data_path_template.format(
            episode_chunk=episode_index // member.chunk_size,
            episode_index=episode_index,
        )
        rows = self._resolver.read_parquet_rows(source=source, relative_path=relative_path)
        self._episode_cache[cache_key] = rows
        while len(self._episode_cache) > self.data_config.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return rows

    def _build_view_sequence(
        self,
        *,
        observation_rows: list[dict[str, Any]],
        selection: ConsortiumChannelSelection,
    ) -> torch.Tensor:
        if selection.source_name is None:
            if self.data_config.missing_channel_policy == ConsortiumMissingChannelPolicy.ERROR:
                raise KeyError(f"Missing required consortium slot '{selection.target_slot}'.")
            placement = next(
                (view for view in self.data_config.view_layout if view.source_name == selection.target_slot),
                None,
            )
            if placement is None:
                raise ValueError(
                    f"Missing view layout configuration for consortium slot '{selection.target_slot}'."
                )
            return torch.zeros(
                self.data_config.num_frames,
                placement.height,
                placement.width,
                3,
                dtype=torch.uint8,
            )
        frames = [self._decode_image_value(row[selection.source_name]) for row in observation_rows]
        return torch.stack(frames, dim=0)

    def _decode_image_value(self, raw_value: Any) -> torch.Tensor:
        image_bytes: bytes | None = None
        if isinstance(raw_value, dict) and "bytes" in raw_value:
            image_bytes = raw_value["bytes"]
        elif isinstance(raw_value, (bytes, bytearray)):
            image_bytes = bytes(raw_value)
        if image_bytes is None:
            raise ValueError("Expected LeRobot consortium image value to provide inline image bytes.")
        with Image.open(BytesIO(image_bytes)) as image:
            rgb = image.convert("RGB")
            tensor = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
            return tensor.reshape(rgb.height, rgb.width, 3)

    def _build_action_targets(
        self,
        *,
        action_rows: list[dict[str, Any]],
        target_state_rows: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        return build_row_action_targets(
            data_config=self.data_config,
            action_rows=action_rows,
            target_state_rows=target_state_rows,
            extract_sequence=self._extract_sequence,
            reference_source_subject="Consortium relative pose targets",
            relative_sequence_name="relative_pose_targets",
            include_pose_dimension_context=False,
        )

    def _extract_sequence(
        self,
        *,
        rows: list[dict[str, Any]],
        key: str,
        target_dim: int,
        target_length: int,
        left_pad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not rows:
            raise ValueError(f"Cannot extract sequence for key '{key}' from an empty row slice.")
        sequence = torch.stack(
            [torch.tensor(row[resolve_row_key(row, key)], dtype=torch.float32) for row in rows],
            dim=0,
        )
        return pack_temporal_sequence(
            sequence=sequence,
            target_dim=target_dim,
            target_length=target_length,
            left_pad=left_pad,
            sequence_name=key,
        )


def build_lerobot_consortium_train_val_datasets(
    data_config: DataConfig,
    *,
    catalog: ConsortiumCatalog | None = None,
    split: ConsortiumResolvedSplit | None = None,
) -> tuple[LeRobotConsortiumWindowDataset, LeRobotConsortiumWindowDataset]:
    if not isinstance(data_config, LeRobotConsortiumDataConfig):
        raise TypeError("Consortium dataset builder requires LeRobotConsortiumDataConfig.")
    resolved_catalog = catalog or build_lerobot_consortium_catalog(data_config)
    resolved_split = split or resolve_lerobot_consortium_train_val_split(data_config, resolved_catalog)
    train_index = build_lerobot_consortium_window_index(data_config, resolved_catalog, resolved_split.train_episodes)
    val_index = build_lerobot_consortium_window_index(data_config, resolved_catalog, resolved_split.val_episodes)
    return (
        LeRobotConsortiumWindowDataset(
            data_config=data_config,
            catalog=resolved_catalog,
            window_index=train_index,
            split_name="train",
            split_audit_payload=resolved_split.audit_payload,
        ),
        LeRobotConsortiumWindowDataset(
            data_config=data_config,
            catalog=resolved_catalog,
            window_index=val_index,
            split_name="val",
            split_audit_payload=resolved_split.audit_payload,
        ),
    )
