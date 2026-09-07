"""LeRobot consortium catalog discovery and metadata snapshot contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any
import warnings

from open_wam.configs import LeRobotConsortiumDataConfig
from open_wam.contracts.paths import find_repo_root

from .lerobot_consortium_contracts import (
    build_lerobot_consortium_contract_catalog_from_inventory_rows,
    write_lerobot_consortium_contract_catalog,
)
from .lerobot_consortium_index import build_lerobot_consortium_inventory
from .lerobot_consortium_inventory_contracts import (
    LeRobotConsortiumInventoryRow,
    LeRobotConsortiumRepoTarget,
)
from .lerobot_consortium_inventory_io import (
    load_lerobot_consortium_inventory_rows,
    write_lerobot_consortium_inventory_csv,
    write_lerobot_consortium_inventory_markdown,
)
from .lerobot_consortium_targets import (
    infer_lerobot_consortium_source_group,
    load_lerobot_consortium_repo_targets,
    write_lerobot_consortium_repo_targets,
)
from .lerobot_consortium_planning import _resolve_member_id
from .lerobot_consortium_storage import (
    ConsortiumSourceResolver,
    ConsortiumSourceSpec,
    discover_local_lerobot_consortium_members,
)


_CONSORTIUM_INDEX_ROOT_ENV = "OPEN_WAM_CONSORTIUM_INDEX_ROOT"


def _resolve_consortium_index_root() -> tuple[Path, bool]:
    """Return the active snapshot root and whether refresh may write to it."""

    explicit_root = os.environ.get(_CONSORTIUM_INDEX_ROOT_ENV)
    if explicit_root:
        return Path(explicit_root).expanduser().resolve(), True

    source_root = find_repo_root(Path(__file__))
    package_source = source_root / "src" / "open_wam"
    module_path = Path(__file__).resolve()
    if module_path.is_relative_to(package_source.resolve()):
        return source_root / "notes" / "index", True

    package_resources = module_path.parents[1] / "resources" / "consortium"
    return package_resources, False


_CONSORTIUM_INDEX_ROOT, _CONSORTIUM_INDEX_MUTABLE = _resolve_consortium_index_root()
_CONSORTIUM_INDEX_REPO_IDS_PATH = _CONSORTIUM_INDEX_ROOT / "lerobot_consortium_hf_repo_ids.txt"
_CONSORTIUM_INDEX_INVENTORY_CSV_PATH = _CONSORTIUM_INDEX_ROOT / "lerobot_consortium_hf_dataset_inventory.csv"
_CONSORTIUM_INDEX_INVENTORY_MD_PATH = _CONSORTIUM_INDEX_ROOT / "lerobot_consortium_hf_dataset_inventory.md"
_CONSORTIUM_INDEX_CONTRACTS_JSON_PATH = _CONSORTIUM_INDEX_ROOT / "lerobot_consortium_hf_dataset_contracts.json"
_CONSORTIUM_INDEX_SANITY_CACHE: set[tuple[str, ...]] = set()


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
    repo_ids = sorted(
        {
            source.repo_id
            for source in _resolve_member_sources(data_config)
            if source.repo_id is not None
        }
    )
    return tuple(repo_ids)


def _configured_remote_repo_targets(
    data_config: LeRobotConsortiumDataConfig,
) -> tuple[LeRobotConsortiumRepoTarget, ...]:
    deduped: dict[str, LeRobotConsortiumRepoTarget] = {}
    for source in _resolve_member_sources(data_config):
        if source.repo_id is None:
            continue
        source_group = _resolve_source_group(
            data_config,
            member_id=source.member_id,
        ) or infer_lerobot_consortium_source_group(
            source.repo_id, default_source_group="manual"
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
    if not _CONSORTIUM_INDEX_MUTABLE:
        raise RuntimeError(
            "The installed consortium snapshot is read-only. Set "
            f"{_CONSORTIUM_INDEX_ROOT_ENV} to a writable snapshot directory "
            "before refreshing it."
        )
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
    merged_repo_targets = tuple(
        sorted(
            target_by_repo_id.values(),
            key=lambda target: (target.source_group, target.repo_id),
        )
    )
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
        repo_list_ids = tuple(
            target.repo_id
            for target in load_lerobot_consortium_repo_targets(
                _CONSORTIUM_INDEX_REPO_IDS_PATH
            )
        )

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

    refresh_help = (
        "Refresh command: PYTHONPATH=src python "
        "scripts/build_lerobot_consortium_index.py "
        "--repo-list notes/index/lerobot_consortium_hf_repo_ids.txt"
        if _CONSORTIUM_INDEX_MUTABLE
        else (
            "The installed snapshot is read-only. Upgrade OpenWAM for a newer "
            f"snapshot, or set {_CONSORTIUM_INDEX_ROOT_ENV} to a writable "
            "snapshot directory and refresh it from a source checkout."
        )
    )
    message = (
        "Detected discrepancy between the configured LeRobot consortium repo "
        "ids and the local parsed inventory/contracts. "
        "This usually means the repo-id list, inventory CSV, and contract JSON are out of sync.\n"
        + "\n".join(f"- {issue}" for issue in issues)
        + f"\n{refresh_help}"
    )

    if _CONSORTIUM_INDEX_MUTABLE and _consortium_index_prompt_available():
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
