from __future__ import annotations

from pathlib import Path

from open_wam.configs import DataConfig

from .artifacts import DatasetArtifactKind, DatasetArtifactRequirement
from .lerobot_v2_latent_storage import discover_local_lerobot_repo_bundles
from .replay_status import resolve_replay_status_path


def resolve_local_lerobot_latent_artifacts(
    data_config: DataConfig,
) -> tuple[DatasetArtifactRequirement, ...]:
    """Declare filesystem dependencies already enforced by the local adapter."""

    train_repository_roots = _discover_replay_roots(data_config.local_root)
    requirements = [
        DatasetArtifactRequirement(
            name="training dataset root",
            path=data_config.local_root,
            kind=DatasetArtifactKind.DIRECTORY,
            required=True,
            config_path="data.local_root",
            purpose="local latent datasets require a discoverable LeRobot repository",
        )
    ]
    if data_config.val_local_root:
        requirements.append(
            DatasetArtifactRequirement(
                name="validation dataset root",
                path=data_config.val_local_root,
                kind=DatasetArtifactKind.DIRECTORY,
                required=True,
                config_path="data.val_local_root",
                purpose="the configured validation split uses a separate repository",
            )
        )

    if data_config.empty_text_embedding_path is not None:
        requirements.append(
            DatasetArtifactRequirement(
                name="empty text embedding",
                path=data_config.empty_text_embedding_path,
                kind=DatasetArtifactKind.FILE,
                required=True,
                config_path="data.empty_text_embedding_path",
                purpose="the adapter uses this tensor for dropped text conditioning",
                remediation=(
                    "Generate the embedding or point the config at an existing tensor "
                    "checkpoint"
                ),
            )
        )

    val_required = (
        data_config.require_replay_status
        if data_config.val_require_replay_status is None
        else data_config.val_require_replay_status
    )
    shared_val_requires_status = bool(
        not data_config.val_local_root
        and data_config.val_replay_status_policy is not None
        and val_required
    )
    requirements.extend(
        _resolve_replay_status_artifacts(
            name=(
                "shared replay-status metadata"
                if shared_val_requires_status and not data_config.require_replay_status
                else "training replay-status metadata"
            ),
            dataset_root=data_config.local_root,
            configured_path=data_config.replay_status_path,
            required=bool(
                data_config.require_replay_status or shared_val_requires_status
            ),
            config_path="data.replay_status_path",
            repository_roots=train_repository_roots,
        )
    )

    if data_config.val_local_root:
        val_repository_roots = _discover_replay_roots(data_config.val_local_root)
        val_replay_status_path = data_config.val_replay_status_path
        if (
            val_replay_status_path is None
            and data_config.replay_status_path is not None
        ):
            train_replay_status_path = Path(data_config.replay_status_path).expanduser()
            if not train_replay_status_path.is_absolute():
                val_replay_status_path = data_config.replay_status_path
        requirements.extend(
            _resolve_replay_status_artifacts(
                name="validation replay-status metadata",
                dataset_root=data_config.val_local_root,
                configured_path=val_replay_status_path,
                required=bool(val_required),
                config_path="data.val_replay_status_path",
                repository_roots=val_repository_roots,
            )
        )
    return tuple(requirements)


def _resolve_replay_status_artifacts(
    *,
    name: str,
    dataset_root: str | None,
    configured_path: str | None,
    required: bool,
    config_path: str,
    repository_roots: tuple[Path, ...],
) -> tuple[DatasetArtifactRequirement, ...]:
    if configured_path is None and not required:
        return ()

    configured = None if configured_path is None else Path(configured_path).expanduser()
    if configured is not None and configured.is_absolute():
        paths = (configured,)
    else:
        paths = tuple(
            resolve_replay_status_path(root, configured_path)
            for root in repository_roots
        )
        if not paths:
            paths = (resolve_replay_status_path(dataset_root, configured_path),)

    return tuple(
        DatasetArtifactRequirement(
            name=name,
            path=path,
            kind=DatasetArtifactKind.FILE,
            required=required,
            config_path=config_path,
            purpose="the configured episode split uses simulator replay labels",
            remediation=(
                "Install replay_status.jsonl under the dataset meta directory or "
                "configure an explicit path"
            ),
        )
        for path in paths
    )


def _discover_replay_roots(dataset_root: str | None) -> tuple[Path, ...]:
    if dataset_root is None:
        return ()
    root = Path(dataset_root).expanduser()
    if not root.is_dir():
        return ()
    return tuple(bundle.root for bundle in discover_local_lerobot_repo_bundles(root))


__all__ = ["resolve_local_lerobot_latent_artifacts"]
