from __future__ import annotations

from pathlib import Path
from typing import Any

from open_wam.configs import DatasetPreflightKind
from open_wam.utils.latent_filenames import match_latent_window_filename

from .types import CheckpointSpec, DatasetPreflight, DatasetProfile


def preflight_dataset(profile: DatasetProfile) -> DatasetPreflight:
    """Check local latent data availability using the same complete-camera rule as the loader."""

    missing: list[dict[str, str]] = []
    warnings: list[str] = []
    root = profile.root
    if not root.is_dir():
        missing.append({"kind": "data.local_root", "path": str(root)})
        return DatasetPreflight(
            key=profile.name,
            data_root=str(root),
            repo_count=0,
            latent_cameras=profile.latent_cameras,
            latent_counts={},
            missing=tuple(missing),
            warnings=tuple(warnings),
        )

    repo_roots = discover_lerobot_roots(root)
    if not repo_roots:
        missing.append({"kind": "data.meta.info_json", "path": str(root)})
    latent_counts = count_latent_windows(repo_roots, profile.latent_cameras)
    if (
        profile.preflight == DatasetPreflightKind.LOCAL_LATENT
        and latent_counts["complete_multicamera_windows"] <= 0
    ):
        missing.append(
            {
                "kind": "data.latent_windows",
                "path": str(root),
                "detail": f"no complete .pth windows for latent cameras {profile.latent_cameras!r}",
            }
        )
    elif latent_counts["complete_multicamera_windows"] < latent_counts["total_primary_windows"]:
        warnings.append(
            f"Only {latent_counts['complete_multicamera_windows']} of "
            f"{latent_counts['total_primary_windows']} primary latent windows have every configured camera; "
            "incomplete windows are skipped by the local latent scanner."
        )
    invalid_filename_total = sum(int(count) for count in latent_counts["invalid_filename_counts"].values())
    if invalid_filename_total:
        warnings.append(
            f"Ignored {invalid_filename_total} latent file(s) whose names do not match "
            "`episode_<six_digit_episode>_<start>_<end>.pth`; these are skipped by the local latent scanner."
        )
    return DatasetPreflight(
        key=profile.name,
        data_root=str(root),
        repo_count=len(repo_roots),
        latent_cameras=profile.latent_cameras,
        latent_counts=latent_counts,
        missing=tuple(missing),
        warnings=tuple(warnings),
    )


def preflight_checkpoint(spec: CheckpointSpec) -> tuple[dict[str, str], ...]:
    missing: list[dict[str, str]] = []
    transformer = spec.transformer_subdir
    if not transformer.is_dir():
        missing.append({"kind": "transformer_subdir", "path": str(transformer)})
        return tuple(missing)
    if not (transformer / "config.json").is_file():
        missing.append({"kind": "transformer_subdir.config.json", "path": str(transformer / "config.json")})
    if not has_transformer_weights(transformer):
        missing.append({"kind": "transformer_subdir.weights", "path": str(transformer)})
    return tuple(missing)


def discover_lerobot_roots(root: Path) -> list[Path]:
    if (root / "meta" / "info.json").is_file():
        return [root]
    return sorted(path.parent.parent for path in root.rglob("meta/info.json"))


def count_latent_windows(repo_roots: list[Path], latent_cameras: tuple[str, ...]) -> dict[str, Any]:
    camera_counts = {camera: 0 for camera in latent_cameras}
    invalid_filename_counts = {camera: 0 for camera in latent_cameras}
    complete_windows = 0
    for repo_root in repo_roots:
        latent_root = repo_root / "latents"
        camera_files: dict[str, set[tuple[str, str]]] = {}
        for camera in latent_cameras:
            files: set[tuple[str, str]] = set()
            for path in latent_root.glob(f"chunk-*/{camera}/episode_*.pth"):
                if match_latent_window_filename(path.name) is None:
                    invalid_filename_counts[camera] += 1
                    continue
                files.add((path.parents[1].name, path.name))
            camera_files[camera] = files
            camera_counts[camera] += len(files)
        if camera_files:
            complete_windows += len(set.intersection(*camera_files.values()))
    return {
        "total_primary_windows": camera_counts[latent_cameras[0]] if latent_cameras else 0,
        "complete_multicamera_windows": complete_windows,
        "camera_counts": camera_counts,
        "invalid_filename_counts": invalid_filename_counts,
    }


def has_transformer_weights(transformer: Path) -> bool:
    if (transformer / "diffusion_pytorch_model.safetensors").is_file():
        return True
    if (transformer / "diffusion_pytorch_model.safetensors.index.json").is_file():
        return True
    return any(transformer.glob("diffusion_pytorch_model-*.safetensors"))
