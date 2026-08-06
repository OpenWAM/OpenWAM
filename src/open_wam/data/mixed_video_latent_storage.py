"""Mixed-video latent sidecar resolution, validation, and bounded caching."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch

from open_wam.artifacts import load_tensor_artifact
from open_wam.configs import MixedVideoDataConfig

from .mixed_video_catalog_contracts import MixedVideoStreamRecord


__all__ = [
    "MixedVideoLatentRepository",
    "load_mixed_video_latent_tensor",
    "mixed_video_latent_cache_key",
    "resolve_mixed_video_latent_path",
]


def resolve_mixed_video_latent_path(
    stream: MixedVideoStreamRecord,
    *,
    cache_dir: str | None,
) -> Path:
    """Resolve one local or Hugging Face latent sidecar path."""

    if stream.latent_path is not None:
        if not stream.latent_path.exists():
            raise FileNotFoundError(
                f"Missing mixed-video latent file for source={stream.source_id}, "
                f"episode={stream.episode_index}, "
                f"stream={stream.stream_key}: {stream.latent_path}"
            )
        return stream.latent_path
    if stream.repo_id is None or stream.latent_shard_relative_path is None:
        raise FileNotFoundError(
            "Mixed-video stream has neither latent_path nor HF latent shard "
            f"path: source={stream.source_id}, "
            f"episode={stream.episode_index}, stream={stream.stream_key}."
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required for remote mixed-video latent manifests."
        ) from exc
    return Path(
        hf_hub_download(
            repo_id=stream.repo_id,
            filename=stream.latent_shard_relative_path,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
    )


def load_mixed_video_latent_tensor(
    path: Path,
    *,
    key: str,
) -> torch.Tensor:
    """Load one `[C,T,H,W]` sidecar as contiguous float32."""

    payload = load_tensor_artifact(path)
    if isinstance(payload, torch.Tensor):
        tensor = payload
    elif isinstance(payload, dict) and key in payload:
        tensor = payload[key]
    else:
        raise ValueError(
            f"Expected latent tensor or key {key!r} in latent payload at {path}."
        )
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
        raise ValueError(
            f"Expected latent tensor [C,T,H,W] at {path}, got {type(tensor)!r}."
        )
    return tensor.to(dtype=torch.float32).contiguous()


def mixed_video_latent_cache_key(
    stream: MixedVideoStreamRecord,
) -> str:
    """Return the stable physical-sidecar component of a cache key."""

    if stream.latent_path is not None:
        return str(stream.latent_path)
    return f"{stream.repo_id}:{stream.latent_shard_relative_path}"


class MixedVideoLatentRepository:
    """Bounded latent-sidecar access for one mixed-video data config."""

    def __init__(self, data_config: MixedVideoDataConfig) -> None:
        self.data_config = data_config
        self.cache: OrderedDict[
            tuple[str, str, str],
            torch.Tensor,
        ] = OrderedDict()

    @property
    def cache_capacity(self) -> int:
        return max(
            1,
            int(self.data_config.episode_cache_size)
            * max(1, len(self.data_config.camera_names)),
        )

    def load(self, stream: MixedVideoStreamRecord) -> torch.Tensor:
        """Load one stream sidecar and update its LRU position."""

        cache_key = (
            stream.source_id,
            mixed_video_latent_cache_key(stream),
            stream.latent_key,
        )
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]
        path = resolve_mixed_video_latent_path(
            stream,
            cache_dir=self.data_config.cache_dir,
        )
        latents = load_mixed_video_latent_tensor(path, key=stream.latent_key)
        self.cache[cache_key] = latents
        while len(self.cache) > self.cache_capacity:
            self.cache.popitem(last=False)
        return latents
