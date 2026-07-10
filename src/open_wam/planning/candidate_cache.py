from __future__ import annotations

"""Persistent policy-candidate cache for planning diagnostics."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .contracts import ActionChunk, PlanningContext, PolicyActionSampler


@dataclass(frozen=True)
class CandidateCacheConfig:
    """Configuration for policy action candidate caching."""

    cache_dir: Path
    namespace: str = "default"
    sample_count: int | None = None

    def __post_init__(self) -> None:
        if self.sample_count is not None and self.sample_count <= 0:
            raise ValueError("sample_count must be positive when provided.")


class CachedPolicyActionSampler:
    """Wrap a policy sampler and persist sampled action chunks by context.

    The cache stores only policy action proposals. Dynamics predictions and
    evaluator scores are intentionally left uncached so different FDM backends
    can compare against the same candidate actions without sharing predictions.
    """

    def __init__(self, policy: PolicyActionSampler, config: CandidateCacheConfig) -> None:
        self.policy = policy
        self.config = config
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def sample_action_chunks(
        self,
        context: PlanningContext,
        *,
        num_samples: int,
        chunk_action_steps: int,
        temperature: float,
        seed: int | None = None,
    ) -> Sequence[ActionChunk]:
        if num_samples < 0:
            raise ValueError("num_samples must be non-negative.")
        if num_samples == 0:
            return ()
        target_count = max(int(num_samples), int(self.config.sample_count or num_samples))
        cache_path = self._cache_path(
            context,
            target_count=target_count,
            chunk_action_steps=int(chunk_action_steps),
            temperature=float(temperature),
            seed=None if seed is None else int(seed),
        )
        cached = self._try_load(cache_path, requested_count=int(num_samples))
        if cached is not None:
            return cached

        chunks = tuple(
            self.policy.sample_action_chunks(
                context,
                num_samples=target_count,
                chunk_action_steps=int(chunk_action_steps),
                temperature=float(temperature),
                seed=seed,
            )
        )
        if len(chunks) < target_count:
            raise RuntimeError(
                "Wrapped policy returned fewer cached candidates than requested: "
                f"requested {target_count}, got {len(chunks)}."
            )
        self._save(
            cache_path,
            chunks=chunks[:target_count],
            target_count=target_count,
            requested_count=int(num_samples),
            chunk_action_steps=int(chunk_action_steps),
            temperature=float(temperature),
            seed=None if seed is None else int(seed),
        )
        return self._with_cache_metadata(
            chunks[: int(num_samples)],
            cache_path=cache_path,
            cache_hit=False,
            cache_sample_count=target_count,
        )

    def reset(self) -> Any:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            return reset()
        return None

    def snapshot_state(self) -> Any:
        snapshot = getattr(self.policy, "snapshot_state", None)
        if callable(snapshot):
            return snapshot()
        return None

    def restore_state(self, state: Any) -> Any:
        restore = getattr(self.policy, "restore_state", None)
        if callable(restore):
            return restore(state)
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.policy, name)

    def _cache_path(
        self,
        context: PlanningContext,
        *,
        target_count: int,
        chunk_action_steps: int,
        temperature: float,
        seed: int | None,
    ) -> Path:
        digest = _policy_query_digest(
            context,
            namespace=self.config.namespace,
            target_count=target_count,
            chunk_action_steps=chunk_action_steps,
            temperature=temperature,
            seed=seed,
        )
        return self.config.cache_dir / f"{digest}.npz"

    def _try_load(self, cache_path: Path, *, requested_count: int) -> Sequence[ActionChunk] | None:
        if not cache_path.is_file():
            return None
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                actions = np.asarray(data["actions"], dtype=np.float32)
                logprobs = np.asarray(data["logprobs"], dtype=np.float32)
                sampler_scores = np.asarray(data["sampler_scores"], dtype=np.float32)
                metadata_json = str(np.asarray(data["metadata_json"]).item())
        except Exception:
            return None
        if actions.ndim != 3 or actions.shape[0] < requested_count:
            return None
        try:
            metadata = json.loads(metadata_json)
        except Exception:
            metadata = {}
        chunks: list[ActionChunk] = []
        for index in range(requested_count):
            logprob = None if np.isnan(logprobs[index]) else float(logprobs[index])
            sampler_score = None if np.isnan(sampler_scores[index]) else float(sampler_scores[index])
            chunk_metadata = {
                "candidate_cache_hit": True,
                "candidate_cache_path": str(cache_path),
                "candidate_cache_index": int(index),
                "candidate_cache_sample_count": int(actions.shape[0]),
                "candidate_cache_namespace": self.config.namespace,
                "candidate_cache_metadata": metadata,
            }
            chunks.append(
                ActionChunk(
                    actions=actions[index],
                    logprob=logprob,
                    sampler_score=sampler_score,
                    metadata=chunk_metadata,
                )
            )
        return tuple(chunks)

    def _save(
        self,
        cache_path: Path,
        *,
        chunks: Sequence[ActionChunk],
        target_count: int,
        requested_count: int,
        chunk_action_steps: int,
        temperature: float,
        seed: int | None,
    ) -> None:
        actions = np.stack([np.asarray(chunk.actions, dtype=np.float32) for chunk in chunks], axis=0)
        logprobs = np.asarray(
            [np.nan if chunk.logprob is None else float(chunk.logprob) for chunk in chunks],
            dtype=np.float32,
        )
        sampler_scores = np.asarray(
            [np.nan if chunk.sampler_score is None else float(chunk.sampler_score) for chunk in chunks],
            dtype=np.float32,
        )
        metadata = {
            "namespace": self.config.namespace,
            "target_count": int(target_count),
            "requested_count_at_write": int(requested_count),
            "chunk_action_steps": int(chunk_action_steps),
            "temperature": float(temperature),
            "seed": seed,
        }
        tmp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        try:
            np.savez_compressed(
                tmp_path,
                actions=actions,
                logprobs=logprobs,
                sampler_scores=sampler_scores,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            saved_path = tmp_path if tmp_path.suffix == ".npz" else tmp_path.with_suffix(tmp_path.suffix + ".npz")
            os.replace(saved_path, cache_path)
        finally:
            for candidate in (tmp_path, tmp_path.with_suffix(tmp_path.suffix + ".npz")):
                try:
                    if candidate.exists() and candidate != cache_path:
                        candidate.unlink()
                except OSError:
                    pass

    def _with_cache_metadata(
        self,
        chunks: Sequence[ActionChunk],
        *,
        cache_path: Path,
        cache_hit: bool,
        cache_sample_count: int,
    ) -> tuple[ActionChunk, ...]:
        out: list[ActionChunk] = []
        for index, chunk in enumerate(chunks):
            metadata = dict(chunk.metadata or {})
            metadata.update(
                {
                    "candidate_cache_hit": bool(cache_hit),
                    "candidate_cache_path": str(cache_path),
                    "candidate_cache_index": int(index),
                    "candidate_cache_sample_count": int(cache_sample_count),
                    "candidate_cache_namespace": self.config.namespace,
                }
            )
            out.append(
                ActionChunk(
                    actions=chunk.actions,
                    logprob=chunk.logprob,
                    sampler_score=chunk.sampler_score,
                    metadata=metadata,
                )
            )
        return tuple(out)


def _policy_query_digest(
    context: PlanningContext,
    *,
    namespace: str,
    target_count: int,
    chunk_action_steps: int,
    temperature: float,
    seed: int | None,
) -> str:
    hasher = hashlib.sha256()
    payload = {
        "version": 1,
        "namespace": str(namespace),
        "target_count": int(target_count),
        "chunk_action_steps": int(chunk_action_steps),
        "temperature": float(temperature),
        "seed": seed,
        "task_text": context.task_text,
        "state": _array_signature(context.state),
        "views": {
            str(key): _array_signature(value)
            for key, value in sorted(context.views.items(), key=lambda item: str(item[0]))
        },
    }
    hasher.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return hasher.hexdigest()


def _array_signature(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256(array.view(np.uint8)).hexdigest()
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": digest,
    }
