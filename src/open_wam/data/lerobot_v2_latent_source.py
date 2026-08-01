"""Canonical source material for local LeRobot latent sample assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .lerobot_v2_latent_storage import (
    LocalEpisodeWindow,
    LocalLatentRepository,
    LocalRepoBundle,
    assemble_canonical_latents,
)


__all__ = [
    "LocalLatentSampleConditioning",
    "LocalLatentSampleSource",
    "LocalLatentSampleSourceLoader",
]


@dataclass(frozen=True, eq=False)
class LocalLatentSampleConditioning:
    """Task and text conditioning resolved at one selected source frame."""

    task_index: int
    task_text: str | None
    text_context: torch.Tensor | None
    negative_text_context: torch.Tensor | None


@dataclass(frozen=True, eq=False)
class LocalLatentSampleSource:
    """Canonical physical-window inputs shared by local sampling modes."""

    window: LocalEpisodeWindow
    repo_bundle: LocalRepoBundle
    rows: list[dict[str, Any]]
    video_latents: torch.Tensor
    latent_layout_metadata: dict[str, dict[str, int]]
    primary_payload: dict[str, Any]
    condition_latents: torch.Tensor | None
    condition_layout_metadata: dict[str, dict[str, int]]
    raw_frame_ids: list[int]

    def conditioning_for_frame(
        self,
        frame_index: int,
        *,
        empty_text_embedding: torch.Tensor | None,
    ) -> LocalLatentSampleConditioning:
        """Resolve task identity and text tensors at a selected frame."""

        task_index = (
            int(
                self.rows[min(frame_index, len(self.rows) - 1)].get(
                    "task_index",
                    0,
                )
            )
            if self.rows
            else 0
        )
        episode_record = self.repo_bundle.episodes_by_index.get(
            self.window.episode_index
        )
        task_text = self.repo_bundle.metadata.tasks_by_index.get(task_index)
        if (
            task_text is None
            and episode_record is not None
            and episode_record.tasks
        ):
            task_text = episode_record.tasks[0]

        text_context = self.primary_payload.get("text_emb")
        if isinstance(text_context, torch.Tensor):
            text_context = text_context.to(dtype=torch.float32)
        else:
            text_context = None
        negative_text_context = (
            empty_text_embedding.clone()
            if empty_text_embedding is not None
            else None
        )
        return LocalLatentSampleConditioning(
            task_index=task_index,
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
        )


class LocalLatentSampleSourceLoader:
    """Load canonical source material without choosing sampling geometry."""

    def __init__(self, repository: LocalLatentRepository) -> None:
        self.repository = repository
        self.data_config = repository.data_config

    def load(
        self,
        window: LocalEpisodeWindow,
        *,
        include_condition_latents: bool,
    ) -> LocalLatentSampleSource:
        """Load one physical latent window in its canonical view layout."""

        repo_bundle = self.repository.repo_bundles[str(window.repo_root)]
        rows = self.repository.load_episode_rows(
            window.repo_root,
            window.episode_index,
            repo_bundle.metadata,
        )
        if include_condition_latents:
            (
                video_latents,
                latent_layout_metadata,
                primary_payload,
                condition_latents,
                condition_layout_metadata,
            ) = self.repository.load_canonical_window_latents(
                window,
                repo_bundle.metadata,
            )
        else:
            latent_payloads = self.repository.load_window_latents(
                window,
                repo_bundle.metadata,
            )
            video_latents, latent_layout_metadata = assemble_canonical_latents(
                self.data_config,
                latent_payloads,
            )
            assert video_latents is not None
            primary_payload = latent_payloads[
                self.data_config.latent_camera_names[0]
            ]
            condition_latents = None
            condition_layout_metadata = {}

        raw_frame_ids = [
            int(value)
            for value in list(primary_payload.get("frame_ids", []))
        ]
        if not raw_frame_ids:
            raw_frame_ids = list(window.observation_frame_indices)
        return LocalLatentSampleSource(
            window=window,
            repo_bundle=repo_bundle,
            rows=rows,
            video_latents=video_latents,
            latent_layout_metadata=latent_layout_metadata,
            primary_payload=primary_payload,
            condition_latents=condition_latents,
            condition_layout_metadata=condition_layout_metadata,
            raw_frame_ids=raw_frame_ids,
        )
