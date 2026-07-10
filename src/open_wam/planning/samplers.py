from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import ActionChunk, PlanningContext


@dataclass(frozen=True)
class GaussianActionSampler:
    """Small dependency-free sampler for dry-runs and control-flow tests."""

    action_dim: int
    mean: float = 0.0
    std: float = 0.2
    clip: float = 1.0

    def sample_action_chunks(
        self,
        context: PlanningContext,
        *,
        num_samples: int,
        chunk_action_steps: int,
        temperature: float,
        seed: int | None = None,
    ) -> list[ActionChunk]:
        del context
        rng = np.random.default_rng(seed)
        scale = float(self.std) * max(float(temperature), 1e-6)
        chunks: list[ActionChunk] = []
        for sample_index in range(int(num_samples)):
            actions = rng.normal(
                loc=float(self.mean),
                scale=scale,
                size=(int(chunk_action_steps), int(self.action_dim)),
            ).astype(np.float32)
            actions = np.clip(actions, -float(self.clip), float(self.clip))
            chunks.append(ActionChunk(actions=actions, metadata={"sample_index": sample_index, "sampler": "gaussian"}))
        return chunks
