"""Deterministic epoch-order planning for LeRobot consortium datasets."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from open_wam.configs import ConsortiumRandomMode, ConsortiumWeightMode

__all__ = ["ConsortiumEpochOrderPlan"]


@dataclass(frozen=True)
class ConsortiumEpochOrderPlan:
    """Immutable source-balancing policy for one consortium sample index."""

    member_indices: tuple[tuple[str, tuple[int, ...]], ...]
    member_weights: tuple[tuple[str, float], ...]
    random_mode: ConsortiumRandomMode
    weight_mode: ConsortiumWeightMode
    sampling_seed: int

    @classmethod
    def from_member_indices(
        cls,
        *,
        member_indices: Mapping[str, tuple[int, ...]],
        member_weights: Mapping[str, float],
        random_mode: ConsortiumRandomMode,
        weight_mode: ConsortiumWeightMode,
        sampling_seed: int,
    ) -> ConsortiumEpochOrderPlan:
        """Freeze dataset membership and balancing controls into a plan."""

        return cls(
            member_indices=tuple(
                (str(member_id), tuple(int(index) for index in indices))
                for member_id, indices in member_indices.items()
            ),
            member_weights=tuple(
                (str(member_id), float(weight))
                for member_id, weight in member_weights.items()
            ),
            random_mode=random_mode,
            weight_mode=weight_mode,
            sampling_seed=int(sampling_seed),
        )

    def build_epoch_index_order(self, *, epoch: int) -> tuple[int, ...]:
        """Build one deterministic global order for an epoch."""

        dataset_indices = dict(self.member_indices)
        target_counts = _resolve_member_target_counts(
            member_indices=dataset_indices,
            member_weights=dict(self.member_weights),
            weight_mode=self.weight_mode,
        )
        per_member_sequences: dict[str, list[int]] = {}
        for member_id, indices in dataset_indices.items():
            base = list(indices)
            if self.random_mode == ConsortiumRandomMode.WITHIN_DATASET:
                seed = _stable_int_seed(self.sampling_seed, epoch, member_id)
                base = _seeded_shuffle(base, seed)
            elif self.random_mode == ConsortiumRandomMode.TRAJECTORY_GLOBAL:
                seed = _stable_int_seed(
                    self.sampling_seed,
                    epoch,
                    member_id,
                    "global",
                )
                base = _seeded_shuffle(base, seed)
            per_member_sequences[member_id] = _cycle_take(
                tuple(base),
                target_counts.get(member_id, 0),
            )

        if self.random_mode == ConsortiumRandomMode.TRAJECTORY_GLOBAL:
            combined = [
                index
                for member_id in sorted(per_member_sequences)
                for index in per_member_sequences[member_id]
            ]
            return tuple(
                _seeded_shuffle(
                    combined,
                    _stable_int_seed(self.sampling_seed, epoch, "global"),
                )
            )

        positions = {member_id: 0 for member_id in per_member_sequences}
        order: list[int] = []
        for member_id in _build_weighted_round_robin_schedule(target_counts):
            sequence = per_member_sequences[member_id]
            position = positions[member_id]
            if position < len(sequence):
                order.append(sequence[position])
                positions[member_id] += 1
        return tuple(order)


def _resolve_member_target_counts(
    *,
    member_indices: Mapping[str, tuple[int, ...]],
    member_weights: Mapping[str, float],
    weight_mode: ConsortiumWeightMode,
) -> dict[str, int]:
    total_samples = sum(len(indices) for indices in member_indices.values())
    if weight_mode == ConsortiumWeightMode.PROPORTIONAL_TO_SIZE:
        raw_weights = {
            key: float(len(indices)) for key, indices in member_indices.items()
        }
    elif weight_mode == ConsortiumWeightMode.PROPORTIONAL_THEN_MANUAL_SCALE:
        raw_weights = {
            key: float(len(indices)) * member_weights.get(key, 1.0)
            for key, indices in member_indices.items()
        }
    else:
        raw_weights = {
            key: member_weights.get(key, 1.0) for key in member_indices
        }
    return _largest_remainder_counts(raw_weights, total_count=total_samples)


def _largest_remainder_counts(
    raw_weights: Mapping[str, float],
    *,
    total_count: int,
) -> dict[str, int]:
    if total_count <= 0:
        return {key: 0 for key in raw_weights}
    positive_items = [
        (key, max(0.0, value)) for key, value in raw_weights.items()
    ]
    weight_sum = sum(value for _, value in positive_items)
    if weight_sum <= 0.0:
        raise ValueError(
            "Consortium weight resolution requires at least one positive "
            "dataset weight."
        )
    floor_counts: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    allocated = 0
    for key, value in positive_items:
        exact = value / weight_sum * total_count
        floor_value = math.floor(exact)
        floor_counts[key] = floor_value
        allocated += floor_value
        remainders.append((exact - floor_value, key))
    remaining = total_count - allocated
    for _, key in sorted(
        remainders,
        key=lambda item: (-item[0], item[1]),
    )[:remaining]:
        floor_counts[key] += 1
    return floor_counts


def _build_weighted_round_robin_schedule(
    target_counts: Mapping[str, int],
) -> tuple[str, ...]:
    total = sum(target_counts.values())
    used = {key: 0 for key in target_counts}
    keys = sorted(target_counts)
    schedule: list[str] = []
    for step in range(total):
        best_key: str | None = None
        best_score: float | None = None
        for key in keys:
            if used[key] >= target_counts[key]:
                continue
            desired = target_counts[key] * float(step + 1) / float(max(total, 1))
            score = desired - float(used[key])
            if best_score is None or score > best_score or (
                math.isclose(score, best_score) and key < (best_key or key)
            ):
                best_key = key
                best_score = score
        if best_key is None:
            break
        used[best_key] += 1
        schedule.append(best_key)
    return tuple(schedule)


def _cycle_take(indices: tuple[int, ...], count: int) -> list[int]:
    if not indices:
        return []
    resolved: list[int] = []
    while len(resolved) < count:
        resolved.extend(indices)
    return resolved[:count]


def _seeded_shuffle(values: list[int], seed: int) -> list[int]:
    rng = random.Random(seed)
    shuffled = list(values)
    rng.shuffle(shuffled)
    return shuffled


def _stable_int_seed(*parts: Any) -> int:
    token = "::".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(token).hexdigest()[:16], 16)
