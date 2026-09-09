"""Deterministic mixed-video window planning and source-balanced ordering."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import random
from typing import TYPE_CHECKING

from open_wam.configs import (
    CausalPrefixSuffixBucketConfig,
    MixedVideoDataConfig,
    MixedVideoRandomMode,
    MixedVideoViewCombinationConfig,
    MixedVideoWeightMode,
)

if TYPE_CHECKING:
    from .mixed_video_catalog_contracts import MixedVideoEpisodeRecord


__all__ = [
    "MixedVideoWindowPlanner",
    "MixedVideoWindowRecord",
]


@dataclass(frozen=True)
class MixedVideoWindowRecord:
    """One deterministic video-only training window."""

    episode_key: str
    observation_start: int
    observed_prefix_frames: int
    future_suffix_frames: int
    view_combination_name: str | None = None
    view_combination_slots: tuple[str, ...] = ()

    @property
    def valid_video_frames(self) -> int:
        return self.observed_prefix_frames + self.future_suffix_frames


class MixedVideoWindowPlanner:
    """Plan sample geometry and epoch order without materializing tensors."""

    def __init__(self, data_config: MixedVideoDataConfig) -> None:
        self.data_config = data_config

    def build_episode_windows(
        self,
        *,
        episode_records: Mapping[str, MixedVideoEpisodeRecord],
        episode_keys: Sequence[str],
    ) -> tuple[MixedVideoWindowRecord, ...]:
        """Plan windows against each episode's canonical frame timeline."""

        windows: list[MixedVideoWindowRecord] = []
        for episode_key in episode_keys:
            episode = episode_records[episode_key]
            episode_length = int(episode.length_frames)
            if episode_length <= 0:
                continue
            for start in range(0, episode_length, self.data_config.sample_stride):
                bucket = self._select_valid_causal_bucket(
                    episode,
                    start,
                    episode_length=episode_length,
                )
                if bucket is None:
                    continue
                windows.append(
                    MixedVideoWindowRecord(
                        episode_key=episode_key,
                        observation_start=start,
                        observed_prefix_frames=bucket.observed_frames,
                        future_suffix_frames=bucket.future_frames,
                    )
                )
        return tuple(windows)

    def build_latent_view_windows(
        self,
        *,
        episode_records: Mapping[str, MixedVideoEpisodeRecord],
        episode_keys: Sequence[str],
    ) -> tuple[MixedVideoWindowRecord, ...]:
        """Plan one weighted window stream per valid latent-view combination."""

        windows: list[MixedVideoWindowRecord] = []
        for episode_key in episode_keys:
            episode = episode_records[episode_key]
            combinations = self.valid_latent_view_combinations(episode)
            for combination in combinations:
                episode_length = _latent_combination_length_frames(episode, combination.slots)
                if episode_length <= 0:
                    continue
                repeat_count = _latent_view_combination_repeat_count(combination, combinations)
                for start in range(0, episode_length, self.data_config.sample_stride):
                    bucket = self._select_valid_causal_bucket(
                        episode,
                        start,
                        episode_length=episode_length,
                    )
                    if bucket is None:
                        continue
                    for _ in range(repeat_count):
                        windows.append(
                            MixedVideoWindowRecord(
                                episode_key=episode_key,
                                observation_start=start,
                                observed_prefix_frames=bucket.observed_frames,
                                future_suffix_frames=bucket.future_frames,
                                view_combination_name=combination.name,
                                view_combination_slots=combination.slots,
                            )
                        )
        return tuple(windows)

    def valid_latent_view_combinations(
        self,
        episode: MixedVideoEpisodeRecord,
    ) -> tuple[MixedVideoViewCombinationConfig, ...]:
        """Resolve ordered combinations available for one physical episode."""

        streams_by_slot = {stream.target_slot: stream for stream in episode.streams}
        configured_slots = tuple(
            dict.fromkeys(self.data_config.latent_camera_names or self.data_config.camera_names)
        )
        present_slots = tuple(slot for slot in configured_slots if slot in streams_by_slot)
        if self.data_config.latent_view_combinations:
            valid: list[MixedVideoViewCombinationConfig] = []
            for combination in self.data_config.latent_view_combinations:
                if not combination.enabled:
                    continue
                if combination.source_ids and episode.source_id not in combination.source_ids:
                    continue
                if all(slot in streams_by_slot for slot in combination.slots):
                    valid.append(combination)
            return tuple(valid)
        if not present_slots:
            return ()
        return (
            MixedVideoViewCombinationConfig(
                name="all_available",
                slots=present_slots,
                sampling_weight=1.0,
            ),
        )

    def build_source_balanced_epoch_order(
        self,
        *,
        sample_index: Sequence[MixedVideoWindowRecord],
        episode_records: Mapping[str, MixedVideoEpisodeRecord],
        epoch: int = 0,
    ) -> tuple[int, ...]:
        """Build one deterministic weighted order across all source IDs."""

        source_to_indices: dict[str, list[int]] = defaultdict(list)
        for sample_index_value, window in enumerate(sample_index):
            episode = episode_records[window.episode_key]
            source_to_indices[episode.source_id].append(sample_index_value)
        if not source_to_indices:
            return ()
        source_counts = {source_id: len(indices) for source_id, indices in source_to_indices.items()}
        target_counts = _source_target_counts(self.data_config, source_counts)
        rng = random.Random(int(self.data_config.sampling_seed) + int(epoch))
        per_source_orders: dict[str, list[int]] = {}
        for source_id, indices in source_to_indices.items():
            order = list(indices)
            if self.data_config.random_mode == MixedVideoRandomMode.WITHIN_SOURCE:
                rng.shuffle(order)
            per_source_orders[source_id] = _repeat_or_trim(order, target_counts[source_id])

        source_cycle = _weighted_source_cycle(target_counts)
        epoch_order: list[int] = []
        source_offsets = {source_id: 0 for source_id in per_source_orders}
        for source_id in source_cycle:
            offset = source_offsets[source_id]
            source_order = per_source_orders[source_id]
            if offset >= len(source_order):
                continue
            epoch_order.append(source_order[offset])
            source_offsets[source_id] = offset + 1
        if self.data_config.random_mode == MixedVideoRandomMode.GLOBAL:
            rng.shuffle(epoch_order)
        return tuple(epoch_order)

    def build_shape_bucketed_epoch_order(
        self,
        *,
        sample_index: Sequence[MixedVideoWindowRecord],
        episode_records: Mapping[str, MixedVideoEpisodeRecord],
        shape_ids: Sequence[int],
        world_size: int,
        batch_size: int,
        epoch: int = 0,
    ) -> tuple[int, ...]:
        """Keep rank-local H/W compatible while preserving source draws.

        Repeat only incomplete shape tails and rank-alignment columns, recording
        their multiplicities. Length bucketing subsequently only permutes this
        existing order; it must not add, remove or resample any of these draws.
        """

        resolved_world_size = max(1, int(world_size))
        resolved_batch_size = max(1, int(batch_size))
        base_order = self.build_source_balanced_epoch_order(
            sample_index=sample_index,
            episode_records=episode_records,
            epoch=epoch,
        )
        if resolved_batch_size <= 1 or not base_order:
            return base_order
        if len(shape_ids) != len(sample_index):
            raise ValueError(
                "Shape-bucketed ordering needs one grid id per planned window, got "
                f"shape_ids={len(shape_ids)}, sample_index={len(sample_index)}."
            )

        # Walk the source-balanced order once, dropping each index into the open
        # column for its grid. A column closes at `batch_size` and becomes one
        # rank-batch. Relative order within a grid is preserved, so the
        # source interleaving the planner produced still shows through.
        open_columns: dict[int, list[int]] = {}
        columns: list[list[int]] = []
        for index in base_order:
            shape_id = int(shape_ids[index])
            column = open_columns.get(shape_id)
            if column is None:
                column = []
                open_columns[shape_id] = column
            column.append(int(index))
            if len(column) == resolved_batch_size:
                columns.append(column)
                del open_columns[shape_id]

        # Close each grid's partial tail column. Borrow the shortfall from
        # ANOTHER full column of the same grid rather than repeating a sample
        # already inside this column: a sample appearing twice in one batch
        # would carry double weight in that batch's gradient, whereas a sample
        # appearing in two different batches is just a sample seen twice.
        columns_by_shape: dict[int, list[int]] = {}
        for column_position, column in enumerate(columns):
            columns_by_shape.setdefault(int(shape_ids[column[0]]), []).append(column_position)
        repeated_samples = 0
        degenerate_shapes: list[int] = []
        for shape_id in sorted(open_columns):
            column = open_columns[shape_id]
            if not column:
                continue
            deficit = resolved_batch_size - len(column)
            donors = columns_by_shape.get(shape_id, ())
            if donors:
                borrowed: list[int] = []
                donor_cursor = 0
                while len(borrowed) < deficit:
                    donor = columns[donors[donor_cursor % len(donors)]]
                    borrowed.append(donor[(donor_cursor // len(donors)) % len(donor)])
                    donor_cursor += 1
                column.extend(borrowed)
            else:
                # This grid holds fewer than `batch_size` windows in the entire
                # epoch, so there is no other batch to borrow from. Repeating
                # within the batch is the only alternative to dropping the grid
                # outright, which would remove a source from the mixture.
                degenerate_shapes.append(shape_id)
                column.extend(column[position % len(column)] for position in range(deficit))
            repeated_samples += deficit
            columns.append(column)
        self.last_shape_bucketing_degenerate_shapes = tuple(degenerate_shapes)

        # Shuffling columns (not samples) restores a globally mixed order while
        # leaving each column homogeneous. Seeded identically on every rank:
        # all ranks must derive the SAME global order or the stride sharding
        # silently overlaps.
        rng = random.Random(
            int(self.data_config.sampling_seed)
            + int(epoch) * 1_000_003
            + resolved_world_size * 7_919
            + resolved_batch_size * 104_729
        )
        rng.shuffle(columns)

        # Pad the column count to a whole number of blocks. Padding columns are
        # taken from the FRONT of the shuffled list, which lands in an early
        # block, so the final block never holds the same column twice and no
        # step sees a duplicated batch across its ranks.
        remainder = len(columns) % resolved_world_size
        if remainder:
            snapshot = list(columns)
            deficit = resolved_world_size - remainder
            if len(snapshot) >= resolved_world_size:
                columns.extend(snapshot[position] for position in range(deficit))
            else:
                columns.extend(snapshot[position % len(snapshot)] for position in range(deficit))
            repeated_samples += deficit * resolved_batch_size

        epoch_order: list[int] = []
        for block_start in range(0, len(columns), resolved_world_size):
            block = columns[block_start : block_start + resolved_world_size]
            for row in range(resolved_batch_size):
                for column in block:
                    epoch_order.append(column[row])
        self.last_shape_bucketing_repeats = repeated_samples
        return tuple(epoch_order)


    def _select_valid_causal_bucket(
        self,
        episode: MixedVideoEpisodeRecord,
        observation_start: int,
        *,
        episode_length: int,
    ) -> CausalPrefixSuffixBucketConfig | None:
        valid_buckets: list[CausalPrefixSuffixBucketConfig] = []
        for bucket in self.data_config.sample_construction.effective_causal_prefix_suffix_buckets:
            total_frames = _causal_bucket_total_frames(self.data_config, bucket)
            required_span = (total_frames - 1) * int(self.data_config.frame_stride) + 1
            if int(observation_start) + required_span <= int(episode_length):
                valid_buckets.append(bucket)
        if not valid_buckets:
            return None
        token = f"{self.data_config.sampling_seed}:{episode.key}:{observation_start}".encode("utf-8")
        bucket_index = int(hashlib.sha256(token).hexdigest()[:16], 16) % len(valid_buckets)
        return valid_buckets[bucket_index]


def _latent_view_combination_repeat_count(
    combination: MixedVideoViewCombinationConfig,
    combinations: Sequence[MixedVideoViewCombinationConfig],
) -> int:
    positive_weights = [float(item.sampling_weight) for item in combinations if item.enabled]
    if not positive_weights:
        return 1
    scale = min(positive_weights)
    return max(1, int(round(float(combination.sampling_weight) / scale)))


def _latent_combination_length_frames(
    episode: MixedVideoEpisodeRecord,
    slots: Sequence[str],
) -> int:
    streams_by_slot = {stream.target_slot: stream for stream in episode.streams}
    lengths: list[int] = []
    for slot in slots:
        stream = streams_by_slot.get(slot)
        if stream is None:
            raise KeyError(f"Mixed-video latent episode {episode.key!r} is missing slot {slot!r}.")
        if stream.latent_length_frames is None:
            raise ValueError(
                f"Mixed-video episode {episode.key!r}, slot {slot!r} "
                "has no latent_length_frames; latent training requires "
                "manifest latent sidecars."
            )
        lengths.append(int(stream.latent_length_frames))
    return min(lengths) if lengths else 0


def _causal_bucket_total_frames(
    data_config: MixedVideoDataConfig,
    bucket: CausalPrefixSuffixBucketConfig,
) -> int:
    total_frames = int(bucket.observed_frames) + int(bucket.future_frames)
    if total_frames <= 0:
        raise ValueError("Mixed-video causal prefix/suffix buckets must request at least one frame.")
    if total_frames > int(data_config.num_frames):
        raise ValueError(
            f"Mixed-video causal bucket requests {total_frames} frames, "
            f"but data.num_frames={data_config.num_frames}."
        )
    return total_frames


def _source_target_counts(
    data_config: MixedVideoDataConfig,
    source_counts: dict[str, int],
) -> dict[str, int]:
    if data_config.weight_mode == MixedVideoWeightMode.PROPORTIONAL_TO_SIZE:
        return dict(source_counts)
    manual_weights = {
        source.source_id: source.sampling_weight
        for source in data_config.video_sources
        if source.enabled and source.sampling_weight is not None
    }
    if data_config.weight_mode == MixedVideoWeightMode.MANUAL_OVERRIDE:
        if set(manual_weights) != set(source_counts):
            missing = sorted(set(source_counts) - set(manual_weights))
            raise ValueError(f"manual_override mixed-video weighting needs sampling_weight for: {missing}")
        total = sum(source_counts.values())
        weight_sum = sum(float(value) for value in manual_weights.values())
        return {
            source_id: max(1, int(round(total * float(manual_weights[source_id]) / weight_sum)))
            for source_id in source_counts
        }
    scaled: dict[str, int] = {}
    for source_id, count in source_counts.items():
        scale = float(manual_weights.get(source_id, 1.0))
        scaled[source_id] = max(1, int(round(count * scale)))
    return scaled


def _weighted_source_cycle(target_counts: dict[str, int]) -> tuple[str, ...]:
    remaining = dict(target_counts)
    total = sum(remaining.values())
    order: list[str] = []
    while len(order) < total:
        source_id = max(
            (source for source, count in remaining.items() if count > 0),
            key=lambda source: remaining[source] / max(1, target_counts[source]),
        )
        order.append(source_id)
        remaining[source_id] -= 1
    return tuple(order)


def _repeat_or_trim(values: Sequence[int], target_count: int) -> list[int]:
    if target_count <= len(values):
        return list(values[:target_count])
    repeats = (target_count + len(values) - 1) // len(values)
    return list((list(values) * repeats)[:target_count])
