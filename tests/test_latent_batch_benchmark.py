"""CPU contracts for the bounded CUDA benchmark (not a CUDA speed claim)."""
from __future__ import annotations

import copy
from dataclasses import replace
import importlib.util
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from open_wam.configs import BatchingConfig, BatchingMode, ExperimentConfig
from open_wam.data.latent_batching import LatentBatchCollator
from open_wam.data.latent_contracts import LatentWAMSample


def load_benchmark():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_latent_batches.py"
    spec = importlib.util.spec_from_file_location("latent_batch_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


benchmark = load_benchmark()


@pytest.mark.parametrize("mode, expected", [("strict", (1, 6)), ("bucket", (2, 3)), ("padded", (2, 3)), ("packed", (2, 3))])
def test_original_benchmark_batch_layouts_are_unchanged(mode, expected):
    assert benchmark.benchmark_batch_layout(mode) == expected


def test_token_budget_candidate_uses_six_sample_logical_batches():
    assert benchmark.benchmark_batch_layout("packed", 150000) == (6, 1)
    args = benchmark.parser().parse_args(["--mode", "packed", "--max-tokens", "150000"])
    assert args.max_tokens == 150000


@pytest.mark.parametrize("mode, budget", [("strict", 1), ("bucket", 1), ("padded", 1), ("packed", 0), ("packed", -1), ("packed", True)])
def test_budget_benchmark_rejects_unsupported_mode_or_budget(mode, budget):
    with pytest.raises(ValueError, match="--max-tokens"):
        benchmark.benchmark_batch_layout(mode, budget)


@pytest.mark.parametrize("mode", ["strict", "bucket", "padded", "packed"])
def test_plan_keeps_measured_multiset_and_duplicate_draws(mode):
    indices = [9, 9, 2, 3, 8, 1, 3, 3, 7, 0, 6, 4]
    lengths = [100, 100, 70, 90, 120, 50, 90, 90, 200, 150, 160, 180]
    positions = benchmark.plan_positions(indices, lengths, warmup_samples=6, mode=mode, pool_size=4)
    assert sorted(positions[:6]) == list(range(6))
    assert sorted(positions[6:]) == list(range(6, 12))
    assert sorted(indices[position] for position in positions[6:]) == sorted(indices[6:])
    if mode == "bucket":
        assert positions.index(0) < positions.index(1)
    else:
        assert positions == list(range(12))


class StochasticDataset:
    def __getitem__(self, index):
        length = 8 + index
        return LatentWAMSample(
            video_latents=torch.rand(2, length, 4, 4),
            actions=torch.rand(length * 4, 3),
            metadata={
                "sampled_chunk_size": random.randint(1, 4),
                "sampled_window_size": int(np.random.randint(4, 65)),
                "history_frames": 1,
                "segment_valid_latent_frames": length - 1,
            },
        )


def dataset_wrapper(geometry="fixed"):
    return benchmark.ReproducibleDataset(StochasticDataset(), [1, 1, 3], seed=123, rank=0,
                                         geometry=geometry, chunk=4, window=64)


def test_draw_materialization_is_order_independent_and_preserves_rng():
    wrapped = dataset_wrapper()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    first = wrapped[0]
    wrapped[2]
    repeated = wrapped[0]
    duplicate_draw = wrapped[1]
    assert torch.equal(first.video_latents, repeated.video_latents)
    assert first.metadata == repeated.metadata
    assert not torch.equal(first.video_latents, duplicate_draw.video_latents)
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_fixed_geometry_survives_all_collators_and_reports_real_tokens():
    samples = [dataset_wrapper()[0], dataset_wrapper()[2]]
    cfg = SimpleNamespace(backbone=SimpleNamespace(patch_size_t=1, patch_size_h=2, patch_size_w=2))
    for mode in (BatchingMode.BUCKET, BatchingMode.PADDED, BatchingMode.PACKED):
        batch = LatentBatchCollator(BatchingConfig(mode=mode))(samples)
        report = benchmark.describe_batch(batch, cfg)
        assert report["materialized_latent_frames"] == 20
        assert report["valid_latent_frames"] == 18
        assert report["valid_video_patch_tokens"] == 72
        assert report["transport_latent_capacity"] == 22
        assert [sample["sampled_chunk_size"] for sample in report["samples"]] == [4, 4]
        assert [sample["sampled_window_size"] for sample in report["samples"]] == [64, 64]


def test_sampled_geometry_keeps_original_materialized_metadata():
    sample = dataset_wrapper("sampled")[0]
    for name, value in sample.metadata["benchmark_original_geometry"].items():
        assert sample.metadata[name] == value


def test_batch_description_reports_true_transformer_budget_separately_from_video_throughput():
    samples = [dataset_wrapper()[0], dataset_wrapper()[2]]
    cfg = SimpleNamespace(backbone=SimpleNamespace(patch_size_t=1, patch_size_h=2, patch_size_w=2))
    batch = LatentBatchCollator(BatchingConfig(mode=BatchingMode.PACKED))(samples)
    report = benchmark.describe_batch(batch, cfg, token_costs=(152, 184))
    assert report["transformer_tokens"] == 336
    assert report["sample_transformer_tokens"] == (152, 184)
    assert report["valid_video_patch_tokens"] == 72
    with pytest.raises(ValueError, match="per original sample"):
        benchmark.describe_batch(batch, cfg, token_costs=(152,))


def test_physical_token_reports_reuse_cpu_costs_without_reading_device_tensors():
    from open_wam.models.policy_variants.dual_expert.token_cost import dual_expert_token_costs

    base = ExperimentConfig()
    cfg = replace(base, data=replace(base.data, batching=BatchingConfig(mode=BatchingMode.PACKED)))
    batch = LatentBatchCollator(cfg.data.batching)([dataset_wrapper()[0], dataset_wrapper()[2]])
    costs = dual_expert_token_costs(config=cfg, batch=batch)
    cache = {item["benchmark_draw_position"]: cost for item, cost in zip(batch.metadata, costs, strict=True)}
    # Meta tensors have shapes but no readable payload: reporting must neither
    # invoke the CPU-only admission API on device data nor copy tensors to CPU.
    device_batch = replace(batch, video_latents=torch.empty_like(batch.video_latents, device="meta"))
    report = benchmark.describe_physical_batch_from_cost_cache(device_batch, cfg, cache)
    assert report["sample_transformer_tokens"] == costs
    assert report["transformer_tokens"] == sum(costs)


def summary_rows():
    rank = {
        "wall_seconds": 1.0, "data_wait_seconds": 0.1,
        "cuda_peak_allocated_bytes": 100, "cuda_peak_reserved_bytes": 200,
        "batches": [{"samples": [{}, {}], "valid_latent_frames": 18,
                     "valid_video_patch_tokens": 72, "transport_latent_capacity": 22,
                     "materialized_latent_frames": 20}],
        "proof": {"microbatch_losses": [0.5], "updates": [{
            "gradient_probes": [{"finite": True, "abs_max": 0.2}],
            "parameter_delta_probes": [{"abs_max": 0.01}],
        }]},
    }
    first = {"optimizer_step": 1, "warmup": True, "ranks": [{**copy.deepcopy(rank), "rank": index} for index in range(4)]}
    second = {"optimizer_step": 2, "warmup": False, "ranks": [{**copy.deepcopy(rank), "rank": index} for index in range(4)]}
    second["ranks"][3]["wall_seconds"] = 2.0
    return [first, second]


def test_summary_excludes_warmup_and_uses_slowest_rank_throughput():
    summary = benchmark.summarize(summary_rows(), world_size=4)
    assert summary["measured_updates"] == 1
    assert summary["mean_update_seconds"] == 2
    assert summary["samples"] == 8
    assert summary["samples_per_second"] == 4
    assert summary["valid_latent_frames_per_second"] == 36
    assert summary["valid_video_patch_tokens_per_second"] == 144
    assert summary["transport_padding_fraction"] == pytest.approx(1 - 20 / 22)
    assert summary["finite_loss"]
    assert summary["finite_gradient_probes"]
    assert summary["finite_parameter_update_probes"]
    assert summary["nonzero_parameter_update_probe"]
    assert summary["passed"]
    assert summary["all_rank_updates_passed"]


def test_summary_detects_nonfinite_and_zero_update():
    rows = summary_rows()
    for rank in rows[1]["ranks"]:
        rank["proof"]["microbatch_losses"] = [float("nan")]
        rank["proof"]["updates"][0]["gradient_probes"][0]["finite"] = False
        rank["proof"]["updates"][0]["parameter_delta_probes"][0]["abs_max"] = 0
    summary = benchmark.summarize(rows, world_size=4)
    assert not summary["finite_loss"]
    assert not summary["finite_gradient_probes"]
    assert not summary["nonzero_parameter_update_probe"]
    assert not summary["passed"]


def test_summary_rejects_nonfinite_parameter_update_even_if_another_probe_moves():
    rows = summary_rows()
    rows[1]["ranks"][0]["proof"]["updates"][0]["parameter_delta_probes"][0]["abs_max"] = float("nan")
    summary = benchmark.summarize(rows, world_size=4)
    assert not summary["finite_parameter_update_probes"]
    assert summary["nonzero_parameter_update_probe"]
    assert not summary["passed"]


@pytest.mark.parametrize("field", ["gradient_probes", "parameter_delta_probes"])
def test_summary_rejects_one_non_updating_rank_despite_active_peers(field):
    rows = summary_rows()
    rows[1]["ranks"][2]["proof"]["updates"][0][field][0]["abs_max"] = 0.0
    summary = benchmark.summarize(rows, world_size=4)
    assert summary["nonzero_gradient_probe"]
    assert summary["nonzero_parameter_update_probe"]
    assert not summary["passed"]
    failed = [check for check in summary["rank_update_admission"] if not check["passed"]]
    assert len(failed) == 1
    assert failed[0]["rank"] == 2
    assert failed[0]["optimizer_step"] == 2


def test_summary_rejects_one_stalled_update_despite_other_successful_updates():
    rows = summary_rows()
    rows.append(copy.deepcopy(rows[1]))
    rows[2]["optimizer_step"] = 3
    rows[2]["ranks"][0]["proof"]["updates"][0]["parameter_delta_probes"][0]["abs_max"] = 0
    summary = benchmark.summarize(rows, world_size=4)
    assert not summary["passed"]
    assert sum(not check["passed"] for check in summary["rank_update_admission"]) == 1


@pytest.mark.parametrize("missing", ["loss", "update", "gradient", "delta"])
def test_summary_rejects_missing_rank_update_evidence(missing):
    rows = summary_rows()
    proof = rows[1]["ranks"][1]["proof"]
    if missing == "loss":
        proof["microbatch_losses"] = []
    elif missing == "update":
        proof["updates"] = []
    elif missing == "gradient":
        proof["updates"][0]["gradient_probes"] = []
    else:
        proof["updates"][0]["parameter_delta_probes"] = []
    assert not benchmark.summarize(rows, world_size=4)["passed"]


def test_summary_rejects_a_missing_rank_report():
    rows = summary_rows()
    rows[1]["ranks"].pop()
    summary = benchmark.summarize(rows, world_size=4)
    assert not summary["complete_measured_rank_sets"]
    assert not summary["passed"]


def token_budget_summary_rows():
    rows = summary_rows()
    for rank in rows[1]["ranks"]:
        original = rank["batches"][0]
        original.update(samples=[{"draw_position": index} for index in range(6)],
                        valid_latent_frames=60, valid_video_patch_tokens=240,
                        materialized_latent_frames=60, transport_latent_capacity=120)
        rank["physical_batches"] = [
            {"samples": [{"draw_position": index} for index in positions],
             "materialized_latent_frames": 10 * len(positions),
             "transport_latent_capacity": 10 * len(positions),
             "transformer_tokens": 100 * len(positions)}
            for positions in ([0, 1], [2], [3], [4, 5])
        ]
        rank["configured_max_tokens"] = 200
        rank["proof"]["microbatch_losses"] = [0.5] * 4
    return rows


def test_budget_summary_admits_multiple_physical_forwards_for_one_logical_batch():
    summary = benchmark.summarize(token_budget_summary_rows(), world_size=4)
    assert summary["passed"]
    assert summary["samples"] == 24
    assert summary["logical_batch_sizes"] == [6] * 4
    assert summary["physical_microbatches_per_rank_update"] == [4] * 4
    assert summary["physical_microbatch_sizes"] == [2, 1, 1, 2] * 4
    assert summary["physical_max_tokens"] == 200
    assert summary["configured_max_tokens"] == [200]
    assert summary["transport_padding_fraction"] == 0


@pytest.mark.parametrize("failure", ["missing_sample", "reordered_sample", "over_budget", "missing_cost", "missing_loss"])
def test_budget_summary_rejects_invalid_physical_execution(failure):
    rows = token_budget_summary_rows()
    rank = rows[1]["ranks"][2]
    if failure == "missing_sample":
        rank["physical_batches"][0]["samples"].pop()
    elif failure == "reordered_sample":
        rank["physical_batches"][0]["samples"].reverse()
    elif failure == "over_budget":
        rank["physical_batches"][0]["transformer_tokens"] = 201
    elif failure == "missing_cost":
        del rank["physical_batches"][0]["transformer_tokens"]
    else:
        rank["proof"]["microbatch_losses"].pop()
    summary = benchmark.summarize(rows, world_size=4)
    assert not summary["passed"]


def test_budget_summary_rejects_inconsistent_physical_microstep_counts_across_ranks():
    rows = token_budget_summary_rows()
    rank = rows[1]["ranks"][0]
    first = rank["physical_batches"].pop(0)
    rank["physical_batches"][:0] = [
        {**first, "samples": [sample], "transformer_tokens": 100}
        for sample in first["samples"]
    ]
    rank["proof"]["microbatch_losses"].append(0.5)
    summary = benchmark.summarize(rows, world_size=4)
    assert all(check["passed"] for check in summary["rank_update_admission"])
    assert not summary["aligned_physical_microstep_counts"]
    assert not summary["passed"]


def test_tensor_probes_do_not_change_optimizer_update():
    model = torch.nn.Linear(3, 2)
    reference = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
    executor = SimpleNamespace(forward_train=lambda value: SimpleNamespace(loss=model(value).square().mean()))
    runtime = SimpleNamespace(model=model, optimizer=optimizer, step_executor=executor)
    pending, names = benchmark.install_probes(runtime, describe_physical_batch=lambda value: {"samples": int(value.shape[0])})
    assert names
    data = torch.randn(4, 3)
    runtime.step_executor.forward_train(data).loss.backward()
    optimizer.step()
    reference(data).square().mean().backward()
    reference_optimizer.step()
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    proof = benchmark.read_probes(pending)
    assert len(proof["microbatch_losses"]) == 1
    assert any(probe["abs_max"] > 0 for probe in proof["updates"][0]["parameter_delta_probes"])
    assert proof["physical_batches"] == [{"samples": 4}]
    assert not pending["losses"] and not pending["updates"]
    assert not pending["physical_batches"]
