"""Matched paired-block microbenchmark, not a trained-policy speed claim.

Reports complete denoising-call time, including binding and first-pass cache
population. Use an otherwise idle GPU for publishable latency measurements.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from open_wam.configs import CurrentBlockCoupling
from open_wam.models.common.denoising_cache import DenoisingCache
from open_wam.models.policy_variants.dual_expert.attention_packed import (
    build_dual_expert_packed_coupling_attention_profile,
)
from open_wam.models.policy_variants.dual_expert.modules import DualExpertActionExpert
from open_wam.models.policy_variants.dual_expert.packed_block import (
    DualExpertPackedBlockStack,
)
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
from open_wam.models.visual_tower.replica_core import SharedVideoTransformerCore


@torch.no_grad()
def benchmark(*, frames: int, coupling: CurrentBlockCoupling, repeats: int):
    torch.manual_seed(891)
    hidden, layers, steps = 512, 2, 20
    core = SharedVideoTransformerCore(
        SharedVideoTransformerConfig(
            hidden_size=hidden,
            num_layers=layers,
            num_heads=8,
            attention_head_dim=64,
            ffn_dim=hidden * 4,
            text_dim=128,
            freq_dim=64,
        ),
        action_dim=7,
        state_dim=8,
    )
    action = DualExpertActionExpert(
        hidden_size=hidden,
        action_dim=7,
        num_layers=layers,
        num_heads=8,
        attention_head_dim=64,
        ffn_dim=hidden * 4,
        text_dim=128,
        freq_dim=64,
    )
    stack = DualExpertPackedBlockStack(core.blocks, action.blocks).cuda().eval()
    nv, na = 2 * frames * 32, 2 * frames * 4
    profile = build_dual_expert_packed_coupling_attention_profile(
        num_video_frames=frames,
        video_tokens_per_frame=32,
        num_action_frames=frames,
        action_tokens_per_frame=4,
        chunk_size_frames=4,
        chunk_origin_frame=1,
        device=torch.device("cuda"),
        current_block_coupling=coupling,
        build_dense_masks=True,
        build_flex_masks=False,
    )
    inputs = {
        "video_hidden_states": torch.randn(1, nv, hidden, device="cuda"),
        "action_hidden_states": torch.randn(1, na, hidden, device="cuda"),
        "video_timestep_proj": torch.randn(1, nv, 6, hidden, device="cuda"),
        "action_temb": torch.randn(1, na, 6, hidden, device="cuda"),
        "video_rotary_emb": None,
        "action_rotary_emb": None,
        "video_attention_mask": profile.self_attention_mask[None, None, :nv],
        "action_attention_mask": profile.self_attention_mask[None, None, nv:],
        "video_text_hidden_states": torch.randn(1, 16, hidden, device="cuda"),
        "action_text_hidden_states": torch.randn(1, 16, hidden, device="cuda"),
    }

    def run(cached):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        starting_memory = torch.cuda.memory_allocated()
        started = time.perf_counter()
        cache = None
        if cached:
            cache = DenoisingCache()
            cache.bind(
                profile=profile,
                invariant_tokens=profile.token_layout.noise_id == 1,
                stream_lengths=(nv, na),
                num_layers=layers,
            )
        # Vary only live features. Constant streams and their dependencies stay fixed.
        output = None
        for step in range(steps):
            changed = dict(inputs)
            for name, size in (
                ("video_hidden_states", nv),
                ("action_hidden_states", na),
            ):
                changed[name] = inputs[name].clone()
                changed[name][:, : size // 2] += step / steps
            output = stack(**changed, denoising_cache=cache)
        torch.cuda.synchronize()
        return (
            output,
            (time.perf_counter() - started) * 1000,
            torch.cuda.max_memory_allocated() - starting_memory,
        )

    run(False)
    run(True)
    times, memory = {False: [], True: []}, {False: [], True: []}
    exact = True
    max_error = 0.0
    for repeat in range(repeats):
        outputs = {}
        for cached in (False, True) if repeat % 2 == 0 else (True, False):
            output, elapsed, peak = run(cached)
            outputs[cached] = tuple(value.cpu() for value in output)
            times[cached].append(elapsed)
            memory[cached].append(peak)
        for expected, actual in zip(outputs[False], outputs[True], strict=True):
            exact &= torch.equal(expected, actual)
            max_error = max(max_error, (expected - actual).abs().max().item())
    return {
        "coupling": coupling.value,
        "frames": frames,
        "hidden_size": hidden,
        "layers": layers,
        "denoising_evaluations": steps,
        "cached_evaluations": steps,
        "cached_self_attention_query_fraction": (1 + (steps - 1) / 2) / steps,
        "exact_output": exact,
        "max_absolute_error": max_error,
        "uncached_ms": statistics.median(times[False]),
        "cached_ms": statistics.median(times[True]),
        "uncached_peak_bytes": max(memory[False]),
        "cached_peak_bytes": max(memory[True]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--frames", type=int, nargs="+", default=(5, 33, 65, 129))
    args = parser.parse_args()
    if (
        args.repeats < 1
        or any(frame <= 0 for frame in args.frames)
        or args.output.exists()
    ):
        parser.error("Use positive lengths/repeats and a fresh output path.")
    results = []
    for frames in args.frames:
        for coupling in CurrentBlockCoupling:
            result = benchmark(frames=frames, coupling=coupling, repeats=args.repeats)
            results.append(result)
            print(json.dumps(result), flush=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
