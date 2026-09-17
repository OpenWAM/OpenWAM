"""Cache eligibility characterized against the existing numerical executors.

These tests establish dependency invariants, not cached/uncached equivalence.
The checkpoint characterization remains the numerical acceptance gate.
"""

from __future__ import annotations

import pytest
import torch

from open_wam.configs import CurrentBlockCoupling
from open_wam.models.policy_variants.dual_expert.attention_packed import (
    build_dual_expert_packed_coupling_attention_mask,
)
from open_wam.models.policy_variants.dual_expert.packed_block import (
    DualExpertPackedBlockStack,
)
from tests.test_dual_expert_packed_block import (
    _build_inputs,
    _make_action_expert,
    _make_video_core,
)


@pytest.fixture(params=("cpu", pytest.param("cuda", marks=pytest.mark.gpu)))
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for the GPU cache-dependency gate.")
    return torch.device(request.param)


def _case(coupling, device):
    torch.manual_seed(89)
    stack = (
        DualExpertPackedBlockStack(
            _make_video_core().blocks, _make_action_expert().blocks
        )
        .eval()
        .to(device)
    )
    inputs = _build_inputs(video_seq_len=18, action_seq_len=18, seed=91)
    inputs = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }
    mask = build_dual_expert_packed_coupling_attention_mask(
        num_video_frames=9,
        video_tokens_per_frame=1,
        num_action_frames=9,
        action_tokens_per_frame=1,
        chunk_size_frames=4,
        chunk_origin_frame=1,
        device=device,
        current_block_coupling=coupling,
    )
    inputs["video_attention_mask"] = mask[None, None, :18]
    inputs["action_attention_mask"] = mask[None, None, 18:]
    return stack, inputs


@pytest.mark.parametrize("coupling", tuple(CurrentBlockCoupling))
@torch.no_grad()
def test_clean_streams_are_invariant_to_current_noise_across_layers(coupling, device):
    stack, inputs = _case(coupling, device)
    reference = stack(**inputs)
    changed = dict(inputs)
    for name in ("video_hidden_states", "action_hidden_states"):
        changed[name] = inputs[name].clone()
        changed[name][:, 5:9] += torch.linspace(-2, 3, 32, device=device)
    changed["video_timestep_proj"] = inputs["video_timestep_proj"].clone()
    changed["action_temb"] = inputs["action_temb"].clone()
    changed["video_timestep_proj"][:, 5:9] += 0.7
    changed["action_temb"][:, 5:9] -= 0.3
    actual = stack(**changed)

    for original, updated in zip(reference, actual, strict=True):
        torch.testing.assert_close(updated[:, 9:], original[:, 9:], rtol=0, atol=0)
    assert any(
        not torch.equal(original[:, 5:9], updated[:, 5:9])
        for original, updated in zip(reference, actual, strict=True)
    ), "The noise perturbation must actually exercise a live computation."


@pytest.mark.parametrize("coupling", tuple(CurrentBlockCoupling))
@torch.no_grad()
def test_clean_stream_cache_depends_on_text_conditioning(coupling, device):
    stack, inputs = _case(coupling, device)
    original = stack(**inputs)
    changed = dict(inputs)
    changed["video_text_hidden_states"] = inputs["video_text_hidden_states"] + 1
    changed["action_text_hidden_states"] = inputs["action_text_hidden_states"] - 1
    actual = stack(**changed)
    assert all(
        not torch.equal(before[:, 9:], after[:, 9:])
        for before, after in zip(original, actual, strict=True)
    )


@torch.no_grad()
def test_fixed_input_in_live_slot_is_not_an_invariant_cache_entry(device):
    stack, inputs = _case(CurrentBlockCoupling.JOINT, device)
    original_video, _ = stack(**inputs)
    changed = dict(inputs)
    changed["action_hidden_states"] = inputs["action_hidden_states"].clone()
    changed["action_hidden_states"][:, 5:9] += torch.linspace(-2, 3, 32, device=device)
    actual_video, _ = stack(**changed)

    # IDM uses a fixed video input in the live/noisy slot. Its hidden features
    # still depend on evolving actions under the joint-like current-block mask.
    assert not torch.equal(actual_video[:, 5:9], original_video[:, 5:9])
    torch.testing.assert_close(
        actual_video[:, 9:], original_video[:, 9:], rtol=0, atol=0
    )


@torch.no_grad()
def test_retained_clean_features_depend_on_evicted_context(device):
    stack, inputs = _case(CurrentBlockCoupling.JOINT, device)
    original_video, _ = stack(**inputs)
    changed = dict(inputs)
    # Remove the first clean video frame from K/V visibility while preserving
    # all tensor shapes. Window eviction can change retained deeper features.
    for name in ("video_attention_mask", "action_attention_mask"):
        changed[name] = inputs[name].clone()
        changed[name][..., 9] = False
    actual_video, _ = stack(**changed)
    assert not torch.equal(actual_video[:, 10:], original_video[:, 10:])
