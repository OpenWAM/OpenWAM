"""Regression coverage for runtime artifact contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from open_wam.configs import ReferenceCoreInitMode
from open_wam.configs.backbone import SharedVideoTransformerConfig
from open_wam.data import WAMBatch
from open_wam.evals.libero_realtime_runtime import apply_inference_overrides
from open_wam.models.common.proprio_conditioning import resolve_hidden_proprio_context
from open_wam.models.visual_tower import reference_transformer, shared_transformer_support
from open_wam.runtime import checkpoints
from open_wam.training.step_executor import ViewBatchAdapter


def _backbone(**overrides):
    return SharedVideoTransformerConfig(
        hidden_size=32, num_layers=1, num_heads=4, attention_head_dim=8,
        ffn_dim=64, text_dim=16, freq_dim=8, **overrides,
    )


@pytest.mark.parametrize("path_kind", ["artifact", "legacy_root", "absolute_component"])
def test_missing_explicit_reference_artifact_never_constructs_random_weights(
    tmp_path, monkeypatch, path_kind
) -> None:
    root = tmp_path / "checkpoint_step_1"
    artifact = root / "transformer"
    fields = {
        "artifact": {"runtime_backbone_artifact_path": str(artifact)},
        "legacy_root": {"pretrained_model_name_or_path": str(root)},
        "absolute_component": {"transformer_subdir": str(artifact)},
    }[path_kind]

    class NeverConstruct:
        def __init__(self, **kwargs):
            raise AssertionError("A missing requested artifact must not reach random initialization")

    monkeypatch.setattr(reference_transformer, "load_wan_transformer_class", lambda config: NeverConstruct)
    with pytest.raises(FileNotFoundError, match="export_runtime_backbone") as caught:
        reference_transformer.build_reference_transformer(_backbone(**fields), action_dim=4)
    assert str(artifact) in str(caught.value)


@pytest.mark.parametrize("init_mode", [ReferenceCoreInitMode.FULL, ReferenceCoreInitMode.VIDEO_ONLY])
def test_canonical_reference_artifact_keeps_checkpoint_native_video_only_load(
    tmp_path, monkeypatch, init_mode
) -> None:
    artifact = tmp_path / "transformer"
    artifact.mkdir()
    calls = []
    model = torch.nn.Linear(1, 1)

    class Loader:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((path, kwargs))
            return model

    monkeypatch.setattr(reference_transformer, "load_wan_transformer_class", lambda config: Loader)
    config = _backbone(
        pretrained_model_name_or_path=str(tmp_path / "missing_legacy_root"),
        runtime_backbone_artifact_path=str(artifact), reference_core_init_mode=init_mode,
    )
    assert reference_transformer.build_reference_transformer(config, action_dim=4) is model
    assert calls[0][0] == str(artifact)
    expected = {"torch_dtype": torch.float32}
    if init_mode is ReferenceCoreInitMode.FULL:
        expected["action_dim"] = 4
    assert calls[0][1] == expected


def test_unloadable_canonical_artifact_stops_with_original_cause(tmp_path, monkeypatch) -> None:
    artifact = tmp_path / "transformer"
    artifact.mkdir()
    cause = OSError("incomplete tensor export")

    class Loader:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            raise cause

    monkeypatch.setattr(reference_transformer, "load_wan_transformer_class", lambda config: Loader)
    with pytest.raises(RuntimeError, match="incomplete tensor export") as caught:
        reference_transformer.build_reference_transformer(
            _backbone(runtime_backbone_artifact_path=str(artifact)), action_dim=4
        )
    assert caught.value.__cause__ is cause
    assert str(artifact) in str(caught.value)


@pytest.mark.parametrize("nested", [False, True])
def test_all_inference_overrides_reach_either_runner(nested) -> None:
    config = SimpleNamespace(video_num_inference_steps=50, action_num_inference_steps=10,
                             guidance_scale=1.0, action_guidance_scale=1.0)
    variant = SimpleNamespace(inference_config=config)
    runner = (SimpleNamespace(pipeline=SimpleNamespace(policy_variant=variant))
              if nested else SimpleNamespace(policy_variant=variant))
    apply_inference_overrides(runner, video_num_inference_steps=8, action_num_inference_steps=4,
                              guidance_scale=2.0, action_guidance_scale=3.0)
    assert vars(config) == dict(video_num_inference_steps=8, action_num_inference_steps=4,
                                guidance_scale=2.0, action_guidance_scale=3.0)


def test_invalid_step_override_does_not_mutate_other_settings() -> None:
    config = SimpleNamespace(video_num_inference_steps=50, action_num_inference_steps=10)
    runner = SimpleNamespace(pipeline=SimpleNamespace(policy_variant=SimpleNamespace(inference_config=config)))
    with pytest.raises(ValueError):
        apply_inference_overrides(runner, video_num_inference_steps=8, action_num_inference_steps=0,
                                  guidance_scale=None, action_guidance_scale=None)
    assert vars(config) == dict(video_num_inference_steps=50, action_num_inference_steps=10)


def test_rgb_proprio_payload_and_mask_reach_chunk_conditioning_without_fake_frames() -> None:
    state = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    mask = torch.ones_like(state)
    mask[:, :, -1] = 0
    batch = WAMBatch(views={"cam": torch.zeros(1, 3, 3, 2, 2)}, actions=torch.zeros(1, 3, 4),
                     state=state, state_mask=mask, metadata=({},))
    prepared = ViewBatchAdapter().prepare(batch)
    extra = prepared.policy_batch.extra
    assert extra["proprio_context_state"] is state
    assert extra["proprio_context_state_mask"] is mask
    assert prepared.policy_batch.state is state
    assert "proprio_context_frames" not in extra
    context = resolve_hidden_proprio_context(extra, require_frame_aligned=False, label="RGB test")
    torch.testing.assert_close(context.values, state * mask, rtol=0, atol=0)
    with pytest.raises(ValueError, match="requires frame-level"):
        resolve_hidden_proprio_context(extra, require_frame_aligned=True, label="RGB test")


def test_rgb_without_state_does_not_invent_proprio() -> None:
    batch = WAMBatch(views={"cam": torch.zeros(1, 2, 3, 2, 2)}, actions=torch.zeros(1, 2, 4))
    extra = ViewBatchAdapter().prepare(batch).policy_batch.extra
    assert "proprio_context_state" not in extra
    assert "proprio_context_state_mask" not in extra


@pytest.mark.parametrize("entry", ["directory", "model_state.pt", "full_training_state.pt"])
def test_metadata_only_config_lookup_does_not_require_weights(tmp_path, monkeypatch, entry) -> None:
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text("schema_version: 2\n", encoding="utf-8")

    def no_tensor_resolution(*args, **kwargs):
        raise AssertionError("Metadata-only lookup must not require a tensor artifact")

    monkeypatch.setattr(checkpoints, "resolve_checkpoint_file", no_tensor_resolution)
    path = tmp_path if entry == "directory" else tmp_path / entry
    assert checkpoints.find_checkpoint_resolved_config(path) == config_path.resolve()


def test_metadata_lookup_does_not_make_model_only_checkpoint_resumable(tmp_path) -> None:
    (tmp_path / "resolved_config.yaml").write_text("schema_version: 2\n", encoding="utf-8")
    (tmp_path / "model_state.pt").write_bytes(b"model-only")
    with pytest.raises(FileNotFoundError, match="full_training_state.pt"):
        checkpoints.resolve_checkpoint_file(tmp_path, operation="resume_training")






def test_shared_block_materializes_table_at_its_own_precision(monkeypatch) -> None:
    block = shared_transformer_support.SharedTransformerBlock(
        dim=8, ffn_dim=16, num_heads=2, cross_attn_norm=True, eps=1e-6
    )
    original = shared_transformer_support._materialize_runtime_parameter
    observed = []

    def checked(parameter, *, device, dtype):
        if parameter is block.scale_shift_table:
            observed.append((device, dtype))
        return original(parameter, device=device, dtype=dtype)

    monkeypatch.setattr(shared_transformer_support, "_materialize_runtime_parameter", checked)
    hidden = torch.randn(1, 2, 8)
    temb = torch.randn(1, 2, 6, 8, dtype=torch.bfloat16)
    output, _, _ = block(hidden, encoder_hidden_states=torch.randn(1, 3, 8), temb=temb, rotary_emb=None)
    output.square().mean().backward()
    assert observed == [(temb.device, torch.float32)]
    assert block.scale_shift_table.dtype is torch.float32
    assert torch.isfinite(output).all()
    assert block.scale_shift_table.grad is not None
    assert torch.isfinite(block.scale_shift_table.grad).all()
