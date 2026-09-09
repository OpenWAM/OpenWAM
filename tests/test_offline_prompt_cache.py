from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from open_wam.configs.asset_cache import PromptCacheConfig
from open_wam.configs.backbone import SharedVideoTransformerConfig
from open_wam.models.visual_tower.prompt_cache import (
    OfflinePromptCache,
)
from open_wam.models.visual_tower.reference_assets import LingbotReferenceAssets


def _cache(
    root: Path, *, extra_prompt: bool = False
) -> tuple[dict, dict[str, torch.Tensor]]:
    (root / "embeddings").mkdir(parents=True)
    tensors = {
        "": torch.arange(4, dtype=torch.bfloat16).reshape(1, 4),
        "pick 红杯": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
    }
    if extra_prompt:
        tensors["place object"] = torch.tensor(
            [[4, -3, 2, 0], [3, 9, -2, 0]], dtype=torch.bfloat16
        )
    entries = []
    for prompt, tensor in tensors.items():
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        torch.save(tensor, root / "embeddings" / f"{key}.pt")
        entries.append({"prompt": prompt, "sha256": key, "tokens": len(tensor)})
    raw = (
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n"
    ).encode()
    (root / "prompts.jsonl").write_bytes(raw)
    hashes = {"text_encoder/weights.bin": "a" * 64, "tokenizer/config.json": "b" * 64}
    index = {
        "format_version": 1,
        "complete": True,
        "dtype": "bfloat16",
        "storage": "unpadded",
        "max_text_tokens": 512,
        "text_dim": 4,
        "encoder_files_sha256": hashes,
        "encoder_fingerprint": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
        "prompts_sha256": hashlib.sha256(raw).hexdigest(),
        "cached_prompts": len(entries),
        "unique_prompts": len(entries),
    }
    _write_index(root, index)
    return index, tensors


def _write_index(root: Path, index: dict) -> None:
    (root / "index.json").write_text(json.dumps(index))


def test_explicit_cache_works_without_loading_text_encoder(tmp_path, monkeypatch):
    index, tensors = _cache(tmp_path)
    assets = LingbotReferenceAssets.maybe_load(
        SharedVideoTransformerConfig(
            text_dim=4,
            prompt_cache=PromptCacheConfig(str(tmp_path), index["encoder_fingerprint"]),
        )
    )
    assert not assets.has_text_encoder
    encoded = assets.encode_prompts(
        ["pick 红杯", "", "pick 红杯"], device=torch.device("cpu"), dtype=torch.float32
    )
    assert encoded.shape == (3, 512, 4)
    torch.testing.assert_close(encoded[0, :3], tensors["pick 红杯"].float())
    torch.testing.assert_close(encoded[0], encoded[2])
    assert encoded[:, 3:].count_nonzero() == 0
    torch.testing.assert_close(
        assets.encode_blank_text(
            batch_size=1, device=torch.device("cpu"), dtype=torch.float32
        )[0, :1],
        tensors[""].float(),
    )
    assert assets.text_embedding_cache == {}


def test_cache_does_not_load_online_assets_or_change_vae_guard(tmp_path, monkeypatch):
    _cache(tmp_path)
    config = SharedVideoTransformerConfig(
        text_dim=4,
        load_text_conditioning=True,
        pretrained_model_name_or_path="/path/to/model",
        prompt_cache=PromptCacheConfig(str(tmp_path)),
    )
    assets = LingbotReferenceAssets.maybe_load(config)
    assert assets.offline_prompt_cache is not None
    from dataclasses import replace

    with pytest.raises(FileNotFoundError, match="placeholder"):
        LingbotReferenceAssets.maybe_load(replace(config, load_wan_vae_frontend=True))


def test_no_cache_preserves_disabled_text_behavior():
    assets = LingbotReferenceAssets(config=SharedVideoTransformerConfig(text_dim=4))
    assert (
        assets.encode_prompts(
            ["anything"], device=torch.device("cpu"), dtype=torch.float32
        )
        is None
    )


@pytest.mark.parametrize("max_text_tokens", [2, 256, 1024])
def test_cache_requires_the_encoded_token_limit(tmp_path, max_text_tokens):
    _cache(tmp_path)
    with pytest.raises(ValueError, match="token limit"):
        OfflinePromptCache(tmp_path, max_text_tokens=max_text_tokens, text_dim=4)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("complete", False, "complete"),
        ("format_version", 2, "format_version"),
        ("dtype", "float32", "bfloat16"),
        ("text_dim", 8, "text_dim"),
        ("max_text_tokens", 128, "token limit"),
        ("encoder_fingerprint", "a" * 64, "fingerprint"),
        ("encoder_files_sha256", {}, "provenance"),
        ("prompts_sha256", "b" * 64, "checksum"),
        ("cached_prompts", 1, "count"),
    ],
)
def test_rejects_incompatible_or_incomplete_index(tmp_path, field, value, match):
    index, _ = _cache(tmp_path)
    index[field] = value
    _write_index(tmp_path, index)
    with pytest.raises(ValueError, match=match):
        OfflinePromptCache(tmp_path, max_text_tokens=512, text_dim=4)


def test_pinned_encoder_mismatch_is_rejected(tmp_path):
    _cache(tmp_path)
    with pytest.raises(ValueError, match="expected encoder"):
        OfflinePromptCache(
            tmp_path,
            max_text_tokens=512,
            text_dim=4,
            expected_encoder_fingerprint="c" * 64,
        )


@pytest.mark.parametrize(
    "tensor",
    [
        torch.ones(3, 4, dtype=torch.float32),
        torch.ones(4, 4, dtype=torch.bfloat16),
        torch.ones(3, 5, dtype=torch.bfloat16),
        torch.full((3, 4), float("nan"), dtype=torch.bfloat16),
        {"embedding": torch.ones(3, 4)},
    ],
)
def test_corrupt_tensor_is_rejected_before_conditioning(tmp_path, tensor):
    _cache(tmp_path)
    key = hashlib.sha256("pick 红杯".encode()).hexdigest()
    torch.save(tensor, tmp_path / "embeddings" / f"{key}.pt")
    cache = OfflinePromptCache(tmp_path, max_text_tokens=512, text_dim=4)
    with pytest.raises(ValueError, match="invalid dtype, shape, or values"):
        cache.encode_prompts(
            ("pick 红杯",), device=torch.device("cpu"), dtype=torch.float32
        )


def test_unknown_prompt_and_missing_tensor_fail_instead_of_zero_fallback(tmp_path):
    _cache(tmp_path)
    cache = OfflinePromptCache(tmp_path, max_text_tokens=512, text_dim=4)
    with pytest.raises(KeyError, match="absent"):
        cache.encode_prompts(
            ("unknown",), device=torch.device("cpu"), dtype=torch.float32
        )
    (tmp_path / "embeddings" / f"{hashlib.sha256(b'').hexdigest()}.pt").unlink()
    with pytest.raises(FileNotFoundError):
        cache.encode_prompts(("",), device=torch.device("cpu"), dtype=torch.float32)


def test_ambient_cache_settings_do_not_redirect_model_assets(monkeypatch):
    monkeypatch.setenv("OPENWAM_PRETRAIN_TEXT_CACHE", "/nonexistent")
    monkeypatch.setenv("OPENWAM_PRETRAIN_TEXT_ENCODER_FINGERPRINT", "a" * 64)
    assets = LingbotReferenceAssets(SharedVideoTransformerConfig(text_dim=4))
    assert assets.offline_prompt_cache is None


def test_tensor_materialization_lease_is_held_through_load(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from open_wam.models.visual_tower import prompt_cache as module

    _cache(tmp_path)
    active = False
    observed = []
    original_load = torch.load

    @contextmanager
    def materialize(path):
        nonlocal active
        active = True
        try:
            yield path
        finally:
            active = False

    def load(path, **kwargs):
        assert active
        observed.append(path)
        return original_load(path, **kwargs)

    monkeypatch.setattr(module.torch, "load", load)
    cache = OfflinePromptCache(tmp_path, max_text_tokens=512, text_dim=4)
    monkeypatch.setattr(cache.resolver, "materialize", materialize)
    cache.encode_prompts(("",), device=torch.device("cpu"), dtype=torch.float32)
    assert len(observed) == 1 and not active


@pytest.mark.parametrize("mode", ["bucket", "packed"])
@pytest.mark.parametrize("drop_probability", [0.0, 0.1])
def test_portable_vpm_recipe_sends_cached_task_text_to_actual_cross_attention(
    tmp_path, monkeypatch, mode, drop_probability
):
    import yaml

    from open_wam.configs import (
        BatchingConfig,
        TextConditioningMode,
        load_experiment_config,
    )
    from open_wam.data.latent_batching import LatentBatchCollator
    from open_wam.data.latent_contracts import LatentWAMSample
    from open_wam.pipelines import build_variant_pipeline_from_config
    from open_wam.training.step_executor import (
        LatentBatchAdapter,
        PipelineTrainStepExecutor,
    )
    from tests.test_causal_video_prediction import _deterministic_cpu_math

    cache_root = tmp_path / "text"
    _, tensors = _cache(cache_root, extra_prompt=True)
    recipe_path = Path(__file__).parents[1] / "configs/examples/video_pretraining.yaml"
    raw = yaml.safe_load(recipe_path.read_text())
    raw["data"]["video_sources"] = [
        {
            "source_id": "tiny",
            "manifest_csv": str(tmp_path / "unused.csv"),
            "source_format": "latent",
        }
    ]
    raw["backbone"].update(
        pretrained_model_name_or_path=None,
        load_reference_core_weights=False,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        attention_head_dim=8,
        ffn_dim=32,
        text_dim=4,
        freq_dim=8,
        prompt_cache=dict(root=str(cache_root)),
    )
    raw["policy_variant"]["hidden_size"] = 16
    raw["action_decoder"]["hidden_size"] = 16
    raw["training"].update(
        video_num_train_timesteps=8, text_condition_dropout_prob=drop_probability
    )
    raw["trainer"].update(
        strategy="single_device", accelerator="cpu", precision="32-true"
    )
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    config = load_experiment_config(config_path)
    assert (
        config.policy_variant.text_conditioning_mode is TextConditioningMode.TASK_PROMPT
    )
    assert config.backbone.load_text_conditioning is False
    with _deterministic_cpu_math():
        torch.manual_seed(24)
        pipeline = build_variant_pipeline_from_config(config)
        assert not pipeline.visual_tower.frontend.reference_assets.has_text_encoder
        samples = [
            LatentWAMSample(
                video_latents=torch.randn(48, n, 2, 4),
                actions=torch.zeros(0, 1),
                task_text=prompt,
                metadata={
                    "dataset_type": "mixed_video",
                    "observed_prefix_frames": 1,
                    "future_suffix_frames": n - 1,
                    "valid_video_frames": n,
                },
            )
            for n, prompt in ((3, "pick 红杯"), (5, "place object"))
        ]
        batch = LatentBatchCollator(BatchingConfig(mode=mode))(samples)
        executor = PipelineTrainStepExecutor(
            pipeline=pipeline,
            batch_adapter=LatentBatchAdapter(),
            training_config=config.training,
        )
        if drop_probability:
            ordinary_dropout = executor._apply_text_condition_dropout

            def selected_dropout(prepared):
                # Admit the production 0.1 recipe and exercise a batch in which
                # all draws select dropout, without changing flow-noise RNG.
                with monkeypatch.context() as context:
                    context.setattr(
                        torch,
                        "rand",
                        lambda size, *, device: torch.zeros(size, device=device),
                    )
                    return ordinary_dropout(prepared)

            monkeypatch.setattr(
                executor, "_apply_text_condition_dropout", selected_dropout
            )
        seen, contexts = [], []
        predict = pipeline.visual_tower.predict_video_flow

        def capture_context(*args, **kwargs):
            contexts.append(kwargs["text_context"].detach().clone())
            return predict(*args, **kwargs)

        monkeypatch.setattr(
            pipeline.visual_tower, "predict_video_flow", capture_context
        )
        handle = pipeline.visual_tower.core.blocks[0].attn2.register_forward_pre_hook(
            lambda module, args, kwargs: seen.append(args[1].detach().clone()),
            with_kwargs=True,
        )
        torch.manual_seed(72)
        result = executor.forward_train(batch)
        result.loss.backward()
        handle.remove()
        assert torch.isfinite(result.loss)
        assert len(seen) == 1 and seen[0].abs().sum() > 0
        assert len(contexts) == 1
        for index, prompt in enumerate(("pick 红杯", "place object")):
            effective = "" if drop_probability else prompt
            expected = tensors[effective].float()
            torch.testing.assert_close(contexts[0][index, : len(expected)], expected)
            assert contexts[0][index, len(expected) :].count_nonzero() == 0
        gradients = [
            parameter.grad
            for parameter in pipeline.visual_tower.core.blocks[0].attn2.parameters()
            if parameter.grad is not None
        ]
        assert gradients and all(
            torch.isfinite(gradient).all() for gradient in gradients
        )
        assert any(gradient.abs().sum() > 0 for gradient in gradients)


@pytest.mark.parametrize(
    "probability,draws,expected",
    [
        (0.0, [0.1, 0.8, 0.2], ("one", "two", "three")),
        (1.0, [0.1, 0.8, 0.2], ("", "", "")),
        (0.5, [0.1, 0.8, 0.2], ("", "two", "")),
    ],
)
def test_lazy_task_text_dropout_uses_one_mask_and_keeps_original_labels(
    monkeypatch, probability, draws, expected
):
    from types import SimpleNamespace

    from open_wam.configs import TrainingConfig
    from open_wam.models.policy_variants import PolicyTrainBatch
    from open_wam.training.step_executor import (
        LatentBatchAdapter,
        PipelineTrainStepExecutor,
        PreparedTrainInput,
    )

    original = ("one", "two", "three")
    prepared = PreparedTrainInput(
        policy_batch=PolicyTrainBatch(
            actions=torch.zeros(3, 0, 1), extra={"task_text": original}
        )
    )
    executor = PipelineTrainStepExecutor(
        pipeline=SimpleNamespace(training=True),
        batch_adapter=LatentBatchAdapter(),
        training_config=TrainingConfig(text_condition_dropout_prob=probability),
    )
    calls = []

    def random_draw(size, *, device):
        calls.append(size)
        return torch.tensor(draws, device=device)

    monkeypatch.setattr(torch, "rand", random_draw)
    output = executor._apply_text_condition_dropout(prepared)
    assert output.policy_batch.extra["task_text"] == expected
    assert prepared.policy_batch.extra == {"task_text": original}
    assert calls == ([] if probability == 0 else [3])
    if probability:
        assert output.policy_batch.extra["source_task_text"] == original
