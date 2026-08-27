from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from open_wam.configs import CausalVideoProgram, load_experiment_config
from open_wam.configs.enums import TextConditioningMode, TrainingComponentSelector
from open_wam.contracts import CanonicalViewLayout, ViewPlacement
from open_wam.data import LatentWAMBatch
from open_wam.data.lerobot_v2_latent_storage import assemble_canonical_latents
from open_wam.evals.video_prediction import rollout_causal_video_prediction

REPO_ROOT = Path(__file__).resolve().parents[1]


class _RecordingVideoPredictionPipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.policy_variant = SimpleNamespace(
            config=SimpleNamespace(program=CausalVideoProgram.PREFIX_SUFFIX)
        )

    def forward_infer_step_from_latents(
        self,
        video_latents: torch.Tensor,
        context,
        *,
        infer_state,
        canonical_video: torch.Tensor | None,
        text_context: torch.Tensor | None,
        negative_text_context: torch.Tensor | None,
    ):
        metadata = context.extra["metadata"][0]
        observed = int(metadata["observed_prefix_frames"])
        future = int(metadata["future_suffix_frames"])
        call_index = len(self.calls) + 1
        generated = video_latents.new_full(
            (
                video_latents.shape[0],
                video_latents.shape[1],
                future,
                video_latents.shape[3],
                video_latents.shape[4],
            ),
            float(10 * call_index),
        )
        predicted = torch.cat([video_latents[:, :, :observed], generated], dim=2)
        self.calls.append(
            {
                "shape": tuple(video_latents.shape),
                "metadata": dict(metadata),
                "task_text": context.extra["task_text"],
                "text_context": text_context,
                "negative_text_context": negative_text_context,
                "infer_state": infer_state,
                "canonical_video": canonical_video,
            }
        )
        return SimpleNamespace(
            decoder_output=SimpleNamespace(aux={"predicted_latents": predicted}),
            policy_output=SimpleNamespace(aux={}, next_state=None),
            visual_outputs=SimpleNamespace(
                frontend=SimpleNamespace(
                    conditioning=SimpleNamespace(
                        text_context=text_context,
                        negative_text_context=negative_text_context,
                    )
                )
            ),
        )


def _latent_batch() -> LatentWAMBatch:
    latents = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8, 1, 1)
    return LatentWAMBatch(
        video_latents=latents,
        actions=torch.zeros(1, 0, 1),
        action_mask=torch.zeros(1, 0, 1),
        state=torch.zeros(1, 0, 1),
        state_mask=torch.zeros(1, 0, 1),
        task_text=("move the object",),
        text_context=torch.ones(1, 2, 3),
        negative_text_context=torch.zeros(1, 2, 3),
        metadata=(
            {
                "observed_prefix_frames": 2,
                "future_suffix_frames": 3,
                "valid_video_frames": 5,
                "padded_video_frames": 8,
            },
        ),
    )


def test_video_prediction_rollout_reuses_exact_latent_dataset_layout() -> None:
    pipeline = _RecordingVideoPredictionPipeline()

    result = rollout_causal_video_prediction(
        pipeline,  # type: ignore[arg-type]
        _latent_batch(),
        num_chunks=2,
    )

    assert result.observed_latent_frames == 2
    assert result.future_latent_frames == 3
    assert result.context_latent_frames == (2, 5, 8)
    assert tuple(result.target_latents.shape) == (1, 1, 5, 1, 1)
    assert tuple(result.predicted_latents.shape) == (1, 1, 8, 1, 1)
    assert [call["shape"] for call in pipeline.calls] == [
        (1, 1, 5, 1, 1),
        (1, 1, 8, 1, 1),
    ]
    assert [call["metadata"] for call in pipeline.calls] == [
        {"observed_prefix_frames": 2, "future_suffix_frames": 3},
        {"observed_prefix_frames": 5, "future_suffix_frames": 3},
    ]
    assert all(call["task_text"] == ("move the object",) for call in pipeline.calls)
    assert result.first_chunk_future_mse == pytest.approx(
        float(((torch.full((3,), 10.0) - torch.arange(2, 5)).square().mean()).item())
    )


def test_canonical_latent_layout_rejects_partial_or_overlapping_metadata() -> None:
    with pytest.raises(ValueError, match="schema"):
        CanonicalViewLayout.from_metadata(
            {
                "camera_0": {
                    "top": 0,
                    "left": 0,
                    "latent_height": 8,
                    "latent_width": 8,
                }
            }
        )

    with pytest.raises(ValueError, match="source name"):
        CanonicalViewLayout.from_metadata(
            {
                "schema_version": CanonicalViewLayout.SCHEMA_VERSION,
                "canvas_height": 8,
                "canvas_width": 8,
                "placements": [
                    {
                        "slot": "legacy_camera",
                        "canonical_name": "camera",
                        "top": 0,
                        "left": 0,
                        "height": 8,
                        "width": 8,
                    }
                ],
            }
        )

    with pytest.raises(ValueError, match="overlap"):
        CanonicalViewLayout.from_metadata(
            {
                "schema_version": CanonicalViewLayout.SCHEMA_VERSION,
                "canvas_height": 8,
                "canvas_width": 12,
                "placements": [
                    {
                        "source_name": "camera_0",
                        "canonical_name": "camera_0",
                        "top": 0,
                        "left": 0,
                        "height": 8,
                        "width": 8,
                    },
                    {
                        "source_name": "camera_1",
                        "canonical_name": "camera_1",
                        "top": 0,
                        "left": 4,
                        "height": 8,
                        "width": 8,
                    },
                ],
            }
        )


def test_canonical_latent_layout_metadata_is_versioned_and_round_trips() -> None:
    layout = CanonicalViewLayout(
        canvas_height=8,
        canvas_width=8,
        placements=(
            ViewPlacement(
                source_name="camera_0",
                canonical_name="image",
                top=0,
                left=0,
                height=8,
                width=8,
            ),
        ),
    )

    metadata = layout.to_metadata()

    assert metadata["schema_version"] == CanonicalViewLayout.SCHEMA_VERSION
    assert CanonicalViewLayout.from_metadata(metadata) == layout

    with pytest.raises(ValueError, match="Unsupported canonical view layout schema"):
        CanonicalViewLayout.from_metadata(
            {**metadata, "schema_version": "open_wam.canonical_view_layout.v2"}
        )


def _view_payload(
    value: float,
    *,
    frames: int = 3,
    channels: int = 2,
    height: int = 8,
    width: int = 8,
    frame_ids: tuple[int, ...] | None = (0, 1, 2),
    dtype: torch.dtype = torch.float32,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "latent": torch.full((frames, height, width, channels), value, dtype=dtype),
        "latent_num_frames": frames,
        "latent_height": height,
        "latent_width": width,
    }
    if frame_ids is not None:
        payload["frame_ids"] = frame_ids
    return payload


def test_canonical_latent_assembly_preserves_valid_multiview_geometry() -> None:
    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/causal_video_prediction_libero_latent_local.yaml"
    )
    first_camera, second_camera = config.data.latent_camera_names

    latents, metadata = assemble_canonical_latents(
        config.data,
        {
            first_camera: _view_payload(1.0),
            second_camera: _view_payload(2.0),
        },
    )

    assert latents is not None
    assert tuple(latents.shape) == (2, 3, 8, 16)
    assert torch.all(latents[:, :, :, :8] == 1.0)
    assert torch.all(latents[:, :, :, 8:] == 2.0)
    assert metadata["schema_version"] == CanonicalViewLayout.SCHEMA_VERSION
    assert CanonicalViewLayout.from_metadata(metadata).canvas_width == 16


def test_canonical_latent_assembly_uses_layout_source_names() -> None:
    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/causal_video_prediction_libero_latent_local.yaml"
    )
    first_camera, second_camera = config.data.latent_camera_names

    latents, metadata = assemble_canonical_latents(
        replace(
            config.data,
            latent_camera_names=(second_camera, first_camera),
        ),
        {
            first_camera: _view_payload(1.0),
            second_camera: _view_payload(2.0),
        },
    )

    assert latents is not None
    assert torch.all(latents[:, :, :, :8] == 1.0)
    assert torch.all(latents[:, :, :, 8:] == 2.0)
    placements = CanonicalViewLayout.from_metadata(metadata).placements
    assert tuple(placement.source_name for placement in placements) == (
        first_camera,
        second_camera,
    )


def test_canonical_latent_assembly_requires_matching_camera_declarations() -> None:
    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/causal_video_prediction_libero_latent_local.yaml"
    )
    first_camera, second_camera = config.data.latent_camera_names

    with pytest.raises(ValueError, match="same camera set"):
        assemble_canonical_latents(
            replace(
                config.data,
                latent_camera_names=(first_camera, "unexpected_camera"),
            ),
            {
                first_camera: _view_payload(1.0),
                second_camera: _view_payload(2.0),
            },
        )


def test_canonical_latent_assembly_rejects_inconsistent_view_contracts() -> None:
    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/causal_video_prediction_libero_latent_local.yaml"
    )
    first_camera, second_camera = config.data.latent_camera_names
    first = _view_payload(1.0)

    with pytest.raises(ValueError, match="frame and channel dimensions"):
        assemble_canonical_latents(
            config.data,
            {
                first_camera: first,
                second_camera: _view_payload(
                    2.0,
                    frames=1,
                    frame_ids=(0,),
                ),
            },
        )

    with pytest.raises(ValueError, match="identical frame_ids"):
        assemble_canonical_latents(
            config.data,
            {
                first_camera: first,
                second_camera: _view_payload(2.0, frame_ids=(1, 2, 3)),
            },
        )

    with pytest.raises(ValueError, match="identical dtypes"):
        assemble_canonical_latents(
            config.data,
            {
                first_camera: first,
                second_camera: _view_payload(2.0, dtype=torch.float16),
            },
        )

    inexact_layout = replace(config.data.view_layout[0], height=127)
    with pytest.raises(ValueError, match="exact positive multiple"):
        assemble_canonical_latents(
            replace(
                config.data,
                latent_camera_names=(first_camera,),
                view_layout=(inexact_layout,),
            ),
            {first_camera: first},
        )


@pytest.mark.parametrize(
    ("frame_ids", "error", "message"),
    (
        ((0.5, 1.5, 2.5), TypeError, "integral values"),
        ((0, 2, 1), ValueError, "nondecreasing"),
        ((), ValueError, "non-empty"),
        ((False, 1, 2), TypeError, "integral values"),
    ),
)
def test_canonical_latent_assembly_rejects_invalid_frame_ids(
    frame_ids: tuple[object, ...],
    error: type[Exception],
    message: str,
) -> None:
    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/causal_video_prediction_libero_latent_local.yaml"
    )
    first_camera, second_camera = config.data.latent_camera_names
    first_payload = _view_payload(1.0, frame_ids=None)
    second_payload = _view_payload(2.0, frame_ids=None)
    first_payload["frame_ids"] = frame_ids
    second_payload["frame_ids"] = frame_ids

    with pytest.raises(error, match=message):
        assemble_canonical_latents(
            config.data,
            {
                first_camera: first_payload,
                second_camera: second_payload,
            },
        )


def test_canonical_latent_assembly_allows_stride_ids_and_padding_duplicates() -> None:
    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/causal_video_prediction_libero_latent_local.yaml"
    )
    first_camera, second_camera = config.data.latent_camera_names

    latents, _ = assemble_canonical_latents(
        config.data,
        {
            first_camera: _view_payload(1.0, frame_ids=(0, 0, 4, 8)),
            second_camera: _view_payload(2.0, frame_ids=(0, 0, 4, 8)),
        },
    )

    assert latents is not None


def test_canonical_video_only_config_uses_latent_units_and_semantic_selectors() -> None:
    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/causal_video_prediction_libero_latent_local.yaml"
    )

    buckets = config.data.sample_construction.effective_causal_prefix_suffix_buckets
    assert [(bucket.observed_frames, bucket.future_frames) for bucket in buckets] == [
        (1, 3),
        (2, 6),
        (3, 9),
        (4, 12),
        (5, 15),
    ]
    assert config.training.trainable_components == (
        TrainingComponentSelector.VISUAL_TOWER_SHARED_VIDEO_BACKBONE,
    )
    assert config.training.frozen_components == (
        TrainingComponentSelector.VISUAL_TOWER_SHARED_ACTION_RUNTIME,
        TrainingComponentSelector.VISUAL_TOWER_SHARED_RUNTIME_ADAPTERS,
    )
    assert (
        config.policy_variant.text_conditioning_mode
        == TextConditioningMode.TASK_PROMPT
    )
    assert config.trainer.runtime_backbone_export_components == (
        TrainingComponentSelector.VISUAL_TOWER_SHARED_VIDEO_BACKBONE,
    )
    assert config.training.text_condition_dropout_prob == pytest.approx(0.1)
    assert config.inference.guidance_scale == pytest.approx(5.0)
    assert config.trainer.limit_train_batches is None
    assert config.trainer.limit_val_batches == 0
    assert config.trainer.max_checkpoints_to_keep == 3
    assert config.trainer.wandb_mode == "online"


def _load_rollout_script():
    script_path = REPO_ROOT / "scripts/generate_video_only_rollout.py"
    spec = importlib.util.spec_from_file_location(
        "generate_video_only_rollout", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_video_only_rollout_cli_inherits_inference_defaults_and_uses_sample_indices() -> (
    None
):
    module = _load_rollout_script()
    parser = module.build_argument_parser()

    args = parser.parse_args(
        [
            "--config",
            "config.yaml",
            "--checkpoint",
            "checkpoint",
            "--reference-assets-root",
            "assets",
        ]
    )

    assert args.guidance_scale is None
    assert args.video_steps is None
    assert args.split == "val"
    assert args.sample_index == 0
    assert "--overwrite" not in parser._option_string_actions
    assert not any(
        option in parser._option_string_actions
        for option in ("--start-frame", "--observed-frames", "--future-frames")
    )


def test_video_only_rollout_uses_checkpoint_transformer_weights(
    tmp_path: Path,
) -> None:
    module = _load_rollout_script()
    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/causal_video_prediction_libero_latent_local.yaml"
    )
    config = replace(
        config,
        backbone=replace(config.backbone, load_reference_core_weights=False),
    )

    resolved = module._resolve_runtime_config(
        config,
        transformer_dir=tmp_path / "checkpoint" / "transformer",
        reference_assets_root=tmp_path / "reference-assets",
        data_root=None,
        empty_text_embedding=None,
        video_steps=None,
        guidance_scale=None,
    )

    assert resolved.backbone.load_reference_core_weights is True
    assert resolved.backbone.runtime_backbone_artifact_path == str(
        tmp_path / "checkpoint" / "transformer"
    )


def test_video_only_rollout_keeps_supplied_config_authoritative(
    tmp_path: Path,
) -> None:
    module = _load_rollout_script()
    config_path = (
        REPO_ROOT
        / "configs/experiments/causal_video_prediction_libero_latent_local.yaml"
    )
    base_config = load_experiment_config(config_path)
    supplied_config = replace(
        base_config,
        data=replace(base_config.data, local_root="/local/dataset"),
        training=replace(
            base_config.training,
            video_sigma_shift=1.25,
        ),
        inference=replace(
            base_config.inference,
            video_num_inference_steps=9,
        ),
    )
    checkpoint_dir = tmp_path / "checkpoint_step_12"
    transformer_dir = checkpoint_dir / "transformer"
    transformer_dir.mkdir(parents=True)
    (transformer_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint_dir / "resolved_config.yaml").write_text(
        "training:\n  video_sigma_shift: 7.25\n"
        "inference:\n  video_num_inference_steps: 37\n",
        encoding="utf-8",
    )

    resolved = module._resolve_runtime_config(
        supplied_config,
        transformer_dir=transformer_dir,
        reference_assets_root=tmp_path / "reference-assets",
        data_root=None,
        empty_text_embedding=None,
        video_steps=None,
        guidance_scale=None,
    )

    assert "merge_runtime_config_from_checkpoint" not in vars(module)
    assert resolved.data.local_root == "/local/dataset"
    assert resolved.training.video_sigma_shift == pytest.approx(1.25)
    assert resolved.inference.video_num_inference_steps == 9
    assert resolved.backbone.runtime_backbone_artifact_path == str(transformer_dir)


def test_video_only_rollout_output_identity_tracks_exact_sample_content(
    tmp_path: Path,
) -> None:
    module = _load_rollout_script()
    first = torch.zeros(1, 2, dtype=torch.bfloat16)
    second = first.clone()
    second[0, 0] = 1
    first_identity = {"sample_sha256": module._tensor_sha256(first)}
    second_identity = {"sample_sha256": module._tensor_sha256(second)}
    common = {
        "root": tmp_path,
        "transformer_dir": tmp_path / "checkpoint" / "transformer",
        "split": "val",
        "sample_index": 0,
        "num_chunks": 1,
        "seed": 7,
    }

    first_path = module._output_directory(identity=first_identity, **common)

    assert first_path == module._output_directory(identity=first_identity, **common)
    assert first_path != module._output_directory(identity=second_identity, **common)


def test_video_only_rollout_uses_standard_checkpoint_provenance(
    tmp_path: Path,
) -> None:
    module = _load_rollout_script()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: fixture\n", encoding="utf-8")
    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (transformer_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
    reference_assets = tmp_path / "reference"
    reference_assets.mkdir()
    empty_embedding = tmp_path / "empty.pt"
    empty_embedding.write_bytes(b"embedding")
    config = SimpleNamespace(
        data=SimpleNamespace(
            local_root=None,
            empty_text_embedding_path=str(empty_embedding),
        )
    )
    batch = SimpleNamespace(
        metadata=({},),
        task_text=("move object",),
        video_latents=torch.zeros(1, 2),
        text_context=torch.zeros(1, 2),
        negative_text_context=torch.zeros(1, 2),
    )

    identity = module._rollout_artifact_identity(
        config=config,
        resolved_config={"name": "fixture"},
        config_path=config_path,
        transformer_dir=transformer_dir,
        reference_assets_root=reference_assets,
        batch=batch,
        split="val",
        sample_index=0,
        num_chunks=1,
        seed=7,
        preview_fps=8.0,
    )
    provenance = module.collect_runtime_provenance(
        config_path=config_path,
        resolved_config={"name": "fixture"},
        checkpoint_path=transformer_dir,
        dataset_root=config.data.local_root,
        argv=(),
    )

    assert identity["checkpoint_transformer"] == provenance["checkpoint"]
    assert "dataset_root" not in identity
    assert provenance["checkpoint"]["exists"] is True


def test_video_only_rollout_rejects_output_collision_before_model_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_rollout_script()
    existing_output = tmp_path / "existing"
    existing_output.mkdir()
    args = SimpleNamespace(
        config="config.yaml",
        checkpoint="checkpoint",
        reference_assets_root="assets",
        data_root=None,
        empty_text_embedding=None,
        split="val",
        sample_index=0,
        num_chunks=1,
        video_steps=None,
        guidance_scale=None,
        device=None,
        decode_device=None,
        seed=7,
        preview_fps=8.0,
        output_dir=str(tmp_path),
    )
    parser = SimpleNamespace(parse_args=lambda: args)
    config = SimpleNamespace(data=SimpleNamespace(local_root=str(tmp_path)))

    monkeypatch.setattr(module, "build_argument_parser", lambda: parser)
    monkeypatch.setattr(
        module,
        "resolve_experiment_config_reference",
        lambda _: tmp_path / "config.yaml",
    )
    monkeypatch.setattr(
        module,
        "resolve_transformer_dir_override",
        lambda *args, **kwargs: tmp_path / "transformer",
    )
    monkeypatch.setattr(
        module, "_existing_path", lambda *args, **kwargs: tmp_path / "assets"
    )
    monkeypatch.setattr(module, "load_experiment_config", lambda _: config)
    monkeypatch.setattr(
        module, "_resolve_runtime_config", lambda *args, **kwargs: config
    )
    monkeypatch.setattr(module, "seed_everywhere", lambda _: None)
    monkeypatch.setattr(
        module, "build_train_val_latent_datasets", lambda _: ([object()], [object()])
    )
    monkeypatch.setattr(module, "collate_latent_wam_samples", lambda _: object())
    monkeypatch.setattr(module, "serialize_experiment_config", lambda _: {})
    monkeypatch.setattr(module, "_rollout_artifact_identity", lambda **kwargs: {})
    monkeypatch.setattr(module, "_output_directory", lambda **kwargs: existing_output)

    def _unexpected_model_construction(*args, **kwargs):
        raise AssertionError("model construction must happen after output preflight")

    monkeypatch.setattr(
        module,
        "build_variant_pipeline_from_config",
        _unexpected_model_construction,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        module.main()


def test_video_only_rollout_publishes_atomically_to_a_new_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_rollout_script()

    def _write_video(path: Path, frames, *, fps: float) -> None:
        assert fps == 8.0
        assert list(frames)
        path.write_bytes(b"video")

    monkeypatch.setattr(module, "write_video_frames", _write_video)
    output_dir = tmp_path / "rollout"
    video = torch.zeros(2, 4, 4, 3).numpy()
    summary = {"artifact_identity": {"schema_version": "test"}}

    published = module._publish_rollout_artifacts(
        output_dir,
        target=video,
        predicted=video,
        fps=8.0,
        summary={
            "schema_version": "open_wam.result.v1",
            "artifacts": {},
            **summary,
        },
    )

    paths = published["artifacts"]["videos"]
    assert paths == [
        str(output_dir / "target.mp4"),
        str(output_dir / "predicted.mp4"),
        str(output_dir / "comparison_target_left_predicted_right.mp4"),
    ]
    persisted = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert persisted == published
    assert persisted["schema_version"] == "open_wam.result.v1"
    assert not list(tmp_path.glob(".rollout.tmp-*"))

    with pytest.raises(FileExistsError, match="already exists"):
        module._publish_rollout_artifacts(
            output_dir,
            target=video,
            predicted=video,
            fps=8.0,
            summary=summary,
        )
