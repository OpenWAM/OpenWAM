from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

import open_wam
import open_wam.configs as open_wam_configs
import open_wam.models.action_decoders as action_decoders
import open_wam.models.policy_variants as policy_variants
from open_wam.cli.inspect_config import build_arg_parser
from open_wam.configs import (
    ActionDecoderName,
    LiberoAbsoluteJointExecutionMode,
    PolicyVariantName,
)
from open_wam.contracts import (
    SampleConstructionMetadata,
    ViewPlacement,
    VideoFrameMapping,
    normalized_video_frame_count,
    resolve_video_source_fps,
)
from open_wam.pipelines import ACTION_DECODER_BUILDERS, POLICY_VARIANT_BUILDERS
from open_wam.pipelines.factory import build_action_decoder, build_policy_variant
from open_wam.runtime import (
    CheckpointArtifactResolution,
    OPEN_WAM_RESULT_SCHEMA_V1,
    build_result_envelope,
    find_repo_root,
    resolve_checkpoint_artifacts,
    resolve_repo_path,
)
from open_wam.configs import load_experiment_config
from open_wam.utils import load_artifact_manifest, validate_artifact_layout


REPO_ROOT = Path(__file__).resolve().parents[1]


def _project_version() -> str:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["project"]["version"]


@pytest.mark.unit
def test_package_exposes_version_without_importing_integrations() -> None:
    code = (
        "import sys, open_wam; "
        "assert open_wam.__version__ == sys.argv[1]; "
        "assert 'open_wam.integrations.calvin_env' not in sys.modules; "
        "assert 'open_wam.integrations.robotwin_env' not in sys.modules; "
        "assert 'open_wam.integrations.libero_env' not in sys.modules; "
        "assert 'open_wam.integrations.libero_tasks' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code, _project_version()], check=True)


@pytest.mark.unit
def test_libero_task_contract_does_not_load_control_or_torch_stacks() -> None:
    code = (
        "import sys; "
        "from open_wam.integrations import LiberoTaskSpec, load_libero_benchmark_init_state_counts, resolve_libero_benchmark_tasks; "
        "assert LiberoTaskSpec.__module__ == 'open_wam.integrations.libero_tasks'; "
        "assert load_libero_benchmark_init_state_counts.__module__ == 'open_wam.integrations.libero_tasks'; "
        "assert resolve_libero_benchmark_tasks.__module__ == 'open_wam.integrations.libero_tasks'; "
        "assert 'open_wam.integrations.libero_tasks' in sys.modules; "
        "assert 'open_wam.integrations.libero_env' not in sys.modules; "
        "assert 'torch' not in sys.modules; "
        "assert 'numpy' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.unit
def test_console_entrypoints_are_declared() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]

    assert scripts["open-wam-train"] == "open_wam.cli.train:main"
    assert scripts["open-wam-eval"] == "open_wam.cli.eval:main"
    assert scripts["open-wam-inspect-config"] == "open_wam.cli.inspect_config:main"
    assert scripts["open-wam-validate-config"] == "open_wam.cli.validate_config:main"
    assert scripts["open-wam-sanity"] == "open_wam.cli.sanity:main"
    assert scripts["open-wam-sim-rollout"] == "open_wam.cli.sim_rollout:main"


@pytest.mark.unit
def test_base_dependencies_stay_minimal_and_extras_are_explicit() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    base_deps = pyproject["project"]["dependencies"]
    extras = pyproject["project"]["optional-dependencies"]
    heavy_base_names = {
        "torch",
        "lightning",
        "diffusers",
        "transformers",
        "numpy",
        "pyarrow",
        "h5py",
        "imageio",
        "imageio-ffmpeg",
        "matplotlib",
        "websockets",
    }

    assert base_deps == ["pyyaml>=6.0"]
    assert not {dependency.split(">=", 1)[0] for dependency in base_deps}.intersection(heavy_base_names)
    assert {
        "core",
        "torch",
        "train",
        "eval",
        "viz",
        "libero",
        "robotwin",
        "calvin",
        "deployment",
        "docs",
        "full",
    }.issubset(extras)


@pytest.mark.unit
def test_public_config_exports_include_libero_execution_mode() -> None:
    assert "LiberoAbsoluteJointExecutionMode" in open_wam_configs.__all__
    assert (
        open_wam_configs.LiberoAbsoluteJointExecutionMode
        is LiberoAbsoluteJointExecutionMode
    )


@pytest.mark.unit
def test_minimal_import_surfaces_do_not_import_torch_stack() -> None:
    code = (
        "import sys; "
        "import open_wam, open_wam.configs, open_wam.contracts, open_wam.extensions, open_wam.runtime, open_wam.utils, open_wam.pipelines, open_wam.simulators; "
        "from open_wam.cli.train import build_arg_parser as train_parser; "
        "from open_wam.cli.eval import build_arg_parser as eval_parser; "
        "from open_wam.cli.sanity import build_arg_parser as sanity_parser; "
        "from open_wam.cli.sim_rollout import build_arg_parser as sim_parser; "
        "from open_wam.cli.validate_config import build_arg_parser as validate_parser; "
        "[factory() for factory in (train_parser, eval_parser, sanity_parser, sim_parser, validate_parser)]; "
        "assert 'torch' not in sys.modules; "
        "assert 'lightning' not in sys.modules; "
        "assert 'diffusers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.unit
def test_sampled_eval_data_contracts_do_not_import_tensor_stacks() -> None:
    code = (
        "import sys; "
        "import open_wam.data, open_wam.data.replay_status; "
        "from open_wam.data import ReplayStatusRecord; "
        "import open_wam.evals.sampled_eval_sampling as sampling; "
        "assert ReplayStatusRecord.__module__ == 'open_wam.data.replay_status'; "
        "assert sampling.SampledEvalMode.FULL.value == 'full'; "
        "assert 'torch' not in sys.modules; "
        "assert 'numpy' not in sys.modules; "
        "assert 'pyarrow' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.unit
def test_sampled_eval_planning_public_facade_is_lazy_and_tensor_free() -> None:
    code = (
        "import sys, open_wam.evals as evals; "
        "assert 'open_wam.evals.sampled_eval_planning' not in sys.modules; "
        "from open_wam.evals import DatasetEpisode, SampledEvalCaseOptions, SampledEvalMethodSpec, parse_sampled_eval_target_requests; "
        "assert DatasetEpisode.__module__ == 'open_wam.evals.sampled_eval_sampling'; "
        "assert SampledEvalCaseOptions.__module__ == 'open_wam.evals.sampled_eval_planning'; "
        "assert SampledEvalMethodSpec.__module__ == 'open_wam.evals.sampled_eval_planning'; "
        "assert parse_sampled_eval_target_requests(['m1:x=/tmp/x'])[0].checkpoint_key == 'x'; "
        "assert 'open_wam.evals.sampled_eval_planning' in sys.modules; "
        "assert 'open_wam.integrations.libero_tasks' not in sys.modules; "
        "assert 'torch' not in sys.modules; "
        "assert 'numpy' not in sys.modules; "
        "assert 'pyarrow' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.unit
def test_generic_eval_contract_facade_is_lazy_and_tensor_free() -> None:
    code = (
        "import sys, open_wam.evals as evals; "
        "assert 'open_wam.evals.evaluation_contracts' not in sys.modules; "
        "assert 'open_wam.evals.evaluate' not in sys.modules; "
        "from open_wam.evals import EvaluationRequest, EvaluationSummary, resolve_evaluation_request; "
        "assert EvaluationRequest.__module__ == 'open_wam.evals.evaluation_contracts'; "
        "assert EvaluationSummary.__module__ == 'open_wam.evals.evaluation_contracts'; "
        "assert resolve_evaluation_request.__module__ == 'open_wam.evals.evaluation_contracts'; "
        "assert 'open_wam.evals.evaluation_contracts' in sys.modules; "
        "assert 'open_wam.evals.evaluate' not in sys.modules; "
        "assert 'torch' not in sys.modules; "
        "assert 'numpy' not in sys.modules; "
        "assert 'pyarrow' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.unit
def test_checkpoint_artifact_contract_does_not_import_tensor_stacks() -> None:
    code = (
        "import sys; "
        "from open_wam.runtime import CheckpointArtifactResolution, resolve_checkpoint_artifacts; "
        "result = resolve_checkpoint_artifacts(None); "
        "assert isinstance(result, CheckpointArtifactResolution); "
        "assert result.problem == 'checkpoint was not provided'; "
        "assert 'open_wam.runtime.checkpoint_artifacts' in sys.modules; "
        "assert 'open_wam.runtime.checkpoints' not in sys.modules; "
        "assert 'torch' not in sys.modules; "
        "assert 'numpy' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.unit
def test_runtime_checkpoint_artifact_exports_preserve_owner_identity() -> None:
    from open_wam.runtime import checkpoint_artifacts

    assert CheckpointArtifactResolution is checkpoint_artifacts.CheckpointArtifactResolution
    assert resolve_checkpoint_artifacts is checkpoint_artifacts.resolve_checkpoint_artifacts


@pytest.mark.unit
def test_lazy_data_facade_preserves_representative_export_identities() -> None:
    from importlib import import_module

    import open_wam.data as public_data

    owners = {
        "PoseSequence": "action_pose",
        "ActionMappingResult": "action_mapping",
        "WAMSample": "contracts",
        "GENERALIST_GJD_CHUNK_CONTRACT_T0_SINGLETON": "conditional_dynamics_layout",
        "ActionBranchSpec": "counterfactual_actions",
        "LatentWAMSample": "latent_contracts",
        "LatentSegmentBoundary": "latent_segment_geometry",
        "LatentSegmentMaterializationPlan": "latent_segment_materialization",
        "assemble_latent_views": "latent_view_assembly",
        "RowSequenceExtractor": "row_action_targets",
        "pack_temporal_sequence": "sequence_packing",
    }

    for name, module_name in owners.items():
        owner = import_module(f"open_wam.data.{module_name}")
        assert getattr(public_data, name) is getattr(owner, name)


@pytest.mark.unit
def test_runtime_entrypoints_accept_ordered_extensions() -> None:
    from open_wam.cli.eval import build_arg_parser as eval_parser
    from open_wam.cli.sanity import build_arg_parser as sanity_parser
    from open_wam.cli.sim_rollout import build_arg_parser as sim_parser
    from open_wam.cli.train import build_arg_parser as train_parser

    extension_args = ["--extension", "acme.data", "--extension", "acme.runtime:install"]

    assert train_parser().parse_args(
        ["--config-name", "smoke", *extension_args]
    ).extension == ["acme.data", "acme.runtime:install"]
    assert eval_parser().parse_args(["--cfg", "eval.yaml", *extension_args]).extension == [
        "acme.data",
        "acme.runtime:install",
    ]
    assert sanity_parser().parse_args(
        ["--cfg", "experiment.yaml", *extension_args]
    ).extension == ["acme.data", "acme.runtime:install"]
    assert sim_parser().parse_args(
        ["--cfg", "experiment.yaml", "--benchmark", "calvin", *extension_args]
    ).extension == ["acme.data", "acme.runtime:install"]


@pytest.mark.unit
def test_public_train_parser_advertises_resume_and_step_controls() -> None:
    from open_wam.cli.train import build_arg_parser as train_parser

    args = train_parser().parse_args(
        [
            "--cfg",
            "experiment.yaml",
            "--checkpoint-root",
            "runs/example/checkpoints/checkpoint_step_100",
            "--num-steps",
            "200",
        ]
    )

    assert args.checkpoint_root.endswith("checkpoint_step_100")
    assert args.num_steps == 200


@pytest.mark.unit
def test_inspect_config_cli_accepts_legacy_and_new_config_flags() -> None:
    parser = build_arg_parser()

    assert parser.parse_args(["--cfg", "a.yaml"]).config == "a.yaml"
    assert parser.parse_args(["--config", "b.yaml"]).config == "b.yaml"


@pytest.mark.unit
def test_result_envelope_schema_is_versioned_and_json_serializable() -> None:
    envelope = build_result_envelope(
        command="open-wam-eval",
        config="configs/evals/example.yaml",
        metrics={"mean_action_mse": 1.0},
        checkpoint=None,
        benchmark="robotwin",
        device="cpu",
        seed=0,
    )

    assert envelope["schema_version"] == OPEN_WAM_RESULT_SCHEMA_V1
    assert envelope["open_wam_version"] == open_wam.__version__ == _project_version()
    assert envelope["metrics"] == {"mean_action_mse": 1.0}
    json.dumps(envelope)


@pytest.mark.unit
def test_result_envelope_preserves_reserved_keys_when_extra_collides() -> None:
    envelope = build_result_envelope(
        command="open-wam-sanity",
        config="configs/experiments/example.yaml",
        metrics={"loss": 1.0},
        extra={
            "schema_version": "legacy.schema",
            "metrics": {"legacy_loss": 2.0},
            "dataset_type": "robotwin",
        },
    )

    assert envelope["schema_version"] == OPEN_WAM_RESULT_SCHEMA_V1
    assert envelope["metrics"] == {"loss": 1.0}
    assert envelope["dataset_type"] == "robotwin"
    assert envelope["legacy"]["schema_version"] == "legacy.schema"
    assert envelope["legacy"]["metrics"] == {"legacy_loss": 2.0}
    assert envelope["legacy_key_collisions"] == ["metrics", "schema_version"]


@pytest.mark.unit
def test_repo_path_resolution_detects_source_root(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    nested = source_root / "src" / "open_wam" / "runtime"
    nested.mkdir(parents=True)
    (source_root / "pyproject.toml").write_text(
        '[project]\nname = "open-wam"\nversion = "9.9.9"\n',
        encoding="utf-8",
    )
    marker = nested / "paths.py"
    marker.write_text("", encoding="utf-8")

    assert find_repo_root(marker) == source_root
    assert resolve_repo_path("configs/example.yaml", repo_root=source_root) == source_root / "configs/example.yaml"


@pytest.mark.unit
def test_repo_path_resolution_ignores_unrelated_git_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unrelated_root = tmp_path / "unrelated"
    unrelated_nested = unrelated_root / "src" / "some_package"
    unrelated_nested.mkdir(parents=True)
    (unrelated_root / ".git").mkdir()
    (unrelated_root / "pyproject.toml").write_text(
        '[project]\nname = "not-open-wam"\n',
        encoding="utf-8",
    )
    marker = unrelated_nested / "module.py"
    marker.write_text("", encoding="utf-8")

    fallback = tmp_path / "fallback"
    fallback.mkdir()
    monkeypatch.chdir(fallback)

    assert find_repo_root(marker) == fallback.resolve()


@pytest.mark.unit
def test_public_video_timeline_contracts_are_typed_and_deterministic() -> None:
    mapping = VideoFrameMapping.wan_causal_prefix_suffix(
        raw_observed_frames=16,
        raw_future_frames=32,
    )

    assert mapping.raw_total_frames == 48
    assert mapping.observed_frames == 4
    assert mapping.future_frames == 8
    assert mapping.total_frames == 12
    assert resolve_video_source_fps(
        None,
        container_fps=24.0,
        missing_observation_fps=15.0,
    ).source == "container"
    assert normalized_video_frame_count(
        101,
        source_fps=30.0,
        target_fps=10.0,
    ) == 34


@pytest.mark.unit
def test_public_cross_layer_contracts_are_dependency_free() -> None:
    placement = ViewPlacement(
        source_name="wrist",
        canonical_name="aux",
        top=128,
        left=0,
        height=64,
        width=64,
    )
    metadata = SampleConstructionMetadata.from_mapping(
        {
            "sampled_chunk_size": 4,
            "loss_frame_start": 1,
            "loss_frame_end": 5,
        }
    )

    assert placement.width == 64
    assert metadata is not None
    assert metadata.sampled_chunk_size_for(2) == 2
    assert metadata.frame_range_or_default(observed_num_frames=8) == (1, 5)


@pytest.mark.unit
def test_public_local_paths_sample_has_no_private_roots() -> None:
    sample = (REPO_ROOT / "configs/local_paths.sample.yaml").read_text(encoding="utf-8")

    assert "/simurgh" not in sample
    assert "/afs/" not in sample
    assert "/sailhome/" not in sample
    assert "private-user" not in sample
    assert "private-user" not in sample


@pytest.mark.unit
def test_artifact_manifest_sample_has_required_fields() -> None:
    raw = yaml.safe_load((REPO_ROOT / "configs/artifacts.sample.yaml").read_text(encoding="utf-8"))
    required = {
        "artifact_id",
        "method_family",
        "variant",
        "benchmark",
        "config",
        "local_path_alias",
        "expected_layout",
        "download_url",
        "checksum",
        "license",
        "source",
        "notes",
    }

    assert isinstance(raw["artifacts"], list)
    assert raw["artifacts"]
    for artifact in raw["artifacts"]:
        assert required.issubset(artifact)


@pytest.mark.unit
def test_artifact_manifest_loader_and_layout_validator(tmp_path: Path) -> None:
    entries = load_artifact_manifest(REPO_ROOT / "configs/artifacts.sample.yaml")
    assert entries
    entry = entries[0]
    root = tmp_path / "checkpoint_step_1"
    (root / "transformer").mkdir(parents=True)
    (root / "full_training_state.pt").write_text("", encoding="utf-8")
    (root / "transformer" / "config.json").write_text("{}", encoding="utf-8")

    assert validate_artifact_layout(root, entry.expected_layout) == ()


@pytest.mark.unit
def test_public_tiny_fixture_artifact_layout_is_valid() -> None:
    entries = load_artifact_manifest(REPO_ROOT / "configs/artifacts.sample.yaml")
    entry = next(item for item in entries if item.artifact_id == "public-tiny-synthetic-contract")
    root = REPO_ROOT / "tests/fixtures/public_tiny/artifacts/checkpoint_step_1"

    assert validate_artifact_layout(root, entry.expected_layout) == ()


@pytest.mark.smoke
@pytest.mark.parametrize(
    "config_name",
    [
        "parallel_stream_robotwin_smoke.yaml",
        "mot_robotwin_smoke.yaml",
    ],
)
def test_builtin_pipeline_registries_construct_smoke_variants(config_name: str) -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments" / config_name)

    assert type(config.policy_variant) in POLICY_VARIANT_BUILDERS.keys()
    assert config.action_decoder.name in ACTION_DECODER_BUILDERS.keys()
    assert build_policy_variant(config) is not None
    assert build_action_decoder(config) is not None


@pytest.mark.unit
def test_retired_register_attached_surface_is_not_publicly_selectable() -> None:
    assert "register_attached" not in {member.value for member in PolicyVariantName}
    assert "register_decoder" not in {member.value for member in ActionDecoderName}
    assert not hasattr(open_wam_configs, "RegisterAttachedPolicyConfig")
    assert not hasattr(open_wam_configs, "RegisterActionDecoderConfig")
    assert not hasattr(policy_variants, "RegisterAttachedPolicyVariant")
    assert not hasattr(action_decoders, "RegisterActionDecoder")
