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
from open_wam.configs import ActionDecoderName, PolicyVariantName
from open_wam.pipelines import ACTION_DECODER_BUILDERS, POLICY_VARIANT_BUILDERS
from open_wam.pipelines.factory import build_action_decoder, build_policy_variant
from open_wam.runtime import (
    OPEN_WAM_RESULT_SCHEMA_V1,
    build_result_envelope,
    find_repo_root,
    resolve_repo_path,
)
from open_wam.utils import load_artifact_manifest, load_experiment_config, validate_artifact_layout


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
        "assert 'open_wam.integrations.libero_env' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code, _project_version()], check=True)


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
def test_minimal_import_surfaces_do_not_import_torch_stack() -> None:
    code = (
        "import sys; "
        "import open_wam, open_wam.configs, open_wam.extensions, open_wam.runtime, open_wam.utils, open_wam.pipelines, open_wam.simulators; "
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
