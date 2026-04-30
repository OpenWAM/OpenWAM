from __future__ import annotations

from pathlib import Path

from baselines.lingbot_va.config import (
    CheckpointSpec,
    RolloutSuiteConfig,
    expand_env_vars,
    iter_episode_specs,
    load_episode_manifest,
    parse_int_selection,
    suite_config_from_mapping,
)
from baselines.lingbot_va.generate_libero10_manifest import _deterministic_pairs, _random_pairs
from baselines.lingbot_va.run_robotwin_client import build_upstream_argv, patch_robotwin_client_source
from baselines.lingbot_va.summarize_results import validate_rows


def test_parse_int_selection_supports_ranges_and_lists() -> None:
    assert parse_int_selection("0") == [0]
    assert parse_int_selection("0,2,4") == [0, 2, 4]
    assert parse_int_selection("0:3") == [0, 1, 2]
    assert parse_int_selection("0-2,5") == [0, 1, 2, 5]


def test_suite_config_expands_episode_grid(tmp_path: Path) -> None:
    config = RolloutSuiteConfig(
        checkpoints=(CheckpointSpec(name="ckpt", model_root=tmp_path / "model"),),
        benchmark="libero_10",
        task_ids=(0, 2),
        episode_indices=(1, 3),
        seed=7,
    )

    episodes = list(iter_episode_specs(config))

    assert [(episode.task_id, episode.episode_idx, episode.seed) for episode in episodes] == [
        (0, 1, 7),
        (0, 3, 7),
        (2, 1, 7),
        (2, 3, 7),
    ]


def test_suite_config_uses_episode_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """
episodes:
  - benchmark: libero_10
    task_id: 2
    episode_idx: 4
    seed: 9
    sample_id: libero_10_random_000
    sample_kind: random
    sample_index: 0
""",
        encoding="utf-8",
    )
    raw = {
        "model_root": str(tmp_path / "model"),
        "episodes": {"manifest": str(manifest)},
    }

    config = suite_config_from_mapping(raw, base_dir=Path.cwd())
    episodes = list(iter_episode_specs(config))

    assert len(episodes) == 1
    assert episodes[0].benchmark == "libero_10"
    assert episodes[0].task_id == 2
    assert episodes[0].episode_idx == 4
    assert episodes[0].sample_id == "libero_10_random_000"
    assert episodes[0].sample_kind == "random"
    assert episodes[0].sample_index == 0
    assert load_episode_manifest(manifest)[0] == episodes[0]


def test_suite_mapping_uses_suite_defaults(tmp_path: Path) -> None:
    raw = {
        "source_repo": "previous_works/lingbot-va",
        "model_root": str(tmp_path / "model"),
        "hf_repo_id": "robbyant/lingbot-va-posttrain-libero-long",
        "checkpoints": [{"name": "native", "enable_offload": "false"}],
        "episodes": {"task_ids": "0:2", "episode_indices": "0,2", "seed": 3},
        "runtime": {"max_timestep": 11, "render_video": "false", "continue_on_error": "true"},
        "output_dir": str(tmp_path / "out"),
    }

    config = suite_config_from_mapping(raw, base_dir=Path.cwd())

    assert config.checkpoints[0].name == "native"
    assert config.checkpoints[0].model_root == tmp_path / "model"
    assert config.checkpoints[0].hf_repo_id == "robbyant/lingbot-va-posttrain-libero-long"
    assert config.checkpoints[0].enable_offload is False
    assert config.task_ids == (0, 1)
    assert config.episode_indices == (0, 2)
    assert config.seed == 3
    assert config.max_timestep == 11
    assert config.render_video is False
    assert config.continue_on_error is True


def test_suite_mapping_rejects_transformer_override(tmp_path: Path) -> None:
    raw = {
        "model_root": str(tmp_path / "model"),
        "checkpoints": [{"name": "bad", "transformer_dir": str(tmp_path / "transformer")}],
    }

    try:
        suite_config_from_mapping(raw, base_dir=Path.cwd())
    except ValueError as exc:
        assert "transformer_dir" in str(exc)
        assert "vanilla LingBot-VA baseline" in str(exc)
    else:
        raise AssertionError("transformer_dir should be rejected for the vanilla baseline")


def test_suite_mapping_rejects_non_libero10_benchmark(tmp_path: Path) -> None:
    raw = {
        "model_root": str(tmp_path / "model"),
        "episodes": {"benchmark": "libero_spatial"},
    }

    try:
        suite_config_from_mapping(raw, base_dir=Path.cwd())
    except ValueError as exc:
        assert "libero_10" in str(exc)
    else:
        raise AssertionError("non-libero_10 suites should be rejected")


def test_libero_subset_sampling_helpers_are_deterministic() -> None:
    assert _deterministic_pairs(task_count=10, init_count=50, count=12) == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (6, 0),
        (7, 0),
        (8, 0),
        (9, 0),
        (0, 1),
        (1, 1),
    ]
    assert _random_pairs(task_count=3, init_count=4, count=5, seed=123) == _random_pairs(
        task_count=3,
        init_count=4,
        count=5,
        seed=123,
    )


def test_expand_env_vars_supports_default(monkeypatch) -> None:
    monkeypatch.delenv("LINGBOT_TEST_PATH", raising=False)
    assert expand_env_vars("${LINGBOT_TEST_PATH:-fallback}/x") == "fallback/x"
    monkeypatch.setenv("LINGBOT_TEST_PATH", "real")
    assert expand_env_vars("${LINGBOT_TEST_PATH:-fallback}/x") == "real/x"


def test_summary_validation_rejects_bad_full_eval_rows(tmp_path: Path) -> None:
    good_row = {
        "benchmark": "libero_10",
        "checkpoint_name": "lingbot_va_posttrain_libero_long",
        "task_id": 0,
        "episode_idx": 0,
        "seed": None,
        "prepared_model": {
            "hf_revision": "rev",
            "model_root": str(tmp_path / "model"),
        },
    }

    validate_rows(
        [good_row],
        expect_count=1,
        expect_benchmark="libero_10",
        expect_task_ids=[0],
        expect_episode_indices=[0],
        require_null_seed=True,
        require_unique=True,
        require_hf_revision="rev",
        require_model_root=str(tmp_path / "model"),
    )

    duplicate_rows = [good_row, dict(good_row)]
    try:
        validate_rows(duplicate_rows, require_unique=True)
    except ValueError as exc:
        assert "Duplicate" in str(exc)
    else:
        raise AssertionError("duplicate rows should be rejected")

    seeded_row = dict(good_row, seed=3)
    try:
        validate_rows([seeded_row], require_null_seed=True)
    except ValueError as exc:
        assert "seeds" in str(exc)
    else:
        raise AssertionError("non-null seed should be rejected")


def test_robotwin_client_patch_only_rewrites_paths(tmp_path: Path) -> None:
    source = """
from pathlib import Path
robowin_root = Path("/path/to/your/robowin")
save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
"""

    patched = patch_robotwin_client_source(source, tmp_path / "RoboTwin")

    assert f"robowin_root = Path('{tmp_path / 'RoboTwin'}')" in patched
    assert 'Path(save_root) / "eval_result"' in patched
    assert 'Path(f"eval_result/' not in patched


def test_robotwin_client_argv_preserves_upstream_defaults(tmp_path: Path) -> None:
    class Args:
        config = "policy/ACT/deploy_policy.yml"
        task_name = "adjust_bottle"
        task_config = "demo_clean"
        train_config_name = "0"
        model_name = "0"
        ckpt_setting = None
        seed = 0
        policy_name = "ACT"
        save_root = str(tmp_path / "out")
        video_guidance_scale = 5.0
        action_guidance_scale = 1.0
        test_num = 100
        port = 29056

    argv = build_upstream_argv(Args(), tmp_path / "eval_polict_client_openpi.py")

    assert argv[:5] == [
        str(tmp_path / "eval_polict_client_openpi.py"),
        "--config",
        "policy/ACT/deploy_policy.yml",
        "--overrides",
        "--task_name",
    ]
    assert argv[argv.index("--task_name") + 1] == "adjust_bottle"
    assert argv[argv.index("--ckpt_setting") + 1] == "0"
    assert argv[argv.index("--test_num") + 1] == "100"
    assert argv[argv.index("--port") + 1] == "29056"
