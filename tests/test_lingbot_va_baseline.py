from __future__ import annotations

from pathlib import Path

from baselines.lingbot_va.assets import safe_link_dir, to_heng_transformer_key
from baselines.lingbot_va.config import (
    CheckpointSpec,
    RolloutSuiteConfig,
    expand_env_vars,
    iter_episode_specs,
    parse_int_selection,
    suite_config_from_mapping,
)


def test_lingbot_va_key_conversion_matches_heng_conditioner_names() -> None:
    assert to_heng_transformer_key("time_conditioner.linear.weight") == "condition_embedder.linear.weight"
    assert to_heng_transformer_key("text_proj.proj.weight") == "condition_embedder.text_embedder.proj.weight"
    assert (
        to_heng_transformer_key("action_time_conditioner.linear.weight")
        == "condition_embedder_action.linear.weight"
    )
    assert (
        to_heng_transformer_key("action_text_proj.proj.weight")
        == "condition_embedder_action.text_embedder.proj.weight"
    )
    assert to_heng_transformer_key("runtime_stream_adapters.foo.weight") is None
    assert to_heng_transformer_key("blocks.0.attn1.to_q.weight") == "blocks.0.attn1.to_q.weight"


def test_parse_int_selection_supports_ranges_and_lists() -> None:
    assert parse_int_selection("0") == [0]
    assert parse_int_selection("0,2,4") == [0, 2, 4]
    assert parse_int_selection("0:3") == [0, 1, 2]
    assert parse_int_selection("0-2,5") == [0, 1, 2, 5]


def test_suite_config_expands_episode_grid(tmp_path: Path) -> None:
    config = RolloutSuiteConfig(
        checkpoints=(CheckpointSpec(name="ckpt", pretrained_root=tmp_path / "base"),),
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


def test_suite_mapping_uses_suite_defaults(tmp_path: Path) -> None:
    raw = {
        "source_repo": "previous_works/lingbot-va",
        "pretrained_root": str(tmp_path / "base"),
        "checkpoints": [{"name": "native", "transformer_dir": str(tmp_path / "transformer")}],
        "episodes": {"task_ids": "0:2", "episode_indices": "0,2", "seed": 3},
        "runtime": {"max_timestep": 11, "render_video": "false", "continue_on_error": "true"},
        "output_dir": str(tmp_path / "out"),
    }

    config = suite_config_from_mapping(raw, base_dir=Path.cwd())

    assert config.checkpoints[0].name == "native"
    assert config.checkpoints[0].pretrained_root == tmp_path / "base"
    assert config.checkpoints[0].transformer_dir == tmp_path / "transformer"
    assert config.task_ids == (0, 1)
    assert config.episode_indices == (0, 2)
    assert config.seed == 3
    assert config.max_timestep == 11
    assert config.render_video is False
    assert config.continue_on_error is True


def test_expand_env_vars_supports_default(monkeypatch) -> None:
    monkeypatch.delenv("LINGBOT_TEST_PATH", raising=False)
    assert expand_env_vars("${LINGBOT_TEST_PATH:-fallback}/x") == "fallback/x"
    monkeypatch.setenv("LINGBOT_TEST_PATH", "real")
    assert expand_env_vars("${LINGBOT_TEST_PATH:-fallback}/x") == "real/x"


def test_safe_link_dir_refuses_to_replace_unrelated_path(tmp_path: Path) -> None:
    source = tmp_path / "source"
    other = tmp_path / "other"
    dest = tmp_path / "dest"
    source.mkdir()
    other.mkdir()
    safe_link_dir(source, dest)
    assert dest.resolve() == source

    replacement = tmp_path / "replacement"
    replacement.mkdir()
    bad_dest = tmp_path / "bad_dest"
    bad_dest.mkdir()
    try:
        safe_link_dir(replacement, bad_dest)
    except FileExistsError as exc:
        assert "Refusing to replace" in str(exc)
    else:
        raise AssertionError("safe_link_dir should not replace an unrelated existing path")
