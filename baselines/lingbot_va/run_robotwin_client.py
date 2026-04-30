from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


UPSTREAM_CLIENT_RELATIVE_PATH = Path("evaluation") / "robotwin" / "eval_polict_client_openpi.py"
ROBOTWIN_ROOT_PLACEHOLDER = 'robowin_root = Path("/path/to/your/robowin")'
UPSTREAM_SAVE_DIR_LINE = (
    'save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")'
)
PATCHED_SAVE_DIR_LINE = (
    'save_dir = Path(save_root) / "eval_result" / str(task_name) / str(policy_name) / '
    'str(task_config) / str(ckpt_setting) / str(current_time)'
)


def patch_robotwin_client_source(source: str, robotwin_root: Path) -> str:
    """Patch only path/output assumptions in the upstream RobotWin client source."""

    if ROBOTWIN_ROOT_PLACEHOLDER not in source:
        raise ValueError("Could not find upstream RobotWin root placeholder in client source.")
    if UPSTREAM_SAVE_DIR_LINE not in source:
        raise ValueError("Could not find upstream eval_result save_dir line in client source.")

    patched = source.replace(
        ROBOTWIN_ROOT_PLACEHOLDER,
        f"robowin_root = Path({str(robotwin_root)!r})",
    )
    return patched.replace(UPSTREAM_SAVE_DIR_LINE, PATCHED_SAVE_DIR_LINE)


def build_upstream_argv(args: argparse.Namespace, source_path: Path) -> list[str]:
    ckpt_setting = args.ckpt_setting if args.ckpt_setting is not None else args.model_name
    return [
        str(source_path),
        "--config",
        args.config,
        "--overrides",
        "--task_name",
        args.task_name,
        "--task_config",
        args.task_config,
        "--train_config_name",
        args.train_config_name,
        "--model_name",
        args.model_name,
        "--ckpt_setting",
        ckpt_setting,
        "--seed",
        str(args.seed),
        "--policy_name",
        args.policy_name,
        "--save_root",
        str(Path(args.save_root).expanduser().resolve()),
        "--video_guidance_scale",
        str(args.video_guidance_scale),
        "--action_guidance_scale",
        str(args.action_guidance_scale),
        "--test_num",
        str(args.test_num),
        "--port",
        str(args.port),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the upstream LingBot-VA RobotWin eval client with local paths.")
    parser.add_argument("--source-repo", type=str, default="previous_works/lingbot-va")
    parser.add_argument(
        "--robotwin-root",
        type=str,
        default=os.environ.get("ROBOTWIN_ROOT"),
        help="Path to a RoboTwin checkout. Defaults to ROBOTWIN_ROOT.",
    )
    parser.add_argument("--save-root", type=str, default="outputs/lingbot_va_robotwin_eval")
    parser.add_argument("--task-name", type=str, default="adjust_bottle")
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=29056)
    parser.add_argument("--config", type=str, default="policy/ACT/deploy_policy.yml")
    parser.add_argument("--policy-name", type=str, default="ACT")
    parser.add_argument("--task-config", type=str, default="demo_clean")
    parser.add_argument("--train-config-name", type=str, default="0")
    parser.add_argument("--model-name", type=str, default="0")
    parser.add_argument("--ckpt-setting", type=str, default=None)
    parser.add_argument("--video-guidance-scale", type=float, default=5.0)
    parser.add_argument("--action-guidance-scale", type=float, default=1.0)
    args = parser.parse_args()

    if not args.robotwin_root:
        raise ValueError("--robotwin-root is required unless ROBOTWIN_ROOT is set.")

    source_repo = Path(args.source_repo).expanduser().resolve()
    robotwin_root = Path(args.robotwin_root).expanduser().resolve()
    source_path = source_repo / UPSTREAM_CLIENT_RELATIVE_PATH
    source = patch_robotwin_client_source(source_path.read_text(encoding="utf-8"), robotwin_root)

    for path in (source_repo, robotwin_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    sys.argv = build_upstream_argv(args, source_path)
    globals_dict = {"__name__": "__main__", "__file__": str(source_path)}
    exec(compile(source, str(source_path), "exec"), globals_dict)


if __name__ == "__main__":
    main()
