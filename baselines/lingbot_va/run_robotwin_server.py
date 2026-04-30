from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _prepare_paths(source_repo: Path) -> None:
    for path in (REPO_ROOT / "src", REPO_ROOT / "outputs" / "lingbot_va_pydeps", source_repo):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the upstream LingBot-VA RobotWin websocket server.")
    parser.add_argument("--source-repo", type=str, default="previous_works/lingbot-va")
    parser.add_argument("--model-root", type=str, required=True, help="Released LingBot-VA RobotWin full model root.")
    parser.add_argument("--save-root", type=str, default="outputs/lingbot_va_robotwin_server_vis")
    parser.add_argument("--port", type=int, default=29056)
    parser.add_argument("--host", type=str, default=None, help="Override the upstream server host.")
    parser.add_argument("--enable-offload", action="store_true")
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use upstream distributed server launch semantics. Direct single-GPU mode is the tested default.",
    )
    args = parser.parse_args()

    source_repo = Path(args.source_repo).expanduser().resolve()
    _prepare_paths(source_repo)

    from open_wam.third_party.lingbot import _ensure_flash_attn_shims

    _ensure_flash_attn_shims()

    from wan_va.configs import VA_CONFIGS
    from wan_va.utils import init_logger, logger, run_async_server_mode
    from wan_va.utils.Simple_Remote_Infer.deploy.websocket_policy_server import WebsocketPolicyServer
    from wan_va.wan_va_server import VA_Server

    init_logger()
    config = copy.deepcopy(VA_CONFIGS["robotwin"])
    config.wan22_pretrained_model_name_or_path = str(Path(args.model_root).expanduser().resolve())
    config.save_root = str(Path(args.save_root).expanduser().resolve())
    config.port = int(args.port)
    config.enable_offload = bool(args.enable_offload)
    if args.host:
        config.host = args.host
    config.rank = int(os.getenv("RANK", 0))
    config.local_rank = int(os.getenv("LOCAL_RANK", 0))
    config.world_size = int(os.getenv("WORLD_SIZE", 1))

    if args.distributed:
        from wan_va.distributed.util import init_distributed

        init_distributed(config.world_size, config.local_rank, config.rank)

    logger.info(
        "Starting RobotWin LingBot-VA server on %s:%s with model root %s",
        config.host,
        config.port,
        config.wan22_pretrained_model_name_or_path,
    )
    model = VA_Server(config)
    if args.distributed:
        run_async_server_mode(model, config.local_rank, config.host, config.port)
    else:
        logger.info("Running direct single-GPU websocket server mode")
        WebsocketPolicyServer(model, host=config.host, port=config.port).serve_forever()


if __name__ == "__main__":
    main()
