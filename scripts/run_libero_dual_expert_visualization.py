from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.evals.libero_dual_expert_rollout import (
    DualExpertLiberoEpisodeOptions,
    construct_dual_expert_libero_env,
    resolve_dual_expert_libero_task_resources,
    run_dual_expert_libero_episode,
)
from open_wam.evals.libero_dual_expert_runtime import (
    CURRENT_FRONTEND_ENCODE_MODE,
    DEPRECATED_FRONTEND_ENCODE_MODE,
    DUAL_EXPERT_GJD_ACTION_ROUTES,
    DualExpertLiberoLoadOptions,
    load_dual_expert_libero_runtime,
)
from open_wam.runtime.checkpoints import CheckpointCompatibilityPolicy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one LIBERO rollout with a DualExpert policy and save a rollout video."
    )
    parser.add_argument(
        "--cfg",
        "--config",
        dest="config",
        type=str,
        default="configs/experiments/dual_expert_libero_joint.yaml",
    )
    parser.add_argument("--benchmark", type=str, default="libero_10")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint file, checkpoint_step_* directory, or run directory. "
            "If omitted, use top-level checkpoint_path in the config, then infer from backbone.transformer_subdir."
        ),
    )
    parser.add_argument(
        "--allow-partial-checkpoint",
        action="store_true",
        help=(
            "Permit missing or unexpected checkpoint keys for migration "
            "diagnostics. Standard rollouts remain strict."
        ),
    )
    parser.add_argument(
        "--merge-checkpoint-runtime-config",
        action="store_true",
        help=(
            "Opt into merging the checkpoint's resolved_config.yaml before rollout. "
            "By default this script treats --cfg as the rollout contract and only uses "
            "the checkpoint directory for weights/exported transformer assets, keeping "
            "old checkpoints with stale resolved_config.yaml files usable."
        ),
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        help="Apply a config override such as `--set policy_variant.generalist_mode_text_token=true`.",
    )
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--max-timestep", type=int, default=800)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--raw-window-frames", type=int, default=None)
    parser.add_argument(
        "--execute-action-steps",
        type=int,
        default=None,
        help=(
            "Execute only the first N predicted actions from each DualExpert chunk before replanning. "
            "Defaults to the full action horizon. N must be positive, <= action_horizon, "
            "and aligned to action_per_frame."
        ),
    )
    parser.add_argument(
        "--execute-frame-chunk-size",
        type=int,
        default=None,
        help=(
            "Execute only the first N latent-frame groups from each DualExpert chunk before replanning. "
            "This preserves the model's configured inference.frame_chunk_size and maps to "
            "N * action_per_frame executed actions."
        ),
    )
    parser.add_argument(
        "--dual-expert-rollout-frame-chunk-size",
        "--mot-rollout-frame-chunk-size",
        dest="dual_expert_rollout_frame_chunk_size",
        type=int,
        default=None,
        help=(
            "Override DualExpert's internal inference chunk to N latent frames. Unlike "
            "--execute-frame-chunk-size, this reduces the generated video frames and action horizon "
            "inside the policy while preserving the checkpoint's action_per_frame."
        ),
    )
    parser.add_argument(
        "--dual-expert-inference-window-size",
        "--mot-inference-window-size",
        dest="dual_expert_inference_window_size",
        type=int,
        default=None,
        help=(
            "Optional DualExpert rollout attention/cache window override in latent-frame block units. "
            "For legacy split-cache modes this overrides the LingBot slot-pool attn_window; "
            "for packed coupling modes it overrides the training_config.window_size used at inference."
        ),
    )
    parser.add_argument(
        "--dual-expert-action-only-rollout",
        "--mot-action-only-rollout",
        dest="dual_expert_action_only_rollout",
        action="store_true",
        help=(
            "Skip imagined-video denoising during DualExpert rollout and produce actions only. "
            "Supported only for action_then_video and decoupled_same_step couplings."
        ),
    )
    parser.add_argument(
        "--dual-expert-generalist-rollout-mode",
        "--mot-generalist-rollout-mode",
        dest="dual_expert_generalist_rollout_mode",
        metavar="{joint,vanilla_joint_rollout}",
        default=None,
        help=(
            "Optional dual-expert GJD rollout mode forwarded to PolicyInferContext. "
            "Live sim rollout supports only joint/vanilla modes; FDM/IDM and "
            "clean-action diagnostic modes require offline GT action/video tensors. "
            "`--dual-expert-action-only-rollout` is not an IDM substitute."
        ),
    )
    parser.add_argument(
        "--dual-expert-gjd-action-route",
        "--mot-gjd-action-route",
        dest="dual_expert_gjd_action_route",
        choices=sorted(DUAL_EXPERT_GJD_ACTION_ROUTES),
        default="joint",
        help=(
            "Diagnostic dual-expert GJD live-sim action route. `joint` is the maintained "
            "normal rollout. `joint_video_then_idm` first generates the current "
            "video chunk with joint denoising, then reruns IDM from the same "
            "pre-step state using only that generated video as clean condition "
            "and executes the IDM action chunk."
        ),
    )
    parser.add_argument(
        "--frontend-encode-mode",
        choices=(DEPRECATED_FRONTEND_ENCODE_MODE, CURRENT_FRONTEND_ENCODE_MODE),
        default=CURRENT_FRONTEND_ENCODE_MODE,
        help=(
            "RGB-to-latent frontend mode. The current supported rollout contract is "
            "`lingbot_streaming_vae`, which keeps the Wan VAE stream cache alive and "
            "encodes only newly executed env observations between chunks. `rolling_offline` "
            "is deprecated historical compatibility and requires "
            "`--allow-deprecated-frontend-encode-mode`."
        ),
    )
    parser.add_argument(
        "--startup-model-obs-frames",
        type=int,
        default=1,
        help=(
            "Number of initial observations fed to the model on chunk 0. "
            "Defaults to 1 to match Method-1 exact startup; the rolling window "
            "used after chunk 0 still keeps `--raw-window-frames`."
        ),
    )
    parser.add_argument(
        "--startup-env-init-steps",
        type=int,
        default=5,
        help=(
            "Number of zero-action environment steps before chunk 0. "
            "Defaults to 5 to match Method-1 exact startup; if smaller than "
            "--startup-model-obs-frames, it is raised to keep enough observations."
        ),
    )
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument("--output-dir", type=str, default="outputs/libero_dual_expert_visualization")
    parser.add_argument("--suffix", type=str, default="open_wam_dual_expert")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-rollout-video", action="store_true")
    parser.add_argument(
        "--max-imagined-latent-frames",
        type=int,
        default=None,
        help=(
            "Optional cap on predicted latent frames retained for the imagined-video panel. "
            "Default keeps all imagined latent frames for full comparison videos. "
            "Use 0 to disable imagined-video decode for long memory-constrained rollouts."
        ),
    )
    parser.add_argument("--runtime-device", type=str, default=None)
    parser.add_argument("--action-device", type=str, default=None)
    parser.add_argument("--frontend-device", type=str, default=None)
    parser.add_argument("--decode-device", type=str, default=None)
    parser.add_argument(
        "--reset-policy-state-each-chunk",
        action="store_true",
        help=(
            "Debug/compatibility mode matching the original standalone DualExpert script: "
            "rebuild the DualExpert policy state for each observation window instead of carrying caches across chunks."
        ),
    )
    parser.add_argument(
        "--allow-deprecated-libero-config",
        action="store_true",
        help=(
            "Allow historical LIBERO dual-expert configs that do not match the current strict fixed-128, "
            "one-frame, proprio-conditioned training/eval paradigm."
        ),
    )
    parser.add_argument(
        "--allow-deprecated-frontend-encode-mode",
        action="store_true",
        help=(
            "Allow non-lingbot_streaming_vae frontend encode modes only for historical debugging. "
            "Current dual-expert rollout comparisons should not use this."
        ),
    )
    args = parser.parse_args()
    runtime = load_dual_expert_libero_runtime(
        DualExpertLiberoLoadOptions(
            config=args.config,
            checkpoint=args.checkpoint,
            merge_checkpoint_runtime_config=bool(
                args.merge_checkpoint_runtime_config
            ),
            set_overrides=tuple(args.set_overrides),
            source="run_libero_dual_expert_visualization.py",
            checkpoint_error=(
                "DualExpert visualization requires a trained checkpoint. Pass "
                "`--checkpoint`, set top-level `checkpoint_path` in the config, "
                "or point `backbone.transformer_subdir` at an exported checkpoint."
            ),
            raw_window_frames=args.raw_window_frames,
            startup_model_obs_frames=args.startup_model_obs_frames,
            startup_env_init_steps=args.startup_env_init_steps,
            dual_expert_inference_window_size=args.dual_expert_inference_window_size,
            dual_expert_rollout_frame_chunk_size=args.dual_expert_rollout_frame_chunk_size,
            dual_expert_action_only_rollout=bool(args.dual_expert_action_only_rollout),
            dual_expert_generalist_rollout_mode=args.dual_expert_generalist_rollout_mode,
            dual_expert_gjd_action_route=args.dual_expert_gjd_action_route,
            execute_action_steps=args.execute_action_steps,
            execute_frame_chunk_size=args.execute_frame_chunk_size,
            frontend_encode_mode=args.frontend_encode_mode,
            reset_policy_state_each_chunk=bool(
                args.reset_policy_state_each_chunk
            ),
            runtime_device=args.runtime_device,
            action_device=args.action_device,
            frontend_device=args.frontend_device,
            decode_device=args.decode_device,
            allow_deprecated_libero_config=bool(
                args.allow_deprecated_libero_config
            ),
            allow_deprecated_frontend_encode_mode=bool(
                args.allow_deprecated_frontend_encode_mode
            ),
            checkpoint_load_policy=(
                CheckpointCompatibilityPolicy.ALLOW_PARTIAL
                if args.allow_partial_checkpoint
                else CheckpointCompatibilityPolicy.STRICT
            ),
        )
    )
    task_resources = resolve_dual_expert_libero_task_resources(
        args.benchmark,
        args.task_id,
    )
    env = construct_dual_expert_libero_env(task_resources.task_spec)
    episode = DualExpertLiberoEpisodeOptions(
        benchmark=args.benchmark,
        task_id=args.task_id,
        episode_idx=args.episode_idx,
        max_timestep=args.max_timestep,
        max_chunks=args.max_chunks,
        execute_action_steps=args.execute_action_steps,
        execute_frame_chunk_size=args.execute_frame_chunk_size,
        dual_expert_rollout_frame_chunk_size=args.dual_expert_rollout_frame_chunk_size,
        dual_expert_inference_window_size=args.dual_expert_inference_window_size,
        dual_expert_action_only_rollout=bool(args.dual_expert_action_only_rollout),
        dual_expert_generalist_rollout_mode=args.dual_expert_generalist_rollout_mode,
        dual_expert_gjd_action_route=args.dual_expert_gjd_action_route,
        reset_policy_state_each_chunk=bool(args.reset_policy_state_each_chunk),
        max_imagined_latent_frames=args.max_imagined_latent_frames,
        output_dir=args.output_dir,
        suffix=args.suffix,
        video_fps=args.video_fps,
        seed=args.seed,
        save_rollout_video=bool(args.save_rollout_video),
    )
    run_dual_expert_libero_episode(
        episode,
        runtime,
        task_resources,
        env,
        include_episode_coordinates=False,
        close_env_after_rollout=True,
    )


if __name__ == "__main__":
    main()
