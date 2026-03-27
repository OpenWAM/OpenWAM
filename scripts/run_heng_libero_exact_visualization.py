from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
HENG_REPO_ROOT = Path("/path/to/private-resource")
HENG_WAN_ROOT = HENG_REPO_ROOT / "wan_va"

for path in (SRC_ROOT, HENG_REPO_ROOT, HENG_WAN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from open_wam.third_party.lingbot import _ensure_flash_attn_shims  # noqa: E402
from open_wam.integrations import ensure_local_libero_config  # noqa: E402
from open_wam.utils import seed_everywhere  # noqa: E402

_ensure_flash_attn_shims()
ensure_local_libero_config(REPO_ROOT)

from libero.libero import benchmark  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from wan_va.configs import VA_CONFIGS  # noqa: E402
from wan_va.wan_va_server import VA_Server  # noqa: E402


LIBERO_OBS_KEYS = (
    "observation.images.agentview_rgb",
    "observation.images.eye_in_hand_rgb",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Heng's LIBERO exact pipeline for one or more chunks and save a comparison video."
    )
    parser.add_argument("--benchmark", type=str, default="libero_10")
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--max-timestep", type=int, default=800)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument("--output-dir", type=str, default="outputs/libero_exact_visualization_firstpass")
    parser.add_argument("--suffix", type=str, default="heng")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Heng comparison runner expects CUDA to be available.")
    torch.cuda.set_device(0)

    model = _build_heng_model(Path(args.output_dir))
    component_report = _build_heng_component_report(model)
    _print_log("load_report", component_report)
    env = None
    try:
        benchmark_instance = benchmark.get_benchmark_dict()[args.benchmark]()
        task = benchmark_instance.get_task(args.task_id)
        prompt = task.language
        env = _construct_single_env(
            {
                "bddl_file_name": benchmark_instance.get_task_bddl_file_path(args.task_id),
                "camera_heights": 128,
                "camera_widths": 128,
            }
        )
        if env is None:
            raise RuntimeError("Failed to construct LIBERO OffScreenRenderEnv after 5 retries.")
        init_states = benchmark_instance.get_task_init_states(args.task_id)
        first_obs = _init_single_env(env, init_states[args.episode_idx % init_states.shape[0]])

        model.infer(dict(reset=True, prompt=prompt, n_view=2))

        full_obs_list: list[dict[str, np.ndarray]] = [
            {key: np.array(value, copy=True) for key, value in first_obs.items()}
        ]
        done = False
        first = True
        chunk_count = 0

        while env.env.timestep < args.max_timestep and not done:
            if args.max_chunks is not None and chunk_count >= args.max_chunks:
                break

            if args.seed is not None:
                seed_everywhere(args.seed + chunk_count)
            timestep_before = int(env.env.timestep)
            ret = model.infer(dict(obs=first_obs, prompt=prompt, save_visualization=False))
            action = ret["action"]
            _print_log(
                f"chunk_{chunk_count}",
                {
                    "phase": "infer",
                    "first_chunk": first,
                    "env_timestep_before": timestep_before,
                    "action_shape": list(action.shape),
                    "frame_st_id": int(getattr(model, "frame_st_id", -1)),
                    "action_preview": _preview_tensor(action[0, 0, 0]),
                },
            )

            key_frame_list: list[dict[str, np.ndarray]] = []
            assert action.shape[2] % 4 == 0
            action_per_frame = action.shape[2] // 4
            start_idx = 1 if first else 0
            for frame_group in range(start_idx, action.shape[1]):
                for action_index in range(action.shape[2]):
                    ee_action = action[:, frame_group, action_index]
                    observes, done = _env_one_step(env, ee_action)
                    if done:
                        break
                    if (action_index + 1) % action_per_frame == 0:
                        copied = {key: np.array(value, copy=True) for key, value in observes.items()}
                        full_obs_list.append(copied)
                        key_frame_list.append(copied)
                if done:
                    break

            chunk_count += 1
            _print_log(
                f"chunk_{chunk_count - 1}",
                {
                    "phase": "env_rollout",
                    "env_timestep_after": int(env.env.timestep),
                    "done": bool(done),
                    "key_frame_count": len(key_frame_list),
                    "start_frame_group": start_idx,
                },
            )
            first = False

            if done:
                break
            if args.max_chunks is not None and chunk_count >= args.max_chunks:
                break
            if not key_frame_list:
                break

            _print_log(
                f"chunk_{chunk_count - 1}",
                {
                    "phase": "warmup_prepare",
                    "key_frame_count": len(key_frame_list),
                    "action_state_shape": list(action.shape),
                    "frame_st_id_before": int(getattr(model, "frame_st_id", -1)),
                },
            )
            model.infer(dict(obs=key_frame_list, compute_kv_cache=True, imagine=False, state=action))
            _print_log(
                f"chunk_{chunk_count - 1}",
                {
                    "phase": "warmup_done",
                    "frame_st_id_after": int(getattr(model, "frame_st_id", -1)),
                },
            )

        video_ret = model.infer(dict(export_imagined_video=True))
        imagined_video = video_ret.get("video")

        output_path = _build_output_path(
            root=Path(args.output_dir),
            benchmark_name=args.benchmark,
            task_id=args.task_id,
            prompt=prompt,
            episode_idx=args.episode_idx,
            done=done,
            suffix=args.suffix,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_libero_comparison_video(
            real_obs_list=full_obs_list,
            imagined_video=imagined_video,
            save_path=output_path,
            fps=args.video_fps,
        )

        summary = {
            "benchmark": args.benchmark,
            "task_id": args.task_id,
            "prompt": prompt,
            "episode_idx": args.episode_idx,
            "success": bool(done),
            "chunk_count": chunk_count,
            "env_timestep": int(env.env.timestep),
            "seed": args.seed,
            "video_path": str(output_path.resolve()),
            "pipeline": "heng",
        }
        summary_path = output_path.with_suffix(".json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        load_report_path = output_path.with_name(f"{output_path.stem}_load_report.json")
        load_report_path.write_text(json.dumps(component_report, indent=2), encoding="utf-8")

        print(json.dumps(summary, indent=2))
    finally:
        if env is not None:
            env.close()
        del model
        torch.cuda.empty_cache()


def _build_heng_model(save_root: Path) -> VA_Server:
    config = copy.deepcopy(VA_CONFIGS["libero"])
    config.rank = 0
    config.local_rank = 0
    config.world_size = 1
    config.save_root = str(save_root.resolve())
    return VA_Server(config)


def _build_heng_component_report(model: VA_Server) -> dict[str, object]:
    transformer_dir = Path(model.job_config.transformer_override_path) / "transformer"
    pretrained_root = Path(model.job_config.wan22_pretrained_model_name_or_path)
    vae_dir = pretrained_root / "vae"
    text_encoder_dir = pretrained_root / "text_encoder"
    tokenizer_dir = pretrained_root / "tokenizer"
    transformer_config = getattr(model.transformer, "config", None)
    return {
        "pipeline": "heng",
        "runtime_device": str(model.device),
        "backbone_pretrained_root": str(pretrained_root.resolve()),
        "transformer_dir": str(transformer_dir.resolve()),
        "transformer_config_sha256": _sha256_if_exists(transformer_dir / "config.json"),
        "transformer_weights_sha256": _sha256_if_exists(transformer_dir / "diffusion_pytorch_model.safetensors"),
        "vae_dir": str(vae_dir.resolve()),
        "vae_config_sha256": _sha256_if_exists(vae_dir / "config.json"),
        "vae_weights_sha256": _sha256_if_exists(vae_dir / "diffusion_pytorch_model.safetensors"),
        "text_encoder_dir": str(text_encoder_dir.resolve()),
        "text_encoder_index_sha256": _sha256_if_exists(text_encoder_dir / "model.safetensors.index.json"),
        "tokenizer_dir": str(tokenizer_dir.resolve()),
        "tokenizer_json_sha256": _sha256_if_exists(tokenizer_dir / "tokenizer.json"),
        "spiece_sha256": _sha256_if_exists(tokenizer_dir / "spiece.model"),
        "transformer_class": model.transformer.__class__.__name__,
        "transformer_num_layers": getattr(transformer_config, "num_layers", None),
        "transformer_action_dim": getattr(transformer_config, "action_dim", None),
        "transformer_attn_mode": getattr(transformer_config, "attn_mode", None),
        "transformer_patch_size": list(getattr(model.transformer, "patch_size", ()) or ()),
        "max_text_tokens": 512,
        "frame_chunk_size": int(model.job_config.frame_chunk_size),
        "action_per_frame": int(model.job_config.action_per_frame),
        "enable_offload": bool(model.enable_offload),
        "action_decoder_class": "native_lingbot_transformer_head",
        "action_decoder_trainable_params": None,
        "policy_variant_class": "VA_Server",
        "runtime_mode": "server_exact",
        "exact_inference_uses_reference_transformer_only": True,
        "visual_tower_decoder_bypassed_in_exact_mode": True,
        "exact_action_adapter_enabled": True,
    }


def _construct_single_env(env_args):
    count = 0
    env = None
    while env is None and count < 5:
        try:
            env = OffScreenRenderEnv(**env_args)
        except Exception as exc:  # pragma: no cover - best-effort retry path
            print(f"construct env failed ({count + 1}/5): {exc}")
            time.sleep(5)
            count += 1
    return env


def _extract_obs(obs) -> dict[str, np.ndarray]:
    return {
        LIBERO_OBS_KEYS[0]: np.ascontiguousarray(obs["agentview_image"][::-1]),
        LIBERO_OBS_KEYS[1]: np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1]),
    }


def _init_single_env(env_in, init_state) -> dict[str, np.ndarray]:
    env_in.reset()
    env_in.set_init_state(init_state)
    obs = None
    for _ in range(5):
        obs, _, _, _ = env_in.step([0.0] * 7)
    if obs is None:
        raise RuntimeError("LIBERO env did not return an observation during initialization.")
    return _extract_obs(obs)


def _env_one_step(env_in, action):
    obs, _, done, _ = env_in.step(action)
    return _extract_obs(obs), done


def _build_output_path(
    *,
    root: Path,
    benchmark_name: str,
    task_id: int,
    prompt: str,
    episode_idx: int,
    done: bool,
    suffix: str,
) -> Path:
    safe_prompt = prompt.replace(" ", "_")
    return root / benchmark_name / f"{task_id}_{safe_prompt}" / f"{episode_idx}_{done}_{suffix}.mp4"


def _sha256_if_exists(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _print_log(label: str, payload: dict[str, object]) -> None:
    print(f"[{label}] {json.dumps(payload, sort_keys=True, default=str)}")


def _preview_tensor(tensor, *, limit: int = 8) -> list[float]:
    if isinstance(tensor, torch.Tensor):
        flat = tensor.detach().reshape(-1).to(dtype=torch.float32).cpu().tolist()
    else:
        flat = np.asarray(tensor, dtype=np.float32).reshape(-1).tolist()
    return [float(value) for value in flat[:limit]]


def add_title_bar(img: np.ndarray, text: str, font_scale: float = 0.65, thickness: int = 2) -> np.ndarray:
    h, w, _ = img.shape
    del h
    bar_height = 36
    title_bar = np.zeros((bar_height, w, 3), dtype=np.uint8)
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = (w - text_w) // 2
    text_y = (bar_height + text_h) // 2 - 4
    cv2.putText(
        title_bar,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return np.vstack([title_bar, img])


def save_libero_comparison_video(
    *,
    real_obs_list: list[dict[str, np.ndarray]],
    imagined_video,
    save_path: Path,
    fps: float,
) -> None:
    if not real_obs_list:
        return

    imagined = None
    if imagined_video is not None:
        if hasattr(imagined_video, "detach"):
            imagined = imagined_video.detach().cpu().numpy()
        elif isinstance(imagined_video, (list, tuple)):
            imagined = np.concatenate([np.asarray(chunk) for chunk in imagined_video], axis=0)
        else:
            imagined = np.asarray(imagined_video)
        while imagined.ndim > 4 and imagined.shape[0] == 1:
            imagined = imagined[0]

    final_frames: list[np.ndarray] = []
    panel_height = 300
    for index, obs in enumerate(real_obs_list):
        agentview = np.ascontiguousarray(obs[LIBERO_OBS_KEYS[0]])
        wrist = np.ascontiguousarray(obs[LIBERO_OBS_KEYS[1]])
        row_real = np.hstack([agentview, wrist])
        row_real = add_title_bar(row_real, "Real (AgentView / Wrist)")
        target_width = row_real.shape[1]

        if imagined is not None and index < len(imagined):
            img_frame = np.asarray(imagined[index])
            if img_frame.dtype != np.uint8 and float(img_frame.max()) <= 1.0001:
                img_frame = (img_frame * 255).astype(np.uint8)
            elif img_frame.dtype != np.uint8:
                img_frame = img_frame.astype(np.uint8)
            scale = min(target_width / img_frame.shape[1], panel_height / img_frame.shape[0])
            resized_w = max(1, int(img_frame.shape[1] * scale))
            resized_h = max(1, int(img_frame.shape[0] * scale))
            resized_img = cv2.resize(img_frame, (resized_w, resized_h))
            row_imagined = np.zeros((panel_height, target_width, 3), dtype=np.uint8)
            offset_x = (target_width - resized_w) // 2
            offset_y = (panel_height - resized_h) // 2
            row_imagined[offset_y : offset_y + resized_h, offset_x : offset_x + resized_w] = resized_img
        else:
            row_imagined = np.zeros((panel_height, target_width, 3), dtype=np.uint8)
            cv2.putText(
                row_imagined,
                "No imagined video",
                (max(10, target_width // 2 - 140), 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (120, 120, 120),
                2,
            )

        row_imagined = add_title_bar(row_imagined, "Imagined (Heng Server Decode)")
        full_frame = np.vstack([row_real, row_imagined])
        final_frames.append(np.ascontiguousarray(full_frame))

    imageio.mimsave(str(save_path), final_frames, fps=fps)


if __name__ == "__main__":
    main()
