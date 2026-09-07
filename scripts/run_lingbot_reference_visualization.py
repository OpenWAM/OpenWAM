from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.integrations import (  # noqa: E402
    LiberoTaskSpec,
    activate_libero_renderer,
    ensure_local_libero_config,
    load_libero_task_init_states,
)
from open_wam.configs import LiberoRendererProfile  # noqa: E402
from open_wam.third_party.lingbot import _ensure_flash_attn_shims  # noqa: E402
from open_wam.utils import seed_everywhere  # noqa: E402

benchmark: Any = None
OffScreenRenderEnv: Any = None
VA_CONFIGS: Any = None
VA_Server: Any = None


LIBERO_OBS_KEYS = (
    "observation.images.agentview_rgb",
    "observation.images.eye_in_hand_rgb",
)


def main() -> None:
    if any(
        argument == "--heng-repo-root" or argument.startswith("--heng-repo-root=")
        for argument in sys.argv[1:]
    ):
        warnings.warn(
            "--heng-repo-root is deprecated; use --reference-repo-root instead.",
            FutureWarning,
            stacklevel=2,
        )
    parser = argparse.ArgumentParser(
        description=(
            "Run the LingBot reference LIBERO exact pipeline for one or more "
            "chunks and save a comparison video."
        )
    )
    parser.add_argument(
        "--reference-repo-root",
        "--heng-repo-root",
        dest="reference_repo_root",
        type=Path,
        required=True,
        help=(
            "Source checkout containing the upstream wan_va package. "
            "--heng-repo-root is a deprecated alias."
        ),
    )
    parser.add_argument("--benchmark", type=str, default="libero_10")
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--max-timestep", type=int, default=800)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument(
        "--output-dir", type=str, default="outputs/libero_exact_visualization_firstpass"
    )
    parser.add_argument("--suffix", type=str, default="lingbot_reference")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--pretrained-root",
        type=str,
        default=None,
        help=(
            "LingBot/Wan base root for the LingBot reference VA_Server. "
            "Overrides placeholder paths in LingBot reference configs."
        ),
    )
    parser.add_argument(
        "--transformer-dir",
        type=str,
        default=None,
        help=(
            "Optional trained transformer directory to mount under the "
            "pretrained root layout expected by the LingBot reference runtime."
        ),
    )
    args = parser.parse_args()

    renderer_config = activate_libero_renderer(
        LiberoRendererProfile.ONLINE_ROLLOUT
    )
    _configure_reference_runtime(args.reference_repo_root)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "LingBot reference comparison runner expects CUDA to be available."
        )
    torch.cuda.set_device(0)

    model = _build_reference_model(
        Path(args.output_dir),
        pretrained_root=Path(args.pretrained_root) if args.pretrained_root else None,
        transformer_dir=Path(args.transformer_dir) if args.transformer_dir else None,
    )
    component_report = _build_reference_component_report(model)
    component_report["libero_renderer"] = renderer_config.to_dict()
    _print_log("load_report", component_report)
    env = None
    try:
        benchmark_instance = benchmark.get_benchmark_dict()[args.benchmark]()
        task = benchmark_instance.get_task(args.task_id)
        prompt = task.language
        env = _construct_single_env(
            {
                "bddl_file_name": benchmark_instance.get_task_bddl_file_path(
                    args.task_id
                ),
                "camera_heights": 128,
                "camera_widths": 128,
                "horizon": max(args.max_timestep + 16, 1000),
                "ignore_done": True,
            }
        )
        if env is None:
            raise RuntimeError(
                "Failed to construct LIBERO OffScreenRenderEnv after 5 retries."
            )
        init_states = _load_reference_init_states(
            args.benchmark, args.task_id, benchmark_instance, task
        )
        first_obs = _init_single_env(
            env, init_states[args.episode_idx % init_states.shape[0]]
        )

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
            ret = model.infer(
                dict(obs=first_obs, prompt=prompt, save_visualization=False)
            )
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
                        copied = {
                            key: np.array(value, copy=True)
                            for key, value in observes.items()
                        }
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
            model.infer(
                dict(
                    obs=key_frame_list,
                    compute_kv_cache=True,
                    imagine=False,
                    state=action,
                )
            )
            _print_log(
                f"chunk_{chunk_count - 1}",
                {
                    "phase": "warmup_done",
                    "frame_st_id_after": int(getattr(model, "frame_st_id", -1)),
                },
            )

        imagined_video = _export_reference_imagined_video(model)

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
            "pipeline": "lingbot_reference",
        }
        summary_path = output_path.with_suffix(".json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        load_report_path = output_path.with_name(f"{output_path.stem}_load_report.json")
        load_report_path.write_text(
            json.dumps(component_report, indent=2), encoding="utf-8"
        )

        print(json.dumps(summary, indent=2))
    finally:
        if env is not None:
            env.close()
        del model
        torch.cuda.empty_cache()


def _configure_reference_runtime(reference_repo_root: Path) -> None:
    """Load the explicitly selected upstream comparison runtime."""

    global benchmark, OffScreenRenderEnv, VA_CONFIGS, VA_Server

    repo_root = reference_repo_root.expanduser().resolve()
    wan_root = repo_root / "wan_va"
    if not repo_root.is_dir():
        raise FileNotFoundError(
            f"LingBot reference source checkout not found: {repo_root}"
        )
    if not wan_root.is_dir():
        raise FileNotFoundError(
            f"LingBot reference wan_va package root not found: {wan_root}"
        )
    for path in (repo_root, wan_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    _ensure_flash_attn_shims()
    ensure_local_libero_config(REPO_ROOT)

    from libero.libero import benchmark as libero_benchmark
    from libero.libero.envs import OffScreenRenderEnv as libero_offscreen_env
    from wan_va.configs import VA_CONFIGS as upstream_va_configs
    from wan_va.wan_va_server import VA_Server as upstream_va_server

    benchmark = libero_benchmark
    OffScreenRenderEnv = libero_offscreen_env
    VA_CONFIGS = upstream_va_configs
    VA_Server = upstream_va_server


def _build_reference_model(
    save_root: Path,
    *,
    pretrained_root: Path | None = None,
    transformer_dir: Path | None = None,
) -> VA_Server:
    config = copy.deepcopy(VA_CONFIGS["libero"])
    config.rank = 0
    config.local_rank = 0
    config.world_size = 1
    config.save_root = str(save_root.resolve())
    if pretrained_root is not None:
        config.wan22_pretrained_model_name_or_path = str(
            pretrained_root.expanduser().resolve()
        )
    if transformer_dir is not None:
        source_transformer_dir = transformer_dir.expanduser().resolve()
        resolved_transformer_dir = _ensure_reference_transformer_dir(
            save_root=save_root,
            pretrained_root=Path(config.wan22_pretrained_model_name_or_path),
            transformer_dir=source_transformer_dir,
        )
        model_root = _prepare_reference_model_root(
            save_root=save_root,
            pretrained_root=Path(config.wan22_pretrained_model_name_or_path),
            transformer_dir=resolved_transformer_dir,
        )
        config.reference_base_pretrained_root = str(
            Path(config.wan22_pretrained_model_name_or_path).resolve()
        )
        config.reference_source_transformer_dir = str(source_transformer_dir)
        config.reference_transformer_dir = str(resolved_transformer_dir)
        config.wan22_pretrained_model_name_or_path = str(model_root)
    else:
        root = Path(config.wan22_pretrained_model_name_or_path).expanduser().resolve()
        config.reference_base_pretrained_root = str(root)
        config.reference_source_transformer_dir = str(root / "transformer")
        config.reference_transformer_dir = str(root / "transformer")
    return VA_Server(config)


def _build_reference_component_report(model: VA_Server) -> dict[str, object]:
    pretrained_root = Path(model.job_config.reference_base_pretrained_root)
    transformer_dir = Path(model.job_config.reference_transformer_dir)
    source_transformer_dir = Path(model.job_config.reference_source_transformer_dir)
    vae_dir = pretrained_root / "vae"
    text_encoder_dir = pretrained_root / "text_encoder"
    tokenizer_dir = pretrained_root / "tokenizer"
    transformer_config = getattr(model.transformer, "config", None)
    return {
        "pipeline": "lingbot_reference",
        "runtime_device": str(model.device),
        "backbone_pretrained_root": str(pretrained_root.resolve()),
        "source_transformer_dir": str(source_transformer_dir.resolve()),
        "transformer_dir": str(transformer_dir.resolve()),
        "transformer_converted_for_reference": (
            source_transformer_dir.resolve() != transformer_dir.resolve()
        ),
        "transformer_config_sha256": _sha256_if_exists(transformer_dir / "config.json"),
        "transformer_weights_sha256": _sha256_if_exists(
            transformer_dir / "diffusion_pytorch_model.safetensors"
        ),
        "vae_dir": str(vae_dir.resolve()),
        "vae_config_sha256": _sha256_if_exists(vae_dir / "config.json"),
        "vae_weights_sha256": _sha256_if_exists(
            vae_dir / "diffusion_pytorch_model.safetensors"
        ),
        "text_encoder_dir": str(text_encoder_dir.resolve()),
        "text_encoder_index_sha256": _sha256_if_exists(
            text_encoder_dir / "model.safetensors.index.json"
        ),
        "tokenizer_dir": str(tokenizer_dir.resolve()),
        "tokenizer_json_sha256": _sha256_if_exists(tokenizer_dir / "tokenizer.json"),
        "spiece_sha256": _sha256_if_exists(tokenizer_dir / "spiece.model"),
        "transformer_class": model.transformer.__class__.__name__,
        "transformer_num_layers": getattr(transformer_config, "num_layers", None),
        "transformer_action_dim": getattr(transformer_config, "action_dim", None),
        "transformer_attn_mode": getattr(transformer_config, "attn_mode", None),
        "transformer_patch_size": list(
            getattr(model.transformer, "patch_size", ()) or ()
        ),
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


def _load_reference_init_states(
    benchmark_name: str,
    task_id: int,
    benchmark_instance,
    task,
):
    config_path = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        libero_config = yaml.safe_load(handle)
    task_spec = LiberoTaskSpec(
        benchmark_name=benchmark_name,
        task_id=task_id,
        task_name=task.name,
        task_language=task.language,
        problem_folder=task.problem_folder,
        bddl_file_path=benchmark_instance.get_task_bddl_file_path(task_id),
        init_states_path=str(
            Path(libero_config["init_states"])
            / task.problem_folder
            / f"{task.name}.pruned_init"
        ),
    )
    return load_libero_task_init_states(task_spec, REPO_ROOT)


def _export_reference_imagined_video(model: VA_Server):
    try:
        video_ret = model.infer(dict(export_imagined_video=True))
    except KeyError as exc:
        if exc.args == ("obs",):
            return None
        raise
    return video_ret.get("video")


def _ensure_reference_transformer_dir(
    *,
    save_root: Path,
    pretrained_root: Path,
    transformer_dir: Path,
) -> Path:
    config_path = transformer_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Transformer config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        source_config = json.load(handle)
    if source_config.get("_class_name") == "WanTransformer3DModel":
        return transformer_dir

    weights_path = transformer_dir / "diffusion_pytorch_model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Only unsharded OpenWAM transformer exports are supported: {weights_path}"
        )
    digest = hashlib.sha256(str(transformer_dir.resolve()).encode("utf-8")).hexdigest()[
        :12
    ]
    converted_dir = save_root.resolve() / "_reference_transformer_converted" / digest
    converted_config_path = converted_dir / "config.json"
    converted_weights_path = converted_dir / "diffusion_pytorch_model.safetensors"
    if converted_config_path.is_file() and converted_weights_path.is_file():
        return converted_dir

    base_config_path = (
        pretrained_root.expanduser().resolve() / "transformer" / "config.json"
    )
    if not base_config_path.is_file():
        raise FileNotFoundError(
            f"LingBot reference transformer base config not found: {base_config_path}"
        )
    converted_dir.mkdir(parents=True, exist_ok=True)
    with base_config_path.open("r", encoding="utf-8") as handle:
        reference_config = json.load(handle)
    if "attn_mode" in source_config:
        reference_config["attn_mode"] = source_config["attn_mode"]

    from safetensors.torch import load_file, save_file

    source_state = load_file(str(weights_path), device="cpu")
    converted_state = {}
    for key, value in source_state.items():
        converted_key = _to_reference_transformer_key(key)
        if converted_key is None:
            continue
        converted_state[converted_key] = value
    with converted_config_path.open("w", encoding="utf-8") as handle:
        json.dump(reference_config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    save_file(converted_state, str(converted_weights_path))
    return converted_dir


def _to_reference_transformer_key(key: str) -> str | None:
    if key.startswith("runtime_stream_adapters."):
        return None
    prefix_pairs = (
        ("time_conditioner.", "condition_embedder."),
        ("text_proj.", "condition_embedder.text_embedder."),
        ("action_time_conditioner.", "condition_embedder_action."),
        ("action_text_proj.", "condition_embedder_action.text_embedder."),
    )
    for source_prefix, target_prefix in prefix_pairs:
        if key.startswith(source_prefix):
            return f"{target_prefix}{key[len(source_prefix) :]}"
    return key


def _prepare_reference_model_root(
    *,
    save_root: Path,
    pretrained_root: Path,
    transformer_dir: Path,
) -> Path:
    pretrained_root = pretrained_root.expanduser().resolve()
    transformer_dir = transformer_dir.expanduser().resolve()
    _require_dir(pretrained_root / "vae")
    _require_dir(pretrained_root / "text_encoder")
    _require_dir(pretrained_root / "tokenizer")
    _require_dir(transformer_dir)

    digest = hashlib.sha256(
        f"{pretrained_root}|{transformer_dir}".encode("utf-8")
    ).hexdigest()[:12]
    model_root = save_root.resolve() / "_reference_model_roots" / digest
    model_root.mkdir(parents=True, exist_ok=True)
    _safe_link_dir(pretrained_root / "vae", model_root / "vae")
    _safe_link_dir(pretrained_root / "text_encoder", model_root / "text_encoder")
    _safe_link_dir(pretrained_root / "tokenizer", model_root / "tokenizer")
    _safe_link_dir(transformer_dir, model_root / "transformer")
    return model_root


def _require_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(
            f"Required LingBot reference model directory not found: {path}"
        )


def _safe_link_dir(source: Path, dest: Path) -> None:
    if dest.is_symlink() and dest.resolve() == source:
        return
    if dest.exists():
        if dest.resolve() == source:
            return
        raise FileExistsError(
            f"Refusing to replace existing LingBot reference model path: {dest}"
        )
    dest.symlink_to(source, target_is_directory=True)


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
        raise RuntimeError(
            "LIBERO env did not return an observation during initialization."
        )
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
    return (
        root
        / benchmark_name
        / f"{task_id}_{safe_prompt}"
        / f"{episode_idx}_{done}_{suffix}.mp4"
    )


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


def add_title_bar(
    img: np.ndarray, text: str, font_scale: float = 0.65, thickness: int = 2
) -> np.ndarray:
    h, w, _ = img.shape
    del h
    bar_height = 36
    title_bar = np.zeros((bar_height, w, 3), dtype=np.uint8)
    (text_w, text_h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
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
            imagined = np.concatenate(
                [np.asarray(chunk) for chunk in imagined_video], axis=0
            )
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
            scale = min(
                target_width / img_frame.shape[1], panel_height / img_frame.shape[0]
            )
            resized_w = max(1, int(img_frame.shape[1] * scale))
            resized_h = max(1, int(img_frame.shape[0] * scale))
            resized_img = cv2.resize(img_frame, (resized_w, resized_h))
            row_imagined = np.zeros((panel_height, target_width, 3), dtype=np.uint8)
            offset_x = (target_width - resized_w) // 2
            offset_y = (panel_height - resized_h) // 2
            row_imagined[
                offset_y : offset_y + resized_h, offset_x : offset_x + resized_w
            ] = resized_img
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

        row_imagined = add_title_bar(
            row_imagined,
            "Imagined (LingBot Reference Decode)",
        )
        full_frame = np.vstack([row_real, row_imagined])
        final_frames.append(np.ascontiguousarray(full_frame))

    imageio.mimsave(str(save_path), final_frames, fps=fps)


if __name__ == "__main__":
    main()
