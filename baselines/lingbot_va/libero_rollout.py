from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .assets import PreparedModel, build_component_report, prepare_model_root
from .config import CheckpointSpec, EpisodeSpec
from .external import ExternalModules, load_external_modules
from .video import LIBERO_OBS_KEYS, save_libero_rollout_video


@dataclass(frozen=True)
class ChunkTrace:
    chunk_index: int
    first_chunk: bool
    env_timestep_before: int
    env_timestep_after: int
    action_shape: tuple[int, ...]
    start_frame_group: int
    key_frame_count: int
    done_after_chunk: bool
    infer_seconds: float
    warmup_seconds: float | None
    frame_st_id_before: int | None
    frame_st_id_after: int | None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "first_chunk": self.first_chunk,
            "env_timestep_before": self.env_timestep_before,
            "env_timestep_after": self.env_timestep_after,
            "action_shape": list(self.action_shape),
            "start_frame_group": self.start_frame_group,
            "key_frame_count": self.key_frame_count,
            "done_after_chunk": self.done_after_chunk,
            "infer_seconds": self.infer_seconds,
            "warmup_seconds": self.warmup_seconds,
            "frame_st_id_before": self.frame_st_id_before,
            "frame_st_id_after": self.frame_st_id_after,
        }


@dataclass(frozen=True)
class RolloutResult:
    checkpoint_name: str
    source_repo: str
    prepared_model: dict[str, Any]
    benchmark: str
    task_id: int
    prompt: str
    episode_idx: int
    seed: int | None
    sample_id: str | None
    sample_kind: str | None
    sample_index: int | None
    success: bool
    chunk_count: int
    env_timestep: int
    executed_actions: int
    max_timestep: int
    max_chunks: int | None
    video_path: str | None
    summary_path: str
    load_report_path: str
    chunk_trace_path: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_name": self.checkpoint_name,
            "source_repo": self.source_repo,
            "prepared_model": self.prepared_model,
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "prompt": self.prompt,
            "episode_idx": self.episode_idx,
            "seed": self.seed,
            "sample_id": self.sample_id,
            "sample_kind": self.sample_kind,
            "sample_index": self.sample_index,
            "success": self.success,
            "chunk_count": self.chunk_count,
            "env_timestep": self.env_timestep,
            "executed_actions": self.executed_actions,
            "max_timestep": self.max_timestep,
            "max_chunks": self.max_chunks,
            "video_path": self.video_path,
            "summary_path": self.summary_path,
            "load_report_path": self.load_report_path,
            "chunk_trace_path": self.chunk_trace_path,
        }


class LingBotVALiberoRunner:
    """Runs the upstream LingBot-VA `VA_Server` with the upstream LIBERO client loop."""

    def __init__(
        self,
        checkpoint: CheckpointSpec,
        *,
        output_dir: str | Path,
        cuda_device: int = 0,
        video_fps: float = 60.0,
        render_video: bool = True,
    ) -> None:
        self.checkpoint = checkpoint
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.cuda_device = cuda_device
        self.video_fps = video_fps
        self.render_video = render_video
        self.modules: ExternalModules = load_external_modules(checkpoint.source_repo)
        self.prepared_model: PreparedModel | None = None
        self.model: Any | None = None
        self.load_report: dict[str, Any] | None = None

    def __enter__(self) -> "LingBotVALiberoRunner":
        self.load()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()

    def load(self) -> None:
        if self.model is not None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("LingBot-VA LIBERO baseline expects CUDA to be available.")
        torch.cuda.set_device(self.cuda_device)

        self.prepared_model = prepare_model_root(
            name=self.checkpoint.name,
            model_root=self.checkpoint.model_root,
            hf_repo_id=self.checkpoint.hf_repo_id,
            hf_revision=self.checkpoint.hf_revision,
        )

        config = copy.deepcopy(self.modules.VA_CONFIGS["libero"])
        config.rank = 0
        config.local_rank = self.cuda_device
        config.world_size = 1
        config.save_root = str((self.output_dir / "server_state" / self.checkpoint.name).resolve())
        config.wan22_pretrained_model_name_or_path = str(self.prepared_model.model_root)
        if self.checkpoint.enable_offload is not None:
            config.enable_offload = bool(self.checkpoint.enable_offload)
        config.lingbot_va_baseline_source_repo = str(self.modules.source_repo)
        config.lingbot_va_baseline_prepared_model = self.prepared_model.to_json_dict()

        self.model = self.modules.VA_Server(config)
        self.load_report = build_component_report(self.model, self.prepared_model)
        self.load_report["source_repo"] = str(self.modules.source_repo)
        load_report_path = self._checkpoint_load_report_path()
        load_report_path.parent.mkdir(parents=True, exist_ok=True)
        load_report_path.write_text(json.dumps(self.load_report, indent=2, sort_keys=True), encoding="utf-8")

    def close(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        torch.cuda.empty_cache()

    def run_episode(
        self,
        episode: EpisodeSpec,
        *,
        max_timestep: int,
        max_chunks: int | None = None,
    ) -> RolloutResult:
        if self.model is None or self.prepared_model is None:
            self.load()
        assert self.model is not None
        assert self.prepared_model is not None
        assert self.load_report is not None
        if episode.benchmark != "libero_10":
            raise ValueError(f"LingBot-VA LIBERO-LONG baseline only supports benchmark 'libero_10', got {episode.benchmark!r}.")

        if episode.seed is not None:
            self.modules.seed_everywhere(episode.seed)

        benchmark_instance = self.modules.benchmark.get_benchmark_dict()[episode.benchmark]()
        task = benchmark_instance.get_task(episode.task_id)
        prompt = task.language
        env = self._construct_env(
            {
                "bddl_file_name": benchmark_instance.get_task_bddl_file_path(episode.task_id),
                "camera_heights": 128,
                "camera_widths": 128,
            }
        )
        if env is None:
            raise RuntimeError("Failed to construct LIBERO OffScreenRenderEnv after retries.")

        traces: list[ChunkTrace] = []
        full_obs_list: list[dict[str, np.ndarray]] = []
        frame_chunk_ids: list[int] = []
        done = False
        executed_actions = 0
        chunk_count = 0
        first = True
        try:
            init_states = benchmark_instance.get_task_init_states(episode.task_id)
            first_obs = self._init_env(env, init_states[episode.episode_idx % init_states.shape[0]])

            self.model.infer(dict(reset=True, prompt=prompt))

            while env.env.timestep < max_timestep and not done:
                if max_chunks is not None and chunk_count >= max_chunks:
                    break

                timestep_before = int(env.env.timestep)
                frame_st_before = _optional_int(getattr(self.model, "frame_st_id", None))
                infer_started = time.perf_counter()
                ret = self.model.infer(dict(obs=first_obs, prompt=prompt, save_visualization=False))
                infer_seconds = time.perf_counter() - infer_started
                action = ret["action"]
                action_shape = tuple(int(dim) for dim in action.shape)

                key_frame_list: list[dict[str, np.ndarray]] = []
                assert action.shape[2] % 4 == 0
                action_per_frame = action.shape[2] // 4
                start_idx = 1 if first else 0
                for frame_group in range(start_idx, action.shape[1]):
                    for action_index in range(action.shape[2]):
                        ee_action = action[:, frame_group, action_index]
                        observes, done = self._env_one_step(env, ee_action)
                        executed_actions += 1
                        if done:
                            break
                        if (action_index + 1) % action_per_frame == 0:
                            copied = {key: np.array(value, copy=True) for key, value in observes.items()}
                            full_obs_list.append(copied)
                            frame_chunk_ids.append(chunk_count)
                            key_frame_list.append(copied)
                    if done:
                        break

                warmup_seconds: float | None = None
                if not done and (max_chunks is None or chunk_count + 1 < max_chunks) and key_frame_list:
                    warmup_started = time.perf_counter()
                    self.model.infer(dict(obs=key_frame_list, compute_kv_cache=True, imagine=False, state=action))
                    warmup_seconds = time.perf_counter() - warmup_started

                traces.append(
                    ChunkTrace(
                        chunk_index=chunk_count,
                        first_chunk=first,
                        env_timestep_before=timestep_before,
                        env_timestep_after=int(env.env.timestep),
                        action_shape=action_shape,
                        start_frame_group=start_idx,
                        key_frame_count=len(key_frame_list),
                        done_after_chunk=bool(done),
                        infer_seconds=infer_seconds,
                        warmup_seconds=warmup_seconds,
                        frame_st_id_before=frame_st_before,
                        frame_st_id_after=_optional_int(getattr(self.model, "frame_st_id", None)),
                    )
                )

                chunk_count += 1
                first = False
                if done or not key_frame_list:
                    break
        finally:
            env.close()

        output_path = self._rollout_video_path(
            episode=episode,
            prompt=prompt,
            success=done,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        video_path: str | None = None
        if self.render_video:
            save_libero_rollout_video(
                real_obs_list=full_obs_list,
                save_path=output_path,
                fps=self.video_fps,
                frame_chunk_ids=frame_chunk_ids,
                title=f"{self.checkpoint.name} | task={episode.task_id} ep={episode.episode_idx}",
            )
            video_path = str(output_path)

        trace_path = output_path.with_name(f"{output_path.stem}_chunks.json")
        trace_path.write_text(
            json.dumps([trace.to_json_dict() for trace in traces], indent=2, sort_keys=True),
            encoding="utf-8",
        )

        summary_path = output_path.with_suffix(".json")
        result = RolloutResult(
            checkpoint_name=self.checkpoint.name,
            source_repo=str(self.modules.source_repo),
            prepared_model=self.prepared_model.to_json_dict(),
            benchmark=episode.benchmark,
            task_id=episode.task_id,
            prompt=prompt,
            episode_idx=episode.episode_idx,
            seed=episode.seed,
            sample_id=episode.sample_id,
            sample_kind=episode.sample_kind,
            sample_index=episode.sample_index,
            success=bool(done),
            chunk_count=chunk_count,
            env_timestep=int(traces[-1].env_timestep_after if traces else 0),
            executed_actions=executed_actions,
            max_timestep=max_timestep,
            max_chunks=max_chunks,
            video_path=video_path,
            summary_path=str(summary_path),
            load_report_path=str(self._checkpoint_load_report_path()),
            chunk_trace_path=str(trace_path),
        )
        summary_path.write_text(json.dumps(result.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return result

    def _construct_env(self, env_args: dict[str, Any]):
        env = None
        for attempt in range(5):
            try:
                env = self.modules.OffScreenRenderEnv(**env_args)
                break
            except Exception as exc:  # pragma: no cover - simulator retry path
                print(f"construct env failed ({attempt + 1}/5): {exc}", flush=True)
                time.sleep(5)
        return env

    def _init_env(self, env, init_state) -> dict[str, np.ndarray]:
        env.reset()
        env.set_init_state(init_state)
        obs = None
        for _ in range(5):
            obs, _, _, _ = env.step([0.0] * 7)
        if obs is None:
            raise RuntimeError("LIBERO env did not return an observation during initialization.")
        return _extract_obs(obs)

    def _env_one_step(self, env, action) -> tuple[dict[str, np.ndarray], bool]:
        obs, _, done, _ = env.step(action)
        return _extract_obs(obs), bool(done)

    def _rollout_video_path(self, *, episode: EpisodeSpec, prompt: str, success: bool) -> Path:
        safe_prompt = prompt.replace(" ", "_")
        safe_name = self.checkpoint.name.replace("/", "_")
        sample_prefix = f"{_safe_path_component(episode.sample_id)}_" if episode.sample_id else ""
        return (
            self.output_dir
            / "rollouts"
            / episode.benchmark
            / f"{episode.task_id}_{safe_prompt}"
            / f"{sample_prefix}{episode.episode_idx}_{success}_{safe_name}_seed{episode.seed}.mp4"
        )

    def _checkpoint_load_report_path(self) -> Path:
        return self.output_dir / "load_reports" / f"{self.checkpoint.name}.json"


def _extract_obs(obs) -> dict[str, np.ndarray]:
    return {
        LIBERO_OBS_KEYS[0]: np.ascontiguousarray(obs["agentview_image"][::-1]),
        LIBERO_OBS_KEYS[1]: np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1]),
    }


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _safe_path_component(value: str | None) -> str:
    if not value:
        return ""
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value))
