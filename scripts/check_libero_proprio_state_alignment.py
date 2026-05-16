from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.integrations import (  # noqa: E402
    LiberoTaskSpec,
    ensure_local_libero_config,
    load_libero_task_init_states,
)
from open_wam.data.action_transforms import quaternion_to_axis_angle  # noqa: E402


LIBERO_INIT_STABILIZATION_STEPS = 5
LIBERO_NOOP_ACTION = [0.0] * 7


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare LeRobot LIBERO observation.state[:8] against live robosuite "
            "eef_pos/axisangle/gripper_qpos after LIBERO init-state reset."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--benchmark", type=str, default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument(
        "--dataset-task-index",
        type=int,
        default=None,
        help=(
            "Optional LeRobot dataset task_index override. By default this is "
            "resolved from meta/tasks.jsonl using the LIBERO benchmark task text."
        ),
    )
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--dataset-frame-index", type=int, default=0)
    parser.add_argument("--env-horizon", type=int, default=None)
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument(
        "--init-state-strategy",
        choices=("episode_idx", "state_match"),
        default="state_match",
        help=(
            "`state_match` searches LIBERO init states for the closest robot state to dataset observation.state[:8]. "
            "`episode_idx` uses the raw episode index within the task."
        ),
    )
    parser.add_argument("--state-match-top-k", type=int, default=5)
    args = parser.parse_args()

    task_spec = _resolve_task_spec(args.benchmark, args.task_id)
    dataset_task_index = (
        int(args.dataset_task_index)
        if args.dataset_task_index is not None
        else _dataset_task_index_for_text(args.data_root, task_spec.task_language)
    )
    dataset_episode = _dataset_episode_for_task(args.data_root, dataset_task_index, args.episode_idx)
    dataset_state, selected_dataset_frame = _load_dataset_state(
        args.data_root,
        episode_index=dataset_episode,
        frame_index=args.dataset_frame_index,
    )
    env_report = _load_env_init_state(
        task_spec,
        episode_idx=args.episode_idx,
        dataset_state=dataset_state,
        strategy=args.init_state_strategy,
        top_k=args.state_match_top_k,
        env_horizon=args.env_horizon,
        camera_size=args.camera_size,
    )
    env_state = env_report["selected_state"]
    diff = np.abs(dataset_state - env_state)
    report = {
        "benchmark": args.benchmark,
        "task_id": int(args.task_id),
        "task_language": task_spec.task_language,
        "dataset_task_index": int(dataset_task_index),
        "episode_idx_within_task": int(args.episode_idx),
        "dataset_episode_index": int(dataset_episode),
        "requested_dataset_frame_index": int(args.dataset_frame_index),
        "selected_dataset_frame_index": int(selected_dataset_frame),
        "dataset_state": dataset_state.tolist(),
        "env_proprio": env_state.tolist(),
        "abs_diff": diff.tolist(),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "init_state_strategy": args.init_state_strategy,
        "init_stabilization_steps": LIBERO_INIT_STABILIZATION_STEPS,
        "selected_init_state_index": int(env_report["selected_init_state_index"]),
        "candidate_count": int(env_report["candidate_count"]),
        "top_matches": env_report["top_matches"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def _resolve_task_spec(benchmark_name: str, task_id: int) -> LiberoTaskSpec:
    ensure_local_libero_config(REPO_ROOT)
    from libero.libero import benchmark  # type: ignore
    import yaml
    import os

    benchmark_instance = benchmark.get_benchmark_dict()[benchmark_name]()
    task = benchmark_instance.get_task(task_id)
    config_path = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        libero_config = yaml.safe_load(handle)
    return LiberoTaskSpec(
        benchmark_name=benchmark_name,
        task_id=task_id,
        task_name=task.name,
        task_language=task.language,
        problem_folder=task.problem_folder,
        bddl_file_path=benchmark_instance.get_task_bddl_file_path(task_id),
        init_states_path=str(Path(libero_config["init_states"]) / task.problem_folder / f"{task.name}.pruned_init"),
    )


def _dataset_task_index_for_text(data_root: Path, task_text: str) -> int:
    tasks_path = data_root / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        raise FileNotFoundError(f"Missing LeRobot task metadata: {tasks_path}")
    target = _canonical_task_text(task_text)
    for line in tasks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if _canonical_task_text(str(row["task"])) == target:
            return int(row["task_index"])
    raise ValueError(f"No LeRobot task_index found for task text {task_text!r} in {tasks_path}.")


def _canonical_task_text(task_text: str) -> str:
    return re.sub(r"\s+", " ", task_text.replace("_", " ").strip()).lower()


def _dataset_episode_for_task(data_root: Path, dataset_task_index: int, episode_idx: int) -> int:
    matches: list[int] = []
    for parquet_path in sorted((data_root / "data").glob("chunk-*/episode_*.parquet")):
        table = pq.read_table(parquet_path, columns=["episode_index", "task_index"])
        if table.num_rows == 0:
            continue
        row = table.slice(0, 1).to_pylist()[0]
        if int(row["task_index"]) == int(dataset_task_index):
            matches.append(int(row["episode_index"]))
    if not matches:
        raise ValueError(f"No LeRobot episode found for task_index={dataset_task_index} under {data_root}.")
    return matches[int(episode_idx) % len(matches)]


def _load_dataset_state(data_root: Path, *, episode_index: int, frame_index: int) -> tuple[np.ndarray, int]:
    parquet_path = data_root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing LeRobot episode parquet: {parquet_path}")
    table = pq.read_table(parquet_path, columns=["frame_index", "observation.state"])
    if table.num_rows == 0:
        raise ValueError(f"Empty LeRobot episode parquet: {parquet_path}")
    rows = table.to_pylist()
    selected = next((row for row in rows if int(row["frame_index"]) == int(frame_index)), None)
    if selected is None:
        selected = rows[min(max(0, int(frame_index)), len(rows) - 1)]
    state = np.asarray(selected["observation.state"], dtype=np.float32).reshape(-1)
    if state.shape[0] < 8:
        raise ValueError(f"Expected observation.state dim >= 8, got {state.shape[0]}.")
    return state[:8].astype(np.float32, copy=False), int(selected["frame_index"])


def _load_env_init_state(
    task_spec: LiberoTaskSpec,
    *,
    episode_idx: int,
    dataset_state: np.ndarray,
    strategy: str,
    top_k: int,
    env_horizon: int | None,
    camera_size: int,
) -> dict[str, object]:
    ensure_local_libero_config(REPO_ROOT)
    from libero.libero.envs import OffScreenRenderEnv  # type: ignore

    init_states = load_libero_task_init_states(task_spec)
    env = None
    count = 0
    while env is None and count < 5:
        try:
            kwargs = {
                "bddl_file_name": task_spec.bddl_file_path,
                "camera_heights": int(camera_size),
                "camera_widths": int(camera_size),
            }
            if env_horizon is not None:
                kwargs["horizon"] = int(env_horizon)
            env = OffScreenRenderEnv(**kwargs)
        except Exception as exc:  # pragma: no cover - best-effort retry path
            print(f"construct env failed ({count + 1}/5): {exc}", file=sys.stderr)
            time.sleep(5)
            count += 1
    if env is None:
        raise RuntimeError("Failed to construct LIBERO OffScreenRenderEnv after 5 retries.")
    try:
        candidate_states: list[np.ndarray] = []
        for init_state in init_states:
            env.reset()
            obs = env.set_init_state(init_state)
            if obs is None:
                raise RuntimeError("LIBERO env did not return an observation from set_init_state.")
            for _ in range(LIBERO_INIT_STABILIZATION_STEPS):
                obs, _, _, _ = env.step(LIBERO_NOOP_ACTION)
            candidate_states.append(_extract_libero_eef_axisangle_gripper_state(obs))
        stacked = np.stack(candidate_states, axis=0).astype(np.float32)
        if strategy == "episode_idx":
            selected_index = int(episode_idx) % len(init_states)
        elif strategy == "state_match":
            selected_index = _rank_candidate_states(
                dataset_state=dataset_state,
                candidate_states=stacked,
                top_k=1,
            )[0]["init_state_index"]
        else:  # pragma: no cover - argparse choices guard
            raise ValueError(f"Unsupported init-state strategy: {strategy!r}")
        top_matches = _rank_candidate_states(
            dataset_state=dataset_state,
            candidate_states=stacked,
            top_k=top_k,
        )
        return {
            "selected_state": stacked[selected_index],
            "selected_init_state_index": int(selected_index),
            "candidate_count": int(len(stacked)),
            "top_matches": top_matches,
        }
    finally:
        env.close()


def _rank_candidate_states(
    *,
    dataset_state: np.ndarray,
    candidate_states: np.ndarray,
    top_k: int,
) -> list[dict[str, object]]:
    dataset = np.asarray(dataset_state, dtype=np.float32).reshape(1, -1)
    candidates = np.asarray(candidate_states, dtype=np.float32)
    abs_diff = np.abs(candidates - dataset)
    order = np.argsort(abs_diff.max(axis=1), kind="stable")[: max(1, int(top_k))]
    return [
        {
            "rank": int(rank),
            "init_state_index": int(index),
            "max_abs_diff": float(abs_diff[index].max()),
            "mean_abs_diff": float(abs_diff[index].mean()),
            "candidate_state": candidates[index].astype(float).tolist(),
        }
        for rank, index in enumerate(order)
    ]


def _extract_libero_eef_axisangle_gripper_state(obs) -> np.ndarray:
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
    eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1)
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    if eef_pos.shape[0] != 3:
        raise ValueError(f"Expected LIBERO robot0_eef_pos to have dim 3, got {eef_pos.shape[0]}.")
    if eef_quat.shape[0] != 4:
        raise ValueError(f"Expected LIBERO robot0_eef_quat to have dim 4, got {eef_quat.shape[0]}.")
    if gripper_qpos.shape[0] != 2:
        raise ValueError(f"Expected LIBERO robot0_gripper_qpos to have dim 2, got {gripper_qpos.shape[0]}.")
    axisangle = (
        quaternion_to_axis_angle(torch.from_numpy(eef_quat).to(dtype=torch.float32).unsqueeze(0))[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    return np.concatenate([eef_pos, axisangle, gripper_qpos], axis=0).astype(np.float32, copy=False)


if __name__ == "__main__":
    main()
