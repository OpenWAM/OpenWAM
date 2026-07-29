from __future__ import annotations

import json
import math
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import imageio.v2 as imageio
import numpy as np

from .contracts import CandidateTrajectory


_CANDIDATE_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_LABEL_COLORS = (
    (230, 57, 70),
    (29, 53, 87),
    (42, 157, 143),
    (244, 162, 97),
    (131, 56, 236),
    (255, 183, 3),
    (69, 123, 157),
    (0, 109, 119),
)


class GeminiPolicyPriorHintMode(str, Enum):
    """How much candidate-role information to expose in Gemini prompts."""

    CONSERVATIVE = "conservative"
    NEUTRAL = "neutral"
    BLIND = "blind"


@dataclass(frozen=True)
class GeminiVlmRerankConfig:
    """Configuration for Gemini-backed candidate video reranking."""

    output_dir: Path
    api_key_env: str = "GEMINI_API_KEY"
    model: str = "gemini-3.5-flash"
    video_fps: float = 4.0
    request_timeout_seconds: float = 120.0
    max_candidates: int = 8
    delete_uploaded_files: bool = False
    policy_prior_hint_mode: GeminiPolicyPriorHintMode | str = GeminiPolicyPriorHintMode.CONSERVATIVE

    def __post_init__(self) -> None:
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive.")
        if self.max_candidates > len(_CANDIDATE_LABELS):
            raise ValueError(f"max_candidates cannot exceed {len(_CANDIDATE_LABELS)}.")
        if self.video_fps <= 0:
            raise ValueError("video_fps must be positive.")
        prior_hint_mode = GeminiPolicyPriorHintMode(self.policy_prior_hint_mode)
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser().resolve())
        object.__setattr__(self, "policy_prior_hint_mode", prior_hint_mode)


class GeminiRestClient:
    """Small stdlib-only Gemini REST client for Files API + structured JSON."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = "GEMINI_API_KEY",
        base_url: str = "https://generativelanguage.googleapis.com",
        timeout_seconds: float = 120.0,
        file_activation_timeout_seconds: float = 120.0,
        file_activation_poll_seconds: float = 1.0,
    ) -> None:
        resolved_key = api_key or os.environ.get(api_key_env)
        if not resolved_key:
            raise RuntimeError(
                f"Gemini VLM reranking requires an API key in ${api_key_env}. "
                "The key is read at runtime and is never written to outputs."
            )
        self.api_key = resolved_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.file_activation_timeout_seconds = float(file_activation_timeout_seconds)
        self.file_activation_poll_seconds = float(file_activation_poll_seconds)

    def upload_file(self, file_path: Path, *, mime_type: str | None = None, display_name: str | None = None) -> dict[str, Any]:
        path = Path(file_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Gemini upload file does not exist: {path}")
        resolved_mime = mime_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        start_url = f"{self.base_url}/upload/v1beta/files"
        start_body = json.dumps({"file": {"display_name": display_name or path.name}}).encode("utf-8")
        start_request = urllib.request.Request(
            start_url,
            data=start_body,
            method="POST",
            headers={
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(len(data)),
                "X-Goog-Upload-Header-Content-Type": resolved_mime,
            },
        )
        with _urlopen_json_errors(start_request, timeout=self.timeout_seconds) as response:
            upload_url = response.headers.get("x-goog-upload-url")
        if not upload_url:
            raise RuntimeError("Gemini Files API did not return x-goog-upload-url.")

        upload_request = urllib.request.Request(
            upload_url,
            data=data,
            method="POST",
            headers={
                "Content-Length": str(len(data)),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
        )
        with _urlopen_json_errors(upload_request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        file_record = payload.get("file", payload)
        if "uri" not in file_record:
            raise RuntimeError(f"Gemini Files API upload response did not include a file uri: {payload}")
        file_record.setdefault("mimeType", resolved_mime)
        return self.wait_for_file_active(file_record)

    def get_file(self, file_name: str) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/v1beta/{file_name}",
            method="GET",
            headers={"x-goog-api-key": self.api_key},
        )
        with _urlopen_json_errors(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def wait_for_file_active(self, file_record: Mapping[str, Any]) -> dict[str, Any]:
        name = file_record.get("name")
        if not name:
            return dict(file_record)
        state = str(file_record.get("state", "")).upper()
        if state == "ACTIVE":
            return dict(file_record)
        if state == "":
            current = self.get_file(str(name))
        else:
            current = dict(file_record)
        deadline = time.monotonic() + self.file_activation_timeout_seconds
        while True:
            state = str(current.get("state", "")).upper()
            if state in {"", "ACTIVE"}:
                return current
            if state in {"FAILED", "ERROR"}:
                raise RuntimeError(f"Gemini uploaded file {name} failed processing: {current}")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for Gemini uploaded file {name} to become ACTIVE; "
                    f"last state={state!r}."
                )
            time.sleep(max(0.1, self.file_activation_poll_seconds))
            current = self.get_file(str(name))

    def generate_json(
        self,
        *,
        model: str,
        prompt: str,
        files: Sequence[Mapping[str, Any]],
        response_schema: Mapping[str, Any],
    ) -> dict[str, Any]:
        parts: list[dict[str, Any]] = []
        for file_record in files:
            file_uri = file_record.get("uri") or file_record.get("fileUri")
            mime_type = file_record.get("mimeType") or file_record.get("mime_type")
            if not file_uri or not mime_type:
                raise ValueError(f"Gemini file record must include uri and mime type, got {file_record}.")
            parts.append({"file_data": {"mime_type": str(mime_type), "file_uri": str(file_uri)}})
        parts.append({"text": prompt})
        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": response_schema,
            },
        }
        request = urllib.request.Request(
            f"{self.base_url}/v1beta/models/{model}:generateContent",
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
            },
        )
        with _urlopen_json_errors(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        text = _extract_gemini_text(payload)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Gemini response was not valid JSON: {text}") from exc

    def delete_file(self, file_name: str) -> None:
        request = urllib.request.Request(
            f"{self.base_url}/v1beta/{file_name}",
            method="DELETE",
            headers={"x-goog-api-key": self.api_key},
        )
        with _urlopen_json_errors(request, timeout=self.timeout_seconds):
            return


class GeminiVlmCandidateRerankEvaluator:
    """Use Gemini to jointly rank FDM-imagined candidate futures."""

    def __init__(
        self,
        config: GeminiVlmRerankConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = client
        self._call_index = 0

    def score(self, candidate: CandidateTrajectory, *, goal: Any | None = None) -> float:
        del goal
        # Scalar fallback keeps the class compatible with TrajectoryEvaluator,
        # but online use should always route through score_candidates().
        return float(candidate.score)

    def score_candidates(
        self,
        candidates: Sequence[CandidateTrajectory],
        *,
        goal: Any | None = None,
    ) -> list[float]:
        if not candidates:
            return []
        if len(candidates) > int(self.config.max_candidates):
            raise ValueError(
                f"Gemini reranking supports at most {self.config.max_candidates} candidates per query, "
                f"got {len(candidates)}."
            )
        labels = _CANDIDATE_LABELS[: len(candidates)]
        call_dir = self.config.output_dir / f"vlm_replan_{self._call_index:04d}"
        self._call_index += 1
        call_dir.mkdir(parents=True, exist_ok=True)

        tiled_video = build_tiled_candidate_video(candidates, labels=labels)
        candidate_video_path = call_dir / "candidate_futures_tiled.mp4"
        imageio.mimsave(candidate_video_path, tiled_video, fps=float(self.config.video_fps))
        sidecar_paths = _write_goal_sidecars(call_dir, goal)
        candidate_roles = _candidate_roles_by_label(candidates, labels)
        prompt = build_gemini_candidate_prompt(
            labels=labels,
            goal=goal,
            candidate_roles=candidate_roles,
            policy_prior_hint_mode=self.config.policy_prior_hint_mode,
        )

        client = self.client or GeminiRestClient(
            api_key_env=self.config.api_key_env,
            timeout_seconds=self.config.request_timeout_seconds,
        )
        uploaded_files: list[dict[str, Any]] = []
        try:
            for path in [candidate_video_path, *sidecar_paths]:
                uploaded_files.append(client.upload_file(path))
            result = client.generate_json(
                model=self.config.model,
                prompt=prompt,
                files=uploaded_files,
                response_schema=gemini_ranking_schema(labels),
            )
        finally:
            if self.config.delete_uploaded_files:
                for file_record in uploaded_files:
                    name = file_record.get("name")
                    if name:
                        try:
                            client.delete_file(str(name))
                        except Exception:
                            pass

        scores = _scores_from_gemini_result(result, labels)
        manifest = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model": self.config.model,
            "labels": list(labels),
            "candidate_video_path": str(candidate_video_path),
            "candidate_video_transform": {"flip_vertical_for_display": True},
            "sidecar_paths": [str(path) for path in sidecar_paths],
            "prompt": prompt,
            "candidate_debug": _candidate_debug_by_label(candidates, labels),
            "gemini_result": result,
            "scores": {label: score for label, score in zip(labels, scores, strict=True)},
        }
        (call_dir / "gemini_rerank_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return scores


def build_tiled_candidate_video(
    candidates: Sequence[CandidateTrajectory],
    *,
    labels: Sequence[str] | None = None,
    border_px: int = 6,
    flip_vertical_for_display: bool = True,
) -> np.ndarray:
    """Build the Gemini-visible tiled candidate video.

    The Open-WAM imagined futures arrive in image-array coordinates that render
    upside-down in the MP4s Gemini sees. Flip only this visualization/export
    view so candidate futures match the current/target/demo sidecars.
    """

    videos = [_candidate_video(candidate) for candidate in candidates]
    if not videos:
        raise ValueError("At least one candidate with predicted video is required.")
    labels = tuple(labels or _CANDIDATE_LABELS[: len(videos)])
    if len(labels) != len(videos):
        raise ValueError(f"Expected one label per video, got labels={len(labels)}, videos={len(videos)}.")
    if border_px < 0:
        raise ValueError("border_px must be non-negative.")

    normalized = _normalize_video_shapes(videos)
    if flip_vertical_for_display:
        normalized = [_flip_video_vertical(video) for video in normalized]
    frames, height, width, channels = normalized[0].shape
    cols = int(math.ceil(math.sqrt(len(normalized))))
    rows = int(math.ceil(len(normalized) / cols))
    canvas = np.zeros((frames, rows * height, cols * width, channels), dtype=np.uint8)
    for index, video in enumerate(normalized):
        row = index // cols
        col = index % cols
        tile = video.copy()
        if border_px > 0:
            color = np.asarray(_LABEL_COLORS[index % len(_LABEL_COLORS)], dtype=np.uint8)
            tile[:, :border_px, :, :] = color
            tile[:, :, :border_px, :] = color
            tile[:, :, -border_px:, :] = color
            tile[:, -border_px:, :, :] = color
        y0 = row * height
        x0 = col * width
        canvas[:, y0 : y0 + height, x0 : x0 + width] = tile
    return canvas


def build_gemini_candidate_prompt(
    *,
    labels: Sequence[str],
    goal: Any | None,
    candidate_roles: Mapping[str, str] | None = None,
    policy_prior_hint_mode: GeminiPolicyPriorHintMode | str = GeminiPolicyPriorHintMode.CONSERVATIVE,
) -> str:
    label_map = ", ".join(
        f"{label}=tile {index} ({'row-major order'})" for index, label in enumerate(labels)
    )
    prior_hint_mode = GeminiPolicyPriorHintMode(policy_prior_hint_mode)
    task_text = _goal_value(goal, "task_text", default="the robot manipulation task")
    demo_note = "A successful demonstration video is included." if _goal_value(goal, "demo_video_path") else "No demo video is included."
    current_note = "A current observation image is included." if _goal_value(goal, "current") is not None else "No current image is included."
    target_note = "A target or reference image is included." if _goal_value(goal, "target") is not None else "No target image is included."
    explicit_stage_criteria = _goal_sequence(goal, "stage_success_criteria")
    subtask_hints = tuple(str(item) for item in explicit_stage_criteria) or infer_libero_progress_hints(str(task_text))
    subtask_lines = "\n".join(f"- {hint}" for hint in subtask_hints)
    planning_scope = str(_goal_value(goal, "planning_scope", default="local_chunk"))
    if planning_scope == "full_trajectory":
        comparison_scope = (
            "You are comparing complete imagined futures generated by repeatedly rolling the policy and FDM forward. "
            "Rank by expected task completion by the end of each video. The robot will execute only the first control "
            "chunk from the winning branch, then observe the real simulator and replan.\n"
        )
        ranking_objective = "Rank candidates by whole-task progress and final task completion in this order:\n"
    else:
        comparison_scope = (
            f"You are comparing exactly {len(labels)} alternatives for the next short control chunk, "
            "not judging final task success.\n"
        )
        ranking_objective = "Rank candidates by local subtask progress in this order:\n"
    role_lines = () if prior_hint_mode is GeminiPolicyPriorHintMode.BLIND else _candidate_role_lines(labels, candidate_roles)
    role_note = "" if not role_lines else "Candidate role notes:\n" + "\n".join(role_lines) + "\n"
    stalled_stage = bool(_goal_value(goal, "intervention_was_stall", default=False))
    if prior_hint_mode is GeminiPolicyPriorHintMode.BLIND:
        baseline_policy_note = (
            "Candidate roles are hidden. Rank only the visible imagined futures and the listed local subtask criteria.\n"
        )
    elif prior_hint_mode is GeminiPolicyPriorHintMode.NEUTRAL:
        baseline_policy_note = (
            "Candidate roles are metadata only. Do not favor or penalize a deterministic baseline-policy candidate because "
            "of its role; rank by visible local subtask progress.\n"
        )
    elif stalled_stage:
        baseline_policy_note = (
            "If a deterministic baseline-policy candidate is present, note that the same baseline policy has just stalled on "
            "this local stage. Do not prefer it by default; choose it only if its imagined future visibly makes the best "
            "local subtask progress.\n"
        )
    else:
        baseline_policy_note = (
            "If a deterministic baseline-policy candidate is present, prefer it unless another candidate clearly improves "
            "the listed local subtask criteria without adding new risks.\n"
        )
    return (
        "You are ranking short imagined robot futures for receding-horizon planning.\n"
        f"Task: {task_text}\n"
        f"{demo_note} {current_note} {target_note}\n"
        "The candidate future video is tiled. Candidates are arranged left-to-right, top-to-bottom. "
        f"Labels: {label_map}.\n"
        f"{role_note}"
        f"{comparison_scope}"
        f"{ranking_objective}"
        f"{subtask_lines}\n"
        "Primary signals: gripper approaches the relevant object, gripper/object alignment improves, stable contact or grasp appears, "
        "a held object moves toward the named receptacle/target, and the motion is consistent with the task text and demo.\n"
        "Penalize: moving away from all task objects, opening/closing the gripper at the wrong place, colliding without useful contact, "
        "dropping or losing the object, motion that only changes camera/background appearance, static futures, and obvious FDM artifacts "
        "such as ghost objects or blurred hallucinated object motion.\n"
        f"{baseline_policy_note}"
        "Do not reward large arm motion, image sharpness, or prettier rendering by itself. "
        "If all candidates are poor, still rank the least bad candidate and keep scores low.\n"
        "Return only JSON matching the schema."
    )


def infer_libero_progress_hints(task_text: str) -> tuple[str, ...]:
    """Return generic short-horizon progress hints for a LIBERO-style task."""

    text = " ".join(str(task_text).lower().split())
    object_phrase, target_phrase = _extract_pick_place_phrases(text)
    object_label = object_phrase or "the task-relevant object"
    target_label = target_phrase or "the named target/receptacle"
    hints: list[str] = []
    if "both" in text:
        hints.extend(
            [
                f"identify which of the requested objects is not yet handled and approach that object",
                "align the gripper with one requested object before moving to the next object",
                f"after grasping, move the held object toward {target_label}",
                f"release the object only when it is inside/on {target_label}, then repeat for the remaining object",
            ]
        )
    elif any(token in text for token in ("put ", "place ", "insert ", "stack ")):
        hints.extend(
            [
                f"approach {object_label} with the gripper",
                f"align and close the gripper on {object_label}",
                f"move the held object toward {target_label}",
                f"release only after the object reaches {target_label}",
            ]
        )
    elif any(token in text for token in ("pick ", "pick up", "grasp ", "lift ")):
        hints.extend(
            [
                f"approach {object_label} with the gripper",
                f"align the gripper around {object_label}",
                f"close the gripper on {object_label}",
                "lift the object without drifting away",
            ]
        )
    else:
        hints.extend(
            [
                "approach the task-relevant object rather than moving in empty space",
                "align the gripper before contact",
                "create stable object contact or grasp",
                "move the affected object toward the task goal",
            ]
        )
    return tuple(hints)


def _extract_pick_place_phrases(task_text: str) -> tuple[str | None, str | None]:
    text = re.sub(r"\s+", " ", task_text.strip().lower())
    object_phrase: str | None = None
    target_phrase: str | None = None

    both_match = re.search(r"\bput both (.+?) (?:in|into|on|onto) (.+)", text)
    if both_match:
        object_phrase = _clean_task_phrase(both_match.group(1))
        target_phrase = _clean_task_phrase(both_match.group(2))
        return object_phrase, target_phrase

    place_match = re.search(r"\b(?:put|place|insert|stack) (.+?) (?:in|into|on|onto) (.+)", text)
    if place_match:
        object_phrase = _clean_task_phrase(place_match.group(1))
        target_phrase = _clean_task_phrase(place_match.group(2))
        return object_phrase, target_phrase

    pick_match = re.search(r"\b(?:pick up|pick|grasp|lift) (.+)", text)
    if pick_match:
        object_phrase = _clean_task_phrase(pick_match.group(1))
    return object_phrase, target_phrase


def _clean_task_phrase(phrase: str) -> str:
    cleaned = phrase.strip(" .")
    cleaned = re.sub(r"\bthe\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or phrase.strip(" .")


def _candidate_debug_by_label(
    candidates: Sequence[CandidateTrajectory],
    labels: Sequence[str],
) -> dict[str, dict[str, Any]]:
    if len(candidates) != len(labels):
        raise ValueError(f"Expected one label per candidate, got {len(labels)} labels and {len(candidates)} candidates.")
    return {
        str(label): _candidate_debug_stats(candidate)
        for label, candidate in zip(labels, candidates, strict=True)
    }


def _candidate_debug_stats(candidate: CandidateTrajectory) -> dict[str, Any]:
    actions = np.asarray(candidate.actions, dtype=np.float32)
    stats: dict[str, Any] = {
        "score_before_vlm": float(candidate.score),
        "action_steps": int(actions.shape[0]) if actions.ndim == 2 else 0,
        "action_dim": int(actions.shape[1]) if actions.ndim == 2 else 0,
    }
    if actions.ndim == 2 and actions.size:
        step_l2 = np.linalg.norm(actions, axis=1)
        stats.update(
            {
                "action_mean_l2": float(np.mean(step_l2)),
                "action_max_l2": float(np.max(step_l2)),
                "action_temporal_delta_mean_l2": _temporal_delta_mean_l2(actions),
            }
        )
    if candidate.action_chunks:
        stats["action_chunk_metadata"] = [_jsonable_mapping(chunk.metadata) for chunk in candidate.action_chunks]
        role = candidate.action_chunks[-1].metadata.get("candidate_role")
        if role:
            stats["candidate_role"] = str(role)
    if candidate.predicted_videos:
        video = _candidate_video(candidate)
        stats.update(_video_debug_stats(video))
    if candidate.metadata:
        stats["candidate_metadata"] = _jsonable_mapping(candidate.metadata)
    return stats


def _temporal_delta_mean_l2(actions: np.ndarray) -> float:
    if int(actions.shape[0]) < 2:
        return 0.0
    deltas = np.diff(actions, axis=0)
    return float(np.linalg.norm(deltas, axis=1).mean())


def _video_debug_stats(video: np.ndarray) -> dict[str, Any]:
    frames = int(video.shape[0])
    stats: dict[str, Any] = {
        "predicted_video_shape": [int(dim) for dim in video.shape],
        "predicted_video_frames": frames,
    }
    if frames >= 2:
        diffs = np.abs(video.astype(np.float32)[1:] - video.astype(np.float32)[:-1])
        stats["predicted_video_mean_frame_delta"] = float(diffs.mean())
        stats["predicted_video_max_frame_delta"] = float(diffs.max())
    else:
        stats["predicted_video_mean_frame_delta"] = 0.0
        stats["predicted_video_max_frame_delta"] = 0.0
    return stats


def _jsonable_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable_value(value) for key, value in mapping.items()}


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size <= 32:
            return value.tolist()
        return {
            "type": "ndarray",
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype),
        }
    if isinstance(value, Mapping):
        return _jsonable_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable_value(item) for item in value]
    if hasattr(value, "detach"):
        try:
            return _jsonable_value(value.detach().cpu().numpy())
        except Exception:
            return repr(value)
    return repr(value)


def gemini_ranking_schema(labels: Sequence[str]) -> dict[str, Any]:
    label_enum = list(labels)
    score_properties = {label: {"type": "number", "description": f"Utility score for candidate {label}."} for label in labels}
    risk_properties = {label: {"type": "string", "description": f"Short risk note for candidate {label}."} for label in labels}
    return {
        "type": "object",
        "properties": {
            "winner": {"type": "string", "enum": label_enum},
            "ranking": {
                "type": "array",
                "items": {"type": "string", "enum": label_enum},
                "minItems": len(label_enum),
                "maxItems": len(label_enum),
            },
            "scores": {
                "type": "object",
                "properties": score_properties,
                "required": label_enum,
            },
            "risks": {
                "type": "object",
                "properties": risk_properties,
                "required": label_enum,
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["winner", "ranking", "scores", "risks", "confidence"],
    }


def _candidate_video(candidate: CandidateTrajectory) -> np.ndarray:
    if not candidate.predicted_videos:
        raise ValueError("Gemini VLM reranking requires every candidate to include predicted video.")
    videos = []
    for index, raw_video in enumerate(candidate.predicted_videos):
        video = np.asarray(raw_video)
        if video.ndim != 4 or int(video.shape[-1]) != 3:
            raise ValueError(
                f"Candidate predicted video chunk {index} must have shape [T,H,W,3], got {video.shape}."
            )
        videos.append(_to_uint8_video(video))
    if len(videos) == 1:
        return videos[0]
    first_shape = videos[0].shape[1:]
    for index, video in enumerate(videos):
        if video.shape[1:] != first_shape:
            raise ValueError(
                "Candidate predicted video chunks must share [H,W,C] before concatenation, "
                f"got first={first_shape}, chunk{index}={video.shape[1:]}."
            )
    return np.concatenate(videos, axis=0)


def _normalize_video_shapes(videos: Sequence[np.ndarray]) -> list[np.ndarray]:
    max_frames = max(int(video.shape[0]) for video in videos)
    height, width, channels = videos[0].shape[1:]
    normalized: list[np.ndarray] = []
    for video in videos:
        if video.shape[1:] != (height, width, channels):
            raise ValueError(
                "All candidate videos must share [H,W,C] for VLM tiling, "
                f"got first={(height, width, channels)}, current={video.shape[1:]}."
            )
        if int(video.shape[0]) < max_frames:
            pad = np.repeat(video[-1:], max_frames - int(video.shape[0]), axis=0)
            video = np.concatenate([video, pad], axis=0)
        normalized.append(video)
    return normalized


def _flip_video_vertical(video: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.flip(video, axis=1))


def _to_uint8_video(video: np.ndarray) -> np.ndarray:
    array = np.asarray(video)
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    array = array.astype(np.float32, copy=False)
    if float(np.nanmax(array)) <= 1.5:
        array = array * 255.0
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


def _write_goal_sidecars(call_dir: Path, goal: Any | None) -> list[Path]:
    paths: list[Path] = []
    demo_path = _goal_value(goal, "demo_video_path")
    if demo_path:
        paths.append(Path(demo_path).expanduser().resolve())
    current = _goal_value(goal, "current")
    if current is not None:
        path = call_dir / "current_observation.png"
        imageio.imwrite(path, _to_uint8_image(current))
        paths.append(path)
    target = _goal_value(goal, "target")
    if target is not None:
        path = call_dir / "target_reference.png"
        imageio.imwrite(path, _to_uint8_image(target))
        paths.append(path)
    return paths


def _to_uint8_image(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or int(array.shape[-1]) != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {array.shape}.")
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    array = array.astype(np.float32, copy=False)
    if float(np.nanmax(array)) <= 1.5:
        array = array * 255.0
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


def _goal_value(goal: Any | None, key: str, default: Any | None = None) -> Any | None:
    if isinstance(goal, Mapping):
        return goal.get(key, default)
    return default


def _goal_sequence(goal: Any | None, key: str) -> tuple[Any, ...]:
    value = _goal_value(goal, key)
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return ()
    if isinstance(value, Sequence):
        return tuple(value)
    return ()


def _candidate_roles_by_label(candidates: Sequence[CandidateTrajectory], labels: Sequence[str]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for label, candidate in zip(labels, candidates, strict=True):
        if not candidate.action_chunks:
            continue
        role = candidate.action_chunks[-1].metadata.get("candidate_role")
        if role:
            roles[str(label)] = str(role)
    return roles


def _candidate_role_lines(labels: Sequence[str], candidate_roles: Mapping[str, str] | None) -> list[str]:
    if not candidate_roles:
        return []
    lines: list[str] = []
    for label in labels:
        role = candidate_roles.get(str(label))
        if role == "policy_prior":
            lines.append(f"- Candidate {label} is the deterministic baseline-policy proposal.")
        elif role:
            lines.append(f"- Candidate {label} role: {role}.")
    return lines


def _scores_from_gemini_result(result: Mapping[str, Any], labels: Sequence[str]) -> list[float]:
    scores_payload = result.get("scores")
    if isinstance(scores_payload, Mapping):
        scores = _normalize_gemini_scores([scores_payload.get(label) for label in labels])
        if scores is not None and any(score != 0.0 for score in scores):
            return scores
    ranking = result.get("ranking")
    if not isinstance(ranking, Sequence) or isinstance(ranking, (str, bytes)):
        winner = result.get("winner")
        ranking = [winner] if winner else []
    denom = max(1, len(labels))
    rank_scores = {str(label): float(len(labels) - index) / float(denom) for index, label in enumerate(ranking)}
    return [rank_scores.get(label, 0.0) for label in labels]


def _normalize_gemini_scores(values: Sequence[Any]) -> list[float] | None:
    raw_scores: list[float] = []
    for value in values:
        score = _finite_float(value)
        if score is None:
            raw_scores.append(0.0)
        else:
            raw_scores.append(score)
    if not raw_scores:
        return None
    min_score = min(raw_scores)
    max_score = max(raw_scores)
    if 0.0 <= min_score and max_score <= 1.0:
        return raw_scores
    if max_score <= 0.0:
        return [0.0 for _ in raw_scores]
    # Gemini often emits human-readable utility scores on a 0-10 scale even
    # though the planner consumes unit scores. Preserve ordering instead of
    # clamping every useful score above 1.0 to the same value.
    scale = 10.0 if max_score <= 10.0 else max_score
    return [max(0.0, min(1.0, score / scale)) for score in raw_scores]


def _finite_float(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return score


def _extract_gemini_text(payload: Mapping[str, Any]) -> str:
    try:
        parts = payload["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Gemini response did not include candidates content: {payload}") from exc
    texts = [str(part["text"]) for part in parts if isinstance(part, Mapping) and "text" in part]
    if not texts:
        raise RuntimeError(f"Gemini response did not include text parts: {payload}")
    return "\n".join(texts)


def _urlopen_json_errors(request: urllib.request.Request, *, timeout: float):
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API request failed with HTTP {exc.code}: {body}") from exc


def write_candidate_dump_manifest(
    *,
    output_dir: Path,
    candidates: Sequence[CandidateTrajectory],
    goal: Any | None = None,
    video_fps: float = 4.0,
) -> Path:
    """Save local candidate futures for offline VLM/debug review without API calls."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    labels = _CANDIDATE_LABELS[: len(candidates)]
    tiled_video = build_tiled_candidate_video(candidates, labels=labels)
    video_path = output / "candidate_futures_tiled.mp4"
    imageio.mimsave(video_path, tiled_video, fps=float(video_fps))
    sidecar_paths = _write_goal_sidecars(output, goal)
    manifest = {
        "labels": list(labels),
        "candidate_video_path": str(video_path),
        "candidate_video_transform": {"flip_vertical_for_display": True},
        "sidecar_paths": [str(path) for path in sidecar_paths],
        "prompt": build_gemini_candidate_prompt(labels=labels, goal=goal),
        "candidate_debug": _candidate_debug_by_label(candidates, labels),
    }
    manifest_path = output / "candidate_dump_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
