from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LingbotReferenceProfile:
    """Reference LingBot parallel-stream runtime settings for a known benchmark."""

    name: str
    max_text_tokens: int
    action_dim: int
    action_per_frame: int
    frame_chunk_size: int
    attn_window: int
    guidance_scale: float
    action_guidance_scale: float
    video_num_inference_steps: int
    action_num_inference_steps: int
    video_exec_step: int
    video_sigma_shift: float
    action_sigma_shift: float
    obs_cam_keys: tuple[str, ...]
    used_action_channel_ids: tuple[int, ...]
    inverse_used_action_channel_ids: tuple[int, ...]
    action_norm_method: str
    norm_q01: tuple[float, ...]
    norm_q99: tuple[float, ...]


_BUILTIN_REFERENCE_PROFILES: dict[str, LingbotReferenceProfile] = {
    "robotwin": LingbotReferenceProfile(
        name="robotwin",
        max_text_tokens=512,
        action_dim=30,
        action_per_frame=16,
        frame_chunk_size=2,
        attn_window=72,
        guidance_scale=5.0,
        action_guidance_scale=1.0,
        video_num_inference_steps=25,
        action_num_inference_steps=50,
        video_exec_step=-1,
        video_sigma_shift=5.0,
        action_sigma_shift=1.0,
        obs_cam_keys=(
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ),
        used_action_channel_ids=tuple(list(range(0, 7)) + [28] + list(range(7, 14)) + [29]),
        inverse_used_action_channel_ids=(0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 7, 15),
        action_norm_method="quantiles",
        norm_q01=(
            -0.06172713458538055,
            -3.6716461181640625e-05,
            -0.08783501386642456,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -0.3547105032205582,
            -1.3113021850585938e-06,
            -0.11975435614585876,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
        )
        + (0.0,) * 16,
        norm_q99=(
            0.3462600058317184,
            0.39966784834861746,
            0.14745532035827624,
            1.0,
            1.0,
            1.0,
            1.0,
            0.034201726913452024,
            0.39142737388610793,
            0.1792279863357542,
            1.0,
            1.0,
            1.0,
            1.0,
        )
        + (0.0,) * 14
        + (1.0, 1.0),
    ),
    "franka": LingbotReferenceProfile(
        name="franka",
        max_text_tokens=512,
        action_dim=30,
        action_per_frame=20,
        frame_chunk_size=4,
        attn_window=30,
        guidance_scale=5.0,
        action_guidance_scale=1.0,
        video_num_inference_steps=5,
        action_num_inference_steps=10,
        video_exec_step=-1,
        video_sigma_shift=5.0,
        action_sigma_shift=1.0,
        obs_cam_keys=(
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ),
        used_action_channel_ids=tuple(list(range(0, 7)) + [28] + list(range(7, 14)) + [29]),
        inverse_used_action_channel_ids=(0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 7, 15),
        action_norm_method="quantiles",
        norm_q01=(
            0.3051295876502991,
            -0.22647984325885773,
            0.19957000017166138,
            -0.022680532187223434,
            -0.05553057789802551,
            -0.2693849802017212,
            -0.29341773986816405,
            0.2935442328453064,
            -0.4431332051753998,
            0.21256473660469055,
            -0.7962440848350525,
            -0.40816226601600647,
            -0.28359392285346985,
            -0.44507765769958496,
        )
        + (0.0,) * 16,
        norm_q99=(
            0.7572150230407715,
            0.47736290097236633,
            0.6428080797195435,
            0.9835678935050964,
            0.9927203059196472,
            0.28041139245033264,
            0.47529348731040877,
            0.7564866304397571,
            0.04082797020673729,
            0.5355993628501885,
            0.9976375699043274,
            0.8973174452781656,
            0.6016915678977965,
            0.5027598619461056,
        )
        + (0.0,) * 14
        + (1.0, 1.0),
    ),
    "demo": LingbotReferenceProfile(
        name="demo",
        max_text_tokens=512,
        action_dim=30,
        action_per_frame=8,
        frame_chunk_size=4,
        attn_window=30,
        guidance_scale=5.0,
        action_guidance_scale=1.0,
        video_num_inference_steps=5,
        action_num_inference_steps=10,
        video_exec_step=-1,
        video_sigma_shift=5.0,
        action_sigma_shift=1.0,
        obs_cam_keys=(
            "observation.images.top",
            "observation.images.wrist",
        ),
        used_action_channel_ids=(0, 1, 2, 3, 4, 28),
        inverse_used_action_channel_ids=(0, 1, 2, 3, 4) + (6,) * 23 + (5, 6),
        action_norm_method="quantiles",
        norm_q01=(
            -90.60303497314453,
            -98.73043060302734,
            -79.9008560180664,
            48.95470428466797,
            -32.794578552246094,
        )
        + (0.0,) * 23
        + (0.8250824809074402, 0.0),
        norm_q99=(
            71.735107421875,
            65.89081573486328,
            92.87967681884766,
            100.0,
            22.784151077270508,
        )
        + (0.0,) * 23
        + (100.0, 0.0),
    ),
    "libero": LingbotReferenceProfile(
        name="libero",
        max_text_tokens=512,
        action_dim=30,
        action_per_frame=4,
        frame_chunk_size=4,
        attn_window=30,
        guidance_scale=5.0,
        action_guidance_scale=1.0,
        video_num_inference_steps=20,
        action_num_inference_steps=50,
        video_exec_step=-1,
        video_sigma_shift=5.0,
        action_sigma_shift=1.0,
        obs_cam_keys=(
            "observation.images.agentview_rgb",
            "observation.images.eye_in_hand_rgb",
        ),
        used_action_channel_ids=(0, 1, 2, 3, 4, 5, 28),
        inverse_used_action_channel_ids=(0, 1, 2, 3, 4, 5) + (7,) * 22 + (6, 7),
        action_norm_method="quantiles",
        norm_q01=(
            -0.6589285731315613,
            -0.84375,
            -0.9375,
            -0.12107142806053162,
            -0.15964286029338837,
            -0.26571428775787354,
        )
        + (0.0,) * 22
        + (-1.0, 0.0),
        norm_q99=(
            0.8999999761581421,
            0.8544642925262451,
            0.9375,
            0.17142857611179352,
            0.1842857152223587,
            0.34392857551574707,
        )
        + (0.0,) * 22
        + (1.0, 0.0),
    ),
    "libero_joint": LingbotReferenceProfile(
        name="libero_joint",
        max_text_tokens=512,
        action_dim=30,
        action_per_frame=4,
        frame_chunk_size=4,
        attn_window=30,
        guidance_scale=5.0,
        action_guidance_scale=1.0,
        video_num_inference_steps=20,
        action_num_inference_steps=20,
        video_exec_step=-1,
        video_sigma_shift=5.0,
        action_sigma_shift=1.0,
        obs_cam_keys=(
            "observation.images.agentview_rgb",
            "observation.images.eye_in_hand_rgb",
        ),
        used_action_channel_ids=(0, 1, 2, 3, 4, 5, 28),
        inverse_used_action_channel_ids=(0, 1, 2, 3, 4, 5) + (7,) * 22 + (6, 7),
        action_norm_method="quantiles",
        norm_q01=(
            -0.6589285731315613,
            -0.84375,
            -0.9375,
            -0.12107142806053162,
            -0.15964286029338837,
            -0.26571428775787354,
        )
        + (0.0,) * 22
        + (-1.0, 0.0),
        norm_q99=(
            0.8999999761581421,
            0.8544642925262451,
            0.9375,
            0.17142857611179352,
            0.1842857152223587,
            0.34392857551574707,
        )
        + (0.0,) * 22
        + (1.0, 0.0),
    ),
}


def load_reference_profile(name: str | None) -> LingbotReferenceProfile | None:
    if name is None:
        return None
    try:
        return _BUILTIN_REFERENCE_PROFILES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_BUILTIN_REFERENCE_PROFILES))
        raise ValueError(f"Unsupported LingBot reference profile '{name}'. Expected one of: {supported}.") from exc
