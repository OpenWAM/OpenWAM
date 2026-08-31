"""Architecture-independent output semantics for video/action programs."""

from __future__ import annotations

from open_wam.configs import VideoActionProgram
from open_wam.configs.policy_video_action import output_flags_for_program

from .contracts import PolicyOutputModality


def video_action_program_output_modalities(
    program: VideoActionProgram | str,
) -> frozenset[PolicyOutputModality]:
    """Return products emitted by a normal inference call for one program."""

    emits_video, emits_action = output_flags_for_program(program)
    modalities: set[PolicyOutputModality] = set()
    if emits_video:
        modalities.add(PolicyOutputModality.VIDEO)
    if emits_action:
        modalities.add(PolicyOutputModality.ACTION)
    if not modalities:  # pragma: no cover - guarded by program declarations.
        raise ValueError(
            f"Video/action program {VideoActionProgram(program).value!r} emits no outputs."
        )
    return frozenset(modalities)


__all__ = ["video_action_program_output_modalities"]
