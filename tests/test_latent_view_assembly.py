from __future__ import annotations

import pytest
import torch

from open_wam.contracts.video import CanonicalViewLayout
from open_wam.data import assemble_latent_views
from open_wam.data.latent_view_assembly import (
    assemble_mixed_video_latent_views,
)
from open_wam.data.mixed_video import (
    assemble_mixed_video_latent_views as legacy_mixed_video_assembly,
)


def _views(count: int) -> list[torch.Tensor]:
    return [
        torch.full((2, 3, 4, 5), float(index + 1))
        for index in range(count)
    ]


def test_mixed_video_assembly_is_identity_compatible_alias() -> None:
    assert assemble_mixed_video_latent_views is assemble_latent_views
    assert legacy_mixed_video_assembly is assemble_latent_views


@pytest.mark.parametrize(
    ("view_count", "expected_shape", "expected_placements"),
    [
        (1, (2, 3, 4, 5), ((0, 0),)),
        (2, (2, 3, 4, 10), ((0, 0), (0, 5))),
        (3, (2, 3, 8, 10), ((0, 0), (0, 5), (4, 2))),
        (4, (2, 3, 8, 10), ((0, 0), (0, 5), (4, 0), (4, 5))),
    ],
)
def test_assemble_latent_views_places_one_to_four_views(
    view_count: int,
    expected_shape: tuple[int, ...],
    expected_placements: tuple[tuple[int, int], ...],
) -> None:
    slots = tuple(f"camera.{index}" for index in range(view_count))

    canvas, metadata = assemble_latent_views(
        _views(view_count),
        slots=slots,
    )

    assert canvas.shape == expected_shape
    assert canvas.is_contiguous()
    layout = CanonicalViewLayout.from_metadata(metadata)
    assert tuple(placement.source_name for placement in layout.placements) == slots
    assert tuple(placement.canonical_name for placement in layout.placements) == slots
    assert tuple(
        (placement.top, placement.left) for placement in layout.placements
    ) == expected_placements


def test_single_view_is_centered_in_larger_configured_canvas() -> None:
    canvas, metadata = assemble_latent_views(
        _views(1),
        slots=("wrist",),
        canvas_view_count=4,
    )

    assert canvas.shape == (2, 3, 8, 10)
    layout = CanonicalViewLayout.from_metadata(metadata)
    assert layout.canvas_height == 8
    assert layout.canvas_width == 10
    assert layout.placements[0].source_name == "wrist"
    assert (layout.placements[0].top, layout.placements[0].left) == (2, 2)
    assert torch.equal(canvas[:, :, 2:6, 2:7], _views(1)[0])


def test_assembly_preserves_dtype_and_accepts_noncontiguous_views() -> None:
    view = torch.arange(2 * 3 * 4 * 5, dtype=torch.float64).reshape(2, 3, 4, 5)
    view = view.transpose(2, 3)
    assert not view.is_contiguous()

    canvas, _ = assemble_latent_views([view], slots=("front",))

    assert canvas.dtype == torch.float64
    assert canvas.is_contiguous()
    assert torch.equal(canvas, view)


def test_assembly_preserves_gradients_for_every_selected_view() -> None:
    views = [
        torch.full(
            (2, 3, 4, 5),
            float(index + 1),
            requires_grad=True,
        )
        for index in range(3)
    ]

    canvas, _ = assemble_latent_views(
        views,
        slots=("front", "left", "wrist"),
        canvas_view_count=4,
    )
    canvas.square().sum().backward()

    for view in views:
        assert view.grad is not None
        assert torch.equal(view.grad, 2 * view)


@pytest.mark.parametrize(
    ("second", "message"),
    [
        (torch.zeros(2, 3, 4, 5, dtype=torch.float64), "requires one dtype"),
        (torch.empty(2, 3, 4, 5, device="meta"), "requires one device"),
    ],
)
def test_assembly_rejects_cross_view_storage_mismatches(
    second: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        assemble_latent_views(
            [torch.zeros(2, 3, 4, 5), second],
            slots=("front", "wrist"),
        )


@pytest.mark.parametrize("canvas_view_count", [True, 2.5])
def test_assembly_requires_integral_canvas_view_count(
    canvas_view_count: object,
) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        assemble_latent_views(
            _views(1),
            slots=("front",),
            canvas_view_count=canvas_view_count,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("views", "slots", "canvas_view_count", "message"),
    [
        (_views(1), (), 1, "Expected one latent tensor per slot"),
        ((), (), 1, "supports 1 to 4 views"),
        (_views(1), ("a",), 0, "canvas supports 1 to 4 views"),
        (_views(1), ("a",), 5, "canvas supports 1 to 4 views"),
        (_views(2), ("a", "b"), 1, "cannot hold 2 selected views"),
        (
            [_views(1)[0], torch.zeros(2, 3, 4, 6)],
            ("a", "b"),
            2,
            "requires same-resolution views",
        ),
    ],
)
def test_assemble_latent_views_rejects_invalid_layouts(
    views: list[torch.Tensor] | tuple[torch.Tensor, ...],
    slots: tuple[str, ...],
    canvas_view_count: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        assemble_latent_views(
            views,
            slots=slots,
            canvas_view_count=canvas_view_count,
        )
