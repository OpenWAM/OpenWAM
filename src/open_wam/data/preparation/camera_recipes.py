"""Camera conventions for the documented dataset recipe, not model semantics."""


def camera_recipe(row):
    """Translate source conventions or reviewed overrides to generic plan inputs."""
    source = row["source_id"]
    selected, reason = select_cameras(source, row["cameras"])
    if source not in ("VPT-10R", "VPT-10S") and row.get("multiview_cameras"):
        selected = row["multiview_cameras"]
        reason = "explicit episode camera-role mapping"
    rotations = dict(row.get("camera_rotations_degrees", {}))
    if source == "VPT-06" and "camera_top" in selected:
        rotations.setdefault("camera_top", 180)
    return dict(
        cameras=selected,
        selection_reason=reason,
        camera_rotations=rotations,
        pair_orientation="horizontal" if source in ("VPT-01", "VPT-09") else "auto",
    )


def select_cameras(source, cameras):
    """Return canonical order plus a reviewable selection explanation.

    Camera ids are dataset metadata, not learned roles. Ambiguous three-view
    arrangements stay in the catalog with a reason instead of inventing a
    right camera or substituting unrelated episodes.
    """
    cameras = sorted(set(cameras))
    if source in ("VPT-10R", "VPT-10S"):
        return (
            [],
            "EgoExo4D multiview augmentation disabled by user; retain original single views",
        )
    if len(cameras) < 2:
        return [], "single view"
    if len(cameras) == 2:
        return cameras, "all available scene views"
    left = [c for c in cameras if "left" in c and ("hand" in c or "wrist" in c)]
    right = [c for c in cameras if "right" in c and ("hand" in c or "wrist" in c)]
    # Prefer a head camera over front/chest; stereo head cameras get a fixed
    # left-camera representative while their original single views remain.
    heads = [c for c in cameras if "head" in c]
    heads.sort(key=lambda c: (("left" in c or "right" in c), "right" in c, c))
    fallback = [
        c
        for c in cameras
        if c in ("camera_front", "camera_top") or "front" in c or "high" in c
    ]
    head = (
        ["camera_top"]
        if source == "VPT-06" and "camera_top" in cameras
        else (heads or fallback)
    )
    if len(left) == 1 and len(right) == 1 and head:
        return [head[0], left[0], right[0]], "head/front above left/right wrist"
    if (
        source == "VPT-06"
        and "camera_left" in cameras
        and "camera_right" in cameras
        and head
    ):
        return [
            head[0],
            "camera_left",
            "camera_right",
        ], "front/top above named left/right external views"
    if len(cameras) == 3 and head:
        remaining = [c for c in cameras if c != head[0]]
        return (
            [head[0]] + remaining,
            "head above remaining cameras in recorded name order; not claimed as anatomical left/right",
        )
    return [], "requires camera-role review"
