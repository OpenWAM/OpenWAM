from __future__ import annotations

import re


LATENT_WINDOW_FILENAME_PATTERN = re.compile(
    r"episode_(?P<episode>\d{6})_(?P<start>\d+)_(?P<end>\d+)\.pth$"
)


def match_latent_window_filename(filename: str) -> re.Match[str] | None:
    return LATENT_WINDOW_FILENAME_PATTERN.match(filename)
