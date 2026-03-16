# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Roman Chernov (romanchernovv@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.

from __future__ import annotations

from dataclasses import dataclass

from models import VideoOrientation


@dataclass(slots=True)
class Layout:
    artist_y: int
    title_y: int
    cover_box: tuple[int, int, int, int]
    lyrics_box: tuple[int, int, int, int]
    date_y: int


BASE_CANVAS_BY_ORIENTATION: dict[VideoOrientation, tuple[int, int]] = {
    "vertical": (540, 960),
    "horizontal": (960, 540),
}


def get_base_canvas(orientation: VideoOrientation) -> tuple[int, int]:
    return BASE_CANVAS_BY_ORIENTATION[orientation]


def _layout_vertical(width: int, height: int) -> Layout:
    lyrics_h = int(height * 0.18)
    lyrics_w = int(width * 0.8)
    lyrics_x = (width - lyrics_w) // 2
    lyrics_y = int(height * 0.67)

    cover_w = lyrics_w
    cover_h = cover_w
    cover_x = lyrics_x
    cover_y = max(0, lyrics_y - cover_h)

    return Layout(
        artist_y=int(height * 0.07),
        title_y=int(height * 0.145),
        cover_box=(cover_x, cover_y, cover_x + cover_w, cover_y + cover_h),
        lyrics_box=(lyrics_x, lyrics_y, lyrics_x + lyrics_w, lyrics_y + lyrics_h),
        date_y=lyrics_y + lyrics_h + int(height * 0.03),
    )


def _layout_horizontal(width: int, height: int) -> Layout:
    side_margin = int(width * 0.035)
    middle_gap = int(width * 0.08)

    box_y = int(height * 0.165)
    box_h = int(height * 0.76)
    box_w = (width - side_margin * 2 - middle_gap) // 2

    cover_x = side_margin
    cover_y = box_y

    right_start = cover_x + box_w + middle_gap
    right_width = box_w
    lyrics_y = box_y
    lyrics_h = box_h

    return Layout(
        artist_y=int(height * 0.02),
        title_y=int(height * 0.095),
        cover_box=(cover_x, cover_y, cover_x + box_w, cover_y + box_h),
        lyrics_box=(right_start, lyrics_y, right_start + right_width, lyrics_y + lyrics_h),
        date_y=int(height * 0.94),
    )


def compute_layout(width: int, height: int, orientation: VideoOrientation) -> Layout:
    _ = (width, height)
    width, height = get_base_canvas(orientation)
    if orientation == "horizontal":
        return _layout_horizontal(width, height)
    return _layout_vertical(width, height)
