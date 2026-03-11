from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Layout:
    artist_y: int
    dash_y: int
    title_y: int
    cover_box: tuple[int, int, int, int]
    lyrics_box: tuple[int, int, int, int]
    date_y: int


def compute_layout(width: int, height: int) -> Layout:
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
        dash_y=int(height * 0.11),
        title_y=int(height * 0.145),
        cover_box=(cover_x, cover_y, cover_x + cover_w, cover_y + cover_h),
        lyrics_box=(lyrics_x, lyrics_y, lyrics_x + lyrics_w, lyrics_y + lyrics_h),
        date_y=lyrics_y + lyrics_h + int(height * 0.03),
    )
