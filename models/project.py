from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class LyricLine:
    start_time: str
    text: str


@dataclass(slots=True)
class PaletteColor:
    rgb: tuple[int, int, int]
    ratio: float


@dataclass(slots=True)
class PaletteInfo:
    colors: list[PaletteColor] = field(default_factory=list)


@dataclass(slots=True)
class RenderSettings:
    width: int = 1080
    height: int = 1920
    fps: int = 30
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    pixel_format: str = "yuv420p"


@dataclass(slots=True)
class ProjectData:
    audio_path: Path | None = None
    image_path: Path | None = None
    artist: str = ""
    title: str = ""
    release_date: str = ""
    lyrics: list[LyricLine] = field(default_factory=list)
