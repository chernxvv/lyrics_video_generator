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
    prefer_hw_encode: bool = True
    video_codec_hw: str = "h264_nvenc"
    video_codec_sw: str = "libx264"
    nvenc_preset: str = "p1"
    nvenc_cq: int = 23
    x264_preset: str = "medium"
    x264_crf: int = 21
    thread_count: int = 2
    frame_chunk_size: int = 60
    audio_codec: str = "aac"
    pixel_format: str = "yuv420p"

    @classmethod
    def preview(cls) -> "RenderSettings":
        return cls(width=540, height=960, fps=24)

    @classmethod
    def final(cls) -> "RenderSettings":
        return cls(width=1080, height=1920, fps=30)


@dataclass(slots=True)
class ProjectData:
    audio_path: Path | None = None
    image_path: Path | None = None
    artist: str = ""
    title: str = ""
    release_date: str = ""
    lyrics: list[LyricLine] = field(default_factory=list)
