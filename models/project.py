from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

VideoOrientation = Literal["vertical", "horizontal"]
SyncMode = Literal["manual", "auto"]
BackgroundMode = Literal["soft_gradient", "bpm_dynamic"]
RenderMode = Literal["Preview", "Final"]


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
class RenderProfile:
    orientation: VideoOrientation
    mode: RenderMode
    width: int
    height: int
    fps: int


@dataclass(slots=True)
class RenderSettings:
    width: int = 1080
    height: int = 1920
    fps: int = 30
    orientation: VideoOrientation = "vertical"
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
    def from_profile(cls, profile: RenderProfile) -> "RenderSettings":
        return cls(
            width=profile.width,
            height=profile.height,
            fps=profile.fps,
            orientation=profile.orientation,
        )


RENDER_PROFILES: dict[tuple[VideoOrientation, RenderMode], RenderProfile] = {
    ("vertical", "Preview"): RenderProfile("vertical", "Preview", 540, 960, 24),
    ("vertical", "Final"): RenderProfile("vertical", "Final", 1080, 1920, 30),
    ("horizontal", "Preview"): RenderProfile("horizontal", "Preview", 960, 540, 24),
    ("horizontal", "Final"): RenderProfile("horizontal", "Final", 1920, 1080, 30),
}


@dataclass(slots=True)
class ProjectData:
    audio_path: Path | None = None
    image_path: Path | None = None
    artist: str = ""
    title: str = ""
    release_date: str = ""
    lyrics: list[LyricLine] = field(default_factory=list)
    orientation: VideoOrientation = "vertical"
    sync_mode: SyncMode = "manual"
    auto_sync_lyrics_text: str = ""
    lyrics_autofilled: bool = False
    auto_sync_audio_path: str = ""
    background_mode: BackgroundMode = "soft_gradient"
