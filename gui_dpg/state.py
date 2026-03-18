from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from models import LyricLine, ProjectData, RenderSettings

TaskStatus = Literal["Idle", "Analyzing", "Auto-sync running", "Rendering", "Ready", "Error"]


@dataclass(slots=True)
class DiagnosticsState:
    entries: list[str] = field(default_factory=list)

    def push(self, message: str) -> None:
        self.entries.append(message)
        del self.entries[:-200]

    @property
    def text(self) -> str:
        return "\n".join(self.entries[-80:])


@dataclass(slots=True)
class WaveformCacheState:
    audio_path: Path | None = None
    duration: float = 0.0
    samples: list[float] = field(default_factory=list)
    ready: bool = False


@dataclass(slots=True)
class PreviewCacheState:
    current_time: float = 0.0
    texture_tag: str = "preview_texture"
    width: int = 640
    height: int = 360
    dirty: bool = True


@dataclass(slots=True)
class UIState:
    project: ProjectData = field(default_factory=ProjectData)
    selected_line_index: int = -1
    playback_position: float = 0.0
    zoom_level: float = 1.0
    timeline_scroll: float = 0.0
    status: TaskStatus = "Idle"
    waveform: WaveformCacheState = field(default_factory=WaveformCacheState)
    preview: PreviewCacheState = field(default_factory=PreviewCacheState)
    diagnostics: DiagnosticsState = field(default_factory=DiagnosticsState)
    render_settings: RenderSettings = field(default_factory=RenderSettings)
    project_path: Path | None = None
    transport_playing: bool = False
    transport_fps: float = 12.0
    last_error: str = ""

    def set_status(self, status: TaskStatus, message: str | None = None) -> None:
        self.status = status
        if message:
            self.diagnostics.push(f"[{status}] {message}")
