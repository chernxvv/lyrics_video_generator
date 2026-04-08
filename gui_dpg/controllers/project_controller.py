from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from models import LyricLine, ProjectData


def save_project_file(path: Path, project: ProjectData) -> None:
    payload = asdict(project)
    if project.audio_path is not None:
        payload["audio_path"] = str(project.audio_path)
    if project.image_path is not None:
        payload["image_path"] = str(project.image_path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_project_file(path: Path) -> ProjectData:
    payload = json.loads(path.read_text(encoding="utf-8"))
    lyrics = [LyricLine(**line) for line in payload.get("lyrics", [])]
    payload["lyrics"] = lyrics
    if payload.get("audio_path"):
        payload["audio_path"] = Path(payload["audio_path"])
    else:
        payload["audio_path"] = None
    if payload.get("image_path"):
        payload["image_path"] = Path(payload["image_path"])
    else:
        payload["image_path"] = None
    return ProjectData(**payload)
