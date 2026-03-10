from __future__ import annotations

import json
import subprocess
from pathlib import Path


class AudioError(RuntimeError):
    pass


def probe_audio_duration(audio_path: Path) -> float:
    if not audio_path.exists():
        raise AudioError(f"Аудиофайл не найден: {audio_path}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioError(f"Не удалось прочитать аудио: {result.stderr.strip()}")

    payload = json.loads(result.stdout)
    duration = float(payload["format"]["duration"])
    if duration <= 0:
        raise AudioError("Некорректная длительность аудио.")
    return duration
