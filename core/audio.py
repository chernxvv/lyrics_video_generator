from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioError(RuntimeError):
    pass


def ensure_ffprobe_available() -> None:
    logger.info("Проверка доступности ffprobe")
    if shutil.which("ffprobe") is None:
        raise AudioError(
            "Не найден ffprobe в PATH. Установите FFmpeg и добавьте ffprobe в переменную PATH."
        )


def probe_audio_duration(audio_path: Path) -> float:
    logger.info("Чтение длительности аудио: %s", audio_path)
    ensure_ffprobe_available()

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
    logger.info("Выполнение команды: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("ffprobe завершился с ошибкой: %s", result.stderr.strip())
        raise AudioError(f"Не удалось прочитать аудио: {result.stderr.strip()}")

    payload = json.loads(result.stdout)
    duration = float(payload["format"]["duration"])
    if duration <= 0:
        raise AudioError("Некорректная длительность аудио.")

    logger.info("Длительность трека: %.3f сек", duration)
    return duration
