from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np

from core.audio import probe_audio_duration

logger = logging.getLogger(__name__)


def build_waveform_envelope(audio_path: Path, bins: int = 2048) -> tuple[list[float], float]:
    duration = probe_audio_duration(audio_path)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg не найден: waveform preview недоступен")

    cmd = [
        "ffmpeg", "-v", "error", "-i", str(audio_path),
        "-ac", "1", "-ar", "8000", "-f", "f32le", "-",
    ]
    logger.info("Building waveform envelope via ffmpeg: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(stderr or "Не удалось декодировать аудио для waveform")

    samples = np.frombuffer(result.stdout, dtype=np.float32)
    if samples.size == 0:
        return [0.0] * bins, duration

    abs_samples = np.abs(samples)
    chunk = max(1, int(np.ceil(abs_samples.size / bins)))
    padded = int(np.ceil(abs_samples.size / chunk) * chunk)
    if padded != abs_samples.size:
        abs_samples = np.pad(abs_samples, (0, padded - abs_samples.size))
    envelope = abs_samples.reshape(-1, chunk).max(axis=1)
    if envelope.size < bins:
        envelope = np.pad(envelope, (0, bins - envelope.size))
    elif envelope.size > bins:
        envelope = envelope[:bins]
    maximum = float(np.max(envelope)) or 1.0
    return (envelope / maximum).astype(float).tolist(), duration
