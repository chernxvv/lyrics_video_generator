from __future__ import annotations

import logging
import re
import time

import numpy as np

from models import LyricLine

logger = logging.getLogger(__name__)


class AutoSyncError(RuntimeError):
    pass


def _split_lyrics_text(full_text: str) -> list[str]:
    lines = [line.strip() for line in full_text.splitlines()]
    return [line for line in lines if line]


def _format_mmss(seconds: float) -> str:
    safe_seconds = max(0, int(seconds))
    return f"{safe_seconds // 60:02d}:{safe_seconds % 60:02d}"


def _line_weight(text: str) -> float:
    letters = re.findall(r"\w", text, flags=re.UNICODE)
    return max(1.0, float(len(letters)))


def auto_sync_lyrics(audio_path: str, full_lyrics_text: str) -> list[LyricLine]:
    logger.info("Автосинхронизация: запуск")
    started = time.perf_counter()

    text = full_lyrics_text.strip()
    if not text:
        raise AutoSyncError("Текст трека пуст. Вставьте полный текст перед запуском анализа.")

    lines = _split_lyrics_text(text)
    if len(lines) < 2:
        raise AutoSyncError("Для автосинхронизации нужно минимум 2 непустые строки.")

    try:
        import librosa
    except ImportError as exc:
        raise AutoSyncError("Для автосинхронизации нужен librosa. Установите зависимости из requirements.txt") from exc

    logger.info("Автосинхронизация: backend=librosa onset+energy alignment")
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        intervals = librosa.effects.split(y, top_db=28)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    except Exception as exc:  # noqa: BLE001
        raise AutoSyncError(f"Ошибка анализа аудио: {exc}") from exc

    if duration <= 0:
        raise AutoSyncError("Не удалось определить длительность аудио для автосинхронизации.")

    starts = [0.0]
    for s, _ in intervals:
        point = float(s / sr)
        if point > 0.35:
            starts.append(point)
    starts.extend(float(t) for t in onset_times.tolist())
    starts = sorted(set(starts))
    starts = [t for t in starts if t < duration - 0.2]

    if len(starts) < 3:
        logger.warning("Автосинхронизация: мало стартовых маркеров, fallback на равномерное распределение")
        starts = np.linspace(0, max(0.0, duration - 0.5), num=max(3, len(lines) + 1)).tolist()

    weights = np.array([_line_weight(line) for line in lines], dtype=np.float32)
    weights /= weights.sum()
    target_times = (weights.cumsum() * max(0.0, duration - 0.5)).tolist()

    aligned: list[LyricLine] = []
    prev = -1.0
    for text_line, target in zip(lines, target_times):
        nearest = min(starts, key=lambda t: abs(t - target))
        start_time = max(prev + 0.3, nearest)
        if start_time >= duration:
            start_time = max(prev + 0.3, duration - 0.3)
        aligned.append(LyricLine(start_time=_format_mmss(start_time), text=text_line))
        prev = start_time

    if len(aligned) < 2:
        raise AutoSyncError("Автосинхронизация вернула слишком мало строк.")

    elapsed = time.perf_counter() - started
    logger.info(
        "Автосинхронизация: завершено за %.2fs, найдено строк=%d, стартовых маркеров=%d",
        elapsed,
        len(aligned),
        len(starts),
    )
    if len(starts) < len(lines):
        logger.warning("Автосинхронизация: частичное совпадение маркеров и строк, возможна ручная корректировка")
    return aligned
