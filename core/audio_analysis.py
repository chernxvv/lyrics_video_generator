from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


class AudioAnalysisError(RuntimeError):
    pass


@dataclass(slots=True)
class BeatAnalysisResult:
    bpm: float
    beats: list[float]
    confidence_low: bool = False


_last_beat_cache: dict[tuple[str, int], BeatAnalysisResult] = {}


def analyze_bpm_and_beats(audio_path: str, fps: int) -> BeatAnalysisResult:
    cache_key = (audio_path, fps)
    if cache_key in _last_beat_cache:
        logger.info("Beat-analysis: использован runtime cache")
        return _last_beat_cache[cache_key]

    logger.info("Beat-analysis: старт для %s", audio_path)
    try:
        import librosa
    except ImportError as exc:
        raise AudioAnalysisError("Для BPM-анализа нужен librosa. Установите зависимости из requirements.txt") from exc

    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr).tolist()
    except Exception as exc:  # noqa: BLE001
        raise AudioAnalysisError(f"Ошибка BPM-анализа: {exc}") from exc

    bpm = float(tempo) if np.isfinite(tempo) else 0.0
    confidence_low = False

    if not beat_times:
        confidence_low = True
        logger.warning("Beat-analysis: детектор не нашел ударов, fallback на равномерную сетку")
        if bpm <= 0:
            bpm = 90.0
        step = 60.0 / bpm
        duration = len(y) / sr
        beat_times = np.arange(0, duration, max(step, 0.25)).tolist()

    if bpm < 60.0 or bpm > 190.0:
        logger.warning("Beat-analysis: BPM %.2f вне типового диапазона, точность может быть ниже", bpm)
        confidence_low = True

    result = BeatAnalysisResult(bpm=bpm, beats=beat_times, confidence_low=confidence_low)
    _last_beat_cache[cache_key] = result
    logger.info("Beat-analysis: bpm=%.2f, beats=%d", result.bpm, len(result.beats))
    return result
