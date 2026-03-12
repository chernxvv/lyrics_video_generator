from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


class AudioAnalysisError(RuntimeError):
    pass


@dataclass(slots=True)
class BeatAnalysisResult:
    bpm: float
    beats: list[float]
    confidence_low: bool = False
    beat_strengths: list[float] = field(default_factory=list)
    local_tempo: list[float] = field(default_factory=list)
    percussion_energy: list[float] = field(default_factory=list)


_last_beat_cache: dict[tuple[str, int], BeatAnalysisResult] = {}


def _to_scalar_bpm(tempo_value) -> float:
    if isinstance(tempo_value, np.ndarray):
        if tempo_value.size == 0:
            return 0.0
        return float(np.ravel(tempo_value)[0])
    if isinstance(tempo_value, (list, tuple)):
        if not tempo_value:
            return 0.0
        return float(tempo_value[0])
    return float(tempo_value)


def _normalize_series(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-8:
        return np.ones_like(values, dtype=np.float32) * 0.5
    return ((values - lo) / (hi - lo)).astype(np.float32)


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
        _, y_percussive = librosa.effects.hpss(y)

        tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr).tolist()

        onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)
        rms = librosa.feature.rms(y=y_percussive, frame_length=2048, hop_length=512).flatten()

        tempo_f = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
    except Exception as exc:  # noqa: BLE001
        raise AudioAnalysisError(f"Ошибка BPM-анализа: {exc}") from exc

    bpm_candidate = _to_scalar_bpm(tempo)
    bpm = bpm_candidate if np.isfinite(bpm_candidate) else 0.0
    confidence_low = False

    if not beat_times:
        confidence_low = True
        logger.warning("Beat-analysis: детектор не нашел ударов, fallback на равномерную сетку")
        if bpm <= 0:
            bpm = 90.0
        step = 60.0 / bpm
        duration = len(y) / sr
        beat_times = np.arange(0, duration, max(step, 0.25)).tolist()
        beats = librosa.time_to_frames(np.array(beat_times), sr=sr)

    beat_frames = np.array(beats, dtype=np.int64)

    if beat_frames.size == 0:
        beat_strengths = []
        percussion_energy = []
        local_tempo = []
    else:
        onset_idx = np.clip(beat_frames, 0, max(0, len(onset_env) - 1))
        rms_idx = np.clip((beat_frames * (512 / 512)).astype(np.int64), 0, max(0, len(rms) - 1))

        beat_strengths_arr = _normalize_series(onset_env[onset_idx])
        percussion_arr = _normalize_series(rms[rms_idx])

        if np.size(tempo_f) > 0:
            tempo_vec = np.ravel(np.array(tempo_f, dtype=np.float32))
            tempo_idx = np.clip(onset_idx, 0, max(0, len(tempo_vec) - 1))
            local_tempo_arr = tempo_vec[tempo_idx]
        else:
            local_tempo_arr = np.ones_like(beat_strengths_arr) * max(80.0, bpm)

        beat_strengths = beat_strengths_arr.tolist()
        percussion_energy = percussion_arr.tolist()
        local_tempo = local_tempo_arr.astype(np.float32).tolist()

    if bpm < 60.0 or bpm > 190.0:
        logger.warning("Beat-analysis: BPM %.2f вне типового диапазона, точность может быть ниже", bpm)
        confidence_low = True

    result = BeatAnalysisResult(
        bpm=bpm,
        beats=beat_times,
        confidence_low=confidence_low,
        beat_strengths=beat_strengths,
        local_tempo=local_tempo,
        percussion_energy=percussion_energy,
    )
    _last_beat_cache[cache_key] = result
    logger.info(
        "Beat-analysis: bpm=%.2f, beats=%d, strengths=%d, perc=%d, local_tempo=%d",
        result.bpm,
        len(result.beats),
        len(result.beat_strengths),
        len(result.percussion_energy),
        len(result.local_tempo),
    )
    return result
