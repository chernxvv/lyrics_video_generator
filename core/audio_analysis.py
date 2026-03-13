from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class AudioAnalysisError(RuntimeError):
    pass


@dataclass(slots=True)
class BeatEvent:
    time: float
    intensity: float
    chaos: float
    source_stem: str
    vocal_mod: float = 0.0


@dataclass(slots=True)
class BeatAnalysisResult:
    bpm: float
    beats: list[float]
    confidence_low: bool = False
    beat_strengths: list[float] = field(default_factory=list)
    local_tempo: list[float] = field(default_factory=list)
    percussion_energy: list[float] = field(default_factory=list)
    events: list[BeatEvent] = field(default_factory=list)


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


def _align_length(source: np.ndarray, target_len: int) -> np.ndarray:
    if len(source) == target_len:
        return source
    if len(source) > target_len:
        return source[:target_len]
    pad = np.zeros(target_len - len(source), dtype=source.dtype)
    return np.concatenate([source, pad])


def _run_demucs_6s(audio_path: str) -> tuple[dict[str, np.ndarray], int]:
    try:
        import librosa
    except ImportError as exc:
        raise AudioAnalysisError("Для BPM-анализа нужен librosa") from exc

    if shutil.which("demucs") is None:
        raise AudioAnalysisError("Demucs CLI не найден в PATH")

    with tempfile.TemporaryDirectory(prefix="lvg_demucs6s_") as tmpdir:
        model_names = ["htdemucs_6s", "htdemucs"]
        last_err = ""
        for model_name in model_names:
            cmd = [
                "demucs",
                "-n",
                model_name,
                "--device",
                "cpu",
                "-o",
                tmpdir,
                audio_path,
            ]
            logger.info("Beat-analysis: запуск Demucs model=%s", model_name)
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                break
            last_err = (res.stderr or res.stdout or "")[-500:]
        else:
            raise AudioAnalysisError(f"Demucs завершился с ошибкой: {last_err}")

        stem_root = Path(tmpdir)
        # find output dir containing stems
        stem_dirs = list(stem_root.glob("**/drums.wav"))
        if not stem_dirs:
            raise AudioAnalysisError("Demucs не вернул stems")
        base_dir = stem_dirs[0].parent

        stem_names = ["drums", "bass", "guitar", "piano", "other", "vocals"]
        loaded: dict[str, np.ndarray] = {}
        sr_ref: int | None = None

        for stem in stem_names:
            stem_file = base_dir / f"{stem}.wav"
            if not stem_file.exists():
                continue
            y, sr = librosa.load(str(stem_file), sr=None, mono=True)
            if sr_ref is None:
                sr_ref = sr
            elif sr != sr_ref:
                y, _ = librosa.load(str(stem_file), sr=sr_ref, mono=True)
            loaded[stem] = y.astype(np.float32)

        if "drums" not in loaded:
            raise AudioAnalysisError("Demucs не вернул обязательный drums stem")

        max_len = max(len(v) for v in loaded.values())
        for k, v in list(loaded.items()):
            loaded[k] = _align_length(v, max_len)

        logger.info("Beat-analysis: Demucs stems loaded=%s", sorted(loaded.keys()))
        return loaded, int(sr_ref or 44100)


def _load_hpss_fallback(audio_path: str) -> tuple[dict[str, np.ndarray], int]:
    try:
        import librosa
    except ImportError as exc:
        raise AudioAnalysisError("Для BPM-анализа нужен librosa") from exc

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    _, y_perc = librosa.effects.hpss(y)
    y = y.astype(np.float32)
    y_perc = y_perc.astype(np.float32)
    return {"drums": y_perc, "other": y}, sr


def _compute_stem_activity(y: np.ndarray, sr: int, hop_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import librosa
    except ImportError as exc:
        raise AudioAnalysisError("Для BPM-анализа нужен librosa") from exc

    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length).flatten()
    envelope = _normalize_series(0.6 * _align_length(onset, len(rms)) + 0.4 * rms)

    transient_ratio = _normalize_series(_align_length(onset, len(rms)) / np.maximum(rms, 1e-4))
    quality = _normalize_series(0.7 * transient_ratio + 0.3 * envelope)
    times = np.arange(len(envelope), dtype=np.float32) * (hop_length / sr)
    return envelope.astype(np.float32), quality.astype(np.float32), times


def _build_event_timeline(
    activities: dict[str, np.ndarray],
    qualities: dict[str, np.ndarray],
    times: np.ndarray,
    vocal_presence: np.ndarray,
    sr: int,
    hop_length: int,
) -> list[BeatEvent]:
    stem_weights = {
        "drums": 1.00,
        "guitar": 0.82,
        "piano": 0.78,
        "other": 0.66,
    }

    target_stems = [stem for stem in ("drums", "guitar", "piano", "other") if stem in activities]
    if not target_stems:
        target_stems = ["drums"] if "drums" in activities else list(activities.keys())[:1]

    frame_count = len(times)
    scores = {stem: np.zeros(frame_count, dtype=np.float32) for stem in target_stems}

    for stem in target_stems:
        scores[stem] = activities[stem] * qualities[stem] * stem_weights.get(stem, 0.55)

    score_matrix = np.vstack([scores[s] for s in target_stems])
    winners = np.argmax(score_matrix, axis=0)
    winner_scores = np.max(score_matrix, axis=0)

    threshold = float(np.percentile(winner_scores, 58)) if winner_scores.size else 0.0
    cooldown_frames = max(1, int(0.09 * sr / hop_length))

    events: list[BeatEvent] = []
    last_event_frame = -cooldown_frames
    prev_stem = ""

    for i in range(1, frame_count - 1):
        if i - last_event_frame < cooldown_frames:
            continue

        current = winner_scores[i]
        if current < threshold:
            continue

        if not (current >= winner_scores[i - 1] and current >= winner_scores[i + 1]):
            continue

        stem = target_stems[int(winners[i])]
        if prev_stem and stem != prev_stem and current < (threshold * 1.2):
            continue

        vocal_mod = float(vocal_presence[i]) if i < len(vocal_presence) else 0.0
        intensity = float(np.clip(0.35 + 0.95 * current + 0.20 * vocal_mod, 0.0, 1.6))
        chaos = float(np.clip(0.25 + 1.10 * current + 0.35 * vocal_mod, 0.1, 1.8))

        events.append(
            BeatEvent(
                time=float(times[i]),
                intensity=intensity,
                chaos=chaos,
                source_stem=stem,
                vocal_mod=vocal_mod,
            )
        )
        prev_stem = stem
        last_event_frame = i

    return events


def analyze_bpm_and_beats(audio_path: str, fps: int) -> BeatAnalysisResult:
    cache_key = (audio_path, fps)
    if cache_key in _last_beat_cache:
        logger.info("Beat-analysis: использован runtime cache")
        return _last_beat_cache[cache_key]

    try:
        import librosa
    except ImportError as exc:
        raise AudioAnalysisError("Для BPM-анализа нужен librosa. Установите зависимости из requirements.txt") from exc

    logger.info("Beat-analysis: старт для %s", audio_path)
    hop_length = 512

    used_demucs = False
    try:
        stems, sr = _run_demucs_6s(audio_path)
        used_demucs = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Beat-analysis: Demucs недоступен/ошибка (%s), fallback на HPSS", exc)
        stems, sr = _load_hpss_fallback(audio_path)

    y_rhythm = stems.get("drums")
    if y_rhythm is None:
        y_rhythm = next(iter(stems.values()))

    tempo, beat_frames = librosa.beat.beat_track(y=y_rhythm, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length).tolist()

    bpm_candidate = _to_scalar_bpm(tempo)
    bpm = bpm_candidate if np.isfinite(bpm_candidate) else 0.0
    confidence_low = False

    duration = len(y_rhythm) / sr
    if not beat_times:
        confidence_low = True
        logger.warning("Beat-analysis: beat_track вернул пустой timeline, fallback на равномерную сетку")
        if bpm <= 0:
            bpm = 90.0
        step = 60.0 / bpm
        beat_times = np.arange(0, duration, max(step, 0.25)).tolist()
        beat_frames = librosa.time_to_frames(np.array(beat_times), sr=sr, hop_length=hop_length)

    activities: dict[str, np.ndarray] = {}
    qualities: dict[str, np.ndarray] = {}
    times_ref: np.ndarray | None = None

    for stem_name, y_stem in stems.items():
        env, quality, times = _compute_stem_activity(y_stem, sr, hop_length)
        activities[stem_name] = env
        qualities[stem_name] = quality
        if times_ref is None:
            times_ref = times

    assert times_ref is not None

    vocal_presence = np.zeros_like(times_ref, dtype=np.float32)
    if "vocals" in activities:
        vocal_presence = _normalize_series(activities["vocals"] * qualities.get("vocals", 1.0))

    events = _build_event_timeline(activities, qualities, times_ref, vocal_presence, sr, hop_length)

    beat_frames = np.array(beat_frames, dtype=np.int64)
    if beat_frames.size > 0:
        drums_env = activities.get("drums", next(iter(activities.values())))
        drums_quality = qualities.get("drums", np.ones_like(drums_env, dtype=np.float32))
        tempo_f = librosa.feature.tempo(onset_envelope=drums_env, sr=sr, hop_length=hop_length, aggregate=None)

        idx = np.clip(beat_frames, 0, len(drums_env) - 1)
        beat_strengths = _normalize_series((drums_env * drums_quality)[idx]).tolist()
        percussion_energy = _normalize_series(drums_env[idx]).tolist()

        if np.size(tempo_f) > 0:
            tempo_vec = np.ravel(np.array(tempo_f, dtype=np.float32))
            tempo_idx = np.clip(idx, 0, max(0, len(tempo_vec) - 1))
            local_tempo = tempo_vec[tempo_idx].astype(np.float32).tolist()
        else:
            local_tempo = (np.ones(len(idx), dtype=np.float32) * max(80.0, bpm)).tolist()
    else:
        beat_strengths = []
        percussion_energy = []
        local_tempo = []
        confidence_low = True

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
        events=events,
    )
    _last_beat_cache[cache_key] = result

    logger.info(
        "Beat-analysis: bpm=%.2f, beats=%d, events=%d, source=%s",
        result.bpm,
        len(result.beats),
        len(result.events),
        "demucs_6s" if used_demucs else "hpss-fallback",
    )
    return result
