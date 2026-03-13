from __future__ import annotations

import csv
import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
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
    confidence: float
    color_index: int
    decay_ms: float
    vocal_mod: float = 0.0


@dataclass(slots=True)
class StemFeatures:
    name: str
    times: list[float] = field(default_factory=list)
    onset_strength: list[float] = field(default_factory=list)
    energy_envelope: list[float] = field(default_factory=list)
    spectral_flux: list[float] = field(default_factory=list)
    transient_ratio: list[float] = field(default_factory=list)
    quality_mask: list[float] = field(default_factory=list)


@dataclass(slots=True)
class BeatAnalysisResult:
    bpm: float
    beats: list[float]
    confidence_low: bool = False
    beat_strengths: list[float] = field(default_factory=list)
    local_tempo: list[float] = field(default_factory=list)
    percussion_energy: list[float] = field(default_factory=list)
    events: list[BeatEvent] = field(default_factory=list)
    stem_features: dict[str, StemFeatures] = field(default_factory=dict)
    macro_intensity: list[float] = field(default_factory=list)
    section_ids: list[int] = field(default_factory=list)
    vocal_presence: list[float] = field(default_factory=list)


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
        return values.astype(np.float32)
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
            cmd = ["demucs", "-n", model_name, "--device", "cpu", "-o", tmpdir, audio_path]
            logger.info("Beat-analysis: запуск Demucs model=%s", model_name)
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                break
            last_err = (res.stderr or res.stdout or "")[-600:]
        else:
            raise AudioAnalysisError(f"Demucs завершился с ошибкой: {last_err}")

        stem_dirs = list(Path(tmpdir).glob("**/drums.wav"))
        if not stem_dirs:
            raise AudioAnalysisError("Demucs не вернул stems")

        base_dir = stem_dirs[0].parent
        stem_names = ["drums", "bass", "guitar", "piano", "other", "vocals"]

        loaded: dict[str, np.ndarray] = {}
        sr_ref: int | None = None
        for stem in stem_names:
            f = base_dir / f"{stem}.wav"
            if not f.exists():
                continue
            y, sr = librosa.load(str(f), sr=None, mono=True)
            if sr_ref is None:
                sr_ref = sr
            elif sr != sr_ref:
                y, _ = librosa.load(str(f), sr=sr_ref, mono=True)
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


def _compute_stem_features(y: np.ndarray, sr: int, hop_length: int, stem_name: str) -> StemFeatures:
    try:
        import librosa
    except ImportError as exc:
        raise AudioAnalysisError("Для BPM-анализа нужен librosa") from exc

    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length).flatten()
    stft_mag = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    flux = np.maximum(0.0, np.diff(stft_mag, axis=1)).mean(axis=0)

    size = min(len(onset), len(rms), len(flux))
    if size <= 4:
        times = np.arange(size, dtype=np.float32) * (hop_length / sr)
        empty = np.zeros(size, dtype=np.float32)
        return StemFeatures(
            name=stem_name,
            times=times.tolist(),
            onset_strength=empty.tolist(),
            energy_envelope=empty.tolist(),
            spectral_flux=empty.tolist(),
            transient_ratio=empty.tolist(),
            quality_mask=empty.tolist(),
        )

    onset = onset[:size]
    rms = rms[:size]
    flux = flux[:size]

    onset_n = _normalize_series(onset)
    rms_n = _normalize_series(rms)
    flux_n = _normalize_series(flux)

    energy_env = _normalize_series(0.55 * rms_n + 0.45 * onset_n)
    transient_ratio = _normalize_series(onset_n / np.maximum(rms_n, 1e-4))

    flatness = librosa.feature.spectral_flatness(y=y, n_fft=2048, hop_length=hop_length).flatten()[:size]
    flatness_n = _normalize_series(flatness)

    quality = _normalize_series(0.45 * transient_ratio + 0.35 * flux_n + 0.25 * energy_env - 0.28 * flatness_n)
    times = np.arange(size, dtype=np.float32) * (hop_length / sr)

    return StemFeatures(
        name=stem_name,
        times=times.tolist(),
        onset_strength=onset_n.tolist(),
        energy_envelope=energy_env.tolist(),
        spectral_flux=flux_n.tolist(),
        transient_ratio=transient_ratio.tolist(),
        quality_mask=quality.tolist(),
    )


def _build_macro_intensity_and_sections(drums_env: np.ndarray, hop_s: float) -> tuple[np.ndarray, np.ndarray]:
    if drums_env.size == 0:
        return drums_env.astype(np.float32), np.zeros(0, dtype=np.int32)

    smooth_win = max(5, int(1.6 / max(hop_s, 1e-4)))
    kernel = np.ones(smooth_win, dtype=np.float32) / smooth_win
    smooth = np.convolve(drums_env, kernel, mode="same")
    macro = _normalize_series(smooth)

    novelty = np.abs(np.diff(macro, prepend=macro[0]))
    boundary_thr = float(np.percentile(novelty, 88))
    section_ids = np.zeros_like(macro, dtype=np.int32)

    sid = 0
    min_gap = max(10, int(2.0 / max(hop_s, 1e-4)))
    last_boundary = -min_gap
    for i in range(len(macro)):
        if novelty[i] > boundary_thr and (i - last_boundary) >= min_gap:
            sid += 1
            last_boundary = i
        section_ids[i] = sid

    return macro.astype(np.float32), section_ids


def _build_event_timeline(
    stem_features: dict[str, StemFeatures],
    macro_intensity: np.ndarray,
    section_ids: np.ndarray,
    vocal_presence: np.ndarray,
    sr: int,
    hop_length: int,
) -> tuple[list[BeatEvent], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_stems = [stem for stem in ("drums", "guitar", "piano", "other") if stem in stem_features]
    if not target_stems:
        target_stems = ["drums"] if "drums" in stem_features else list(stem_features.keys())[:1]

    frame_count = min(len(stem_features[s].times) for s in target_stems)
    if frame_count <= 2:
        return [], np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)

    stem_weights = {"drums": 1.0, "guitar": 0.82, "piano": 0.78, "other": 0.64}

    score_matrix = []
    quality_matrix = []
    for s in target_stems:
        sf = stem_features[s]
        onset = np.array(sf.onset_strength[:frame_count], dtype=np.float32)
        env = np.array(sf.energy_envelope[:frame_count], dtype=np.float32)
        flux = np.array(sf.spectral_flux[:frame_count], dtype=np.float32)
        quality = np.array(sf.quality_mask[:frame_count], dtype=np.float32)

        raw_score = (0.45 * onset + 0.35 * env + 0.32 * flux) * quality * stem_weights.get(s, 0.6)
        score_matrix.append(raw_score)
        quality_matrix.append(quality)

    score_mat = np.vstack(score_matrix)
    quality_mat = np.vstack(quality_matrix)
    winners = np.argmax(score_mat, axis=0)
    winner_scores = np.max(score_mat, axis=0)
    winner_quality = np.max(quality_mat, axis=0)

    global_quality_mask = _normalize_series(np.mean(quality_mat, axis=0))
    score_thr = float(np.percentile(winner_scores, 60)) if winner_scores.size else 0.0
    conf_thr = float(np.percentile(global_quality_mask, 42)) if global_quality_mask.size else 0.0

    cooldown_frames = max(1, int(0.09 * sr / hop_length))
    events: list[BeatEvent] = []
    last_event_frame = -cooldown_frames
    prev_stem = ""

    chosen_stem_idx = np.full(frame_count, -1, dtype=np.int32)
    event_intensity_line = np.zeros(frame_count, dtype=np.float32)

    for i in range(1, frame_count - 1):
        if i - last_event_frame < cooldown_frames:
            continue

        current = winner_scores[i]
        confidence = float(np.clip(0.55 * winner_quality[i] + 0.45 * global_quality_mask[i], 0.0, 1.0))

        if current < score_thr or confidence < conf_thr:
            continue
        if not (current >= winner_scores[i - 1] and current >= winner_scores[i + 1]):
            continue

        stem = target_stems[int(winners[i])]
        if prev_stem and stem != prev_stem and current < score_thr * 1.2:
            continue

        macro = float(macro_intensity[i]) if i < len(macro_intensity) else 0.5
        vocal_mod = float(vocal_presence[i]) if i < len(vocal_presence) else 0.0
        section = int(section_ids[i]) if i < len(section_ids) else 0

        # тихие section-ы suppress, активные разрешают больше динамики
        section_gain = 0.72 + 0.60 * macro

        intensity = float(np.clip((0.30 + 0.95 * current + 0.55 * confidence) * section_gain + 0.18 * vocal_mod, 0.0, 1.7))
        chaos = float(np.clip(0.20 + 1.10 * current + 0.42 * macro + 0.25 * vocal_mod, 0.08, 1.9))

        color_index_map = {"drums": 0, "guitar": 1, "piano": 2, "other": 3, "bass": 4, "vocals": 5}
        color_index = color_index_map.get(stem, 0)
        decay_ms = float(np.clip(180 + 260 * (1.0 - macro) + 150 * (1.0 - confidence), 120, 520))

        events.append(
            BeatEvent(
                time=float(stem_features[stem].times[i]),
                intensity=intensity,
                chaos=chaos,
                source_stem=stem,
                confidence=confidence,
                color_index=color_index,
                decay_ms=decay_ms,
                vocal_mod=vocal_mod,
            )
        )
        chosen_stem_idx[i] = color_index
        event_intensity_line[i] = intensity
        prev_stem = stem
        last_event_frame = i

    return events, global_quality_mask, winner_scores, chosen_stem_idx, event_intensity_line


def _save_debug_output(
    debug_prefix: str,
    times: np.ndarray,
    stem_features: dict[str, StemFeatures],
    global_quality_mask: np.ndarray,
    winner_scores: np.ndarray,
    chosen_stem_idx: np.ndarray,
    event_intensity_line: np.ndarray,
    macro_intensity: np.ndarray,
    events: list[BeatEvent],
) -> None:
    prefix = Path(debug_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")

    stem_names = sorted(stem_features.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        header = [
            "t",
            *[f"score_{stem}" for stem in stem_names],
            *[f"quality_{stem}" for stem in stem_names],
            "quality_mask",
            "winner_score",
            "chosen_stem_idx",
            "event_intensity",
            "macro_intensity",
        ]
        writer.writerow(header)

        for i in range(len(times)):
            row = [float(times[i])]
            for stem in stem_names:
                sf = stem_features[stem]
                onset = sf.onset_strength[i] if i < len(sf.onset_strength) else 0.0
                env = sf.energy_envelope[i] if i < len(sf.energy_envelope) else 0.0
                flux = sf.spectral_flux[i] if i < len(sf.spectral_flux) else 0.0
                quality = sf.quality_mask[i] if i < len(sf.quality_mask) else 0.0
                row.append(float(0.45 * onset + 0.35 * env + 0.32 * flux) * float(quality))
            for stem in stem_names:
                sf = stem_features[stem]
                row.append(float(sf.quality_mask[i] if i < len(sf.quality_mask) else 0.0))
            row.extend(
                [
                    float(global_quality_mask[i] if i < len(global_quality_mask) else 0.0),
                    float(winner_scores[i] if i < len(winner_scores) else 0.0),
                    int(chosen_stem_idx[i] if i < len(chosen_stem_idx) else -1),
                    float(event_intensity_line[i] if i < len(event_intensity_line) else 0.0),
                    float(macro_intensity[i] if i < len(macro_intensity) else 0.0),
                ]
            )
            writer.writerow(row)

    payload = {
        "events": [asdict(e) for e in events],
        "stem_names": stem_names,
        "size": int(len(times)),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Beat-analysis debug exported: csv=%s json=%s", csv_path, json_path)


def analyze_beats_multistem(audio_path: str, fps: int, debug_output_prefix: str | None = None) -> BeatAnalysisResult:
    try:
        import librosa
    except ImportError as exc:
        raise AudioAnalysisError("Для BPM-анализа нужен librosa. Установите зависимости из requirements.txt") from exc

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

    stem_features: dict[str, StemFeatures] = {}
    for stem_name, y_stem in stems.items():
        stem_features[stem_name] = _compute_stem_features(y_stem, sr, hop_length, stem_name)

    ref_name = "drums" if "drums" in stem_features else next(iter(stem_features.keys()))
    times_ref = np.array(stem_features[ref_name].times, dtype=np.float32)
    drums_env = np.array(stem_features[ref_name].energy_envelope, dtype=np.float32)

    macro_intensity, section_ids = _build_macro_intensity_and_sections(drums_env, hop_length / sr)

    vocal_presence = np.zeros_like(times_ref, dtype=np.float32)
    if "vocals" in stem_features:
        vp = np.array(stem_features["vocals"].energy_envelope, dtype=np.float32)
        vq = np.array(stem_features["vocals"].quality_mask, dtype=np.float32)
        vocal_presence = _normalize_series(vp * vq)

        # обнуляем длинные участки без вокала (длительные тишины вокала)
        absent_thr = 0.12
        min_absent_frames = max(1, int(10.0 / (hop_length / sr)))
        absent_count = 0
        for i in range(len(vocal_presence)):
            if vocal_presence[i] < absent_thr:
                absent_count += 1
            else:
                absent_count = 0
            if absent_count >= min_absent_frames:
                vocal_presence[i] = 0.0

    events, quality_mask, winner_scores, chosen_stem_idx, event_intensity_line = _build_event_timeline(
        stem_features,
        macro_intensity,
        section_ids,
        vocal_presence,
        sr,
        hop_length,
    )

    beat_frames_arr = np.array(beat_frames, dtype=np.int64)
    if beat_frames_arr.size > 0 and len(drums_env) > 0:
        idx = np.clip(beat_frames_arr, 0, len(drums_env) - 1)
        drums_quality = np.array(stem_features[ref_name].quality_mask, dtype=np.float32)

        beat_strengths = _normalize_series((drums_env * drums_quality)[idx]).tolist()
        percussion_energy = _normalize_series(drums_env[idx]).tolist()

        tempo_f = librosa.feature.tempo(onset_envelope=drums_env, sr=sr, hop_length=hop_length, aggregate=None)
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

    if quality_mask.size > 0:
        low_ratio = float(np.mean(quality_mask < (np.percentile(quality_mask, 35))))
    else:
        low_ratio = 1.0
    vocal_ratio = float(np.mean(vocal_presence > 0.18)) if vocal_presence.size > 0 else 0.0
    avg_vocal_mod = float(np.mean([e.vocal_mod for e in events])) if events else 0.0

    logger.info(
        "Beat-analysis diagnostics: low_conf_ratio=%.3f, vocal_presence_ratio=%.3f, avg_vocal_mod=%.3f",
        low_ratio,
        vocal_ratio,
        avg_vocal_mod,
    )

    if debug_output_prefix:
        _save_debug_output(
            debug_output_prefix,
            times_ref,
            stem_features,
            quality_mask,
            winner_scores,
            chosen_stem_idx,
            event_intensity_line,
            macro_intensity,
            events,
        )

    result = BeatAnalysisResult(
        bpm=bpm,
        beats=beat_times,
        confidence_low=confidence_low,
        beat_strengths=beat_strengths,
        local_tempo=local_tempo,
        percussion_energy=percussion_energy,
        events=events,
        stem_features=stem_features,
        macro_intensity=macro_intensity.astype(np.float32).tolist(),
        section_ids=section_ids.astype(int).tolist(),
        vocal_presence=vocal_presence.astype(np.float32).tolist(),
    )

    logger.info(
        "Beat-analysis: bpm=%.2f, beats=%d, events=%d, source=%s",
        result.bpm,
        len(result.beats),
        len(result.events),
        "demucs_6s" if used_demucs else "hpss-fallback",
    )
    return result


def analyze_bpm_and_beats(audio_path: str, fps: int) -> BeatAnalysisResult:
    cache_key = (audio_path, fps)
    if cache_key in _last_beat_cache:
        logger.info("Beat-analysis: использован runtime cache")
        return _last_beat_cache[cache_key]

    # optional debug path controlled by env var to avoid GUI/API signature changes
    debug_prefix = None
    import os

    env_debug = os.getenv("LVG_BPM_DEBUG_PREFIX", "").strip()
    if env_debug:
        debug_prefix = env_debug

    result = analyze_beats_multistem(audio_path, fps, debug_output_prefix=debug_prefix)
    _last_beat_cache[cache_key] = result
    return result
