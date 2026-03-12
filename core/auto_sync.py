from __future__ import annotations

import logging
import re
import time
from difflib import SequenceMatcher

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


def _normalize_text(value: str) -> str:
    lowered = value.lower().replace("ё", "е")
    cleaned = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    return " ".join(cleaned.split())


def _line_weight(text: str) -> float:
    letters = re.findall(r"\w", text, flags=re.UNICODE)
    return max(1.0, float(len(letters)))


def _align_text_to_time_candidates(
    lines: list[str],
    candidates: list[tuple[float, str]],
    duration: float,
) -> list[LyricLine]:
    if not candidates:
        raise AutoSyncError("Не удалось получить кандидатов тайминга для автосинхронизации.")

    normalized_candidates = [(_normalize_text(text), t) for t, text in candidates if _normalize_text(text)]
    if not normalized_candidates:
        raise AutoSyncError("Кандидаты тайминга пусты после нормализации текста.")

    aligned: list[LyricLine] = []
    prev = -1.0
    seek_idx = 0

    for line in lines:
        target = _normalize_text(line)
        if not target:
            continue

        best_idx = seek_idx
        best_score = -1.0
        lookahead_end = min(len(normalized_candidates), seek_idx + 24)

        for idx in range(seek_idx, lookahead_end):
            cand_text, _ = normalized_candidates[idx]
            score_single = SequenceMatcher(None, target, cand_text).ratio()

            score_window = score_single
            if idx + 1 < len(normalized_candidates):
                merged = f"{cand_text} {normalized_candidates[idx + 1][0]}"
                score_window = max(score_window, SequenceMatcher(None, target, merged).ratio())

            if score_window > best_score:
                best_score = score_window
                best_idx = idx

        candidate_time = normalized_candidates[best_idx][1]
        start_time = max(prev + 0.25, candidate_time)
        start_time = min(start_time, max(0.0, duration - 0.2))

        aligned.append(LyricLine(start_time=_format_mmss(start_time), text=line))
        prev = start_time
        seek_idx = min(len(normalized_candidates) - 1, best_idx + 1)

    if len(aligned) < 2:
        raise AutoSyncError("Автосинхронизация вернула слишком мало строк.")

    low_match_count = 0
    for idx, line in enumerate(aligned):
        if idx >= len(lines):
            break
        score = SequenceMatcher(None, _normalize_text(lines[idx]), _normalize_text(line.text)).ratio()
        if score < 0.42:
            low_match_count += 1
    if low_match_count > max(1, len(aligned) // 3):
        logger.warning(
            "Автосинхронизация: много частичных совпадений текста (%d из %d), потребуется ручная правка",
            low_match_count,
            len(aligned),
        )

    return aligned


def _auto_sync_whisper(audio_path: str, lines: list[str]) -> list[LyricLine]:
    try:
        import whisper
    except ImportError as exc:
        raise AutoSyncError("backend whisper недоступен") from exc

    logger.info("Автосинхронизация: backend=whisper")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, task="transcribe", word_timestamps=False, verbose=False)

    segments = result.get("segments") or []
    duration = float(result.get("duration") or 0.0)
    candidates: list[tuple[float, str]] = []

    for segment in segments:
        text = str(segment.get("text") or "").strip()
        start = float(segment.get("start") or 0.0)
        if text:
            candidates.append((start, text))

    if duration <= 0:
        max_end = max((float(seg.get("end") or 0.0) for seg in segments), default=0.0)
        duration = max_end

    if duration <= 0 or len(candidates) < 2:
        raise AutoSyncError("Whisper вернул недостаточно сегментов для выравнивания строк")

    return _align_text_to_time_candidates(lines, candidates, duration)


def _auto_sync_librosa(audio_path: str, lines: list[str]) -> list[LyricLine]:
    try:
        import librosa
    except ImportError as exc:
        raise AutoSyncError("Для fallback-автосинхронизации нужен librosa") from exc

    logger.info("Автосинхронизация: backend=librosa (tuned onset+energy)")
    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        y_harmonic, _ = librosa.effects.hpss(y)
        onset_env = librosa.onset.onset_strength(y=y_harmonic, sr=sr, aggregate=np.median)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            backtrack=True,
            pre_max=12,
            post_max=12,
            pre_avg=20,
            post_avg=20,
            delta=0.14,
            wait=8,
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()

        intervals = librosa.effects.split(y_harmonic, top_db=26)
    except Exception as exc:  # noqa: BLE001
        raise AutoSyncError(f"Ошибка анализа аудио librosa: {exc}") from exc

    if duration <= 0:
        raise AutoSyncError("Не удалось определить длительность аудио для автосинхронизации.")

    starts = [0.0]
    for s, _ in intervals:
        point = float(s / sr)
        if point > 0.28:
            starts.append(point)
    starts.extend(float(t) for t in onset_times)
    starts = sorted(set(t for t in starts if t < duration - 0.12))

    if len(starts) < 4:
        logger.warning("Автосинхронизация/librosa: мало стартовых маркеров, fallback на равномерное распределение")
        starts = np.linspace(0, max(0.0, duration - 0.35), num=max(4, len(lines) + 1)).tolist()

    weights = np.array([_line_weight(line) for line in lines], dtype=np.float32)
    weights /= weights.sum()
    target_times = (weights.cumsum() * max(0.0, duration - 0.3)).tolist()

    # Выравниваем строки по ближайшим onset-маркерам с монотонностью.
    aligned: list[LyricLine] = []
    prev = -1.0
    for idx, line_text in enumerate(lines):
        target = target_times[idx]
        nearest = min(starts, key=lambda t: abs(t - target))
        start_time = max(prev + 0.22, nearest)
        start_time = min(start_time, max(0.0, duration - 0.2))
        aligned.append(LyricLine(start_time=_format_mmss(start_time), text=line_text))
        prev = start_time

    if len(aligned) < 2:
        raise AutoSyncError("Автосинхронизация/librosa вернула слишком мало строк.")

    return aligned


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
        aligned = _auto_sync_whisper(audio_path, lines)
        backend = "whisper"
    except AutoSyncError as whisper_exc:
        logger.warning("Автосинхронизация: whisper недоступен/ошибка (%s), fallback на librosa", whisper_exc)
        aligned = _auto_sync_librosa(audio_path, lines)
        backend = "librosa"

    elapsed = time.perf_counter() - started
    logger.info("Автосинхронизация: завершено за %.2fs, backend=%s, найдено строк=%d", elapsed, backend, len(aligned))
    return aligned
