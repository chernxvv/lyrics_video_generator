from __future__ import annotations

import logging
import re
import time

import numpy as np

from pathlib import Path

from core.audio import probe_audio_duration
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


def _guess_language_code(lines: list[str]) -> str:
    joined = " ".join(lines)
    if re.search(r"[а-яА-ЯёЁ]", joined):
        return "ru"
    return "en"


def _build_text_segments(lines: list[str], duration: float) -> list[dict[str, float | str]]:
    weights = np.array([max(1.0, len(re.findall(r"\w", line, flags=re.UNICODE))) for line in lines], dtype=np.float32)
    weights /= weights.sum()
    total = max(0.5, duration - 0.2)
    cumulative = (weights.cumsum() * total).tolist()

    starts = [0.0] + cumulative[:-1]
    ends = cumulative
    segments: list[dict[str, float | str]] = []
    for i, line in enumerate(lines):
        segments.append(
            {
                "id": i,
                "start": float(starts[i]),
                "end": float(max(starts[i] + 0.15, ends[i])),
                "text": line,
            }
        )
    return segments


def _extract_lines_from_aligned_segments(aligned_segments: list[dict]) -> list[tuple[float, str]]:
    result: list[tuple[float, str]] = []
    for segment in aligned_segments:
        text = str(segment.get("text") or "").strip()
        start = segment.get("start")

        if start is None:
            words = segment.get("words") or []
            for word in words:
                if word.get("start") is not None:
                    start = float(word["start"])
                    break

        if text and start is not None:
            result.append((float(start), text))
    return result


def _auto_sync_whisperx_forced(audio_path: str, lines: list[str]) -> list[LyricLine]:
    try:
        import whisperx
    except ImportError as exc:
        raise AutoSyncError("backend whisperx недоступен") from exc

    duration = probe_audio_duration(Path(audio_path))
    if duration <= 0:
        raise AutoSyncError("Не удалось определить длительность аудио для forced alignment")

    language_code = _guess_language_code(lines)
    logger.info("Автосинхронизация: backend=whisperx forced alignment, lang=%s", language_code)

    text_segments = _build_text_segments(lines, duration)

    device = "cpu"
    compute_type = "int8"

    try:
        audio = whisperx.load_audio(audio_path)
        align_model, metadata = whisperx.load_align_model(language_code=language_code, device=device)
        aligned = whisperx.align(
            text_segments,
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise AutoSyncError(f"Ошибка WhisperX forced alignment: {exc}") from exc

    aligned_segments = aligned.get("segments") or []
    line_starts = _extract_lines_from_aligned_segments(aligned_segments)

    if len(line_starts) < 2:
        raise AutoSyncError("WhisperX вернул слишком мало выровненных сегментов")

    lyrics: list[LyricLine] = []
    prev = -1.0
    for idx, original_line in enumerate(lines):
        if idx < len(line_starts):
            start = line_starts[idx][0]
        else:
            start = prev + 0.35
        start = max(prev + 0.2, min(start, max(0.0, duration - 0.2)))
        lyrics.append(LyricLine(start_time=_format_mmss(start), text=original_line))
        prev = start

    logger.info(
        "Автосинхронизация/whisperx: duration=%.2f, aligned_segments=%d, lyric_lines=%d",
        duration,
        len(aligned_segments),
        len(lyrics),
    )
    return lyrics


def _auto_sync_librosa(audio_path: str, lines: list[str]) -> list[LyricLine]:
    try:
        import librosa
    except ImportError as exc:
        raise AutoSyncError("Для fallback-автосинхронизации нужен librosa") from exc

    logger.info("Автосинхронизация: backend=librosa fallback (tuned onset+energy)")
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

    weights = np.array([max(1.0, float(len(re.findall(r"\w", line, flags=re.UNICODE)))) for line in lines], dtype=np.float32)
    weights /= weights.sum()
    target_times = (weights.cumsum() * max(0.0, duration - 0.3)).tolist()

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
        aligned = _auto_sync_whisperx_forced(audio_path, lines)
        backend = "whisperx_forced"
    except AutoSyncError as align_exc:
        logger.warning("Автосинхронизация: whisperx недоступен/ошибка (%s), fallback на librosa", align_exc)
        aligned = _auto_sync_librosa(audio_path, lines)
        backend = "librosa"

    elapsed = time.perf_counter() - started
    logger.info("Автосинхронизация: завершено за %.2fs, backend=%s, найдено строк=%d", elapsed, backend, len(aligned))
    return aligned
