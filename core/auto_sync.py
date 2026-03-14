from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np

from pathlib import Path

from core.audio import AudioError, probe_audio_duration
from core.line_alignment import (
    LineAlignmentConfig,
    RecognizedWord,
    SegmentInfo,
    align_lyric_lines,
    build_recognized_words,
)
from models import LyricLine

logger = logging.getLogger(__name__)


class AutoSyncError(RuntimeError):
    pass


def _split_lyrics_text(full_text: str) -> list[str]:
    lines = [line.strip() for line in full_text.splitlines()]
    return [line for line in lines if line]


def _format_mmss(seconds: float) -> str:
    safe_seconds = max(0.0, seconds)
    mm = int(safe_seconds // 60)
    ss = safe_seconds - (mm * 60)
    return f"{mm:02d}:{ss:05.2f}"


def _guess_language_code(lines: list[str]) -> str:
    joined = " ".join(lines)
    if re.search(r"[а-яА-ЯёЁ]", joined):
        return "ru"
    return "en"


def _run_demucs_vocals_stem(audio_path: str) -> str | None:
    if shutil.which("demucs") is None:
        logger.info("Автосинхронизация: Demucs CLI не найден, используем full mix")
        return None

    src_path = Path(audio_path)
    if not src_path.exists():
        return None

    stat = src_path.stat()
    cache_key = f"{src_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
    cache_hash = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:16]
    cache_dir = Path(tempfile.gettempdir()) / "lvg_demucs_cache" / cache_hash
    cache_vocals = cache_dir / "vocals.wav"

    if cache_vocals.exists() and cache_vocals.stat().st_size > 0:
        logger.info("Автосинхронизация: переиспользуем vocals stem из кэша: %s", cache_vocals)
        return str(cache_vocals)

    cache_dir.mkdir(parents=True, exist_ok=True)

    last_err = ""
    for model_name in ("htdemucs_6s", "htdemucs"):
        cmd = [
            "demucs",
            "-n",
            model_name,
            "--device",
            "cpu",
            "-o",
            str(cache_dir),
            audio_path,
        ]
        logger.info("Автосинхронизация: запуск Demucs model=%s", model_name)
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0:
            vocals_candidates = list(cache_dir.glob("**/vocals.wav"))
            if vocals_candidates:
                cached = vocals_candidates[0]
                shutil.copy2(cached, cache_vocals)
                logger.info("Автосинхронизация: Demucs vocals stem подготовлен: %s", cache_vocals)
                return str(cache_vocals)
            last_err = "Demucs завершился без vocals stem"
            continue
        last_err = (res.stderr or res.stdout or "")[-800:]

    logger.warning("Автосинхронизация: Demucs недоступен/ошибка, fallback на full mix: %s", last_err)
    return None


def _extract_words_and_segments(aligned_result: dict) -> tuple[list[dict], list[dict]]:
    words: list[dict] = []
    segments: list[dict] = []
    for seg in aligned_result.get("segments") or []:
        segments.append(
            {
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": seg.get("text") or "",
            }
        )
        for word in seg.get("words") or []:
            words.append(
                {
                    "word": word.get("word") or word.get("text") or "",
                    "start": word.get("start"),
                    "end": word.get("end"),
                    "score": word.get("score"),
                }
            )
    return words, segments


def _auto_sync_whisperx_word_level(audio_path: str, lines: list[str]) -> list[LyricLine]:
    try:
        import whisperx
    except ImportError as exc:
        raise AutoSyncError("backend whisperx недоступен") from exc

    try:
        duration = probe_audio_duration(Path(audio_path))
    except AudioError as exc:
        raise AutoSyncError(f"Не удалось подготовить alignment: {exc}") from exc
    if duration <= 0:
        raise AutoSyncError("Не удалось определить длительность аудио для alignment")

    language_code = _guess_language_code(lines)
    russian_mode = language_code == "ru"

    source_audio = audio_path
    source_type = "full_mix"
    vocals_stem = _run_demucs_vocals_stem(audio_path)
    if vocals_stem:
        source_audio = vocals_stem
        source_type = "vocals_stem"

    logger.info("Автосинхронизация: backend=whisperx word-level, lang=%s, source=%s", language_code, source_type)

    device = "cpu"
    compute_type = "int8"

    try:
        audio = whisperx.load_audio(source_audio)
        model = whisperx.load_model("small", device, compute_type=compute_type, language=language_code)
        transcription = model.transcribe(audio, batch_size=8)

        segments = transcription.get("segments") or []
        if not segments:
            raise AutoSyncError("WhisperX не вернул сегменты транскрипции")

        align_model, metadata = whisperx.load_align_model(language_code=language_code, device=device)
        aligned = whisperx.align(
            segments,
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
    except AutoSyncError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AutoSyncError(f"Ошибка WhisperX word alignment: {exc}") from exc

    words_raw, segments_raw = _extract_words_and_segments(aligned)
    recognized_words: list[RecognizedWord] = build_recognized_words(words_raw, russian_mode=russian_mode)
    segment_infos = [SegmentInfo(start=s.get("start"), end=s.get("end"), text=str(s.get("text") or "")) for s in segments_raw]

    if len(recognized_words) < 2:
        raise AutoSyncError("WhisperX вернул слишком мало слов с таймингом")

    cfg = LineAlignmentConfig(russian_mode=russian_mode)
    line_results = align_lyric_lines(lines, recognized_words, segments=segment_infos, config=cfg)

    lyrics: list[LyricLine] = [
        LyricLine(start_time=_format_mmss(item.start_time_seconds), text=item.text)
        for item in line_results
    ]

    fallback_lines = sum(1 for item in line_results if item.status.startswith("fallback"))
    logger.info(
        "Автосинхронизация/whisperx: source=%s words=%d segments=%d lyric_lines=%d fallback_lines=%d",
        source_type,
        len(recognized_words),
        len(segment_infos),
        len(lyrics),
        fallback_lines,
    )
    logger.debug(
        "Автосинхронизация/whisperx: statuses=%s",
        json.dumps([item.status for item in line_results], ensure_ascii=False),
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
        aligned = _auto_sync_whisperx_word_level(audio_path, lines)
        backend = "whisperx_word_level"
    except AutoSyncError as align_exc:
        logger.warning("Автосинхронизация: whisperx недоступен/ошибка (%s), fallback на librosa", align_exc)
        aligned = _auto_sync_librosa(audio_path, lines)
        backend = "librosa"

    elapsed = time.perf_counter() - started
    logger.info("Автосинхронизация: завершено за %.2fs, backend=%s, найдено строк=%d", elapsed, backend, len(aligned))
    return aligned
