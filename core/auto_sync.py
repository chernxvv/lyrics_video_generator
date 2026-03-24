# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Roman Chernov (romanchernovv@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.

from __future__ import annotations

import json
import logging
import os
import re
import time
import warnings

import numpy as np

from pathlib import Path

from core.audio import AudioError, probe_audio_duration
from core.demucs_cache import ensure_demucs_stems_cached
from core.line_alignment import (
    LineAlignmentConfig,
    RecognizedWord,
    SegmentInfo,
    LineTimingResult,
    align_lyric_lines,
    build_recognized_words,
)
from models import LyricLine

logger = logging.getLogger(__name__)


class AutoSyncError(RuntimeError):
    pass


OPTIONAL_AUTOSYNC_PACKAGES = ("librosa", "soundfile", "whisperx", "demucs")


def get_missing_autosync_packages() -> list[str]:
    missing: list[str] = []
    for module_name in OPTIONAL_AUTOSYNC_PACKAGES:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    return missing


def build_autosync_dependency_error(missing_packages: list[str]) -> str:
    packages = ", ".join(missing_packages)
    return (
        "Автосинхронизация требует optional-зависимости, которые не установлены: "
        f"{packages}.\n\n"
        "Установите их командой:\n"
        "pip install -r requirements-autosync.txt"
    )


def _split_lyrics_text(full_text: str) -> list[str]:
    lines = [line.strip() for line in full_text.splitlines()]
    return [line for line in lines if line]


def _format_mmss(seconds: float) -> str:
    safe_seconds = max(0.0, seconds)
    total_centiseconds = int(round(safe_seconds * 100))
    mm, centiseconds_remainder = divmod(total_centiseconds, 60 * 100)
    ss = centiseconds_remainder / 100
    return f"{mm:02d}:{ss:05.2f}"


def _guess_language_code(lines: list[str]) -> str:
    joined = " ".join(lines)
    if re.search(r"[а-яА-ЯёЁ]", joined):
        return "ru"
    return "en"


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


def _transcription_coverage_is_too_low(
    *,
    transcription: dict,
    audio_duration_s: float,
    lyric_line_count: int,
) -> bool:
    segments = transcription.get("segments") or []
    if not segments:
        return True

    last_segment_end = 0.0
    for segment in segments:
        seg_end = segment.get("end")
        if seg_end is None:
            continue
        last_segment_end = max(last_segment_end, float(seg_end))

    word_count = 0
    for segment in segments:
        words = segment.get("words")
        if words:
            word_count += len(words)

    covered_ratio = 0.0 if audio_duration_s <= 0 else (last_segment_end / max(1e-6, audio_duration_s))
    min_expected_words = max(8, lyric_line_count * 2)
    return covered_ratio < 0.78 or word_count < min_expected_words


def _alignment_material_is_too_sparse(*, recognized_word_count: int, lyric_line_count: int) -> bool:
    return recognized_word_count < max(24, lyric_line_count * 3)


def _should_prefer_retry_source(*, current_word_count: int, retry_word_count: int) -> bool:
    if retry_word_count <= current_word_count:
        return False
    # Избегаем лишних переключений источника ради минимального прироста.
    min_delta = max(8, int(current_word_count * 0.15))
    return (retry_word_count - current_word_count) >= min_delta


def _whisperx_alignment_quality_is_poor(
    *,
    line_results: list[LineTimingResult],
    track_duration_s: float,
    recognized_word_count: int,
) -> tuple[bool, str]:
    if not line_results:
        return True, "empty_line_results"

    total = len(line_results)
    fallback_lines = sum(1 for item in line_results if item.status.startswith("fallback"))
    matched_lines = [item for item in line_results if item.status.startswith("matched")]
    fallback_ratio = fallback_lines / max(1, total)
    latest_matched_start = max((item.start_time_seconds for item in matched_lines), default=0.0)
    coverage_ratio = 0.0 if track_duration_s <= 0 else (latest_matched_start / max(1e-6, track_duration_s))
    words_per_line = recognized_word_count / max(1, total)

    if coverage_ratio < 0.62:
        return True, f"low_timeline_coverage:{coverage_ratio:.3f}"
    if fallback_ratio > 0.35 and coverage_ratio < 0.78:
        return True, f"high_fallback_ratio:{fallback_ratio:.3f}"
    if fallback_ratio > 0.42:
        return True, f"too_many_fallbacks:{fallback_ratio:.3f}"
    if words_per_line < 2.8 and coverage_ratio < 0.82:
        return True, f"sparse_alignment_material:{words_per_line:.3f}"
    return False, ""


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

    # Убираем шумный warning от HuggingFace про Xet, если ускоритель не установлен.
    # Это не влияет на корректность загрузки моделей, только на способ скачивания.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    source_audio = audio_path
    source_type = "full_mix"
    try:
        demucs_cache = ensure_demucs_stems_cached(audio_path, preferred_models=("htdemucs_6s", "htdemucs"))
        vocals_stem = demucs_cache.stem_paths.get("vocals")
    except RuntimeError as exc:
        logger.warning("Автосинхронизация: Demucs недоступен/ошибка, fallback на full mix: %s", exc)
        vocals_stem = None

    if vocals_stem:
        source_audio = str(vocals_stem)
        source_type = "vocals_stem"
        logger.info(
            "Автосинхронизация: Demucs cache reused=%s model=%s stems=%s",
            demucs_cache.reused,
            demucs_cache.model_name,
            sorted(demucs_cache.stem_paths.keys()),
        )

    logger.info("Автосинхронизация: backend=whisperx word-level, lang=%s, source=%s", language_code, source_type)

    device = "cpu"
    compute_type = "int8"

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*torchcodec is not installed correctly.*",
                category=UserWarning,
            )

            align_model, metadata = whisperx.load_align_model(language_code=language_code, device=device)

            def transcribe_with_recovery(audio_file: str, *, label: str) -> tuple[dict, np.ndarray]:
                loaded_audio = whisperx.load_audio(audio_file)
                try:
                    model = whisperx.load_model(
                        "small",
                        device,
                        compute_type=compute_type,
                        language=language_code,
                        vad_method="silero",
                    )
                    used_vad_local = True
                except TypeError:
                    model = whisperx.load_model("small", device, compute_type=compute_type, language=language_code)
                    used_vad_local = False
                local_transcription = model.transcribe(loaded_audio, batch_size=8)

                if used_vad_local and _transcription_coverage_is_too_low(
                    transcription=local_transcription,
                    audio_duration_s=duration,
                    lyric_line_count=len(lines),
                ):
                    logger.warning(
                        "Автосинхронизация: низкое покрытие сегментов с VAD на %s, повторяем транскрипцию через "
                        "альтернативный VAD (duration=%.2fs, segments=%d)",
                        label,
                        duration,
                        len(local_transcription.get("segments") or []),
                    )
                    model_retry = whisperx.load_model(
                        "small",
                        device,
                        compute_type=compute_type,
                        language=language_code,
                    )
                    retry_transcription = model_retry.transcribe(loaded_audio, batch_size=8)
                    if not _transcription_coverage_is_too_low(
                        transcription=retry_transcription,
                        audio_duration_s=duration,
                        lyric_line_count=len(lines),
                    ):
                        local_transcription = retry_transcription

                return local_transcription, loaded_audio

            def align_for_source(audio_file: str, *, label: str) -> tuple[list[dict], list[dict]]:
                local_transcription, loaded_audio = transcribe_with_recovery(audio_file, label=label)
                local_segments = local_transcription.get("segments") or []
                if not local_segments:
                    return [], []
                aligned_local = whisperx.align(
                    local_segments,
                    align_model,
                    metadata,
                    loaded_audio,
                    device,
                    return_char_alignments=False,
                )
                return _extract_words_and_segments(aligned_local)

            words_raw, segments_raw = align_for_source(source_audio, label=source_type)
            recognized_words = build_recognized_words(words_raw, russian_mode=russian_mode)

            if source_type == "vocals_stem" and _alignment_material_is_too_sparse(
                recognized_word_count=len(recognized_words),
                lyric_line_count=len(lines),
            ):
                logger.warning(
                    "Автосинхронизация: слишком мало выровненных слов на vocals stem (words=%d, lines=%d), "
                    "повторяем pipeline на full mix",
                    len(recognized_words),
                    len(lines),
                )
                words_mix_raw, segments_mix_raw = align_for_source(audio_path, label="full_mix")
                recognized_words_mix = build_recognized_words(words_mix_raw, russian_mode=russian_mode)
                if _should_prefer_retry_source(
                    current_word_count=len(recognized_words),
                    retry_word_count=len(recognized_words_mix),
                ):
                    source_audio = audio_path
                    source_type = "full_mix_recovered"
                    words_raw = words_mix_raw
                    segments_raw = segments_mix_raw
                    recognized_words = recognized_words_mix

        if not segments_raw:
            raise AutoSyncError("WhisperX не вернул сегменты транскрипции")
    except AutoSyncError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AutoSyncError(f"Ошибка WhisperX word alignment: {exc}") from exc

    segment_infos = [SegmentInfo(start=s.get("start"), end=s.get("end"), text=str(s.get("text") or "")) for s in segments_raw]

    if len(recognized_words) < 2:
        raise AutoSyncError("WhisperX вернул слишком мало слов с таймингом")

    cfg = LineAlignmentConfig(russian_mode=russian_mode)
    line_results = align_lyric_lines(lines, recognized_words, segments=segment_infos, config=cfg)
    quality_is_poor, quality_reason = _whisperx_alignment_quality_is_poor(
        line_results=line_results,
        track_duration_s=duration,
        recognized_word_count=len(recognized_words),
    )
    if quality_is_poor:
        raise AutoSyncError(f"WhisperX alignment quality is too low ({quality_reason})")

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
    missing_packages = get_missing_autosync_packages()
    if missing_packages:
        raise AutoSyncError(build_autosync_dependency_error(missing_packages))

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
