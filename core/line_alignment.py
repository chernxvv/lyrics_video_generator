from __future__ import annotations

import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_STOPWORDS_RU = {
    "и",
    "а",
    "но",
    "я",
    "ты",
    "мы",
    "вы",
    "он",
    "она",
    "они",
    "в",
    "на",
    "по",
    "с",
    "к",
    "не",
    "да",
    "же",
    "ли",
    "бы",
    "то",
}
_STOPWORDS_EN = {
    "a",
    "an",
    "and",
    "or",
    "the",
    "to",
    "of",
    "in",
    "on",
    "for",
    "at",
    "i",
    "you",
    "we",
    "he",
    "she",
    "it",
    "they",
}


@dataclass(slots=True)
class LineAlignmentConfig:
    pre_roll_ms: int = 120
    min_line_gap_ms: int = 120
    max_line_jump_ms: int = 8000
    max_anchor_search_words: int = 10
    min_anchor_word_length: int = 3
    prefer_non_stopword_anchor: bool = True
    low_confidence_threshold: float = 0.45
    allow_segment_fallback: bool = True
    max_window_extra_words: int = 8
    max_candidate_lookahead_words: int = 120
    min_local_match_score: float = 0.52
    time_prior_weight: float = 0.35
    cursor_prior_weight: float = 0.25
    russian_mode: bool = False


@dataclass(slots=True)
class RecognizedWord:
    raw: str
    normalized: str
    start: float | None
    end: float | None
    confidence: float | None
    index: int


@dataclass(slots=True)
class SegmentInfo:
    start: float | None
    end: float | None
    text: str


@dataclass(slots=True)
class LineTimingResult:
    text: str
    start_time_seconds: float
    confidence: float
    status: str
    anchor_word: str = ""
    anchor_word_index: int = -1
    raw_start_seconds: float | None = None
    details: dict[str, str | float | int] = field(default_factory=dict)


def normalize_for_matching(text: str, *, russian_mode: bool = False) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = normalized.replace("ё", "е") if russian_mode else normalized
    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "—": "-",
        "–": "-",
        "‑": "-",
    }
    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst)

    normalized = re.sub(r"[^\w\s\-']", " ", normalized, flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def tokenize_for_matching(text: str, *, russian_mode: bool = False) -> list[str]:
    normalized = normalize_for_matching(text, russian_mode=russian_mode)
    tokens: list[str] = []
    for raw in normalized.split():
        token = re.sub(r"^\W+|\W+$", "", raw, flags=re.UNICODE)
        if token:
            tokens.append(token)
    return tokens


def _is_stopword(token: str, *, russian_mode: bool) -> bool:
    return token in (_STOPWORDS_RU if russian_mode else _STOPWORDS_EN)


def build_recognized_words(words: list[dict], *, russian_mode: bool = False) -> list[RecognizedWord]:
    built: list[RecognizedWord] = []
    for idx, word in enumerate(words):
        raw = str(word.get("word") or word.get("text") or "").strip()
        if not raw:
            continue
        token = tokenize_for_matching(raw, russian_mode=russian_mode)
        normalized = token[0] if token else ""
        if not normalized:
            continue
        start = word.get("start")
        end = word.get("end")
        conf = word.get("score")
        if conf is None:
            conf = word.get("confidence")
        built.append(
            RecognizedWord(
                raw=raw,
                normalized=normalized,
                start=float(start) if start is not None else None,
                end=float(end) if end is not None else None,
                confidence=float(conf) if conf is not None else None,
                index=idx,
            )
        )
    return built


def _score_candidate(
    line_tokens: list[str],
    recognized: list[RecognizedWord],
    start_idx: int,
    window_size: int,
    *,
    expected_time: float | None,
    time_span: float,
    time_prior_weight: float,
    cursor_index: int,
    cursor_span_words: int,
    cursor_prior_weight: float,
) -> tuple[float, list[int], int]:
    if not line_tokens:
        return 0.0, [], 0
    end_idx = min(len(recognized), start_idx + window_size)
    rec_tokens = [w.normalized for w in recognized[start_idx:end_idx]]

    matches: list[int] = []
    cursor = 0
    for tok in line_tokens:
        try:
            rel = rec_tokens.index(tok, cursor)
        except ValueError:
            continue
        abs_idx = start_idx + rel
        matches.append(abs_idx)
        cursor = rel + 1

    coverage = len(matches) / max(1, len(line_tokens))
    if not matches:
        return 0.0, [], 0

    first_rel = max(0, matches[0] - start_idx)
    compactness = 1.0 - (first_rel / max(1, window_size))
    score = 0.75 * coverage + 0.25 * compactness

    if expected_time is not None and 0 <= matches[0] < len(recognized):
        first_word_time = recognized[matches[0]].start
        if first_word_time is not None and time_span > 0:
            dist = abs(float(first_word_time) - expected_time)
            time_score = max(0.0, 1.0 - (dist / time_span))
            score = (1.0 - time_prior_weight) * score + time_prior_weight * time_score

    # Prior к ближайшему окну после текущего курсора, чтобы не прыгать на дальние повторяющиеся припевы.
    distance_words = max(0, start_idx - cursor_index)
    if cursor_span_words > 0:
        cursor_score = max(0.0, 1.0 - (distance_words / float(cursor_span_words)))
        score = (1.0 - cursor_prior_weight) * score + cursor_prior_weight * cursor_score

    return score, matches, first_rel


def _find_segment_fallback_start(segments: list[SegmentInfo], prev_start: float, min_gap_s: float) -> float | None:
    target = prev_start + min_gap_s
    for seg in segments:
        if seg.start is not None and seg.start >= target:
            return float(seg.start)
    for seg in segments:
        if seg.start is not None:
            return float(seg.start)
    return None


def _estimate_expected_time(
    line_idx: int,
    line_count: int,
    first_word_time: float,
    last_word_time: float,
) -> float:
    if line_count <= 1:
        return first_word_time
    frac = line_idx / max(1, line_count - 1)
    return first_word_time + (last_word_time - first_word_time) * frac


def align_lyric_lines(
    lyric_lines: list[str],
    recognized_words: list[RecognizedWord],
    *,
    segments: list[SegmentInfo] | None = None,
    config: LineAlignmentConfig | None = None,
) -> list[LineTimingResult]:
    cfg = config or LineAlignmentConfig()
    started = time.perf_counter()

    pre_roll_s = max(0.0, min(float(cfg.pre_roll_ms), 300.0)) / 1000.0
    min_gap_s = max(0.0, float(cfg.min_line_gap_ms)) / 1000.0
    max_jump_s = max(min_gap_s * 2.0, float(cfg.max_line_jump_ms) / 1000.0)
    segments = segments or []

    logger.info(
        "Line alignment: lines=%d words=%d pre_roll_ms=%d min_gap_ms=%d max_jump_ms=%d",
        len(lyric_lines),
        len(recognized_words),
        cfg.pre_roll_ms,
        cfg.min_line_gap_ms,
        cfg.max_line_jump_ms,
    )

    if not recognized_words:
        return [
            LineTimingResult(
                text=text,
                start_time_seconds=(idx * min_gap_s),
                confidence=0.0,
                status="fallback_no_words",
                details={"line_idx": idx},
            )
            for idx, text in enumerate(lyric_lines)
        ]

    valid_times = [w.start for w in recognized_words if w.start is not None]
    first_word_time = float(valid_times[0]) if valid_times else 0.0
    last_word_time = float(valid_times[-1]) if valid_times else first_word_time + 60.0
    timeline_span = max(5.0, last_word_time - first_word_time)

    results: list[LineTimingResult] = []
    word_cursor = 0
    fallback_count = 0
    non_first_anchor_count = 0
    far_jump_corrections = 0

    for line_idx, line_text in enumerate(lyric_lines):
        tokens = tokenize_for_matching(line_text, russian_mode=cfg.russian_mode)
        if not tokens:
            fallback_count += 1
            base = 0.0 if not results else (results[-1].start_time_seconds + min_gap_s)
            results.append(
                LineTimingResult(
                    text=line_text,
                    start_time_seconds=base,
                    confidence=0.0,
                    status="fallback_empty_line",
                    details={"line_idx": line_idx},
                )
            )
            continue

        expected_time = _estimate_expected_time(line_idx, len(lyric_lines), first_word_time, last_word_time)
        max_window = max(len(tokens) + cfg.max_window_extra_words, len(tokens) * 3)

        best_score = -1.0
        best_matches: list[int] = []
        best_start_idx = -1

        local_lookahead = max(20, cfg.max_candidate_lookahead_words)
        local_end = min(len(recognized_words), word_cursor + local_lookahead)
        for ridx in range(word_cursor, local_end):
            score, matches, _ = _score_candidate(
                tokens,
                recognized_words,
                ridx,
                max_window,
                expected_time=expected_time,
                time_span=timeline_span,
                time_prior_weight=cfg.time_prior_weight,
                cursor_index=word_cursor,
                cursor_span_words=local_lookahead,
                cursor_prior_weight=cfg.cursor_prior_weight,
            )
            if score > best_score:
                best_score = score
                best_matches = matches
                best_start_idx = ridx
            if best_score >= 0.94:
                break

        # Если локальный поиск слабый — расширяем область ограниченно,
        # а не до конца трека, чтобы не улетать на дальние повторы припева.
        if best_score < cfg.min_local_match_score:
            remote_end = min(len(recognized_words), local_end + local_lookahead)
            for ridx in range(local_end, remote_end):
                score, matches, _ = _score_candidate(
                    tokens,
                    recognized_words,
                    ridx,
                    max_window,
                    expected_time=expected_time,
                    time_span=timeline_span,
                    time_prior_weight=min(0.85, cfg.time_prior_weight + 0.25),
                    cursor_index=word_cursor,
                    cursor_span_words=local_lookahead * 2,
                    cursor_prior_weight=min(0.45, cfg.cursor_prior_weight + 0.10),
                )
                if score > best_score:
                    best_score = score
                    best_matches = matches
                    best_start_idx = ridx

        anchor_idx = -1
        anchor_word = ""
        raw_start = None
        confidence = max(0.0, best_score)
        status = "matched"

        if best_matches:
            for pos, matched_idx in enumerate(best_matches[: max(1, cfg.max_anchor_search_words)]):
                rec_word = recognized_words[matched_idx]
                token = rec_word.normalized
                short = len(token) < cfg.min_anchor_word_length
                stop = cfg.prefer_non_stopword_anchor and _is_stopword(token, russian_mode=cfg.russian_mode)
                weak_conf = rec_word.confidence is not None and rec_word.confidence < cfg.low_confidence_threshold
                suspicious = short or stop or weak_conf
                if suspicious and pos + 1 < len(best_matches):
                    continue
                anchor_idx = matched_idx
                anchor_word = rec_word.raw
                raw_start = rec_word.start
                if pos > 0:
                    non_first_anchor_count += 1
                break

        if raw_start is None:
            fallback_count += 1
            status = "fallback"
            confidence = min(confidence, 0.35)
            if best_matches:
                candidate = recognized_words[best_matches[0]].start
                if candidate is not None:
                    raw_start = candidate
                    status = "fallback_partial_word"
            if raw_start is None and cfg.allow_segment_fallback:
                seg_start = _find_segment_fallback_start(
                    segments,
                    prev_start=results[-1].start_time_seconds if results else -min_gap_s,
                    min_gap_s=min_gap_s,
                )
                if seg_start is not None:
                    raw_start = seg_start
                    status = "fallback_segment"
            if raw_start is None:
                raw_start = 0.0 if not results else (results[-1].start_time_seconds + min_gap_s)
                status = "fallback_gap"

        adjusted = max(0.0, float(raw_start) - pre_roll_s)

        if results:
            prev = results[-1].start_time_seconds
            if adjusted - prev > max_jump_s:
                far_jump_corrections += 1
                # Далёкий скачок считаем подозрительным и притягиваем к ожидаемой точке,
                # но не нарушая монотонность и минимум зазора.
                expected_from_prev = prev + max(min_gap_s, timeline_span / max(6.0, len(lyric_lines) * 0.65))
                adjusted = max(prev + min_gap_s, min(adjusted, expected_from_prev + max_jump_s * 0.35))
                status = f"{status}_far_jump_limited"

            if adjusted < prev + min_gap_s:
                adjusted = prev + min_gap_s
                status = f"{status}_gap_adjusted"

        if anchor_idx >= 0:
            word_cursor = max(word_cursor, anchor_idx + 1)

        result = LineTimingResult(
            text=line_text,
            start_time_seconds=adjusted,
            confidence=confidence,
            status=status,
            anchor_word=anchor_word,
            anchor_word_index=anchor_idx,
            raw_start_seconds=raw_start,
            details={
                "line_idx": line_idx,
                "tokens": len(tokens),
                "best_score": best_score,
                "matches": len(best_matches),
                "candidate_start_idx": best_start_idx,
                "expected_time": expected_time,
            },
        )
        logger.debug(
            "Line alignment debug: idx=%d status=%s anchor=%s raw=%.3f adjusted=%.3f score=%.3f",
            line_idx,
            status,
            anchor_word or "-",
            float(raw_start),
            adjusted,
            confidence,
        )
        results.append(result)

    nearly_equal_before_fix = 0
    for i in range(1, len(results)):
        if abs(results[i].start_time_seconds - results[i - 1].start_time_seconds) < 0.01:
            nearly_equal_before_fix += 1

    confident_count = sum(1 for r in results if r.status.startswith("matched") and r.confidence >= cfg.low_confidence_threshold)
    elapsed = time.perf_counter() - started
    logger.info(
        "Line alignment: done in %.3fs, confident=%d fallback=%d non_first_anchor=%d equal_starts=%d far_jump_limited=%d",
        elapsed,
        confident_count,
        fallback_count,
        non_first_anchor_count,
        nearly_equal_before_fix,
        far_jump_corrections,
    )

    return results
