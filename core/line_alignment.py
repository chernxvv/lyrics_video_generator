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
    min_cursor_advance_confidence: float = 0.58
    low_info_line_max_tokens: int = 2
    use_global_alignment: bool = True
    lookahead_lines: int = 2
    lookahead_weight: float = 0.22
    max_context_jump_ms: int = 4500
    global_match_score: float = 1.8
    global_mismatch_penalty: float = 0.9
    global_gap_penalty: float = 0.7
    global_time_bonus: float = 0.45
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


@dataclass(slots=True)
class _FlatTextToken:
    token: str
    line_idx: int
    token_idx_in_line: int


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


def _find_segment_fallback_start(segments: list[SegmentInfo], prev_start: float, min_gap_s: float) -> float | None:
    target = prev_start + min_gap_s
    for seg in segments:
        if seg.start is not None and seg.start >= target:
            return float(seg.start)
    for seg in segments:
        if seg.start is not None:
            return float(seg.start)
    return None


def _estimate_expected_time(line_idx: int, line_count: int, first_word_time: float, last_word_time: float) -> float:
    if line_count <= 1:
        return first_word_time
    frac = line_idx / max(1, line_count - 1)
    return first_word_time + (last_word_time - first_word_time) * frac


def _is_low_information_line(tokens: list[str], cfg: LineAlignmentConfig) -> bool:
    if not tokens:
        return True
    if len(tokens) <= cfg.low_info_line_max_tokens:
        if len(tokens) == 1 and len(tokens[0]) >= 4 and not tokens[0].isdigit() and not _is_stopword(tokens[0], russian_mode=cfg.russian_mode):
            return False
        return True
    if len(set(tokens)) <= 1:
        return True
    short_ratio = sum(1 for tok in tokens if len(tok) <= 2) / max(1, len(tokens))
    digit_ratio = sum(1 for tok in tokens if tok.isdigit()) / max(1, len(tokens))
    return short_ratio >= 0.75 or digit_ratio >= 0.5


def _build_flat_text_tokens(lyric_lines: list[str], line_tokens: list[list[str]], low_info_mask: list[bool]) -> list[_FlatTextToken]:
    flat: list[_FlatTextToken] = []
    for li, _line in enumerate(lyric_lines):
        if low_info_mask[li]:
            continue
        for ti, tok in enumerate(line_tokens[li]):
            flat.append(_FlatTextToken(token=tok, line_idx=li, token_idx_in_line=ti))
    return flat


def _global_align_tokens(
    flat_tokens: list[_FlatTextToken],
    recognized_words: list[RecognizedWord],
    *,
    cfg: LineAlignmentConfig,
    first_word_time: float,
    last_word_time: float,
    total_lines: int,
) -> dict[int, list[int]]:
    n = len(flat_tokens)
    m = len(recognized_words)
    if n == 0 or m == 0:
        return {}

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    ptr = [[0] * (m + 1) for _ in range(n + 1)]  # 1=diag 2=up 3=left

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] - cfg.global_gap_penalty
        ptr[i][0] = 2
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] - cfg.global_gap_penalty
        ptr[0][j] = 3

    time_span = max(5.0, last_word_time - first_word_time)

    for i in range(1, n + 1):
        text_tok = flat_tokens[i - 1]
        expected_time = _estimate_expected_time(text_tok.line_idx, total_lines, first_word_time, last_word_time)
        for j in range(1, m + 1):
            rec = recognized_words[j - 1]
            diag_score = dp[i - 1][j - 1]
            if text_tok.token == rec.normalized:
                diag_score += cfg.global_match_score
            else:
                diag_score -= cfg.global_mismatch_penalty

            if rec.start is not None:
                dist = abs(float(rec.start) - expected_time)
                diag_score += cfg.global_time_bonus * max(0.0, 1.0 - (dist / time_span))

            up_score = dp[i - 1][j] - cfg.global_gap_penalty
            left_score = dp[i][j - 1] - cfg.global_gap_penalty

            if diag_score >= up_score and diag_score >= left_score:
                dp[i][j] = diag_score
                ptr[i][j] = 1
            elif up_score >= left_score:
                dp[i][j] = up_score
                ptr[i][j] = 2
            else:
                dp[i][j] = left_score
                ptr[i][j] = 3

    line_matches: dict[int, list[int]] = {}
    i, j = n, m
    while i > 0 and j > 0:
        p = ptr[i][j]
        if p == 1:
            ft = flat_tokens[i - 1]
            rw = recognized_words[j - 1]
            if ft.token == rw.normalized:
                line_matches.setdefault(ft.line_idx, []).append(j - 1)
            i -= 1
            j -= 1
        elif p == 2:
            i -= 1
        else:
            j -= 1

    for line_idx, values in list(line_matches.items()):
        line_matches[line_idx] = sorted(set(values))

    return line_matches


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

    distance_words = max(0, start_idx - cursor_index)
    if cursor_span_words > 0:
        cursor_score = max(0.0, 1.0 - (distance_words / float(cursor_span_words)))
        score = (1.0 - cursor_prior_weight) * score + cursor_prior_weight * cursor_score

    return score, matches, first_rel


def _context_score_candidate(
    line_idx: int,
    start_idx: int,
    line_tokens_all: list[list[str]],
    recognized_words: list[RecognizedWord],
    cfg: LineAlignmentConfig,
    expected_time: float,
    time_span: float,
    local_lookahead: int,
) -> float:
    base, base_matches, _ = _score_candidate(
        line_tokens_all[line_idx],
        recognized_words,
        start_idx,
        max(len(line_tokens_all[line_idx]) + cfg.max_window_extra_words, len(line_tokens_all[line_idx]) * 3),
        expected_time=expected_time,
        time_span=time_span,
        time_prior_weight=cfg.time_prior_weight,
        cursor_index=start_idx,
        cursor_span_words=local_lookahead,
        cursor_prior_weight=cfg.cursor_prior_weight,
    )
    if not base_matches:
        return base

    score = base
    anchor_pos = base_matches[0]
    for step in range(1, cfg.lookahead_lines + 1):
        li = line_idx + step
        if li >= len(line_tokens_all):
            break
        next_tokens = line_tokens_all[li]
        if not next_tokens:
            continue
        next_expected = _estimate_expected_time(li, len(line_tokens_all), expected_time - (line_idx * 0.01), expected_time + time_span)
        next_s, next_matches, _ = _score_candidate(
            next_tokens,
            recognized_words,
            min(len(recognized_words) - 1, anchor_pos + 1),
            max(len(next_tokens) + cfg.max_window_extra_words, len(next_tokens) * 3),
            expected_time=next_expected,
            time_span=time_span,
            time_prior_weight=min(0.85, cfg.time_prior_weight + 0.2),
            cursor_index=anchor_pos,
            cursor_span_words=local_lookahead,
            cursor_prior_weight=min(0.5, cfg.cursor_prior_weight + 0.1),
        )
        if not next_matches:
            score -= cfg.lookahead_weight * 0.9
            continue
        jump_ms = 0.0
        if recognized_words[next_matches[0]].start is not None and recognized_words[anchor_pos].start is not None:
            jump_ms = max(0.0, (recognized_words[next_matches[0]].start - recognized_words[anchor_pos].start) * 1000.0)
        if jump_ms > cfg.max_context_jump_ms:
            score -= cfg.lookahead_weight * 1.4
        score += cfg.lookahead_weight * next_s / float(step)

    return score


def _resolve_timing_conflicts(
    results: list[LineTimingResult],
    strong_mask: list[bool],
    *,
    min_gap_s: float,
) -> tuple[int, list[LineTimingResult]]:
    relocated = 0
    if not results:
        return relocated, results

    strong_indices = [i for i, is_strong in enumerate(strong_mask) if is_strong]
    if len(strong_indices) < 2:
        return relocated, results

    for idx, res in enumerate(results):
        if strong_mask[idx]:
            continue

        prev_strong = max((si for si in strong_indices if si < idx), default=None)
        next_strong = min((si for si in strong_indices if si > idx), default=None)
        if prev_strong is None and next_strong is None:
            continue

        left_bound = (results[prev_strong].start_time_seconds + min_gap_s) if prev_strong is not None else 0.0
        right_bound = (
            max(left_bound, results[next_strong].start_time_seconds - min_gap_s)
            if next_strong is not None
            else left_bound + min_gap_s
        )

        desired = res.start_time_seconds
        if desired < left_bound or (next_strong is not None and desired >= results[next_strong].start_time_seconds):
            relocated += 1
            relocated_time = min(right_bound, max(left_bound, desired))
            if idx > 0:
                relocated_time = max(relocated_time, results[idx - 1].start_time_seconds + min_gap_s)
            res.start_time_seconds = relocated_time
            res.status = f"{res.status}_conflict_relocated"

    return relocated, results


def _align_lyric_lines_greedy(
    lyric_lines: list[str],
    recognized_words: list[RecognizedWord],
    *,
    segments: list[SegmentInfo],
    cfg: LineAlignmentConfig,
) -> tuple[list[LineTimingResult], list[bool], dict[str, int]]:
    pre_roll_s = max(0.0, min(float(cfg.pre_roll_ms), 300.0)) / 1000.0
    min_gap_s = max(0.0, float(cfg.min_line_gap_ms)) / 1000.0
    max_jump_s = max(min_gap_s * 2.0, float(cfg.max_line_jump_ms) / 1000.0)

    valid_times = [w.start for w in recognized_words if w.start is not None]
    first_word_time = float(valid_times[0]) if valid_times else 0.0
    last_word_time = float(valid_times[-1]) if valid_times else first_word_time + 60.0
    timeline_span = max(5.0, last_word_time - first_word_time)

    line_tokens_all = [tokenize_for_matching(line, russian_mode=cfg.russian_mode) for line in lyric_lines]
    low_info_mask = [_is_low_information_line(tokens, cfg) for tokens in line_tokens_all]
    strong_mask = [not flag for flag in low_info_mask]

    results: list[LineTimingResult] = []
    word_cursor = 0
    fallback_count = 0
    non_first_anchor_count = 0
    far_jump_corrections = 0
    context_override_count = 0

    for line_idx, line_text in enumerate(lyric_lines):
        tokens = line_tokens_all[line_idx]
        if not tokens:
            fallback_count += 1
            base = 0.0 if not results else (results[-1].start_time_seconds + min_gap_s)
            results.append(LineTimingResult(text=line_text, start_time_seconds=base, confidence=0.0, status="fallback_empty_line"))
            continue

        is_low_info_line = low_info_mask[line_idx]
        expected_time = _estimate_expected_time(line_idx, len(lyric_lines), first_word_time, last_word_time)
        max_window = max(len(tokens) + cfg.max_window_extra_words, len(tokens) * 3)
        local_lookahead = max(20, cfg.max_candidate_lookahead_words)
        local_end = min(len(recognized_words), word_cursor + local_lookahead)

        candidates: list[tuple[float, int, list[int]]] = []
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
            if matches:
                candidates.append((score, ridx, matches))

        candidates.sort(key=lambda item: item[0], reverse=True)
        top_candidates = candidates[:4] if candidates else []

        if not top_candidates:
            best_score = -1.0
            best_start_idx = -1
            best_matches: list[int] = []
        else:
            local_best_score, local_best_start_idx, local_best_matches = top_candidates[0]
            selected = (local_best_score, local_best_start_idx, local_best_matches)
            best_context = _context_score_candidate(
                line_idx,
                local_best_start_idx,
                line_tokens_all,
                recognized_words,
                cfg,
                expected_time,
                timeline_span,
                local_lookahead,
            )
            for score, ridx, matches in top_candidates[1:]:
                alt_context = _context_score_candidate(
                    line_idx,
                    ridx,
                    line_tokens_all,
                    recognized_words,
                    cfg,
                    expected_time,
                    timeline_span,
                    local_lookahead,
                )
                if alt_context > best_context + 0.08:
                    selected = (score, ridx, matches)
                    best_context = alt_context
                    context_override_count += 1
            best_score, best_start_idx, best_matches = selected

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
            if best_matches and not (is_low_info_line and confidence < cfg.min_local_match_score):
                candidate = recognized_words[best_matches[0]].start
                if candidate is not None:
                    raw_start = candidate
                    status = "fallback_partial_word"
            if raw_start is None and is_low_info_line and results:
                raw_start = results[-1].start_time_seconds + min_gap_s
                status = "interpolated_low_info"
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
                adjusted = max(prev + min_gap_s, prev + max_jump_s * 0.8)
                status = f"{status}_far_jump_limited"
            if adjusted < prev + min_gap_s:
                adjusted = prev + min_gap_s
                status = f"{status}_gap_adjusted"

        should_advance_cursor = (
            anchor_idx >= 0
            and confidence >= cfg.min_cursor_advance_confidence
            and not status.startswith("fallback")
            and not is_low_info_line
        )
        if should_advance_cursor:
            word_cursor = max(word_cursor, anchor_idx + 1)

        results.append(
            LineTimingResult(
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
                    "low_info_line": int(is_low_info_line),
                },
            )
        )

    stats = {
        "fallback_count": fallback_count,
        "non_first_anchor_count": non_first_anchor_count,
        "far_jump_corrections": far_jump_corrections,
        "context_override_count": context_override_count,
    }
    return results, strong_mask, stats


def _align_lyric_lines_global(
    lyric_lines: list[str],
    recognized_words: list[RecognizedWord],
    *,
    segments: list[SegmentInfo],
    cfg: LineAlignmentConfig,
) -> tuple[list[LineTimingResult], list[bool], dict[str, int]]:
    pre_roll_s = max(0.0, min(float(cfg.pre_roll_ms), 300.0)) / 1000.0
    min_gap_s = max(0.0, float(cfg.min_line_gap_ms)) / 1000.0

    line_tokens = [tokenize_for_matching(line, russian_mode=cfg.russian_mode) for line in lyric_lines]
    low_info_mask = [_is_low_information_line(tokens, cfg) for tokens in line_tokens]
    strong_mask = [not low for low in low_info_mask]

    valid_times = [w.start for w in recognized_words if w.start is not None]
    first_word_time = float(valid_times[0]) if valid_times else 0.0
    last_word_time = float(valid_times[-1]) if valid_times else first_word_time + 60.0

    flat_tokens = _build_flat_text_tokens(lyric_lines, line_tokens, low_info_mask)
    match_map = _global_align_tokens(
        flat_tokens,
        recognized_words,
        cfg=cfg,
        first_word_time=first_word_time,
        last_word_time=last_word_time,
        total_lines=len(lyric_lines),
    )

    raw_starts: list[float | None] = [None] * len(lyric_lines)
    confidences: list[float] = [0.0] * len(lyric_lines)
    statuses: list[str] = ["unresolved"] * len(lyric_lines)
    anchor_words: list[str] = ["" for _ in lyric_lines]
    anchor_indices: list[int] = [-1 for _ in lyric_lines]

    anchored_strong_lines = 0
    interpolated_low_info_lines = 0

    for i in range(len(lyric_lines)):
        if not strong_mask[i]:
            continue
        matched = match_map.get(i) or []
        if matched:
            idx = matched[0]
            start = recognized_words[idx].start
            if start is not None:
                raw_starts[i] = float(start)
                confidences[i] = min(1.0, len(matched) / max(1.0, len(line_tokens[i])))
                statuses[i] = "matched_global"
                anchor_words[i] = recognized_words[idx].raw
                anchor_indices[i] = idx
                anchored_strong_lines += 1
                continue

        statuses[i] = "fallback_global"

    # Fill strong unresolved with segment/gap fallback
    for i in range(len(lyric_lines)):
        if not strong_mask[i] or raw_starts[i] is not None:
            continue
        prev = max((raw_starts[k] for k in range(i - 1, -1, -1) if raw_starts[k] is not None), default=None)
        if cfg.allow_segment_fallback:
            seg_start = _find_segment_fallback_start(segments, prev_start=(prev or -min_gap_s), min_gap_s=min_gap_s)
            if seg_start is not None:
                raw_starts[i] = seg_start
                statuses[i] = "fallback_segment"
                continue
        raw_starts[i] = (prev + min_gap_s) if prev is not None else 0.0
        statuses[i] = "fallback_gap"

    # Phase B: interpolate low-information lines between strong anchors
    strong_indices = [idx for idx, is_strong in enumerate(strong_mask) if is_strong]
    for i in range(len(lyric_lines)):
        if strong_mask[i]:
            continue
        prev_strong = max((si for si in strong_indices if si < i and raw_starts[si] is not None), default=None)
        next_strong = min((si for si in strong_indices if si > i and raw_starts[si] is not None), default=None)

        if prev_strong is not None and next_strong is not None:
            between = [k for k in range(prev_strong + 1, next_strong) if not strong_mask[k]]
            pos = between.index(i)
            step = (raw_starts[next_strong] - raw_starts[prev_strong]) / max(1, len(between) + 1)
            raw_starts[i] = raw_starts[prev_strong] + step * (pos + 1)
            statuses[i] = "interpolated_low_info"
            confidences[i] = 0.25
            interpolated_low_info_lines += 1
        elif prev_strong is not None:
            raw_starts[i] = raw_starts[prev_strong] + min_gap_s
            statuses[i] = "interpolated_low_info_tail"
            confidences[i] = 0.2
            interpolated_low_info_lines += 1
        else:
            next_known = min((raw_starts[k] for k in range(i + 1, len(lyric_lines)) if raw_starts[k] is not None), default=None)
            if next_known is not None:
                raw_starts[i] = max(0.0, next_known - min_gap_s)
            else:
                raw_starts[i] = 0.0 if i == 0 else raw_starts[i - 1] + min_gap_s
            statuses[i] = "interpolated_low_info_head"
            confidences[i] = 0.2
            interpolated_low_info_lines += 1

    # Build ordered/adjusted results
    results: list[LineTimingResult] = []
    for i, text in enumerate(lyric_lines):
        raw_start = max(0.0, float(raw_starts[i] or 0.0))
        adjusted = max(0.0, raw_start - pre_roll_s)
        if results:
            adjusted = max(adjusted, results[-1].start_time_seconds + min_gap_s)

        results.append(
            LineTimingResult(
                text=text,
                start_time_seconds=adjusted,
                confidence=confidences[i],
                status=statuses[i],
                anchor_word=anchor_words[i],
                anchor_word_index=anchor_indices[i],
                raw_start_seconds=raw_start,
                details={
                    "line_idx": i,
                    "tokens": len(line_tokens[i]),
                    "low_info_line": int(low_info_mask[i]),
                },
            )
        )

    stats = {
        "anchored_strong_lines": anchored_strong_lines,
        "interpolated_low_info_lines": interpolated_low_info_lines,
    }
    return results, strong_mask, stats


def align_lyric_lines(
    lyric_lines: list[str],
    recognized_words: list[RecognizedWord],
    *,
    segments: list[SegmentInfo] | None = None,
    config: LineAlignmentConfig | None = None,
) -> list[LineTimingResult]:
    cfg = config or LineAlignmentConfig()
    started = time.perf_counter()
    segments = segments or []

    logger.info(
        "Line alignment: lines=%d words=%d pre_roll_ms=%d min_gap_ms=%d max_jump_ms=%d use_global=%s",
        len(lyric_lines),
        len(recognized_words),
        cfg.pre_roll_ms,
        cfg.min_line_gap_ms,
        cfg.max_line_jump_ms,
        cfg.use_global_alignment,
    )

    if not recognized_words:
        min_gap_s = max(0.0, float(cfg.min_line_gap_ms)) / 1000.0
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

    try:
        if cfg.use_global_alignment:
            results, strong_mask, stats = _align_lyric_lines_global(
                lyric_lines,
                recognized_words,
                segments=segments,
                cfg=cfg,
            )
            mode = "global"
        else:
            results, strong_mask, stats = _align_lyric_lines_greedy(
                lyric_lines,
                recognized_words,
                segments=segments,
                cfg=cfg,
            )
            mode = "greedy"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Line alignment: %s path failed (%s), fallback to greedy", "global" if cfg.use_global_alignment else "selected", exc)
        results, strong_mask, stats = _align_lyric_lines_greedy(
            lyric_lines,
            recognized_words,
            segments=segments,
            cfg=cfg,
        )
        mode = "greedy_fallback"

    min_gap_s = max(0.0, float(cfg.min_line_gap_ms)) / 1000.0
    relocated_low_info_conflicts, results = _resolve_timing_conflicts(results, strong_mask, min_gap_s=min_gap_s)

    nearly_equal = 0
    for i in range(1, len(results)):
        if abs(results[i].start_time_seconds - results[i - 1].start_time_seconds) < 0.01:
            nearly_equal += 1

    fallback_count = sum(1 for r in results if r.status.startswith("fallback"))
    confident_count = sum(1 for r in results if r.status.startswith("matched") and r.confidence >= cfg.low_confidence_threshold)
    elapsed = time.perf_counter() - started
    logger.info(
        "Line alignment: done mode=%s in %.3fs, confident=%d fallback=%d equal_starts=%d anchored_strong_lines=%d interpolated_low_info_lines=%d relocated_low_info_conflicts=%d",
        mode,
        elapsed,
        confident_count,
        fallback_count,
        nearly_equal,
        stats.get("anchored_strong_lines", 0),
        stats.get("interpolated_low_info_lines", 0),
        relocated_low_info_conflicts,
    )

    if stats.get("context_override_count", 0) > 0:
        logger.info("Line alignment: context overrides=%d", stats["context_override_count"])

    return results
