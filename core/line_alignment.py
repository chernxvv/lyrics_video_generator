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
    global_line_jump_penalty: float = 1.2
    global_skipped_line_penalty: float = 1.6
    global_expected_time_penalty: float = 1.35
    global_expected_index_penalty: float = 0.045
    global_soft_line_time_factor: float = 1.8
    global_soft_line_word_factor: float = 2.2
    global_hard_line_time_factor: float = 3.6
    global_hard_line_word_factor: float = 4.5
    global_constraint_penalty: float = 7.5
    global_min_adjacent_gap_ratio: float = 0.18
    global_min_adjacent_gap_floor_s: float = 0.35
    global_small_gap_strict_token_threshold: int = 3
    segment_expected_match_bonus: float = 0.32
    segment_jump_penalty: float = 0.58
    segment_far_penalty: float = 1.35
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


@dataclass(slots=True)
class _TokenMatch:
    token_idx_in_line: int
    recognized_idx: int




@dataclass(slots=True)
class _LocalBlockCandidate:
    start_idx: int
    matches: list[_TokenMatch]
    score: float
    raw_start_seconds: float | None
    anchor_idx: int
    anchor_word: str
    confidence: float


@dataclass(slots=True)
class _SearchWindow:
    level: str
    start_idx: int
    end_idx: int
    span_seconds: float
    material_state: str


@dataclass(slots=True)
class _SegmentPrior:
    expected_start_idx: int
    expected_end_idx: int
    matched_segment_idx: int
    segment_jump_count: int
    prior_score: float


def _time_to_segment_idx(segments: list[SegmentInfo], value: float | None) -> int:
    if value is None or not segments:
        return -1
    candidate_idx = -1
    for idx, seg in enumerate(segments):
        seg_start = float(seg.start) if seg.start is not None else None
        seg_end = float(seg.end) if seg.end is not None else None
        if seg_start is not None and value < seg_start:
            return max(0, idx - 1) if idx > 0 else 0
        if seg_start is not None and seg_end is not None and seg_start <= value < seg_end:
            return idx
        if seg_start is not None and value >= seg_start:
            candidate_idx = idx
    return candidate_idx


def _estimate_segment_prior(
    *,
    segments: list[SegmentInfo],
    matched_time: float | None,
    line_idx: int,
    total_lines: int,
    prev_confirmed_line_idx: int,
    prev_confirmed_segment_idx: int,
) -> _SegmentPrior:
    matched_segment_idx = _time_to_segment_idx(segments, matched_time)
    if matched_segment_idx < 0 or not segments:
        return _SegmentPrior(-1, -1, matched_segment_idx, 0, 0.0)

    last_segment_idx = len(segments) - 1
    if prev_confirmed_segment_idx < 0:
        expected_progress = (line_idx / max(1, total_lines - 1)) * last_segment_idx
        expected_start_idx = max(0, int(expected_progress) - 1)
        expected_end_idx = min(last_segment_idx, int(expected_progress) + 1)
    else:
        line_delta = max(1, line_idx - prev_confirmed_line_idx)
        remaining_lines = max(1, total_lines - 1 - prev_confirmed_line_idx)
        remaining_segments = max(0, last_segment_idx - prev_confirmed_segment_idx)
        expected_step = remaining_segments / float(remaining_lines)
        expected_center = prev_confirmed_segment_idx + (line_delta * expected_step)
        slack = max(1, int(round(max(1.0, expected_step * line_delta * 0.75))))
        expected_start_idx = max(prev_confirmed_segment_idx, int(expected_center) - slack)
        expected_end_idx = min(last_segment_idx, int(expected_center) + slack)

    if matched_segment_idx < expected_start_idx:
        distance = expected_start_idx - matched_segment_idx
    elif matched_segment_idx > expected_end_idx:
        distance = matched_segment_idx - expected_end_idx
    else:
        distance = 0

    segment_jump_count = 0
    if prev_confirmed_segment_idx >= 0:
        line_delta = max(1, line_idx - prev_confirmed_line_idx)
        allowed_jump = max(1, line_delta)
        actual_jump = max(0, matched_segment_idx - prev_confirmed_segment_idx)
        segment_jump_count = max(0, actual_jump - allowed_jump)

    prior_score = 0.0
    if distance == 0:
        prior_score += 1.0
    else:
        prior_score -= float(distance)
    if segment_jump_count > 0:
        prior_score -= segment_jump_count * 0.75

    return _SegmentPrior(expected_start_idx, expected_end_idx, matched_segment_idx, segment_jump_count, prior_score)


def _words_per_second(recognized_words: list[RecognizedWord], default: float = 2.5) -> float:
    valid = [w.start for w in recognized_words if w.start is not None]
    if len(valid) < 2:
        return default
    span = max(1.0, float(valid[-1]) - float(valid[0]))
    return max(0.5, min(6.0, len(valid) / span))


def _find_last_reliable_anchor(results: list[LineTimingResult], cfg: LineAlignmentConfig) -> tuple[int, float | None]:
    for result in reversed(results):
        if result.anchor_word_index < 0:
            continue
        if result.confidence < cfg.min_cursor_advance_confidence:
            continue
        if result.status.startswith("fallback"):
            continue
        return result.anchor_word_index, result.raw_start_seconds
    return -1, None


def _build_candidate_windows(
    *,
    recognized_words: list[RecognizedWord],
    anchor_word_index: int,
    prev_raw_start_seconds: float | None,
    expected_line_duration_range: tuple[float, float],
    cfg: LineAlignmentConfig,
    line_token_count: int,
    cursor_index: int,
) -> list[_SearchWindow]:
    if not recognized_words:
        return []

    words_per_second = _words_per_second(recognized_words)
    local_lookahead = max(20, cfg.max_candidate_lookahead_words)
    anchor_idx = max(0, min(len(recognized_words) - 1, anchor_word_index if anchor_word_index >= 0 else cursor_index))
    base_start = max(cursor_index, anchor_idx)

    expected_min_duration, expected_max_duration = expected_line_duration_range
    expected_max_duration = max(expected_min_duration, expected_max_duration)
    expected_end_time = None if prev_raw_start_seconds is None else prev_raw_start_seconds + expected_max_duration
    expected_word_limit = max(
        line_token_count + cfg.max_window_extra_words,
        int(expected_max_duration * words_per_second) + cfg.max_window_extra_words,
    )
    if expected_end_time is not None:
        time_end_idx = len(recognized_words) - 1
        for idx in range(base_start, len(recognized_words)):
            start_time = recognized_words[idx].start
            if start_time is None:
                continue
            if float(start_time) > expected_end_time:
                time_end_idx = max(base_start, idx)
                break
    else:
        time_end_idx = min(len(recognized_words) - 1, base_start + local_lookahead)

    narrow_end = min(len(recognized_words) - 1, max(base_start, time_end_idx, base_start + expected_word_limit))
    material_state = "time_and_word_bounded" if expected_end_time is not None else "word_bounded"

    windows: list[_SearchWindow] = [
        _SearchWindow(
            level="local",
            start_idx=base_start,
            end_idx=narrow_end,
            span_seconds=max(0.0, expected_max_duration),
            material_state=material_state,
        )
    ]

    expansions = [
        ("medium", max(local_lookahead, line_token_count * 8), max(expected_max_duration * 2.0, 8.0)),
        ("wide", max(local_lookahead * 2, line_token_count * 14), max(expected_max_duration * 3.5, 16.0)),
        ("global", max(local_lookahead * 4, len(recognized_words) - base_start), max(expected_max_duration * 6.0, 32.0)),
    ]
    last_end = narrow_end
    for level, extra_words, span_seconds in expansions:
        end_idx = min(len(recognized_words) - 1, max(last_end, base_start + extra_words))
        windows.append(
            _SearchWindow(
                level=level,
                start_idx=base_start,
                end_idx=end_idx,
                span_seconds=span_seconds,
                material_state="expanded",
            )
        )
        last_end = end_idx
    return windows


def _evaluate_window_candidates(
    tokens: list[str],
    recognized_words: list[RecognizedWord],
    window: _SearchWindow,
    *,
    expected_time: float,
    timeline_span: float,
    cfg: LineAlignmentConfig,
    cursor_index: int,
    segments: list[SegmentInfo],
    line_idx: int,
    total_lines: int,
    prev_confirmed_line_idx: int,
    prev_confirmed_segment_idx: int,
) -> tuple[list[tuple[float, int, list[_TokenMatch], _SegmentPrior | None]], str]:
    max_window = max(len(tokens) + cfg.max_window_extra_words, len(tokens) * 3)
    cursor_prior_span_words = max(1, cfg.max_candidate_lookahead_words)
    candidates: list[tuple[float, int, list[_TokenMatch], _SegmentPrior | None]] = []
    matched_starts = 0
    for ridx in range(window.start_idx, min(len(recognized_words), window.end_idx + 1)):
        score, matches, _, _segment_prior = _score_candidate(
            tokens,
            recognized_words,
            ridx,
            max_window,
            expected_time=expected_time,
            time_span=timeline_span,
            time_prior_weight=cfg.time_prior_weight,
            cursor_index=cursor_index,
            cursor_span_words=cursor_prior_span_words,
            cursor_prior_weight=cfg.cursor_prior_weight,
            segment_expected_match_bonus=cfg.segment_expected_match_bonus,
            segment_jump_penalty=cfg.segment_jump_penalty,
            segments=segments,
            line_idx=line_idx,
            total_lines=total_lines,
            prev_confirmed_line_idx=prev_confirmed_line_idx,
            prev_confirmed_segment_idx=prev_confirmed_segment_idx,
        )
        if matches:
            matched_starts += 1
            candidates.append((score, ridx, matches, _segment_prior))

    if not candidates:
        state = "no_material" if matched_starts == 0 else "no_candidates"
    else:
        best_score = max(score for score, *_rest in candidates)
        state = "weak_score" if best_score < cfg.min_local_match_score else "matched"
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[:4], state


def _build_local_block_candidates(
    line_idx: int,
    line_tokens_all: list[list[str]],
    recognized_words: list[RecognizedWord],
    *,
    cfg: LineAlignmentConfig,
    first_word_time: float,
    last_word_time: float,
    cursor_start: int,
    cursor_end: int,
    segments: list[SegmentInfo],
    prev_confirmed_line_idx: int,
    prev_confirmed_segment_idx: int,
) -> list[_LocalBlockCandidate]:
    tokens = line_tokens_all[line_idx]
    if not tokens or cursor_start > cursor_end or not recognized_words:
        return []

    expected_time = _estimate_expected_time(line_idx, len(line_tokens_all), first_word_time, last_word_time)
    time_span = max(5.0, last_word_time - first_word_time)
    max_window = max(len(tokens) + cfg.max_window_extra_words, len(tokens) * 3)
    cursor_span_words = max(1, cfg.max_candidate_lookahead_words)

    built: list[_LocalBlockCandidate] = []
    for ridx in range(max(0, cursor_start), min(len(recognized_words) - 1, cursor_end) + 1):
        score, matches, _, _segment_prior = _score_candidate(
            tokens,
            recognized_words,
            ridx,
            max_window,
            expected_time=expected_time,
            time_span=time_span,
            time_prior_weight=cfg.time_prior_weight,
            cursor_index=cursor_start,
            cursor_span_words=cursor_span_words,
            cursor_prior_weight=cfg.cursor_prior_weight,
            segment_expected_match_bonus=cfg.segment_expected_match_bonus,
            segment_jump_penalty=cfg.segment_jump_penalty,
            segments=segments,
            line_idx=line_idx,
            total_lines=len(line_tokens_all),
            prev_confirmed_line_idx=prev_confirmed_line_idx,
            prev_confirmed_segment_idx=prev_confirmed_segment_idx,
        )
        if not matches:
            continue
        raw_start, anchor_idx = _estimate_line_raw_start(
            recognized_words,
            matches,
            tokens,
            russian_mode=cfg.russian_mode,
        )
        if raw_start is None or anchor_idx < 0:
            continue
        anchor_word = recognized_words[anchor_idx].raw
        confidence = min(1.0, len(matches) / max(1.0, len(tokens)))
        built.append(
            _LocalBlockCandidate(
                start_idx=ridx,
                matches=matches,
                score=score,
                raw_start_seconds=float(raw_start),
                anchor_idx=anchor_idx,
                anchor_word=anchor_word,
                confidence=confidence,
            )
        )

    built.sort(key=lambda item: (item.score, item.confidence, -item.anchor_idx), reverse=True)
    return built[:6]


def _score_block_path(
    path: list[_LocalBlockCandidate],
    line_indices: list[int],
    recognized_words: list[RecognizedWord],
    *,
    first_word_time: float,
    last_word_time: float,
) -> float:
    if not path:
        return float('-inf')
    total = 0.0
    time_span = max(5.0, last_word_time - first_word_time)
    for pos, candidate in enumerate(path):
        line_idx = line_indices[pos]
        total += candidate.score + candidate.confidence
        if candidate.raw_start_seconds is not None:
            expected_time = _estimate_expected_time(line_idx, max(line_indices) + 1, first_word_time, last_word_time)
            total -= abs(candidate.raw_start_seconds - expected_time) / time_span
        if pos == 0:
            continue
        prev = path[pos - 1]
        if candidate.anchor_idx <= prev.anchor_idx:
            return float('-inf')
        jump_words = candidate.anchor_idx - prev.anchor_idx
        jump_time = 0.0
        prev_time = recognized_words[prev.anchor_idx].start
        next_time = recognized_words[candidate.anchor_idx].start
        if prev_time is not None and next_time is not None:
            jump_time = max(0.0, float(next_time) - float(prev_time))
        line_delta = max(1, line_idx - line_indices[pos - 1])
        total -= max(0.0, jump_words - (line_delta * 8)) * 0.08
        total -= max(0.0, jump_time - (line_delta * 4.0)) * 0.12
    return total


def _refine_global_results_locally(
    lyric_lines: list[str],
    line_tokens: list[list[str]],
    strong_mask: list[bool],
    recognized_words: list[RecognizedWord],
    match_map: dict[int, list[_TokenMatch]],
    raw_starts: list[float | None],
    confidences: list[float],
    statuses: list[str],
    anchor_words: list[str],
    anchor_indices: list[int],
    line_details: list[dict[str, str | float | int]],
    *,
    cfg: LineAlignmentConfig,
    segments: list[SegmentInfo],
    first_word_time: float,
    last_word_time: float,
) -> int:
    reliable_indices = [idx for idx, status in enumerate(statuses) if status == "matched_global" and strong_mask[idx] and raw_starts[idx] is not None]
    if not reliable_indices:
        return 0

    refined_blocks = 0
    last_reliable_idx = reliable_indices[0]
    for current_idx in range(last_reliable_idx + 1, len(line_tokens)):
        if not strong_mask[current_idx]:
            continue
        prev_anchor_idx = anchor_indices[last_reliable_idx]
        matched_current = match_map.get(current_idx) or []
        estimated_current_start = raw_starts[current_idx]
        current_anchor_idx = anchor_indices[current_idx]
        if matched_current and (estimated_current_start is None or current_anchor_idx < 0):
            estimated_current_start, current_anchor_idx = _estimate_line_raw_start(
                recognized_words,
                matched_current,
                line_tokens[current_idx],
                russian_mode=cfg.russian_mode,
            )
        if prev_anchor_idx < 0 or current_anchor_idx < 0 or estimated_current_start is None:
            if statuses[current_idx] == "matched_global":
                last_reliable_idx = current_idx
            continue

        unresolved_before = sum(1 for li in range(last_reliable_idx + 1, current_idx) if statuses[li].startswith('fallback') or statuses[li] == 'unresolved')
        matched_tokens = sum(len(match_map.get(li) or []) for li in range(last_reliable_idx + 1, current_idx + 1) if strong_mask[li])
        total_tokens = sum(len(line_tokens[li]) for li in range(last_reliable_idx + 1, current_idx + 1) if strong_mask[li])
        match_density = matched_tokens / max(1, total_tokens)
        line_delta = max(1, current_idx - last_reliable_idx)
        gap_s = max(0.0, float(estimated_current_start or 0.0) - float(raw_starts[last_reliable_idx] or 0.0))
        word_jump = max(0, current_anchor_idx - prev_anchor_idx)
        density_ratio_time = line_details[current_idx].get("density_ratio_time", 0.0)
        density_ratio_words = line_details[current_idx].get("density_ratio_words", 0.0)
        local_confidence = line_details[current_idx].get("local_confidence", confidences[current_idx])
        suspicious = (
            gap_s > max(cfg.max_line_jump_ms / 1000.0, line_delta * 3.5)
            or word_jump > line_delta * 10
            or unresolved_before >= 2
            or (unresolved_before >= 1 and match_density < 0.45)
            or match_density < 0.34
            or (isinstance(density_ratio_time, (int, float)) and float(density_ratio_time) > cfg.global_soft_line_time_factor * 1.35)
            or (isinstance(density_ratio_words, (int, float)) and isinstance(local_confidence, (int, float)) and float(density_ratio_words) > cfg.global_soft_line_word_factor * 0.65 and float(local_confidence) < 0.85)
        )
        if not suspicious:
            if statuses[current_idx] == "matched_global":
                last_reliable_idx = current_idx
            continue

        block_line_indices = [li for li in range(last_reliable_idx + 1, current_idx + 1) if strong_mask[li]]
        if not block_line_indices:
            if statuses[current_idx] == "matched_global":
                last_reliable_idx = current_idx
            continue

        block_start_cursor = prev_anchor_idx + 1
        block_end_cursor = current_anchor_idx
        if block_start_cursor >= block_end_cursor:
            if statuses[current_idx] == "matched_global":
                last_reliable_idx = current_idx
            continue

        candidates_per_line: list[list[_LocalBlockCandidate]] = []
        cursor_floor = block_start_cursor
        for li in block_line_indices:
            candidates = _build_local_block_candidates(
                li,
                line_tokens,
                recognized_words,
                cfg=cfg,
                first_word_time=first_word_time,
                last_word_time=last_word_time,
                cursor_start=cursor_floor,
                cursor_end=block_end_cursor,
                segments=segments,
                prev_confirmed_line_idx=last_reliable_idx,
                prev_confirmed_segment_idx=_time_to_segment_idx(segments, raw_starts[last_reliable_idx]),
            )
            if not candidates:
                candidates_per_line = []
                break
            candidates_per_line.append(candidates)
        if not candidates_per_line:
            if statuses[current_idx] == "matched_global":
                last_reliable_idx = current_idx
            continue

        paths: list[tuple[float, list[_LocalBlockCandidate]]] = [(0.0, [])]
        for candidates in candidates_per_line:
            new_paths: list[tuple[float, list[_LocalBlockCandidate]]] = []
            for base_score, base_path in paths:
                prev_anchor = base_path[-1].anchor_idx if base_path else prev_anchor_idx
                for candidate in candidates:
                    if candidate.anchor_idx <= prev_anchor or candidate.anchor_idx > block_end_cursor:
                        continue
                    next_path = base_path + [candidate]
                    path_score = _score_block_path(
                        next_path,
                        block_line_indices[: len(next_path)],
                        recognized_words,
                        first_word_time=first_word_time,
                        last_word_time=last_word_time,
                    )
                    if path_score == float('-inf'):
                        continue
                    new_paths.append((path_score, next_path))
            new_paths.sort(key=lambda item: item[0], reverse=True)
            paths = new_paths[:12]
            if not paths:
                break
        if not paths:
            if statuses[current_idx] == "matched_global":
                last_reliable_idx = current_idx
            continue

        best_score, best_path = paths[0]
        baseline_candidates = []
        for li in block_line_indices:
            matched = match_map.get(li) or []
            if not matched or raw_starts[li] is None or anchor_indices[li] < 0:
                continue
            baseline_candidates.append(
                _LocalBlockCandidate(
                    start_idx=anchor_indices[li],
                    matches=matched,
                    score=float(line_details[li].get('local_confidence', confidences[li])) if isinstance(line_details[li].get('local_confidence', confidences[li]), (int, float)) else confidences[li],
                    raw_start_seconds=float(raw_starts[li]),
                    anchor_idx=anchor_indices[li],
                    anchor_word=anchor_words[li],
                    confidence=confidences[li],
                )
            )
        baseline_score = _score_block_path(
            baseline_candidates,
            block_line_indices[: len(baseline_candidates)],
            recognized_words,
            first_word_time=first_word_time,
            last_word_time=last_word_time,
        ) if len(baseline_candidates) == len(block_line_indices) else float('-inf')

        if best_score <= baseline_score + 0.05:
            if statuses[current_idx] == "matched_global":
                last_reliable_idx = current_idx
            continue

        for li, candidate in zip(block_line_indices, best_path):
            raw_starts[li] = candidate.raw_start_seconds
            confidences[li] = max(confidences[li], candidate.confidence)
            statuses[li] = 'matched_global_refined_block'
            anchor_words[li] = candidate.anchor_word
            anchor_indices[li] = candidate.anchor_idx
            line_details[li]['refined_from_block'] = f'{last_reliable_idx}:{current_idx}'
            line_details[li]['refined_block_score'] = round(best_score, 4)
            line_details[li]['suspicious_gap_s'] = round(gap_s, 4)
            line_details[li]['suspicious_unresolved_before'] = unresolved_before
            line_details[li]['suspicious_match_density'] = round(match_density, 4)
            refined_prior = _estimate_segment_prior(
                segments=segments,
                matched_time=candidate.raw_start_seconds,
                line_idx=li,
                total_lines=len(lyric_lines),
                prev_confirmed_line_idx=last_reliable_idx,
                prev_confirmed_segment_idx=_time_to_segment_idx(segments, raw_starts[last_reliable_idx]),
            )
            line_details[li]['matched_segment_idx'] = refined_prior.matched_segment_idx
            line_details[li]['segment_jump_count'] = refined_prior.segment_jump_count
            line_details[li]['segment_prior_score'] = round(refined_prior.prior_score, 4)
        refined_blocks += 1
        last_reliable_idx = current_idx

    return refined_blocks

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
    return None


def _is_segment_fallback_plausible(
    *,
    candidate_start: float,
    prev_start: float | None,
    line_idx: int,
    total_lines: int,
    first_word_time: float,
    last_word_time: float,
    cfg: LineAlignmentConfig,
    min_gap_s: float,
) -> bool:
    if prev_start is None:
        return True

    expected_current = _estimate_expected_time(line_idx, total_lines, first_word_time, last_word_time)
    expected_previous = _estimate_expected_time(max(0, line_idx - 1), total_lines, first_word_time, last_word_time)
    expected_gap_s = max(min_gap_s, expected_current - expected_previous)
    allowed_gap_s = max(
        cfg.max_line_jump_ms / 1000.0,
        expected_gap_s * max(1.0, cfg.global_soft_line_time_factor),
    )
    return (candidate_start - prev_start) <= allowed_gap_s


def _should_block_segment_fallback(
    rejected_reason: str,
    diagnostic_prior: _SegmentPrior | None,
    *,
    line_idx: int,
    prev_line_idx: int,
) -> bool:
    if rejected_reason in {"gap_from_prev_line_too_large", "recognized_jump_too_large", "line_density_conflict"}:
        return True
    if rejected_reason != "segment_jump_too_large":
        return False
    if diagnostic_prior is None:
        return True
    if diagnostic_prior.segment_jump_count <= max(0, line_idx - prev_line_idx):
        return False
    return diagnostic_prior.prior_score < -1.5


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


def _estimate_word_step_seconds(
    recognized_words: list[RecognizedWord],
    match_pairs: list[_TokenMatch] | None = None,
) -> float | None:
    candidate_steps: list[float] = []

    if match_pairs and len(match_pairs) >= 2:
        ordered = sorted(match_pairs, key=lambda item: item.token_idx_in_line)
        for left, right in zip(ordered, ordered[1:]):
            token_gap = right.token_idx_in_line - left.token_idx_in_line
            if token_gap <= 0:
                continue
            left_time = recognized_words[left.recognized_idx].start
            right_time = recognized_words[right.recognized_idx].start
            if left_time is None or right_time is None or right_time <= left_time:
                continue
            candidate_steps.append((float(right_time) - float(left_time)) / token_gap)

    if not candidate_steps:
        return None

    ordered_steps = sorted(candidate_steps)
    mid = len(ordered_steps) // 2
    if len(ordered_steps) % 2 == 0:
        median_step = (ordered_steps[mid - 1] + ordered_steps[mid]) / 2.0
    else:
        median_step = ordered_steps[mid]
    return min(0.8, max(0.08, median_step))


def _estimate_line_raw_start(
    recognized_words: list[RecognizedWord],
    match_pairs: list[_TokenMatch],
    line_tokens: list[str] | None = None,
    *,
    russian_mode: bool = False,
) -> tuple[float | None, int]:
    if not match_pairs:
        return None, -1

    earliest = min(match_pairs, key=lambda item: (item.token_idx_in_line, item.recognized_idx))
    anchor_time = recognized_words[earliest.recognized_idx].start
    if anchor_time is None:
        return None, earliest.recognized_idx

    missing_prefix_tokens = []
    if line_tokens is not None and earliest.token_idx_in_line > 0:
        missing_prefix_tokens = line_tokens[: earliest.token_idx_in_line]

    low_info_prefix = all(
        _is_stopword(token, russian_mode=russian_mode)
        for token in missing_prefix_tokens
    )
    allow_prefix_backdating = (
        not missing_prefix_tokens
        or (low_info_prefix and len(match_pairs) >= 3)
    )
    if not allow_prefix_backdating:
        return float(anchor_time), earliest.recognized_idx

    word_step = _estimate_word_step_seconds(recognized_words, match_pairs)
    if word_step is None:
        return float(anchor_time), earliest.recognized_idx

    estimated_start = max(0.0, float(anchor_time) - (earliest.token_idx_in_line * word_step))
    return estimated_start, earliest.recognized_idx


def _is_global_match_reliable(
    *,
    line_idx: int,
    line_count: int,
    line_tokens: list[str],
    matched: list[_TokenMatch],
    recognized_words: list[RecognizedWord],
    raw_start_seconds: float | None,
    confidence: float,
    prev_reliable_line_idx: int,
    prev_reliable_token_count: int,
    prev_reliable_raw_start_seconds: float | None,
    prev_reliable_anchor_idx: int,
    last_used_recognized_idx: int,
    first_word_time: float,
    last_word_time: float,
    cfg: LineAlignmentConfig,
    segments: list[SegmentInfo],
) -> tuple[bool, dict[str, str | float | int]]:
    details: dict[str, str | float | int] = {
        "line_idx": line_idx,
        "matches": len(matched),
        "tokens": len(line_tokens),
        "local_confidence": confidence,
        "gap_from_prev_line_s": -1.0,
        "recognized_jump_words": 0,
        "expected_gap_s": 0.0,
        "expected_jump_words": 0.0,
        "density_ratio_time": 0.0,
        "density_ratio_words": 0.0,
        "matched_segment_idx": -1,
        "segment_jump_count": 0,
        "segment_prior_score": 0.0,
    }
    if not matched or raw_start_seconds is None:
        details["rejected_reason"] = "missing_anchor"
        return False, details

    anchor_idx = min(match.recognized_idx for match in matched)
    details["anchor_word_index"] = anchor_idx

    segment_prior = _estimate_segment_prior(
        segments=segments,
        matched_time=raw_start_seconds,
        line_idx=line_idx,
        total_lines=line_count,
        prev_confirmed_line_idx=prev_reliable_line_idx,
        prev_confirmed_segment_idx=_time_to_segment_idx(segments, prev_reliable_raw_start_seconds),
    )
    details["matched_segment_idx"] = segment_prior.matched_segment_idx
    details["segment_jump_count"] = segment_prior.segment_jump_count
    details["segment_prior_score"] = round(segment_prior.prior_score, 4)
    if segment_prior.segment_jump_count > max(0, line_idx - prev_reliable_line_idx) and segment_prior.prior_score < -1.5:
        details["rejected_reason"] = "segment_jump_too_large"
        return False, details

    if prev_reliable_line_idx >= 0 and confidence < cfg.min_local_match_score and len(matched) < 2:
        details["rejected_reason"] = "low_local_similarity"
        return False, details

    if line_count <= 1:
        return True, details

    timeline_span = max(5.0, last_word_time - first_word_time)
    seconds_per_line = timeline_span / max(1, line_count - 1)
    words_per_line = max(1.0, len(recognized_words) / max(1, line_count))

    if prev_reliable_raw_start_seconds is not None and prev_reliable_line_idx >= 0 and prev_reliable_anchor_idx >= 0:
        line_delta = max(1, line_idx - prev_reliable_line_idx)
        gap_from_prev = max(0.0, float(raw_start_seconds) - float(prev_reliable_raw_start_seconds))
        recognized_jump_words = max(0, anchor_idx - last_used_recognized_idx)
        expected_gap_s = seconds_per_line * line_delta
        expected_jump_words = words_per_line * line_delta
        details["gap_from_prev_line_s"] = gap_from_prev
        details["recognized_jump_words"] = recognized_jump_words
        details["expected_gap_s"] = expected_gap_s
        details["expected_jump_words"] = expected_jump_words
        details["density_ratio_time"] = gap_from_prev / max(0.001, expected_gap_s)
        details["density_ratio_words"] = recognized_jump_words / max(1.0, expected_jump_words)

        hard_gap_limit = max(cfg.max_line_jump_ms / 1000.0, expected_gap_s * cfg.global_hard_line_time_factor)
        hard_word_limit = expected_jump_words * cfg.global_hard_line_word_factor
        soft_word_limit = max(len(line_tokens) + 1.0, expected_jump_words * cfg.global_soft_line_word_factor)
        adjacent_gap_limit = max(seconds_per_line * cfg.global_soft_line_time_factor, cfg.max_line_jump_ms / 1000.0)
        adjacent_word_limit = max(len(line_tokens) * 2.0, words_per_line * cfg.global_soft_line_word_factor)

        if line_delta == 1 and gap_from_prev > adjacent_gap_limit:
            details["rejected_reason"] = "gap_from_prev_line_too_large"
            return False, details
        strict_token_floor = max(1, int(cfg.global_small_gap_strict_token_threshold))
        if line_delta == 1 and prev_reliable_token_count >= strict_token_floor and len(line_tokens) >= strict_token_floor:
            adjacent_gap_floor = max(
                float(cfg.global_min_adjacent_gap_floor_s),
                expected_gap_s * float(cfg.global_min_adjacent_gap_ratio),
                max(0.0, float(cfg.min_line_gap_ms)) / 1000.0 * 2.0,
            )
            if gap_from_prev < adjacent_gap_floor:
                details["rejected_reason"] = "gap_from_prev_line_too_small"
                return False, details
        if line_delta == 1 and recognized_jump_words > adjacent_word_limit:
            details["rejected_reason"] = "recognized_jump_too_large"
            return False, details
        if gap_from_prev > hard_gap_limit:
            details["rejected_reason"] = "gap_from_prev_line_too_large"
            return False, details
        if recognized_jump_words > hard_word_limit:
            details["rejected_reason"] = "recognized_jump_too_large"
            return False, details
        if (
            details["density_ratio_time"] > cfg.global_soft_line_time_factor
            and details["density_ratio_words"] > cfg.global_soft_line_word_factor
        ):
            details["rejected_reason"] = "line_density_conflict"
            return False, details
        if recognized_jump_words > soft_word_limit and gap_from_prev < max(seconds_per_line, expected_gap_s * 0.75):
            details["rejected_reason"] = "recognized_jump_without_time_support"
            return False, details

    return True, details


def _global_align_tokens(
    flat_tokens: list[_FlatTextToken],
    recognized_words: list[RecognizedWord],
    *,
    cfg: LineAlignmentConfig,
    first_word_time: float,
    last_word_time: float,
    total_lines: int,
    segments: list[SegmentInfo],
) -> dict[int, list[_TokenMatch]]:
    if not flat_tokens or not recognized_words:
        return {}

    time_span = max(5.0, last_word_time - first_word_time)
    words_per_line = max(1.0, len(recognized_words) / max(1, total_lines))
    seconds_per_line = time_span / max(1, total_lines - 1)

    candidates: list[tuple[int, int, _FlatTextToken, RecognizedWord, float]] = []
    for flat_idx, text_tok in enumerate(flat_tokens):
        expected_time = _estimate_expected_time(text_tok.line_idx, total_lines, first_word_time, last_word_time)
        for rec_idx, rec in enumerate(recognized_words):
            if text_tok.token != rec.normalized:
                continue
            score = cfg.global_match_score
            if rec.start is not None:
                dist = abs(float(rec.start) - expected_time)
                score += cfg.global_time_bonus * max(0.0, 1.0 - (dist / time_span))
            segment_prior = _estimate_segment_prior(
                segments=segments,
                matched_time=rec.start,
                line_idx=text_tok.line_idx,
                total_lines=total_lines,
                prev_confirmed_line_idx=-1,
                prev_confirmed_segment_idx=-1,
            )
            if segment_prior.matched_segment_idx >= 0:
                if segment_prior.prior_score >= 0:
                    score += cfg.segment_expected_match_bonus * (1.0 + min(1.0, segment_prior.prior_score))
                else:
                    score -= cfg.segment_jump_penalty * abs(segment_prior.prior_score)
            candidates.append((flat_idx, rec_idx, text_tok, rec, score))

    if not candidates:
        return {}

    strong_line_order: dict[int, int] = {}
    for text_tok in flat_tokens:
        if text_tok.line_idx not in strong_line_order:
            strong_line_order[text_tok.line_idx] = len(strong_line_order)

    best_scores = [float('-inf')] * len(candidates)
    prev_ptr = [-1] * len(candidates)

    for idx, (flat_idx, rec_idx, text_tok, rec, base_score) in enumerate(candidates):
        best_score = base_score - (flat_idx + rec_idx) * cfg.global_gap_penalty

        current_time = float(rec.start) if rec.start is not None else _estimate_expected_time(text_tok.line_idx, total_lines, first_word_time, last_word_time)

        for prev_idx in range(idx):
            prev_flat_idx, prev_rec_idx, prev_text_tok, prev_rec, _ = candidates[prev_idx]
            if prev_flat_idx >= flat_idx or prev_rec_idx >= rec_idx:
                continue

            token_gap = max(0, (flat_idx - prev_flat_idx) - 1)
            rec_gap = max(0, (rec_idx - prev_rec_idx) - 1)
            transition_score = best_scores[prev_idx] - (token_gap + rec_gap) * cfg.global_gap_penalty

            line_delta = text_tok.line_idx - prev_text_tok.line_idx
            if line_delta < 0:
                continue
            strong_line_delta = strong_line_order[text_tok.line_idx] - strong_line_order[prev_text_tok.line_idx]
            effective_line_delta = max(0, strong_line_delta)
            skipped_lines = max(0, effective_line_delta - 1)
            if effective_line_delta > 1:
                transition_score -= cfg.global_line_jump_penalty * float(effective_line_delta - 1)
                transition_score -= cfg.global_skipped_line_penalty * float(skipped_lines * skipped_lines)

            prev_time = float(prev_rec.start) if prev_rec.start is not None else _estimate_expected_time(prev_text_tok.line_idx, total_lines, first_word_time, last_word_time)
            remaining_lines = max(1, total_lines - 1 - prev_text_tok.line_idx)
            remaining_words = max(1, len(recognized_words) - 1 - prev_rec_idx)
            expected_index_step = remaining_words / float(remaining_lines)
            expected_time_step = max(seconds_per_line, (last_word_time - prev_time) / float(remaining_lines))

            expected_rec_idx = prev_rec_idx + (effective_line_delta * expected_index_step)
            expected_time = prev_time + (effective_line_delta * expected_time_step)

            rec_idx_dist = abs(rec_idx - expected_rec_idx)
            time_dist = abs(current_time - expected_time)

            if effective_line_delta > 0:
                line_scale = float(effective_line_delta)
                transition_score -= cfg.global_expected_index_penalty * (rec_idx_dist / line_scale)
                transition_score -= cfg.global_expected_time_penalty * (time_dist / line_scale)

                current_segment_prior = _estimate_segment_prior(
                    segments=segments,
                    matched_time=current_time,
                    line_idx=text_tok.line_idx,
                    total_lines=total_lines,
                    prev_confirmed_line_idx=prev_text_tok.line_idx,
                    prev_confirmed_segment_idx=_time_to_segment_idx(segments, prev_time),
                )
                if current_segment_prior.matched_segment_idx >= 0:
                    if current_segment_prior.prior_score >= 0:
                        transition_score += cfg.segment_expected_match_bonus * (1.0 + min(1.0, current_segment_prior.prior_score))
                    else:
                        transition_score -= cfg.segment_jump_penalty * abs(current_segment_prior.prior_score)
                    if current_segment_prior.segment_jump_count > 0:
                        transition_score -= cfg.segment_far_penalty * current_segment_prior.segment_jump_count

                soft_word_limit = words_per_line * cfg.global_soft_line_word_factor * line_scale
                soft_time_limit = seconds_per_line * cfg.global_soft_line_time_factor * line_scale
                hard_word_limit = words_per_line * cfg.global_hard_line_word_factor * line_scale
                hard_time_limit = seconds_per_line * cfg.global_hard_line_time_factor * line_scale

                if rec_idx_dist > soft_word_limit:
                    transition_score -= cfg.global_constraint_penalty * (rec_idx_dist - soft_word_limit) / max(1.0, words_per_line)
                if time_dist > soft_time_limit:
                    transition_score -= cfg.global_constraint_penalty * (time_dist - soft_time_limit) / max(0.5, seconds_per_line)
                if rec_idx_dist > hard_word_limit or time_dist > hard_time_limit:
                    overflow = max(
                        0.0 if hard_word_limit <= 0 else (rec_idx_dist - hard_word_limit) / max(1.0, words_per_line),
                        0.0 if hard_time_limit <= 0 else (time_dist - hard_time_limit) / max(0.5, seconds_per_line),
                    )
                    transition_score -= cfg.global_constraint_penalty * (4.0 + overflow + skipped_lines)

            if transition_score + base_score > best_score:
                best_score = transition_score + base_score
                prev_ptr[idx] = prev_idx

        best_scores[idx] = best_score

    best_idx = max(range(len(candidates)), key=lambda candidate_idx: best_scores[candidate_idx])
    line_matches: dict[int, list[_TokenMatch]] = {}
    while best_idx >= 0:
        _flat_idx, rec_idx, text_tok, _rec, _score = candidates[best_idx]
        line_matches.setdefault(text_tok.line_idx, []).append(
            _TokenMatch(token_idx_in_line=text_tok.token_idx_in_line, recognized_idx=rec_idx)
        )
        best_idx = prev_ptr[best_idx]

    for line_idx, values in list(line_matches.items()):
        dedup: dict[tuple[int, int], _TokenMatch] = {}
        for item in values:
            dedup[(item.token_idx_in_line, item.recognized_idx)] = item
        line_matches[line_idx] = sorted(dedup.values(), key=lambda item: (item.token_idx_in_line, item.recognized_idx))

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
    segment_expected_match_bonus: float = 0.0,
    segment_jump_penalty: float = 0.0,
    segments: list[SegmentInfo] | None = None,
    line_idx: int | None = None,
    total_lines: int | None = None,
    prev_confirmed_line_idx: int = -1,
    prev_confirmed_segment_idx: int = -1,
) -> tuple[float, list[_TokenMatch], int, _SegmentPrior | None]:
    if not line_tokens:
        return 0.0, [], 0, None
    end_idx = min(len(recognized), start_idx + window_size)
    rec_tokens = [w.normalized for w in recognized[start_idx:end_idx]]

    matches: list[_TokenMatch] = []
    cursor = 0
    for token_idx, tok in enumerate(line_tokens):
        try:
            rel = rec_tokens.index(tok, cursor)
        except ValueError:
            continue
        abs_idx = start_idx + rel
        matches.append(_TokenMatch(token_idx_in_line=token_idx, recognized_idx=abs_idx))
        cursor = rel + 1

    coverage = len(matches) / max(1, len(line_tokens))
    if not matches:
        return 0.0, [], 0, None

    first_rel = max(0, matches[0].recognized_idx - start_idx)
    compactness = 1.0 - (first_rel / max(1, window_size))
    score = 0.75 * coverage + 0.25 * compactness

    if expected_time is not None and 0 <= matches[0].recognized_idx < len(recognized):
        first_word_time = recognized[matches[0].recognized_idx].start
        if first_word_time is not None and time_span > 0:
            dist = abs(float(first_word_time) - expected_time)
            time_score = max(0.0, 1.0 - (dist / time_span))
            score = (1.0 - time_prior_weight) * score + time_prior_weight * time_score

    distance_words = max(0, start_idx - cursor_index)
    if cursor_span_words > 0:
        cursor_score = max(0.0, 1.0 - (distance_words / float(cursor_span_words)))
        score = (1.0 - cursor_prior_weight) * score + cursor_prior_weight * cursor_score

    segment_prior = None
    if segments and line_idx is not None and total_lines is not None:
        matched_time = recognized[matches[0].recognized_idx].start if matches else None
        segment_prior = _estimate_segment_prior(
            segments=segments,
            matched_time=matched_time,
            line_idx=line_idx,
            total_lines=total_lines,
            prev_confirmed_line_idx=prev_confirmed_line_idx,
            prev_confirmed_segment_idx=prev_confirmed_segment_idx,
        )
        if segment_prior.matched_segment_idx >= 0:
            if segment_prior.prior_score >= 0:
                score += segment_expected_match_bonus * (1.0 + min(1.0, segment_prior.prior_score))
            else:
                score -= segment_jump_penalty * abs(segment_prior.prior_score)

    return score, matches, first_rel, segment_prior


def _context_score_candidate(
    line_idx: int,
    start_idx: int,
    line_tokens_all: list[list[str]],
    recognized_words: list[RecognizedWord],
    cfg: LineAlignmentConfig,
    expected_time: float,
    time_span: float,
    local_lookahead: int,
    segments: list[SegmentInfo],
    prev_confirmed_line_idx: int,
    prev_confirmed_segment_idx: int,
) -> float:
    base, base_matches, _, _ = _score_candidate(
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
        segment_expected_match_bonus=cfg.segment_expected_match_bonus,
        segment_jump_penalty=cfg.segment_jump_penalty,
        segments=segments,
        line_idx=line_idx,
        total_lines=len(line_tokens_all),
        prev_confirmed_line_idx=prev_confirmed_line_idx,
        prev_confirmed_segment_idx=prev_confirmed_segment_idx,
    )
    if not base_matches:
        return base

    score = base
    anchor_pos = base_matches[0].recognized_idx
    for step in range(1, cfg.lookahead_lines + 1):
        li = line_idx + step
        if li >= len(line_tokens_all):
            break
        next_tokens = line_tokens_all[li]
        if not next_tokens:
            continue
        next_expected = _estimate_expected_time(li, len(line_tokens_all), expected_time - (line_idx * 0.01), expected_time + time_span)
        next_s, next_matches, _, _ = _score_candidate(
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
            segment_expected_match_bonus=cfg.segment_expected_match_bonus,
            segment_jump_penalty=cfg.segment_jump_penalty,
            segments=segments,
            line_idx=li,
            total_lines=len(line_tokens_all),
            prev_confirmed_line_idx=line_idx,
            prev_confirmed_segment_idx=_time_to_segment_idx(segments, recognized_words[anchor_pos].start),
        )
        if not next_matches:
            score -= cfg.lookahead_weight * 0.9
            continue
        jump_ms = 0.0
        next_anchor = next_matches[0].recognized_idx
        if recognized_words[next_anchor].start is not None and recognized_words[anchor_pos].start is not None:
            jump_ms = max(0.0, (recognized_words[next_anchor].start - recognized_words[anchor_pos].start) * 1000.0)
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


def _bridge_unresolved_strong_blocks(
    *,
    raw_starts: list[float | None],
    statuses: list[str],
    strong_mask: list[bool],
    line_tokens: list[list[str]],
    min_gap_s: float,
) -> int:
    bridged = 0
    if not raw_starts:
        return bridged

    idx = 0
    total = len(raw_starts)
    while idx < total:
        if not strong_mask[idx] or not str(statuses[idx]).startswith("fallback"):
            idx += 1
            continue

        block_start = idx
        while idx < total and strong_mask[idx] and str(statuses[idx]).startswith("fallback"):
            idx += 1
        block_end = idx - 1

        prev_strong = max(
            (j for j in range(block_start - 1, -1, -1) if strong_mask[j] and raw_starts[j] is not None),
            default=None,
        )
        next_strong = min(
            (j for j in range(block_end + 1, total) if strong_mask[j] and raw_starts[j] is not None),
            default=None,
        )
        if prev_strong is None or next_strong is None:
            continue

        left = float(raw_starts[prev_strong])
        right = float(raw_starts[next_strong])
        block_len = block_end - block_start + 1
        if right - left <= min_gap_s * (block_len + 1):
            continue

        weights: list[float] = []
        for j in range(block_start, block_end + 1):
            token_count = len(line_tokens[j])
            weights.append(max(1.0, token_count / 2.0))
        total_weight = sum(weights)
        if total_weight <= 0:
            continue

        span = max(min_gap_s * (block_len + 1), right - left)
        cursor = left
        for offset, j in enumerate(range(block_start, block_end + 1)):
            share = span * (weights[offset] / total_weight)
            cursor = min(right - (min_gap_s * (block_end - j + 1)), cursor + share)
            bridged_time = max(left + min_gap_s * (offset + 1), cursor)
            raw_starts[j] = bridged_time
            statuses[j] = f"{statuses[j]}_bridged"
            bridged += 1

    return bridged


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
        cursor_index = word_cursor
        anchor_word_index, prev_raw_start_seconds = _find_last_reliable_anchor(results, cfg)
        prev_confirmed_line_idx = max((idx for idx, result in enumerate(results) if result.raw_start_seconds is not None and not result.status.startswith("fallback")), default=-1)
        prev_confirmed_segment_idx = _time_to_segment_idx(segments, prev_raw_start_seconds)
        if prev_raw_start_seconds is None and results:
            prev_raw_start_seconds = results[-1].raw_start_seconds
        expected_duration = max(
            min_gap_s,
            (timeline_span / max(1, len(lyric_lines))) * max(0.65, len(tokens) / 4.0),
        )
        expected_line_duration_range = (
            min_gap_s,
            max(min_gap_s, min(max_jump_s, expected_duration * 1.8)),
        )
        search_windows = _build_candidate_windows(
            recognized_words=recognized_words,
            anchor_word_index=anchor_word_index,
            prev_raw_start_seconds=prev_raw_start_seconds,
            expected_line_duration_range=expected_line_duration_range,
            cfg=cfg,
            line_token_count=len(tokens),
            cursor_index=cursor_index,
        )

        top_candidates: list[tuple[float, int, list[_TokenMatch], _SegmentPrior | None]] = []
        weak_top_candidates: list[tuple[float, int, list[_TokenMatch], _SegmentPrior | None]] = []
        window_level = "unsearched"
        window_state = "no_windows"
        search_span_words = 0
        search_span_seconds = 0.0
        for window in search_windows:
            window_candidates, candidate_state = _evaluate_window_candidates(
                tokens,
                recognized_words,
                window,
                expected_time=expected_time,
                timeline_span=timeline_span,
                cfg=cfg,
                cursor_index=cursor_index,
                segments=segments,
                line_idx=line_idx,
                total_lines=len(lyric_lines),
                prev_confirmed_line_idx=prev_confirmed_line_idx,
                prev_confirmed_segment_idx=prev_confirmed_segment_idx,
            )
            if candidate_state == "weak_score":
                if window_candidates and not weak_top_candidates:
                    weak_top_candidates = window_candidates
            if candidate_state == "matched":
                top_candidates = window_candidates
                window_level = window.level
                window_state = candidate_state
                search_span_words = max(1, window.end_idx - window.start_idx + 1)
                search_span_seconds = window.span_seconds
                break
            if candidate_state == "weak_score":
                window_state = candidate_state
                search_span_words = max(1, window.end_idx - window.start_idx + 1)
                search_span_seconds = window.span_seconds
                continue
            if candidate_state == "no_material":
                window_state = candidate_state
                search_span_words = max(1, window.end_idx - window.start_idx + 1)
                search_span_seconds = window.span_seconds
                continue
            window_state = candidate_state

        if not top_candidates and weak_top_candidates:
            top_candidates = weak_top_candidates

        if not top_candidates:
            best_score = -1.0
            best_start_idx = -1
            best_matches: list[_TokenMatch] = []
            best_segment_prior = None
        else:
            local_best_score, local_best_start_idx, local_best_matches, local_best_segment_prior = top_candidates[0]
            selected = (local_best_score, local_best_start_idx, local_best_matches, local_best_segment_prior)
            best_context = _context_score_candidate(
                line_idx,
                local_best_start_idx,
                line_tokens_all,
                recognized_words,
                cfg,
                expected_time,
                timeline_span,
                search_span_words,
                segments,
                prev_confirmed_line_idx,
                prev_confirmed_segment_idx,
            )
            for score, ridx, matches, segment_prior in top_candidates[1:]:
                alt_context = _context_score_candidate(
                    line_idx,
                    ridx,
                    line_tokens_all,
                    recognized_words,
                    cfg,
                    expected_time,
                    timeline_span,
                    search_span_words,
                    segments,
                    prev_confirmed_line_idx,
                    prev_confirmed_segment_idx,
                )
                if alt_context > best_context + 0.08:
                    selected = (score, ridx, matches, segment_prior)
                    best_context = alt_context
                    context_override_count += 1
            best_score, best_start_idx, best_matches, best_segment_prior = selected

        anchor_idx = -1
        anchor_word = ""
        raw_start = None
        confidence = max(0.0, best_score)
        status = "matched"

        if best_matches:
            for pos, match in enumerate(best_matches[: max(1, cfg.max_anchor_search_words)]):
                rec_word = recognized_words[match.recognized_idx]
                token = rec_word.normalized
                short = len(token) < cfg.min_anchor_word_length
                stop = cfg.prefer_non_stopword_anchor and _is_stopword(token, russian_mode=cfg.russian_mode)
                weak_conf = rec_word.confidence is not None and rec_word.confidence < cfg.low_confidence_threshold
                suspicious = short or stop or weak_conf
                if suspicious and pos + 1 < len(best_matches):
                    continue
                trusted_matches = best_matches[pos:] if pos > 0 else best_matches
                anchor_idx = match.recognized_idx
                anchor_word = rec_word.raw
                estimated_start, _ = _estimate_line_raw_start(
                    recognized_words,
                    trusted_matches,
                    tokens,
                    russian_mode=cfg.russian_mode,
                )
                raw_start = estimated_start if estimated_start is not None else rec_word.start
                if pos > 0:
                    non_first_anchor_count += 1
                break

        if raw_start is None:
            fallback_count += 1
            status = "fallback"
            confidence = min(confidence, 0.35)
            if best_matches and not (is_low_info_line and confidence < cfg.min_local_match_score):
                candidate, _ = _estimate_line_raw_start(
                    recognized_words,
                    best_matches,
                    tokens,
                    russian_mode=cfg.russian_mode,
                )
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
                    "window_level": window_level,
                    "window_state": window_state,
                    "search_span_words": search_span_words,
                    "search_span_seconds": round(search_span_seconds, 3),
                    "matched_segment_idx": -1 if best_segment_prior is None else best_segment_prior.matched_segment_idx,
                    "segment_jump_count": 0 if best_segment_prior is None else best_segment_prior.segment_jump_count,
                    "segment_prior_score": 0.0 if best_segment_prior is None else round(best_segment_prior.prior_score, 4),
                },
            )
        )
        logger.debug(
            "Line %s matched with window_level=%s span_words=%s span_seconds=%.3f state=%s score=%.3f",
            line_idx,
            window_level,
            search_span_words,
            search_span_seconds,
            window_state,
            best_score,
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
        segments=segments,
    )

    raw_starts: list[float | None] = [None] * len(lyric_lines)
    confidences: list[float] = [0.0] * len(lyric_lines)
    statuses: list[str] = ["unresolved"] * len(lyric_lines)
    anchor_words: list[str] = ["" for _ in lyric_lines]
    anchor_indices: list[int] = [-1 for _ in lyric_lines]
    line_details: list[dict[str, str | float | int]] = [
        {
            "line_idx": i,
            "tokens": len(line_tokens[i]),
            "low_info_line": int(low_info_mask[i]),
            "matched_segment_idx": -1,
            "segment_jump_count": 0,
            "segment_prior_score": 0.0,
        }
        for i in range(len(lyric_lines))
    ]

    anchored_strong_lines = 0
    interpolated_low_info_lines = 0
    global_rejection_count = 0
    prev_reliable_line_idx = -1
    prev_reliable_token_count = 0
    prev_reliable_raw_start: float | None = None
    prev_reliable_anchor_idx = -1
    last_used_recognized_idx = -1

    for i in range(len(lyric_lines)):
        if not strong_mask[i]:
            continue
        matched = match_map.get(i) or []
        if matched:
            start, idx = _estimate_line_raw_start(
                recognized_words,
                matched,
                line_tokens[i],
                russian_mode=cfg.russian_mode,
            )
            confidence = min(1.0, len(matched) / max(1.0, len(line_tokens[i])))
            reliable, reliability_details = _is_global_match_reliable(
                line_idx=i,
                line_count=len(lyric_lines),
                line_tokens=line_tokens[i],
                matched=matched,
                recognized_words=recognized_words,
                raw_start_seconds=start,
                confidence=confidence,
                prev_reliable_line_idx=prev_reliable_line_idx,
                prev_reliable_token_count=prev_reliable_token_count,
                prev_reliable_raw_start_seconds=prev_reliable_raw_start,
                prev_reliable_anchor_idx=prev_reliable_anchor_idx,
                last_used_recognized_idx=last_used_recognized_idx,
                first_word_time=first_word_time,
                last_word_time=last_word_time,
                cfg=cfg,
                segments=segments,
            )
            line_details[i].update(reliability_details)
            if reliable and start is not None and idx >= 0:
                raw_starts[i] = float(start)
                confidences[i] = confidence
                statuses[i] = "matched_global"
                anchor_words[i] = recognized_words[idx].raw
                anchor_indices[i] = idx
                anchored_strong_lines += 1
                prev_reliable_line_idx = i
                prev_reliable_token_count = len(line_tokens[i])
                prev_reliable_raw_start = float(start)
                prev_reliable_anchor_idx = idx
                last_used_recognized_idx = max(last_used_recognized_idx, max(match.recognized_idx for match in matched))
                continue
            global_rejection_count += 1

        statuses[i] = "fallback_global"

    refined_block_count = _refine_global_results_locally(
        lyric_lines,
        line_tokens,
        strong_mask,
        recognized_words,
        match_map,
        raw_starts,
        confidences,
        statuses,
        anchor_words,
        anchor_indices,
        line_details,
        cfg=cfg,
        segments=segments,
        first_word_time=first_word_time,
        last_word_time=last_word_time,
    )

    # Fill strong unresolved with segment/gap fallback
    for i in range(len(lyric_lines)):
        if not strong_mask[i] or raw_starts[i] is not None:
            continue
        prev = max((raw_starts[k] for k in range(i - 1, -1, -1) if raw_starts[k] is not None), default=None)
        prev_line_idx = max((k for k in range(i - 1, -1, -1) if raw_starts[k] is not None and strong_mask[k]), default=-1)
        prev_segment_idx = _time_to_segment_idx(segments, prev)
        diagnostic_prior: _SegmentPrior | None = None
        diagnostic_candidates = _build_local_block_candidates(
            i,
            line_tokens,
            recognized_words,
            cfg=cfg,
            first_word_time=first_word_time,
            last_word_time=last_word_time,
            cursor_start=0 if prev_line_idx < 0 else max(0, anchor_indices[prev_line_idx] + 1),
            cursor_end=len(recognized_words) - 1,
            segments=segments,
            prev_confirmed_line_idx=prev_line_idx,
            prev_confirmed_segment_idx=prev_segment_idx,
        )
        if diagnostic_candidates:
            diagnostic = diagnostic_candidates[0]
            diagnostic_prior = _estimate_segment_prior(
                segments=segments,
                matched_time=diagnostic.raw_start_seconds,
                line_idx=i,
                total_lines=len(lyric_lines),
                prev_confirmed_line_idx=prev_line_idx,
                prev_confirmed_segment_idx=prev_segment_idx,
            )
            line_details[i]["matched_segment_idx"] = diagnostic_prior.matched_segment_idx
            line_details[i]["segment_jump_count"] = diagnostic_prior.segment_jump_count
            line_details[i]["segment_prior_score"] = round(diagnostic_prior.prior_score, 4)
            if (
                diagnostic_prior.segment_jump_count > max(0, i - prev_line_idx)
                and diagnostic_prior.prior_score < -1.5
            ):
                line_details[i].setdefault("rejected_reason", "segment_jump_too_large")
        allow_segment_fallback = cfg.allow_segment_fallback
        rejected_reason = str(line_details[i].get("rejected_reason") or "")
        if _should_block_segment_fallback(
            rejected_reason,
            diagnostic_prior,
            line_idx=i,
            prev_line_idx=prev_line_idx,
        ):
            allow_segment_fallback = False
        if allow_segment_fallback:
            seg_start = _find_segment_fallback_start(segments, prev_start=(prev or -min_gap_s), min_gap_s=min_gap_s)
            if seg_start is not None and _is_segment_fallback_plausible(
                candidate_start=seg_start,
                prev_start=prev,
                line_idx=i,
                total_lines=len(lyric_lines),
                first_word_time=first_word_time,
                last_word_time=last_word_time,
                cfg=cfg,
                min_gap_s=min_gap_s,
            ):
                raw_starts[i] = seg_start
                statuses[i] = "fallback_segment"
                continue
            if seg_start is not None:
                line_details[i].setdefault("rejected_reason", "segment_fallback_too_far_ahead")
        raw_starts[i] = (prev + min_gap_s) if prev is not None else 0.0
        statuses[i] = "fallback_gap"

    # Phase B: bridge unresolved strong blocks between reliable strong anchors.
    bridged_strong_fallback_lines = _bridge_unresolved_strong_blocks(
        raw_starts=raw_starts,
        statuses=statuses,
        strong_mask=strong_mask,
        line_tokens=line_tokens,
        min_gap_s=min_gap_s,
    )

    # Phase C: interpolate low-information lines between strong anchors
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
                details=line_details[i],
            )
        )

    stats = {
        "anchored_strong_lines": anchored_strong_lines,
        "interpolated_low_info_lines": interpolated_low_info_lines,
        "global_rejection_count": global_rejection_count,
        "refined_block_count": refined_block_count,
        "bridged_strong_fallback_lines": bridged_strong_fallback_lines,
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
