from __future__ import annotations

from dataclasses import dataclass, field

from core.line_alignment import LineAlignmentConfig, RecognizedWord, SegmentInfo


@dataclass(frozen=True, slots=True)
class AlignmentExpectation:
    statuses: tuple[str | None, ...]
    raw_start_ranges: tuple[tuple[float | None, float | None] | None, ...]
    anchor_words: tuple[str | None, ...] = ()
    anchor_word_indices: tuple[int | None, ...] = ()
    detail_equals: tuple[dict[str, object], ...] = ()
    detail_predicates: tuple[dict[str, tuple[str, object]], ...] = ()


@dataclass(frozen=True, slots=True)
class AlignmentScenario:
    scenario_id: str
    lyric_lines: tuple[str, ...]
    recognized_words: tuple[RecognizedWord, ...]
    expected: dict[str, AlignmentExpectation]
    segments: tuple[SegmentInfo, ...] = ()
    config: LineAlignmentConfig = field(default_factory=LineAlignmentConfig)


def _word(raw: str, start: float, end: float, index: int, normalized: str | None = None, confidence: float = 0.99) -> RecognizedWord:
    return RecognizedWord(raw=raw, normalized=normalized or raw.lower(), start=start, end=end, confidence=confidence, index=index)


def _segment(start: float, end: float, text: str) -> SegmentInfo:
    return SegmentInfo(start=start, end=end, text=text)


SCENARIOS: tuple[AlignmentScenario, ...] = (
    AlignmentScenario(
        scenario_id="missing_first_tokens_ru",
        lyric_lines=(
            "Это стало ужасно",
            "Не резко, а медленно",
            "Как будто мир стирает краски небрежно",
        ),
        recognized_words=(
            _word("это", 0.00, 0.20, 0, normalized="это"),
            _word("стало", 0.30, 0.50, 1, normalized="стало"),
            _word("ужасно", 0.60, 0.80, 2, normalized="ужасно"),
            _word("резко", 1.20, 1.40, 3, normalized="резко"),
            _word("медленно", 1.80, 2.00, 4, normalized="медленно"),
            _word("будто", 2.40, 2.60, 5, normalized="будто"),
            _word("мир", 2.70, 2.90, 6, normalized="мир"),
            _word("стирает", 3.00, 3.20, 7, normalized="стирает"),
            _word("краски", 3.30, 3.50, 8, normalized="краски"),
            _word("небрежно", 3.60, 3.80, 9, normalized="небрежно"),
        ),
        config=LineAlignmentConfig(russian_mode=True),
        expected={
            "global": AlignmentExpectation(
                statuses=("matched_global", "matched_global", "matched_global"),
                raw_start_ranges=((0.0, 0.01), (1.20, 1.21), (2.40, 2.41)),
            ),
        },
    ),
    AlignmentScenario(
        scenario_id="single_sparse_match",
        lyric_lines=("this is a long line now",),
        recognized_words=(
            _word("intro", 0.0, 0.1, 0),
            _word("now", 10.0, 10.2, 1),
        ),
        expected={
            "greedy": AlignmentExpectation(statuses=("matched",), raw_start_ranges=((10.0, 10.01),)),
            "global": AlignmentExpectation(statuses=("matched_global",), raw_start_ranges=((10.0, 10.01),)),
        },
    ),
    AlignmentScenario(
        scenario_id="missing_leading_content_word",
        lyric_lines=("darling run into the night now",),
        recognized_words=(
            _word("run", 10.0, 10.2, 0),
            _word("into", 10.3, 10.5, 1),
            _word("the", 10.6, 10.8, 2),
            _word("night", 10.9, 11.1, 3),
            _word("now", 11.2, 11.4, 4),
        ),
        expected={
            "greedy": AlignmentExpectation(statuses=("matched",), raw_start_ranges=((10.0, 10.01),)),
            "global": AlignmentExpectation(statuses=("matched_global",), raw_start_ranges=((10.0, 10.01),)),
        },
    ),
    AlignmentScenario(
        scenario_id="unsung_lead_in_tokens",
        lyric_lines=("maybe tonight we run",),
        recognized_words=(
            _word("tonight", 10.0, 10.2, 0),
            _word("we", 10.3, 10.5, 1),
            _word("run", 10.6, 10.8, 2),
        ),
        expected={
            "greedy": AlignmentExpectation(statuses=("matched",), raw_start_ranges=((10.0, 10.01),)),
            "global": AlignmentExpectation(statuses=("matched_global",), raw_start_ranges=((10.0, 10.01),)),
        },
    ),
    AlignmentScenario(
        scenario_id="skipped_stopword_prefix_ad_lib",
        lyric_lines=("oh tonight we run",),
        recognized_words=(
            _word("tonight", 10.0, 10.2, 0),
            _word("we", 10.3, 10.5, 1),
            _word("run", 10.6, 10.8, 2),
        ),
        expected={
            "greedy": AlignmentExpectation(statuses=("matched",), raw_start_ranges=((10.0, 10.01),)),
            "global": AlignmentExpectation(statuses=("matched_global",), raw_start_ranges=((10.0, 10.01),)),
        },
    ),
    AlignmentScenario(
        scenario_id="rejected_early_match_ignored_for_raw_start",
        lyric_lines=("a miracle happens",),
        recognized_words=(
            _word("a", 9.5, 9.6, 0),
            _word("miracle", 10.2, 10.4, 1),
            _word("happens", 10.6, 10.8, 2),
        ),
        expected={
            "greedy": AlignmentExpectation(
                statuses=("matched",),
                raw_start_ranges=((10.2, 10.21),),
                anchor_words=("miracle",),
            ),
        },
    ),
    AlignmentScenario(
        scenario_id="stopword_anchor_backdating_uses_later_matches",
        lyric_lines=("and we can still go home",),
        recognized_words=(
            _word("and", 10.0, 10.1, 0),
            _word("can", 10.4, 10.5, 1),
            _word("still", 10.7, 10.8, 2),
            _word("go", 11.0, 11.1, 3),
            _word("home", 11.3, 11.4, 4),
        ),
        expected={
            "greedy": AlignmentExpectation(
                statuses=("matched",),
                raw_start_ranges=((9.79, 9.81),),
                anchor_words=("can",),
            ),
        },
    ),
    AlignmentScenario(
        scenario_id="true_median_gap_for_backdating",
        lyric_lines=("and we can go",),
        recognized_words=(
            _word("we", 10.0, 10.1, 0),
            _word("can", 10.2, 10.3, 1),
            _word("go", 10.8, 10.9, 2),
        ),
        expected={
            "greedy": AlignmentExpectation(statuses=("matched",), raw_start_ranges=((10.2, 10.21),)),
            "global": AlignmentExpectation(statuses=("matched_global",), raw_start_ranges=((9.59, 9.61),)),
        },
    ),
    AlignmentScenario(
        scenario_id="stopword_prefix_may_be_unsung",
        lyric_lines=("I remember you",),
        recognized_words=(
            _word("remember", 10.0, 10.2, 0),
            _word("you", 10.3, 10.5, 1),
        ),
        expected={
            "greedy": AlignmentExpectation(statuses=("matched",), raw_start_ranges=((10.0, 10.01),)),
            "global": AlignmentExpectation(statuses=("matched_global",), raw_start_ranges=((10.0, 10.01),)),
        },
    ),
    AlignmentScenario(
        scenario_id="global_penalizes_far_line_jump_even_with_overlap",
        lyric_lines=tuple("alpha start begins" if idx == 0 else "alpha finish ending" if idx == 17 else f"line {idx} unique" for idx in range(18)),
        recognized_words=(
            _word("alpha", 0.0, 0.2, 0),
            _word("start", 0.2, 0.4, 1),
            _word("begins", 0.4, 0.6, 2),
            _word("alpha", 25.0, 25.2, 3),
            _word("finish", 25.2, 25.4, 4),
            _word("ending", 25.4, 25.6, 5),
        ),
        expected={
            "global": AlignmentExpectation(
                statuses=tuple("matched_global" if idx == 0 else None for idx in range(18)),
                raw_start_ranges=tuple((0.0, 0.01) if idx == 0 else (0.01, 24.99) if idx == 17 else None for idx in range(18)),
            ),
        },
    ),
    AlignmentScenario(
        scenario_id="global_skips_low_info_lines_for_penalties",
        lyric_lines=(
            "alpha beta gamma",
            "now run echo",
            "we echo",
            "bright the",
            "and go now alpha",
        ),
        recognized_words=(
            _word("alpha", 0.0, 0.2, 0),
            _word("beta", 0.3, 0.5, 1),
            _word("gamma", 0.6, 0.8, 2),
            _word("now", 1.8, 2.0, 3),
            _word("run", 2.1, 2.3, 4),
            _word("echo", 2.4, 2.6, 5),
            _word("and", 5.0, 5.2, 6),
            _word("go", 5.3, 5.5, 7),
            _word("now", 5.6, 5.8, 8),
            _word("alpha", 5.9, 6.1, 9),
        ),
        expected={
            "global": AlignmentExpectation(
                statuses=(None, None, None, None, "matched_global"),
                raw_start_ranges=(None, None, None, None, (5.0, 5.01)),
                anchor_words=(None, None, None, None, "and"),
            ),
        },
    ),
    AlignmentScenario(
        scenario_id="late_local_match_rejected_when_previous_recent",
        lyric_lines=(
            "alpha start now",
            "gamma after long section",
            "delta missing outro",
        ),
        recognized_words=(
            _word("alpha", 0.0, 0.2, 0),
            _word("start", 0.2, 0.4, 1),
            _word("now", 0.4, 0.6, 2),
            _word("gamma", 12.0, 12.2, 3),
            _word("after", 12.2, 12.4, 4),
            _word("long", 12.4, 12.6, 5),
            _word("section", 12.6, 12.8, 6),
        ),
        expected={
            "global": AlignmentExpectation(
                statuses=("matched_global", None, None),
                raw_start_ranges=((0.0, 0.01), None, None),
                detail_predicates=(
                    {},
                    {
                        "gap_from_prev_line_s": (">", 10.0),
                        "recognized_jump_words": (">=", 1),
                        "rejected_reason": ("in", {"gap_from_prev_line_too_large", "recognized_jump_too_large", "line_density_conflict"}),
                    },
                    {},
                ),
            ),
        },
    ),
    AlignmentScenario(
        scenario_id="matched_global_refined_block",
        lyric_lines=tuple(f"line{idx} aa bb" for idx in range(20)),
        recognized_words=tuple(
            _word(token, float(idx) + offset * 0.2, float(idx) + offset * 0.2 + 0.2, idx * 3 + offset)
            for idx in range(20)
            for offset, token in enumerate((f"line{idx}", "aa", "bb"))
        ) + tuple(
            _word(token, 25.0 + offset * 0.2, 25.2 + offset * 0.2, 60 + offset)
            for offset, token in enumerate(("line19", "aa", "bb"))
        ),
        expected={
            "global": AlignmentExpectation(
                statuses=tuple("matched_global_refined_block" if idx == 18 else None for idx in range(20)),
                raw_start_ranges=tuple((18.0, 18.01) if idx == 18 else (24.01, None) if idx == 19 else None for idx in range(20)),
                detail_equals=tuple({"refined_from_block": "17:18"} if idx == 18 else {} for idx in range(20)),
                detail_predicates=tuple({"suspicious_gap_s": (">", 2.0)} if idx == 18 else {} for idx in range(20)),
            ),
        },
    ),
    AlignmentScenario(
        scenario_id="segment_jump_too_large",
        lyric_lines=(
            "alpha intro start",
            "beta second line",
            "gamma closing line",
        ),
        recognized_words=(
            _word("alpha", 0.0, 0.2, 0),
            _word("intro", 0.2, 0.4, 1),
            _word("start", 0.4, 0.6, 2),
            _word("beta", 29.0, 29.2, 3),
            _word("second", 29.2, 29.4, 4),
            _word("line", 29.4, 29.6, 5),
            _word("gamma", 30.0, 30.2, 6),
            _word("closing", 30.2, 30.4, 7),
            _word("line", 30.4, 30.6, 8),
        ),
        segments=tuple(_segment(float(idx), float(idx) + 1.0, f"seg{idx}") for idx in range(32)),
        expected={
            "global": AlignmentExpectation(
                statuses=("matched_global", None, None),
                raw_start_ranges=((0.0, 0.01), None, None),
                detail_equals=(
                    {"matched_segment_idx": 0},
                    {"matched_segment_idx": 29, "rejected_reason": "segment_jump_too_large"},
                    {},
                ),
                detail_predicates=(
                    {},
                    {"segment_jump_count": (">=", 27), "segment_prior_score": ("<", -10.0)},
                    {},
                ),
            ),
        },
    ),
    AlignmentScenario(
        scenario_id="fallback_segment",
        lyric_lines=("opening anchor intro", "gamma alpha echo"),
        recognized_words=(
            _word("opening", 0.20, 0.35, 0),
            _word("anchor", 0.36, 0.50, 1),
            _word("intro", 0.51, 0.66, 2),
            _word("gamma", 1.40, 1.55, 3),
        ),
        segments=(
            _segment(0.20, 0.70, "opening anchor intro"),
            _segment(1.40, 1.90, "gamma alpha echo"),
        ),
        expected={
            "global": AlignmentExpectation(
                statuses=("matched_global", "fallback_segment"),
                raw_start_ranges=((0.20, 0.21), (1.40, 1.41)),
                detail_equals=({}, {"matched_segment_idx": 1, "segment_jump_count": 0}),
            ),
        },
    ),
    AlignmentScenario(
        scenario_id="segment_fallback_too_far_ahead",
        lyric_lines=(
            "Это стало ужасно",
            "Не резко, а медленно",
            "Как будто мир стирает краски небрежно",
        ),
        recognized_words=(
            _word("Это", 0.30, 0.45, 0, normalized="это"),
            _word("стало", 0.46, 0.60, 1, normalized="стало"),
            _word("ужасно", 0.61, 0.82, 2, normalized="ужасно"),
            _word("не", 30.17, 30.25, 3, normalized="не"),
            _word("резко", 30.26, 30.40, 4, normalized="резко"),
            _word("а", 30.41, 30.46, 5, normalized="а"),
            _word("медленно", 30.47, 30.70, 6, normalized="медленно"),
        ),
        segments=(
            _segment(0.30, 0.90, "Это стало ужасно"),
            _segment(30.17, 30.80, "Не резко, а медленно"),
        ),
        config=LineAlignmentConfig(russian_mode=True, use_global_alignment=True),
        expected={
            "global": AlignmentExpectation(
                statuses=("matched_global", "fallback_gap", None),
                raw_start_ranges=((0.30, 0.31), (0.31, 4.99), None),
                detail_equals=({}, {"rejected_reason": "segment_fallback_too_far_ahead"}, {}),
            ),
        },
    ),
    AlignmentScenario(
        scenario_id="repeated_tokens_choose_exact_later_match",
        lyric_lines=(
            "home oh we",
            "bright go alpha alpha",
            "alpha night i night",
            "now delta run",
        ),
        recognized_words=(
            _word("home", 0.10, 0.25, 0),
            _word("oh", 0.35, 0.45, 1),
            _word("we", 0.55, 0.65, 2),
            _word("bright", 1.20, 1.35, 3),
            _word("go", 1.45, 1.55, 4),
            _word("alpha", 1.65, 1.78, 5),
            _word("alpha", 2.30, 2.45, 6),
            _word("alpha", 3.70, 3.85, 7),
            _word("night", 3.95, 4.08, 8),
            _word("i", 4.18, 4.28, 9),
            _word("night", 4.38, 4.52, 10),
            _word("now", 5.10, 5.20, 11),
            _word("delta", 5.30, 5.45, 12),
            _word("run", 5.50, 5.62, 13),
        ),
        expected={
            "greedy": AlignmentExpectation(
                statuses=(None, None, "matched", None),
                raw_start_ranges=(None, None, (3.70, 3.71), None),
                anchor_word_indices=(None, None, 7, None),
            ),
        },
    ),
    AlignmentScenario(
        scenario_id="global_transition_penalties_ignore_low_info_lines",
        lyric_lines=(
            "alpha beta gamma",
            "now run echo",
            "we echo",
            "bright the",
            "and go now alpha",
        ),
        recognized_words=(
            _word("alpha", 0.20, 0.32, 0),
            _word("beta", 0.40, 0.52, 1),
            _word("gamma", 0.60, 0.72, 2),
            _word("now", 1.30, 1.42, 3),
            _word("run", 1.50, 1.62, 4),
            _word("echo", 1.70, 1.82, 5),
            _word("we", 2.40, 2.52, 6),
            _word("echo", 2.60, 2.72, 7),
            _word("bright", 3.30, 3.42, 8),
            _word("the", 3.50, 3.62, 9),
            _word("and", 5.60, 5.72, 10),
            _word("go", 5.80, 5.92, 11),
            _word("now", 6.00, 6.12, 12),
            _word("alpha", 6.20, 6.32, 13),
        ),
        expected={
            "global": AlignmentExpectation(
                statuses=(None, None, None, None, "matched_global"),
                raw_start_ranges=(None, None, None, None, (5.60, 5.61)),
                anchor_word_indices=(None, None, None, None, 10),
            ),
        },
    ),
)


def get_scenarios(mode: str) -> list[AlignmentScenario]:
    return [scenario for scenario in SCENARIOS if mode in scenario.expected]
