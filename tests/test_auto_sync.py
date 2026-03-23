from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.auto_sync import AutoSyncError, _split_lyrics_text, auto_sync_lyrics
from models import LyricLine
from core.line_alignment import (
    LineAlignmentConfig,
    RecognizedWord,
    SegmentInfo,
    _find_segment_fallback_start,
    align_lyric_lines,
)


def test_split_lyrics_text_strips_and_ignores_empty_lines() -> None:
    source = "\n first line \n\n  second line\n   \nthird line  "

    assert _split_lyrics_text(source) == ["first line", "second line", "third line"]


def test_auto_sync_lyrics_rejects_single_non_empty_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.auto_sync.get_missing_autosync_packages", lambda: [])

    with pytest.raises(AutoSyncError, match="минимум 2 непустые строки"):
        auto_sync_lyrics("fake.wav", "only one line")


def test_auto_sync_lyrics_falls_back_to_librosa_when_whisperx_backend_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = [
        LyricLine(start_time="00:01.00", text="line 1"),
        LyricLine(start_time="00:02.00", text="line 2"),
    ]
    monkeypatch.setattr("core.auto_sync.get_missing_autosync_packages", lambda: [])
    monkeypatch.setattr(
        "core.auto_sync._auto_sync_whisperx_word_level",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AutoSyncError("whisperx unavailable")),
    )
    monkeypatch.setattr("core.auto_sync._auto_sync_librosa", lambda *_args, **_kwargs: expected)

    result = auto_sync_lyrics("fake.wav", "line 1\nline 2")

    assert result == expected


def test_auto_sync_lyrics_reports_missing_optional_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.auto_sync.get_missing_autosync_packages", lambda: ["whisperx", "demucs"])

    with pytest.raises(AutoSyncError, match="optional-зависимости"):
        auto_sync_lyrics("fake.wav", "line 1\nline 2")



def test_auto_sync_lyrics_rejects_blank_full_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.auto_sync.get_missing_autosync_packages", lambda: [])

    with pytest.raises(AutoSyncError, match="Текст трека пуст"):
        auto_sync_lyrics("fake.wav", "   \n\t")


def test_build_dependency_error_contains_package_list() -> None:
    from core.auto_sync import build_autosync_dependency_error

    message = build_autosync_dependency_error(["whisperx", "demucs"])
    assert "whisperx, demucs" in message
    assert "requirements-autosync.txt" in message


def test_format_mmss_rounding_and_non_negative() -> None:
    from core.auto_sync import _format_mmss

    assert _format_mmss(-1.0) == "00:00.00"
    assert _format_mmss(61.239) == "01:01.24"


def test_guess_language_code_detects_russian_and_english() -> None:
    from core.auto_sync import _guess_language_code

    assert _guess_language_code(["Привет мир"]) == "ru"
    assert _guess_language_code(["hello world"]) == "en"


def test_extract_words_and_segments_handles_missing_fields() -> None:
    from core.auto_sync import _extract_words_and_segments

    aligned = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "Hello",
                "words": [{"text": "Hello", "start": 0.0, "end": 0.5}],
            },
            {
                "start": 1.0,
                "end": 2.0,
                "text": None,
                "words": [{"word": "world", "start": 1.1, "end": 1.5, "score": 0.9}],
            },
        ]
    }
    words, segments = _extract_words_and_segments(aligned)
    assert words[0]["word"] == "Hello"
    assert words[1]["word"] == "world"
    assert segments[1]["text"] == ""


def test_get_missing_autosync_packages_reports_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    from core.auto_sync import get_missing_autosync_packages

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "whisperx":
            raise ImportError("no whisperx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    missing = get_missing_autosync_packages()
    assert "whisperx" in missing



def test_auto_sync_librosa_returns_aligned_lines_with_mocked_librosa(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    class FakeEffects:
        @staticmethod
        def hpss(y):
            return y, y

        @staticmethod
        def split(_y, top_db=26):
            return np.array([[1000, 3000], [6000, 8000]], dtype=np.int64)

    class FakeOnset:
        @staticmethod
        def onset_strength(**_kwargs):
            return np.array([0.1, 0.2, 0.3], dtype=np.float32)

        @staticmethod
        def onset_detect(**_kwargs):
            return np.array([1, 5, 9], dtype=np.int64)

    class FakeLibrosa:
        effects = FakeEffects
        onset = FakeOnset

        @staticmethod
        def load(_path, sr=22050, mono=True):
            return np.ones(22050, dtype=np.float32), sr

        @staticmethod
        def get_duration(y, sr):
            return float(len(y) / sr)

        @staticmethod
        def frames_to_time(frames, sr):
            return np.array(frames, dtype=np.float32) / float(sr)

    import core.auto_sync as mod

    monkeypatch.setitem(sys.modules, "librosa", FakeLibrosa)
    result = mod._auto_sync_librosa("fake.wav", ["first line", "second line"])  # noqa: SLF001

    assert len(result) == 2
    assert result[0].text == "first line"
    assert result[1].text == "second line"


def test_auto_sync_librosa_raises_on_analysis_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLibrosa:
        @staticmethod
        def load(*_args, **_kwargs):
            raise RuntimeError("broken")

    import core.auto_sync as mod

    monkeypatch.setitem(sys.modules, "librosa", FakeLibrosa)

    with pytest.raises(AutoSyncError, match="Ошибка анализа аудио librosa"):
        mod._auto_sync_librosa("fake.wav", ["line 1", "line 2"])  # noqa: SLF001


def test_align_lyric_lines_recovers_line_start_when_first_tokens_are_missing() -> None:
    lyric_lines = [
        "Это стало ужасно",
        "Не резко, а медленно",
        "Как будто мир стирает краски небрежно",
    ]
    recognized = [
        RecognizedWord(raw="это", normalized="это", start=0.00, end=0.20, confidence=0.99, index=0),
        RecognizedWord(raw="стало", normalized="стало", start=0.30, end=0.50, confidence=0.99, index=1),
        RecognizedWord(raw="ужасно", normalized="ужасно", start=0.60, end=0.80, confidence=0.99, index=2),
        RecognizedWord(raw="резко", normalized="резко", start=1.20, end=1.40, confidence=0.99, index=3),
        RecognizedWord(raw="медленно", normalized="медленно", start=1.80, end=2.00, confidence=0.99, index=4),
        RecognizedWord(raw="будто", normalized="будто", start=2.40, end=2.60, confidence=0.99, index=5),
        RecognizedWord(raw="мир", normalized="мир", start=2.70, end=2.90, confidence=0.99, index=6),
        RecognizedWord(raw="стирает", normalized="стирает", start=3.00, end=3.20, confidence=0.99, index=7),
        RecognizedWord(raw="краски", normalized="краски", start=3.30, end=3.50, confidence=0.99, index=8),
        RecognizedWord(raw="небрежно", normalized="небрежно", start=3.60, end=3.80, confidence=0.99, index=9),
    ]

    result = align_lyric_lines(
        lyric_lines,
        recognized,
        config=LineAlignmentConfig(russian_mode=True),
    )

    assert result[0].raw_start_seconds == pytest.approx(0.0, abs=0.01)
    assert result[1].raw_start_seconds == pytest.approx(1.20, abs=0.01)
    assert result[2].raw_start_seconds == pytest.approx(2.40, abs=0.01)


def test_align_lyric_lines_does_not_extrapolate_single_sparse_match() -> None:
    lyric_lines = ["this is a long line now"]
    recognized = [
        RecognizedWord(raw="intro", normalized="intro", start=0.0, end=0.1, confidence=0.99, index=0),
        RecognizedWord(raw="now", normalized="now", start=10.0, end=10.2, confidence=0.99, index=1),
    ]

    greedy = align_lyric_lines(
        lyric_lines,
        recognized,
        config=LineAlignmentConfig(use_global_alignment=False),
    )
    global_result = align_lyric_lines(
        lyric_lines,
        recognized,
        config=LineAlignmentConfig(use_global_alignment=True),
    )

    assert greedy[0].raw_start_seconds == pytest.approx(10.0, abs=0.01)
    assert global_result[0].raw_start_seconds == pytest.approx(10.0, abs=0.01)


def test_align_lyric_lines_does_not_backdate_missing_leading_content_word() -> None:
    lyric_lines = ["darling run into the night now"]
    recognized = [
        RecognizedWord(raw="run", normalized="run", start=10.0, end=10.2, confidence=0.99, index=0),
        RecognizedWord(raw="into", normalized="into", start=10.3, end=10.5, confidence=0.99, index=1),
        RecognizedWord(raw="the", normalized="the", start=10.6, end=10.8, confidence=0.99, index=2),
        RecognizedWord(raw="night", normalized="night", start=10.9, end=11.1, confidence=0.99, index=3),
        RecognizedWord(raw="now", normalized="now", start=11.2, end=11.4, confidence=0.99, index=4),
    ]

    greedy = align_lyric_lines(
        lyric_lines,
        recognized,
        config=LineAlignmentConfig(use_global_alignment=False),
    )
    global_result = align_lyric_lines(
        lyric_lines,
        recognized,
        config=LineAlignmentConfig(use_global_alignment=True),
    )

    assert greedy[0].raw_start_seconds == pytest.approx(10.0, abs=0.01)
    assert global_result[0].raw_start_seconds == pytest.approx(10.0, abs=0.01)


def test_align_lyric_lines_does_not_backdate_unsung_lead_in_tokens() -> None:
    lyric_lines = ["maybe tonight we run"]
    recognized = [
        RecognizedWord(raw="tonight", normalized="tonight", start=10.0, end=10.2, confidence=0.99, index=0),
        RecognizedWord(raw="we", normalized="we", start=10.3, end=10.5, confidence=0.99, index=1),
        RecognizedWord(raw="run", normalized="run", start=10.6, end=10.8, confidence=0.99, index=2),
    ]

    greedy = align_lyric_lines(
        lyric_lines,
        recognized,
        config=LineAlignmentConfig(use_global_alignment=False),
    )
    global_result = align_lyric_lines(
        lyric_lines,
        recognized,
        config=LineAlignmentConfig(use_global_alignment=True),
    )

    assert greedy[0].raw_start_seconds == pytest.approx(10.0, abs=0.01)
    assert global_result[0].raw_start_seconds == pytest.approx(10.0, abs=0.01)


def test_align_lyric_lines_does_not_backdate_skipped_ad_lib_prefix() -> None:
    lyric_lines = ["oh tonight we run"]
    recognized = [
        RecognizedWord(raw="tonight", normalized="tonight", start=10.0, end=10.2, confidence=0.99, index=0),
        RecognizedWord(raw="we", normalized="we", start=10.3, end=10.5, confidence=0.99, index=1),
        RecognizedWord(raw="run", normalized="run", start=10.6, end=10.8, confidence=0.99, index=2),
    ]

    greedy = align_lyric_lines(
        lyric_lines,
        recognized,
        config=LineAlignmentConfig(use_global_alignment=False),
    )
    global_result = align_lyric_lines(
        lyric_lines,
        recognized,
        config=LineAlignmentConfig(use_global_alignment=True),
    )

    assert greedy[0].raw_start_seconds == pytest.approx(10.0, abs=0.01)
    assert global_result[0].raw_start_seconds == pytest.approx(10.0, abs=0.01)


def test_align_lyric_lines_greedy_ignores_rejected_early_match_for_raw_start() -> None:
    lyric_lines = ["a miracle happens"]
    recognized = [
        RecognizedWord(raw="a", normalized="a", start=9.5, end=9.6, confidence=0.99, index=0),
        RecognizedWord(raw="miracle", normalized="miracle", start=10.2, end=10.4, confidence=0.99, index=1),
        RecognizedWord(raw="happens", normalized="happens", start=10.6, end=10.8, confidence=0.99, index=2),
    ]

    result = align_lyric_lines(
        lyric_lines,
        recognized,
        config=LineAlignmentConfig(use_global_alignment=False),
    )

    assert result[0].anchor_word == "miracle"
    assert result[0].raw_start_seconds == pytest.approx(10.2, abs=0.01)


def test_align_lyric_lines_greedy_keeps_later_matches_for_backdating_after_skipping_stopword_anchor() -> None:
    lyric_lines = ["and we can still go home"]
    recognized = [
        RecognizedWord(raw="and", normalized="and", start=10.0, end=10.1, confidence=0.99, index=0),
        RecognizedWord(raw="can", normalized="can", start=10.4, end=10.5, confidence=0.99, index=1),
        RecognizedWord(raw="still", normalized="still", start=10.7, end=10.8, confidence=0.99, index=2),
        RecognizedWord(raw="go", normalized="go", start=11.0, end=11.1, confidence=0.99, index=3),
        RecognizedWord(raw="home", normalized="home", start=11.3, end=11.4, confidence=0.99, index=4),
    ]

    result = align_lyric_lines(
        lyric_lines,
        recognized,
        config=LineAlignmentConfig(use_global_alignment=False),
    )

    assert result[0].anchor_word == "can"
    assert result[0].raw_start_seconds == pytest.approx(9.8, abs=0.01)

def test_align_lyric_lines_uses_true_median_gap_for_backdating() -> None:
    lyric_lines = ["and we can go"]
    recognized = [
        RecognizedWord(raw="we", normalized="we", start=10.0, end=10.1, confidence=0.99, index=0),
        RecognizedWord(raw="can", normalized="can", start=10.2, end=10.3, confidence=0.99, index=1),
        RecognizedWord(raw="go", normalized="go", start=10.8, end=10.9, confidence=0.99, index=2),
    ]

    greedy = align_lyric_lines(
        lyric_lines,
        recognized,
        config=LineAlignmentConfig(use_global_alignment=False),
    )
    global_result = align_lyric_lines(
        lyric_lines,
        recognized,
        config=LineAlignmentConfig(use_global_alignment=True),
    )

    assert greedy[0].raw_start_seconds == pytest.approx(10.2, abs=0.01)
    assert global_result[0].raw_start_seconds == pytest.approx(9.6, abs=0.01)


def test_align_lyric_lines_does_not_backdate_stopword_prefix_that_may_be_unsung() -> None:
    lyric_lines = ["I remember you"]
    recognized = [
        RecognizedWord(raw="remember", normalized="remember", start=10.0, end=10.2, confidence=0.99, index=0),
        RecognizedWord(raw="you", normalized="you", start=10.3, end=10.5, confidence=0.99, index=1),
    ]

    greedy = align_lyric_lines(
        lyric_lines,
        recognized,
        config=LineAlignmentConfig(use_global_alignment=False),
    )
    global_result = align_lyric_lines(
        lyric_lines,
        recognized,
        config=LineAlignmentConfig(use_global_alignment=True),
    )

    assert greedy[0].raw_start_seconds == pytest.approx(10.0, abs=0.01)
    assert global_result[0].raw_start_seconds == pytest.approx(10.0, abs=0.01)


def test_align_lyric_lines_global_penalizes_far_line_jump_even_with_overlap() -> None:
    lyric_lines = [f"line {idx} unique" for idx in range(18)]
    lyric_lines[0] = "alpha start begins"
    lyric_lines[17] = "alpha finish ending"

    recognized = [
        RecognizedWord(raw="alpha", normalized="alpha", start=0.0, end=0.2, confidence=0.99, index=0),
        RecognizedWord(raw="start", normalized="start", start=0.2, end=0.4, confidence=0.99, index=1),
        RecognizedWord(raw="begins", normalized="begins", start=0.4, end=0.6, confidence=0.99, index=2),
        RecognizedWord(raw="alpha", normalized="alpha", start=25.0, end=25.2, confidence=0.99, index=3),
        RecognizedWord(raw="finish", normalized="finish", start=25.2, end=25.4, confidence=0.99, index=4),
        RecognizedWord(raw="ending", normalized="ending", start=25.4, end=25.6, confidence=0.99, index=5),
    ]

    result = align_lyric_lines(
        lyric_lines,
        recognized,
        config=LineAlignmentConfig(use_global_alignment=True),
    )

    assert result[0].status == "matched_global"
    assert result[0].raw_start_seconds == pytest.approx(0.0, abs=0.01)
    assert result[17].status != "matched_global"
    assert result[17].raw_start_seconds > result[0].raw_start_seconds


def test_align_lyric_lines_global_rejects_late_local_match_when_previous_line_is_recent() -> None:
    lyric_lines = [
        "alpha start now",
        "gamma after long section",
        "delta missing outro",
    ]
    recognized = [
        RecognizedWord(raw="alpha", normalized="alpha", start=0.0, end=0.2, confidence=0.99, index=0),
        RecognizedWord(raw="start", normalized="start", start=0.2, end=0.4, confidence=0.99, index=1),
        RecognizedWord(raw="now", normalized="now", start=0.4, end=0.6, confidence=0.99, index=2),
        RecognizedWord(raw="gamma", normalized="gamma", start=12.0, end=12.2, confidence=0.99, index=3),
        RecognizedWord(raw="after", normalized="after", start=12.2, end=12.4, confidence=0.99, index=4),
        RecognizedWord(raw="long", normalized="long", start=12.4, end=12.6, confidence=0.99, index=5),
        RecognizedWord(raw="section", normalized="section", start=12.6, end=12.8, confidence=0.99, index=6),
    ]

    result = align_lyric_lines(
        lyric_lines,
        recognized,
        config=LineAlignmentConfig(use_global_alignment=True),
    )

    assert result[0].status == "matched_global"
    assert result[1].status != "matched_global"
    assert result[1].details["rejected_reason"] in {
        "gap_from_prev_line_too_large",
        "recognized_jump_too_large",
        "line_density_conflict",
    }
    assert result[1].details["gap_from_prev_line_s"] > 10.0
    assert result[1].details["recognized_jump_words"] >= 1


def test_align_lyric_lines_global_refines_suspicious_anchor_block_instead_of_locking_tail() -> None:
    lyric_lines = [f"line{idx} aa bb" for idx in range(20)]
    recognized: list[RecognizedWord] = []

    word_index = 0
    for idx, line in enumerate(lyric_lines):
        current_time = float(idx)
        for token in line.split():
            recognized.append(
                RecognizedWord(
                    raw=token,
                    normalized=token,
                    start=current_time,
                    end=current_time + 0.2,
                    confidence=0.99,
                    index=word_index,
                )
            )
            word_index += 1
            current_time += 0.2

    for offset, token in enumerate(lyric_lines[-1].split()):
        recognized.append(
            RecognizedWord(
                raw=token,
                normalized=token,
                start=25.0 + (offset * 0.2),
                end=25.2 + (offset * 0.2),
                confidence=0.99,
                index=word_index,
            )
        )
        word_index += 1

    result = align_lyric_lines(
        lyric_lines,
        recognized,
        config=LineAlignmentConfig(use_global_alignment=True),
    )

    assert result[18].status == "matched_global_refined_block"
    assert result[18].raw_start_seconds == pytest.approx(18.0, abs=0.01)
    assert result[18].details["refined_from_block"] == "17:18"
    assert result[18].details["suspicious_gap_s"] > 2.0
    assert result[19].raw_start_seconds > 24.0


def test_align_lyric_lines_segment_prior_rejects_far_future_match() -> None:
    lyric_lines = [
        "alpha intro start",
        "beta second line",
        "gamma closing line",
    ]
    recognized = [
        RecognizedWord(raw="alpha", normalized="alpha", start=0.0, end=0.2, confidence=0.99, index=0),
        RecognizedWord(raw="intro", normalized="intro", start=0.2, end=0.4, confidence=0.99, index=1),
        RecognizedWord(raw="start", normalized="start", start=0.4, end=0.6, confidence=0.99, index=2),
        RecognizedWord(raw="beta", normalized="beta", start=29.0, end=29.2, confidence=0.99, index=3),
        RecognizedWord(raw="second", normalized="second", start=29.2, end=29.4, confidence=0.99, index=4),
        RecognizedWord(raw="line", normalized="line", start=29.4, end=29.6, confidence=0.99, index=5),
        RecognizedWord(raw="gamma", normalized="gamma", start=30.0, end=30.2, confidence=0.99, index=6),
        RecognizedWord(raw="closing", normalized="closing", start=30.2, end=30.4, confidence=0.99, index=7),
        RecognizedWord(raw="line", normalized="line", start=30.4, end=30.6, confidence=0.99, index=8),
    ]
    segments = [SegmentInfo(start=float(idx), end=float(idx) + 1.0, text=f"seg{idx}") for idx in range(32)]

    result = align_lyric_lines(
        lyric_lines,
        recognized,
        segments=segments,
        config=LineAlignmentConfig(use_global_alignment=True),
    )

    assert result[0].status == "matched_global"
    assert result[0].details["matched_segment_idx"] == 0
    assert result[1].status != "matched_global"
    assert result[1].details["matched_segment_idx"] == 29
    assert result[1].details["segment_jump_count"] >= 27
    assert result[1].details["segment_prior_score"] < -10.0
    assert result[1].details["rejected_reason"] == "segment_jump_too_large"


def test_align_lyric_lines_global_does_not_promote_far_future_segment_fallback() -> None:
    lyric_lines = [
        "Это стало ужасно",
        "Не резко, а медленно",
        "Как будто мир стирает краски небрежно",
    ]
    recognized = [
        RecognizedWord(raw="Это", normalized="это", start=0.30, end=0.45, confidence=0.99, index=0),
        RecognizedWord(raw="стало", normalized="стало", start=0.46, end=0.60, confidence=0.99, index=1),
        RecognizedWord(raw="ужасно", normalized="ужасно", start=0.61, end=0.82, confidence=0.99, index=2),
        RecognizedWord(raw="не", normalized="не", start=30.17, end=30.25, confidence=0.99, index=3),
        RecognizedWord(raw="резко", normalized="резко", start=30.26, end=30.40, confidence=0.99, index=4),
        RecognizedWord(raw="а", normalized="а", start=30.41, end=30.46, confidence=0.99, index=5),
        RecognizedWord(raw="медленно", normalized="медленно", start=30.47, end=30.70, confidence=0.99, index=6),
    ]
    segments = [
        SegmentInfo(start=0.30, end=0.90, text="Это стало ужасно"),
        SegmentInfo(start=30.17, end=30.80, text="Не резко, а медленно"),
    ]

    result = align_lyric_lines(
        lyric_lines,
        recognized,
        segments=segments,
        config=LineAlignmentConfig(russian_mode=True, use_global_alignment=True),
    )

    assert result[0].status == "matched_global"
    assert result[1].status == "fallback_gap"
    assert result[1].raw_start_seconds < 5.0
    assert result[1].details["rejected_reason"] == "segment_jump_too_large"
    assert result[2].raw_start_seconds >= result[1].raw_start_seconds


def test_find_segment_fallback_start_does_not_wrap_to_first_segment() -> None:
    segments = [SegmentInfo(start=3.0, end=4.0, text="seg")]

    assert _find_segment_fallback_start(segments, prev_start=10.0, min_gap_s=0.12) is None
