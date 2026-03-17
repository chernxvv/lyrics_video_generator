from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.lyrics import active_line_index, active_line_index_precomputed, parse_mmss, prepare_timeline, sort_lyrics
from models import LyricLine


def test_parse_mmss_accepts_comma_and_dot() -> None:
    assert parse_mmss("01:02,50") == pytest.approx(62.5)
    assert parse_mmss("01:02.50") == pytest.approx(62.5)


def test_parse_mmss_rejects_seconds_over_59() -> None:
    with pytest.raises(ValueError, match="Секунды должны быть меньше 60"):
        parse_mmss("00:60")


def test_sort_and_prepare_timeline() -> None:
    lines = [LyricLine("00:20", "b"), LyricLine("00:10", "a")]
    sorted_lines = sort_lyrics(lines)
    assert [line.text for line in sorted_lines] == ["a", "b"]

    prepared_lines, starts = prepare_timeline(lines)
    assert [line.text for line in prepared_lines] == ["a", "b"]
    assert starts == [10.0, 20.0]


def test_active_line_index_behaviour() -> None:
    lines = [LyricLine("00:10", "a"), LyricLine("00:20", "b")]
    assert active_line_index([], 5.0) == -1
    assert active_line_index(lines, 5.0) == 0
    assert active_line_index(lines, 15.0) == 0
    assert active_line_index(lines, 21.0) == 1


def test_active_line_index_precomputed_behaviour() -> None:
    assert active_line_index_precomputed([], 12.0) == -1
    assert active_line_index_precomputed([10.0, 20.0], 5.0) == -1
    assert active_line_index_precomputed([10.0, 20.0], 10.0) == 0
    assert active_line_index_precomputed([10.0, 20.0], 25.0) == 1
