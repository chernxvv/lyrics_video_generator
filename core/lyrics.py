from __future__ import annotations

from bisect import bisect_right
import re

from models import LyricLine


def parse_mmss(value: str) -> float:
    raw_value = value.strip()
    if not re.fullmatch(r"\d{2}:\d{2}(?:[.,]\d{1,2})?", raw_value):
        raise ValueError("Время должно быть в формате мм:сс или мм:сс.сс")
    mm, ss = raw_value.split(":", 1)
    ss = ss.replace(",", ".")
    seconds = int(mm) * 60 + float(ss)
    if float(ss) >= 60:
        raise ValueError("Секунды должны быть меньше 60")
    return float(seconds)


def sort_lyrics(lines: list[LyricLine]) -> list[LyricLine]:
    return sorted(lines, key=lambda x: parse_mmss(x.start_time))


def active_line_index(lines: list[LyricLine], current_time: float) -> int:
    if not lines:
        return -1
    sorted_lines = sort_lyrics(lines)
    idx = 0
    for i, line in enumerate(sorted_lines):
        if parse_mmss(line.start_time) <= current_time:
            idx = i
        else:
            break
    return idx


def prepare_timeline(lines: list[LyricLine]) -> tuple[list[LyricLine], list[float]]:
    sorted_lines = sort_lyrics(lines)
    start_times = [parse_mmss(line.start_time) for line in sorted_lines]
    return sorted_lines, start_times


def active_line_index_precomputed(start_times: list[float], current_time: float) -> int:
    if not start_times:
        return -1
    return bisect_right(start_times, current_time) - 1
