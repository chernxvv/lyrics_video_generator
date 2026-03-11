from __future__ import annotations

from models import LyricLine


def parse_mmss(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) != 2:
        raise ValueError("Время должно быть в формате мм:сс")
    mm, ss = parts
    if not (mm.isdigit() and ss.isdigit()):
        raise ValueError("Время должно содержать только цифры")
    seconds = int(mm) * 60 + int(ss)
    if int(ss) >= 60:
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
