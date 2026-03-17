from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.auto_sync import AutoSyncError, _split_lyrics_text, auto_sync_lyrics
from models import LyricLine


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
