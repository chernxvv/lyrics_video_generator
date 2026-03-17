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
