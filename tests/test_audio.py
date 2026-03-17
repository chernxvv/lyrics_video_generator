from pathlib import Path
import sys
from types import SimpleNamespace
import json

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.audio import AudioError, ensure_ffprobe_available, probe_audio_duration


def test_ensure_ffprobe_available_raises_when_binary_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.audio.shutil.which", lambda _: None)

    with pytest.raises(AudioError, match="Не найден ffprobe"):
        ensure_ffprobe_available()


def test_probe_audio_duration_raises_on_invalid_probe_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"x")
    monkeypatch.setattr("core.audio.shutil.which", lambda _: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        "core.audio.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="not-json", stderr=""),
    )

    with pytest.raises(json.JSONDecodeError, match="Expecting value"):
        probe_audio_duration(audio)
