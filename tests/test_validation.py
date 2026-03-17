from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.validation import ValidationError, validate_project
from models import LyricLine, ProjectData


def _base_project(tmp_path: Path) -> ProjectData:
    audio = tmp_path / "track.mp3"
    image = tmp_path / "cover.jpg"
    audio.write_bytes(b"audio")
    image.write_bytes(b"image")
    return ProjectData(
        audio_path=audio,
        image_path=image,
        artist="Artist",
        title="Title",
        lyrics=[LyricLine(start_time="00:10", text="line 1")],
        sync_mode="manual",
    )


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("artist", "   ", "Укажите исполнителя."),
        ("title", "\n", "Укажите название трека."),
    ],
)
def test_validate_project_rejects_empty_required_text_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: str, expected: str) -> None:
    project = _base_project(tmp_path)
    setattr(project, field, value)
    monkeypatch.setattr("core.validation.probe_audio_duration", lambda _: 120.0)

    with pytest.raises(ValidationError, match=expected):
        validate_project(project)


def test_validate_project_rejects_invalid_mmss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _base_project(tmp_path)
    project.lyrics = [LyricLine(start_time="bad", text="line")]
    monkeypatch.setattr("core.validation.probe_audio_duration", lambda _: 120.0)

    with pytest.raises(ValidationError, match="Время должно быть в формате"):
        validate_project(project)


def test_validate_project_rejects_line_after_audio_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _base_project(tmp_path)
    project.lyrics = [LyricLine(start_time="02:31", text="late line")]
    monkeypatch.setattr("core.validation.probe_audio_duration", lambda _: 60.0)

    with pytest.raises(ValidationError, match="начинается позже конца трека"):
        validate_project(project)


def test_validate_project_auto_mode_requires_full_text_when_not_autofilled(tmp_path: Path) -> None:
    project = _base_project(tmp_path)
    project.sync_mode = "auto"
    project.auto_sync_lyrics_text = "   "
    project.lyrics_autofilled = False
    project.auto_sync_audio_path = ""

    with pytest.raises(ValidationError, match="В режиме автосинхронизации нужно вставить полный текст трека"):
        validate_project(project)
