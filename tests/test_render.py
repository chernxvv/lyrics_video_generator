from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.render import (
    RenderDependencyError,
    RenderError,
    _compute_uniform_scale,
    _ensure_ffmpeg_available,
    _escape_drawtext,
    _ffmpeg_probe_output,
    _require_font_path,
    _resolve_video_codec,
    _supports_encoder,
    _supports_filter,
)
from models import RenderSettings


def test_require_font_path_raises_when_font_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.render.FONT_DIR", tmp_path)

    with pytest.raises(RenderError, match="Не найден обязательный шрифт"):
        _require_font_path("missing.ttf")


def test_ensure_ffmpeg_available_raises_when_binary_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.render.shutil.which", lambda _: None)

    with pytest.raises(RenderDependencyError, match="Не найден ffmpeg"):
        _ensure_ffmpeg_available()


def test_resolve_video_codec_falls_back_to_software_when_hw_not_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = RenderSettings(prefer_hw_encode=True, video_codec_hw="h264_nvenc", video_codec_sw="libx264")
    monkeypatch.setattr("core.render._supports_encoder", lambda _enc: False)

    codec, reason = _resolve_video_codec(settings)

    assert codec == "libx264"
    assert "NVENC недоступен" in reason


def test_resolve_video_codec_uses_software_when_hw_disabled() -> None:
    settings = RenderSettings(prefer_hw_encode=False, video_codec_hw="h264_nvenc", video_codec_sw="libx264")

    codec, reason = _resolve_video_codec(settings)

    assert codec == "libx264"
    assert "hardware encoding отключен" in reason



def test_ffmpeg_probe_output_returns_empty_on_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_args, **_kwargs):
        raise OSError("boom")

    monkeypatch.setattr("core.render.subprocess.run", _raise)
    assert _ffmpeg_probe_output(["ffmpeg", "-h"]) == ""


def test_ffmpeg_probe_output_returns_empty_on_nonzero_rc(monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        returncode = 1
        stdout = "ignored"

    monkeypatch.setattr("core.render.subprocess.run", lambda *_args, **_kwargs: Result())
    assert _ffmpeg_probe_output(["ffmpeg", "-h"]) == ""


def test_supports_encoder_and_filter_use_probe_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.render._ffmpeg_probe_output", lambda cmd: "h264_nvenc\nscale_cuda" if "-encoders" in cmd else "scale_cuda")
    assert _supports_encoder("h264_nvenc") is True
    assert _supports_filter("scale_cuda") is True


def test_compute_uniform_scale_rejects_invalid_base_canvas() -> None:
    with pytest.raises(RenderError, match="Некорректный базовый canvas"):
        _compute_uniform_scale(0, 100, 100, 100)


def test_escape_drawtext_escapes_special_characters() -> None:
    raw = "a:b'c%\\n"
    escaped = _escape_drawtext(raw)
    assert r"\:" in escaped
    assert r"\'" in escaped
    assert r"\%" in escaped
