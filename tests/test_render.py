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
    _drawtext_style,
    _load_font,
    _measure_wrapped_height,
    _scale_box,
    _scale_value,
    _wrap_text_by_pixel_width,
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



def test_scale_helpers_behaviour() -> None:
    assert _scale_value(10, 1.5) == 15
    assert _scale_box((1, 2, 10, 20), 2.0) == (2, 4, 20, 40)


def test_drawtext_style_requires_font(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.render.FONT_DIR", tmp_path)

    with pytest.raises(RenderError, match="Не найден обязательный шрифт"):
        _drawtext_style(24, "missing.ttf", bordered=True)


def test_load_font_wraps_oserror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    font_file = tmp_path / "NotoSerif-Regular.ttf"
    font_file.write_bytes(b"not-a-real-font")
    monkeypatch.setattr("core.render.FONT_DIR", tmp_path)

    with pytest.raises(RenderError, match="Не удалось загрузить шрифт"):
        _load_font(20, "NotoSerif-Regular.ttf")


def test_wrap_and_measure_text_helpers() -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGBA", (300, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    lines = _wrap_text_by_pixel_width(draw, "one two three four five", font, max_width=40, stroke_width=0)
    assert len(lines) >= 2

    h = _measure_wrapped_height(draw, "one two three", 120, font, stroke_width=0, line_gap=4)
    assert h > 0
