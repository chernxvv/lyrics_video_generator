from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.render import RenderDependencyError, RenderError, _ensure_ffmpeg_available, _require_font_path, _resolve_video_codec
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
