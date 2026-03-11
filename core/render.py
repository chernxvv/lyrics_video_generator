from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path
from textwrap import wrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.background import build_background_frame
from core.layout import compute_layout
from core.lyrics import active_line_index_precomputed, prepare_timeline
from models import PaletteInfo, ProjectData, RenderSettings

logger = logging.getLogger(__name__)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_centered(draw: ImageDraw.ImageDraw, text: str, y: int, width: int, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (width - (bbox[2] - bbox[0])) // 2
    draw.text((x, y), text, font=font, fill=fill)


def _ensure_ffmpeg_available() -> None:
    logger.info("Проверка доступности ffmpeg")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "Не найден ffmpeg в PATH. Установите FFmpeg и добавьте ffmpeg в PATH перед генерацией видео."
        )


def _supports_encoder(encoder_name: str) -> bool:
    try:
        result = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=False)
    except OSError:
        return False
    if result.returncode != 0:
        logger.warning("Не удалось проверить список энкодеров ffmpeg: %s", result.stderr[-500:])
        return False
    return encoder_name in result.stdout


def _resolve_video_codec(settings: RenderSettings) -> tuple[str, str]:
    if not settings.prefer_hw_encode:
        return settings.video_codec_sw, "hardware encoding отключен в настройках"
    if _supports_encoder(settings.video_codec_hw):
        return settings.video_codec_hw, "обнаружена поддержка NVENC"
    return settings.video_codec_sw, "NVENC недоступен в ffmpeg -encoders"


def _build_static_overlay(
    project: ProjectData,
    layout,
    width: int,
    height: int,
    cover: Image.Image,
    font_artist,
    font_title,
    font_date,
) -> Image.Image:
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    _draw_centered(draw, project.artist, layout.artist_y, width, font_artist, (255, 255, 255, 255))
    _draw_centered(draw, "—", layout.dash_y, width, font_artist, (255, 255, 255, 255))
    _draw_centered(draw, project.title, layout.title_y, width, font_title, (255, 255, 255, 255))
    _draw_centered(draw, project.release_date, layout.date_y, width, font_date, (245, 245, 245, 255))
    overlay.paste(cover.convert("RGBA"), (layout.cover_box[0], layout.cover_box[1]))
    return overlay


def _build_lyrics_overlay(lines, current_index: int, width: int, height: int, font_lyrics) -> Image.Image:
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 105))
    draw = ImageDraw.Draw(overlay)
    visible = range(max(0, current_index - 2), min(len(lines), current_index + 3))
    y = 22
    for idx in visible:
        prefix = "▶ " if idx == current_index else ""
        color = (255, 255, 255, 255) if idx == current_index else (220, 220, 220, 255)
        for part in wrap(prefix + lines[idx].text, width=34):
            draw.text((24, y), part, font=font_lyrics, fill=color)
            y += 52
    return overlay


def render_video(
    project: ProjectData,
    palette: PaletteInfo,
    duration: float,
    output_path: Path,
    settings: RenderSettings,
    progress_callback=None,
) -> str:
    logger.info("Старт рендера видео в файл: %s", output_path)
    _ensure_ffmpeg_available()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height, fps = settings.width, settings.height, settings.fps
    total_frames = int(duration * fps)
    lines, start_times = prepare_timeline(project.lyrics)
    layout = compute_layout(width, height)

    codec, codec_reason = _resolve_video_codec(settings)
    logger.info("Выбран видеокодек: %s (%s)", codec, codec_reason)
    logger.info(
        "Параметры рендера: %dx%d, fps=%d, длительность=%.2fs, кадров=%d",
        width,
        height,
        fps,
        duration,
        total_frames,
    )

    cover = Image.open(project.image_path).convert("RGB")
    cover = cover.resize((layout.cover_box[2] - layout.cover_box[0], layout.cover_box[3] - layout.cover_box[1]))

    font_artist = _load_font(58)
    font_title = _load_font(52)
    font_lyrics = _load_font(46)
    font_date = _load_font(36)

    static_overlay = _build_static_overlay(project, layout, width, height, cover, font_artist, font_title, font_date)

    lx1, ly1, lx2, ly2 = layout.lyrics_box
    lyrics_width = lx2 - lx1
    lyrics_height = ly2 - ly1
    cached_idx: int | None = None
    cached_lyrics_overlay: Image.Image | None = None

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-i",
        str(project.audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        codec,
    ]

    if codec == settings.video_codec_hw:
        cmd.extend(["-preset", settings.nvenc_preset, "-cq", str(settings.nvenc_cq), "-b:v", "0"])
    else:
        cmd.extend(["-preset", settings.x264_preset, "-crf", str(settings.x264_crf), "-threads", "0"])

    cmd.extend(
        [
            "-pix_fmt",
            settings.pixel_format,
            "-c:a",
            settings.audio_codec,
            "-shortest",
            str(output_path),
        ]
    )

    logger.info("Старт ffmpeg pipe: %s", " ".join(cmd))
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    render_started = time.perf_counter()
    try:
        assert process.stdin is not None
        logger.info("Начало записи кадров в ffmpeg stdin")
        for i in range(total_frames):
            t = i / fps
            frame = Image.fromarray(build_background_frame(t, width, height, palette)).convert("RGBA")
            frame.alpha_composite(static_overlay)

            current_idx = active_line_index_precomputed(start_times, t)
            if cached_lyrics_overlay is None or current_idx != cached_idx:
                cached_idx = current_idx
                cached_lyrics_overlay = _build_lyrics_overlay(lines, current_idx, lyrics_width, lyrics_height, font_lyrics)
            frame.alpha_composite(cached_lyrics_overlay, (lx1, ly1))

            process.stdin.write(frame.convert("RGB").tobytes())

            if progress_callback and (i % max(1, fps // 2) == 0):
                progress_callback(int(i / total_frames * 100))

        logger.info("Завершение записи кадров, закрытие stdin")
        process.stdin.close()
        stderr_text = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        elapsed = time.perf_counter() - render_started
        fps_actual = (total_frames / elapsed) if elapsed > 0 else 0.0
        logger.info("ffmpeg завершен с кодом: %s, фактическая скорость=%.2f fps", return_code, fps_actual)
        if return_code != 0:
            logger.error("ffmpeg ошибка: %s", stderr_text[-2000:])
            raise RuntimeError(f"Ошибка ffmpeg: {stderr_text[-1200:]}")
    finally:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        if process.stderr:
            process.stderr.close()

    if progress_callback:
        progress_callback(100)
    logger.info("Рендер завершен успешно: %s", output_path)
    return codec
