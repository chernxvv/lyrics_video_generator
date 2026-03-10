from __future__ import annotations

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import wrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.background import build_background_frame
from core.layout import compute_layout
from core.lyrics import active_line_index, sort_lyrics
from models import PaletteInfo, ProjectData, RenderSettings


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


def render_video(
    project: ProjectData,
    palette: PaletteInfo,
    duration: float,
    output_path: Path,
    settings: RenderSettings,
    progress_callback=None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height, fps = settings.width, settings.height, settings.fps
    total_frames = int(duration * fps)
    lines = sort_lyrics(project.lyrics)
    layout = compute_layout(width, height)

    cover = Image.open(project.image_path).convert("RGB")
    cover = cover.resize((layout.cover_box[2] - layout.cover_box[0], layout.cover_box[3] - layout.cover_box[1]))

    font_artist = _load_font(58)
    font_title = _load_font(52)
    font_lyrics = _load_font(46)
    font_date = _load_font(36)

    with TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "video.rgb"
        with raw_path.open("wb") as raw_file:
            for i in range(total_frames):
                t = i / fps
                bg = build_background_frame(t, width, height, palette)
                img = Image.fromarray(bg)
                draw = ImageDraw.Draw(img)

                _draw_centered(draw, project.artist, layout.artist_y, width, font_artist, (255, 255, 255))
                _draw_centered(draw, "—", layout.dash_y, width, font_artist, (255, 255, 255))
                _draw_centered(draw, project.title, layout.title_y, width, font_title, (255, 255, 255))

                img.paste(cover, (layout.cover_box[0], layout.cover_box[1]))

                lx1, ly1, lx2, ly2 = layout.lyrics_box
                overlay = Image.new("RGBA", (lx2 - lx1, ly2 - ly1), (0, 0, 0, 105))
                img.paste(overlay, (lx1, ly1), overlay)

                current = active_line_index(lines, t)
                visible = range(max(0, current - 2), min(len(lines), current + 3))
                y = ly1 + 22
                for idx in visible:
                    prefix = "▶ " if idx == current else ""
                    color = (255, 255, 255) if idx == current else (220, 220, 220)
                    text = prefix + lines[idx].text
                    for part in wrap(text, width=34):
                        draw.text((lx1 + 24, y), part, font=font_lyrics, fill=color)
                        y += 52
                _draw_centered(draw, project.release_date, layout.date_y, width, font_date, (245, 245, 245))

                raw_file.write(np.array(img, dtype=np.uint8).tobytes())

                if progress_callback and (i % max(1, fps // 2) == 0):
                    progress_callback(int(i / total_frames * 100))

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
            str(raw_path),
            "-i",
            str(project.audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            settings.video_codec,
            "-pix_fmt",
            settings.pixel_format,
            "-c:a",
            settings.audio_codec,
            "-shortest",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Ошибка ffmpeg: {result.stderr[-1200:]}")

    if progress_callback:
        progress_callback(100)
