# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Roman Chernov (romanchernovv@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from core.audio_analysis import BeatAnalysisResult, analyze_bpm_and_beats
from core.background import build_background_frame
from core.layout import compute_layout, get_base_canvas
from core.lyrics import active_line_index_precomputed, prepare_timeline
from models import PaletteInfo, ProjectData, RenderSettings, VideoOrientation

logger = logging.getLogger(__name__)


class RenderError(RuntimeError):
    pass


class RenderDependencyError(RenderError):
    pass

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
LYRICS_FONT_REGULAR_FILE = "NotoSerif-Regular.ttf"
LYRICS_FONT_BOLD_FILE = "NotoSerif-Bold.ttf"
META_FONT_FILE = "NotoSerif-Regular.ttf"


def _require_font_path(filename: str) -> Path:
    font_path = FONT_DIR / filename
    if not font_path.exists():
        raise RenderError(
            "Не найден обязательный шрифт: "
            f"{font_path}. "
            "Создайте папку assets/fonts и положите туда нужные .ttf файлы: "
            f"{LYRICS_FONT_REGULAR_FILE}, {LYRICS_FONT_BOLD_FILE}."
        )
    return font_path


def _load_font(size: int, filename: str) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = _require_font_path(filename)
    try:
        return ImageFont.truetype(str(font_path), size=size)
    except OSError as exc:
        raise RenderError(f"Не удалось загрузить шрифт {font_path}: {exc}") from exc




def _ensure_ffmpeg_available() -> None:
    logger.info("Проверка доступности ffmpeg")
    if shutil.which("ffmpeg") is None:
        raise RenderDependencyError(
            "Не найден ffmpeg в PATH. Установите FFmpeg и добавьте ffmpeg в PATH перед генерацией видео."
        )


def _ffmpeg_probe_output(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout


def _supports_encoder(encoder_name: str) -> bool:
    output = _ffmpeg_probe_output(["ffmpeg", "-hide_banner", "-encoders"])
    return encoder_name in output


def _supports_filter(filter_name: str) -> bool:
    output = _ffmpeg_probe_output(["ffmpeg", "-hide_banner", "-filters"])
    return filter_name in output


def _probe_cuda_runtime(image_path: Path) -> tuple[bool, str]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-vf",
        "format=nv12,hwupload_cuda,scale_cuda=16:16:format=nv12,hwdownload,format=nv12",
        "-frames:v",
        "1",
        "-f",
        "null",
        "-",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=8)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    if res.returncode == 0:
        return True, "ok"
    err = (res.stderr or "").strip()
    return False, (err[-300:] if err else f"returncode={res.returncode}")


def _resolve_video_codec(settings: RenderSettings) -> tuple[str, str]:
    if not settings.prefer_hw_encode:
        return settings.video_codec_sw, "hardware encoding отключен в настройках"
    if _supports_encoder(settings.video_codec_hw):
        return settings.video_codec_hw, "обнаружена поддержка NVENC"
    return settings.video_codec_sw, "NVENC недоступен в ffmpeg -encoders"


def _escape_drawtext(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("%", "\\%")
        .replace("\n", " ")
    )




def _drawtext_style(fontsize: int, font_filename: str, *, bordered: bool = False) -> str:
    fontfile = str(_require_font_path(font_filename)).replace("\\", "/")
    border = ":borderw=1:bordercolor=black" if bordered else ""
    return f"fontfile='{_escape_drawtext(fontfile)}':fontsize={fontsize}:fontcolor=white{border}"


def _compute_uniform_scale(base_width: int, base_height: int, out_width: int, out_height: int) -> float:
    if base_width <= 0 or base_height <= 0:
        raise RenderError("Некорректный базовый canvas: ширина/высота должны быть > 0")
    return min(out_width / base_width, out_height / base_height)


def _scale_value(value: int, scale: float, minimum: int = 1) -> int:
    return max(minimum, int(round(value * scale)))


def _scale_box(box: tuple[int, int, int, int], scale: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    sx1 = _scale_value(x1, scale, minimum=0)
    sy1 = _scale_value(y1, scale, minimum=0)
    sx2 = _scale_value(x2, scale, minimum=sx1 + 1)
    sy2 = _scale_value(y2, scale, minimum=sy1 + 1)
    return (sx1, sy1, sx2, sy2)


def _lyrics_spacing(scale_factor: float) -> tuple[int, int]:
    line_gap = _scale_value(8, scale_factor)
    block_gap = _scale_value(16, scale_factor)
    return line_gap, block_gap


def _wrap_text_by_pixel_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    stroke_width: int,
) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        candidate_width = draw.textbbox((0, 0), candidate, font=font, stroke_width=stroke_width)[2]
        if candidate_width <= max_width:
            current = candidate
            continue
        lines.append(current)
        current = word
    lines.append(current)
    return lines


def _line_height(draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, stroke_width: int) -> int:
    bbox = draw.textbbox((0, 0), "Ag", font=font, stroke_width=stroke_width)
    return bbox[3] - bbox[1]


def _draw_wrapped_block(
    draw: ImageDraw.ImageDraw,
    text: str,
    y: int,
    width: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    color: tuple[int, int, int, int],
    stroke_width: int = 0,
    line_gap: int = 8,
) -> tuple[int, int]:
    wrapped = _wrap_text_by_pixel_width(draw, text, font, max_width=max(80, width - 24), stroke_width=stroke_width)
    h = _line_height(draw, font, stroke_width)
    for part in wrapped:
        text_bbox = draw.textbbox((0, 0), part, font=font, stroke_width=stroke_width)
        text_width = text_bbox[2] - text_bbox[0]
        x = max(0, (width - text_width) // 2)
        if stroke_width > 0:
            draw.text((x, y), part, font=font, fill=color, stroke_width=stroke_width, stroke_fill=(0, 0, 0, 255))
        else:
            draw.text((x, y), part, font=font, fill=color)
        y += h + line_gap
    return y, len(wrapped)


def _measure_wrapped_height(
    draw: ImageDraw.ImageDraw,
    text: str,
    width: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    stroke_width: int = 0,
    line_gap: int = 8,
) -> int:
    wrapped = _wrap_text_by_pixel_width(draw, text, font, max_width=max(80, width - 24), stroke_width=stroke_width)
    h = _line_height(draw, font, stroke_width)
    return len(wrapped) * h + max(0, len(wrapped) - 1) * line_gap


def _iter_next_lyric_indices(current_index: int, total_lines: int, orientation: VideoOrientation) -> range:
    if orientation == "vertical":
        return range(current_index + 1, min(total_lines, current_index + 2))
    return range(current_index + 1, total_lines)


def _build_lyrics_overlay(
    lines,
    current_index: int,
    width: int,
    height: int,
    font_lyrics_regular,
    font_lyrics_bold,
    orientation: VideoOrientation,
    line_gap: int,
    block_gap: int,
    active_center_target_y: int | None = None,
) -> Image.Image:
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if current_index < 0 or current_index >= len(lines):
        return overlay

    draw = ImageDraw.Draw(overlay)

    active_text = lines[current_index].text
    active_h = _measure_wrapped_height(draw, active_text, width, font_lyrics_bold, line_gap=line_gap)

    if active_center_target_y is None:
        active_center_target_y = height // 2
    min_top = 0
    max_top = max(0, height - active_h)
    center_y = min(max(active_center_target_y - (active_h // 2), min_top), max_top)

    side_color = (215, 215, 215, 255)

    prev_idx = current_index - 1
    if prev_idx >= 0:
        prev_h = _measure_wrapped_height(draw, lines[prev_idx].text, width, font_lyrics_regular, line_gap=line_gap)
        prev_y = center_y - block_gap - prev_h
        if prev_y >= 0:
            _draw_wrapped_block(
                draw,
                lines[prev_idx].text,
                prev_y,
                width,
                font_lyrics_regular,
                side_color,
                line_gap=line_gap,
            )

    active_bottom, _ = _draw_wrapped_block(
        draw,
        active_text,
        center_y,
        width,
        font_lyrics_bold,
        (255, 255, 255, 255),
        line_gap=line_gap,
    )

    next_y = active_bottom + block_gap
    for idx in _iter_next_lyric_indices(current_index, len(lines), orientation):
        next_h = _measure_wrapped_height(draw, lines[idx].text, width, font_lyrics_regular, line_gap=line_gap)
        if next_y + next_h > height:
            break
        next_y, _ = _draw_wrapped_block(
            draw,
            lines[idx].text,
            next_y,
            width,
            font_lyrics_regular,
            side_color,
            line_gap=line_gap,
        )
        next_y += block_gap

    return overlay


def _render_chunk(
    chunk_start: int,
    chunk_end: int,
    fps: int,
    width: int,
    height: int,
    palette: PaletteInfo,
    lyrics_overlays: dict[int, Image.Image],
    start_times: list[float],
    lyrics_pos: tuple[int, int],
    background_mode: str,
    beat_result: BeatAnalysisResult | None,
) -> tuple[int, list[bytes]]:
    lx1, ly1 = lyrics_pos
    frames: list[bytes] = []
    for frame_index in range(chunk_start, chunk_end):
        t = frame_index / fps
        frame = Image.fromarray(
            build_background_frame(
                t,
                width,
                height,
                palette,
                mode=background_mode,
                beat_result=beat_result,
            )
        ).convert("RGBA")
        frame = frame.filter(ImageFilter.GaussianBlur(radius=5))
        current_idx = active_line_index_precomputed(start_times, t)
        frame.alpha_composite(lyrics_overlays[current_idx], (lx1, ly1))
        frames.append(frame.convert("RGB").tobytes())
    return chunk_start, frames


def _build_filter_complex(project: ProjectData, layout, use_cuda: bool, scale_factor: float) -> str:
    scaled_cover_box = _scale_box(layout.cover_box, scale_factor)
    cover_w = scaled_cover_box[2] - scaled_cover_box[0]
    cover_h = scaled_cover_box[3] - scaled_cover_box[1]
    cover_box_x = scaled_cover_box[0]
    cover_box_y = scaled_cover_box[1]

    artist = _escape_drawtext(project.artist)
    title = _escape_drawtext(project.title)
    release_date_raw = project.release_date.strip()
    release_date = _escape_drawtext(release_date_raw) if release_date_raw else ""

    if use_cuda:
        logger.info("Выбрана ветка filter_complex: CUDA (hwupload_cuda/scale_cuda/hwdownload)")
        cover_chain = (
            f"[1:v]format=nv12,hwupload_cuda,"
            f"scale_cuda={cover_w}:{cover_h}:force_original_aspect_ratio=increase:format=nv12,"
            f"hwdownload,format=nv12,crop={cover_w}:{cover_h}:(in_w-{cover_w})/2:(in_h-{cover_h})/2[cover]"
        )
    else:
        logger.info("Выбрана ветка filter_complex: CPU (scale/crop)")
        cover_chain = (
            f"[1:v]scale={cover_w}:{cover_h}:force_original_aspect_ratio=increase,"
            f"crop={cover_w}:{cover_h}:(in_w-{cover_w})/2:(in_h-{cover_h})/2[cover]"
        )

    if project.orientation == "horizontal":
        artist_base_size = 37
        title_base_size = 32
        date_base_size = 24
    else:
        artist_base_size = 58
        title_base_size = 52
        date_base_size = 36

    artist_size = _scale_value(artist_base_size, scale_factor)
    title_size = _scale_value(title_base_size, scale_factor)
    date_size = _scale_value(date_base_size, scale_factor)
    separator_h = _scale_value(3, scale_factor)
    min_separator_margin = _scale_value(8, scale_factor)

    artist_y = _scale_value(layout.artist_y, scale_factor, minimum=0)
    title_y = _scale_value(layout.title_y, scale_factor, minimum=0)
    date_y = _scale_value(layout.date_y, scale_factor, minimum=0)

    artist_bottom_y = artist_y + artist_size
    title_top_y = title_y

    free_space = max(0, title_top_y - artist_bottom_y)
    centered_separator_y = artist_bottom_y + (free_space // 2) - (separator_h // 2)

    min_separator_y = artist_bottom_y + min_separator_margin
    max_separator_y = title_top_y - min_separator_margin - separator_h
    if max_separator_y < min_separator_y:
        separator_y = max(0, centered_separator_y)
    else:
        separator_y = min(max(centered_separator_y, min_separator_y), max_separator_y)

    separator_w = max(_scale_value(20, scale_factor), cover_w // 12)
    separator_x_expr = f"(iw-{separator_w})/2"

    logger.debug(
        "Separator line layout: artist_bottom=%d title_top=%d free_space=%d separator_y=%d separator_w=%d separator_x=%s",
        artist_bottom_y,
        title_top_y,
        free_space,
        separator_y,
        separator_w,
        separator_x_expr,
    )

    filter_chain = (
        f"{cover_chain};"
        f"[0:v]format=nv12[base];"
        f"[base][cover]overlay={cover_box_x}:{cover_box_y}[v1];"
        f"[v1]drawtext=text='{artist}':x=(w-text_w)/2:y={artist_y}:{_drawtext_style(artist_size, META_FONT_FILE)},"
        f"drawbox=x={separator_x_expr}:y={separator_y}:w={separator_w}:h={separator_h}:color=white@1:t=fill,"
        f"drawtext=text='{title}':x=(w-text_w)/2:y={title_y}:{_drawtext_style(title_size, META_FONT_FILE)}"
    )

    if release_date:
        filter_chain += (
            f",drawtext=text='{release_date}':x=(w-text_w)/2:y={date_y}:{_drawtext_style(date_size, META_FONT_FILE)}"
        )

    return f"{filter_chain}[vout]"


def _drain_stderr(stderr_pipe, sink: list[str], stop_event: threading.Event) -> None:
    if stderr_pipe is None:
        return
    while not stop_event.is_set():
        line = stderr_pipe.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace").rstrip()
        if text:
            sink.append(text)
            if len(sink) > 3000:
                del sink[:1500]
            logger.debug("ffmpeg: %s", text)


def _collect_stderr_text(stderr_lines: list[str]) -> str:
    return "\n".join(stderr_lines)


def _is_cuda_runtime_failure(stderr_text: str) -> bool:
    low = stderr_text.lower()
    markers = (
        "cuda",
        "hwupload_cuda",
        "scale_cuda",
        "cannot load nvcuda",
        "no device",
        "init_hw_device",
    )
    return any(m in low for m in markers)


def _render_stream_to_ffmpeg(
    cmd: list[str],
    total_frames: int,
    total_chunks: int,
    thread_count: int,
    chunk_size: int,
    fps: int,
    width: int,
    height: int,
    palette: PaletteInfo,
    lyrics_overlays: dict[int, Image.Image],
    start_times: list[float],
    lyrics_pos: tuple[int, int],
    background_mode: str,
    beat_result: BeatAnalysisResult | None,
    progress_callback: Callable[[int], None] | None,
) -> float:
    logger.info("Старт ffmpeg pipe: %s", " ".join(cmd))
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr_lines: list[str] = []
    stop_stderr = threading.Event()
    stderr_thread = threading.Thread(target=_drain_stderr, args=(process.stderr, stderr_lines, stop_stderr), daemon=True)
    stderr_thread.start()

    render_started = time.perf_counter()
    chunks = [(i, min(total_frames, i + chunk_size)) for i in range(0, total_frames, chunk_size)]

    try:
        assert process.stdin is not None
        logger.info("Начало многопоточной генерации кадров: чанков=%d", total_chunks)
        if progress_callback:
            progress_callback(0)

        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            pending: dict[Future, int] = {}
            ready_chunks: dict[int, list[bytes]] = {}
            next_submit = 0
            next_write = 0
            written_frames = 0
            written_chunks = 0
            max_pending = max(1, thread_count)
            last_heartbeat = time.perf_counter()

            while next_submit < len(chunks) and len(pending) < max_pending:
                start_idx, end_idx = chunks[next_submit]
                fut = executor.submit(
                    _render_chunk,
                    start_idx,
                    end_idx,
                    fps,
                    width,
                    height,
                    palette,
                    lyrics_overlays,
                    start_times,
                    lyrics_pos,
                    background_mode,
                    beat_result,
                )
                pending[fut] = start_idx
                logger.debug("Submit chunk %d/%d: frames %d..%d", next_submit + 1, total_chunks, start_idx, end_idx - 1)
                next_submit += 1

            while pending:
                if process.poll() is not None:
                    stderr_text = _collect_stderr_text(stderr_lines)
                    raise RenderError(f"ffmpeg завершился до окончания генерации кадров: {stderr_text[-1200:]}")

                done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED, timeout=1.0)
                if not done:
                    now = time.perf_counter()
                    if now - last_heartbeat >= 2.0:
                        logger.debug(
                            "Render progress heartbeat: written_frames=%d/%d, written_chunks=%d/%d, pending=%d, ready=%d",
                            written_frames,
                            total_frames,
                            written_chunks,
                            total_chunks,
                            len(pending),
                            len(ready_chunks),
                        )
                        last_heartbeat = now
                    continue

                for fut in done:
                    pending.pop(fut)
                    chunk_start, chunk_frames = fut.result()
                    ready_chunks[chunk_start] = chunk_frames
                    logger.debug("Chunk ready: start=%d, size=%d", chunk_start, len(chunk_frames))

                while next_write in ready_chunks:
                    chunk_frames = ready_chunks.pop(next_write)
                    try:
                        for frame_bytes in chunk_frames:
                            process.stdin.write(frame_bytes)
                    except BrokenPipeError as exc:
                        stderr_text = _collect_stderr_text(stderr_lines)
                        raise RenderError(f"Broken pipe при записи в ffmpeg: {stderr_text[-1200:]}") from exc

                    written_frames += len(chunk_frames)
                    written_chunks += 1
                    progress = int(written_frames / total_frames * 100)
                    logger.info(
                        "Chunk written: %d/%d, frames=%d/%d, progress=%d%%",
                        written_chunks,
                        total_chunks,
                        written_frames,
                        total_frames,
                        progress,
                    )
                    if progress_callback:
                        progress_callback(progress)
                    next_write += chunk_size

                while next_submit < len(chunks) and len(pending) < max_pending:
                    start_idx, end_idx = chunks[next_submit]
                    fut = executor.submit(
                        _render_chunk,
                        start_idx,
                        end_idx,
                        fps,
                        width,
                        height,
                        palette,
                        lyrics_overlays,
                        start_times,
                        lyrics_pos,
                        background_mode,
                        beat_result,
                    )
                    pending[fut] = start_idx
                    logger.debug("Submit chunk %d/%d: frames %d..%d", next_submit + 1, total_chunks, start_idx, end_idx - 1)
                    next_submit += 1

        logger.info("Завершение записи кадров, закрытие stdin")
        process.stdin.close()
        return_code = process.wait()

        elapsed = time.perf_counter() - render_started
        fps_actual = (total_frames / elapsed) if elapsed > 0 else 0.0
        logger.info("ffmpeg завершен с кодом: %s, фактическая скорость=%.2f fps", return_code, fps_actual)

        if return_code != 0:
            stderr_text = _collect_stderr_text(stderr_lines)
            logger.error("ffmpeg ошибка: %s", stderr_text[-2000:])
            raise RenderError(f"Ошибка ffmpeg: {stderr_text[-1200:]}")

        return fps_actual
    finally:
        stop_stderr.set()
        stderr_thread.join(timeout=2.0)
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        if process.stderr and not process.stderr.closed:
            process.stderr.close()


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
    if total_frames <= 0:
        raise RenderError("Ошибка рендера: длительность слишком мала, кадров=0")

    lines, start_times = prepare_timeline(project.lyrics)
    base_width, base_height = get_base_canvas(project.orientation)
    layout = compute_layout(width, height, project.orientation)
    scale_factor = _compute_uniform_scale(base_width, base_height, width, height)

    thread_count = max(1, settings.thread_count)
    chunk_size = max(1, settings.frame_chunk_size)

    codec, codec_reason = _resolve_video_codec(settings)
    has_cuda_filters = _supports_filter("scale_cuda") and _supports_filter("hwupload_cuda")
    use_cuda_filters = False
    cuda_reason = "disabled"
    if codec == settings.video_codec_hw and has_cuda_filters:
        probe_ok, probe_reason = _probe_cuda_runtime(Path(project.image_path))
        use_cuda_filters = probe_ok
        cuda_reason = "runtime-probe ok" if probe_ok else f"runtime-probe failed: {probe_reason}"

    logger.info("Выбран видеокодек: %s (%s)", codec, codec_reason)
    logger.info("CUDA filtergraph: %s (%s)", "enabled" if use_cuda_filters else "disabled", cuda_reason)
    effective_background_mode = project.background_mode

    logger.info(
        "Выбраны параметры сцены: orientation=%s, background_mode=%s",
        project.orientation,
        effective_background_mode,
    )
    logger.info(
        "Ключевые layout-параметры (base): cover_box=%s, lyrics_box=%s, artist_y=%d, title_y=%d",
        layout.cover_box,
        layout.lyrics_box,
        layout.artist_y,
        layout.title_y,
    )
    logger.info("Базовый canvas=%dx%d, scale_factor=%.4f", base_width, base_height, scale_factor)
    logger.info(
        "Параметры рендера: %dx%d, fps=%d, длительность=%.2fs, кадров=%d, threads=%d, chunk=%d",
        width,
        height,
        fps,
        duration,
        total_frames,
        thread_count,
        chunk_size,
    )

    lyrics_regular_size = _scale_value(25, scale_factor)
    lyrics_bold_size = _scale_value(28, scale_factor)
    font_lyrics_regular = _load_font(lyrics_regular_size, LYRICS_FONT_REGULAR_FILE)
    font_lyrics_bold = _load_font(lyrics_bold_size, LYRICS_FONT_BOLD_FILE)
    lx1, ly1, lx2, ly2 = _scale_box(layout.lyrics_box, scale_factor)
    lyrics_width = lx2 - lx1
    lyrics_height = ly2 - ly1
    lyrics_line_gap, lyrics_block_gap = _lyrics_spacing(scale_factor)
    frame_center_in_lyrics = (height // 2) - ly1
    lyrics_overlays = {
        idx: _build_lyrics_overlay(
            lines,
            idx,
            lyrics_width,
            lyrics_height,
            font_lyrics_regular,
            font_lyrics_bold,
            project.orientation,
            lyrics_line_gap,
            lyrics_block_gap,
            active_center_target_y=frame_center_in_lyrics,
        )
        for idx in range(-1, len(lines))
    }

    filter_complex = _build_filter_complex(project, layout, use_cuda=use_cuda_filters, scale_factor=scale_factor)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostats",
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
        "-loop",
        "1",
        "-i",
        str(project.image_path),
        "-i",
        str(project.audio_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-map",
        "2:a:0",
        "-c:v",
        codec,
    ]


    if codec == settings.video_codec_hw:
        cmd.extend(["-preset", settings.nvenc_preset, "-cq", str(settings.nvenc_cq), "-b:v", "0"])
    else:
        cmd.extend(["-preset", settings.x264_preset, "-crf", str(settings.x264_crf), "-threads", "0"])

    cmd.extend(["-pix_fmt", settings.pixel_format, "-c:a", settings.audio_codec, "-shortest", str(output_path)])

    total_chunks = len([(i, min(total_frames, i + chunk_size)) for i in range(0, total_frames, chunk_size)])

    beat_result: BeatAnalysisResult | None = None
    if effective_background_mode == "bpm_dynamic":
        try:
            beat_result = analyze_bpm_and_beats(str(project.audio_path), fps)
            logger.info(
                "BPM background: bpm=%.2f, beats=%d, confidence_low=%s",
                beat_result.bpm,
                len(beat_result.beats),
                beat_result.confidence_low,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("BPM-анализ не удался, fallback на мягкий фон: %s", exc)
            effective_background_mode = "soft_gradient"

    try:
        _render_stream_to_ffmpeg(
            cmd=cmd,
            total_frames=total_frames,
            total_chunks=total_chunks,
            thread_count=thread_count,
            chunk_size=chunk_size,
            fps=fps,
            width=width,
            height=height,
            palette=palette,
            lyrics_overlays=lyrics_overlays,
            start_times=start_times,
            lyrics_pos=(lx1, ly1),
            background_mode=effective_background_mode,
            beat_result=beat_result,
            progress_callback=progress_callback,
        )
    except RenderError as exc:
        err = str(exc)
        if use_cuda_filters and _is_cuda_runtime_failure(err):
            logger.warning("CUDA runtime-сбой, выполняется fallback на CPU filtergraph: %s", err[-400:])
            fallback_filter = _build_filter_complex(project, layout, use_cuda=False, scale_factor=scale_factor)
            fallback_cmd = cmd.copy()
            fc_idx = fallback_cmd.index("-filter_complex")
            fallback_cmd[fc_idx + 1] = fallback_filter
            _render_stream_to_ffmpeg(
                cmd=fallback_cmd,
                total_frames=total_frames,
                total_chunks=total_chunks,
                thread_count=thread_count,
                chunk_size=chunk_size,
                fps=fps,
                width=width,
                height=height,
                palette=palette,
                lyrics_overlays=lyrics_overlays,
                start_times=start_times,
                lyrics_pos=(lx1, ly1),
                background_mode=effective_background_mode,
                beat_result=beat_result,
                progress_callback=progress_callback,
            )
        else:
            raise

    if progress_callback:
        progress_callback(100)
    logger.info("Рендер завершен успешно: %s", output_path)
    return codec
