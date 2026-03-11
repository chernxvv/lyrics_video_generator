from __future__ import annotations

import logging
import shutil
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from textwrap import wrap

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


def _ensure_ffmpeg_available() -> None:
    logger.info("Проверка доступности ffmpeg")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
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
) -> tuple[int, list[bytes]]:
    lx1, ly1 = lyrics_pos
    frames: list[bytes] = []
    for frame_index in range(chunk_start, chunk_end):
        t = frame_index / fps
        frame = Image.fromarray(build_background_frame(t, width, height, palette)).convert("RGBA")
        current_idx = active_line_index_precomputed(start_times, t)
        frame.alpha_composite(lyrics_overlays[current_idx], (lx1, ly1))
        frames.append(frame.convert("RGB").tobytes())
    return chunk_start, frames


def _build_filter_complex(project: ProjectData, layout, use_cuda: bool) -> str:
    cover_w = layout.cover_box[2] - layout.cover_box[0]
    cover_h = layout.cover_box[3] - layout.cover_box[1]
    cover_x = layout.cover_box[0]
    cover_y = layout.cover_box[1]

    artist = _escape_drawtext(project.artist)
    title = _escape_drawtext(project.title)
    release_date = _escape_drawtext(project.release_date)

    if use_cuda:
        cover_chain = (
            f"[1:v]format=rgba,hwupload_cuda,scale_cuda={cover_w}:{cover_h},"
            f"hwdownload,format=rgba[cover]"
        )
    else:
        cover_chain = f"[1:v]scale={cover_w}:{cover_h}[cover]"

    return (
        f"{cover_chain};"
        f"[0:v][cover]overlay={cover_x}:{cover_y}[v1];"
        f"[v1]drawtext=text='{artist}':x=(w-text_w)/2:y={layout.artist_y}:fontsize=58:fontcolor=white,"
        f"drawtext=text='—':x=(w-text_w)/2:y={layout.dash_y}:fontsize=58:fontcolor=white,"
        f"drawtext=text='{title}':x=(w-text_w)/2:y={layout.title_y}:fontsize=52:fontcolor=white,"
        f"drawtext=text='{release_date}':x=(w-text_w)/2:y={layout.date_y}:fontsize=36:fontcolor=white[vout]"
    )


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

    thread_count = max(1, settings.thread_count)
    chunk_size = max(1, settings.frame_chunk_size)

    codec, codec_reason = _resolve_video_codec(settings)
    use_cuda_filters = codec == settings.video_codec_hw and _supports_filter("scale_cuda") and _supports_filter("hwupload_cuda")
    logger.info("Выбран видеокодек: %s (%s)", codec, codec_reason)
    logger.info("CUDA filtergraph: %s", "enabled" if use_cuda_filters else "disabled")
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

    font_lyrics = _load_font(46)
    lx1, ly1, lx2, ly2 = layout.lyrics_box
    lyrics_width = lx2 - lx1
    lyrics_height = ly2 - ly1
    lyrics_overlays = {idx: _build_lyrics_overlay(lines, idx, lyrics_width, lyrics_height, font_lyrics) for idx in range(-1, len(lines))}

    filter_complex = _build_filter_complex(project, layout, use_cuda_filters)

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

    if use_cuda_filters:
        cmd[1:1] = ["-init_hw_device", "cuda=gpu:0", "-filter_hw_device", "gpu"]

    if codec == settings.video_codec_hw:
        cmd.extend(["-preset", settings.nvenc_preset, "-cq", str(settings.nvenc_cq), "-b:v", "0"])
    else:
        cmd.extend(["-preset", settings.x264_preset, "-crf", str(settings.x264_crf), "-threads", "0"])

    cmd.extend(["-pix_fmt", settings.pixel_format, "-c:a", settings.audio_codec, "-shortest", str(output_path)])

    logger.info("Старт ffmpeg pipe: %s", " ".join(cmd))
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    render_started = time.perf_counter()
    chunks = [(i, min(total_frames, i + chunk_size)) for i in range(0, total_frames, chunk_size)]

    try:
        assert process.stdin is not None
        logger.info("Начало многопоточной генерации кадров")
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            pending: dict[Future, int] = {}
            ready_chunks: dict[int, list[bytes]] = {}
            next_submit = 0
            next_write = 0
            written_frames = 0
            max_pending = max(1, thread_count * 2)

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
                    (lx1, ly1),
                )
                pending[fut] = start_idx
                next_submit += 1

            while pending:
                done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    pending.pop(fut)
                    chunk_start, chunk_frames = fut.result()
                    ready_chunks[chunk_start] = chunk_frames

                while next_write in ready_chunks:
                    chunk_frames = ready_chunks.pop(next_write)
                    for frame_bytes in chunk_frames:
                        process.stdin.write(frame_bytes)
                    written_frames += len(chunk_frames)
                    if progress_callback:
                        progress_callback(int(written_frames / total_frames * 100))
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
                        (lx1, ly1),
                    )
                    pending[fut] = start_idx
                    next_submit += 1

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
