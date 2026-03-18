from __future__ import annotations

from pathlib import Path

import dearpygui.dearpygui as dpg
import numpy as np

from core.image_analysis import extract_dominant_palette
from core.render import compose_preview_frame
from core.audio import probe_audio_duration


def ensure_preview_texture(tag: str, width: int, height: int) -> None:
    if dpg.does_item_exist(tag):
        return
    data = np.zeros((height, width, 4), dtype=np.float32)
    with dpg.texture_registry(show=False):
        dpg.add_dynamic_texture(width, height, data.flatten().tolist(), tag=tag)


def update_preview_texture(state) -> None:
    ensure_preview_texture(state.preview.texture_tag, state.preview.width, state.preview.height)
    image = None
    try:
        duration = probe_audio_duration(Path(state.project.audio_path)) if state.project.audio_path else 0.0
        palette = extract_dominant_palette(Path(state.project.image_path)) if state.project.image_path else None
        if palette is not None:
            image = compose_preview_frame(
                state.project,
                palette,
                duration,
                current_time=state.playback_position,
                width=state.preview.width,
                height=state.preview.height,
            )
    except Exception:
        image = None
    if image is None:
        image = np.zeros((state.preview.height, state.preview.width, 4), dtype=np.uint8)
        image[..., 0] = 20
        image[..., 1] = 24
        image[..., 2] = 32
        image[..., 3] = 255
    else:
        image = np.array(image.convert("RGBA"), dtype=np.uint8)
    dpg.set_value(state.preview.texture_tag, (image.astype(np.float32) / 255.0).flatten().tolist())
    state.preview.dirty = False
