from __future__ import annotations

import math

import numpy as np

from core.audio_analysis import BeatAnalysisResult
from models import PaletteInfo


def _soft_gradient_frame(t: float, width: int, height: int, palette: PaletteInfo) -> np.ndarray:
    colors = [c.rgb for c in palette.colors[:4]] or [(20, 20, 20), (70, 70, 120)]
    weights = [max(0.05, c.ratio) for c in palette.colors[:4]] or [0.5, 0.5]
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum()

    y, x = np.mgrid[0:height, 0:width]
    accum = np.zeros((height, width, 3), dtype=np.float32)
    norm = np.zeros((height, width, 1), dtype=np.float32)

    for i, color in enumerate(colors):
        cx = width * (0.2 + 0.6 * (0.5 + 0.5 * math.sin(t * 0.08 + i * 1.2)))
        cy = height * (0.2 + 0.6 * (0.5 + 0.5 * math.cos(t * 0.07 + i * 1.6)))
        sigma = min(width, height) * (0.25 + 0.1 * math.sin(t * 0.03 + i))
        blob = np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma**2))).astype(np.float32)
        blob = (blob * weights[i % len(weights)]).reshape(height, width, 1)
        accum += blob * np.array(color, dtype=np.float32).reshape(1, 1, 3)
        norm += blob

    frame = accum / np.maximum(norm, 1e-6)
    vignette = 0.9 - 0.15 * (((x - width / 2) / width) ** 2 + ((y - height / 2) / height) ** 2)
    frame *= vignette[..., None]
    return np.clip(frame, 0, 255).astype(np.uint8)


def _bpm_dynamic_frame(
    t: float,
    width: int,
    height: int,
    palette: PaletteInfo,
    beat_result: BeatAnalysisResult | None,
) -> np.ndarray:
    colors = [c.rgb for c in palette.colors] or [(28, 28, 36), (90, 100, 140), (150, 95, 80)]
    base_color = np.array(colors[0], dtype=np.float32)
    flash_colors = colors[1:] or [tuple(min(255, c + 25) for c in colors[0])]

    frame = np.ones((height, width, 3), dtype=np.float32)
    frame *= base_color.reshape(1, 1, 3)

    if not beat_result or not beat_result.beats:
        return np.clip(frame, 0, 255).astype(np.uint8)

    y, x = np.mgrid[0:height, 0:width]
    beat_times = beat_result.beats
    idx = np.searchsorted(beat_times, t) - 1
    if idx < 0:
        return np.clip(frame, 0, 255).astype(np.uint8)

    for beat_idx in (idx, idx - 1):
        if beat_idx < 0:
            continue
        delta = t - beat_times[beat_idx]
        if delta < 0 or delta > 0.45:
            continue

        fade = math.exp(-delta * 8.0)
        rnd = np.random.default_rng(seed=beat_idx + 1337)
        cx = rnd.uniform(0.1, 0.9) * width
        cy = rnd.uniform(0.15, 0.85) * height
        sigma = min(width, height) * rnd.uniform(0.12, 0.24)
        intensity = rnd.uniform(0.18, 0.33) * fade
        color = np.array(flash_colors[beat_idx % len(flash_colors)], dtype=np.float32)
        blob = np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma**2))).astype(np.float32)
        frame += blob[..., None] * color.reshape(1, 1, 3) * intensity

    vignette = 0.93 - 0.17 * (((x - width / 2) / width) ** 2 + ((y - height / 2) / height) ** 2)
    frame *= vignette[..., None]
    return np.clip(frame, 0, 255).astype(np.uint8)


def build_background_frame(
    t: float,
    width: int,
    height: int,
    palette: PaletteInfo,
    mode: str = "soft_gradient",
    beat_result: BeatAnalysisResult | None = None,
) -> np.ndarray:
    if mode == "bpm_dynamic":
        return _bpm_dynamic_frame(t, width, height, palette, beat_result)
    return _soft_gradient_frame(t, width, height, palette)
