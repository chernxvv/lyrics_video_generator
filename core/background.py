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


def _add_flash_blob(
    frame: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    cx: float,
    cy: float,
    sigma: float,
    color: np.ndarray,
    intensity: float,
) -> None:
    blob = np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma**2))).astype(np.float32)
    # Мягкое "осветление" + легкая цветовая подкраска.
    frame += (255.0 - frame) * blob[..., None] * (intensity * 0.62)
    frame += blob[..., None] * color.reshape(1, 1, 3) * (intensity * 0.25)


def _bpm_dynamic_frame(
    t: float,
    width: int,
    height: int,
    palette: PaletteInfo,
    beat_result: BeatAnalysisResult | None,
) -> np.ndarray:
    colors = [c.rgb for c in palette.colors] or [(28, 28, 36), (90, 100, 140), (150, 95, 80)]
    base_color = np.array(colors[0], dtype=np.float32)
    flash_colors = colors[1:] or [tuple(min(255, c + 30) for c in colors[0])]

    frame = np.ones((height, width, 3), dtype=np.float32)
    frame *= base_color.reshape(1, 1, 3)

    if not beat_result or not beat_result.beats:
        return np.clip(frame, 0, 255).astype(np.uint8)

    y, x = np.mgrid[0:height, 0:width]
    beat_times = beat_result.beats
    idx = np.searchsorted(beat_times, t) - 1
    if idx < 0:
        return np.clip(frame, 0, 255).astype(np.uint8)

    if idx + 1 < len(beat_times):
        beat_interval = max(0.24, beat_times[idx + 1] - beat_times[idx])
    elif idx > 0:
        beat_interval = max(0.24, beat_times[idx] - beat_times[idx - 1])
    else:
        beat_interval = 0.45

    phase = (t - beat_times[idx]) / beat_interval
    phase = max(0.0, min(1.25, phase))
    # Явный пик в начале бита + плавное затухание до следующего бита.
    peak = math.exp(-((phase - 0.06) ** 2) / 0.018)
    tail = math.exp(-phase * 2.6)
    beat_envelope = min(1.0, 0.85 * peak + 0.6 * tail)

    for beat_idx in (idx, idx - 1):
        if beat_idx < 0 or beat_idx >= len(beat_times):
            continue

        delta = t - beat_times[beat_idx]
        if delta < -0.05 or delta > min(0.52, beat_interval * 1.2):
            continue

        rnd = np.random.default_rng(seed=beat_idx + 7331)
        cx = rnd.uniform(0.12, 0.88) * width
        cy = rnd.uniform(0.14, 0.86) * height
        # Чуть меньшие вспышки, чтобы не перекрывали полэкрана.
        sigma = min(width, height) * rnd.uniform(0.085, 0.16)
        color = np.array(flash_colors[beat_idx % len(flash_colors)], dtype=np.float32)

        if delta <= 0:
            flash_fade = math.exp(delta * 9.0)
        else:
            flash_fade = math.exp(-delta * 6.2)

        intensity = rnd.uniform(0.42, 0.68) * beat_envelope * flash_fade
        _add_flash_blob(frame, x, y, cx=cx, cy=cy, sigma=sigma, color=color, intensity=intensity)

        # Пост-эффект: вторичная более слабая вспышка на том же месте.
        post_delta = delta - 0.08
        if post_delta >= 0:
            post_fade = math.exp(-post_delta * 9.0)
            post_intensity = intensity * 0.42 * post_fade
            if post_intensity > 0.01:
                _add_flash_blob(
                    frame,
                    x,
                    y,
                    cx=cx,
                    cy=cy,
                    sigma=sigma * 1.08,
                    color=color,
                    intensity=post_intensity,
                )

    vignette = 0.93 - 0.14 * (((x - width / 2) / width) ** 2 + ((y - height / 2) / height) ** 2)
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
