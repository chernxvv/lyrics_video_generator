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
    frame += (255.0 - frame) * blob[..., None] * (intensity * 0.60)
    frame += blob[..., None] * color.reshape(1, 1, 3) * (intensity * 0.23)


def _safe_pick(values: list[float], idx: int, default: float) -> float:
    if idx < 0 or idx >= len(values):
        return default
    return float(values[idx])


def _tempo_change_factor(beat_times: list[float], idx: int) -> float:
    if idx < 2 or idx >= len(beat_times):
        return 0.0
    prev_interval = max(1e-4, beat_times[idx - 1] - beat_times[idx - 2])
    curr_interval = max(1e-4, beat_times[idx] - beat_times[idx - 1])
    return (prev_interval - curr_interval) / prev_interval


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
        beat_interval = max(0.21, beat_times[idx + 1] - beat_times[idx])
    elif idx > 0:
        beat_interval = max(0.21, beat_times[idx] - beat_times[idx - 1])
    else:
        beat_interval = 0.45

    phase = (t - beat_times[idx]) / beat_interval
    phase = max(0.0, min(1.35, phase))

    peak = math.exp(-((phase - 0.05) ** 2) / 0.013)
    tail = math.exp(-phase * 2.9)
    beat_envelope = min(1.1, 0.95 * peak + 0.62 * tail)

    strength = _safe_pick(beat_result.beat_strengths, idx, 0.5)
    percussion = _safe_pick(beat_result.percussion_energy, idx, 0.5)
    local_tempo = _safe_pick(beat_result.local_tempo, idx, max(80.0, beat_result.bpm))
    tempo_change = _tempo_change_factor(beat_times, idx)

    # Чем выше перкуссия/сила бита/локальный темп, тем ярче и хаотичнее поведение.
    tempo_boost = min(1.25, max(0.8, local_tempo / max(60.0, beat_result.bpm or 120.0)))
    dynamics = 0.35 + 0.40 * strength + 0.35 * percussion + 0.25 * max(0.0, tempo_change)
    dynamics *= tempo_boost

    extra_flashes = 1
    if dynamics > 0.95:
        extra_flashes = 2
    if dynamics > 1.20:
        extra_flashes = 3

    for beat_idx in (idx, idx - 1):
        if beat_idx < 0 or beat_idx >= len(beat_times):
            continue

        delta = t - beat_times[beat_idx]
        if delta < -0.05 or delta > min(0.52, beat_interval * 1.4):
            continue

        rnd = np.random.default_rng(seed=beat_idx + 7331)
        anchor_x = rnd.uniform(0.12, 0.88) * width
        anchor_y = rnd.uniform(0.14, 0.86) * height

        if delta <= 0:
            flash_fade = math.exp(delta * 9.5)
        else:
            flash_fade = math.exp(-delta * (6.0 + 1.8 * percussion))

        for n in range(extra_flashes):
            sub = np.random.default_rng(seed=beat_idx * 101 + n * 17 + 9)
            jitter = (0.012 + 0.045 * dynamics) * min(width, height)
            cx = anchor_x + sub.uniform(-1.0, 1.0) * jitter
            cy = anchor_y + sub.uniform(-1.0, 1.0) * jitter
            sigma = min(width, height) * sub.uniform(0.072, 0.145)
            color = np.array(flash_colors[(beat_idx + n) % len(flash_colors)], dtype=np.float32)

            intensity = sub.uniform(0.34, 0.56) * beat_envelope * flash_fade * (0.85 + 0.45 * dynamics)
            _add_flash_blob(frame, x, y, cx=cx, cy=cy, sigma=sigma, color=color, intensity=intensity)

            post_delta = delta - (0.07 + 0.015 * n)
            if post_delta >= 0:
                post_fade = math.exp(-post_delta * (8.2 + 1.1 * n))
                post_intensity = intensity * (0.34 - 0.05 * n) * post_fade
                if post_intensity > 0.01:
                    _add_flash_blob(
                        frame,
                        x,
                        y,
                        cx=cx,
                        cy=cy,
                        sigma=sigma * (1.05 + 0.03 * n),
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
