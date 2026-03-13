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
    frame += (255.0 - frame) * blob[..., None] * (intensity * 0.58)
    frame += blob[..., None] * color.reshape(1, 1, 3) * (intensity * 0.24)


def _flash_color_for_source(source_stem: str, flash_colors: list[tuple[int, int, int]], fallback_idx: int) -> np.ndarray:
    mapping = {
        "drums": 0,
        "guitar": 1,
        "piano": 2,
        "other": 3,
        "bass": 4,
        "vocals": 5,
    }
    idx = mapping.get(source_stem, fallback_idx) % len(flash_colors)
    return np.array(flash_colors[idx], dtype=np.float32)


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

    if not beat_result:
        return np.clip(frame, 0, 255).astype(np.uint8)

    y, x = np.mgrid[0:height, 0:width]

    # Приоритет: событийная линия из multi-stem анализа (без скрещивания stem-ов).
    events = beat_result.events if beat_result.events else []
    if events:
        window = 0.55
        candidates = [ev for ev in events if -0.08 <= (t - ev.time) <= window]
        # Поддерживаем паузы: если событий нет в окне, оставляем базовый фон.
        if not candidates:
            vignette = 0.94 - 0.14 * (((x - width / 2) / width) ** 2 + ((y - height / 2) / height) ** 2)
            frame *= vignette[..., None]
            return np.clip(frame, 0, 255).astype(np.uint8)

        for idx, ev in enumerate(candidates[-4:]):
            delta = t - ev.time
            if delta < 0:
                fade = math.exp(delta * 8.0)
            else:
                fade = math.exp(-delta * 6.4)

            intensity = np.clip(ev.intensity * fade, 0.0, 1.7)
            chaos = np.clip(ev.chaos, 0.1, 1.8)
            vocal_mod = np.clip(ev.vocal_mod, 0.0, 1.0)

            seed_base = int(ev.time * 1000) + idx * 97
            rng = np.random.default_rng(seed=seed_base)
            anchor_x = rng.uniform(0.12, 0.88) * width
            anchor_y = rng.uniform(0.14, 0.86) * height
            sub_count = 1 + (1 if chaos > 0.8 else 0) + (1 if chaos > 1.25 else 0)

            for n in range(sub_count):
                sub = np.random.default_rng(seed=seed_base + n * 31)
                jitter = (0.012 + 0.05 * chaos) * min(width, height)
                cx = anchor_x + sub.uniform(-1.0, 1.0) * jitter
                cy = anchor_y + sub.uniform(-1.0, 1.0) * jitter
                sigma = min(width, height) * sub.uniform(0.068, 0.14)

                color = _flash_color_for_source(ev.source_stem, flash_colors, idx + n)
                local_intensity = intensity * sub.uniform(0.34, 0.58) * (1.0 + 0.20 * vocal_mod)
                _add_flash_blob(frame, x, y, cx=cx, cy=cy, sigma=sigma, color=color, intensity=local_intensity)

                # Afterglow на том же месте.
                post_delta = delta - (0.075 + 0.015 * n)
                if post_delta >= 0:
                    post_fade = math.exp(-post_delta * (8.8 + 1.2 * n))
                    post_intensity = local_intensity * (0.36 - 0.05 * n) * post_fade
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
    else:
        # Legacy fallback if no events available.
        beat_times = beat_result.beats
        if not beat_times:
            return np.clip(frame, 0, 255).astype(np.uint8)

        idx = np.searchsorted(beat_times, t) - 1
        if idx < 0:
            return np.clip(frame, 0, 255).astype(np.uint8)

        delta = t - beat_times[idx]
        if -0.05 <= delta <= 0.40:
            fade = math.exp(-max(0.0, delta) * 6.0)
            color = np.array(flash_colors[idx % len(flash_colors)], dtype=np.float32)
            rng = np.random.default_rng(seed=idx + 7331)
            cx = rng.uniform(0.15, 0.85) * width
            cy = rng.uniform(0.18, 0.82) * height
            sigma = min(width, height) * 0.11
            _add_flash_blob(frame, x, y, cx=cx, cy=cy, sigma=sigma, color=color, intensity=0.55 * fade)

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
