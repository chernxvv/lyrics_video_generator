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


def _color_from_index(color_index: int, flash_colors: list[tuple[int, int, int]]) -> np.ndarray:
    return np.array(flash_colors[color_index % len(flash_colors)], dtype=np.float32)


def _smoothed_chaos(events, t: float) -> float:
    if not events:
        return 0.8
    weighted = 0.0
    wsum = 0.0
    for ev in events:
        dist = abs(t - ev.time)
        if dist > 0.9:
            continue
        w = math.exp(-dist / 0.22)
        weighted += float(ev.chaos) * w
        wsum += w
    if wsum <= 1e-6:
        return float(events[-1].chaos)
    return float(weighted / wsum)


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

    events = beat_result.events if beat_result.events else []
    if events:
        candidates = [ev for ev in events if -0.08 <= (t - ev.time) <= 0.65]
        if not candidates:
            vignette = 0.94 - 0.14 * (((x - width / 2) / width) ** 2 + ((y - height / 2) / height) ** 2)
            frame *= vignette[..., None]
            return np.clip(frame, 0, 255).astype(np.uint8)

        for idx, ev in enumerate(candidates[-5:]):
            delta = t - ev.time
            decay_sec = max(0.08, ev.decay_ms / 1000.0)
            if delta < 0:
                fade = math.exp(delta * 8.5)
            else:
                fade = math.exp(-delta / decay_sec)

            confidence = float(np.clip(ev.confidence, 0.0, 1.0))
            intensity = float(np.clip(ev.intensity * fade * (0.78 + 0.30 * confidence), 0.0, 1.9))
            chaos_raw = float(np.clip(ev.chaos * (0.75 + 0.35 * confidence), 0.08, 2.0))
            # Лёгкое сглаживание скачков хаоса между соседними событиями (без сильного blur-эффекта).
            chaos_near = _smoothed_chaos(candidates, t)
            chaos = float(np.clip(0.82 * chaos_raw + 0.18 * chaos_near, 0.08, 2.0))
            vocal_mod = float(np.clip(ev.vocal_mod, 0.0, 1.0))

            seed_base = int(ev.time * 1000) + idx * 97 + ev.color_index * 13
            rng = np.random.default_rng(seed=seed_base)
            anchor_x = rng.uniform(0.12, 0.88) * width
            anchor_y = rng.uniform(0.14, 0.86) * height
            sub_count = 1 + (1 if chaos > 0.8 else 0) + (1 if chaos > 1.25 else 0)

            for n in range(sub_count):
                sub = np.random.default_rng(seed=seed_base + n * 31)
                jitter = (0.012 + 0.047 * chaos) * min(width, height)
                cx = anchor_x + sub.uniform(-1.0, 1.0) * jitter
                cy = anchor_y + sub.uniform(-1.0, 1.0) * jitter
                sigma = min(width, height) * sub.uniform(0.066, 0.14)

                color = _color_from_index(ev.color_index + n, flash_colors)
                local_intensity = intensity * sub.uniform(0.34, 0.58) * (1.0 + 0.22 * vocal_mod)
                _add_flash_blob(frame, x, y, cx=cx, cy=cy, sigma=sigma, color=color, intensity=local_intensity)

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
        beat_times = beat_result.beats
        if not beat_times:
            return np.clip(frame, 0, 255).astype(np.uint8)

        idx = np.searchsorted(beat_times, t) - 1
        if idx >= 0:
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
