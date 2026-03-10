from __future__ import annotations

import math

import numpy as np

from models import PaletteInfo


def build_background_frame(
    t: float,
    width: int,
    height: int,
    palette: PaletteInfo,
) -> np.ndarray:
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
