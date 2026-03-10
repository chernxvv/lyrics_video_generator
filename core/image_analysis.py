from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from models import PaletteColor, PaletteInfo


def extract_dominant_palette(image_path: Path, k: int = 5) -> PaletteInfo:
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((320, 320))
    data = np.array(image).reshape(-1, 3)

    k = max(2, min(k, len(data)))
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(data)

    counts = np.bincount(labels)
    total = counts.sum()
    colors: list[PaletteColor] = []
    for idx, center in enumerate(model.cluster_centers_):
        rgb = tuple(int(max(0, min(255, c))) for c in center)
        ratio = float(counts[idx] / total)
        colors.append(PaletteColor(rgb=rgb, ratio=ratio))

    colors.sort(key=lambda c: c.ratio, reverse=True)
    return PaletteInfo(colors=colors)
