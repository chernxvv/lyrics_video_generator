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
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from models import PaletteColor, PaletteInfo

logger = logging.getLogger(__name__)


def extract_dominant_palette(image_path: Path, k: int = 5) -> PaletteInfo:
    logger.info("Анализ обложки и извлечение палитры: %s", image_path)
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
    logger.info("Палитра извлечена: %s", [(c.rgb, round(c.ratio, 3)) for c in colors])
    return PaletteInfo(colors=colors)
