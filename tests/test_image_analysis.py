from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.image_analysis import extract_dominant_palette


def test_extract_dominant_palette_returns_sorted_ratios(tmp_path: Path) -> None:
    image_path = tmp_path / "cover.png"
    img = Image.new("RGB", (20, 20), (255, 0, 0))
    for x in range(5):
        for y in range(20):
            img.putpixel((x, y), (0, 0, 255))
    img.save(image_path)

    palette = extract_dominant_palette(image_path, k=2)

    assert len(palette.colors) >= 2
    ratios = [c.ratio for c in palette.colors]
    assert ratios == sorted(ratios, reverse=True)
    assert abs(sum(ratios) - 1.0) < 1e-6
