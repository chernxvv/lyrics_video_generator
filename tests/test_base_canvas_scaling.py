from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.layout import compute_layout, get_base_canvas
from core.render import _build_filter_complex, _compute_uniform_scale, _iter_next_lyric_indices, _scale_box, _scale_value
from models import ProjectData


def _project(orientation: str) -> ProjectData:
    return ProjectData(
        audio_path=Path("audio.mp3"),
        image_path=Path("cover.jpg"),
        artist="Artist",
        title="Title",
        release_date="2026",
        orientation=orientation,
    )


def _relative_layout_values(width: int, height: int, orientation: str) -> dict[str, float]:
    base_w, base_h = get_base_canvas(orientation)
    layout = compute_layout(width, height, orientation)
    scale = _compute_uniform_scale(base_w, base_h, width, height)
    cover = _scale_box(layout.cover_box, scale)
    lyrics = _scale_box(layout.lyrics_box, scale)

    return {
        "artist_y": _scale_value(layout.artist_y, scale, minimum=0) / height,
        "title_y": _scale_value(layout.title_y, scale, minimum=0) / height,
        "date_y": _scale_value(layout.date_y, scale, minimum=0) / height,
        "cover_w": (cover[2] - cover[0]) / width,
        "cover_h": (cover[3] - cover[1]) / height,
        "lyrics_w": (lyrics[2] - lyrics[0]) / width,
        "lyrics_h": (lyrics[3] - lyrics[1]) / height,
        "lyrics_y": lyrics[1] / height,
        "lyrics_font_regular": _scale_value(25, scale) / height,
        "lyrics_font_bold": _scale_value(28, scale) / height,
    }


def _assert_close_dict(left: dict[str, float], right: dict[str, float], eps: float = 1e-6) -> None:
    assert left.keys() == right.keys()
    for k in left:
        assert abs(left[k] - right[k]) <= eps, f"Mismatch for {k}: {left[k]} != {right[k]}"


def test_vertical_preview_final_relative_sizes_match() -> None:
    preview = _relative_layout_values(540, 960, "vertical")
    final = _relative_layout_values(1080, 1920, "vertical")
    _assert_close_dict(preview, final)


def test_horizontal_preview_final_relative_sizes_match() -> None:
    preview = _relative_layout_values(960, 540, "horizontal")
    final = _relative_layout_values(1920, 1080, "horizontal")
    _assert_close_dict(preview, final)


def test_filter_complex_scales_metadata_and_cover_consistently() -> None:
    orientation = "vertical"
    project = _project(orientation)
    base_w, base_h = get_base_canvas(orientation)

    preview_layout = compute_layout(base_w, base_h, orientation)
    preview_scale = _compute_uniform_scale(base_w, base_h, base_w, base_h)
    preview_fc = _build_filter_complex(project, preview_layout, use_cuda=False, scale_factor=preview_scale)

    final_w, final_h = 1080, 1920
    final_layout = compute_layout(final_w, final_h, orientation)
    final_scale = _compute_uniform_scale(base_w, base_h, final_w, final_h)
    final_fc = _build_filter_complex(project, final_layout, use_cuda=False, scale_factor=final_scale)

    preview_cover = _scale_box(preview_layout.cover_box, preview_scale)
    final_cover = _scale_box(final_layout.cover_box, final_scale)

    assert f"overlay={preview_cover[0]}:{preview_cover[1]}" in preview_fc
    assert f"overlay={final_cover[0]}:{final_cover[1]}" in final_fc
    assert f"fontsize={_scale_value(58, preview_scale)}" in preview_fc
    assert f"fontsize={_scale_value(58, final_scale)}" in final_fc
    assert f"fontsize={_scale_value(36, preview_scale)}" in preview_fc
    assert f"fontsize={_scale_value(36, final_scale)}" in final_fc


def test_vertical_orientation_limits_next_lyrics_to_single_line() -> None:
    assert list(_iter_next_lyric_indices(2, 8, "vertical")) == [3]


def test_horizontal_orientation_keeps_scrollable_next_lyrics() -> None:
    assert list(_iter_next_lyric_indices(2, 8, "horizontal")) == [3, 4, 5, 6, 7]
