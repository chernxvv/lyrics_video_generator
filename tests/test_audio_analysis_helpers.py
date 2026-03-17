from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.audio_analysis import (
    BeatEvent,
    StemFeatures,
    _align_length,
    _build_event_timeline,
    _build_macro_intensity_and_sections,
    _normalize_series,
    _save_debug_output,
    _to_scalar_bpm,
)


def test_scalar_and_series_helpers() -> None:
    assert _to_scalar_bpm(np.array([123.0])) == 123.0
    assert _to_scalar_bpm([99.0]) == 99.0
    assert _to_scalar_bpm(88) == 88.0

    zeros = _normalize_series(np.array([1.0, 1.0], dtype=np.float32))
    assert np.allclose(zeros, np.zeros(2, dtype=np.float32))

    norm = _normalize_series(np.array([1.0, 3.0], dtype=np.float32))
    assert np.allclose(norm, np.array([0.0, 1.0], dtype=np.float32))


def test_align_length_truncate_and_pad() -> None:
    src = np.array([1, 2, 3], dtype=np.float32)
    assert _align_length(src, 2).tolist() == [1, 2]
    assert _align_length(src, 5).tolist() == [1, 2, 3, 0, 0]


def test_macro_intensity_empty_and_nonempty() -> None:
    macro, sections = _build_macro_intensity_and_sections(np.array([], dtype=np.float32), 0.01)
    assert macro.size == 0
    assert sections.size == 0

    env = np.array([0.1, 0.3, 0.2, 0.9, 0.4, 0.8, 0.1], dtype=np.float32)
    macro2, sections2 = _build_macro_intensity_and_sections(env, 0.1)
    assert len(macro2) >= len(env)
    assert len(sections2) == len(macro2)


def test_build_event_timeline_handles_short_and_normal_inputs() -> None:
    short = StemFeatures(name="drums", times=[0.0, 0.1], onset_strength=[0.1, 0.1], energy_envelope=[0.1, 0.1], spectral_flux=[0.1, 0.1], quality_mask=[1.0, 1.0])
    events, *_ = _build_event_timeline({"drums": short}, np.zeros(2), np.zeros(2, dtype=np.int32), np.zeros(2), 44100, 512)
    assert events == []

    n = 8
    sf = StemFeatures(
        name="drums",
        times=[i * 0.1 for i in range(n)],
        onset_strength=[0.1, 0.6, 0.2, 0.9, 0.2, 0.85, 0.2, 0.1],
        energy_envelope=[0.1, 0.5, 0.2, 0.8, 0.2, 0.7, 0.2, 0.1],
        spectral_flux=[0.1, 0.5, 0.2, 0.7, 0.2, 0.6, 0.2, 0.1],
        quality_mask=[0.9] * n,
    )
    events2, quality, scores, chosen, intensity = _build_event_timeline(
        {"drums": sf}, np.linspace(0.1, 0.9, n), np.zeros(n, dtype=np.int32), np.zeros(n), 44100, 512
    )
    assert len(quality) == n
    assert len(scores) == n
    assert len(chosen) == n
    assert len(intensity) == n
    assert all(isinstance(ev, BeatEvent) for ev in events2)


def test_save_debug_output_writes_csv_and_json(tmp_path: Path) -> None:
    prefix = tmp_path / "dbg" / "analysis"
    sf = StemFeatures(
        name="drums",
        times=[0.0, 0.1],
        onset_strength=[0.2, 0.3],
        energy_envelope=[0.2, 0.3],
        spectral_flux=[0.2, 0.3],
        quality_mask=[0.9, 0.8],
    )
    _save_debug_output(
        str(prefix),
        np.array([0.0, 0.1], dtype=np.float32),
        {"drums": sf},
        np.array([0.5, 0.6], dtype=np.float32),
        np.array([0.2, 0.3], dtype=np.float32),
        np.array([0, -1], dtype=np.int32),
        np.array([0.1, 0.0], dtype=np.float32),
        np.array([0.3, 0.4], dtype=np.float32),
        [BeatEvent(time=0.1, intensity=1.0, chaos=0.5, source_stem="drums", confidence=0.9, color_index=0, decay_ms=200.0)],
    )

    assert prefix.with_suffix('.csv').exists()
    assert prefix.with_suffix('.json').exists()
