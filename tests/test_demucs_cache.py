from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.demucs_cache import (
    _collect_stems,
    _discover_cached_stems,
    demucs_cache_dir_for_audio,
    ensure_demucs_stems_cached,
)


def _write_stem(dir_path: Path, stem: str, size: int = 4) -> None:
    p = dir_path / f"{stem}.wav"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * size)


def test_collect_stems_ignores_empty_files(tmp_path: Path) -> None:
    _write_stem(tmp_path, "vocals", size=10)
    (tmp_path / "drums.wav").write_bytes(b"")

    stems = _collect_stems(tmp_path)

    assert "vocals" in stems
    assert "drums" not in stems


def test_discover_cached_stems_prefers_htdemucs_6s_and_more_coverage(tmp_path: Path) -> None:
    d1 = tmp_path / "htdemucs" / "song"
    _write_stem(d1, "vocals")
    _write_stem(d1, "drums")

    d2 = tmp_path / "htdemucs_6s" / "song"
    for stem in ("vocals", "drums", "bass", "piano"):
        _write_stem(d2, stem)

    result = _discover_cached_stems(tmp_path)

    assert result is not None
    assert result.model_name == "htdemucs_6s"
    assert len(result.stem_paths) == 4
    assert result.reused is True


def test_ensure_demucs_stems_cached_raises_when_demucs_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"a")
    monkeypatch.setattr("core.demucs_cache.shutil.which", lambda _: None)

    with pytest.raises(RuntimeError, match="Demucs CLI не найден"):
        ensure_demucs_stems_cached(str(audio))


def test_ensure_demucs_stems_cached_raises_when_audio_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.demucs_cache.shutil.which", lambda _: "/usr/bin/demucs")

    with pytest.raises(RuntimeError, match="Аудиофайл для Demucs не найден"):
        ensure_demucs_stems_cached("/no/such/file.mp3")


def test_ensure_demucs_stems_cached_returns_existing_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"a")
    cache_dir = tmp_path / "cache"
    stems_dir = cache_dir / "htdemucs_6s" / "song"
    _write_stem(stems_dir, "vocals")
    _write_stem(stems_dir, "drums")

    monkeypatch.setattr("core.demucs_cache.shutil.which", lambda _: "/usr/bin/demucs")
    monkeypatch.setattr("core.demucs_cache.demucs_cache_dir_for_audio", lambda _p: cache_dir)

    result = ensure_demucs_stems_cached(str(audio))

    assert result.reused is True
    assert result.model_name == "htdemucs_6s"


def test_ensure_demucs_stems_cached_runs_demucs_and_returns_new_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"a")
    cache_dir = tmp_path / "cache"

    def fake_run(cmd, **kwargs):
        model = cmd[cmd.index("-n") + 1]
        out_dir = Path(cmd[cmd.index("-o") + 1]) / model / audio.stem
        _write_stem(out_dir, "vocals")
        _write_stem(out_dir, "drums")
        return SimpleNamespace(returncode=0, stderr="", stdout="ok")

    monkeypatch.setattr("core.demucs_cache.shutil.which", lambda _: "/usr/bin/demucs")
    monkeypatch.setattr("core.demucs_cache.demucs_cache_dir_for_audio", lambda _p: cache_dir)
    monkeypatch.setattr("core.demucs_cache.subprocess.run", fake_run)

    result = ensure_demucs_stems_cached(str(audio), preferred_models=("htdemucs_6s",))

    assert result.reused is False
    assert "vocals" in result.stem_paths


def test_demucs_cache_dir_for_audio_is_stable_for_same_file(tmp_path: Path) -> None:
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"same")

    p1 = demucs_cache_dir_for_audio(str(audio))
    p2 = demucs_cache_dir_for_audio(str(audio))

    assert p1 == p2
    assert "lvg_demucs_cache" in str(p1)
