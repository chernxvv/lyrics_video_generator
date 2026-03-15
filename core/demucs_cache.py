from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEMUCS_STEM_NAMES = ("drums", "bass", "guitar", "piano", "other", "vocals")


@dataclass(slots=True)
class DemucsCacheResult:
    cache_dir: Path
    stems_dir: Path
    model_name: str
    stem_paths: dict[str, Path]
    reused: bool


def demucs_cache_dir_for_audio(audio_path: str) -> Path:
    src_path = Path(audio_path)
    stat = src_path.stat()
    cache_key = f"{src_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
    cache_hash = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / "lvg_demucs_cache" / cache_hash


def _collect_stems(base_dir: Path) -> dict[str, Path]:
    stems: dict[str, Path] = {}
    for stem in DEMUCS_STEM_NAMES:
        candidate = base_dir / f"{stem}.wav"
        if candidate.exists() and candidate.stat().st_size > 0:
            stems[stem] = candidate
    return stems


def _discover_cached_stems(cache_dir: Path) -> DemucsCacheResult | None:
    vocals_candidates = list(cache_dir.glob("**/vocals.wav"))
    best: tuple[int, int, DemucsCacheResult] | None = None

    for vocals in vocals_candidates:
        stems_dir = vocals.parent
        stem_paths = _collect_stems(stems_dir)
        if "vocals" not in stem_paths:
            continue

        model_name = stems_dir.parent.name if stems_dir.parent else "unknown"
        model_score = 2 if model_name == "htdemucs_6s" else 1
        coverage = len(stem_paths)
        result = DemucsCacheResult(
            cache_dir=cache_dir,
            stems_dir=stems_dir,
            model_name=model_name,
            stem_paths=stem_paths,
            reused=True,
        )
        rank = (model_score, coverage, result)
        if best is None or rank[:2] > best[:2]:
            best = rank

    return best[2] if best else None


def ensure_demucs_stems_cached(
    audio_path: str,
    *,
    preferred_models: tuple[str, ...] = ("htdemucs_6s", "htdemucs"),
) -> DemucsCacheResult:
    if shutil.which("demucs") is None:
        raise RuntimeError("Demucs CLI не найден в PATH")

    src_path = Path(audio_path)
    if not src_path.exists():
        raise RuntimeError("Аудиофайл для Demucs не найден")

    cache_dir = demucs_cache_dir_for_audio(audio_path)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached = _discover_cached_stems(cache_dir)
    if cached and (not preferred_models or cached.model_name in preferred_models):
        return cached

    last_err = ""
    for model_name in preferred_models:
        cmd = ["demucs", "-n", model_name, "--device", "cpu", "-o", str(cache_dir), audio_path]
        logger.info("Demucs-cache: запуск Demucs model=%s", model_name)
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            last_err = (res.stderr or res.stdout or "")[-800:]
            continue

        cached_after_run = _discover_cached_stems(cache_dir)
        if cached_after_run is not None and model_name in cached_after_run.stems_dir.parts:
            return DemucsCacheResult(
                cache_dir=cached_after_run.cache_dir,
                stems_dir=cached_after_run.stems_dir,
                model_name=model_name,
                stem_paths=cached_after_run.stem_paths,
                reused=False,
            )

        if cached_after_run is not None:
            cached_after_run.reused = False
            return cached_after_run

        last_err = "Demucs завершился без пригодных stem-файлов"

    cached_fallback = _discover_cached_stems(cache_dir)
    if cached_fallback is not None:
        cached_fallback.reused = True
        return cached_fallback

    raise RuntimeError(f"Demucs завершился с ошибкой: {last_err}")
