# Lyrics Video Generator — Technical Specification

This document is the full technical specification of the project. It preserves engineering details, current limitations, fallback behavior, and installation/runtime notes.

## 1. Project overview

Lyrics Video Generator is a desktop GUI application (PySide6) for generating lyric videos from:
- audio track;
- cover image;
- song metadata;
- lyric timing (manual or auto-generated).

The project follows a practical MVP philosophy: produce a usable output quickly, keep fallback paths robust, and allow manual correction where fully automatic quality is not guaranteed.

## 2. Supported features

- Two video orientations:
  - `9:16` (vertical)
  - `16:9` (horizontal)
- Render profiles per orientation:
  - Vertical Preview: `540x960`, `24 fps`
  - Vertical Final: `1080x1920`, `30 fps`
  - Horizontal Preview: `960x540`, `24 fps`
  - Horizontal Final: `1920x1080`, `30 fps`
- Synchronization modes:
  - Manual (`мм:сс + строка` table)
  - Automatic (audio analysis + full lyrics text + editable output table)
- Background modes:
  - `soft_gradient`
  - `bpm_dynamic`
- NVENC with automatic fallback to `libx264`
- Streaming/chunked render (multithreaded frame generation + ffmpeg pipe)

## 3. Project structure

- `gui/main_window.py` — GUI (orientation, sync mode, auto-sync flow, background mode).
- `models/project.py` — project models + render profiles.
- `core/audio.py` — duration probing via `ffprobe`.
- `core/image_analysis.py` — palette extraction from cover.
- `core/layout.py` — adaptive frame geometry (`vertical`/`horizontal`).
- `core/lyrics.py` — timing parsing/sorting, active-line handling.
- `core/auto_sync.py` — baseline lyrics auto-sync pipeline.
- `core/audio_analysis.py` — BPM/beat analysis (Demucs 6-stem + fallback), unified event timeline.
- `core/background.py` — background strategies (`soft_gradient`, `bpm_dynamic`).
- `core/validation.py` — project validation.
- `core/render.py` — streaming render and muxing via `ffmpeg`.

## 4. Installation

### 4.1 Installation via `pyproject.toml` (recommended)

Base install (GUI + manual sync):

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

Install with `autosync` extra (WhisperX/Librosa/Demucs):

```bash
python -m venv .venv
source .venv/bin/activate
pip install .[autosync]
```

### 4.2 Stable installation mode (`constraints.txt`)

For reproducible environments, use pinned constraints:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -c constraints.txt .
# optional autosync:
pip install -c constraints.txt .[autosync]
```

`bootstrap.py` and `run.sh`/`run.bat` use this stable mode:
- install `base` by default;
- install `base + autosync` with `--autosync`;
- run preflight version-resolution checks;
- install project with `-c constraints.txt`.

Short commands:

```bash
# Interactive mode (asks which dependency mode to install)
python -m bootstrap

# Non-interactive autosync install
python -m bootstrap --autosync
```

If you started with `base`, optional autosync dependencies can be installed later:

```bash
python -m bootstrap --autosync
```

### 4.3 Installation via requirements files (alternative)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# plus autosync stack:
pip install -r requirements-autosync.txt
```

### 4.4 System dependencies (`ffmpeg` / `ffprobe`)

`ffmpeg` and `ffprobe` must be available in `PATH`.

### 4.5 Windows notes

If you get `WinError 2`, in most cases `ffmpeg`/`ffprobe` are missing in `PATH`. Install FFmpeg and add paths to `ffmpeg.exe` and `ffprobe.exe`.

### 4.6 Manual app launch

```bash
python main.py
```

### 4.7 pip / deprecation notes

Warnings like:
- `DEPRECATION: ... using the legacy setup.py install method`

do not usually block runtime now, but indicate legacy install paths that may change in future pip versions.

Safe actions:
- upgrade `pip`, `setuptools`, `wheel`;
- if needed, install with `--use-pep517`;
- if install succeeds and app runs, treat it as warning (not render runtime failure).

## 5. Render modes

Select one mode before export:
- **Preview** — faster draft render.
- **Final** — full-quality export.

Profiles:
- **Vertical (9:16)**
  - Preview: `540x960`, `24 fps`
  - Final: `1080x1920`, `30 fps`
- **Horizontal (16:9)**
  - Preview: `960x540`, `24 fps`
  - Final: `1920x1080`, `30 fps`

GUI also includes **Performance settings**:
- frame-generation thread count;
- chunk size (default: 60 frames).

## 6. Performance settings and render pipeline behavior

The renderer uses streaming/chunked architecture:
- frames are generated in Python (multithreaded chunk processing);
- output is piped into ffmpeg for encoding/muxing;
- codec choice and effective render FPS are logged.

Parameter trade-offs:
- larger chunk => higher peak memory;
- too small chunk => more overhead;
- to reduce CPU peaks, try `threads=1` and/or smaller chunk size.

## 7. GPU encoding, filtergraph, runtime-probe and fallback

By default, rendering attempts hardware encoder `h264_nvenc`.
If unavailable, automatic software fallback to `libx264` is applied.

Important architecture note:
- GPU accelerates **encoding** (NVENC);
- frame generation (background, text overlay, composition) remains CPU-side.

Low GPU utilization with slow render often indicates CPU-bound stages.

### 7.1 NVENC availability check

```bash
ffmpeg -hide_banner -encoders | rg h264_nvenc
```

If the encoder is listed, NVENC is available in current ffmpeg build.

### 7.2 `filter_complex` branches (cover preprocess)

Two branches exist:
- **CPU branch**: `scale + crop`
- **CUDA branch**: `format=nv12 -> hwupload_cuda -> scale_cuda -> hwdownload -> format=nv12 -> crop`

Selection logic:
1. probe NVENC + CUDA filters (`scale_cuda`, `hwupload_cuda`);
2. execute short runtime CUDA probe on actual ffmpeg;
3. log explicit decision (`CUDA filtergraph: enabled/disabled`) and active branch (`CPU`/`CUDA`).

If CUDA branch fails at runtime (driver/device/ffmpeg issues), render automatically restarts with CPU branch and logs rollback reason.

### 7.3 What is accelerated vs CPU-bound

Accelerated when NVENC/CUDA available:
- video encoding via `h264_nvenc`;
- cover scaling in CUDA `filter_complex` branch.

CPU-bound in current architecture:
- background generation (including BPM dynamics);
- lyric overlay preparation in Pillow;
- frame composition in Python before ffmpeg pipe.

## 8. Synchronization modes

### 8.1 Manual synchronization

- User enters start times and lines manually (`мм:сс` or `мм:сс.сс`).
- Fully compatible with current MVP flow.

### 8.2 Automatic synchronization

Workflow in GUI:
1. select **Automatic** mode;
2. paste full lyrics (or load from file);
3. click **Auto-sync**;
4. edit generated table manually if needed.

Pipeline details:
- Demucs is attempted first; if `vocals` stem is available, transcription uses vocals, otherwise source falls back to full mix.
- WhisperX transcription + alignment provide word-level timestamps.
- Line start timestamps are computed by separate post-processing (`core/line_alignment.py`) based on words (not raw `segment.start`).
- When alignment misses the first token(s) of a line, line post-processing estimates the true line start from later matched words using local speech-rate heuristics instead of anchoring strictly to the first detected matched word.
- Global alignment is additionally validated by sequential local re-checks: if a low-information line or a suspiciously late repeated phrase jumps too far from nearby lines, the algorithm reanchors it near the current cursor instead of blindly keeping a later chorus/reprise match.
- Weak global matches are rejected before anchoring if they rely only on stopwords/particles (for example a lone `не`) or otherwise do not show enough overall line similarity; such cases fall back to stronger local validation instead of producing a false timestamp.
- UX heuristics include `pre-roll`, `min_line_gap`, false-start protection, and fallback strategies for weakly recognized lines.
- If WhisperX is unavailable or alignment fails, pipeline falls back to tuned librosa backend (onset/energy improvements).

### 8.3 WhisperX notes

- First run may be slower due to model/component downloads.
- If backend unavailable, app automatically switches to librosa fallback.
- Missing optional `torchcodec`/`hf_xet` can affect model download speed but does not block autosync.

### 8.4 Demucs notes

For more precise BPM-reactive background, analysis first attempts Demucs (`htdemucs_6s`) to extract stems (`drums`, `bass`, `guitar`, `piano`, `other`, `vocals`).
Then a unified event timeline (winner-take-all by stem source) is built with quality filtering and pause handling.
If Demucs is unavailable, fallback to HPSS-based librosa approach is used.
First Demucs run can be slower due to model loading.
Demucs cache is shared: stems extracted during autosync are reused in BPM analysis (and vice versa) for the same source track.

## 9. BPM debug export

For tuning rhythmic background alignment, diagnostic export can be enabled:

```bash
LVG_BPM_DEBUG_PREFIX=./debug/bpm_analysis python main.py
```

Exports:
- CSV time-series (`score_*`, `quality_*`, `quality_mask`, `chosen_stem_idx`, `event_intensity`, `macro_intensity`);
- JSON with `BeatEvent` list and parameters.

Path to debug files is logged to console.

## 10. Background modes

> In `base` mode, `Dynamic BPM background` works in fallback analysis scenarios.
> In `base + autosync` mode (Demucs/Librosa stack), beat alignment is usually more accurate.

- **Soft gradient** — smooth animated background based on cover palette.
- **Dynamic BPM background** — dominant-color base layer + heartbeat-like events from multi-stem analysis; intensity/chaos depend on stem source, local tempo, percussive energy, and optional vocal contribution. Pauses are preserved between events, abrupt chaos jumps are slightly smoothed, and flash color selection is two-step: first a palette anchor by `BeatEvent.color_index`, then weighted local-neighborhood selection by `ratio` (excluding base color) to keep musical coherence while preserving dominant cover palette.

## 11. Quality limitations and expectations

### 11.1 Auto-sync

- Usable baseline, not perfect alignment.
- Main mode uses WhisperX word-level timestamps + heuristic line post-processing, but manual correction is still often required.
- Best on tracks with clearly separable vocals.
- Accuracy degrades on heavily processed vocals, noisy mixes, or extreme vocal techniques.
- Post-analysis table remains editable and manual finishing is recommended.

### 11.2 BPM analysis and dynamic background

- Beat detection runs once before render.
- Effect modulation uses beat events and local tempo/percussive energy when available.
- With Demucs, unified event bitmap line is built across stems (`drums/guitar/piano/other`) without crossing sources inside one event.
- Vocals are used only as additional modulation where vocal stem is actually active.
- Weak analyzer output triggers fallback behavior (warnings in logs).
- Effect is designed to stay soft/atmospheric (not aggressive strobe), but complex scenes can still slow rendering.

## 12. Render performance factors

Render speed depends on:
- resolution and FPS (Final heavier than Preview; 16:9 Final also resource-intensive);
- NVENC availability and selected codec/preset (`nvenc_preset`, `nvenc_cq`);
- background mode (`bpm_dynamic` can be heavier than `soft_gradient`);
- CPU/GPU performance and disk throughput;
- number of active-line changes (text redraw only when active line changes);
- threading/chunk parameters.

## 13. Benchmark

Use same machine/scene for fair comparison.

| Scenario | Mode | Codec | Render time |
|---|---|---|---|
| Baseline | Vertical Final | libx264 | _measure locally_ |
| New modes | Horizontal Preview | h264_nvenc/libx264 | _measure locally_ |
| New modes | Horizontal Final + bpm_dynamic | h264_nvenc/libx264 | _measure locally_ |

Minimal syntax check command:

```bash
python -m compileall .
```

## 14. Dependencies

### 14.1 Base (`requirements.txt`)

- `PySide6`
- `Pillow`
- `numpy`
- `scikit-learn`

### 14.2 Optional AutoSync (`requirements-autosync.txt`)

- `librosa`
- `soundfile`
- `whisperx`
- `demucs`

### 14.3 Stable pins (`constraints.txt`)

Pinned versions for reproducible installation:
- `PySide6==6.10.2`
- `Pillow==10.4.0`
- `numpy==2.4.3`
- `scikit-learn==1.8.0`
- `librosa==0.11.0`
- `soundfile==0.13.1`
- `whisperx==3.8.2`
- `demucs==4.0.1`

## 15. Additional technical notes

- `run.sh` / `run.bat` are intended as the easiest entrypoint for end users.
- The app logs key runtime decisions for diagnostics (encoder, fallback reasons, render fps).
- Current implementation intentionally combines automation with manual editability for production usability under real-world audio variability.
