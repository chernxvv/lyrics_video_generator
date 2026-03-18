from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path

import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.auto_sync import (
    AutoSyncError,
    auto_sync_lyrics,
    build_autosync_dependency_error,
    get_missing_autosync_packages,
)
from core.logging_config import setup_logging
from core.render import RenderDependencyError, RenderError, render_video
from core.validation import DependencyError, ValidationError
from gui_dpg import dialogs, layout, theme
from gui_dpg.controllers.project_controller import load_project_file, save_project_file
from gui_dpg.preview import ensure_preview_texture, update_preview_texture
from gui_dpg.state import UIState
from gui_dpg.waveform import build_waveform_envelope
from models import LyricLine, RENDER_PROFILES, RenderSettings

logger = logging.getLogger(__name__)


class DPGApplication:
    def __init__(self) -> None:
        self.state = UIState()
        self._timeline_drag_index: int | None = None
        self._worker_lock = threading.Lock()
        self._timeline_drag_active = False
        self._timeline_dirty = True
        self._last_preview_render_at = 0.0
        self._timeline_texture_tag = "timeline_texture"
        self._timeline_width = 1080
        self._timeline_height = 320
        self._last_timeline_render_at = 0.0
        self._audio_process: subprocess.Popen | None = None
        self._playback_anchor: float = 0.0
        self._playback_start_position: float = 0.0
        self._root_margin_x = 12
        self._root_margin_y = 12
        self._toolbar_height = 44
        self._left_panel_width = 320
        self._right_panel_width = 360

    def log(self, message: str) -> None:
        logger.info(message)
        self.state.diagnostics.push(message)
        if dpg.does_item_exist("diagnostics_text"):
            dpg.set_value("diagnostics_text", self.state.diagnostics.text)

    def set_status(self, status: str, message: str | None = None) -> None:
        self.state.set_status(status, message)
        text = status if not message else f"{status} · {message}"
        if dpg.does_item_exist("status_text"):
            dpg.set_value("status_text", text)
        self.log(text)

    def build(self) -> None:
        dpg.create_context()
        dpg.create_viewport(title="Lyrics Video Generator", width=1680, height=960)
        dpg.setup_dearpygui()
        dpg.configure_app(docking=True, docking_space=True)
        dpg.bind_theme(theme.build_theme())
        self._bind_default_font()
        self._sync_preview_geometry()
        self._ensure_timeline_texture()
        with dpg.window(
            tag="root_window",
            label="Lyrics Video Generator",
            pos=(0, 0),
            width=1680,
            height=960,
            no_title_bar=True,
            no_move=True,
            no_resize=True,
            no_collapse=True,
            no_close=True,
            no_bring_to_front_on_focus=True,
        ):
            layout.build_toolbar(self)
            dpg.add_separator()
            layout.build_workspace(self)
        dpg.set_primary_window("root_window", True)
        self.refresh_all()
        if hasattr(dpg, "set_viewport_drop_callback"):
            dpg.set_viewport_drop_callback(self.handle_file_drop)
        if hasattr(dpg, "set_viewport_resize_callback"):
            dpg.set_viewport_resize_callback(self._on_viewport_resize)
        dpg.show_viewport()
        viewport_w = dpg.get_viewport_client_width() if hasattr(dpg, "get_viewport_client_width") else 1680
        viewport_h = dpg.get_viewport_client_height() if hasattr(dpg, "get_viewport_client_height") else 960
        self._on_viewport_resize(None, (viewport_w, viewport_h))

    def build_left_panel(self) -> None:
        with dpg.child_window(border=False):
            dpg.add_text("Project Setup", color=(235, 235, 245))
            dpg.add_text("1. Add source files and basic metadata.", color=(170, 180, 195))
            with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchProp):
                dpg.add_table_column(init_width_or_weight=0.34)
                dpg.add_table_column(init_width_or_weight=0.66)
                with dpg.table_row():
                    dpg.add_text("Audio")
                    dpg.add_button(tag="audio_asset_button", label="Import audio", callback=self.pick_audio, width=-1, height=34)
                with dpg.table_row():
                    dpg.add_text("Cover")
                    dpg.add_button(tag="image_asset_button", label="Import cover", callback=self.pick_image, width=-1, height=34)
                with dpg.table_row():
                    dpg.add_text("Lyrics")
                    dpg.add_button(label="Import lyrics", callback=self.pick_lyrics, width=-1, height=34)
                with dpg.table_row():
                    dpg.add_text("Artist")
                    dpg.add_input_text(tag="artist", callback=lambda s, a, u: self.sync_project_from_ui(), width=-1)
                with dpg.table_row():
                    dpg.add_text("Title")
                    dpg.add_input_text(tag="title", callback=lambda s, a, u: self.sync_project_from_ui(), width=-1)
                with dpg.table_row():
                    dpg.add_text("Release")
                    dpg.add_input_text(tag="release_date", callback=lambda s, a, u: self.sync_project_from_ui(), width=-1)

            dpg.add_separator()
            dpg.add_text("Generation Settings", color=(235, 235, 245))
            dpg.add_text("2. Choose the look and generation mode.", color=(170, 180, 195))
            with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchProp):
                dpg.add_table_column(init_width_or_weight=0.42)
                dpg.add_table_column(init_width_or_weight=0.58)
                with dpg.table_row():
                    dpg.add_text("Orientation")
                    dpg.add_combo(("9:16", "16:9"), default_value="9:16", tag="orientation", callback=lambda s, a, u: self.on_settings_changed(), width=-1)
                with dpg.table_row():
                    dpg.add_text("Background")
                    dpg.add_combo(("soft_gradient", "bpm_dynamic"), default_value="soft_gradient", tag="background_mode", callback=lambda s, a, u: self.on_settings_changed(), width=-1)
                with dpg.table_row():
                    dpg.add_text("Profile")
                    dpg.add_combo(("Preview", "Final"), default_value="Final", tag="render_profile", callback=lambda s, a, u: self.on_settings_changed(), width=-1)
            dpg.add_separator()
            dpg.add_text("Sync", color=(235, 235, 245))
            dpg.add_text("3. Auto-sync if you want a fast starting point.", color=(170, 180, 195))
            with dpg.group(horizontal=True):
                dpg.add_button(label="Auto-sync", callback=self.run_auto_sync, width=120, height=34)
                dpg.add_button(label="Rebuild waveform", callback=self.rebuild_waveform, width=130, height=34)
            dpg.add_spacer(height=6)

            dpg.add_separator()
            dpg.add_text("Generate", color=(235, 235, 245))
            dpg.add_text("4. Generate a quick preview, then render final output.", color=(170, 180, 195))
            dpg.add_button(label="Generate Preview", callback=lambda s, a, u: self.render_video("Preview"), width=-1, height=42)
            dpg.add_spacer(height=6)
            dpg.add_button(label="Render Final", callback=lambda s, a, u: self.render_video("Final"), width=-1, height=38)

            dpg.add_separator()
            with dpg.collapsing_header(label="Advanced", default_open=False):
                with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchProp):
                    dpg.add_table_column(init_width_or_weight=0.42)
                    dpg.add_table_column(init_width_or_weight=0.58)
                    with dpg.table_row():
                        dpg.add_text("Threads")
                        dpg.add_input_int(tag="thread_count", default_value=2, min_value=1, min_clamped=True, callback=lambda s, a, u: self.on_settings_changed(), width=-1)
                    with dpg.table_row():
                        dpg.add_text("Frame chunk")
                        dpg.add_input_int(tag="chunk_size", default_value=60, min_value=1, min_clamped=True, callback=lambda s, a, u: self.on_settings_changed(), width=-1)
                    with dpg.table_row():
                        dpg.add_text("Timeline zoom")
                        dpg.add_input_float(tag="timeline_zoom", default_value=1.0, min_value=0.2, max_value=8.0, min_clamped=True, max_clamped=True, callback=lambda s, a, u: self.on_zoom_changed(), width=-1)
                dpg.add_spacer(height=8)
                dpg.add_button(label="Refresh Preview", callback=self.refresh_preview, width=-1, height=32)

            with dpg.collapsing_header(label="Diagnostics / Logs", default_open=False):
                dpg.add_input_text(tag="diagnostics_text", multiline=True, readonly=True, width=-1, height=220)

    def build_center_panel(self) -> None:
        dpg.add_text("Preview", color=(235, 235, 245))
        with dpg.child_window(height=610, border=False):
            with dpg.group(horizontal=False):
                dpg.add_spacer(height=8)
                dpg.add_image(self.state.preview.texture_tag, tag="preview_image", width=self.state.preview.width, height=self.state.preview.height)
        with dpg.child_window(height=64, border=False):
            dpg.add_text("Transport", color=(220, 220, 230))
            with dpg.group(horizontal=True):
                dpg.add_button(label="Play/Pause", callback=self.toggle_playback, width=86, height=30)
                dpg.add_button(label="Stop", callback=self.stop_playback, width=54, height=30)
                dpg.add_button(label="Frame -", callback=lambda s, a, u: self.nudge_time(-1.0 / max(self.state.transport_fps, 1.0)), width=62, height=30)
                dpg.add_button(label="Frame +", callback=lambda s, a, u: self.nudge_time(1.0 / max(self.state.transport_fps, 1.0)), width=62, height=30)
                dpg.add_text("00:00.00", tag="playback_label")
                dpg.add_spacer(width=16)
                dpg.add_text("Active line: none", tag="active_line_indicator", color=(180, 190, 210))
        dpg.add_text("Timeline", color=(235, 235, 245))
        dpg.add_text("Scrub, click, and drag lyric blocks to retime sync.", color=(170, 180, 195))
        dpg.add_image(self._timeline_texture_tag, tag="timeline_image", width=self._timeline_width, height=self._timeline_height)

    def build_right_panel(self) -> None:
        with dpg.group(horizontal=True):
            dpg.add_text("Lyrics Lines")
            dpg.add_spacer(width=12)
            dpg.add_button(label="Import lyrics", callback=self.pick_lyrics)
        with dpg.child_window(height=360, border=True):
            dpg.add_input_text(
                tag="lyrics_bulk_editor",
                multiline=True,
                width=-1,
                height=336,
                tab_input=True,
                callback=self.on_bulk_lyrics_changed,
            )
        dpg.add_separator()
        dpg.add_text("Selected Line", color=(235, 235, 245))
        dpg.add_text("Selected/active state syncs with preview and timeline.", tag="selected_line_status", color=(170, 180, 195), wrap=260)
        dpg.add_text("Index")
        dpg.add_input_int(tag="selected_index", readonly=True, width=-1)
        dpg.add_text("Start mm:ss")
        dpg.add_input_text(tag="selected_time", callback=lambda s, a, u: self.apply_selected_line_edits(), width=-1)
        dpg.add_text("Text")
        dpg.add_input_text(tag="selected_text", multiline=True, height=130, callback=lambda s, a, u: self.apply_selected_line_edits(), width=-1)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Add line", callback=self.add_line)
            dpg.add_button(label="Delete line", callback=self.delete_selected_line)

    def _on_viewport_resize(self, sender=None, app_data=None) -> None:
        if isinstance(app_data, (list, tuple)) and len(app_data) >= 2:
            viewport_w, viewport_h = int(app_data[0]), int(app_data[1])
        else:
            viewport_w, viewport_h = 1680, 960
        root_width = max(960, viewport_w)
        root_height = max(640, viewport_h)
        content_width = max(720, root_width - (self._root_margin_x * 2))
        content_height = max(480, root_height - (self._root_margin_y * 2))
        center_width = max(360, content_width - self._left_panel_width - self._right_panel_width - 32)
        self._timeline_width = max(360, center_width - 24)
        self._timeline_height = max(220, min(420, int(content_height * 0.28)))
        preview_size = max(420, min(760, center_width - 24, int(content_height * 0.52)))
        self.state.preview.width = preview_size
        self.state.preview.height = preview_size
        if dpg.does_item_exist("root_window"):
            dpg.configure_item("root_window", pos=(0, 0), width=root_width, height=root_height)
        if dpg.does_item_exist("preview_image"):
            dpg.configure_item("preview_image", width=preview_size, height=preview_size)
        if dpg.does_item_exist("timeline_image"):
            dpg.configure_item("timeline_image", width=self._timeline_width, height=self._timeline_height)
        self._timeline_dirty = True
        self.state.preview.dirty = True

    def _stop_audio_playback(self) -> None:
        if self._audio_process is None:
            return
        if self._audio_process.poll() is None:
            self._audio_process.terminate()
        self._audio_process = None

    def _start_audio_playback(self) -> bool:
        self._stop_audio_playback()
        audio_path = self.state.project.audio_path
        if not audio_path or shutil.which("ffplay") is None:
            return False
        start_position = max(0.0, self.state.playback_position)
        cmd = [
            "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
            "-ss", f"{start_position:.3f}",
            "-sync", "audio",
            "-i", str(audio_path),
        ]
        try:
            self._audio_process = subprocess.Popen(cmd)
            self._playback_anchor = time.perf_counter()
            self._playback_start_position = start_position
            return True
        except OSError:
            self._audio_process = None
            return False

    def _restart_audio_if_needed(self) -> None:
        if self.state.transport_playing and not self._start_audio_playback():
            self.state.transport_playing = False

    def _ensure_timeline_texture(self) -> None:
        if dpg.does_item_exist(self._timeline_texture_tag):
            return
        data = np.zeros((self._timeline_height, self._timeline_width, 4), dtype=np.float32)
        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(self._timeline_width, self._timeline_height, data.flatten().tolist(), tag=self._timeline_texture_tag)

    def _sync_preview_geometry(self) -> None:
        self.state.preview.width = max(420, self.state.preview.width or 568)
        self.state.preview.height = max(420, self.state.preview.height or 568)
        ensure_preview_texture(self.state.preview.texture_tag, self.state.preview.width, self.state.preview.height)

    def _bind_default_font(self) -> None:
        font_path = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "NotoSans-Regular.ttf"
        if not font_path.exists():
            return
        with dpg.font_registry():
            with dpg.font(str(font_path), 18) as default_font:
                if hasattr(dpg, "add_font_range_hint"):
                    dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
                    dpg.add_font_range_hint(dpg.mvFontRangeHint_Cyrillic)
                if hasattr(dpg, "add_font_range"):
                    dpg.add_font_range(0x0100, 0x024F)
                    dpg.add_font_range(0x0400, 0x052F)
        dpg.bind_font(default_font)

    def _update_asset_buttons(self) -> None:
        audio_label = self.state.project.audio_path.name if self.state.project.audio_path else "Import audio"
        image_label = self.state.project.image_path.name if self.state.project.image_path else "Import cover"
        if dpg.does_item_exist("audio_asset_button"):
            dpg.configure_item("audio_asset_button", label=audio_label)
        if dpg.does_item_exist("image_asset_button"):
            dpg.configure_item("image_asset_button", label=image_label)

    def pick_audio(self, *args) -> None:
        path = dialogs.pick_open_file(title="Import audio", filetypes=[("Audio files", "*.mp3 *.wav *.flac *.m4a"), ("All files", "*.*")])
        if path is None:
            return
        self.state.project.audio_path = path
        self.sync_ui_from_project()
        self.rebuild_waveform()
        self.set_status("Ready", f"Audio imported: {path.name}")

    def pick_image(self, *args) -> None:
        path = dialogs.pick_open_file(title="Import cover", filetypes=[("Image files", "*.png *.jpg *.jpeg *.webp"), ("All files", "*.*")])
        if path is None:
            return
        self.state.project.image_path = path
        self.sync_ui_from_project()
        self.state.preview.dirty = True
        self._last_preview_render_at = 0.0
        self.set_status("Ready", f"Cover imported: {path.name}")

    def pick_lyrics(self, *args) -> None:
        path = dialogs.pick_open_file(title="Import lyrics", filetypes=[("Lyrics files", "*.txt *.lrc"), ("All files", "*.*")])
        if path is not None:
            self._load_lyrics_from_path(path)

    def pick_project(self, *args) -> None:
        path = dialogs.pick_open_file(title="Open project", filetypes=[("Project files", "*.json"), ("All files", "*.*")])
        if path is None:
            return
        self.state.project = load_project_file(path)
        self.state.project_path = path
        self.state.preview.dirty = True
        self.sync_ui_from_project()
        self.rebuild_waveform()
        self.refresh_all()
        self.set_status("Ready", f"Project loaded: {path.name}")

    def _choose_project_save_path(self) -> Path | None:
        return dialogs.pick_save_file(title="Save project", default_extension=".json", filetypes=[("Project files", "*.json"), ("All files", "*.*")])

    def _choose_render_path(self, mode: str) -> Path | None:
        return dialogs.pick_save_file(title=f"Render {mode}", default_extension=".mp4", filetypes=[("Video files", "*.mp4"), ("All files", "*.*")])

    def refresh_all(self) -> None:
        self.sync_ui_from_project()
        self._update_asset_buttons()
        self.refresh_lyrics_list()
        self.refresh_selected_line_panel()
        self.refresh_preview(force=True)
        self.redraw_timeline()
        dpg.set_value("diagnostics_text", self.state.diagnostics.text)

    def sync_project_from_ui(self, *args) -> None:
        p = self.state.project
        p.artist = dpg.get_value("artist") or ""
        p.title = dpg.get_value("title") or ""
        p.release_date = dpg.get_value("release_date") or ""
        p.orientation = "vertical" if dpg.get_value("orientation") == "9:16" else "horizontal"
        self._sync_preview_geometry()
        p.background_mode = dpg.get_value("background_mode")
        if dpg.does_item_exist("sync_mode"):
            p.sync_mode = dpg.get_value("sync_mode")
        self.state.render_settings.thread_count = max(1, int(dpg.get_value("thread_count") or 1))
        self.state.render_settings.frame_chunk_size = max(1, int(dpg.get_value("chunk_size") or 1))

    def sync_ui_from_project(self) -> None:
        p = self.state.project
        if dpg.does_item_exist("artist"):
            dpg.set_value("artist", p.artist)
            dpg.set_value("title", p.title)
            dpg.set_value("release_date", p.release_date)
            dpg.set_value("orientation", "9:16" if p.orientation == "vertical" else "16:9")
            dpg.set_value("background_mode", p.background_mode)
            if dpg.does_item_exist("sync_mode"):
                dpg.set_value("sync_mode", p.sync_mode)
            dpg.set_value("lyrics_bulk_editor", "\n".join(line.text for line in p.lyrics))
            self._update_asset_buttons()

    def on_settings_changed(self, *args) -> None:
        self.sync_project_from_ui()
        self.state.preview.dirty = True
        self._last_preview_render_at = 0.0
        self._timeline_dirty = True

    def on_zoom_changed(self, *args) -> None:
        self.state.zoom_level = float(dpg.get_value("timeline_zoom") or 1.0)
        self.redraw_timeline()

    def new_project(self, *args) -> None:
        self.state = UIState()
        self.sync_ui_from_project()
        self.set_status("Idle", "New project")
        self.refresh_all()

    def save_project(self, *args) -> None:
        if self.state.project_path is None:
            self.state.project_path = self._choose_project_save_path()
        if self.state.project_path is None:
            return
        self.sync_project_from_ui()
        save_project_file(self.state.project_path, self.state.project)
        self.set_status("Ready", f"Project saved: {self.state.project_path.name}")

    def _load_lyrics_from_path(self, path: Path) -> None:
        for encoding in ("utf-8", "utf-8-sig", "cp1251"):
            try:
                lines = [line.strip() for line in path.read_text(encoding=encoding).splitlines() if line.strip()]
                break
            except UnicodeDecodeError:
                continue
        else:
            dialogs.show_message("Lyrics import", "Не удалось декодировать текстовый файл.")
            return
        self.state.project.auto_sync_lyrics_text = "\n".join(lines)
        if not self.state.project.lyrics:
            self.state.project.lyrics = [LyricLine(start_time=f"{index:02d}:00", text=line) for index, line in enumerate(lines)]
        self.refresh_lyrics_list()
        self._timeline_dirty = True
        self.state.preview.dirty = True
        if lines and self.state.selected_line_index < 0:
            self.state.selected_line_index = 0
        self.refresh_selected_line_panel()
        self.set_status("Ready", f"Lyrics imported: {len(lines)} lines")

    def refresh_lyrics_list(self) -> None:
        if dpg.does_item_exist("lyrics_bulk_editor"):
            current_text = "\n".join(line.text for line in self.state.project.lyrics)
            if dpg.get_value("lyrics_bulk_editor") != current_text:
                dpg.set_value("lyrics_bulk_editor", current_text)

    def on_bulk_lyrics_changed(self, sender, app_data, user_data=None) -> None:
        raw_lines = [line.rstrip() for line in (app_data or "").splitlines()]
        kept_lines = [line for line in raw_lines if line.strip()]
        existing = self.state.project.lyrics
        rebuilt: list[LyricLine] = []
        for index, text_value in enumerate(kept_lines):
            if index < len(existing):
                start_time = existing[index].start_time
            else:
                start_time = self._format_mmss(index * 5.0)
            rebuilt.append(LyricLine(start_time=start_time, text=text_value))
        self.state.project.lyrics = rebuilt
        self.state.project.auto_sync_lyrics_text = "\n".join(kept_lines)
        if rebuilt and self.state.selected_line_index < 0:
            self.state.selected_line_index = 0
        elif self.state.selected_line_index >= len(rebuilt):
            self.state.selected_line_index = len(rebuilt) - 1
        self._timeline_dirty = True
        self.state.preview.dirty = True
        self.refresh_selected_line_panel()

    def _current_active_line_index(self) -> int:
        if not self.state.project.lyrics:
            return -1
        current = self.state.playback_position
        active_index = -1
        for idx, line in enumerate(self.state.project.lyrics):
            if self._parse_time(line.start_time) <= current:
                active_index = idx
            else:
                break
        return active_index

    def refresh_selected_line_panel(self) -> None:
        index = self.state.selected_line_index
        active_index = self._current_active_line_index()
        dpg.set_value("selected_index", index)
        if 0 <= index < len(self.state.project.lyrics):
            line = self.state.project.lyrics[index]
            dpg.set_value("selected_time", line.start_time)
            dpg.set_value("selected_text", line.text)
            state_text = f"Selected line {index + 1}"
            if active_index == index:
                state_text += " · currently active"
            dpg.set_value("selected_line_status", state_text)
        else:
            dpg.set_value("selected_time", "")
            dpg.set_value("selected_text", "")
            dpg.set_value("selected_line_status", "No line selected")
        if 0 <= active_index < len(self.state.project.lyrics):
            dpg.set_value("active_line_indicator", f"Active line: {self.state.project.lyrics[active_index].text[:42]}")
        else:
            dpg.set_value("active_line_indicator", "Active line: none")

    def select_line(self, index: int) -> None:
        self.state.selected_line_index = index
        self.refresh_lyrics_list()
        self.refresh_selected_line_panel()
        self.redraw_timeline()

    def _edit_lyric_time(self, index: int, value: str) -> None:
        self.state.project.lyrics[index] = LyricLine(start_time=value, text=self.state.project.lyrics[index].text)
        if self.state.selected_line_index == index:
            self.refresh_selected_line_panel()
        self.refresh_lyrics_list()
        self._timeline_dirty = True
        self.state.preview.dirty = True

    def _edit_lyric_text(self, index: int, value: str) -> None:
        self.state.project.lyrics[index] = LyricLine(start_time=self.state.project.lyrics[index].start_time, text=value)
        if self.state.selected_line_index == index:
            self.refresh_selected_line_panel()
        self.refresh_lyrics_list()
        self._timeline_dirty = True
        self.state.preview.dirty = True

    def add_line(self, *args) -> None:
        self.state.project.lyrics.append(LyricLine(start_time="00:00", text="New lyric line"))
        self.select_line(len(self.state.project.lyrics) - 1)
        self.refresh_lyrics_list()
        self._restart_audio_if_needed()
        self.state.preview.dirty = True
        self._timeline_dirty = True

    def delete_selected_line(self, *args) -> None:
        index = self.state.selected_line_index
        if 0 <= index < len(self.state.project.lyrics):
            del self.state.project.lyrics[index]
            self.state.selected_line_index = min(index, len(self.state.project.lyrics) - 1)
            self.refresh_lyrics_list()
            self.refresh_selected_line_panel()
            self._timeline_dirty = True
            self.state.preview.dirty = True

    def apply_selected_line_edits(self, *args) -> None:
        index = self.state.selected_line_index
        if 0 <= index < len(self.state.project.lyrics):
            self.state.project.lyrics[index] = LyricLine(start_time=dpg.get_value("selected_time"), text=dpg.get_value("selected_text"))
            self.refresh_lyrics_list()
            self._timeline_dirty = True
            self.state.preview.dirty = True

    def _project_duration(self) -> float:
        return self.state.waveform.duration or max(5.0, len(self.state.project.lyrics) * 2.0)

    def nudge_time(self, delta: float, *args) -> None:
        self.state.playback_position = max(0.0, min(self._project_duration(), self.state.playback_position + delta))
        self._restart_audio_if_needed()
        self.state.preview.dirty = True
        self._timeline_dirty = True

    def toggle_playback(self, *args) -> None:
        self.state.transport_playing = not self.state.transport_playing
        if self.state.transport_playing:
            if not self._start_audio_playback():
                self.state.transport_playing = False
                self.set_status("Error", "Audio playback unavailable (ffplay not found or failed to start)")
                return
        else:
            if self._playback_anchor:
                self.state.playback_position = self._playback_start_position + (time.perf_counter() - self._playback_anchor)
            self._stop_audio_playback()
        self.set_status("Ready", "Playback running" if self.state.transport_playing else "Playback paused")

    def stop_playback(self, *args) -> None:
        self.state.transport_playing = False
        self._stop_audio_playback()
        self.state.playback_position = 0.0
        self.state.preview.dirty = True
        self._timeline_dirty = True

    def refresh_preview(self, *args, force: bool = False) -> None:
        dpg.set_value("playback_label", self._format_time(self.state.playback_position))
        if not force and not self.state.preview.dirty:
            return
        now = time.perf_counter()
        if not force and (now - self._last_preview_render_at) < 0.10:
            return
        try:
            self.sync_project_from_ui()
            update_preview_texture(self.state)
            self._last_preview_render_at = now
        except Exception as exc:  # noqa: BLE001
            self.state.last_error = str(exc)
            self.set_status("Error", f"Preview error: {exc}")

    def rebuild_waveform(self, *args) -> None:
        audio_path = self.state.project.audio_path
        if not audio_path:
            return
        try:
            self.set_status("Analyzing", f"Waveform: {audio_path.name}")
            samples, duration = build_waveform_envelope(audio_path)
            self.state.waveform.audio_path = audio_path
            self.state.waveform.samples = samples
            self.state.waveform.duration = duration
            self.state.waveform.ready = True
            self._timeline_dirty = True
            self.set_status("Ready", f"Waveform rebuilt ({len(samples)} bins)")
        except Exception as exc:  # noqa: BLE001
            self.set_status("Error", f"Waveform failed: {exc}")

    def redraw_timeline(self) -> None:
        if not dpg.does_item_exist(self._timeline_texture_tag):
            return
        width = self._timeline_width
        height = self._timeline_height
        duration = max(self._project_duration(), 1.0)
        zoom = max(self.state.zoom_level, 0.2)
        visible_duration = max(5.0, duration / zoom)
        scroll = min(self.state.timeline_scroll, max(0.0, duration - visible_duration))
        self.state.timeline_scroll = scroll

        image = Image.new("RGBA", (width, height), (25, 29, 36, 255))
        draw = ImageDraw.Draw(image)
        timeline_font_path = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "NotoSans-Regular.ttf"
        timeline_font = ImageFont.truetype(str(timeline_font_path), 14) if timeline_font_path.exists() else ImageFont.load_default()
        draw.rectangle((0, 0, width - 1, height - 1), outline=(80, 90, 110, 255), width=1)

        wf_top = 24
        wf_bottom = 96
        if self.state.waveform.ready and self.state.waveform.samples:
            samples = self.state.waveform.samples
            step = visible_duration / width
            for x in range(width):
                t = scroll + x * step
                idx = min(len(samples) - 1, max(0, int((t / duration) * len(samples))))
                amp = samples[idx]
                y_mid = (wf_top + wf_bottom) / 2
                half = amp * ((wf_bottom - wf_top) / 2)
                draw.line((x, y_mid - half, x, y_mid + half), fill=(110, 181, 255, 220), width=1)

        if visible_duration <= 12:
            ruler_step = 1
        elif visible_duration <= 35:
            ruler_step = 2
        elif visible_duration <= 90:
            ruler_step = 5
        else:
            ruler_step = 10
        start_marker = int(scroll // ruler_step) * ruler_step
        for sec in range(start_marker, int(scroll + visible_duration) + ruler_step, ruler_step):
            x = int((sec - scroll) / visible_duration * width)
            if x < 0 or x > width:
                continue
            draw.line((x, 0, x, height), fill=(58, 63, 74, 160), width=1)
            draw.text((x + 6, 6), self._format_time(sec), fill=(220, 225, 235, 230), font=timeline_font)

        track_y1, track_y2 = 136, 250
        active_index = self._current_active_line_index()
        for index, line in enumerate(self.state.project.lyrics):
            start_time = self._parse_time(line.start_time)
            if index + 1 < len(self.state.project.lyrics):
                end_time = self._parse_time(self.state.project.lyrics[index + 1].start_time)
            else:
                end_time = min(duration, start_time + 3.0)
            end_time = max(start_time + 0.2, end_time)
            if end_time < scroll or start_time > scroll + visible_duration:
                continue
            x1 = max(0, int((start_time - scroll) / visible_duration * width))
            x2 = min(width - 1, int((end_time - scroll) / visible_duration * width))
            if index == self.state.selected_line_index and index == active_index:
                fill = (112, 160, 226, 255)
                outline = (255, 226, 128, 255)
            elif index == self.state.selected_line_index:
                fill = (96, 136, 198, 255)
                outline = (228, 236, 248, 255)
            elif index == active_index:
                fill = (78, 122, 182, 255)
                outline = (255, 204, 102, 255)
            else:
                fill = (66, 93, 125, 230)
                outline = (170, 184, 205, 255)
            draw.rounded_rectangle((x1, track_y1, max(x1 + 14, x2), track_y2), radius=10, fill=fill, outline=outline, width=2)
            text_value = line.text if len(line.text) <= 28 else f"{line.text[:25]}..."
            draw.text((x1 + 10, track_y1 + 16), text_value, fill=(255, 255, 255, 255), font=timeline_font)

        play_x = int((self.state.playback_position - scroll) / visible_duration * width)
        draw.line((play_x, 0, play_x, height), fill=(255, 190, 64, 255), width=2)

        np_image = np.array(image, dtype=np.uint8)
        dpg.set_value(self._timeline_texture_tag, (np_image.astype(np.float32) / 255.0).flatten().tolist())
        self._last_timeline_render_at = time.perf_counter()

        if dpg.does_item_exist("timeline_image") and not dpg.does_item_exist("timeline_handlers"):
            with dpg.item_handler_registry(tag="timeline_handlers"):
                dpg.add_item_clicked_handler(callback=self._on_timeline_click)
                dpg.add_item_double_clicked_handler(callback=self._on_timeline_double_click)
            dpg.bind_item_handler_registry("timeline_image", "timeline_handlers")

    def _update_timeline_interaction(self) -> None:
        if not dpg.does_item_exist("timeline_image"):
            return
        hovered = dpg.is_item_hovered("timeline_image") if hasattr(dpg, "is_item_hovered") else False
        mouse_down = dpg.is_mouse_button_down(0) if hasattr(dpg, "is_mouse_button_down") else False
        if hovered and mouse_down and self._timeline_drag_index is not None:
            self._timeline_drag_active = True
            self._on_timeline_drag(None, None)
            return
        if self._timeline_drag_active and not mouse_down:
            self._timeline_drag_active = False
            self._on_timeline_drag_release(None, None)

    def _on_timeline_click(self, sender, app_data):
        self._seek_or_select_from_mouse(select=True)

    def _on_timeline_double_click(self, sender, app_data):
        self._seek_or_select_from_mouse(select=True)

    def _seek_or_select_from_mouse(self, select: bool) -> None:
        mouse = dpg.get_mouse_pos(local=False)
        origin = dpg.get_item_rect_min("timeline_image")
        size = dpg.get_item_rect_size("timeline_image")
        width = size[0] or 1
        duration = max(self._project_duration(), 1.0)
        visible = max(5.0, duration / max(self.state.zoom_level, 0.2))
        x = min(max(mouse[0] - origin[0], 0.0), width)
        t = self.state.timeline_scroll + (x / width) * visible
        self.state.playback_position = max(0.0, min(duration, t))
        if select:
            self._timeline_drag_index = self._line_index_at_mouse()
            if self._timeline_drag_index is not None:
                self.select_line(self._timeline_drag_index)
        self.state.preview.dirty = True
        self._timeline_dirty = True

    def _line_index_at_mouse(self) -> int | None:
        mouse = dpg.get_mouse_pos(local=False)
        origin = dpg.get_item_rect_min("timeline_image")
        size = dpg.get_item_rect_size("timeline_image")
        width = size[0] or 1
        y = mouse[1] - origin[1]
        if not (126 <= y <= 206):
            return None
        duration = max(self._project_duration(), 1.0)
        visible = max(5.0, duration / max(self.state.zoom_level, 0.2))
        t = self.state.timeline_scroll + ((mouse[0] - origin[0]) / width) * visible
        lyrics = self.state.project.lyrics
        for index, line in enumerate(lyrics):
            start = self._parse_time(line.start_time)
            end = self._parse_time(lyrics[index + 1].start_time) if index + 1 < len(lyrics) else min(duration, start + 3.0)
            if start <= t <= max(end, start + 0.2):
                return index
        return None

    def _on_timeline_drag(self, sender, app_data):
        if self._timeline_drag_index is None:
            self._timeline_drag_index = self._line_index_at_mouse()
        index = self._timeline_drag_index
        if index is None:
            return
        mouse = dpg.get_mouse_pos(local=False)
        origin = dpg.get_item_rect_min("timeline_image")
        size = dpg.get_item_rect_size("timeline_image")
        width = size[0] or 1
        duration = max(self._project_duration(), 1.0)
        visible = max(5.0, duration / max(self.state.zoom_level, 0.2))
        x = min(max(mouse[0] - origin[0], 0.0), width)
        t = max(0.0, min(duration, self.state.timeline_scroll + (x / width) * visible))
        self.state.project.lyrics[index] = LyricLine(start_time=self._format_mmss(t), text=self.state.project.lyrics[index].text)
        self.state.selected_line_index = index
        self.state.playback_position = t
        self.refresh_selected_line_panel()
        self.refresh_lyrics_list()
        self._timeline_dirty = True
        self.state.preview.dirty = True

    def _on_timeline_drag_release(self, sender, app_data):
        self._timeline_drag_index = None

    def run_auto_sync(self, *args) -> None:
        self.sync_project_from_ui()
        audio_path = self.state.project.audio_path
        text = self.state.project.auto_sync_lyrics_text.strip() or "\n".join(line.text for line in self.state.project.lyrics).strip()
        if not audio_path:
            dialogs.show_message("Auto-sync", "Сначала импортируйте аудио.")
            return
        if not text:
            dialogs.show_message("Auto-sync", "Сначала импортируйте или введите текст песни.")
            return
        missing = get_missing_autosync_packages()
        if missing:
            dialogs.show_message("Auto-sync", build_autosync_dependency_error(missing))
            return
        def worker():
            try:
                self.set_status("Auto-sync running", "Analyzing vocals")
                lines = auto_sync_lyrics(str(audio_path), text)
                self.state.project.lyrics = lines
                self.state.project.sync_mode = "auto"
                self.state.project.lyrics_autofilled = True
                self.state.project.auto_sync_audio_path = str(audio_path)
                self.state.selected_line_index = 0 if lines else -1
                self.refresh_all()
                self.set_status("Ready", f"Auto-sync complete: {len(lines)} lines")
            except AutoSyncError as exc:
                self.set_status("Error", str(exc))
                dialogs.show_message("Auto-sync error", str(exc))
        threading.Thread(target=worker, daemon=True).start()

    def render_video(self, mode: str, *args) -> None:
        self.sync_project_from_ui()
        path = self._choose_render_path(mode)
        if path is None:
            return
        if path.suffix.lower() != ".mp4":
            path = path.with_suffix(".mp4")
        orientation = self.state.project.orientation
        profile = RENDER_PROFILES[(orientation, mode)]
        settings = RenderSettings.from_profile(profile)
        settings.thread_count = max(1, int(dpg.get_value("thread_count") or 1))
        settings.frame_chunk_size = max(1, int(dpg.get_value("chunk_size") or 1))
        def worker():
            try:
                self.set_status("Rendering", f"{mode}: {path.name}")
                from core.image_analysis import extract_dominant_palette
                from core.validation import validate_project
                duration = validate_project(self.state.project)
                palette = extract_dominant_palette(Path(self.state.project.image_path))
                codec = render_video(self.state.project, palette, duration, path, settings)
                self.set_status("Ready", f"Rendered {path.name} ({codec})")
                dialogs.show_message("Render complete", f"Видео сохранено:\n{path}")
            except (ValidationError, DependencyError, RenderDependencyError, RenderError) as exc:
                self.set_status("Error", str(exc))
                dialogs.show_message("Render error", str(exc))
        threading.Thread(target=worker, daemon=True).start()

    def handle_file_drop(self, files):
        if not isinstance(files, (list, tuple)):
            return
        for raw in files:
            path = Path(raw)
            suffix = path.suffix.lower()
            if suffix in {".mp3", ".wav", ".flac", ".m4a"}:
                self.state.project.audio_path = path
                self.rebuild_waveform()
            elif suffix in {".png", ".jpg", ".jpeg", ".webp"}:
                self.state.project.image_path = path
            elif suffix in {".txt", ".lrc"}:
                self._load_lyrics_from_path(path)
            elif suffix == ".json":
                self.state.project = load_project_file(path)
                self.state.project_path = path
        self.sync_ui_from_project()
        self.refresh_all()

    def run(self) -> None:
        self.build()
        last = time.perf_counter()
        while dpg.is_dearpygui_running():
            now = time.perf_counter()
            if self.state.transport_playing:
                if self._audio_process is not None and self._audio_process.poll() is not None:
                    self.state.transport_playing = False
                    self._audio_process = None
                self.state.playback_position = min(
                    self._project_duration(),
                    self._playback_start_position + (now - self._playback_anchor),
                )
                if self.state.playback_position >= self._project_duration():
                    self.state.transport_playing = False
                    self._stop_audio_playback()
                self.state.preview.dirty = True
                self._timeline_dirty = True
            last = now
            self._update_timeline_interaction()
            if self._timeline_dirty and (now - self._last_timeline_render_at) >= 0.05:
                self.redraw_timeline()
                self._timeline_dirty = False
            self.refresh_preview()
            dpg.render_dearpygui_frame()
        dpg.destroy_context()

    @staticmethod
    def _parse_time(value: str) -> float:
        try:
            mm, ss = value.replace(",", ".").split(":", 1)
            return int(mm) * 60 + float(ss)
        except Exception:
            return 0.0

    @staticmethod
    def _format_time(seconds: float) -> str:
        minutes = int(seconds // 60)
        rem = seconds - minutes * 60
        return f"{minutes:02d}:{rem:05.2f}"

    @staticmethod
    def _format_mmss(seconds: float) -> str:
        minutes = int(seconds // 60)
        rem = seconds - minutes * 60
        return f"{minutes:02d}:{rem:05.2f}"


def run_app() -> int:
    setup_logging()
    try:
        DPGApplication().run()
        return 0
    except Exception:  # noqa: BLE001
        logger.exception("Dear PyGui application crashed during startup or runtime")
        raise
