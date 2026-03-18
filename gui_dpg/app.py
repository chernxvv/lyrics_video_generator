from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import dearpygui.dearpygui as dpg

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
        ensure_preview_texture(self.state.preview.texture_tag, self.state.preview.width, self.state.preview.height)
        self._register_dialogs()
        with dpg.window(tag="root_window", label="Lyrics Video Generator", width=1660, height=930):
            layout.build_toolbar(self)
            dpg.add_separator()
            layout.build_workspace(self)
        self.refresh_all()
        if hasattr(dpg, "set_viewport_drop_callback"):
            dpg.set_viewport_drop_callback(self.handle_file_drop)
        dpg.show_viewport()

    def build_left_panel(self) -> None:
        dpg.add_text("Project Assets")
        dpg.add_input_text(tag="audio_path", label="Audio", readonly=True)
        dpg.add_input_text(tag="image_path", label="Cover", readonly=True)
        dpg.add_input_text(tag="artist", label="Artist", callback=lambda s, a, u: self.sync_project_from_ui())
        dpg.add_input_text(tag="title", label="Title", callback=lambda s, a, u: self.sync_project_from_ui())
        dpg.add_input_text(tag="release_date", label="Release", callback=lambda s, a, u: self.sync_project_from_ui())
        dpg.add_separator()
        dpg.add_text("Video Settings")
        dpg.add_combo(("9:16", "16:9"), default_value="9:16", tag="orientation", callback=lambda s, a, u: self.on_settings_changed())
        dpg.add_combo(("Preview", "Final"), default_value="Final", tag="render_profile", callback=lambda s, a, u: self.on_settings_changed())
        dpg.add_combo(("soft_gradient", "bpm_dynamic"), default_value="soft_gradient", tag="background_mode", callback=lambda s, a, u: self.on_settings_changed())
        dpg.add_separator()
        dpg.add_text("Sync Settings")
        dpg.add_combo(("manual", "auto"), default_value="manual", tag="sync_mode", callback=lambda s, a, u: self.on_settings_changed())
        dpg.add_input_int(tag="thread_count", label="Threads", default_value=2, min_value=1, min_clamped=True, callback=lambda s, a, u: self.on_settings_changed())
        dpg.add_input_int(tag="chunk_size", label="Frame chunk", default_value=60, min_value=1, min_clamped=True, callback=lambda s, a, u: self.on_settings_changed())
        dpg.add_input_float(tag="timeline_zoom", label="Timeline zoom", default_value=1.0, min_value=0.2, max_value=8.0, min_clamped=True, max_clamped=True, callback=lambda s, a, u: self.on_zoom_changed())
        dpg.add_separator()
        dpg.add_text("Preview")
        with dpg.group(horizontal=True):
            dpg.add_button(label="<< 1s", callback=lambda s, a, u: self.nudge_time(-1.0))
            dpg.add_button(label="1s >>", callback=lambda s, a, u: self.nudge_time(1.0))
        dpg.add_text("Diagnostics / Logs")
        dpg.add_input_text(tag="diagnostics_text", multiline=True, readonly=True, width=-1, height=280)

    def build_center_panel(self) -> None:
        dpg.add_text("Preview")
        dpg.add_image(self.state.preview.texture_tag)
        dpg.add_text("Transport")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Play/Pause", callback=self.toggle_playback)
            dpg.add_button(label="Stop", callback=self.stop_playback)
            dpg.add_button(label="Frame -", callback=lambda s, a, u: self.nudge_time(-1.0 / max(self.state.transport_fps, 1.0)))
            dpg.add_button(label="Frame +", callback=lambda s, a, u: self.nudge_time(1.0 / max(self.state.transport_fps, 1.0)))
            dpg.add_text("00:00.00", tag="playback_label")
        dpg.add_separator()
        dpg.add_text("Timeline")
        dpg.add_text("Click to seek. Drag lyric blocks horizontally to retime.")
        with dpg.drawlist(width=-1, height=280, tag="timeline_drawlist"):
            pass

    def build_right_panel(self) -> None:
        dpg.add_text("Lyrics Lines")
        with dpg.child_window(height=360, border=True):
            dpg.add_group(tag="lyrics_list")
        dpg.add_separator()
        dpg.add_text("Selected Line")
        dpg.add_input_int(tag="selected_index", label="Index", readonly=True)
        dpg.add_input_text(tag="selected_time", label="Start mm:ss", callback=lambda s, a, u: self.apply_selected_line_edits())
        dpg.add_input_text(tag="selected_text", label="Text", multiline=True, height=130, callback=lambda s, a, u: self.apply_selected_line_edits())
        with dpg.group(horizontal=True):
            dpg.add_button(label="Add line", callback=self.add_line)
            dpg.add_button(label="Delete line", callback=self.delete_selected_line)

    def _register_dialogs(self) -> None:
        dialogs.attach_file_dialog("audio_dialog", "Import audio", self._on_audio_dialog, [(".mp3", "Audio"), (".wav", "Audio"), (".flac", "Audio"), (".m4a", "Audio")])
        dialogs.attach_file_dialog("image_dialog", "Import cover", self._on_image_dialog, [(".png", "Image"), (".jpg", "Image"), (".jpeg", "Image"), (".webp", "Image")])
        dialogs.attach_file_dialog("lyrics_dialog", "Import lyrics", self._on_lyrics_dialog, [(".txt", "Lyrics"), (".lrc", "Lyrics")])
        dialogs.attach_file_dialog("open_project_dialog", "Open project", self._on_open_project, [(".json", "Project")])
        dialogs.attach_file_dialog("save_project_dialog", "Save project", self._on_save_project, [(".json", "Project")])
        dialogs.attach_file_dialog("render_dialog", "Render output", self._on_render_path, [(".mp4", "Video")])

    def refresh_all(self) -> None:
        self.sync_ui_from_project()
        self.refresh_lyrics_list()
        self.refresh_selected_line_panel()
        self.refresh_preview()
        self.redraw_timeline()
        dpg.set_value("diagnostics_text", self.state.diagnostics.text)

    def sync_project_from_ui(self, *args) -> None:
        p = self.state.project
        p.artist = dpg.get_value("artist") or ""
        p.title = dpg.get_value("title") or ""
        p.release_date = dpg.get_value("release_date") or ""
        p.orientation = "vertical" if dpg.get_value("orientation") == "9:16" else "horizontal"
        p.background_mode = dpg.get_value("background_mode")
        p.sync_mode = dpg.get_value("sync_mode")
        self.state.render_settings.thread_count = max(1, int(dpg.get_value("thread_count") or 1))
        self.state.render_settings.frame_chunk_size = max(1, int(dpg.get_value("chunk_size") or 1))

    def sync_ui_from_project(self) -> None:
        p = self.state.project
        if dpg.does_item_exist("audio_path"):
            dpg.set_value("audio_path", str(p.audio_path or ""))
            dpg.set_value("image_path", str(p.image_path or ""))
            dpg.set_value("artist", p.artist)
            dpg.set_value("title", p.title)
            dpg.set_value("release_date", p.release_date)
            dpg.set_value("orientation", "9:16" if p.orientation == "vertical" else "16:9")
            dpg.set_value("background_mode", p.background_mode)
            dpg.set_value("sync_mode", p.sync_mode)

    def on_settings_changed(self, *args) -> None:
        self.sync_project_from_ui()
        self.state.preview.dirty = True
        self.redraw_timeline()
        self.refresh_preview()

    def on_zoom_changed(self, *args) -> None:
        self.state.zoom_level = float(dpg.get_value("timeline_zoom") or 1.0)
        self.redraw_timeline()

    def new_project(self, *args) -> None:
        self.state = UIState()
        self.sync_ui_from_project()
        self.set_status("Idle", "New project")
        self.refresh_all()

    def save_project(self, *args) -> None:
        if self.state.project_path is not None:
            self.sync_project_from_ui()
            save_project_file(self.state.project_path, self.state.project)
            self.set_status("Ready", f"Project saved: {self.state.project_path.name}")
            return
        dpg.configure_item("save_project_dialog", show=True)

    def _on_save_project(self, sender, app_data):
        path = dialogs._normalize_selection(app_data)
        if path is None:
            return
        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")
        self.state.project_path = path
        self.sync_project_from_ui()
        save_project_file(path, self.state.project)
        self.set_status("Ready", f"Project saved: {path.name}")

    def _on_open_project(self, sender, app_data):
        path = dialogs._normalize_selection(app_data)
        if path is None:
            return
        self.state.project = load_project_file(path)
        self.state.project_path = path
        self.state.preview.dirty = True
        self.sync_ui_from_project()
        self.rebuild_waveform()
        self.refresh_all()
        self.set_status("Ready", f"Project loaded: {path.name}")

    def _on_audio_dialog(self, sender, app_data):
        path = dialogs._normalize_selection(app_data)
        if path is None:
            return
        self.state.project.audio_path = path
        self.sync_ui_from_project()
        self.rebuild_waveform()
        self.set_status("Ready", f"Audio imported: {path.name}")

    def _on_image_dialog(self, sender, app_data):
        path = dialogs._normalize_selection(app_data)
        if path is None:
            return
        self.state.project.image_path = path
        self.sync_ui_from_project()
        self.refresh_preview()
        self.set_status("Ready", f"Cover imported: {path.name}")

    def _on_lyrics_dialog(self, sender, app_data):
        path = dialogs._normalize_selection(app_data)
        if path is None:
            return
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
        self.redraw_timeline()
        self.refresh_preview()
        self.set_status("Ready", f"Lyrics imported: {len(lines)} lines")

    def refresh_lyrics_list(self) -> None:
        if not dpg.does_item_exist("lyrics_list"):
            return
        for child in dpg.get_item_children("lyrics_list", 1) or []:
            dpg.delete_item(child)
        for index, line in enumerate(self.state.project.lyrics):
            label = f"{line.start_time}  {line.text[:48]}"
            dpg.add_selectable(label=label, parent="lyrics_list", default_value=index == self.state.selected_line_index, callback=lambda s, a, u=index: self.select_line(u))

    def refresh_selected_line_panel(self) -> None:
        index = self.state.selected_line_index
        dpg.set_value("selected_index", index)
        if 0 <= index < len(self.state.project.lyrics):
            line = self.state.project.lyrics[index]
            dpg.set_value("selected_time", line.start_time)
            dpg.set_value("selected_text", line.text)
        else:
            dpg.set_value("selected_time", "")
            dpg.set_value("selected_text", "")

    def select_line(self, index: int) -> None:
        self.state.selected_line_index = index
        self.refresh_lyrics_list()
        self.refresh_selected_line_panel()
        self.redraw_timeline()

    def add_line(self, *args) -> None:
        self.state.project.lyrics.append(LyricLine(start_time="00:00", text="New lyric line"))
        self.select_line(len(self.state.project.lyrics) - 1)
        self.state.preview.dirty = True
        self.refresh_preview()

    def delete_selected_line(self, *args) -> None:
        index = self.state.selected_line_index
        if 0 <= index < len(self.state.project.lyrics):
            del self.state.project.lyrics[index]
            self.state.selected_line_index = min(index, len(self.state.project.lyrics) - 1)
            self.refresh_lyrics_list()
            self.refresh_selected_line_panel()
            self.redraw_timeline()
            self.refresh_preview()

    def apply_selected_line_edits(self, *args) -> None:
        index = self.state.selected_line_index
        if 0 <= index < len(self.state.project.lyrics):
            self.state.project.lyrics[index] = LyricLine(start_time=dpg.get_value("selected_time"), text=dpg.get_value("selected_text"))
            self.refresh_lyrics_list()
            self.redraw_timeline()
            self.refresh_preview()

    def _project_duration(self) -> float:
        return self.state.waveform.duration or max(5.0, len(self.state.project.lyrics) * 2.0)

    def nudge_time(self, delta: float, *args) -> None:
        self.state.playback_position = max(0.0, min(self._project_duration(), self.state.playback_position + delta))
        self.state.preview.dirty = True
        self.redraw_timeline()
        self.refresh_preview()

    def toggle_playback(self, *args) -> None:
        self.state.transport_playing = not self.state.transport_playing
        self.set_status("Ready", "Playback running" if self.state.transport_playing else "Playback paused")

    def stop_playback(self, *args) -> None:
        self.state.transport_playing = False
        self.state.playback_position = 0.0
        self.state.preview.dirty = True
        self.redraw_timeline()
        self.refresh_preview()

    def refresh_preview(self, *args) -> None:
        try:
            self.sync_project_from_ui()
            update_preview_texture(self.state)
        except Exception as exc:  # noqa: BLE001
            self.state.last_error = str(exc)
            self.set_status("Error", f"Preview error: {exc}")
        dpg.set_value("playback_label", self._format_time(self.state.playback_position))

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
            self.redraw_timeline()
            self.set_status("Ready", f"Waveform rebuilt ({len(samples)} bins)")
        except Exception as exc:  # noqa: BLE001
            self.set_status("Error", f"Waveform failed: {exc}")

    def redraw_timeline(self) -> None:
        if not dpg.does_item_exist("timeline_drawlist"):
            return
        dpg.delete_item("timeline_drawlist", children_only=True)
        width = dpg.get_item_rect_size("timeline_drawlist")[0] or 900
        height = dpg.get_item_rect_size("timeline_drawlist")[1] or 280
        duration = max(self._project_duration(), 1.0)
        zoom = max(self.state.zoom_level, 0.2)
        visible_duration = max(5.0, duration / zoom)
        scroll = min(self.state.timeline_scroll, max(0.0, duration - visible_duration))
        self.state.timeline_scroll = scroll
        pmin = [0, 0]
        pmax = [width, height]
        dpg.draw_rectangle(pmin, pmax, fill=(25, 29, 36, 255), color=(80, 90, 110, 255), parent="timeline_drawlist")
        wf_top = 24
        wf_bottom = 120
        if self.state.waveform.ready and self.state.waveform.samples:
            samples = self.state.waveform.samples
            step = visible_duration / width
            for x in range(int(width)):
                t = scroll + x * step
                idx = min(len(samples) - 1, max(0, int((t / duration) * len(samples))))
                amp = samples[idx]
                y_mid = (wf_top + wf_bottom) / 2
                half = amp * ((wf_bottom - wf_top) / 2)
                dpg.draw_line((x, y_mid - half), (x, y_mid + half), color=(110, 181, 255, 180), parent="timeline_drawlist")
        for sec in range(int(scroll), int(scroll + visible_duration) + 1):
            x = (sec - scroll) / visible_duration * width
            dpg.draw_line((x, 0), (x, height), color=(58, 63, 74, 140), parent="timeline_drawlist")
            dpg.draw_text((x + 4, 4), self._format_time(sec), size=12, color=(210, 210, 220, 180), parent="timeline_drawlist")
        track_y1, track_y2 = 150, 220
        for index, line in enumerate(self.state.project.lyrics):
            start = self._parse_time(line.start_time)
            if index + 1 < len(self.state.project.lyrics):
                end = self._parse_time(self.state.project.lyrics[index + 1].start_time)
            else:
                end = min(duration, start + 3.0)
            end = max(start + 0.2, end)
            if end < scroll or start > scroll + visible_duration:
                continue
            x1 = max(0.0, (start - scroll) / visible_duration * width)
            x2 = min(width, (end - scroll) / visible_duration * width)
            fill = (91, 133, 190, 255) if index == self.state.selected_line_index else (66, 93, 125, 220)
            dpg.draw_rectangle((x1, track_y1), (x2, track_y2), fill=fill, color=(190, 200, 215, 255), parent="timeline_drawlist")
            dpg.draw_text((x1 + 6, track_y1 + 10), line.text[:30], size=14, color=(255, 255, 255, 255), parent="timeline_drawlist")
        play_x = (self.state.playback_position - scroll) / visible_duration * width
        dpg.draw_line((play_x, 0), (play_x, height), color=(255, 190, 64, 255), thickness=2, parent="timeline_drawlist")
        if not dpg.does_item_exist("timeline_handlers"):
            with dpg.item_handler_registry(tag="timeline_handlers"):
                dpg.add_item_clicked_handler(callback=self._on_timeline_click)
                dpg.add_item_double_clicked_handler(callback=self._on_timeline_double_click)
        dpg.bind_item_handler_registry("timeline_drawlist", "timeline_handlers")

    def _update_timeline_interaction(self) -> None:
        if not dpg.does_item_exist("timeline_drawlist"):
            return
        hovered = dpg.is_item_hovered("timeline_drawlist") if hasattr(dpg, "is_item_hovered") else False
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
        origin = dpg.get_item_rect_min("timeline_drawlist")
        size = dpg.get_item_rect_size("timeline_drawlist")
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
        self.redraw_timeline()
        self.refresh_preview()

    def _line_index_at_mouse(self) -> int | None:
        mouse = dpg.get_mouse_pos(local=False)
        origin = dpg.get_item_rect_min("timeline_drawlist")
        size = dpg.get_item_rect_size("timeline_drawlist")
        width = size[0] or 1
        y = mouse[1] - origin[1]
        if not (150 <= y <= 220):
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
        origin = dpg.get_item_rect_min("timeline_drawlist")
        size = dpg.get_item_rect_size("timeline_drawlist")
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
        self.redraw_timeline()
        self.refresh_preview()

    def _on_timeline_drag_release(self, sender, app_data):
        self._timeline_drag_index = None

    def run_auto_sync(self, *args) -> None:
        self.sync_project_from_ui()
        audio_path = self.state.project.audio_path
        text = self.state.project.auto_sync_lyrics_text.strip()
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
        dpg.set_item_user_data("render_dialog", mode)
        dpg.configure_item("render_dialog", show=True)

    def _on_render_path(self, sender, app_data):
        path = dialogs._normalize_selection(app_data)
        if path is None:
            return
        mode = dpg.get_item_user_data("render_dialog") or "Final"
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
                self._on_lyrics_dialog(None, {"selections": {path.name: str(path)}})
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
                dt = now - last
                self.state.playback_position = min(self._project_duration(), self.state.playback_position + dt)
                if self.state.playback_position >= self._project_duration():
                    self.state.transport_playing = False
                self.state.preview.dirty = True
                self.redraw_timeline()
                self.refresh_preview()
            last = now
            self._update_timeline_interaction()
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
    DPGApplication().run()
    return 0
