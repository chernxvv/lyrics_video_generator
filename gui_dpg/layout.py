from __future__ import annotations

import dearpygui.dearpygui as dpg


def build_toolbar(app) -> None:
    with dpg.group(horizontal=True, parent="root_window"):
        for label, cb in [
            ("New Project", app.new_project),
            ("Open Project", app.pick_project),
            ("Save Project", app.save_project),
            ("Auto-sync", app.run_auto_sync),
            ("Play / Pause", app.toggle_playback),
            ("Stop", app.stop_playback),
            ("Render Preview", lambda s, a, u: app.render_video("Preview")),
            ("Render Final", lambda s, a, u: app.render_video("Final")),
            ("Rebuild Waveform", app.rebuild_waveform),
            ("Refresh Preview", app.refresh_preview),
        ]:
            dpg.add_button(label=label, callback=cb)
        dpg.add_spacer(width=18)
        dpg.add_text("Idle", tag="status_text")


def build_workspace(app) -> None:
    with dpg.group(horizontal=True, parent="root_window"):
        with dpg.child_window(width=330, autosize_y=True, border=True):
            app.build_left_panel()
        with dpg.child_window(width=-360, autosize_y=True, border=False):
            app.build_center_panel()
        with dpg.child_window(width=340, autosize_y=True, border=True):
            app.build_right_panel()
