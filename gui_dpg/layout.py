from __future__ import annotations

import dearpygui.dearpygui as dpg


def build_toolbar(app) -> None:
    with dpg.group(horizontal=True, parent="root_window"):
        for label, cb in [
            ("New Project", app.new_project),
            ("Open Project", app.pick_project),
            ("Save Project", app.save_project),
        ]:
            dpg.add_button(label=label, callback=cb)
        dpg.add_spacer(width=18)
        dpg.add_text("Idle", tag="status_text")


def build_workspace(app) -> None:
    with dpg.table(
        parent="root_window",
        header_row=False,
        resizable=False,
        borders_innerV=False,
        borders_outerV=False,
        borders_innerH=False,
        borders_outerH=False,
        policy=dpg.mvTable_SizingStretchProp,
        tag="workspace_table",
        width=-1,
    ):
        dpg.add_table_column(init_width_or_weight=320, width_fixed=True)
        dpg.add_table_column(init_width_or_weight=0.50)
        dpg.add_table_column(init_width_or_weight=360, width_fixed=True)
        with dpg.table_row():
            with dpg.table_cell():
                with dpg.child_window(tag="left_panel", width=-1, autosize_y=True, border=False):
                    app.build_left_panel()
            with dpg.table_cell():
                with dpg.child_window(tag="center_panel", width=-1, autosize_y=True, border=False):
                    app.build_center_panel()
            with dpg.table_cell():
                with dpg.child_window(tag="right_panel", width=-1, autosize_y=True, border=False):
                    app.build_right_panel()
