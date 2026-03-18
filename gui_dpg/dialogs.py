from __future__ import annotations

from pathlib import Path

import dearpygui.dearpygui as dpg


def _normalize_selection(app_data) -> Path | None:
    if not app_data:
        return None
    selections = app_data.get("selections") or {}
    if selections:
        return Path(next(iter(selections.values())))
    current = app_data.get("current_path")
    file_name = app_data.get("file_name")
    if current and file_name:
        return Path(current) / file_name
    return None


def attach_file_dialog(tag: str, label: str, callback, extensions: list[tuple[str, str]], directory_selector: bool = False):
    with dpg.file_dialog(
        directory_selector=directory_selector,
        show=False,
        callback=callback,
        tag=tag,
        width=900,
        height=560,
        modal=True,
        label=label,
    ):
        for extension, display in extensions:
            dpg.add_file_extension(extension, color=(120, 180, 255, 255), custom_text=display)


def open_dialog(tag: str) -> None:
    dpg.configure_item(tag, show=True)


def show_message(title: str, message: str) -> None:
    tag = f"modal::{title}"
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)
    with dpg.window(label=title, modal=True, tag=tag, no_resize=True, autosize=True):
        dpg.add_text(message, wrap=520)
        dpg.add_button(label="OK", width=90, callback=lambda: dpg.delete_item(tag))
