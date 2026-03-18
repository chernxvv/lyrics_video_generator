from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import dearpygui.dearpygui as dpg

_DIALOG_ROOT: tk.Tk | None = None


def _get_dialog_root() -> tk.Tk:
    global _DIALOG_ROOT
    if _DIALOG_ROOT is None:
        _DIALOG_ROOT = tk.Tk()
        _DIALOG_ROOT.withdraw()
        _DIALOG_ROOT.attributes("-topmost", True)
    return _DIALOG_ROOT


def pick_open_file(*, title: str, filetypes: list[tuple[str, str]]) -> Path | None:
    root = _get_dialog_root()
    selected = filedialog.askopenfilename(parent=root, title=title, filetypes=filetypes)
    return Path(selected) if selected else None


def pick_save_file(*, title: str, default_extension: str, filetypes: list[tuple[str, str]]) -> Path | None:
    root = _get_dialog_root()
    selected = filedialog.asksaveasfilename(
        parent=root,
        title=title,
        defaultextension=default_extension,
        filetypes=filetypes,
    )
    return Path(selected) if selected else None


def show_message(title: str, message: str) -> None:
    tag = f"modal::{title}"
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)
    with dpg.window(label=title, modal=True, tag=tag, no_resize=True, autosize=True):
        dpg.add_text(message, wrap=520)
        dpg.add_button(label="OK", width=90, callback=lambda s, a, u: dpg.delete_item(tag))
