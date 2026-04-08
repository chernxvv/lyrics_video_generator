"""Deprecated compatibility shim.

The application frontend was migrated from PySide6 to Dear PyGui.
This module preserves the historical import path used by older scripts.
"""

from gui_dpg.app import DPGApplication as MainWindow

__all__ = ["MainWindow"]
