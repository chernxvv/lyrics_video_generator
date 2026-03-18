from __future__ import annotations

import dearpygui.dearpygui as dpg


def build_theme() -> int:
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (24, 27, 34), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (30, 34, 42), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (40, 44, 54), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Button, (61, 88, 122), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (79, 112, 148), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Header, (56, 66, 84), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Tab, (44, 48, 60), category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 10, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 8, category=dpg.mvThemeCat_Core)
    return theme
