# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Roman Chernov (romanchernovv@gmail.com)

from __future__ import annotations

import os
import argparse
import shutil
import subprocess
import sys
import venv
from pathlib import Path


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _print_step(step: int, total: int, message: str) -> None:
    print(f"[шаг {step}/{total}] {message}")


def _ensure_ffmpeg_tools() -> bool:
    """Проверяет ffmpeg/ffprobe и пытается установить, если их нет."""
    required_tools = ("ffmpeg", "ffprobe")
    missing_tools = [tool for tool in required_tools if shutil.which(tool) is None]
    if not missing_tools:
        print("✅ Системные утилиты ffmpeg и ffprobe уже доступны.")
        return True

    print(f"⚠️ Не найдены системные утилиты: {', '.join(missing_tools)}")
    install_attempts = _build_ffmpeg_install_commands()
    if not install_attempts:
        print(
            "❌ Не удалось определить подходящий менеджер пакетов для автоустановки ffmpeg. "
            "Установите ffmpeg вручную и запустите bootstrap снова."
        )
        return False

    for command in install_attempts:
        print(f"Пробую установить ffmpeg через: {' '.join(command)}")
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            continue

        still_missing = [tool for tool in required_tools if shutil.which(tool) is None]
        if not still_missing:
            print("✅ Успешно установлены ffmpeg и ffprobe.")
            return True

    print(
        "❌ Автоматическая установка ffmpeg не удалась. "
        "Установите ffmpeg вручную (должны появиться и ffmpeg, и ffprobe), затем повторите запуск."
    )
    return False


def _build_ffmpeg_install_commands() -> list[list[str]]:
    if sys.platform.startswith("linux"):
        commands: list[list[str]] = []
        if shutil.which("apt-get"):
            commands.extend([
                ["sudo", "apt-get", "update"],
                ["sudo", "apt-get", "install", "-y", "ffmpeg"],
            ])
        elif shutil.which("dnf"):
            commands.append(["sudo", "dnf", "install", "-y", "ffmpeg"])
        elif shutil.which("yum"):
            commands.append(["sudo", "yum", "install", "-y", "ffmpeg"])
        elif shutil.which("pacman"):
            commands.append(["sudo", "pacman", "-S", "--noconfirm", "ffmpeg"])
        elif shutil.which("zypper"):
            commands.append(["sudo", "zypper", "--non-interactive", "install", "ffmpeg"])
        return commands

    if sys.platform == "darwin" and shutil.which("brew"):
        return [["brew", "install", "ffmpeg"]]

    if os.name == "nt":
        commands = []
        if shutil.which("winget"):
            commands.append(["winget", "install", "--id", "Gyan.FFmpeg", "-e", "--accept-source-agreements", "--accept-package-agreements"])
        if shutil.which("choco"):
            commands.append(["choco", "install", "ffmpeg", "-y"])
        if shutil.which("scoop"):
            commands.append(["scoop", "install", "ffmpeg"])
        return commands

    return []


def _dependency_preflight(
    venv_python: Path,
    project_root: Path,
    constraints_path: Path,
    install_target: str,
) -> bool:
    """Проверяет заранее, что зафиксированные версии можно разрешить без конфликтов."""
    check_cmd = [
        str(venv_python),
        "-m",
        "pip",
        "install",
        "--dry-run",
        "-c",
        str(constraints_path),
        install_target,
    ]
    result = subprocess.run(check_cmd, cwd=project_root, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        print("✅ Preflight: версии из constraints.txt успешно разрешаются.")
        return True

    print("❌ Preflight не пройден: не удалось разрешить версии из constraints.txt.")
    details = (result.stderr or result.stdout or "").strip()
    if details:
        print("--- pip output ---")
        print(details)
        print("------------------")

    print(
        "Подсказка: проверьте доступность указанных версий в constraints.txt "
        "и совместимость их диапазонов в pyproject.toml. "
        "После исправлений запустите bootstrap снова."
    )
    return False




def _select_install_target(args: argparse.Namespace) -> tuple[str, str]:
    if args.autosync:
        return ".[autosync]", "base + autosync"

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return ".", "base"

    print("Выберите режим установки:")
    print("  1) base (быстрее, без автосинхронизации)")
    print("  2) base + autosync (WhisperX/Librosa/Demucs, установка дольше)")

    while True:
        choice = input("Введите 1 или 2 (Enter = 1): ").strip()
        if choice in {"", "1"}:
            return ".", "base"
        if choice == "2":
            return ".[autosync]", "base + autosync"
        print("Неверный ввод. Укажите 1 или 2.")

def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap окружения для Lyrics Video Generator")
    parser.add_argument(
        "--autosync",
        action="store_true",
        help="Установить extra-зависимости для автосинхронизации",
    )
    args = parser.parse_args()

    total_steps = 5
    project_root = _project_root()
    venv_dir = project_root / ".venv"
    constraints_path = project_root / "constraints.txt"
    install_target, mode_label = _select_install_target(args)

    try:
        _print_step(1, total_steps, "Проверка системных зависимостей (ffmpeg/ffprobe)")
        if not _ensure_ffmpeg_tools():
            return 1

        _print_step(2, total_steps, "Проверка виртуального окружения (.venv)")
        if not venv_dir.exists():
            print(f"Создаю окружение: {venv_dir}")
            venv.EnvBuilder(with_pip=True).create(venv_dir)
        else:
            print(f"Окружение уже существует: {venv_dir}")

        venv_python = _venv_python(venv_dir)
        if not venv_python.exists():
            print("❌ Не найден интерпретатор внутри .venv. Попробуйте удалить .venv и запустить снова.")
            return 1
        if not constraints_path.exists():
            print("❌ Не найден constraints.txt в корне проекта. Восстановите файл и запустите bootstrap снова.")
            return 1

        _print_step(3, total_steps, "Обновление pip/setuptools/wheel")
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"],
            check=True,
            cwd=project_root,
        )

        print(f"ℹ️ Режим установки: {mode_label}.")
        if install_target == ".[autosync]":
            print("ℹ️ Включён autosync: установка может занять больше времени из-за дополнительных зависимостей.")
        else:
            print("ℹ️ Позже можно добавить autosync командой: python -m bootstrap --autosync")

        _print_step(4, total_steps, "Preflight-проверка зависимостей (constraints)")
        if not _dependency_preflight(venv_python, project_root, constraints_path, install_target):
            return 1

        subprocess.run(
            [str(venv_python), "-m", "pip", "install", install_target, "-c", str(constraints_path)],
            check=True,
            cwd=project_root,
        )

        _print_step(5, total_steps, "Запуск приложения")
        result = subprocess.run([str(venv_python), "main.py"], cwd=project_root, check=False)
        return result.returncode

    except subprocess.CalledProcessError as exc:
        print("❌ Не удалось выполнить один из шагов установки.")
        print(f"Команда завершилась с кодом {exc.returncode}: {' '.join(map(str, exc.cmd))}")
        return exc.returncode or 1
    except Exception as exc:  # дружелюбная ошибка без stack trace
        print(f"❌ Ошибка bootstrap: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
