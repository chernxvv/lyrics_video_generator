# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Roman Chernov (romanchernovv@gmail.com)

from __future__ import annotations

import os
import subprocess
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


def main() -> int:
    total_steps = 3
    project_root = _project_root()
    venv_dir = project_root / ".venv"

    try:
        _print_step(1, total_steps, "Проверка виртуального окружения (.venv)")
        if not venv_dir.exists():
            print(f"Создаю окружение: {venv_dir}")
            venv.EnvBuilder(with_pip=True).create(venv_dir)
        else:
            print(f"Окружение уже существует: {venv_dir}")

        venv_python = _venv_python(venv_dir)
        if not venv_python.exists():
            print("❌ Не найден интерпретатор внутри .venv. Попробуйте удалить .venv и запустить снова.")
            return 1

        _print_step(2, total_steps, "Установка/обновление приложения в .venv")
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"],
            check=True,
            cwd=project_root,
        )
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "."],
            check=True,
            cwd=project_root,
        )

        _print_step(3, total_steps, "Запуск приложения")
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
