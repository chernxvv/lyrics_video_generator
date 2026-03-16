# Copyright 2026 Roman Chernov (romanchernovv@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.

from __future__ import annotations

import logging
from pathlib import Path

from core.audio import AudioError, probe_audio_duration
from core.lyrics import parse_mmss, sort_lyrics
from models import ProjectData

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    pass


class DependencyError(RuntimeError):
    pass


def validate_project(data: ProjectData) -> float:
    logger.info("Старт валидации проекта")

    if not data.audio_path or not Path(data.audio_path).exists():
        raise ValidationError("Выберите существующий аудиофайл.")
    if not data.image_path or not Path(data.image_path).exists():
        raise ValidationError("Выберите существующее изображение.")
    if not data.artist.strip():
        raise ValidationError("Укажите исполнителя.")
    if not data.title.strip():
        raise ValidationError("Укажите название трека.")
    if not data.release_date.strip():
        raise ValidationError("Укажите дату релиза.")
    auto_sync_ready = data.lyrics_autofilled and bool(data.audio_path) and data.auto_sync_audio_path == str(data.audio_path)
    if data.sync_mode == "auto" and not data.auto_sync_lyrics_text.strip() and not auto_sync_ready:
        raise ValidationError("В режиме автосинхронизации нужно вставить полный текст трека.")
    if not data.lyrics:
        raise ValidationError("Добавьте хотя бы одну строку текста.")

    try:
        duration = probe_audio_duration(Path(data.audio_path))
    except AudioError as exc:
        raise DependencyError(str(exc)) from exc

    try:
        sorted_lines = sort_lyrics(data.lyrics)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    logger.info("Проверка %d строк текста", len(sorted_lines))

    for line in sorted_lines:
        if not line.text.strip():
            raise ValidationError("Текст строки не может быть пустым.")
        try:
            start = parse_mmss(line.start_time)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        if start > duration:
            raise ValidationError(f"Строка '{line.text[:24]}' начинается позже конца трека.")

    logger.info("Валидация завершена успешно")
    return duration
