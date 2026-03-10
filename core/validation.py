from __future__ import annotations

import logging
from pathlib import Path

from core.audio import probe_audio_duration
from core.lyrics import parse_mmss, sort_lyrics
from models import ProjectData

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
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
    if not data.lyrics:
        raise ValidationError("Добавьте хотя бы одну строку текста.")

    duration = probe_audio_duration(Path(data.audio_path))
    sorted_lines = sort_lyrics(data.lyrics)
    logger.info("Проверка %d строк текста", len(sorted_lines))

    for line in sorted_lines:
        if not line.text.strip():
            raise ValidationError("Текст строки не может быть пустым.")
        start = parse_mmss(line.start_time)
        if start > duration:
            raise ValidationError(f"Строка '{line.text[:24]}' начинается позже конца трека.")

    logger.info("Валидация завершена успешно")
    return duration
