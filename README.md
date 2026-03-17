# Lyrics Video Generator

Lyrics Video Generator — это desktop GUI-приложение на PySide6 для сборки lyric-video из аудиотрека, обложки и текста. Проект ориентирован на быстрый практический результат: можно работать полностью вручную или использовать auto-sync как стартовую точку с последующей ручной доводкой. Поддерживаются вертикальные и горизонтальные форматы, а также два визуальных режима фона.

## Main features

- Desktop GUI workflow: загрузка аудио/обложки, ввод метаданных, настройка синхронизации и экспорт видео.
- Две ориентации: `9:16` (vertical) и `16:9` (horizontal).
- Два режима синхронизации: manual и auto-sync.
- Два режима фона: `soft gradient` и `dynamic BPM background`.
- Два режима рендера: `Preview` (быстрая проверка) и `Final` (финальный экспорт).
- Запуск для обычного пользователя через `run.sh` (Linux/macOS) или `run.bat` (Windows).

## Quick start

### Linux/macOS

```bash
./run.sh
```

### Windows

```bat
run.bat
```

Скрипт автоматически создаёт `.venv`, устанавливает зависимости и запускает GUI. Можно выбрать базовый режим (`base`) или установку с авто-синхронизацией (`base + autosync`).

> В системе должны быть установлены `ffmpeg` и `ffprobe`.

## How to use

1. Выберите аудиофайл и изображение обложки.
2. Заполните метаданные трека.
3. Выберите режим синхронизации:
   - **Manual**: заполните таблицу `мм:сс + строка`.
   - **Auto-sync**: вставьте полный текст, запустите автоанализ, затем при необходимости поправьте тайминги вручную.
4. Выберите ориентацию, режим фона и render mode (`Preview`/`Final`).
5. Нажмите «Сгенерировать видео».

## Limitations and quality expectations

- Auto-sync — это usable baseline, а не идеальный alignment для всех треков.
- На сложных/плотных миксах и нетипичном вокале возможны ошибки тайминга.
- После авто-синхронизации обычно нужна ручная доводка.
- Для полного auto-sync pipeline требуются дополнительные зависимости (`autosync` extras).

## Full technical documentation

Полная техническая спецификация (архитектура, fallback-логика, производительность, GPU/NVENC, debug-export, benchmark и инженерные ограничения):

- [SPECIFICATION.md](SPECIFICATION.md)

## Project policies

- [Contributing guide](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md)
