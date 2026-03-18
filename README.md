# Lyrics Video Generator

Lyrics Video Generator — это desktop GUI-приложение на Dear PyGui для сборки lyric-video из аудиотрека, обложки и текста. Проект ориентирован на быстрый практический результат: можно работать полностью вручную или использовать auto-sync как стартовую точку с последующей ручной доводкой. Поддерживаются вертикальные и горизонтальные форматы, а также два визуальных режима фона.

## Статус проекта

Public beta.

Базовый workflow уже пригоден для реальной работы: доступны ручная синхронизация, рендер, вертикальный и горизонтальный форматы, а также режимы фона.

Auto-sync и BPM-reactive background работают, но в зависимости от трека, сложности микса и локального окружения могут требовать ручной доводки.

## Быстрый запуск

### Linux/macOS

```bash
./run.sh
```

### Windows

```bat
run.bat
```

Скрипт автоматически создаёт `.venv`, устанавливает зависимости и запускает Dear PyGui GUI. Можно выбрать базовый режим (`base`) или установку с авто-синхронизацией (`base + autosync`).

> В системе должны быть установлены `ffmpeg` и `ffprobe`.

## Примеры результата

**Vertical (`9:16`) · Мягкий градиент · Preview**

https://github.com/user-attachments/assets/ca55d529-f876-4a39-afbf-0d0ac4412a87

**Vertical (`9:16`) · Динамический BPM-фон · Preview**

https://github.com/user-attachments/assets/0e230c52-47be-4ab2-b99e-bf1f6c71c6a0

**Horizontal (`16:9`) · Мягкий градиент · Preview**

https://github.com/user-attachments/assets/79e4012c-45dd-4145-85e4-22393296b4d1

**Horizontal (`16:9`) · Динамический BPM-фон · Preview**

https://github.com/user-attachments/assets/130ea04b-9a82-4d6d-97f0-3f9bb9726ef9

## Как использовать

1. Импортируйте аудио, обложку и lyrics через toolbar или drag & drop.
2. Настройте проект в левой панели: ориентацию, профиль рендера, фон и параметры производительности.
3. Используйте preview по центру для проверки текущего кадра и transport controls для scrub/playback.
4. Работайте с waveform + timeline: кликайте для seek, перетаскивайте lyric blocks для ретайминга, редактируйте текст выбранной строки справа.
5. Запускайте Auto-sync и экспорт через `Render Preview` / `Render Final`.

## Основные возможности

- Editor-like Dear PyGui workspace: toolbar, project inspector, preview, timeline, waveform, lyric properties и экспорт видео.
- Две ориентации: `9:16` (vertical) и `16:9` (horizontal).
- Два режима синхронизации: manual и auto-sync, плюс ручной ретайминг lyric blocks на timeline.
- Два режима фона: `soft gradient` и `dynamic BPM background`.
- Два режима рендера: `Preview` (быстрая проверка) и `Final` (финальный экспорт), с preview frame в основном окне.
- Запуск для обычного пользователя через `run.sh` (Linux/macOS) или `run.bat` (Windows).

## Минимальные системные ожидания

- Python 3.11+.
- В системе должны быть установлены `ffmpeg` и `ffprobe`, доступные через `PATH`.
- Основные целевые среды — Windows и Linux.
- NVIDIA GPU не обязателен: он используется только для ускорения кодирования через NVENC, если доступен.
- Для auto-sync нужны дополнительные опциональные зависимости (`autosync` extras).

## Ограничения и ожидания в отношении качества

- Auto-sync — это usable baseline, а не идеальный alignment для всех треков.
- На сложных/плотных миксах и нетипичном вокале возможны ошибки тайминга.
- После авто-синхронизации обычно нужна ручная доводка.
- Для полного auto-sync pipeline требуются дополнительные зависимости (`autosync` extras).

## Полная техническая документация

Полная техническая спецификация (архитектура, fallback-логика, производительность, GPU/NVENC, debug-export, benchmark и инженерные ограничения):

- [SPECIFICATION.md](SPECIFICATION.md)

## Project policies and contribution

- [Contributing guide](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md)
