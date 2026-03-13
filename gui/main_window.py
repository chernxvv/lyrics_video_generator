from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.auto_sync import AutoSyncError, auto_sync_lyrics
from core.image_analysis import extract_dominant_palette
from core.render import RenderDependencyError, RenderError, render_video
from core.validation import DependencyError, ValidationError, validate_project
from models import LyricLine, ProjectData, RenderSettings, RENDER_PROFILES

logger = logging.getLogger(__name__)


class PerformanceSettingsDialog(QDialog):
    def __init__(self, thread_count: int, chunk_size: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки производительности")
        layout = QFormLayout(self)

        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 32)
        self.threads_spin.setValue(thread_count)

        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(1, 600)
        self.chunk_spin.setValue(chunk_size)

        layout.addRow("Количество потоков", self.threads_spin)
        layout.addRow("Размер чанка (кадров)", self.chunk_spin)

        buttons = QHBoxLayout()
        save_btn = QPushButton("Сохранить")
        cancel_btn = QPushButton("Отмена")
        save_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(save_btn)
        buttons.addWidget(cancel_btn)
        layout.addRow(buttons)


class AutoSyncWorker(QThread):
    done = Signal(list)
    failed = Signal(str)

    def __init__(self, audio_path: Path, lyrics_text: str):
        super().__init__()
        self.audio_path = audio_path
        self.lyrics_text = lyrics_text

    def run(self):
        logger.info("AutoSyncWorker: запуск")
        try:
            lines = auto_sync_lyrics(str(self.audio_path), self.lyrics_text)
            self.done.emit(lines)
        except AutoSyncError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("AutoSyncWorker: неизвестная ошибка")
            self.failed.emit(str(exc))


class RenderWorker(QThread):
    progress = Signal(int)
    finished_ok = Signal(str)
    failed = Signal(str, str)

    def __init__(self, project: ProjectData, output_path: Path, settings: RenderSettings, mode: str):
        super().__init__()
        self.project = project
        self.output_path = output_path
        self.settings = settings
        self.mode = mode

    def run(self):
        logger.info(
            "Worker: запуск генерации, режим=%s, orientation=%s, background=%s, threads=%d, chunk=%d",
            self.mode,
            self.project.orientation,
            self.project.background_mode,
            self.settings.thread_count,
            self.settings.frame_chunk_size,
        )
        try:
            duration = validate_project(self.project)
            palette = extract_dominant_palette(Path(self.project.image_path))
            codec = render_video(
                self.project,
                palette,
                duration,
                self.output_path,
                self.settings,
                progress_callback=lambda value: self.progress.emit(value),
            )
            logger.info("Worker: генерация завершена, режим=%s, codec=%s", self.mode, codec)
            self.finished_ok.emit(
                f"{self.output_path}|{self.mode}|{codec}|{self.settings.thread_count}|{self.settings.frame_chunk_size}"
            )
        except ValidationError as exc:
            logger.exception("Worker: ошибка валидации")
            self.failed.emit(str(exc), "validation")
        except DependencyError as exc:
            logger.exception("Worker: ошибка зависимостей")
            self.failed.emit(str(exc), "dependencies")
        except RenderDependencyError as exc:
            logger.exception("Worker: ошибка зависимостей рендера")
            self.failed.emit(str(exc), "dependencies")
        except RenderError as exc:
            logger.exception("Worker: runtime-ошибка рендера")
            self.failed.emit(str(exc), "runtime")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Worker: неизвестная ошибка генерации")
            self.failed.emit(str(exc), "runtime")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lyric Video Generator")
        self.resize(980, 760)
        self.project = ProjectData()
        self._worker: RenderWorker | None = None
        self._auto_sync_worker: AutoSyncWorker | None = None
        self._auto_sync_input_audio_path: str = ""
        self.thread_count = 2
        self.chunk_size = 60
        self._build_ui()
        logger.info("Окно приложения инициализировано")

    def _build_ui(self):
        root = QWidget()
        main = QVBoxLayout(root)

        files = QGroupBox("Исходные файлы")
        files_layout = QGridLayout(files)
        self.audio_label = QLabel("Не выбран")
        self.image_label = QLabel("Не выбрано")
        btn_audio = QPushButton("Выбрать аудио")
        btn_image = QPushButton("Выбрать обложку")
        btn_audio.clicked.connect(self.pick_audio)
        btn_image.clicked.connect(self.pick_image)
        files_layout.addWidget(btn_audio, 0, 0)
        files_layout.addWidget(self.audio_label, 0, 1)
        files_layout.addWidget(btn_image, 1, 0)
        files_layout.addWidget(self.image_label, 1, 1)

        meta = QGroupBox("Метаданные")
        meta_layout = QFormLayout(meta)
        self.artist_input = QLineEdit()
        self.title_input = QLineEdit()
        self.date_input = QLineEdit()
        meta_layout.addRow("Исполнитель", self.artist_input)
        meta_layout.addRow("Название", self.title_input)
        meta_layout.addRow("Дата релиза", self.date_input)

        sync_box = QGroupBox("Синхронизация")
        sync_layout = QVBoxLayout(sync_box)
        self.sync_mode_combo = QComboBox()
        self.sync_mode_combo.addItems(["Ручная", "Автоматическая"])
        self.sync_mode_combo.currentTextChanged.connect(self._on_sync_mode_changed)

        self.sync_stack = QStackedWidget()
        manual_widget = QWidget()
        manual_layout = QVBoxLayout(manual_widget)
        manual_layout.addWidget(QLabel("Ручной ввод таймингов в таблице ниже"))

        auto_widget = QWidget()
        auto_layout = QVBoxLayout(auto_widget)
        self.auto_text = QPlainTextEdit()
        self.auto_text.setPlaceholderText("Вставьте полный текст трека (по одной строке на строку)")
        self.auto_text.textChanged.connect(self._on_auto_text_changed)
        self.auto_file_btn = QPushButton("Загрузить текст из файла")
        self.auto_sync_btn = QPushButton("Синхронизировать автоматически")
        self.auto_file_btn.clicked.connect(self.load_auto_text_file)
        self.auto_sync_btn.clicked.connect(self.run_auto_sync)
        auto_layout.addWidget(QLabel("Текст трека для авто-синхронизации"))
        auto_layout.addWidget(self.auto_text)
        auto_buttons = QHBoxLayout()
        auto_buttons.addWidget(self.auto_file_btn)
        auto_buttons.addWidget(self.auto_sync_btn)
        auto_layout.addLayout(auto_buttons)

        self.sync_stack.addWidget(manual_widget)
        self.sync_stack.addWidget(auto_widget)
        sync_layout.addWidget(QLabel("Режим синхронизации"))
        sync_layout.addWidget(self.sync_mode_combo)
        sync_layout.addWidget(self.sync_stack)

        lyrics = QGroupBox("Текст и тайминги")
        lyrics_layout = QVBoxLayout(lyrics)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Время (мм:сс)", "Строка"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        add_row = QPushButton("Добавить строку")
        del_row = QPushButton("Удалить строку")
        add_row.clicked.connect(self.add_row)
        del_row.clicked.connect(self.delete_row)
        row_buttons = QHBoxLayout()
        row_buttons.addWidget(add_row)
        row_buttons.addWidget(del_row)
        lyrics_layout.addWidget(self.table)
        lyrics_layout.addLayout(row_buttons)

        actions = QGroupBox("Генерация")
        actions_layout = QVBoxLayout(actions)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Preview", "Final"])
        self.mode_combo.setCurrentText("Final")
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(["Вертикальное (9:16)", "Горизонтальное (16:9)"])
        self.background_combo = QComboBox()
        self.background_combo.addItems(["Мягкий градиент", "Динамический BPM-фон"])
        self.settings_button = QPushButton("Настройки производительности")
        self.settings_button.clicked.connect(self.open_performance_settings)
        self.status_label = QLabel("Режим: Final")
        self.mode_combo.currentTextChanged.connect(self._refresh_status)
        self.orientation_combo.currentTextChanged.connect(self._refresh_status)
        self.background_combo.currentTextChanged.connect(self._refresh_status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.generate_button = QPushButton("Сгенерировать видео")
        self.generate_button.clicked.connect(self.generate)
        actions_layout.addWidget(QLabel("Режим рендера"))
        actions_layout.addWidget(self.mode_combo)
        actions_layout.addWidget(QLabel("Ориентация видео"))
        actions_layout.addWidget(self.orientation_combo)
        actions_layout.addWidget(QLabel("Режим фона"))
        actions_layout.addWidget(self.background_combo)
        actions_layout.addWidget(self.settings_button)
        actions_layout.addWidget(self.status_label)
        actions_layout.addWidget(self.generate_button)
        actions_layout.addWidget(self.progress)

        main.addWidget(files)
        main.addWidget(meta)
        main.addWidget(sync_box)
        main.addWidget(lyrics)
        main.addWidget(actions)

        self.setCentralWidget(root)
        self._refresh_status()

    def _on_sync_mode_changed(self, mode: str):
        is_auto = mode == "Автоматическая"
        self.sync_stack.setCurrentIndex(1 if is_auto else 0)
        self.project.sync_mode = "auto" if is_auto else "manual"
        logger.info("Выбран режим синхронизации: %s", self.project.sync_mode)

    def _refresh_status(self):
        orientation = "vertical" if self.orientation_combo.currentIndex() == 0 else "horizontal"
        mode = self.mode_combo.currentText()
        profile = RENDER_PROFILES[(orientation, mode)]
        self.status_label.setText(
            f"Режим: {mode} | {profile.width}x{profile.height}@{profile.fps} | "
            f"Потоки: {self.thread_count}, чанк: {self.chunk_size}"
        )

    def open_performance_settings(self):
        dialog = PerformanceSettingsDialog(self.thread_count, self.chunk_size, self)
        if dialog.exec():
            self.thread_count = dialog.threads_spin.value()
            self.chunk_size = dialog.chunk_spin.value()
            self._refresh_status()
            logger.info("Обновлены настройки производительности: threads=%d, chunk=%d", self.thread_count, self.chunk_size)

    def add_row(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        time_item = QTableWidgetItem("00:00")
        text_item = QTableWidgetItem("")
        time_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        text_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.table.setItem(row, 0, time_item)
        self.table.setItem(row, 1, text_item)
        logger.info("Добавлена строка текста: row=%d", row)

    def delete_row(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)
            logger.info("Удалена строка текста: row=%d", row)

    def _reset_auto_sync_state(self):
        if self.project.lyrics_autofilled:
            logger.info("Сброс флага авто-синхронизации из-за изменения входных данных")
        self.project.lyrics_autofilled = False
        self.project.auto_sync_audio_path = ""

    def _on_auto_text_changed(self):
        self._reset_auto_sync_state()

    def load_auto_text_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите текст", "", "Text (*.txt *.lrc)")
        if not path:
            return
        text_path = Path(path)
        encodings = ("utf-8", "utf-8-sig", "cp1251")
        for encoding in encodings:
            try:
                text = text_path.read_text(encoding=encoding)
                self.auto_text.setPlainText(text)
                logger.info("Загружен текст для авто-синхронизации: %s (encoding=%s)", path, encoding)
                return
            except UnicodeDecodeError:
                logger.warning("Не удалось декодировать файл %s с encoding=%s", path, encoding)
            except OSError as exc:
                logger.warning("Не удалось прочитать файл %s: %s", path, exc)
                QMessageBox.warning(self, "Автосинхронизация", f"Не удалось прочитать файл:\n{exc}")
                return

        logger.warning("Файл %s не удалось декодировать поддерживаемыми кодировками", path)
        QMessageBox.warning(
            self,
            "Автосинхронизация",
            "Не удалось прочитать текстовый файл. Проверьте кодировку (поддерживаются UTF-8/UTF-8 BOM/CP1251).",
        )

    def _fill_lyrics_table(self, lines: list[LyricLine]):
        self.table.setRowCount(0)
        for line in lines:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(line.start_time))
            self.table.setItem(row, 1, QTableWidgetItem(line.text))

    def run_auto_sync(self):
        if not self.project.audio_path:
            QMessageBox.warning(self, "Автосинхронизация", "Сначала выберите аудиофайл.")
            return

        full_text = self.auto_text.toPlainText().strip()
        if not full_text:
            QMessageBox.warning(self, "Автосинхронизация", "Введите полный текст трека.")
            return

        self.auto_sync_btn.setEnabled(False)
        self.status_label.setText("Автосинхронизация: выполняется анализ...")
        logger.info("Запуск авто-синхронизации")
        self._auto_sync_input_audio_path = str(self.project.audio_path)
        self._auto_sync_worker = AutoSyncWorker(Path(self._auto_sync_input_audio_path), full_text)
        self._auto_sync_worker.done.connect(self._on_auto_sync_done)
        self._auto_sync_worker.failed.connect(self._on_auto_sync_failed)
        self._auto_sync_worker.start()

    def _on_auto_sync_done(self, lines: list[LyricLine]):
        self.auto_sync_btn.setEnabled(True)
        current_audio_path = str(self.project.audio_path) if self.project.audio_path else ""
        if self._auto_sync_input_audio_path != current_audio_path:
            self.status_label.setText("Автосинхронизация: результат устарел и был отброшен")
            logger.warning(
                "Автосинхронизация отброшена из-за смены аудио: worker_audio=%s, current_audio=%s",
                self._auto_sync_input_audio_path,
                current_audio_path,
            )
            return

        if len(lines) < 2:
            QMessageBox.warning(self, "Автосинхронизация", "Получено слишком мало строк. Попробуйте другой текст/аудио.")
            return

        self._fill_lyrics_table(lines)
        self.project.lyrics_autofilled = True
        self.project.auto_sync_audio_path = str(self.project.audio_path) if self.project.audio_path else ""
        self.status_label.setText(f"Автосинхронизация завершена: строк={len(lines)}")
        logger.info("Автосинхронизация успешна, таблица заполнена: lines=%d", len(lines))

    def _on_auto_sync_failed(self, error: str):
        self.auto_sync_btn.setEnabled(True)
        self.status_label.setText("Автосинхронизация: ошибка")
        logger.warning("Автосинхронизация завершилась ошибкой: %s", error)
        QMessageBox.warning(self, "Автосинхронизация", error)

    def pick_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите аудио", "", "Audio (*.mp3 *.wav *.flac *.m4a)")
        if path:
            self.project.audio_path = Path(path)
            self.audio_label.setText(Path(path).name)
            self._reset_auto_sync_state()
            logger.info("Выбран аудиофайл: %s", path)

    def pick_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите обложку", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if path:
            self.project.image_path = Path(path)
            self.image_label.setText(Path(path).name)
            logger.info("Выбрана обложка: %s", path)

    def _collect_project(self):
        lyrics: list[LyricLine] = []
        for row in range(self.table.rowCount()):
            t_item = self.table.item(row, 0)
            l_item = self.table.item(row, 1)
            if t_item and l_item:
                normalized_text = " ".join(l_item.text().splitlines())
                l_item.setText(normalized_text)
                l_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                lyrics.append(LyricLine(start_time=t_item.text(), text=normalized_text))

        self.project.artist = self.artist_input.text()
        self.project.title = self.title_input.text()
        self.project.release_date = self.date_input.text()
        self.project.lyrics = lyrics
        self.project.sync_mode = "auto" if self.sync_mode_combo.currentIndex() == 1 else "manual"
        self.project.auto_sync_lyrics_text = self.auto_text.toPlainText().strip()
        self.project.orientation = "vertical" if self.orientation_combo.currentIndex() == 0 else "horizontal"
        self.project.background_mode = "soft_gradient" if self.background_combo.currentIndex() == 0 else "bpm_dynamic"

        logger.info(
            "Собраны данные проекта: artist='%s', title='%s', lines=%d, orientation=%s, sync_mode=%s, background=%s",
            self.project.artist,
            self.project.title,
            len(lyrics),
            self.project.orientation,
            self.project.sync_mode,
            self.project.background_mode,
        )

    def _selected_render_settings(self) -> tuple[str, RenderSettings]:
        mode = self.mode_combo.currentText()
        orientation = "vertical" if self.orientation_combo.currentIndex() == 0 else "horizontal"
        profile = RENDER_PROFILES[(orientation, mode)]
        settings = RenderSettings.from_profile(profile)
        settings.thread_count = self.thread_count
        settings.frame_chunk_size = self.chunk_size
        return mode, settings

    def generate(self):
        self._collect_project()
        output, _ = QFileDialog.getSaveFileName(self, "Сохранить видео", "lyrics_video.mp4", "Video (*.mp4)")
        if not output:
            logger.info("Генерация отменена: путь сохранения не выбран")
            return

        mode, settings = self._selected_render_settings()
        logger.info(
            "Старт генерации в файл: %s, режим=%s, orientation=%s, %dx%d@%dfps, background=%s, threads=%d, chunk=%d",
            output,
            mode,
            settings.orientation,
            settings.width,
            settings.height,
            settings.fps,
            self.project.background_mode,
            settings.thread_count,
            settings.frame_chunk_size,
        )
        self.generate_button.setEnabled(False)
        self.progress.setValue(0)
        self._worker = RenderWorker(self.project, Path(output), settings, mode)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.finished_ok.connect(self._on_success)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_success(self, payload: str):
        output, mode, codec, threads, chunk = payload.split("|", 4)
        logger.info(
            "Генерация успешно завершена: %s, режим=%s, codec=%s, threads=%s, chunk=%s",
            output,
            mode,
            codec,
            threads,
            chunk,
        )
        self._refresh_status()
        self.generate_button.setEnabled(True)
        QMessageBox.information(
            self,
            "Готово",
            f"Видео сохранено:\n{output}\n\nРежим: {mode}\nКодек: {codec}\nПотоки: {threads}\nЧанк: {chunk}",
        )

    def _on_fail(self, error: str, error_type: str):
        logger.error("Генерация завершена с ошибкой [%s]: %s", error_type, error)
        self.status_label.setText("Ошибка рендера")
        self.generate_button.setEnabled(True)
        if error_type == "validation":
            QMessageBox.warning(self, "Ошибка валидации", error)
            return
        if error_type == "dependencies":
            QMessageBox.warning(self, "Ошибка зависимостей", error)
            return
        QMessageBox.critical(self, "Ошибка рендера", error)
