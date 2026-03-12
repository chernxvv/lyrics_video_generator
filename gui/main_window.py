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
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QVBoxLayout,
    QWidget,
)

from core.image_analysis import extract_dominant_palette
from core.render import RenderDependencyError, RenderError, render_video
from core.validation import DependencyError, ValidationError, validate_project
from models import LyricLine, ProjectData, RenderSettings

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
            "Worker: запуск генерации, режим=%s, threads=%d, chunk=%d",
            self.mode,
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
        self.settings_button = QPushButton("Настройки производительности")
        self.settings_button.clicked.connect(self.open_performance_settings)
        self.status_label = QLabel("Режим: Final")
        self.mode_combo.currentTextChanged.connect(lambda mode: self.status_label.setText(f"Режим: {mode}"))
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.generate_button = QPushButton("Сгенерировать видео")
        self.generate_button.clicked.connect(self.generate)
        actions_layout.addWidget(QLabel("Режим рендера"))
        actions_layout.addWidget(self.mode_combo)
        actions_layout.addWidget(self.settings_button)
        actions_layout.addWidget(self.status_label)
        actions_layout.addWidget(self.generate_button)
        actions_layout.addWidget(self.progress)

        main.addWidget(files)
        main.addWidget(meta)
        main.addWidget(lyrics)
        main.addWidget(actions)

        self.setCentralWidget(root)

    def open_performance_settings(self):
        dialog = PerformanceSettingsDialog(self.thread_count, self.chunk_size, self)
        if dialog.exec():
            self.thread_count = dialog.threads_spin.value()
            self.chunk_size = dialog.chunk_spin.value()
            self.status_label.setText(
                f"Режим: {self.mode_combo.currentText()} | Потоки: {self.thread_count}, чанк: {self.chunk_size}"
            )
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

    def pick_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите аудио", "", "Audio (*.mp3 *.wav *.flac *.m4a)")
        if path:
            self.project.audio_path = Path(path)
            self.audio_label.setText(Path(path).name)
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

        logger.info(
            "Собраны данные проекта: artist='%s', title='%s', lines=%d",
            self.project.artist,
            self.project.title,
            len(lyrics),
        )

    def _selected_render_settings(self) -> tuple[str, RenderSettings]:
        mode = self.mode_combo.currentText()
        settings = RenderSettings.preview() if mode == "Preview" else RenderSettings.final()
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
            "Старт генерации в файл: %s, режим=%s, %dx%d@%dfps, threads=%d, chunk=%d",
            output,
            mode,
            settings.width,
            settings.height,
            settings.fps,
            settings.thread_count,
            settings.frame_chunk_size,
        )
        self.status_label.setText(f"Рендер: {mode}, threads={settings.thread_count}, chunk={settings.frame_chunk_size}")
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
        self.status_label.setText(f"Готово: {mode}, codec={codec}, threads={threads}, chunk={chunk}")
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
