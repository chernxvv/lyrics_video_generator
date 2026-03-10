from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
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
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.image_analysis import extract_dominant_palette
from core.render import render_video
from core.validation import validate_project
from models import LyricLine, ProjectData, RenderSettings


class RenderWorker(QThread):
    progress = Signal(int)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, project: ProjectData, output_path: Path):
        super().__init__()
        self.project = project
        self.output_path = output_path

    def run(self):
        try:
            duration = validate_project(self.project)
            palette = extract_dominant_palette(Path(self.project.image_path))
            render_video(
                self.project,
                palette,
                duration,
                self.output_path,
                RenderSettings(),
                progress_callback=lambda value: self.progress.emit(value),
            )
            self.finished_ok.emit(str(self.output_path))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lyric Video Generator")
        self.resize(980, 760)
        self.project = ProjectData()
        self._worker: RenderWorker | None = None
        self._build_ui()

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
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.generate_button = QPushButton("Сгенерировать видео")
        self.generate_button.clicked.connect(self.generate)
        actions_layout.addWidget(self.generate_button)
        actions_layout.addWidget(self.progress)

        main.addWidget(files)
        main.addWidget(meta)
        main.addWidget(lyrics)
        main.addWidget(actions)

        self.setCentralWidget(root)

    def add_row(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem("00:00"))
        self.table.setItem(row, 1, QTableWidgetItem(""))

    def delete_row(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def pick_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите аудио", "", "Audio (*.mp3 *.wav *.flac *.m4a)")
        if path:
            self.project.audio_path = Path(path)
            self.audio_label.setText(Path(path).name)

    def pick_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите обложку", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if path:
            self.project.image_path = Path(path)
            self.image_label.setText(Path(path).name)

    def _collect_project(self):
        lyrics: list[LyricLine] = []
        for row in range(self.table.rowCount()):
            t_item = self.table.item(row, 0)
            l_item = self.table.item(row, 1)
            if t_item and l_item:
                lyrics.append(LyricLine(start_time=t_item.text(), text=l_item.text()))

        self.project.artist = self.artist_input.text()
        self.project.title = self.title_input.text()
        self.project.release_date = self.date_input.text()
        self.project.lyrics = lyrics

    def generate(self):
        self._collect_project()
        output, _ = QFileDialog.getSaveFileName(self, "Сохранить видео", "lyrics_video.mp4", "Video (*.mp4)")
        if not output:
            return

        self.generate_button.setEnabled(False)
        self.progress.setValue(0)
        self._worker = RenderWorker(self.project, Path(output))
        self._worker.progress.connect(self.progress.setValue)
        self._worker.finished_ok.connect(self._on_success)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_success(self, output: str):
        self.generate_button.setEnabled(True)
        QMessageBox.information(self, "Готово", f"Видео сохранено:\n{output}")

    def _on_fail(self, error: str):
        self.generate_button.setEnabled(True)
        QMessageBox.critical(self, "Ошибка", error)
