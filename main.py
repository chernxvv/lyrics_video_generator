import logging

from PySide6.QtWidgets import QApplication

from core.logging_config import setup_logging
from gui import MainWindow


def main() -> int:
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Запуск приложения")

    app = QApplication([])
    window = MainWindow()
    window.show()

    logger.info("GUI готов, вход в event loop")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
