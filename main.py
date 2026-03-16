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
