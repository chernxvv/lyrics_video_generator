from pathlib import Path
import sys
import logging

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.logging_config import setup_logging


def test_setup_logging_emits_init_message(caplog) -> None:
    caplog.set_level(logging.INFO)

    setup_logging()

    assert any("Логирование инициализировано" in rec.message for rec in caplog.records)
