import logging

import main


def test_setup_logging_writes_to_file(tmp_path):
    root = logging.getLogger()
    old_level = root.level
    handler = main.setup_logging(log_dir=tmp_path)
    try:
        logging.getLogger("datasmith.test").info("hello log")
        handler.flush()
        text = (tmp_path / "datasmith.log").read_text(encoding="utf-8")
        assert "hello log" in text
    finally:
        root.removeHandler(handler)
        handler.close()
        root.setLevel(old_level)