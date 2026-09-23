import logging

from provenance.logging_setup import close_logger, setup_stage_logger

from conftest import make_scratch_dir


def test_setup_stage_logger_writes_debug_to_file_but_filters_console_level():
    scratch = make_scratch_dir("logging_setup_basic")
    work_dir = scratch / "work"

    logger = setup_stage_logger("unit_test_stage", work_dir, "run001", console_level=logging.WARNING)
    logger.debug("debug detail")
    logger.warning("a warning")
    close_logger(logger)

    log_path = work_dir / "logs" / "unit_test_stage" / "run001.log"
    contents = log_path.read_text()
    assert "debug detail" in contents
    assert "a warning" in contents


def test_console_handler_level_is_set_as_requested_and_file_handler_stays_at_debug():
    scratch = make_scratch_dir("logging_setup_levels")
    work_dir = scratch / "work"

    logger = setup_stage_logger("unit_test_stage", work_dir, "run002", console_level=logging.ERROR)
    file_handler = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
    console_handler = next(
        h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    )
    assert file_handler.level == logging.DEBUG
    assert console_handler.level == logging.ERROR
    close_logger(logger)


def test_latest_symlink_points_to_the_most_recent_run():
    scratch = make_scratch_dir("logging_setup_symlink")
    work_dir = scratch / "work"

    logger1 = setup_stage_logger("unit_test_stage", work_dir, "run_a")
    logger1.info("first run")
    close_logger(logger1)

    logger2 = setup_stage_logger("unit_test_stage", work_dir, "run_b")
    logger2.info("second run")
    close_logger(logger2)

    latest_link = work_dir / "logs" / "unit_test_stage" / "unit_test_stage_latest.log"
    assert latest_link.is_symlink()
    assert latest_link.resolve().name == "run_b.log"
    assert "second run" in latest_link.read_text()


def test_close_logger_detaches_all_handlers():
    scratch = make_scratch_dir("logging_setup_close")
    work_dir = scratch / "work"

    logger = setup_stage_logger("unit_test_stage", work_dir, "run003")
    assert len(logger.handlers) == 2
    close_logger(logger)
    assert logger.handlers == []
