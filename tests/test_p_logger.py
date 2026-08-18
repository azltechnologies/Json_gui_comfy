"""Tests for json_gui.p_logger."""

from __future__ import annotations

import logging
import sys
from queue import Queue

from json_gui import p_logger


def test_pretty_traceback_formatter_rewrites_traceback_locations() -> None:
    """The formatter should shorten file paths in tracebacks."""
    formatter = p_logger.PrettyTracebackFormatter("%(message)s")

    try:
        raise RuntimeError("boom")
    except RuntimeError:
        output = formatter.formatException(sys.exc_info())

    assert ", line " not in output
    assert " in test_pretty_traceback_formatter_rewrites_traceback_locations" in output


def test_error_and_default_filters_split_log_levels() -> None:
    """ErrorFilter should keep only error records and DefaultFilter should drop them."""
    error_filter = p_logger.ErrorFilter()
    default_filter = p_logger.DefaultFilter()

    info_record = logging.LogRecord("test", logging.INFO, __file__, 10, "info", (), None)
    error_record = logging.LogRecord("test", logging.ERROR, __file__, 11, "error", (), None)

    assert not error_filter.filter(info_record)
    assert error_filter.filter(error_record)
    assert default_filter.filter(info_record)
    assert not default_filter.filter(error_record)


def test_poll_log_queue_drains_messages(monkeypatch) -> None:
    """poll_log_queue should drain the global queue and hand records to the logger."""
    queue = Queue()
    record = logging.LogRecord("test", logging.INFO, __file__, 12, "queued", (), None)
    queue.put(record)
    monkeypatch.setattr(p_logger, "LOG_QUEUE", queue)

    handled = []
    monkeypatch.setattr(p_logger.logger, "handle", lambda rec: handled.append(rec.msg))

    assert p_logger.poll_log_queue() == 1
    assert handled == ["queued"]


def test_getters_return_expected_singletons() -> None:
    """Accessor helpers should expose the configured queue and spawn context."""
    assert p_logger.get_log_queue() is p_logger.LOG_QUEUE
    assert p_logger.get_mp_context().Queue is not None
