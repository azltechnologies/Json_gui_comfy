"""Tests for json_gui.c_logger."""

from __future__ import annotations

import io
import logging
from queue import Queue
from unittest.mock import MagicMock

from json_gui import c_logger


def test_stream_to_logger_buffers_lines_and_flushes() -> None:
    """StreamToLogger should emit complete lines and flush the final partial line."""
    logger = MagicMock()
    stream = c_logger.StreamToLogger(logger, logging.INFO)

    stream.write("first line\nsecond")
    stream.flush()

    assert logger.log.call_args_list[0].args[1] == "first line"
    assert logger.log.call_args_list[1].args[1] == "second"


def test_setup_child_logger_replaces_root_handlers() -> None:
    """Child logger setup should clear inherited handlers and install a queue handler."""
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    queue = Queue()

    try:
        c_logger.setup_child_logger(queue)
        assert len(root.handlers) == 1
        assert root.handlers[0].__class__.__name__ == "QueueHandler"
    finally:
        root.handlers = original_handlers


def test_worker_wrapper_executes_callable_and_redirects_streams(monkeypatch) -> None:
    """The worker wrapper should configure logging and return the callable result."""
    log_queue = Queue()
    flow_queue = Queue()
    monkeypatch.setattr(c_logger.sys, "stdout", io.StringIO())
    monkeypatch.setattr(c_logger.sys, "stderr", io.StringIO())

    callback = MagicMock(return_value=["done"])

    assert c_logger.worker_wrapper(callback, log_queue, flow_queue) == ["done"]
    callback.assert_called_once_with(flow_queue)

