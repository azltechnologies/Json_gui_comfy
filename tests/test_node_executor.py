"""Tests for json_gui.scripts.node_executor."""

from __future__ import annotations

from queue import Queue
from unittest.mock import MagicMock

import pytest
import torch

from json_gui import c_logger, p_logger
from json_gui.scripts.mimic import MimicNode
from json_gui.scripts.node_executor import NodeExecutor, _move_tensors_to_device


class DependencyNode(MimicNode[str]):
    """Simple dependency node used in executor tests."""

    @classmethod
    def _class_param_definitions(cls):
        return []

    @classmethod
    def key(cls) -> str:
        return "dependency"

    def __init__(self, label: str = "") -> None:
        super().__init__()
        self.update(label=label)

    def _update_impl(self, label: str) -> None:
        self._label = label

    def _process_impl(self) -> str:
        return self._label


class ExecutionNode(MimicNode[str]):
    """Node used to exercise executor reconstruction."""

    @classmethod
    def _class_param_definitions(cls):
        return []

    @classmethod
    def key(cls) -> str:
        return "execution"

    def __init__(self, prefix: str = "") -> None:
        super().__init__()
        self.update(prefix=prefix)

    def _update_impl(self, prefix: str) -> None:
        self._prefix = prefix

    def _process_impl(self, value: str, dependency: DependencyNode) -> str:
        return f"{self._prefix}-{value}-{dependency.process()}"


def test_move_tensors_to_device_recurses_and_filters_callables() -> None:
    """Tensor movement should recurse into containers and replace callables with None."""
    tensor = torch.tensor([1.0, 2.0])
    payload = {
        "tensor": tensor,
        "items": [tensor, lambda: None],
        "nested": ({"callable": lambda: None},),
    }

    result = _move_tensors_to_device(payload, torch.device("cpu"))

    assert result["tensor"].device.type == "cpu"
    assert result["items"][1] is None
    assert result["nested"][0]["callable"] is None


def test_node_executor_init_moves_tensors_and_strips_save_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The constructor should sanitize tensors and nested MimicNode instances."""
    node = ExecutionNode()
    node.update(prefix="pre")
    provider = DependencyNode()
    provider.update(label="dep")
    provider.save_tensor = MagicMock()

    fake_context = type("Ctx", (), {"Queue": Queue, "Process": object})()
    monkeypatch.setattr(p_logger, "get_mp_context", lambda: fake_context)
    monkeypatch.setattr(p_logger, "get_log_queue", Queue)

    executor = NodeExecutor(
        node,
        {"value": torch.tensor([1.0]), "dependency": provider},
        {DependencyNode: provider.init_args},
        {"created_images": [], "last_saved_to_temp": None},
    )

    assert executor.node_process_args["value"].device.type == "cpu"
    assert "dependency" not in executor.node_process_args
    assert provider._save_tensor is None


def test_execute_returns_result_from_child_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """execute should start the worker and return the child result."""
    node = ExecutionNode()
    node.update(prefix="pre")
    provider = DependencyNode()
    provider.update(label="dep")

    class FakeProcess:
        def __init__(self, target, name, args):
            self.target = target
            self.args = args
            self.started = False
            self._alive = False

        def start(self) -> None:
            self.started = True
            self._alive = True
            self.target(*self.args)
            self._alive = False

        def is_alive(self) -> bool:
            return self._alive

        def terminate(self) -> None:
            self._alive = False

        def join(self, timeout=None) -> None:
            self._alive = False

    fake_context = type("Ctx", (), {"Queue": Queue, "Process": FakeProcess})()
    monkeypatch.setattr(p_logger, "get_mp_context", lambda: fake_context)
    monkeypatch.setattr(p_logger, "get_log_queue", Queue)
    monkeypatch.setattr(c_logger, "worker_wrapper", lambda func, log_queue, flow_queue: func(flow_queue))

    def fake_target(node_cls, node_init_args, node_exec_args, raw_nodes_serialized, save_call, save_data, result_queue):
        result_queue.put(("success", save_data, "result"))

    monkeypatch.setattr(NodeExecutor, "_node_executor_target", staticmethod(fake_target))

    executor = NodeExecutor(
        node,
        {"value": "mid"},
        {DependencyNode: provider.init_args},
        {"created_images": [], "last_saved_to_temp": None},
    )

    result, saved_data = executor.execute(lambda *args, **kwargs: None)

    assert result == "result"
    assert saved_data["created_images"] == []


def test_execute_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """execute should raise TimeoutError when the child never returns."""
    node = ExecutionNode()
    node.update(prefix="pre")

    class HangingProcess:
        def __init__(self, target, name, args):
            self.target = target
            self.args = args

        def start(self) -> None:
            return None

        def is_alive(self) -> bool:
            return True

        def terminate(self) -> None:
            return None

        def join(self, timeout=None) -> None:
            return None

    fake_context = type("Ctx", (), {"Queue": Queue, "Process": HangingProcess})()
    monkeypatch.setattr(p_logger, "get_mp_context", lambda: fake_context)
    monkeypatch.setattr(p_logger, "get_log_queue", Queue)
    monkeypatch.setattr(c_logger, "worker_wrapper", lambda func, log_queue, flow_queue: func(flow_queue))
    monkeypatch.setattr(NodeExecutor, "_node_executor_target", staticmethod(lambda *args, **kwargs: None))

    executor = NodeExecutor(node, {"value": "mid"}, {}, {"created_images": [], "last_saved_to_temp": None})

    with pytest.raises(TimeoutError):
        executor.execute(lambda *args, **kwargs: None, timeout=0.01, poll_interval=0.001)


def test_execute_target_node_reconstructs_dependency_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """The target node should rebuild the node and its dependencies before execution."""
    node = ExecutionNode()
    node.update(prefix="pre")
    dependency = DependencyNode()
    dependency.update(label="dep")
    result_queue: Queue = Queue()

    monkeypatch.setattr("signal.pause", lambda: None)

    NodeExecutor._node_executor_target(
        ExecutionNode,
        node.init_args,
        {"value": "mid"},
        {"dependency": (DependencyNode, dependency.init_args)},
        lambda *args, **kwargs: None,
        {"created_images": [], "last_saved_to_temp": None},
        result_queue,
    )

    status, saved_data, output = result_queue.get_nowait()
    assert status == "success"
    assert saved_data["created_images"] == []
    assert output == "pre-mid-dep"
