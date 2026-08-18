"""Tests for json_gui.scripts.mimic."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from json_gui.scripts.mimic import DataWrapper, MimicNode, safe_reference_compare
from json_gui.utils import EndOfFlowException


def build_value(multiplier: int) -> int:
    """Simple top-level builder used for DataWrapper tests."""
    return multiplier * 2


class ProviderNode(MimicNode[int]):
    """Simple dependency node used in class param tests."""

    _calls = 0

    @classmethod
    def _class_param_definitions(cls):
        return []

    @classmethod
    def key(cls) -> str:
        return "provider"

    def _update_impl(self, value: int) -> None:
        self._value = value

    def _process_impl(self) -> int:
        ProviderNode._calls += 1
        return self._value


class ConsumerNode(MimicNode[int]):
    """Node that receives a provider through ClassParam injection."""

    @classmethod
    def _class_param_definitions(cls):
        return [cls.build_class_param(ProviderNode, lambda inst: {"extra": inst.process()})]

    @classmethod
    def key(cls) -> str:
        return "consumer"

    def _update_impl(self, base: int) -> None:
        self._base = base

    def _process_impl(self, value: int, extra: int) -> int:
        return self._base + value + extra


class CacheNode(MimicNode[int]):
    """Node used to validate caching behavior."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @classmethod
    def _class_param_definitions(cls):
        return []

    @classmethod
    def key(cls) -> str:
        return "cache"

    def _update_impl(self, offset: int) -> None:
        self._offset = offset

    def _process_impl(self, value: int) -> int:
        self.calls += 1
        return value + self._offset


class EndOfFlowNode(MimicNode[int]):
    """Node that stops immediately with an EndOfFlowException."""

    @classmethod
    def _class_param_definitions(cls):
        return []

    @classmethod
    def key(cls) -> str:
        return "end_of_flow"

    def _update_impl(self) -> None:
        return None

    def _process_impl(self) -> int:
        raise EndOfFlowException(3)


def test_data_wrapper_supports_lazy_evaluation_and_validation() -> None:
    """DataWrapper should build lazily and reject non-serializable args."""

    wrapper = DataWrapper(builder=build_value, args={"multiplier": 5})
    assert wrapper.get() == 10

    with pytest.raises(ValueError):
        DataWrapper(builder=build_value, args={"multiplier": lambda: None})


def test_safe_reference_compare_handles_nested_values_and_tensors() -> None:
    """Comparison should recurse into lists, tuples, dicts, and tensors."""
    tensor_a = torch.tensor([1, 2])
    tensor_b = torch.tensor([1, 2])
    tensor_c = torch.tensor([2, 3])
    wrapper_a = DataWrapper(value=1, identifier="same")
    wrapper_b = DataWrapper(value=2, identifier="same")

    assert safe_reference_compare(wrapper_a, wrapper_b)
    assert safe_reference_compare(tensor_a, tensor_b)
    assert not safe_reference_compare(tensor_a, tensor_c)
    assert safe_reference_compare((1, [2, {"x": 3}]), (1, [2, {"x": 3}]))
    assert not safe_reference_compare({"a": 1}, {"a": 2})


def test_mimic_node_update_and_process_use_cache() -> None:
    """Repeated calls with the same arguments should use the cached result."""
    node = CacheNode()
    node.update(offset=4)

    first = node.process(3)
    second = node.process(3)

    assert first == 7
    assert second == 7
    assert node.calls == 1

    node.update(offset=5)
    third = node.process(3)
    assert third == 8
    assert node.calls == 2


def test_mimic_node_exec_node_routes_to_sync_or_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    """exec_node should dispatch to the correct execution path."""
    node = CacheNode()
    node.update(offset=1)

    sync_mock = MagicMock(return_value="sync")
    spawn_mock = MagicMock(return_value="spawn")
    monkeypatch.setattr(node, "_exec_node_sync", sync_mock)
    monkeypatch.setattr(node, "_exec_node_spawn", spawn_mock)

    MimicNode._do_multiprocess = False
    assert node.exec_node({"value": 2}, []) == "sync"
    sync_mock.assert_called_once()
    assert sync_mock.call_args.args[0] == {"value": 2}

    sync_mock.reset_mock()

    MimicNode._do_multiprocess = True
    assert node.exec_node({"value": 2}, [node]) == "spawn"
    spawn_mock.assert_called_once()


def test_class_param_injects_dependency_into_process() -> None:
    """ClassParam should pop the dependency and inject derived kwargs."""
    ProviderNode._calls = 0
    provider = ProviderNode()
    provider.update(value=4)
    consumer = ConsumerNode()
    consumer.update(base=3)

    assert consumer.process(provider=provider, value=5) == 12
    assert ProviderNode._calls == 1


def test_process_surfaces_end_of_flow_exception_result() -> None:
    """EndOfFlowException raised by _process_impl should propagate cleanly."""
    node = EndOfFlowNode()
    node.update()

    with pytest.raises(EndOfFlowException) as exc_info:
        node.process()

    assert exc_info.value.steps == 3
    assert exc_info.value.result is None


def test_save_all_unsaved_tensors_raises_end_of_flow_after_saving() -> None:
    """Saving tensors should continue through all items and re-raise EndOfFlowException."""
    node = CacheNode()
    node.save_tensor = MagicMock(side_effect=[None, EndOfFlowException(2)])
    node.add_unsaved_tensor(torch.ones((1, 1)), "one")
    node.add_unsaved_tensor(torch.ones((1, 1)), "two")

    with pytest.raises(EndOfFlowException) as exc_info:
        node.save_all_unsaved_tensors()

    assert exc_info.value.steps == 2
    assert node.unsaved_tensors == []
    assert node.save_tensor.call_count == 2
