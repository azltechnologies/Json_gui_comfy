"""Tests for json_gui.typedicts."""

from __future__ import annotations

from json_gui.typedicts import BodyDict, get_empty_creation_dict, is_bodydict, is_creation_dict, is_empty_creation_dict


def test_get_empty_creation_dict_returns_independent_dicts() -> None:
    """The helper should return a fresh empty creation dict each time."""
    first = get_empty_creation_dict()
    second = get_empty_creation_dict()

    assert first == {"args": [], "kwargs": {}}
    assert second == {"args": [], "kwargs": {}}
    assert first is not second
    assert first["args"] is not second["args"]
    assert first["kwargs"] is not second["kwargs"]


def test_is_creation_dict_accepts_only_expected_shape() -> None:
    """Creation dicts should have args and kwargs with the right container types."""
    assert is_creation_dict({"args": [], "kwargs": {}})
    assert not is_creation_dict({"args": (), "kwargs": {}})
    assert not is_creation_dict({"args": [], "kwargs": []})
    assert not is_creation_dict({"kwargs": {}})
    assert not is_creation_dict(None)


def test_is_empty_creation_dict_requires_empty_containers() -> None:
    """Empty creation dicts must validate the inner containers as empty."""
    assert is_empty_creation_dict({"args": [], "kwargs": {}})
    assert not is_empty_creation_dict({"args": [1], "kwargs": {}})
    assert not is_empty_creation_dict({"args": [], "kwargs": {"x": 1}})


def test_is_bodydict_validates_nested_props() -> None:
    """Body dicts must expose a props mapping with node-like dictionaries."""
    valid: BodyDict = {"props": {"prompt": {"type": "string", "isArray": False, "props": {}}}}

    assert is_bodydict(valid)
    assert not is_bodydict({"props": {"prompt": {"type": "string", "props": {}}}})
    assert not is_bodydict({"props": {"prompt": []}})
    assert not is_bodydict(None)
