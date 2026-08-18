"""Tests for json_gui.utils."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
from unittest.mock import patch

import pytest
import torch

from json_gui import utils
from json_gui.utils import (
    AbsFlow,
    EndOfFlowException,
    copy_images,
    get_flow_and_body_paths,
    is_unserializable_callable,
    save_image,
)


def test_is_unserializable_callable_detects_lambdas_closures_and_partials() -> None:
    """Callable serialization checks should flag lambdas, locals, and wrapped locals."""

    def outer() -> Callable[[], int]:
        value = 1

        def inner() -> int:
            return value

        return inner

    assert is_unserializable_callable(lambda: None)
    assert is_unserializable_callable(outer())
    assert is_unserializable_callable(utils.partial(lambda x: x, 1))
    assert not is_unserializable_callable(len)
    assert not is_unserializable_callable(42)


def test_get_flow_and_body_paths_validates_missing_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Flow and body paths should resolve inside the scripts folder and validate existence."""
    scripts_dir = tmp_path / "scripts"
    flow_dir = scripts_dir / "sample_flow"
    flow_dir.mkdir(parents=True)
    (flow_dir / "flow.py").write_text("print('hi')", encoding="utf-8")
    (flow_dir / "body.yml").write_text("props: {}", encoding="utf-8")

    monkeypatch.setattr(utils, "get_scripts_folder_path", lambda: str(scripts_dir))

    flow_path, body_path = get_flow_and_body_paths("sample_flow")

    assert flow_path == str(flow_dir / "flow.py")
    assert body_path == str(flow_dir / "body.yml")

    with pytest.raises(AssertionError):
        get_flow_and_body_paths("missing_flow")


def test_save_image_writes_file_and_raises_end_of_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Saving the last step should persist the image and stop the flow."""
    monkeypatch.setattr(utils.folder_paths, "get_temp_directory", lambda: str(tmp_path))
    data = {"created_images": [], "last_saved_to_temp": None}
    image = torch.ones((1, 2, 2, 3), dtype=torch.float32)

    with pytest.raises(EndOfFlowException) as exc_info:
        save_image(data, image, "node", steps=1, file_identifier="flow_1", is_temp=True)

    saved_path = tmp_path / "flow_1_node_0.png"
    assert saved_path.exists()
    assert data["created_images"] == [str(saved_path)]
    assert data["last_saved_to_temp"] is True
    assert exc_info.value.steps == 1


def test_copy_images_copies_only_new_files(tmp_path: Path) -> None:
    """Only the newly created files should be copied into the target naming scheme."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    src1 = source_dir / "flow_1_node_0.png"
    src2 = source_dir / "flow_1_node_1.png"
    src1.write_bytes(b"one")
    src2.write_bytes(b"two")

    data = {"created_images": [str(src1)], "last_saved_to_temp": True}
    new_data = {"created_images": [str(src1), str(src2)], "last_saved_to_temp": False}

    with pytest.raises(EndOfFlowException):
        copy_images(steps=2, data=data, file_identifier="flow_2", regex_pattern=r"^flow_1_node", new_data=new_data)

    copied = source_dir / "flow_2_1.png"
    assert copied.exists()
    assert data["created_images"] == [str(src1), str(copied)]
    assert data["last_saved_to_temp"] is False


class _DemoFlow(AbsFlow):
    def _run_impl(self, steps: int, multiprocess: bool) -> None:
        image = torch.ones((1, 2, 2, 3), dtype=torch.float32)
        self.save_image(self.saved_data, image, "demo")


def test_abs_flow_run_saves_image_and_updates_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AbsFlow should create output files and return the saved image paths."""
    main_dir = tmp_path / "main"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"
    main_dir.mkdir()
    output_dir.mkdir()
    temp_dir.mkdir()

    monkeypatch.setattr(utils, "get_main_images_path", lambda: str(main_dir))
    monkeypatch.setattr(utils.folder_paths, "get_output_directory", lambda: str(output_dir))
    monkeypatch.setattr(utils.folder_paths, "get_temp_directory", lambda: str(temp_dir))
    monkeypatch.setattr(utils.folder_paths, "recursive_search", lambda folder, excluded_dir_names=None: ([], {}))

    flow_json = main_dir / "folder" / "flow.json"
    flow_json.parent.mkdir()
    flow_json.write_text("{}", encoding="utf-8")

    flow = _DemoFlow("folder", "flow")
    flow._json_path = str(flow_json)

    with patch.object(utils.shutil, "copy2", wraps=utils.shutil.copy2) as copy_mock:
        result = flow.run(steps=1, multiprocess=False)

    assert len(result) == 1
    assert result[0].startswith(str(output_dir))
    assert Path(result[0]).exists()
    assert copy_mock.call_count >= 1
