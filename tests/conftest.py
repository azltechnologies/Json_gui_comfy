"""Shared pytest fixtures and import stubs for json_gui tests."""

from __future__ import annotations

import collections
import importlib.util
import logging
import multiprocessing as std_multiprocessing
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import pytest
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeDevice:
    """Minimal stand-in for torch.device."""

    def __init__(self, device_type: str) -> None:
        self.type = device_type

    def __repr__(self) -> str:
        return f"device('{self.type}')"


class FakeTensor:
    """Very small tensor implementation backed by numpy arrays."""

    def __init__(self, data: Any, device: FakeDevice | None = None, requires_grad: bool = False) -> None:
        self._data = np.array(data)
        self.device = device or FakeDevice("cpu")
        self.requires_grad = requires_grad

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    def detach(self) -> "FakeTensor":
        return FakeTensor(self._data.copy(), self.device, False)

    def cpu(self) -> "FakeTensor":
        return FakeTensor(self._data.copy(), FakeDevice("cpu"), self.requires_grad)

    def clone(self) -> "FakeTensor":
        return FakeTensor(self._data.copy(), self.device, self.requires_grad)

    def contiguous(self) -> "FakeTensor":
        return self

    def to(self, torch_device: Any) -> "FakeTensor":
        device = (
            torch_device
            if isinstance(torch_device, FakeDevice)
            else FakeDevice(getattr(torch_device, "type", "cpu"))
        )
        return FakeTensor(self._data.copy(), device, self.requires_grad)

    def numpy(self) -> np.ndarray:
        return self._data.copy()

    def permute(self, *dims: int) -> "FakeTensor":
        return FakeTensor(np.transpose(self._data, axes=dims), self.device, self.requires_grad)

    def movedim(self, src: int, dst: int) -> "FakeTensor":
        return FakeTensor(np.moveaxis(self._data, src, dst), self.device, self.requires_grad)

    def __iter__(self):
        for item in self._data:
            yield FakeTensor(item, self.device, self.requires_grad)

    def __getitem__(self, item: Any) -> "FakeTensor":
        return FakeTensor(self._data[item], self.device, self.requires_grad)

    def __len__(self) -> int:
        return len(self._data)

    def __array__(self, dtype=None):
        return np.asarray(self._data, dtype=dtype)

    def equal(self, other: Any) -> bool:
        return isinstance(other, FakeTensor) and np.array_equal(self._data, other._data)

    def __mul__(self, other: Any) -> "FakeTensor":
        return FakeTensor(self._data * other, self.device, self.requires_grad)

    def __rmul__(self, other: Any) -> "FakeTensor":
        return self.__mul__(other)

    def __add__(self, other: Any) -> "FakeTensor":
        return FakeTensor(self._data + other, self.device, self.requires_grad)

    def __repr__(self) -> str:
        return f"FakeTensor({self._data!r}, device={self.device!r})"


class _InferenceMode:
    def __enter__(self) -> "_InferenceMode":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _install_fake_torch() -> None:
    torch_module = types.ModuleType("torch")
    torch_module.Tensor = FakeTensor
    torch_module.device = FakeDevice
    torch_module.float32 = np.float32
    torch_module.uint8 = np.uint8
    torch_module.tensor = lambda data, dtype=None, device=None: FakeTensor(data, device=device)
    torch_module.ones = lambda shape, dtype=None, device=None: FakeTensor(np.ones(shape), device=device)
    torch_module.zeros = lambda shape, dtype=None, device=None: FakeTensor(np.zeros(shape), device=device)
    torch_module.zeros_like = lambda tensor, device=None: FakeTensor(
        np.zeros_like(tensor._data), device=device or tensor.device
    )
    torch_module.from_numpy = lambda array: FakeTensor(np.array(array))
    torch_module.stack = lambda tensors, dim=0: FakeTensor(
        np.stack([t._data for t in tensors], axis=dim), tensors[0].device
    )
    torch_module.cat = lambda tensors, dim=0: FakeTensor(
        np.concatenate([t._data for t in tensors], axis=dim), tensors[0].device
    )
    torch_module.equal = lambda left, right: isinstance(left, FakeTensor) and left.equal(right)
    torch_module.inference_mode = _InferenceMode
    torch_module.cuda = types.SimpleNamespace(empty_cache=lambda: None)
    torch_module.nn = types.SimpleNamespace(
        functional=types.SimpleNamespace(interpolate=lambda input, size, mode=None, align_corners=None: input)
    )
    torch_module.multiprocessing = std_multiprocessing
    sys.modules["torch"] = torch_module


_install_fake_torch()
import torch  # noqa: E402  # pylint: disable=wrong-import-position


def _install_module(name: str, module: types.ModuleType) -> types.ModuleType:
    sys.modules[name] = module
    parent_name, _, child_name = name.rpartition(".")
    if parent_name:
        parent = sys.modules[parent_name]
        setattr(parent, child_name, module)
    return module


def _install_stub(name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return _install_module(name, module)


def _load_module(module_name: str, file_path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    _install_module(module_name, module)
    spec.loader.exec_module(module)
    return module


def _install_test_package() -> None:
    if "json_gui" not in sys.modules:
        package = types.ModuleType("json_gui")
        package.__path__ = [str(REPO_ROOT)]
        _install_module("json_gui", package)

    for package_name in ("json_gui.scripts", "json_gui.json_manager"):
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [str(REPO_ROOT / package_name.rsplit(".", 1)[-1])]
            _install_module(package_name, package)

    if "app" not in sys.modules:
        app_module = types.ModuleType("app")
        app_module.__path__ = []
        _install_module("app", app_module)

    if "app.logger" not in sys.modules:
        app_logger = types.ModuleType("app.logger")
        app_logger.logs = collections.deque(maxlen=300)
        app_logger.stdout_interceptor = None
        app_logger.stderr_interceptor = None

        class _LogInterceptor:
            def __init__(self, stream: Any) -> None:
                self.stream = stream

            def write(self, message: str) -> None:
                self.stream.write(message)

            def flush(self) -> None:
                self.stream.flush()

        app_logger.LogInterceptor = _LogInterceptor
        app_logger.deque = collections.deque
        app_logger.get_logs = lambda: [object()]
        _install_module("app.logger", app_logger)

    if "comfy" not in sys.modules:
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        _install_module("comfy", comfy)

    _install_stub("comfy.options", enable_args_parsing=lambda: None)
    _install_stub(
        "comfy.cli_args",
        args=types.SimpleNamespace(verbose=logging.INFO, log_stdout=False),
    )
    _install_stub(
        "comfy.model_management",
        unload_all_models=lambda: None,
        soft_empty_cache=lambda: None,
        intermediate_device=lambda: torch.device("cpu"),
        get_torch_device=lambda: torch.device("cpu"),
    )
    _install_stub("comfy.model_patcher", ModelPatcher=type("ModelPatcher", (), {}))
    _install_stub(
        "comfy.sd",
        VAE=type("VAE", (), {}),
        CLIP=type("CLIP", (), {}),
        load_checkpoint_guess_config=lambda *args, **kwargs: (object(), None, object(), None),
        load_clip=lambda *args, **kwargs: object(),
    )
    _install_stub(
        "comfy.sample",
        fix_empty_latent_channels=lambda model, latent: latent,
        prepare_noise=lambda latent, seed, noise: torch.zeros_like(latent),
        sample=lambda **kwargs: kwargs["latent_image"],
    )
    _install_stub("comfy_extras", __path__=[])
    _install_stub(
        "comfy_extras.nodes_sd3",
        SkipLayerGuidanceSD3=types.SimpleNamespace(execute=lambda *args, **kwargs: (args[0],)),
    )
    _install_stub(
        "comfy_extras.nodes_mask",
        MaskToImage=type("MaskToImage", (), {"execute": lambda self, mask: types.SimpleNamespace(result=(mask,))}),
    )
    _install_stub(
        "folder_paths",
        get_user_directory=lambda: tempfile.gettempdir(),
        get_input_directory=lambda: tempfile.gettempdir(),
        get_output_directory=lambda: tempfile.gettempdir(),
        get_temp_directory=lambda: tempfile.gettempdir(),
        get_scripts_folder_path=lambda: str(REPO_ROOT / "scripts"),
        recursive_search=lambda folder, excluded_dir_names=None: ([], {}),
        filter_files_content_types=lambda files, content_types: files,
        get_filename_list_=lambda folder: ([], {str(Path(tempfile.gettempdir()) / folder): []}),
        get_full_path_or_raise=lambda *args, **kwargs: str(Path(tempfile.gettempdir()) / "stub"),
        get_folder_paths=lambda *args, **kwargs: [tempfile.gettempdir()],
    )
    _install_stub("node_helpers", pillow=lambda func, *args, **kwargs: func(*args, **kwargs))


_install_test_package()

typedicts = _load_module("json_gui.typedicts", REPO_ROOT / "typedicts.py")
utils = _load_module("json_gui.utils", REPO_ROOT / "utils.py")
c_logger = _load_module("json_gui.c_logger", REPO_ROOT / "c_logger.py")
p_logger = _load_module("json_gui.p_logger", REPO_ROOT / "p_logger.py")
mimic = _load_module("json_gui.scripts.mimic", REPO_ROOT / "scripts" / "mimic.py")
mimic_classes = _load_module("json_gui.scripts.mimic_classes", REPO_ROOT / "scripts" / "mimic_classes.py")
mimic_ksamplers = _load_module(
    "json_gui.scripts.mimic_ksamplers", REPO_ROOT / "scripts" / "mimic_ksamplers.py"
)
node_executor = _load_module("json_gui.scripts.node_executor", REPO_ROOT / "scripts" / "node_executor.py")


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory path."""
    return tmp_path


@pytest.fixture
def saved_images_dict() -> dict[str, Any]:
    """Provide a standard SavedImagesDict fixture."""
    return {"created_images": [], "last_saved_to_temp": None}


@pytest.fixture
def torch_tensor() -> torch.Tensor:
    """Provide a simple tensor fixture."""
    return torch.tensor([[1.0, 2.0], [3.0, 4.0]])


@pytest.fixture(autouse=True)
def reset_mimic_globals() -> None:
    """Reset MimicNode globals between tests."""
    from json_gui.scripts.mimic import DataWrapper, MimicNode

    MimicNode._node_executor_factory = None
    MimicNode._current_model = None
    MimicNode._do_multiprocess = False
    DataWrapper._skip_pickle_check = False
    yield
    MimicNode._node_executor_factory = None
    MimicNode._current_model = None
    MimicNode._do_multiprocess = False
    DataWrapper._skip_pickle_check = False
