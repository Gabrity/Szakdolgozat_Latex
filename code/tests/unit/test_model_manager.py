"""Unit tests for src.model_manager.

ModelManager builds real torchvision ResNet50/MobileNetV2 backbones, which
would mean network access and multi-second construction in a "unit" test.
The ``mock_torchvision_models`` fixture replaces the two constructors with a
tiny stand-in nn.Module (same idea as the ``tiny_model`` fixture used for
attack_engine), so caching/dispatch/build logic can be tested in isolation
from the real architectures.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from src.model_manager import _IMAGENET_MEAN, _IMAGENET_STD, ModelManager

pytestmark = pytest.mark.unit


class _TinyBackbone(nn.Module):
    """Minimal stand-in for a torchvision classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@pytest.fixture()
def mock_torchvision_models(mocker):
    """Replace models.resnet50 / models.mobilenet_v2 with tiny fakes so
    building a bundle never downloads weights or builds a real backbone."""
    resnet_stub = _TinyBackbone()
    mobilenet_stub = _TinyBackbone()
    resnet_ctor = mocker.patch("src.model_manager.models.resnet50", return_value=resnet_stub)
    mobilenet_ctor = mocker.patch("src.model_manager.models.mobilenet_v2", return_value=mobilenet_stub)
    return SimpleNamespace(
        resnet_ctor=resnet_ctor,
        mobilenet_ctor=mobilenet_ctor,
        resnet_stub=resnet_stub,
        mobilenet_stub=mobilenet_stub,
    )


# ---------------------------------------------------------------------- #
# cache_dir / TORCH_HOME
# ---------------------------------------------------------------------- #
def test_cache_dir_is_created_and_made_absolute(tmp_path):
    relative = tmp_path / "sub" / "models"

    manager = ModelManager(cache_dir=str(relative))

    assert relative.is_dir()
    assert manager.cache_dir == str(relative.resolve())


def test_cache_dir_sets_torch_home_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("TORCH_HOME", raising=False)

    manager = ModelManager(cache_dir=str(tmp_path))

    assert __import__("os").environ["TORCH_HOME"] == manager.cache_dir


def test_cache_dir_does_not_override_existing_torch_home(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_HOME", "sentinel-value")

    ModelManager(cache_dir=str(tmp_path))

    assert __import__("os").environ["TORCH_HOME"] == "sentinel-value"


# ---------------------------------------------------------------------- #
# device selection
# ---------------------------------------------------------------------- #
def test_device_defaults_to_cpu_when_cuda_unavailable(tmp_path, mocker):
    mocker.patch("torch.cuda.is_available", return_value=False)

    manager = ModelManager(cache_dir=str(tmp_path))

    assert manager.device == torch.device("cpu")


def test_device_uses_cuda_when_available(tmp_path, mocker):
    mocker.patch("torch.cuda.is_available", return_value=True)

    manager = ModelManager(cache_dir=str(tmp_path))

    assert manager.device == torch.device("cuda")


def test_explicit_device_skips_cuda_autodetect(tmp_path, mocker):
    cuda_check = mocker.patch("torch.cuda.is_available")

    manager = ModelManager(cache_dir=str(tmp_path), device="cpu")

    assert manager.device == torch.device("cpu")
    cuda_check.assert_not_called()


# ---------------------------------------------------------------------- #
# unsupported model
# ---------------------------------------------------------------------- #
def test_get_unknown_model_raises_with_supported_list(tmp_path):
    manager = ModelManager(cache_dir=str(tmp_path))

    with pytest.raises(ValueError, match="ResNet50.*MobileNetV2|MobileNetV2.*ResNet50"):
        manager.get("NEM_LETEZO_MODELL")


# ---------------------------------------------------------------------- #
# caching behaviour
# ---------------------------------------------------------------------- #
def test_get_caches_bundle_and_builds_only_once(tmp_path, mock_torchvision_models):
    manager = ModelManager(cache_dir=str(tmp_path))

    first = manager.get("ResNet50")
    second = manager.get("ResNet50")

    assert first is second
    mock_torchvision_models.resnet_ctor.assert_called_once()


@pytest.mark.parametrize(
    "name, should_call, should_not_call",
    [
        ("ResNet50", "resnet_ctor", "mobilenet_ctor"),
        ("MobileNetV2", "mobilenet_ctor", "resnet_ctor"),
    ],
)
def test_get_dispatches_to_matching_constructor_only(
    tmp_path, mock_torchvision_models, name, should_call, should_not_call
):
    manager = ModelManager(cache_dir=str(tmp_path))

    manager.get(name)

    getattr(mock_torchvision_models, should_call).assert_called_once()
    getattr(mock_torchvision_models, should_not_call).assert_not_called()


# ---------------------------------------------------------------------- #
# built bundle contents
# ---------------------------------------------------------------------- #
def test_bundle_name_matches_requested_architecture(tmp_path, mock_torchvision_models):
    manager = ModelManager(cache_dir=str(tmp_path))

    bundle = manager.get("MobileNetV2")

    assert bundle.name == "MobileNetV2"
    assert bundle.model is mock_torchvision_models.mobilenet_stub


def test_build_puts_model_in_eval_mode_and_freezes_parameters(tmp_path, mock_torchvision_models):
    manager = ModelManager(cache_dir=str(tmp_path))

    bundle = manager.get("ResNet50")

    assert bundle.model.training is False
    assert all(not p.requires_grad for p in bundle.model.parameters())


def test_bundle_mean_std_tensors_match_imagenet_constants(tmp_path, mock_torchvision_models):
    manager = ModelManager(cache_dir=str(tmp_path))

    bundle = manager.get("ResNet50")

    expected_mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    expected_std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    assert bundle.mean.shape == (1, 3, 1, 1)
    assert bundle.std.shape == (1, 3, 1, 1)
    assert torch.allclose(bundle.mean.cpu(), expected_mean)
    assert torch.allclose(bundle.std.cpu(), expected_std)


def test_bundle_preprocess_resizes_and_crops_to_224(tmp_path, mock_torchvision_models):
    manager = ModelManager(cache_dir=str(tmp_path))
    bundle = manager.get("ResNet50")
    image = Image.new("RGB", (300, 300), color=(10, 20, 30))

    tensor = bundle.preprocess(image)

    assert tensor.shape == (3, 224, 224)


# ---------------------------------------------------------------------- #
# mean_std property
# ---------------------------------------------------------------------- #
def test_mean_std_property_returns_imagenet_constants(tmp_path):
    manager = ModelManager(cache_dir=str(tmp_path))

    mean, std = manager.mean_std

    assert mean == _IMAGENET_MEAN
    assert std == _IMAGENET_STD
