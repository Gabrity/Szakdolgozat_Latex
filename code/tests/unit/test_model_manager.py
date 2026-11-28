import os
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from src.model_manager import _IMAGENET_MEAN, _IMAGENET_STD, ModelManager

pytestmark = pytest.mark.unit


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture()
def fake_models(mocker):
    resnet = _TinyBackbone()
    mobilenet = _TinyBackbone()
    return SimpleNamespace(
        resnet_ctor=mocker.patch("src.model_manager.models.resnet50", return_value=resnet),
        mobilenet_ctor=mocker.patch("src.model_manager.models.mobilenet_v2", return_value=mobilenet),
        resnet=resnet,
        mobilenet=mobilenet,
    )


def test_init_creates_cache_dir(tmp_path):
    relative = tmp_path / "sub" / "models"

    manager = ModelManager(cache_dir=str(relative))

    assert relative.is_dir()
    assert manager.cache_dir == str(relative.resolve())

def test_init_sets_torch_home(tmp_path, monkeypatch):
    monkeypatch.delenv("TORCH_HOME", raising=False)

    manager = ModelManager(cache_dir=str(tmp_path))

    assert os.environ["TORCH_HOME"] == manager.cache_dir


def test_init_device_cpu_without_cuda(tmp_path, mocker):
    mocker.patch("torch.cuda.is_available", return_value=False)

    manager = ModelManager(cache_dir=str(tmp_path))

    assert manager.device == torch.device("cpu")


def test_init_device_cuda_when_available(tmp_path, mocker):
    mocker.patch("torch.cuda.is_available", return_value=True)

    manager = ModelManager(cache_dir=str(tmp_path))

    assert manager.device == torch.device("cuda")


def test_init_device_skips_check(tmp_path, mocker):
    cuda_check = mocker.patch("torch.cuda.is_available")

    manager = ModelManager(cache_dir=str(tmp_path), device="cpu")

    assert manager.device == torch.device("cpu")
    cuda_check.assert_not_called()


def test_get_unknown_model_raises(tmp_path):
    manager = ModelManager(cache_dir=str(tmp_path))

    with pytest.raises(ValueError, match="Unsupported model 'NEM_LETEZO_NEV')"):
        manager.get("NEM_LETEZO_NEV")


def test_get_caches_bundle(tmp_path, fake_models):
    manager = ModelManager(cache_dir=str(tmp_path))

    first = manager.get("ResNet50")
    second = manager.get("ResNet50")

    assert first is second
    fake_models.resnet_ctor.assert_called_once()


@pytest.mark.parametrize(
    "name, called, not_called",
    [
        ("ResNet50", "resnet_ctor", "mobilenet_ctor"),
        ("MobileNetV2", "mobilenet_ctor", "resnet_ctor"),
    ],
)
def test_get_dispatches_to_constructor(tmp_path, fake_models, name, called, not_called):
    manager = ModelManager(cache_dir=str(tmp_path))

    manager.get(name)

    getattr(fake_models, called).assert_called_once()
    getattr(fake_models, not_called).assert_not_called()


def test_build_sets_bundle_name(tmp_path, fake_models):
    manager = ModelManager(cache_dir=str(tmp_path))

    bundle = manager.get("MobileNetV2")

    assert bundle.name == "MobileNetV2"
    assert bundle.model is fake_models.mobilenet


def test_build_freezes_model(tmp_path):
    manager = ModelManager(cache_dir=str(tmp_path))

    bundle = manager.get("ResNet50")

    assert bundle.model.training is False
    assert all(not p.requires_grad for p in bundle.model.parameters())



def test_build_mean_std_match_imagenet(tmp_path):
    manager = ModelManager(cache_dir=str(tmp_path))

    bundle = manager.get("ResNet50")

    assert bundle.mean.shape == (1, 3, 1, 1)
    assert bundle.std.shape == (1, 3, 1, 1)
    assert torch.allclose(bundle.mean.cpu(), torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
    assert torch.allclose(bundle.std.cpu(), torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))


def test_build_preprocess_crops_to_224(tmp_path):
    manager = ModelManager(cache_dir=str(tmp_path))
    bundle = manager.get("ResNet50")
    image = Image.new("RGB", (300, 300), color=(10, 20, 30))

    tensor = bundle.preprocess(image)

    assert tensor.shape == (3, 224, 224)


def test_mean_std_returns_imagenet_constants(tmp_path):
    manager = ModelManager(cache_dir=str(tmp_path))

    mean, std = manager.mean_std

    assert mean == _IMAGENET_MEAN
    assert std == _IMAGENET_STD
