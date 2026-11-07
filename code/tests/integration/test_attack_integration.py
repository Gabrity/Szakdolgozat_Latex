"""Integration tests for image loading, model management and attacks."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torchvision import models

from src.attack_engine import AttackContext, attack, predict
from src.model_manager import ModelManager
from src.utils import load_image_as_tensor

pytestmark = pytest.mark.integration

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
EPSILON = 8 / 255


@pytest.mark.parametrize("method", ["FGSM", "PGD"])
def test_image_loading_and_attack(method, sample_png, tiny_model_224, tmp_path, monkeypatch):
    monkeypatch.setattr(models, "mobilenet_v2", lambda *, weights: tiny_model_224)
    manager = ModelManager(cache_dir=str(tmp_path), device="cpu")
    bundle = manager.get("MobileNetV2")
    context = AttackContext(bundle.model, bundle.mean, bundle.std)
    image, _ = load_image_as_tensor(str(sample_png), manager.device)

    torch.manual_seed(0)
    label = int(predict(context, image, topk=1)[1][0, 0])
    result = attack(context, image, label, method, epsilon=EPSILON, iterations=5)
    probabilities, indices = predict(context, result.adversarial, topk=5)

    assert result.adversarial.shape == image.shape
    assert (result.adversarial - image).abs().max() <= EPSILON + 1e-6
    assert torch.all((result.adversarial >= 0) & (result.adversarial <= 1))
    assert not torch.equal(result.adversarial, image)
    assert probabilities.shape == indices.shape == (1, 5)
    assert torch.allclose(probabilities.sum(), torch.tensor(1.0))


def test_cached_mobilenet_pgd(sample_png, monkeypatch):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
    checkpoint = MODELS_DIR / "hub" / "checkpoints" / weights.url.rsplit("/", 1)[-1]
    if not checkpoint.is_file():
        pytest.skip("MobileNetV2 checkpoint is not cached locally")

    monkeypatch.setenv("TORCH_HOME", str(MODELS_DIR))
    manager = ModelManager(cache_dir=str(MODELS_DIR), device="cpu")
    bundle = manager.get("MobileNetV2")
    context = AttackContext(bundle.model, bundle.mean, bundle.std)
    image, _ = load_image_as_tensor(str(sample_png), manager.device)

    torch.manual_seed(0)
    label = int(predict(context, image, topk=1)[1][0, 0])
    result = attack(context, image, label, "PGD", epsilon=EPSILON, iterations=5)
    probabilities, indices = predict(context, result.adversarial, topk=3)

    assert result.adversarial.shape == image.shape
    assert (result.adversarial - image).abs().max() <= EPSILON + 1e-6
    assert not torch.equal(result.adversarial, image)
    assert probabilities.shape == indices.shape == (1, 3)
    assert torch.all((indices >= 0) & (indices < 1000))
