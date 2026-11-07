"""Unit tests for src.utils image/tensor conversion helpers."""
from __future__ import annotations

import pytest
import torch

from src.utils import (
    build_adversarial_filename,
    load_image_as_tensor,
    perturbation_to_pil,
    tensor_to_pil,
)

pytestmark = pytest.mark.unit


def test_load_image_as_tensor_resizes_and_crops(sample_png, device):
    tensor, pil = load_image_as_tensor(str(sample_png), device)

    assert tensor.shape == (1, 3, 224, 224)
    assert pil.size == (224, 224)
    assert torch.all(tensor >= 0) and torch.all(tensor <= 1)


def test_load_image_as_tensor_keeps_224_image_unchanged(tmp_path, device):
    torch.manual_seed(0)
    original = torch.rand(1, 3, 224, 224)
    path = tmp_path / "exact.png"
    tensor_to_pil(original).save(path)

    tensor, pil = load_image_as_tensor(str(path), device)

    expected = (original * 255).round() / 255
    assert pil.size == (224, 224)
    assert torch.allclose(tensor.cpu(), expected, atol=1e-6)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (dict(model="ResNet50", method="FGSM", epsilon=0.03, iterations=10),
         "ResNet50_FGSM_eps0.030_untargeted"),
        (dict(model="MobileNetV2", method="PGD", epsilon=0.05, iterations=20),
         "MobileNetV2_PGD_eps0.050_iter20_untargeted"),
        (dict(model="ResNet50", method="PGD", epsilon=0.1, iterations=5,
              target_idx=207, target_name="golden retriever"),
         "ResNet50_PGD_eps0.100_iter5_targeted_207_golden-retriever"),
        (dict(model="ResNet50", method="FGSM", epsilon=0.03, iterations=10,
              target_idx=1, target_name="Mr. X/Y's \"dog\""),
         "ResNet50_FGSM_eps0.030_targeted_1_Mr-X-Y-s-dog"),
    ],
)
def test_build_adversarial_filename(kwargs, expected):
    assert build_adversarial_filename(**kwargs) == expected


def test_tensor_to_pil_roundtrip_size(sample_image_tensor):
    pil = tensor_to_pil(sample_image_tensor)

    assert pil.mode == "RGB"
    assert pil.size == (sample_image_tensor.shape[3], sample_image_tensor.shape[2])


def test_perturbation_to_pil_zero_delta_is_mid_gray(device):
    zero_delta = torch.zeros(1, 3, 4, 4, device=device)

    pil = perturbation_to_pil(zero_delta, magnify=10.0)
    px = pil.getpixel((0, 0))

    assert px in {(127, 127, 127), (128, 128, 128)}
