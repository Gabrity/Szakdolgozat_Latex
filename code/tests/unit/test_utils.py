"""Unit tests for src.utils image/tensor conversion helpers."""
from __future__ import annotations

import pytest
import torch

from src.utils import load_image_as_tensor, perturbation_to_pil, tensor_to_pil

pytestmark = pytest.mark.unit


def test_load_image_as_tensor_resizes_and_crops(sample_png, device):
    tensor, pil = load_image_as_tensor(str(sample_png), device)

    assert tensor.shape == (1, 3, 224, 224)
    assert pil.size == (224, 224)
    assert torch.all(tensor >= 0) and torch.all(tensor <= 1)


def test_tensor_to_pil_roundtrip_size(sample_image_tensor):
    pil = tensor_to_pil(sample_image_tensor)

    assert pil.mode == "RGB"
    assert pil.size == (sample_image_tensor.shape[3], sample_image_tensor.shape[2])


def test_perturbation_to_pil_zero_delta_is_mid_gray(device):
    zero_delta = torch.zeros(1, 3, 4, 4, device=device)

    pil = perturbation_to_pil(zero_delta, magnify=10.0)
    px = pil.getpixel((0, 0))

    assert px in {(127, 127, 127), (128, 128, 128)}
