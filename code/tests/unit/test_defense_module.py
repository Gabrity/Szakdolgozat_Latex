from __future__ import annotations

import pytest
import torch

from src.defense_module import DefenseModule

pytestmark = pytest.mark.unit

DEFENSES = ["gaussian", "jpeg"]


@pytest.fixture()
def defense() -> DefenseModule:
    return DefenseModule()


@pytest.mark.parametrize("method", DEFENSES)
def test_defense_preserves_shape_dtype_and_device(defense, sample_image_tensor, method):
    out = getattr(defense, method)(sample_image_tensor)

    assert out.shape == sample_image_tensor.shape
    assert out.dtype == sample_image_tensor.dtype
    assert out.device == sample_image_tensor.device


@pytest.mark.parametrize("method", DEFENSES)
def test_defense_output_stays_in_valid_pixel_range(defense, sample_image_tensor, method):
    out = getattr(defense, method)(sample_image_tensor * 3.0 - 1.0)

    assert out.min() >= 0.0 and out.max() <= 1.0


def test_gaussian_blur_keeps_constant_image_unchanged(defense):
    x = torch.full((1, 3, 8, 8), 0.5)

    out = defense.gaussian(x)

    assert torch.allclose(out, x, atol=1 / 255)


def test_apply_unknown_method_raises(defense, sample_image_tensor):
    with pytest.raises(ValueError, match="Unknown defense"):
        defense.apply(sample_image_tensor, "NEM VALID")
