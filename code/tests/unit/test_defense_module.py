"""Unit tests for src.defense_module.

DefenseModule only depends on Pillow and torchvision's ``ToTensor``, so the
tests run on tiny random tensors without any mocking (except the dispatch
spies in ``apply``).
"""
from __future__ import annotations

import pytest
import torch

from src.defense_module import DefenseModule

pytestmark = pytest.mark.unit

DEFENSES = ["gaussian_blur", "jpeg_compression"]


@pytest.fixture()
def defense() -> DefenseModule:
    return DefenseModule()


def _total_variation(x: torch.Tensor) -> torch.Tensor:
    """Sum of absolute neighbour differences: a simple high-frequency measure."""
    return (x[..., 1:, :] - x[..., :-1, :]).abs().sum() + (x[..., :, 1:] - x[..., :, :-1]).abs().sum()


@pytest.mark.parametrize("method", DEFENSES)
def test_defense_preserves_shape_dtype_and_device(defense, sample_image_tensor, method):
    out = getattr(defense, method)(sample_image_tensor)

    assert out.shape == sample_image_tensor.shape
    assert out.dtype == sample_image_tensor.dtype
    assert out.device == sample_image_tensor.device


@pytest.mark.parametrize("method", DEFENSES)
def test_defense_output_stays_in_valid_pixel_range(defense, sample_image_tensor, method):
    # Out-of-range input must be clamped by the tensor -> PIL conversion.
    out = getattr(defense, method)(sample_image_tensor * 3.0 - 1.0)

    assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.parametrize("method", DEFENSES)
def test_defense_result_is_detached_from_autograd(defense, sample_image_tensor, method):
    x = sample_image_tensor.clone().requires_grad_(True)

    out = getattr(defense, method)(x)

    assert out.grad_fn is None
    assert out.is_leaf is True


@pytest.mark.parametrize("method", DEFENSES)
def test_defense_does_not_modify_its_input(defense, sample_image_tensor, method):
    before = sample_image_tensor.clone()

    getattr(defense, method)(sample_image_tensor)

    assert torch.equal(sample_image_tensor, before)


def test_gaussian_blur_keeps_constant_image_unchanged(defense):
    x = torch.full((1, 3, 8, 8), 0.5)

    out = defense.gaussian_blur(x)

    assert torch.allclose(out, x, atol=1 / 255)


def test_gaussian_blur_reduces_high_frequency_content(defense, sample_image_tensor):
    out = defense.gaussian_blur(sample_image_tensor, radius=1.5)

    assert _total_variation(out) < _total_variation(sample_image_tensor)


def test_gaussian_blur_larger_radius_smooths_more(defense, sample_image_tensor):
    mild = defense.gaussian_blur(sample_image_tensor, radius=0.5)
    strong = defense.gaussian_blur(sample_image_tensor, radius=2.0)

    assert _total_variation(strong) < _total_variation(mild)


def test_jpeg_lower_quality_distorts_more(defense, sample_image_tensor):
    high = defense.jpeg_compression(sample_image_tensor, quality=95)
    low = defense.jpeg_compression(sample_image_tensor, quality=10)

    error_high = (high - sample_image_tensor).abs().mean()
    error_low = (low - sample_image_tensor).abs().mean()
    assert error_low > error_high


def test_jpeg_is_close_to_original_on_smooth_image_at_high_quality(defense):
    # A smooth gradient is what JPEG is designed for, unlike white noise.
    ramp = torch.linspace(0.2, 0.8, 16)
    x = ramp.view(1, 1, 1, 16).expand(1, 3, 16, 16).contiguous()

    out = defense.jpeg_compression(x, quality=95)

    assert (out - x).abs().max() < 0.05


@pytest.mark.parametrize(
    "name, expected",
    [
        ("gaussian", "gaussian_blur"),
        ("blur", "gaussian_blur"),
        ("gaussian_blur", "gaussian_blur"),
        ("BLUR", "gaussian_blur"),
        ("jpeg", "jpeg_compression"),
        ("jpeg_compression", "jpeg_compression"),
        ("JPEG", "jpeg_compression"),
    ],
)
def test_apply_dispatches_only_to_matching_defense(mocker, defense, sample_image_tensor, name, expected):
    # wraps= keeps the real implementation running while recording the calls.
    spies = {
        method: mocker.patch.object(defense, method, wraps=getattr(defense, method))
        for method in DEFENSES
    }

    defense.apply(sample_image_tensor, name)

    spies[expected].assert_called_once()
    for method, spy in spies.items():
        if method != expected:
            spy.assert_not_called()


@pytest.mark.parametrize(
    "name, kwargs, forwarded",
    [
        ("blur", {"radius": 0.7}, "gaussian_blur"),
        ("jpeg", {"quality": 30}, "jpeg_compression"),
    ],
)
def test_apply_forwards_keyword_arguments(mocker, defense, sample_image_tensor, name, kwargs, forwarded):
    spy = mocker.patch.object(defense, forwarded, wraps=getattr(defense, forwarded))

    defense.apply(sample_image_tensor, name, **kwargs)

    spy.assert_called_once_with(sample_image_tensor, **kwargs)


def test_apply_unknown_method_raises(defense, sample_image_tensor):
    with pytest.raises(ValueError, match="Unknown defense"):
        defense.apply(sample_image_tensor, "NEM VALID")
