"""Unit tests for src.attack_engine.

attack_engine has no hidden dependencies (model/mean/std are passed in via
AttackContext), so these tests need zero mocking - a good example of a
module that is already isolated by construction ("functional core").
"""
from __future__ import annotations

import pytest
import torch

from src.attack_engine import attack, predict, _fgsm, _pgd

pytestmark = pytest.mark.unit


def test_predict_returns_topk_shapes(attack_ctx, sample_image_tensor):
    probs, idxs = predict(attack_ctx, sample_image_tensor, topk=3)

    assert probs.shape == (1, 3)
    assert idxs.shape == (1, 3)


def test_predict_returns_valid_distribution(attack_ctx, sample_image_tensor):
    probs, _ = predict(attack_ctx, sample_image_tensor, topk=5)

    assert torch.all(probs >= 0) and torch.all(probs <= 1)
    assert torch.allclose(probs.sum(), torch.tensor(1.0))


def test_predict_topk_returns_highest_first(attack_ctx, sample_image_tensor):
    probs, _ = predict(attack_ctx, sample_image_tensor, topk=5)

    assert torch.all(probs[0, :-1] >= probs[0, 1:])


def test_attack_unknown_method_raises(attack_ctx, sample_image_tensor):
    with pytest.raises(ValueError):
        attack(attack_ctx, sample_image_tensor, label=0, method="NEM VALID", epsilon=0.1, iterations=1)


def test_attack_fgsm_ignores_iterations_argument(attack_ctx, sample_image_tensor):
    result = attack(attack_ctx, sample_image_tensor, label=0, method="FGSM", epsilon=0.1, iterations=99)

    assert result.iterations_run == 1


def test_attack_pgd_reports_requested_iteration_count(attack_ctx, sample_image_tensor):
    result = attack(attack_ctx, sample_image_tensor, label=0, method="PGD", epsilon=0.08, iterations=4)

    assert result.iterations_run == 4



def test_attack_fgsm_dispatches_only_to_fgsm(mocker, attack_ctx, sample_image_tensor):
    # wraps= keeps the real implementation running so AttackResult assembly still works.
    fgsm_spy = mocker.patch("src.attack_engine._fgsm", wraps=_fgsm)
    pgd_spy = mocker.patch("src.attack_engine._pgd", wraps=_pgd)

    attack(attack_ctx, sample_image_tensor, label=0, method="FGSM", epsilon=0.1, iterations=5)

    fgsm_spy.assert_called_once()
    pgd_spy.assert_not_called()


def test_attack_pgd_dispatches_only_to_pgd(mocker, attack_ctx, sample_image_tensor):
    fgsm_spy = mocker.patch("src.attack_engine._fgsm", wraps=_fgsm)
    pgd_spy = mocker.patch("src.attack_engine._pgd", wraps=_pgd)

    attack(attack_ctx, sample_image_tensor, label=0, method="PGD", epsilon=0.1, iterations=5)

    pgd_spy.assert_called_once()
    fgsm_spy.assert_not_called()


def test_fgsm_targeted_steps_opposite_direction_of_untargeted(attack_ctx, sample_image_tensor):
    x = sample_image_tensor.clamp(0.3, 0.7)
    y = torch.tensor([0])
    eps = 0.05

    untargeted = _fgsm(attack_ctx, x, y, epsilon=eps, targeted=False)
    targeted = _fgsm(attack_ctx, x, y, epsilon=eps, targeted=True)

    assert torch.allclose(untargeted - x, -(targeted - x))


def test_fgsm_clamps_to_valid_pixel_range(attack_ctx):
    x = torch.zeros(1, 3, 8, 8)
    y = torch.tensor([0])

    x_adv = _fgsm(attack_ctx, x, y, epsilon=0.5, targeted=False)

    assert torch.all(x_adv >= 0.0) and torch.all(x_adv <= 1.0)


def test_fgsm_result_is_detached_from_autograd(attack_ctx, sample_image_tensor):
    x = sample_image_tensor
    y = torch.tensor([0])

    x_adv = _fgsm(attack_ctx, x, y, epsilon=0.05, targeted=False)

    assert x_adv.requires_grad is False


def test_fgsm_result_has_no_graph_history(attack_ctx, sample_image_tensor):
    x = sample_image_tensor
    y = torch.tensor([0])

    x_adv = _fgsm(attack_ctx, x, y, epsilon=0.05, targeted=False)

    assert x_adv.grad_fn is None
    assert x_adv.is_leaf is True


def test_fgsm_zero_epsilon_returns_original(attack_ctx, sample_image_tensor):
    x = sample_image_tensor
    y = torch.tensor([0])

    x_adv = _fgsm(attack_ctx, x, y, epsilon=0.0, targeted=False)

    assert torch.allclose(x_adv, x)


def test_fgsm_untargeted_reduces_confidence_in_true_label(attack_ctx, sample_image_tensor):
    x = sample_image_tensor
    true_label = int(predict(attack_ctx, x, topk=1)[1][0, 0])
    y = torch.tensor([true_label])

    probs_before, idxs_before = predict(attack_ctx, x, topk=5)
    prob_before = probs_before[0, (idxs_before[0] == true_label).nonzero()[0, 0]]

    x_adv = _fgsm(attack_ctx, x, y, epsilon=0.1, targeted=False)

    probs_after, idxs_after = predict(attack_ctx, x_adv, topk=5)
    prob_after = probs_after[0, (idxs_after[0] == true_label).nonzero()[0, 0]]

    assert prob_after < prob_before


def test_fgsm_targeted_increases_confidence_in_target_label(attack_ctx, sample_image_tensor):
    x = sample_image_tensor
    true_label = int(predict(attack_ctx, x, topk=1)[1][0, 0])
    target_label = (true_label + 1) % 5
    y = torch.tensor([target_label])

    probs_before, idxs_before = predict(attack_ctx, x, topk=5)
    prob_before = probs_before[0, (idxs_before[0] == target_label).nonzero()[0, 0]]

    x_adv = _fgsm(attack_ctx, x, y, epsilon=0.1, targeted=True)

    probs_after, idxs_after = predict(attack_ctx, x_adv, topk=5)
    prob_after = probs_after[0, (idxs_after[0] == target_label).nonzero()[0, 0]]

    assert prob_after > prob_before


def test_pgd_result_stays_within_epsilon_ball(attack_ctx, sample_image_tensor):
    x = sample_image_tensor.clamp(0.3, 0.7)
    y = torch.tensor([0])
    eps = 0.05

    x_adv = _pgd(attack_ctx, x, y, epsilon=eps, iterations=5, step_size=0.02, targeted=False)

    assert torch.all((x_adv - x).abs() <= eps + 1e-6)


def test_pgd_hits_ball_when_step_too_large(attack_ctx, sample_image_tensor):
    x = sample_image_tensor.clamp(0.3, 0.7)
    y = torch.tensor([0])
    eps = 0.02

    # step_size far larger than epsilon forces every iteration to overshoot,
    # so only the projection clamp keeps the final delta within the ball.
    x_adv = _pgd(attack_ctx, x, y, epsilon=eps, iterations=3, step_size=10.0, targeted=False)

    assert torch.allclose((x_adv - x).abs(), torch.full_like(x, eps), atol=1e-6)


def test_pgd_clamps_to_valid_pixel_range(attack_ctx):
    x = torch.zeros(1, 3, 8, 8)
    y = torch.tensor([0])

    x_adv = _pgd(attack_ctx, x, y, epsilon=0.5, iterations=5, step_size=0.3, targeted=False)

    assert torch.all(x_adv >= 0.0) and torch.all(x_adv <= 1.0)


def test_pgd_result_has_no_graph_history(attack_ctx, sample_image_tensor):
    x = sample_image_tensor
    y = torch.tensor([0])

    x_adv = _pgd(attack_ctx, x, y, epsilon=0.05, iterations=5, step_size=0.02, targeted=False)

    assert x_adv.grad_fn is None
    assert x_adv.is_leaf is True


def test_pgd_zero_epsilon_returns_original(attack_ctx, sample_image_tensor):
    x = sample_image_tensor
    y = torch.tensor([0])

    x_adv = _pgd(attack_ctx, x, y, epsilon=0.0, iterations=5, step_size=0.01, targeted=False)

    assert torch.allclose(x_adv, x)


def test_pgd_untargeted_reduces_confidence_in_true_label(attack_ctx, sample_image_tensor):
    x = sample_image_tensor
    true_label = int(predict(attack_ctx, x, topk=1)[1][0, 0])
    y = torch.tensor([true_label])

    probs_before, idxs_before = predict(attack_ctx, x, topk=5)
    prob_before = probs_before[0, (idxs_before[0] == true_label).nonzero()[0, 0]]

    x_adv = _pgd(attack_ctx, x, y, epsilon=0.1, iterations=5, step_size=0.05, targeted=False)

    probs_after, idxs_after = predict(attack_ctx, x_adv, topk=5)
    prob_after = probs_after[0, (idxs_after[0] == true_label).nonzero()[0, 0]]

    assert prob_after < prob_before


def test_pgd_targeted_increases_confidence_in_target_label(attack_ctx, sample_image_tensor):
    x = sample_image_tensor
    true_label = int(predict(attack_ctx, x, topk=1)[1][0, 0])
    target_label = (true_label + 1) % 5
    y = torch.tensor([target_label])

    probs_before, idxs_before = predict(attack_ctx, x, topk=5)
    prob_before = probs_before[0, (idxs_before[0] == target_label).nonzero()[0, 0]]

    x_adv = _pgd(attack_ctx, x, y, epsilon=0.1, iterations=5, step_size=0.05, targeted=True)

    probs_after, idxs_after = predict(attack_ctx, x_adv, topk=5)
    prob_after = probs_after[0, (idxs_after[0] == target_label).nonzero()[0, 0]]

    assert prob_after > prob_before