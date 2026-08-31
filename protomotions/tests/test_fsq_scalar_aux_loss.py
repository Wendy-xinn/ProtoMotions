import torch

from protomotions.agents.peft.sft_model import factorized_fsq_cross_entropy


def test_factorized_fsq_loss_rewards_each_mixed_radix_digit():
    # levels=3, two scalars: packed token = d0 + 3*d1.
    target = torch.tensor([[7]])  # d0=1, d1=2
    logits = torch.full((1, 1, 9), -12.0)
    logits[..., 7] = 12.0
    loss, accuracy = factorized_fsq_cross_entropy(
        logits, target, num_levels=3, scalars_per_token=2
    )
    assert loss < 1.0e-5
    torch.testing.assert_close(accuracy, torch.tensor(1.0))


def test_factorized_fsq_loss_has_finite_gradients_for_real_shape():
    torch.manual_seed(41)
    logits = torch.randn(2, 8, 9**5, requires_grad=True)
    targets = torch.randint(0, 9**5, (2, 8))
    loss, accuracy = factorized_fsq_cross_entropy(
        logits, targets, num_levels=9, scalars_per_token=5
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert 0.0 <= float(accuracy) <= 1.0
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
