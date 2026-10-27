import pytest
import torch

from soundsense.models import CNN, MLP


@pytest.mark.parametrize('cls', [MLP, CNN])
def test_logits_and_gradient_flow(cls):
    model = cls()
    x = torch.randn(2, 1, 64, 128)
    logits = model(x)
    assert logits.shape == (2, 10) and torch.isfinite(logits).all()
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
