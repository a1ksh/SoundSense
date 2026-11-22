import torch
from soundsense.models import TransferHead, parameter_count


def test_yamnet_head_logits_gradients_and_parameter_scope():
    model=TransferHead()
    logits=model(torch.randn(2,1024))
    assert logits.shape==(2,10) and torch.isfinite(logits).all()
    assert parameter_count(model)==10250
    torch.nn.functional.cross_entropy(logits,torch.tensor([1,2])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
