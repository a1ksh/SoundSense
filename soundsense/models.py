"""The trainable ESC-10 head on frozen 1024-dimensional YAMNet embeddings."""
from torch import nn


class TransferHead(nn.Module):
    def __init__(self, dimension=1024):
        super().__init__()
        if dimension != 1024:
            raise ValueError('YAMNet requires 1024 embedding features')
        self.net = nn.Sequential(nn.Dropout(.2), nn.Linear(dimension, 10))

    def forward(self, x):
        return self.net(x)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
