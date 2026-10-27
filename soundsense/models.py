"""Two deliberately compact, randomly initialized classifiers."""
from torch import nn


class MLP(nn.Module):
    def __init__(self, shape=(64, 128), hidden=(128, 64), dropout=0.4, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(shape[0] * shape[1], hidden[0]),
                                 nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden[0], hidden[1]),
                                 nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden[1], num_classes))

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, shape=(64, 128), channels=(16, 32, 64), dropout=0.3, num_classes=10):
        super().__init__()
        layers, incoming = [], 1
        for outgoing in channels:
            layers.extend([nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                           nn.BatchNorm2d(outgoing), nn.ReLU(), nn.MaxPool2d(2)])
            incoming = outgoing
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(incoming, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name, cfg, preprocessing):
    cls = {'mlp': MLP, 'cnn': CNN}[name]
    return cls(shape=(preprocessing['n_mels'], preprocessing['time_frames']), **cfg)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
