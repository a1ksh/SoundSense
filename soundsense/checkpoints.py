"""Checkpoint loading requires the patched PyTorch weights-only unpickler."""
from packaging.version import Version
import torch


def load_weights(path, device='cpu'):
    if Version(torch.__version__.split('+')[0]) < Version('2.10.0'):
        raise RuntimeError('Checkpoint loading requires torch>=2.10.0 (CVE-2026-24747). '
                           'Install requirements.txt with the matching torchvision build.')
    return torch.load(path, map_location=device, weights_only=True)
