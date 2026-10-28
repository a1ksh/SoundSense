"""Shared split, model, transform and checkpoint contract for V2 and app."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from soundsense.common import ROOT, config, identity, read_json, sha256
from soundsense.models import CNN

REPORT = ROOT / 'reports/accuracy_v2'
MODELS = ROOT / 'models/accuracy_v2'
CACHE = ROOT / 'data/accuracy_v2_cache'
RECIPE_PATH = Path(__file__).with_name('config.json')
YAMNET_URL = 'https://tfhub.dev/google/yamnet/1'


def manifest():
    df = pd.read_csv(ROOT / 'data/metadata/manifest.csv', dtype={'src_file': str})
    if len(df) != 400 or not df.filename.is_unique:
        raise ValueError('Expected exactly 400 unique original ESC-10 clips')
    if df.category.nunique() != 10 or not (df.groupby('category').size() == 40).all():
        raise ValueError('Expected ten classes of forty clips')
    if not (df.groupby(['fold', 'category']).size() == 8).all() or set(df.fold) != set(range(1, 6)):
        raise ValueError('Invalid official class/fold counts')
    if (df.groupby('src_file').fold.nunique() != 1).any():
        raise ValueError('Source recording leakage between folds')
    mapping = read_json(ROOT / 'artifacts/class_mapping.json')
    label_index = {row['label']: row['index'] for row in mapping}
    if not (df.category.map(label_index) == df.class_index).all():
        raise ValueError('Class order differs from baseline mapping')
    return df


def split_masks(df, test_fold=5):
    if test_fold not in range(1, 6):
        raise ValueError('Test fold must be 1..5')
    validation_fold = test_fold - 1 or 5
    train = ~df.fold.isin([test_fold, validation_fold]).to_numpy()
    val = (df.fold == validation_fold).to_numpy()
    test = (df.fold == test_fold).to_numpy()
    if [int(m.sum()) for m in (train, val, test)] != [240, 80, 80]:
        raise ValueError('Invalid clip split')
    for a, b in [(train, val), (train, test), (val, test)]:
        if set(df.loc[a, 'src_file']) & set(df.loc[b, 'src_file']):
            raise ValueError('Cross-partition source leakage')
    return train, val, test


def efficientnet_input(mels):
    """Identical per-clip transform for frozen extraction and app inference.

    Fixed 80-dB relative log-Mel range mapped to [0,1], RGB repeat, bilinear
    224x224 resize, ImageNet channel normalization. No data-fitted statistics.
    """
    x = torch.as_tensor(mels, dtype=torch.float32)
    if x.ndim == 2:
        x = x[None]
    maximum = x.amax(dim=(-2, -1), keepdim=True)
    x = ((x - maximum + 80) / 80).clamp(0, 1)
    x = F.interpolate(x[:, None].repeat(1, 3, 1, 1), size=(224, 224), mode='bilinear', align_corners=False)
    mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
    std = x.new_tensor([.229, .224, .225])[None, :, None, None]
    return (x - mean) / std


def efficientnet_backbone(pretrained=False):
    try:
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        net = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError('EfficientNet requires matching torchvision==0.22.1 and packaged pretrained weights. See experiments/accuracy_v2/README.md.') from exc
    net.classifier = nn.Identity()
    net.requires_grad_(False)
    return net.eval()


class TransferHead(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(.2), nn.Linear(dimension, 10))

    def forward(self, x):
        return self.net(x)


def build_candidate(name, dimension=None):
    if name == 'cnn_v2':
        return CNN()
    if name in ('efficientnet_b0_mel', 'yamnet_transfer') and dimension:
        return TransferHead(dimension)
    raise ValueError(f'Unsupported V2 model: {name}')


def load_candidate(path, device='cpu'):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get('schema') != 'accuracy_v2_1':
        raise ValueError('Unsupported V2 checkpoint format')
    labels = [r['label'] for r in payload['class_mapping']]
    if len(labels) != 10 or len(set(labels)) != 10:
        raise ValueError('V2 checkpoint must contain ten ordered labels')
    model = build_candidate(payload['name'], payload.get('dimension')).to(device)
    model.load_state_dict(payload['state_dict'], strict=True)
    return model.eval(), payload


def raw_mels(df):
    # Baseline cache is read, never rewritten; verify manifest ordering first.
    cache = ROOT / 'data/cache/features.npz'
    meta = ROOT / 'data/cache/features.json'
    if cache.exists() and meta.exists() and sha256(cache) == read_json(meta)['cache_sha256']:
        with np.load(cache, allow_pickle=False) as data:
            values = data['features'].copy()
        if values.shape == (400, 64, 128) and np.isfinite(values).all():
            return values
    from soundsense.preprocess import feature
    cfg = config()['preprocessing']
    return np.stack([feature(ROOT / 'data/raw' / f, cfg) for f in df.filename])


def cache_key(df, name, extra=None):
    return identity({'name': name, 'files': df[['filename', 'sha256']].to_dict('records'),
                     'preprocessing': config()['preprocessing'], 'version': 1, 'extra': extra})


def cache_environment():
    os.environ.setdefault('TORCH_HOME', str(CACHE / 'torch'))
    os.environ.setdefault('TFHUB_CACHE_DIR', str(CACHE / 'tfhub'))
    CACHE.mkdir(parents=True, exist_ok=True)
