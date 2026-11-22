"""YAMNet-only split, frozen feature cache and checkpoint contracts."""
import os
import importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from soundsense.common import ROOT, identity, read_json, code_identity, sha256
from soundsense.models import TransferHead
from soundsense.checkpoints import load_weights

REPORT = ROOT / 'reports/accuracy_v2'
MODELS = ROOT / 'models/accuracy_v2'
CACHE = ROOT / 'data/accuracy_v2_cache'
RECIPE_PATH = Path(__file__).with_name('config.json')
YAMNET_URL = 'https://tfhub.dev/google/yamnet/1'
MODEL_NAME = 'yamnet_transfer'

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
        raise ValueError('Class order differs from ESC-10 mapping')
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



def build_candidate(name=MODEL_NAME, dimension=1024):
    if name != MODEL_NAME or dimension != 1024:
        raise ValueError('Only YAMNet with 1024 features is supported')
    return TransferHead(dimension)


def load_candidate(path, device='cpu'):
    payload = load_weights(path, device)
    if payload.get('schema') != 'accuracy_v2_1':
        raise ValueError('Unsupported YAMNet checkpoint format')
    model = build_candidate(payload.get('name'), payload.get('dimension')).to(device)
    mapping = payload.get('class_mapping', [])
    if mapping != read_json(ROOT / 'artifacts/class_mapping.json'):
        raise ValueError('YAMNet checkpoint class order differs from ESC-10 mapping')
    cfg = payload.get('preprocessing', {})
    if cfg.get('sample_rate') != 16000 or cfg.get('duration_seconds') != 5:
        raise ValueError('YAMNet requires 16 kHz five-second waveform preprocessing')
    for key in ['mean', 'std']:
        values = np.asarray(payload.get(key), dtype=np.float64)
        if values.shape != (1024,) or not np.isfinite(values).all() or (key == 'std' and (values < 1e-6).any()):
            raise ValueError(f'Invalid YAMNet train {key}')
    model.load_state_dict(payload['state_dict'], strict=True)
    return model.eval(), payload


def cache_key(df, name=MODEL_NAME, extra=None):
    if name != MODEL_NAME:
        raise ValueError('Only YAMNet embeddings are supported')
    return identity({'name':name, 'files':df[['filename','sha256']].to_dict('records'),
                     'waveform':{'sample_rate':16000,'duration_seconds':5,'mono':'mean','resample':'soxr_hq'},
                     'implementation':code_identity(['preprocess.py']),
                     'extractor_sha256':sha256(Path(__file__).with_name('extract_yamnet.py')),
                     'packages':{p:importlib.metadata.version(p) for p in ['numpy','librosa','soundfile','soxr']},
                     'version':2,'extra':extra})


def cache_environment():
    os.environ.setdefault('TFHUB_CACHE_DIR', str(CACHE / 'tfhub'))
    CACHE.mkdir(parents=True, exist_ok=True)
