"""The single audio transformation used by both training and inference."""
import importlib.metadata
import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import zoom
from tqdm import tqdm

from .common import ROOT, code_identity, identity, read_json, sha256, write_json


def load_audio(path, cfg):
    y, sr = sf.read(path, dtype='float32', always_2d=True)
    if not len(y) or not np.isfinite(y).all():
        raise ValueError('Audio must be nonempty and finite')
    y = y.mean(axis=1)
    if sr != cfg['sample_rate']:
        y = librosa.resample(y, orig_sr=sr, target_sr=cfg['sample_rate'], res_type='soxr_hq')
    n = int(cfg['sample_rate'] * cfg['duration_seconds'])
    # Deterministic first-five-seconds crop, right-zero-pad for short clips.
    return np.pad(y[:n], (0, max(0, n - len(y)))).astype(np.float32)


def logmel(y, cfg):
    power = librosa.feature.melspectrogram(
        y=y, sr=cfg['sample_rate'], n_fft=cfg['n_fft'], hop_length=cfg['hop_length'],
        n_mels=cfg['n_mels'], fmin=cfg['fmin'], fmax=cfg['fmax'],
        power=2.0, center=True, pad_mode='constant', window='hann',
        htk=False, norm='slaney')
    db = librosa.power_to_db(power, ref=1.0, top_db=cfg['top_db'])
    # Bilinear time interpolation of the whole clip, no frequency-axis resizing.
    resized = zoom(db, (1, cfg['time_frames'] / db.shape[1]), order=1, prefilter=False)
    if resized.shape != (cfg['n_mels'], cfg['time_frames']):
        raise ValueError(f'Unexpected spectrogram shape {resized.shape}')
    return resized.astype(np.float32)


def feature(path, cfg):
    return logmel(load_audio(path, cfg), cfg)


def training_statistics(features, train_mask):
    train = features[np.asarray(train_mask)]
    if not len(train):
        raise ValueError('Empty training set')
    mean, std = float(train.mean(dtype=np.float64)), float(train.std(dtype=np.float64))
    return {'mean': mean, 'std': max(std, 1e-6), 'fit_split': 'train', 'fit_folds': [1, 2, 3],
            'fit_count': len(train), 'method': 'one global scalar mean/std over training pixels'}


def standardize(x, stats):
    return ((x - stats['mean']) / stats['std']).astype(np.float32)


def prepare_features(df, cfg):
    cache = ROOT / 'data/cache/features.npz'
    meta = ROOT / 'data/cache/features.json'
    signature = identity({'preprocessing': cfg, 'files': df[['filename', 'sha256']].to_dict('records'),
                          'implementation': code_identity(['preprocess.py']),
                          'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'librosa', 'scipy', 'soundfile', 'soxr']}})
    if cache.exists() and meta.exists() and read_json(meta)['signature'] == signature:
        if sha256(cache) != read_json(meta)['cache_sha256']:
            raise ValueError('Feature cache checksum mismatch; remove data/cache to rebuild')
        with np.load(cache, allow_pickle=False) as data:
            features = data['features']
        if features.shape != (400, cfg['n_mels'], cfg['time_frames']) or not np.isfinite(features).all():
            raise ValueError('Invalid feature cache')
    else:
        features = np.stack([feature(ROOT / 'data/raw' / f, cfg) for f in tqdm(df.filename, desc='Log-Mel')])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, features=features)
        write_json(meta, {'signature': signature, 'cache_sha256': sha256(cache), 'shape': list(features.shape)})
    stats = training_statistics(features, df.split.to_numpy() == 'train')
    stats['feature_signature'] = signature
    stats['preprocessing'] = cfg
    write_json(ROOT / 'artifacts/preprocessing_stats.json', stats)
    return standardize(features, stats), stats
