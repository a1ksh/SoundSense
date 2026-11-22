"""Deterministic waveform preparation shared by YAMNet training and inference.

The log-Mel frontend lives inside the official YAMNet model.
"""
import numpy as np
import librosa
import soundfile as sf


def load_audio(path, cfg):
    y, sr = sf.read(path, dtype='float32', always_2d=True)
    if not len(y) or not np.isfinite(y).all():
        raise ValueError('Audio must be nonempty and finite')
    y = y.mean(axis=1)
    if sr != cfg['sample_rate']:
        y = librosa.resample(y, orig_sr=sr, target_sr=cfg['sample_rate'], res_type='soxr_hq')
    n = int(cfg['sample_rate'] * cfg['duration_seconds'])
    return np.pad(y[:n], (0, max(0,n-len(y)))).astype(np.float32)
