import numpy as np
import soundfile as sf
import pytest

from soundsense.common import config
from soundsense.preprocess import feature, load_audio, standardize, training_statistics


def test_resampling_stereo_padding_and_finite_silence(tmp_path):
    cfg = config()['preprocessing']
    path = tmp_path / 'stereo.wav'
    stereo = np.column_stack([np.ones(8000) * .25, np.ones(8000) * -.25])
    sf.write(path, stereo, 8000, subtype='FLOAT')
    y = load_audio(path, cfg)
    assert y.shape == (80000,)
    assert np.max(np.abs(y)) < 1e-6
    x = feature(path, cfg)
    assert x.shape == (64, 128) and np.isfinite(x).all()


def test_long_clip_first_five_seconds(tmp_path):
    cfg = config()['preprocessing']
    path = tmp_path / 'long.wav'
    y = np.r_[np.zeros(80000), np.ones(16000)].astype('float32')
    sf.write(path, y, 16000, subtype='FLOAT')
    assert not load_audio(path, cfg).any()


def test_statistics_exclude_validation_and_test():
    x = np.array([np.zeros((64, 128)), np.ones((64, 128)) * 2, np.ones((64, 128)) * 1000], dtype='float32')
    stats = training_statistics(x, [True, True, False])
    assert stats['mean'] == 1 and stats['std'] == 1 and stats['fit_count'] == 2
    normalized = standardize(x[:2], stats)
    assert abs(normalized.mean()) < 1e-7 and abs(normalized.std() - 1) < 1e-7


def test_nonfinite_audio_rejected(tmp_path):
    path = tmp_path / 'nan.wav'
    sf.write(path, np.array([np.nan], dtype='float32'), 16000, subtype='FLOAT')
    with pytest.raises(ValueError, match='finite'):
        load_audio(path, config()['preprocessing'])
