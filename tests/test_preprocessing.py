import numpy as np
import pytest
import soundfile as sf
from soundsense.common import config
from soundsense.preprocess import load_audio


def test_resampling_stereo_padding_and_finite_silence(tmp_path):
    path=tmp_path/'stereo.wav'
    sf.write(path,np.column_stack([np.ones(8000)*.25,np.ones(8000)*-.25]),8000,subtype='FLOAT')
    y=load_audio(path,config()['preprocessing'])
    assert y.shape==(80000,) and np.max(np.abs(y))<1e-6


def test_long_clip_uses_first_five_seconds(tmp_path):
    path=tmp_path/'long.wav'
    sf.write(path,np.r_[np.zeros(80000),np.ones(16000)].astype('float32'),16000,subtype='FLOAT')
    assert not load_audio(path,config()['preprocessing']).any()


def test_nonfinite_audio_is_rejected(tmp_path):
    path=tmp_path/'nan.wav'
    sf.write(path,np.array([np.nan],dtype='float32'),16000,subtype='FLOAT')
    with pytest.raises(ValueError,match='finite'):
        load_audio(path,config()['preprocessing'])
