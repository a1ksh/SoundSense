"""Deployment regressions for the trained EfficientNet and two-model app."""
import numpy as np
import pandas as pd
import pytest
import torch
from soundsense.common import ROOT
from soundsense.efficientnet import EfficientNetPredictor, backbone_input


def test_efficientnet_live_audio_matches_colab_prediction():
    torch.set_num_threads(4)
    frame=pd.read_csv(ROOT/'colab_two_models/run/predictions/efficientnet_b0_mel_fixed.csv')
    row=frame.iloc[0]
    predictor=EfficientNetPredictor()
    predictor.warmup()
    result=predictor.predict(ROOT/'data/raw'/row.filename)
    expected=row[[f'prob_{label}' for label in predictor.labels]].to_numpy(dtype=float)
    actual=np.asarray(list(result['probabilities'].values()))
    np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=2e-5)
    assert result['label']==row.predicted_label
    assert result['waveform'].shape==(80000,) and result['mel'].shape==(64,128)
    assert result['mel_frame_hop']==5/128 and result['mel_scale_label']=='Қуат (дБ)'
    assert len(result['top3'])==3 and not predictor.backbone.training
    assert not any(p.requires_grad for p in predictor.backbone.parameters())
    predictor.close()


def test_efficientnet_transform_is_per_clip():
    values=torch.linspace(-80,10,2*64*128).reshape(2,64,128)
    together=backbone_input(values)
    assert together.shape==(2,3,224,224) and torch.isfinite(together).all()
    torch.testing.assert_close(together[:1],backbone_input(values[0]))


def test_failed_second_model_closes_ready_yamnet(monkeypatch):
    import app
    closed=[]
    class Ready:
        def close(self): closed.append(True)
    monkeypatch.setattr(app,'ready_yamnet',lambda checkpoint:Ready())
    def broken(): raise RuntimeError('EfficientNet weights missing')
    monkeypatch.setattr(app,'ready_efficientnet',broken)
    with pytest.raises(RuntimeError,match='EfficientNet weights missing'):
        app.create_app()
    assert closed==[True]


def test_efficientnet_missing_artifacts_is_clear(tmp_path,monkeypatch):
    import soundsense.efficientnet as module
    monkeypatch.setattr(module,'DEPLOYMENT',tmp_path)
    with pytest.raises(RuntimeError,match='моделі табылмады'):
        module.EfficientNetPredictor()
