import inspect
import numpy as np
import pytest
from soundsense.common import ROOT
from soundsense.infer import Predictor
from experiments.accuracy_v2.core import manifest


def test_real_yamnet_frontend_and_repeated_prediction():
    df=manifest()
    path=ROOT/'data/raw'/df[df.fold==1].iloc[0].filename
    with Predictor() as predictor:
        first=predictor.predict(path)
        process=predictor.worker.process
        second=predictor.predict(path)
        assert first['model']=='yamnet_transfer'
        assert first['waveform'].shape==(80000,)
        assert first['mel'].shape[0]==64 and first['mel'].shape[1]>128
        assert np.isfinite(first['mel']).all()
        np.testing.assert_array_equal(first['mel'],second['mel'])
        p=np.array(list(first['probabilities'].values()))
        assert p.shape==(10,) and p.sum()==pytest.approx(1,abs=1e-5)
        np.testing.assert_array_equal(p,list(second['probabilities'].values()))
        assert len(first['top3'])==3 and predictor.worker.process is process
    assert process.poll() is not None
