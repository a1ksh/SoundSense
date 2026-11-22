"""YAMNet splits, checkpoint contracts and preserved evaluation regressions."""
import numpy as np
import pandas as pd
import pytest
import torch
from soundsense.common import ROOT, read_json
from experiments.accuracy_v2.core import MODELS, REPORT, build_candidate, load_candidate, manifest, split_masks


@pytest.mark.parametrize('fold',range(1,6))
def test_official_partitions_and_source_disjointness(fold):
    df=manifest()
    train,val,test=split_masks(df,fold)
    assert [int(x.sum()) for x in (train,val,test)]==[240,80,80]
    assert set(df.loc[test,'fold'])=={fold}
    assert set(df.loc[val,'fold'])=={fold-1 or 5}
    for a,b in [(train,val),(train,test),(val,test)]:
        assert not set(df.loc[a,'src_file']) & set(df.loc[b,'src_file'])
        assert not np.any(a&b)
    assert np.all(train|val|test)


def test_all_saved_heads_reproduce_400_oof_probabilities_and_train_scalers():
    from experiments.accuracy_v2.verify import verify_predictions
    result,_,_=verify_predictions()
    assert result['oof_count']==400 and result['oof_correct']==391
    assert result['train_only_scalers_verified']==5


def test_fixed_and_oof_metrics_recompute_from_saved_scores():
    from experiments.accuracy_v2.report import score_predictions
    labels=[r['label'] for r in read_json(ROOT/'artifacts/class_mapping.json')]
    for run,count,correct in [('fixed',80,78),('oof',400,391)]:
        frame=pd.read_csv(REPORT/f'predictions/yamnet_transfer_{run}.csv')
        actual=score_predictions(frame,labels,count)
        saved=read_json(REPORT/f'metrics/yamnet_transfer_{run}.json')
        assert actual['count']==count and actual['correct']==correct
        for key in ['accuracy','macro_f1','macro_roc_auc_ovr','macro_average_precision']:
            assert actual[key]==pytest.approx(saved[key],abs=1e-12)


@pytest.mark.parametrize('name',['mlp','cnn_v2','efficientnet_b0_mel'])
def test_yamnet_experiment_rejects_other_architecture_checkpoints(name):
    with pytest.raises(ValueError,match='Only YAMNet'):
        build_candidate(name,1024)


def test_missing_checkpoint_fails_clearly(tmp_path):
    from soundsense.infer import Predictor
    with pytest.raises(RuntimeError,match='табылмады'):
        Predictor(tmp_path/'missing.pt')


def test_unsupported_checkpoint_is_rejected(tmp_path):
    path=tmp_path/'bad.pt'
    torch.save({'schema':'unsupported'},path)
    with pytest.raises(ValueError,match='Unsupported YAMNet'):
        load_candidate(path)


@pytest.mark.parametrize('change',['class_order','mean_shape','std_zero','mean_nan','dimension','sample_rate'])
def test_malformed_checkpoint_cannot_predict(change,tmp_path):
    from soundsense.checkpoints import load_weights
    payload=load_weights(MODELS/'yamnet_transfer_fixed.pt')
    if change=='class_order':
        payload['class_mapping']=payload['class_mapping'][::-1]
    elif change=='mean_shape':
        payload['mean']=[0.]
    elif change=='std_zero':
        payload['std'][0]=0
    elif change=='mean_nan':
        payload['mean'][0]=float('nan')
    elif change=='dimension':
        payload['dimension']=1280
    else:
        payload['preprocessing']['sample_rate']=44100
    path=tmp_path/'bad.pt'
    torch.save(payload,path)
    with pytest.raises(ValueError):
        load_candidate(path)


def test_missing_tensorflow_runtime_is_explicit(monkeypatch,tmp_path):
    from soundsense.infer import Predictor
    monkeypatch.setenv('SOUNDSENSE_YAMNET_PYTHON',str(tmp_path/'missing.exe'))
    with pytest.raises(RuntimeError,match='SOUNDSENSE_YAMNET_PYTHON'):
        Predictor()


def test_yamnet_experiment_weights_and_result_plots_are_preserved():
    checkpoints=list(MODELS.glob('*.pt'))
    assert len(checkpoints)==5 and all(p.name.startswith('yamnet_transfer_') for p in checkpoints)
    plots=list((REPORT/'plots').glob('*.png'))
    assert len(plots)==16 and all(p.name.startswith('yamnet_transfer_') for p in plots)
    all_plots=list((ROOT/'results/figures').glob('*.png'))
    assert len(all_plots)==18
    assert all(p.name.startswith('yamnet_transfer_') or p.name in ['class_distribution.png','audio_waveform_examples.png'] for p in all_plots)
    assert set(read_json(REPORT/'cross_validation_summary.json'))=={'yamnet_transfer'}
