"""V2 leakage, preprocessing, checkpoint and persisted evaluation regression."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from soundsense.common import ROOT, read_json
from experiments.accuracy_v2.core import (CACHE, MODELS, REPORT, efficientnet_input,
                                         load_candidate, manifest, raw_mels, split_masks)


@pytest.mark.parametrize('fold', range(1, 6))
def test_official_nested_partitions_and_source_disjointness(fold):
    df = manifest()
    train, val, test = split_masks(df, fold)
    assert [int(x.sum()) for x in (train, val, test)] == [240, 80, 80]
    assert set(df.loc[test, 'fold']) == {fold}
    assert set(df.loc[val, 'fold']) == {fold - 1 or 5}
    for left, right in ((train, val), (train, test), (val, test)):
        assert not set(df.loc[left, 'src_file']) & set(df.loc[right, 'src_file'])
        assert not np.any(left & right)
    assert np.all(train | val | test)


def test_imagenet_transform_is_clipwise_and_finite():
    mels = torch.linspace(-90, 20, 2 * 64 * 128).reshape(2, 64, 128)
    batched = efficientnet_input(mels)
    assert batched.shape == (2, 3, 224, 224)
    torch.testing.assert_close(batched[:1], efficientnet_input(mels[0]))
    assert torch.isfinite(batched).all()


@pytest.mark.parametrize('name', ['cnn_v2', 'efficientnet_b0_mel', 'yamnet_transfer'])
def test_checkpoint_label_order_output_and_training_only_scaling(name):
    path = MODELS / f'{name}_fixed.pt'
    if not path.exists():
        pytest.skip(f'{name} was not trained; see run_status.json')
    df = manifest()
    train, _, _ = split_masks(df, 5)
    model, payload = load_candidate(path)
    assert payload['class_mapping'] == read_json(ROOT / 'artifacts/class_mapping.json')
    assert payload['fit_count'] == 240 and payload['train_folds'] == [1, 2, 3]
    assert payload['validation_fold'] == 4 and payload['test_fold'] == 5
    if name == 'cnn_v2':
        values = raw_mels(df)
        mean, std = float(values[train].mean(dtype=np.float64)), float(values[train].std(dtype=np.float64))
    else:
        with np.load(CACHE / f'{name}_embeddings.npz', allow_pickle=False) as z:
            values = z['embeddings']
        mean = values[train].mean(axis=0, dtype=np.float64)
        std = np.maximum(values[train].std(axis=0, dtype=np.float64), 1e-6)
    np.testing.assert_allclose(payload['mean'], mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(payload['std'], std, rtol=1e-12, atol=1e-12)
    x = ((values[train][:2] - mean) / std).astype(np.float32)
    if name == 'cnn_v2':
        x = x[:, None]
    with torch.inference_mode():
        first = model(torch.from_numpy(x)).softmax(1)
        second = model(torch.from_numpy(x)).softmax(1)
    assert first.shape == (2, 10)
    torch.testing.assert_close(first.sum(1), torch.ones(2))
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_all_oof_rows_are_original_clips_and_each_model_excludes_its_test_fold():
    status = read_json(REPORT / 'run_status.json')
    df = manifest()
    for name in status['cv_candidates']:
        frame = pd.read_csv(REPORT / f'predictions/{name}_oof.csv')
        assert len(frame) == 400 and frame.filename.is_unique
        merged = frame.merge(df, on='filename', suffixes=('_prediction', '_manifest'), validate='one_to_one')
        assert len(merged) == 400
        np.testing.assert_array_equal(merged.fold_prediction, merged.fold_manifest)
        np.testing.assert_array_equal(merged.true_index, merged.class_index)
        for fold in range(1, 6):
            run_id = f'{name}_fixed' if fold == 5 else f'{name}_fold{fold}'
            _, payload = load_candidate(MODELS / f'{run_id}.pt')
            assert fold not in payload['train_folds']
            assert payload['validation_fold'] not in payload['train_folds']
            assert payload['test_fold'] == fold


def test_frozen_selection_agrees_with_validation_ranking_and_best_checkpoint():
    from soundsense.common import sha256
    selection = read_json(REPORT / 'selection.json')
    ranking = selection['validation_ranking']
    assert selection['name'] == min(ranking, key=lambda r: r['validation_loss'])['name']
    assert sha256(MODELS / 'best_v2.pt') == selection['checkpoint_sha256']


def test_missing_checkpoint_fails_clearly(tmp_path):
    from experiments.accuracy_v2.inference import V2Predictor
    with pytest.raises(RuntimeError, match='табылмады'):
        V2Predictor(tmp_path / 'missing.pt')


def test_unsupported_checkpoint_does_not_predict(tmp_path):
    path = tmp_path / 'unsupported.pt'
    torch.save({'schema': 'unsupported'}, path)
    with pytest.raises(ValueError, match='Unsupported V2'):
        load_candidate(path)


def test_waveform_augmentation_reads_training_clips_only(tmp_path, monkeypatch):
    from experiments.accuracy_v2 import run
    from experiments.accuracy_v2.core import RECIPE_PATH
    df = manifest()
    train, _, _ = split_masks(df, 5)
    recipe = dict(read_json(RECIPE_PATH)['cnn_v2'], waveform_variants=1)
    seen = []
    def load_training(path, cfg):
        seen.append(path.name)
        return np.ones(80000, dtype=np.float32) * .1
    monkeypatch.setattr(run, 'CACHE', tmp_path)
    monkeypatch.setattr(run, 'load_audio', load_training)
    monkeypatch.setattr(run, 'logmel', lambda y, cfg: np.zeros((64, 128), dtype=np.float32))
    values = run.augmented_train_mels(df, train, recipe, np.zeros((400, 64, 128), dtype=np.float32))
    assert values.shape == (2, 240, 64, 128)
    assert len(seen) == 240 and set(seen) == set(df.loc[train, 'filename'])


def test_missing_yamnet_runtime_has_clear_error(monkeypatch, tmp_path):
    from experiments.accuracy_v2.inference import V2Predictor
    path = MODELS / 'yamnet_transfer_fixed.pt'
    if not path.exists():
        pytest.skip('YAMNet did not train; see run_status.json')
    monkeypatch.setenv('SOUNDSENSE_YAMNET_PYTHON', str(tmp_path / 'missing_python.exe'))
    with pytest.raises(RuntimeError, match='SOUNDSENSE_YAMNET_PYTHON'):
        V2Predictor(path)


def test_missing_efficientnet_weights_has_clear_error(monkeypatch, tmp_path):
    from experiments.accuracy_v2 import inference
    path = MODELS / 'efficientnet_b0_mel_fixed.pt'
    monkeypatch.setattr(inference, 'MODELS', tmp_path)
    with pytest.raises(RuntimeError, match='backbone'):
        inference.V2Predictor(path)
