import numpy as np
import pandas as pd
import torch

from soundsense.common import ROOT, read_json
from soundsense.evaluate import load_checkpoint
from soundsense.infer import Predictor
from soundsense.preprocess import feature, standardize


def test_checkpoint_inference_matches_training_transform():
    df = pd.read_csv(ROOT / 'data/metadata/manifest.csv')
    path = ROOT / 'data/raw' / df[df.split == 'train'].iloc[0].filename
    predictor = Predictor()
    result = predictor.predict(path)
    stats = read_json(ROOT / 'artifacts/preprocessing_stats.json')
    x = standardize(feature(path, stats['preprocessing']), stats)
    model, metadata = load_checkpoint(ROOT / 'models/cnn_best.pt')
    with torch.inference_mode():
        expected = torch.softmax(model(torch.from_numpy(x[None, None])), 1)[0].numpy()
    actual = np.array(list(result['probabilities'].values()))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert abs(actual.sum() - 1) < 1e-5
    assert len(result['top3']) == 3
    assert metadata['statistics'] == stats


def test_saved_predictions_match_metrics_and_are_fold_five_only():
    from sklearn.metrics import accuracy_score
    for name in ['mlp', 'cnn']:
        p = pd.read_csv(ROOT / f'results/predictions/{name}_test.csv')
        metrics = read_json(ROOT / f'results/metrics/{name}_metrics.json')
        assert len(p) == 80 and (p.fold == 5).all()
        assert accuracy_score(p.true_index, p.predicted_index) == metrics['accuracy']
        assert len([c for c in p if c.startswith('prob_')]) == 10
