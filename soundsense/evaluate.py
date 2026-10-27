"""One test inference per frozen checkpoint; all metrics derive from saved rows."""
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

from .common import ROOT, identity, read_json, sha256, write_json
from .dataset import split_dataset
from .models import build_model


def load_checkpoint(path, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt['name'], ckpt['model_config'], ckpt['preprocessing']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt


def test_predictions(name, features, manifest, stats, device):
    checkpoint = ROOT / f'models/{name}_best.pt'
    path = ROOT / f'results/predictions/{name}_test.csv'
    meta_path = path.with_suffix('.json')
    expected = manifest[manifest.split == 'test'].reset_index(drop=True)
    signature = identity({'checkpoint_sha256': sha256(checkpoint),
                          'feature_signature': stats['feature_signature'],
                          'test_files': expected[['filename', 'sha256']].to_dict('records')})
    if path.exists() and meta_path.exists():
        meta = read_json(meta_path)
        if meta['signature'] == signature and meta['csv_sha256'] == sha256(path):
            print(f'{name}: reusing saved test predictions; no repeated test inference', flush=True)
            return pd.read_csv(path)
    # Append an honest audit record if the user explicitly retrains later.
    audit_path = ROOT / 'results/predictions/test_access_audit.json'
    audit = read_json(audit_path) if audit_path.exists() else []
    if any(entry['model'] == name for entry in audit):
        warnings.warn('This model has previously been evaluated on fold 5; this is a repeated experiment, not an untouched test.')
    model, ckpt = load_checkpoint(checkpoint, device)
    if ckpt['statistics'] != stats:
        raise ValueError('Checkpoint and preprocessing statistics differ')
    loader = DataLoader(split_dataset(features, manifest, 'test'), batch_size=16, shuffle=False, num_workers=0)
    probabilities = []
    with torch.inference_mode():
        for x, _ in loader:
            probabilities.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
    probabilities = np.concatenate(probabilities)
    labels = [item['label'] for item in ckpt['class_mapping']]
    result = expected[['filename', 'fold', 'src_file', 'class_index', 'category']].rename(
        columns={'class_index': 'true_index', 'category': 'true_label'})
    result['predicted_index'] = probabilities.argmax(1)
    result['predicted_label'] = [labels[i] for i in result.predicted_index]
    for i, label in enumerate(labels):
        result[f'prob_{label}'] = probabilities[:, i]
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    write_json(meta_path, {'signature': signature, 'checkpoint_sha256': sha256(checkpoint),
                           'csv_sha256': sha256(path), 'count': len(result), 'fold': 5,
                           'probability_order': labels})
    from datetime import datetime, timezone
    audit.append({'model': name, 'signature': signature, 'utc': datetime.now(timezone.utc).isoformat(), 'count': len(result)})
    write_json(audit_path, audit)
    return result


def metrics_from_predictions(name, predictions, mapping):
    labels = [item['label'] for item in mapping]
    y = predictions.true_index.to_numpy(dtype=int)
    pred = predictions.predicted_index.to_numpy(dtype=int)
    probs = predictions[[f'prob_{label}' for label in labels]].to_numpy()
    if len(y) != 80 or not (predictions.fold == 5).all() or not predictions.filename.is_unique:
        raise ValueError('Expected exactly 80 unique held-out fold-5 predictions')
    if not np.isfinite(probs).all() or not np.allclose(probs.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid probabilities')
    if not (pred == probs.argmax(1)).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError('Invalid predicted indices or probabilities')
    folder = ROOT / 'results/metrics'
    folder.mkdir(parents=True, exist_ok=True)
    report = classification_report(y, pred, labels=list(range(10)), target_names=labels, output_dict=True, zero_division=0)
    write_json(folder / f'{name}_classification_report.json', report)
    (folder / f'{name}_classification_report.txt').write_text(
        classification_report(y, pred, labels=list(range(10)), target_names=labels, digits=4, zero_division=0), encoding='utf-8')
    pd.DataFrame([{'class': label, **report[label]} for label in labels]).to_csv(folder / f'{name}_per_class.csv', index=False)
    matrix = confusion_matrix(y, pred, labels=range(10))
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(folder / f'{name}_confusion_matrix.csv', index_label='true_class')
    pd.DataFrame(matrix / matrix.sum(1, keepdims=True), index=labels, columns=labels).to_csv(
        folder / f'{name}_confusion_matrix_normalized.csv', index_label='true_class')
    summary = {'accuracy': accuracy_score(y, pred), 'balanced_accuracy': balanced_accuracy_score(y, pred)}
    for average in ['macro', 'weighted']:
        p, r, f1, _ = precision_recall_fscore_support(y, pred, average=average, zero_division=0)
        summary.update({f'{average}_precision': float(p), f'{average}_recall': float(r), f'{average}_f1': float(f1)})
    curves, aucs, aps = [], [], []
    curve_dir = folder / f'{name}_curves'
    curve_dir.mkdir(exist_ok=True)
    for i, label in enumerate(labels):
        binary = (y == i).astype(int)
        if len(np.unique(binary)) < 2:
            curves.append({'class': label, 'roc_auc': None, 'average_precision': None, 'undefined_reason': 'Positive or negative class missing'})
            continue
        fpr, tpr, thresholds = roc_curve(binary, probs[:, i])
        # sklearn uses inf as the initial ROC threshold; empty CSV field denotes it.
        pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': np.where(np.isfinite(thresholds), thresholds, np.nan)}).to_csv(
            curve_dir / f'{label}_roc.csv', index=False)
        precision, recall, thresholds = precision_recall_curve(binary, probs[:, i])
        pd.DataFrame({'precision': precision, 'recall': recall, 'threshold': np.r_[thresholds, np.nan]}).to_csv(
            curve_dir / f'{label}_pr.csv', index=False)
        auc = float(roc_auc_score(binary, probs[:, i]))
        ap = float(average_precision_score(binary, probs[:, i]))
        aucs.append(auc)
        aps.append(ap)
        curves.append({'class': label, 'roc_auc': auc, 'average_precision': ap})
    summary.update({'macro_roc_auc_ovr': float(np.mean(aucs)) if len(aucs) == 10 else None,
                    'macro_average_precision': float(np.mean(aps)) if len(aps) == 10 else None,
                    'per_class_ranking': curves, 'test_count': len(y),
                    'zero_division_policy': 'precision/recall/F1=0 when undefined; all ten classes have eight test positives',
                    'ap_definition': 'non-interpolated average precision, not trapezoidal PR-AUC'})
    write_json(folder / f'{name}_metrics.json', summary)
    return summary
