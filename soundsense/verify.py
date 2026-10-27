"""Audit the requested deliverables without performing additional test inference."""
import numpy as np
import pandas as pd
from PIL import Image

from .common import ROOT, read_json, sha256, write_json

REQUIRED_FIGURES = [
    'class_distribution.png', 'audio_waveform_examples.png', 'mel_spectrogram_examples.png',
    'mlp_loss_curve.png', 'cnn_loss_curve.png', 'mlp_accuracy_curve.png', 'cnn_accuracy_curve.png',
    'models_training_comparison.png', 'mlp_confusion_matrix.png', 'mlp_confusion_matrix_normalized.png',
    'cnn_confusion_matrix.png', 'cnn_confusion_matrix_normalized.png', 'per_class_f1_comparison.png',
    'precision_recall_per_class.png', 'roc_curves_per_class.png', 'model_metrics_comparison.png',
    'cnn_prediction_confidence.png', 'error_analysis_examples.png',
]


def verify():
    inventory = read_json(ROOT / 'results/figures/inventory.json')
    actual = {item['file'].split('/')[-1] for item in inventory}
    if not set(REQUIRED_FIGURES) <= actual:
        raise ValueError(f'Missing coursework figures: {set(REQUIRED_FIGURES) - actual}')
    for item in inventory:
        path = ROOT / item['file']
        if sha256(path) != item['sha256']:
            raise ValueError(f'Figure checksum mismatch: {path.name}')
        with Image.open(path) as image:
            image.verify()
    summary = read_json(ROOT / 'results/summary.json')
    audit = read_json(ROOT / 'results/predictions/test_access_audit.json')
    model_results = {}
    for name in ['mlp', 'cnn']:
        training = summary['models'][name]['training']
        history = pd.read_csv(ROOT / f'results/history/{name}.csv')
        checkpoint = ROOT / f'models/{name}_best.pt'
        if len(history) != training['epochs_completed']:
            raise ValueError('Recorded epoch count differs from history')
        best = history.loc[history.validation_loss.idxmin()]
        if int(best.epoch) != training['best_epoch'] or not np.isclose(best.validation_loss, training['best_validation_loss']):
            raise ValueError('Checkpoint was not selected by minimum validation loss')
        if sha256(checkpoint) != training['checkpoint_sha256']:
            raise ValueError('Checkpoint checksum mismatch')
        predictions_path = ROOT / f'results/predictions/{name}_test.csv'
        prediction_meta = read_json(predictions_path.with_suffix('.json'))
        if prediction_meta['checkpoint_sha256'] != sha256(checkpoint) or prediction_meta['csv_sha256'] != sha256(predictions_path):
            raise ValueError('Predictions do not match frozen checkpoint metadata')
        p = pd.read_csv(predictions_path)
        manifest = pd.read_csv(ROOT / 'data/metadata/manifest.csv')
        heldout = manifest[manifest.split == 'test'].reset_index(drop=True)
        if list(p.filename) != list(heldout.filename) or list(p.true_index) != list(heldout.class_index):
            raise ValueError('Saved predictions differ from held-out manifest')
        model_results[name] = {'epochs': len(history), 'best_epoch': int(best.epoch),
                               'checkpoint_bytes': checkpoint.stat().st_size, 'test_prediction_rows': len(p),
                               'test_inference_passes_recorded': sum(entry['model'] == name for entry in audit)}
    if not read_json(ROOT / 'results/gradio_smoke.json')['passed']:
        raise ValueError('Gradio HTTP smoke test did not pass')
    result = {'passed': True, 'required_figures': len(REQUIRED_FIGURES), 'total_figures': len(inventory),
              'all_300_dpi': all(min(item['dpi']) >= 299 for item in inventory), 'models': model_results,
              'study_design': summary['study_design']}
    write_json(ROOT / 'results/verification.json', result)
    return result


if __name__ == '__main__':
    print(verify())
