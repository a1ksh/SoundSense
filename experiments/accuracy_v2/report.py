"""Render figures and Kazakh reports exclusively from persisted predictions/logs."""
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

from soundsense.common import ROOT, read_json, sha256, write_json
from .core import REPORT, split_masks


def write_audit_manifests():
    source = pd.read_csv(REPORT / 'split_manifest.csv', dtype={'src_file': str})
    partitions = []
    for fold in range(1, 6):
        train, val, test = split_masks(source, fold)
        part = source[['filename', 'fold', 'src_file', 'category', 'class_index']].copy()
        part['outer_test_fold'] = fold
        part['validation_fold'] = fold - 1 or 5
        part['role'] = np.select([train, val, test], ['train', 'validation', 'test'], default='invalid')
        partitions.append(part)
    pd.concat(partitions, ignore_index=True).to_csv(REPORT / 'cv_split_manifest.csv', index=False)
    shutil.copy2(ROOT / 'artifacts/class_mapping.json', REPORT / 'class_mapping.json')


def save(fig, name):
    fig.tight_layout()
    fig.savefig(REPORT / 'plots' / f'{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def score_predictions(frame, labels, expected_count):
    if len(frame) != expected_count or not frame.filename.is_unique:
        raise ValueError('Wrong evaluation denominator or repeated original clip')
    y = frame.true_index.to_numpy(dtype=int)
    prediction = frame.predicted_index.to_numpy(dtype=int)
    p = frame[[f'prob_{label}' for label in labels]].to_numpy()
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid stored probabilities')
    if not np.array_equal(prediction, p.argmax(1)):
        raise ValueError('Stored predictions do not match probability argmax')
    correct = int((y == prediction).sum())
    return {'count': len(y), 'correct': correct, 'accuracy': accuracy_score(y, prediction),
            'macro_f1': f1_score(y, prediction, average='macro'),
            'macro_roc_auc_ovr': roc_auc_score(y, p, multi_class='ovr', average='macro', labels=np.arange(10)),
            'macro_average_precision': float(np.mean([average_precision_score(y == i, p[:, i]) for i in range(10)])),
            'classification_report': classification_report(y, prediction, labels=list(range(10)), target_names=labels, output_dict=True, zero_division=0)}


def plot_predictions(name, frame, labels, metrics):
    y = frame.true_index.to_numpy(dtype=int)
    prediction = frame.predicted_index.to_numpy(dtype=int)
    probabilities = frame[[f'prob_{label}' for label in labels]].to_numpy()
    matrix = confusion_matrix(y, prediction, labels=np.arange(10))
    for normalized in (False, True):
        values = matrix / matrix.sum(1, keepdims=True) if normalized else matrix
        suffix = '_normalized' if normalized else ''
        pd.DataFrame(values, index=labels, columns=labels).to_csv(REPORT / 'metrics' / f'{name}_confusion{suffix}.csv', index_label='true_label')
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(values, cmap='Blues', vmin=0, vmax=1 if normalized else None)
        ax.set(xticks=range(10), yticks=range(10), xticklabels=labels, yticklabels=labels,
               xlabel='Predicted label', ylabel='True label', title=f'{name}: {metrics["correct"]}/{metrics["count"]} correct')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        for i in range(10):
            for j in range(10):
                ax.text(j, i, f'{values[i,j]:.2f}' if normalized else str(values[i,j]), ha='center', va='center',
                        fontsize=8, color='white' if values[i,j] > values.max() / 2 else 'black')
        fig.colorbar(im, ax=ax)
        save(fig, f'{name}_confusion{suffix}')
    report = metrics['classification_report']
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, [report[label]['f1-score'] for label in labels], color='#7960c4')
    ax.set(ylim=(0, 1.05), ylabel='F1', title=f'{name}: per-class F1')
    plt.setp(ax.get_xticklabels(), rotation=40, ha='right')
    save(fig, f'{name}_per_class_f1')
    curves = {'roc': {}, 'precision_recall': {}}
    fig, ax = plt.subplots(figsize=(8, 6))
    grid = np.linspace(0, 1, 501)
    interpolated = []
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y == i, probabilities[:, i])
        auc = roc_auc_score(y == i, probabilities[:, i])
        ax.plot(fpr, tpr, lw=1, label=f'{label}: {auc:.3f}')
        interpolated.append(np.interp(grid, fpr, tpr))
        curves['roc'][label] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': float(auc)}
    ax.plot(grid, np.mean(interpolated, axis=0), color='black', lw=2, linestyle='--', label=f'macro mean AUC={metrics["macro_roc_auc_ovr"]:.3f}')
    ax.plot([0, 1], [0, 1], color='gray', linestyle=':')
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title=f'{name}: one-vs-rest ROC')
    ax.legend(fontsize=7, loc='lower right')
    save(fig, f'{name}_roc')
    fig, ax = plt.subplots(figsize=(8, 6))
    interpolated = []
    for i, label in enumerate(labels):
        precision, recall, _ = precision_recall_curve(y == i, probabilities[:, i])
        ap = average_precision_score(y == i, probabilities[:, i])
        ax.plot(recall, precision, lw=1, label=f'{label}: AP={ap:.3f}')
        interpolated.append(np.interp(grid, recall[::-1], precision[::-1]))
        curves['precision_recall'][label] = {'precision': precision.tolist(), 'recall': recall.tolist(), 'average_precision': float(ap)}
    ax.plot(grid, np.mean(interpolated, axis=0), color='black', lw=2, linestyle='--', label=f'macro AP={metrics["macro_average_precision"]:.3f}')
    ax.set(xlabel='Recall', ylabel='Precision', title=f'{name}: one-vs-rest precision–recall', ylim=(0, 1.03))
    ax.legend(fontsize=7, loc='lower left')
    save(fig, f'{name}_precision_recall')
    write_json(REPORT / 'metrics' / f'{name}_curves.json', curves)
    errors = frame.loc[y != prediction].copy()
    errors['predicted_probability'] = probabilities.max(axis=1)[y != prediction]
    errors.to_csv(REPORT / 'predictions' / f'{name}_misclassifications.csv', index=False)


def generate():
    for directory in ['plots', 'metrics', 'predictions', 'training_history']:
        (REPORT / directory).mkdir(parents=True, exist_ok=True)
    write_audit_manifests()
    labels = [r['label'] for r in read_json(ROOT / 'artifacts/class_mapping.json')]
    baseline = read_json(ROOT / 'results/summary.json')
    all_metrics, rows, cv = {}, [], {}
    inference = read_json(REPORT / 'inference_verification.json') if (REPORT / 'inference_verification.json').exists() else {}
    for name in ['cnn', 'mlp']:
        shutil.copy2(ROOT / f'results/predictions/{name}_test.csv', REPORT / f'predictions/{name}_v1_fixed.csv')
        shutil.copy2(ROOT / f'results/history/{name}.csv', REPORT / f'training_history/{name}_v1.csv')
    paths = sorted((REPORT / 'predictions').glob('*.csv'))
    for path in paths:
        if path.stem.endswith('_misclassifications'):
            continue
        frame = pd.read_csv(path)
        if len(frame) not in (80, 400):
            raise ValueError(f'Invalid prediction row count: {path}')
        if len(frame) == 80 and frame.fold.nunique() != 1:
            raise ValueError('A single-fold evaluation must contain one official fold')
        metrics = score_predictions(frame, labels, len(frame))
        metrics['predictions_sha256'] = sha256(path)
        write_json(REPORT / 'metrics' / f'{path.stem}.json', metrics)
        (REPORT / 'metrics' / f'{path.stem}_classification_report.txt').write_text(
            classification_report(frame.true_index, frame.predicted_index, labels=list(range(10)), target_names=labels, digits=4, zero_division=0), encoding='utf-8')
        all_metrics[path.stem] = metrics
        if path.stem.endswith(('_fixed', '_oof')):
            plot_predictions(path.stem, frame, labels, metrics)
            row = {'model': path.stem, 'evaluation': '400-clip OOF' if len(frame) == 400 else 'exploratory fixed fold 5',
                   **{k: metrics[k] for k in ['correct', 'count', 'accuracy', 'macro_f1', 'macro_roc_auc_ovr', 'macro_average_precision']}}
            meta = REPORT / 'training_history' / f'{path.stem}.json'
            if meta.exists():
                training = read_json(meta)
                row.update({key: training[key] for key in ['training_seconds', 'feature_extraction_seconds', 'parameter_count', 'model_bytes']})
                row['parameter_count_scope'] = 'CNN' if path.stem.startswith('cnn_v2') else 'trainable transfer head only'
                if path.stem.startswith('efficientnet_b0_mel'):
                    row['backbone_bytes'] = (ROOT / 'models/accuracy_v2/efficientnet_backbone.pt').stat().st_size
                elif path.stem.startswith('yamnet_transfer'):
                    row['backbone_bytes'] = sum(p.stat().st_size for p in (ROOT / 'data/accuracy_v2_cache/tfhub').rglob('*') if p.is_file())
                else:
                    row['backbone_bytes'] = 0
                row['deployment_bytes_including_backbone'] = row['model_bytes'] + row['backbone_bytes']
            elif '_v1_' in path.stem:
                training = baseline['models'][path.stem.split('_')[0]]['training']
                row.update({'training_seconds': training['training_seconds'], 'parameter_count': training['parameter_count'],
                            'model_bytes': (ROOT / f'models/{path.stem.split("_")[0]}_best.pt').stat().st_size})
            inference_name = path.stem.removesuffix('_fixed')
            if inference_name in inference:
                row['end_to_end_seconds_one_clip_cpu'] = inference[inference_name]['end_to_end_seconds_one_clip']
            rows.append(row)
        if path.stem.endswith('_oof'):
            if set(frame.fold) != set(range(1, 6)) or not (frame.groupby('fold').size() == 80).all():
                raise ValueError('OOF requires all five folds with eighty clips each')
            original = pd.read_csv(REPORT / 'split_manifest.csv')
            if set(frame.filename) != set(original.filename):
                raise ValueError('OOF does not cover the original dataset')
            per_fold = []
            for fold, group in frame.groupby('fold'):
                s = score_predictions(group, labels, 80)
                per_fold.append({'fold': int(fold), **{k: s[k] for k in ['correct', 'count', 'accuracy', 'macro_f1']}})
            cv[path.stem.removesuffix('_oof')] = {'per_fold': per_fold,
                'accuracy_mean': float(np.mean([r['accuracy'] for r in per_fold])),
                'accuracy_std_sample': float(np.std([r['accuracy'] for r in per_fold], ddof=1)),
                'macro_f1_mean': float(np.mean([r['macro_f1'] for r in per_fold])),
                'macro_f1_std_sample': float(np.std([r['macro_f1'] for r in per_fold], ddof=1)),
                'aggregate_correct': metrics['correct'], 'aggregate_count': 400, 'aggregate_macro_f1': metrics['macro_f1'],
                'single_seed': True, 'std_definition': 'sample standard deviation over five outer folds, ddof=1'}
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot([r['fold'] for r in per_fold], [r['accuracy'] for r in per_fold], 'o-', label='Accuracy')
            ax.plot([r['fold'] for r in per_fold], [r['macro_f1'] for r in per_fold], 's-', label='Macro F1')
            ax.set(xticks=range(1, 6), ylim=(0, 1.03), xlabel='Outer test fold', title=f'{path.stem}: fixed recipe, seed 42')
            ax.legend()
            save(fig, f'{path.stem}_fold_metrics')
    for path in sorted((REPORT / 'training_history').glob('*.csv')):
        history = pd.read_csv(path)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for phase in ['train', 'validation']:
            axes[0].plot(history.epoch, history[f'{phase}_loss'], label=phase)
            axes[1].plot(history.epoch, history[f'{phase}_accuracy'], label=phase)
        axes[0].set(xlabel='Epoch', ylabel='Cross entropy', title=f'{path.stem}: loss')
        axes[1].set(xlabel='Epoch', ylabel='Accuracy', title=f'{path.stem}: accuracy', ylim=(0, 1.03))
        for ax in axes:
            ax.legend()
        save(fig, f'{path.stem}_training')
    comparison = pd.DataFrame(rows)
    comparison.to_csv(REPORT / 'comparison.csv', index=False)
    fixed = comparison[comparison['evaluation'] == 'exploratory fixed fold 5']
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(fixed))
    ax.bar(x - .18, fixed.accuracy, width=.36, label='Accuracy')
    ax.bar(x + .18, fixed.macro_f1, width=.36, label='Macro F1')
    ax.set(xticks=x, xticklabels=fixed.model, ylim=(0, 1.05), title='Exploratory fixed split: 80 original fold-5 clips')
    plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
    ax.legend()
    save(fig, 'model_comparison_fixed')
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    resource_labels = fixed.model.tolist()
    axes[0].bar(resource_labels, fixed.training_seconds.fillna(0), label='training loop')
    if 'feature_extraction_seconds' in fixed:
        axes[0].bar(resource_labels, fixed.feature_extraction_seconds.fillna(0), bottom=fixed.training_seconds.fillna(0), label='frozen feature extraction')
    axes[0].set(ylabel='Seconds', title='Measured training and feature time')
    axes[0].legend(fontsize=7)
    sizes = fixed['deployment_bytes_including_backbone'].fillna(fixed.model_bytes) if 'deployment_bytes_including_backbone' in fixed else fixed.model_bytes
    axes[1].bar(resource_labels, sizes / 1024**2)
    axes[1].set(ylabel='MiB', title='Checkpoint + frozen backbone')
    if 'end_to_end_seconds_one_clip_cpu' in fixed:
        latency = fixed.dropna(subset=['end_to_end_seconds_one_clip_cpu'])
        axes[2].bar(latency.model, latency.end_to_end_seconds_one_clip_cpu)
    axes[2].set(ylabel='Seconds', title='CPU inference, one training clip')
    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=40, ha='right', fontsize=7)
    save(fig, 'model_resources')
    write_json(REPORT / 'metrics.json', all_metrics)
    write_json(REPORT / 'cross_validation_summary.json', cv)
    selection = read_json(REPORT / 'selection.json')
    lines = ['# SoundSense Accuracy V2 — нақты нәтижелер', '',
             'Барлық көрсеткіш сақталған нақты болжамдардан есептелді. Seed=42; ресми ESC-10 fold-тары сақталды.', '',
             '| Модель / бағалау | Дұрыс / барлығы | Accuracy | Macro-F1 |', '|---|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["model"]} | {row["correct"]}/{row["count"]} | {row["accuracy"]:.2%} | {row["macro_f1"]:.4f} |')
    lines += ['', f'Қолданбаға таңдалған модель: **{selection["name"]}**. Таңдау fold-4 validation cross-entropy арқылы, жаңа fold-5 бағалауынан бұрын бекітілді.', '',
              '## Бес fold бойынша бағалау', '']
    for name, summary in cv.items():
        lines.append(f'- {name}: accuracy {summary["accuracy_mean"]:.2%} ± {summary["accuracy_std_sample"]:.2%}; macro-F1 {summary["macro_f1_mean"]:.4f} ± {summary["macro_f1_std_sample"]:.4f}. Жиынтық {summary["aggregate_correct"]}/400, OOF macro-F1={summary["aggregate_macro_f1"]:.4f}.')
    lines += ['', 'Стандарттық ауытқу бес fold бойынша ddof=1 арқылы есептелді; бұл сенімділік интервалы емес.', '',
              '## Өзгерістер және дәлелдер', '',
              '- CNN V2: тек train дыбысына уақыт ығысуы, gain, әлсіз шу және SpecAugment; AdamW және validation scheduler.',
              '- EfficientNet: нақты ImageNet pretrained backbone бекітілген; 3 арналы 224×224 Mel, ImageNet normalization; жаңа 10-класты head.',
              '- YAMNet орындалса: ресми AudioSet pretrained модельдің 1024-өлшемді frame embedding-терінің орташа мәні; ESC-10 үшін бөлек 10-класты head. 521 AudioSet label-і ESC-10 label-і ретінде қолданылмайды.',
              '- Нақты орындалған модельдер мен қателер: `run_status.json`, `blockers.json`; ортамен командалар: `environment.json`.',
              '- `predictions/`: бастапқы файл, нақты/болжанған класс, 10 ықтималдық, қателер тізімі. `training_history/`: нақты epoch журналдары.',
              '- `plots/`: 300 dpi confusion matrix, normalized matrix, F1, ROC, PR, оқыту және fold графиктері. `metrics/`: сандық негізі.', '',
              '## Шектеулер', '',
              'ESC-10 шағын: барлығы 400 дыбыс, әр test fold-та 80 дыбыс, әр класта 8 мысал. Бір fold-та бір қате accuracy-ді 1.25 пайыздық тармаққа өзгертеді. '
              'Бір seed қолданылды; fold нәтижелері өзгереді. Fold 5 бұрын белгілі болғандықтан fixed-split тәжірибе exploratory. CV кандидаттары fixed validation бойынша таңдалды: бұл тәуелсіз жаңа сыртқы dataset емес, кандидат іздеудің толық nested бағалауы емес. '
              'AudioSet/ImageNet pretraining көздерінің ESC аудиосымен ықтимал қабаттасуы тексерілген жоқ. 90% алдын ала кепілдендірілген нәтиже емес; кестеде тек өлшенген сандар берілген. '
              'Open-world белгісіз санаттар қамтылмайды; softmax сенімділігі калибрленбеген. Барлық 400 дыбыспен deployment retraining жасалмады; app validation арқылы таңдалған fixed checkpoint-ті қолданады.', '',
              '## Іске қосу', '', '```powershell', '.\\.runtime\\python311\\python.exe app.py', '.\\.runtime\\python311\\python.exe -m pytest -q', '```', '',
              'Негізгі орнату және қайталау: `experiments/accuracy_v2/README.md`.']
    quality_path = REPORT / 'quality_checks.json'
    if quality_path.exists():
        quality = read_json(quality_path)
        lines += ['', '## Сапаны тексеру', '',
                  f'Нақты pytest: **{quality["pytest_passed"]} passed**, {quality["pytest_skipped"]} skipped; exit code={quality["pytest_exit_code"]}.',
                  'Бастапқы checkpoint-тер SHA-256 арқылы сақталғаны тексерілді. Train/app ықтималдықтары сәйкес, қайталама inference детерминді. Екі модельдің жергілікті Gradio HTTP API, бес output және default selector тексерілді.',
                  f'Барлық {quality["plot_count"]} нәтиже графигінің resolution метадерегі 300 dpi. Толық журнал: `pytest.txt`, `verification.log`, `quality_checks.json`.']
    if 'yamnet_transfer' in inference:
        elapsed = inference['yamnet_transfer']['end_to_end_seconds_one_clip']
        lines += ['', '## Inference уақыты', '',
                  f'Осы компьютерде YAMNet толық CPU inference бір train клипке **{elapsed:.2f} секунд** алды. Бұл бөлек TensorFlow process-інің іске қосылуын қамтиды; әр талдауда осы process қайта іске қосылады. '
                  'CNN v1 және Best V2 таңдауын пайдалануға болады. `comparison.csv` пен `inference_verification.json` нақты өлшемдерді сақтайды. '
                  'Барлық latency бір клипке тізбектей өлшенді; бастапқы librosa/JIT және framework startup әр модельде әртүрлі. Бұл бақыланатын көп-қайталаулы жылдамдық benchmark-і емес.']
    (REPORT / 'report_kk.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    methodology = '''# Accuracy V2 әдіснамасы

Дерек: Piczak, K. J. (2015), ESC: Dataset for Environmental Sound Classification.
https://github.com/karolpiczak/ESC-50 ; ESC-10 CC BY 3.0. Түпнұсқа лицензия мен
авторлар `data/metadata/LICENSE`, `attributions.csv` ішінде сақталған.

Fixed split: train fold 1–3 (240), validation fold 4 (80), test fold 5 (80).
Seed 42. Checkpoint ең төмен validation cross-entropy бойынша сақталады.
Кандидат та осы validation бойынша таңдалады; test accuracy таңдау үшін
қолданылмайды. Әр аяқталған checkpoint үшін сақталған test болжамы қайта
қолданылады, оның SHA-256 тексеріледі. Бұрын белгілі fold-5 нәтижесі ашық көрсетіледі.

5-fold CV: test K, validation K−1 (1 үшін 5), қалған 3 fold train. Әр бастапқы
клиптің OOF болжамы бір рет қана кездеседі. Бастапқы source recording бір fold-та
сақталады. Train-ға арналған augmentation validation/test-ке қолданылмайды.
Scaler mean/std тек train partition-нан есептеледі; checkpoint ішінде сақталады.
Frozen embedding extraction label-дерді қолданбайды. YAMNet subprocess-іне
label немесе class index жіберілмейді; тек 16 кГц mono waveform мен файл атаулары.

Accuracy = дұрыс болжам / бастапқы клип саны. Macro-F1 — 10 класс F1 орташа мәні.
ROC әр классқа one-vs-rest есептеледі; macro ROC-AUC — класс AUC орташа мәні.
PR көрсеткіші — sklearn non-interpolated average precision, трапециялық PR-AUC
емес; macro AP — он AP орташа мәні. Қисықтарды көрсету үшін ғана interpolation
қолданылады. Confusion matrix жолы — нақты класс, бағаны — болжанған класс.
Fold standard deviation ddof=1; aggregate OOF F1 fold F1 орташа мәнінен өзгеше.

EfficientNet тек frozen backbone режимінде; head optimizer шағын batch-пен
оқытылады, backbone BatchNorm eval режимінде қалады. YAMNet те frozen, CPU
арқылы extraction; жаңа head PyTorch CUDA қолжетімді болса GPU-да оқиды.
CNN V2 train-only shift ±0.35 с, gain ±4 dB, SNR 25–35 dB және modest
frequency/time mask қолданады. Толық recipe `config.json` ішінде бекітілген.

Training time transfer head үшін backbone extraction уақытын қамтымайды;
`comparison.csv` ішінде extraction бөлек көрсетіледі. Prediction receipt-теріндегі
head timing толық audio decoding/backbone inference latency емес. App толық
inference уақыты бөлек end-to-end тексеру журналында өлшенеді.

Ресми implementation деректері:
https://www.tensorflow.org/tutorials/audio/transfer_learning_audio
https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html
'''
    (REPORT / 'methodology_kk.md').write_text(methodology, encoding='utf-8')


if __name__ == '__main__':
    generate()
