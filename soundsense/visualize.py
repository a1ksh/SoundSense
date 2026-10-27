"""300-dpi coursework figures, exclusively from actual audio and saved results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageStat

from .common import ROOT, read_json, sha256, write_json
from .preprocess import load_audio, logmel

FIGURES = ROOT / 'results/figures'
SOURCE = ROOT / 'results/figure_data'
PALETTE = ['#266b8f', '#d46b45']


def pretty(label):
    return label.replace('_', ' ')


def save(fig, filename, title=None):
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.tight_layout(pad=1.8)
    fig.savefig(FIGURES / filename, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def generate():
    FIGURES.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.15,
                  rc={'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 110})
    mapping = read_json(ROOT / 'artifacts/class_mapping.json')
    labels = [v['label'] for v in mapping]
    names = [pretty(v) for v in labels]
    manifest = pd.read_csv(ROOT / 'data/metadata/manifest.csv')
    stats = read_json(ROOT / 'artifacts/preprocessing_stats.json')
    cfg = stats['preprocessing']
    summary = read_json(ROOT / 'results/summary.json')
    histories = {name: pd.read_csv(ROOT / f'results/history/{name}.csv') for name in ['mlp', 'cnn']}
    predictions = {name: pd.read_csv(ROOT / f'results/predictions/{name}_test.csv') for name in ['mlp', 'cnn']}

    distribution = pd.crosstab(manifest.category, manifest.split).reindex(labels)[['train', 'validation', 'test']]
    distribution.to_csv(SOURCE / 'class_distribution.csv')
    fig, ax = plt.subplots(figsize=(11, 5))
    distribution.set_axis(names).plot.bar(ax=ax, color=['#266b8f', '#b4c7a2', '#d46b45'], width=.78)
    ax.set(xlabel='Sound category', ylabel='Number of clips', title='ESC-10 · original-fold split (240 / 80 / 80 clips)')
    ax.tick_params(axis='x', rotation=35)
    ax.legend(title='Split', frameon=False, loc='center left', bbox_to_anchor=(1.01, .5))
    save(fig, 'class_distribution.png')

    examples = [manifest[(manifest.category == label) & (manifest.split == 'train')].iloc[0] for label in labels]
    pd.DataFrame(examples).to_csv(SOURCE / 'training_example_manifest.csv', index=False)
    for mode in ['waveform', 'mel']:
        fig, axes = plt.subplots(5, 2, figsize=(13, 14))
        for ax, row in zip(axes.flat, examples):
            y = load_audio(ROOT / 'data/raw' / row.filename, cfg)
            if mode == 'waveform':
                # Render every sample and retain the exact plotted source values.
                ax.plot(np.arange(len(y)) / cfg['sample_rate'], y, lw=.45, color=PALETTE[0])
                pd.DataFrame({'time_seconds': np.arange(len(y)) / cfg['sample_rate'], 'amplitude': y}).to_csv(
                    SOURCE / f'example_{row.category}_waveform.csv', index=False)
                ax.set(ylabel='Amplitude', xlabel='Time (s)', xlim=(0, 5))
            else:
                mel = logmel(y, cfg)
                im = ax.imshow(mel, origin='lower', aspect='auto', extent=(0, 5, 0, 64), cmap='magma')
                fig.colorbar(im, ax=ax, label='dB', pad=.02)
                pd.DataFrame(mel).to_csv(SOURCE / f'example_{row.category}_mel.csv', index=False)
                ax.set(xlabel='Time (s)', ylabel='Mel band')
                ax.grid(False)
            ax.set_title(f'{pretty(row.category)} · {row.filename}', fontsize=10)
        save(fig, f'{"audio_waveform" if mode == "waveform" else "mel_spectrogram"}_examples.png',
             'Actual training-fold audio · one example per ESC-10 category')

    for name, history in histories.items():
        for metric, ylabel in [('loss', 'Cross-entropy loss'), ('accuracy', 'Accuracy')]:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            ax.plot(history.epoch, history[f'train_{metric}'], label='Training', color=PALETTE[0], lw=2)
            ax.plot(history.epoch, history[f'validation_{metric}'], label='Validation (fold 4)', color=PALETTE[1], lw=2)
            epoch = summary['models'][name]['training']['best_epoch']
            ax.axvline(epoch, color='#666666', ls='--', lw=1, label=f'Best validation loss: epoch {epoch}')
            ax.set(xlabel='Epoch', ylabel=ylabel, title=f'{name.upper()} · {ylabel.lower()} (seed 42)')
            if metric == 'accuracy':
                ax.set_ylim(0, 1)
            ax.legend(frameon=False)
            save(fig, f'{name}_{metric}_curve.png')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for color, (name, history) in zip(PALETTE, histories.items()):
        for ax, metric in zip(axes, ['loss', 'accuracy']):
            ax.plot(history.epoch, history[f'validation_{metric}'], color=color, label=f'{name.upper()} validation', lw=2)
            ax.plot(history.epoch, history[f'train_{metric}'], color=color, ls=':', alpha=.65, label=f'{name.upper()} training')
            ax.set(xlabel='Epoch', ylabel='Cross-entropy' if metric == 'loss' else 'Accuracy')
            ax.legend(frameon=False, fontsize=9)
    axes[1].set_ylim(0, 1)
    save(fig, 'models_training_comparison.png', 'Measured learning histories · identical fold split')

    for name in ['mlp', 'cnn']:
        for normalized in [False, True]:
            suffix = '_normalized' if normalized else ''
            cm = pd.read_csv(ROOT / f'results/metrics/{name}_confusion_matrix{suffix}.csv', index_col=0)
            fig, ax = plt.subplots(figsize=(10, 8.4))
            sns.heatmap(cm.to_numpy(), ax=ax, annot=True, fmt='.2f' if normalized else 'd', cmap='Blues',
                        vmin=0, vmax=1 if normalized else None, xticklabels=names, yticklabels=names,
                        cbar_kws={'label': 'Fraction within true class' if normalized else 'Number of clips'},
                        square=True, linewidths=.4, annot_kws={'fontsize': 9})
            ax.set(xlabel='Predicted class', ylabel='True class', title=f'{name.upper()} · {"row-normalized " if normalized else ""}confusion matrix · fold 5 (n=80)')
            plt.setp(ax.get_xticklabels(), rotation=40, ha='right')
            plt.setp(ax.get_yticklabels(), rotation=0)
            save(fig, f'{name}_confusion_matrix{suffix}.png')

    per_class = {name: pd.read_csv(ROOT / f'results/metrics/{name}_per_class.csv').set_index('class').reindex(labels)
                 for name in ['mlp', 'cnn']}
    f1 = pd.DataFrame({name.upper(): frame['f1-score'] for name, frame in per_class.items()})
    f1.to_csv(SOURCE / 'per_class_f1.csv')
    fig, ax = plt.subplots(figsize=(11, 5))
    f1.set_axis(names).plot.bar(ax=ax, color=PALETTE, width=.8)
    ax.set(xlabel='True category (eight test clips each)', ylabel='F1 score', ylim=(0, 1), title='Per-class F1 · held-out fold 5')
    ax.tick_params(axis='x', rotation=35)
    ax.legend(frameon=False)
    save(fig, 'per_class_f1_comparison.png')

    colors = plt.get_cmap('tab10').colors
    for name in ['mlp', 'cnn']:
        ranking = summary['models'][name]['metrics']['per_class_ranking']
        for kind in ['roc', 'pr']:
            fig, ax = plt.subplots(figsize=(11, 7))
            for color, row in zip(colors, ranking):
                curve = pd.read_csv(ROOT / f'results/metrics/{name}_curves/{row["class"]}_{kind}.csv')
                if kind == 'roc':
                    ax.plot(curve.fpr, curve.tpr, color=color, lw=1.7,
                            label=f'{pretty(row["class"])} (AUC {row["roc_auc"]:.3f})')
                else:
                    ax.step(curve.recall, curve.precision, where='post', color=color, lw=1.7,
                            label=f'{pretty(row["class"])} (AP {row["average_precision"]:.3f})')
            if kind == 'roc':
                ax.plot([0, 1], [0, 1], '--', color='#777777', label='Chance ranking')
                ax.set(xlabel='False positive rate', ylabel='True positive rate')
                filename = 'roc_curves_per_class.png'
            else:
                ax.axhline(.1, ls='--', color='#777777', label='Prevalence = 0.10')
                ax.set(xlabel='Recall', ylabel='Precision')
                filename = 'precision_recall_per_class.png'
            ax.set(xlim=(0, 1), ylim=(0, 1.03), title=f'{name.upper()} · one-vs-rest {"ROC" if kind == "roc" else "precision–recall"} · fold 5')
            ax.legend(loc='center left', bbox_to_anchor=(1.01, .5), frameon=False, fontsize=10)
            save(fig, filename if name == 'cnn' else f'mlp_{filename}')

    metrics = ['accuracy', 'balanced_accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'macro_roc_auc_ovr', 'macro_average_precision']
    compare = pd.DataFrame({name.upper(): {metric: summary['models'][name]['metrics'][metric] for metric in metrics}
                            for name in ['mlp', 'cnn']})
    compare.to_csv(SOURCE / 'model_metrics_comparison.csv')
    fig, ax = plt.subplots(figsize=(12, 5))
    compare.set_axis(['Accuracy', 'Balanced\naccuracy', 'Macro\nprecision', 'Macro\nrecall', 'Macro F1', 'Macro\nROC-AUC', 'Macro AP']).plot.bar(ax=ax, color=PALETTE, rot=0)
    ax.set(ylabel='Score', ylim=(0, 1.12), title='Frozen-checkpoint metrics · fold 5, single seed')
    ax.legend(frameon=False)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)
    save(fig, 'model_metrics_comparison.png')

    cnn = predictions['cnn']
    prob_columns = [f'prob_{label}' for label in labels]
    confidence = cnn[prob_columns].max(axis=1).to_numpy()
    correct = (cnn.true_index == cnn.predicted_index).to_numpy()
    pd.DataFrame({'filename': cnn.filename, 'max_softmax': confidence, 'correct': correct}).to_csv(SOURCE / 'cnn_prediction_confidence.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    for mask, label, color in [(correct, 'Correct', PALETTE[0]), (~correct, 'Incorrect', PALETTE[1])]:
        ax.hist(confidence[mask], bins=np.linspace(0, 1, 16), alpha=.7, color=color, label=f'{label} (n={mask.sum()})')
    ax.set(xlabel='Maximum softmax probability (uncalibrated)', ylabel='Number of clips',
           title='CNN prediction scores · softmax is not calibrated confidence', xlim=(0, 1))
    ax.legend(frameon=False)
    save(fig, 'cnn_prediction_confidence.png')

    errors = cnn[~correct].head(6)
    errors.to_csv(SOURCE / 'error_analysis_examples.csv', index=False)
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for ax in axes.flat:
        ax.axis('off')
    if errors.empty:
        axes[1, 0].text(.5, .5, 'No misclassifications in the 80 held-out clips.\nThis does not establish general performance.', ha='center', transform=axes[1, 0].transAxes)
    for ax, row in zip(axes.flat, errors.itertuples()):
        ax.axis('on')
        y = load_audio(ROOT / 'data/raw' / row.filename, cfg)
        mel = logmel(y, cfg)
        pd.DataFrame(mel).to_csv(SOURCE / f'error_{row.filename}.csv', index=False)
        im = ax.imshow(mel, origin='lower', aspect='auto', extent=(0, 5, 0, 64), cmap='magma')
        fig.colorbar(im, ax=ax, label='dB', pad=.02)
        ax.set(xlabel='Time (s)', ylabel='Mel band', title=f'True: {pretty(row.true_label)} → predicted: {pretty(row.predicted_label)}\n{row.filename} · p={getattr(row, "prob_" + row.predicted_label):.3f}')
        ax.grid(False)
    save(fig, 'error_analysis_examples.png', 'CNN errors · first six misclassified test clips in manifest order')

    inventory = []
    for path in sorted(FIGURES.glob('*.png')):
        with Image.open(path) as image:
            if min(image.size) < 600 or max(ImageStat.Stat(image.convert('RGB')).stddev) < 5:
                raise ValueError(f'Blank or undersized figure: {path}')
            dpi = image.info.get('dpi', (0, 0))
            if min(dpi) < 299:
                raise ValueError(f'Figure lacks 300 dpi: {path}')
            inventory.append({'file': str(path.relative_to(ROOT)).replace('\\', '/'), 'width': image.width,
                              'height': image.height, 'dpi': list(dpi), 'sha256': sha256(path)})
    write_json(FIGURES / 'inventory.json', inventory)
    return inventory


if __name__ == '__main__':
    generate()
