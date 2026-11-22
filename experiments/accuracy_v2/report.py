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



def generate(sync_results=True):
    for directory in ['plots','metrics','predictions','training_history']:
        (REPORT/directory).mkdir(parents=True,exist_ok=True)
    write_audit_manifests()
    labels=[r['label'] for r in read_json(ROOT/'artifacts/class_mapping.json')]
    all_metrics,rows,cv={ },[],{}
    paths=sorted((REPORT/'predictions').glob('yamnet_transfer_*.csv'))
    for path in paths:
        if path.stem.endswith('_misclassifications'):
            continue
        frame=pd.read_csv(path)
        count=400 if path.stem.endswith('_oof') else 80
        metrics=score_predictions(frame,labels,count)
        if count==80 and frame.fold.nunique()!=1:
            raise ValueError('Single-fold predictions must contain one official fold')
        metrics['predictions_sha256']=sha256(path)
        write_json(REPORT/'metrics'/f'{path.stem}.json',metrics)
        (REPORT/'metrics'/f'{path.stem}_classification_report.txt').write_text(
            classification_report(frame.true_index,frame.predicted_index,labels=list(range(10)),target_names=labels,digits=4,zero_division=0),encoding='utf-8')
        all_metrics[path.stem]=metrics
        if path.stem.endswith(('_fixed','_oof')):
            plot_predictions(path.stem,frame,labels,metrics)
            rows.append({'model':path.stem,'evaluation':'OOF' if count==400 else 'fixed test',
                         **{k:metrics[k] for k in ['correct','count','accuracy','macro_f1','macro_roc_auc_ovr','macro_average_precision']}})
        if count==400:
            source=pd.read_csv(REPORT/'split_manifest.csv')
            merged=frame.merge(source[['filename','fold','class_index']],on='filename',suffixes=('_prediction','_source'),validate='one_to_one')
            if len(merged)!=400 or not np.array_equal(merged.fold_prediction,merged.fold_source) or not np.array_equal(merged.true_index,merged.class_index):
                raise ValueError('OOF does not match original clips and labels')
            if set(frame.fold)!=set(range(1,6)) or not (frame.groupby('fold').size()==80).all():
                raise ValueError('Invalid OOF fold coverage')
            per_fold=[]
            for fold,group in frame.groupby('fold'):
                scores=score_predictions(group,labels,80)
                per_fold.append({'fold':int(fold),**{k:scores[k] for k in ['correct','count','accuracy','macro_f1']}})
            cv['yamnet_transfer']={'per_fold':per_fold,
                'accuracy_mean':float(np.mean([r['accuracy'] for r in per_fold])),
                'accuracy_std_sample':float(np.std([r['accuracy'] for r in per_fold],ddof=1)),
                'macro_f1_mean':float(np.mean([r['macro_f1'] for r in per_fold])),
                'macro_f1_std_sample':float(np.std([r['macro_f1'] for r in per_fold],ddof=1)),
                'aggregate_correct':metrics['correct'],'aggregate_count':400,'aggregate_macro_f1':metrics['macro_f1'],
                'single_seed':True,'std_definition':'sample standard deviation over five outer folds, ddof=1'}
            fig,ax=plt.subplots(figsize=(7,4))
            ax.plot([r['fold'] for r in per_fold],[r['accuracy'] for r in per_fold],'o-',label='Accuracy')
            ax.plot([r['fold'] for r in per_fold],[r['macro_f1'] for r in per_fold],'s-',label='Macro F1')
            ax.set(xticks=range(1,6),ylim=(0,1.03),xlabel='Outer test fold',title='YAMNet OOF: seed 42')
            ax.legend()
            save(fig,'yamnet_transfer_oof_fold_metrics')
    for path in sorted((REPORT/'training_history').glob('yamnet_transfer_*.csv')):
        history=pd.read_csv(path)
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for phase in ['train','validation']:
            axes[0].plot(history.epoch,history[f'{phase}_loss'],label=phase)
            axes[1].plot(history.epoch,history[f'{phase}_accuracy'],label=phase)
        axes[0].set(xlabel='Epoch',ylabel='Cross entropy',title=f'{path.stem}: loss')
        axes[1].set(xlabel='Epoch',ylabel='Accuracy',title=f'{path.stem}: accuracy',ylim=(0,1.03))
        for ax in axes:
            ax.legend()
        save(fig,f'{path.stem}_training')
    write_json(REPORT/'metrics.json',all_metrics)
    write_json(REPORT/'cross_validation_summary.json',cv)
    pd.DataFrame(rows).to_csv(REPORT/'yamnet_results.csv',index=False)
    lines=['# SoundSense YAMNet нәтижелері','',
           'Бұл есеп AudioSet-те алдын ала оқытылған frozen YAMNet және ESC-10 үшін жаңа head нәтижелерін сипаттайды. Қолданба YAMNet пен EfficientNet-B0 модельдерін қолдайды.',
           'MobileNetV1 терең CNN backbone 1024 embedding береді. Train-only mean/std → Dropout 0.2 → Linear 1024→10.',
           'Оқытылатын head параметрлері 10 250. Backbone қайта оқытылмаған.','',
           '| Бағалау | Дұрыс / барлығы | Accuracy | Macro F1 | ROC AUC | Macro AP |','|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["evaluation"]} | {row["correct"]}/{row["count"]} | {row["accuracy"]:.2%} | {row["macro_f1"]:.4f} | {row["macro_roc_auc_ovr"]:.4f} | {row["macro_average_precision"]:.4f} |')
    lines+=['','## Әдіснама','',
        'ESC-10: 400 бастапқы клип, он санат, ресми бес fold. Fixed train 1–3 (240), validation 4 (80), test 5 (80).',
        'Seed 42. AdamW lr=0.001 weight_decay=0.001 batch=16; max 160 epoch patience=25. Checkpoint ең төмен validation cross-entropy арқылы таңдалады.',
        'YAMNet frontend: 16 кГц mono; STFT 25 мс және 10 мс; 64 Mel жолағы 125–7500 Гц; log offset 0.001; патч 96×64.',
        'Бес секунд қысқа болса нөлмен толықтырылады, ұзын болса алғашқы бөлігі алынады. Frame embedding уақыт бойынша орташаланады.',
        'CV: test K, validation K−1 (1 үшін 5), қалған үш fold train. Әр клип бір OOF болжам алады. Fold 5 fixed checkpoint-ін қолданады.',
        'Scaler тек train арқылы fit жасалады. Backbone frozen, waveform augmentation және SpecAugment қолданылмаған.','',
        '## Графиктер','']
    plots=sorted((REPORT/'plots').glob('yamnet_transfer_*.png'))
    for path in plots:
        lines.append(f'- [{path.stem}](plots/{path.name})')
    lines+=['','## Шектеулер','',
        '400 клип және бір seed. Fixed test бұрын белгілі болған; бастапқы модель іріктеуі fixed validation арқылы жүргізілген. Бұл толық nested model selection емес.',
        'AudioSet пен ESC жазбаларының ықтимал қабаттасуы жеке тексерілмеген. Fold SD confidence interval емес.',
        'Белгісіз дыбыстар да он санаттың біріне жіктеледі. Softmax калибрленбеген. Сыртқы test, multi-label және оқиғаны уақыт бойынша локализациялау бағаланбаған.',
        'Барлық 400 клиппен соңғы deployment retraining жасалмаған. Бастапқы бес YAMNet checkpoint пен тарих сақталған.','',
        'Ресми YAMNet: https://www.tensorflow.org/tutorials/audio/transfer_learning_audio',
        'ESC-50 және ESC-10: https://github.com/karolpiczak/ESC-50']
    (REPORT/'report_kk.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (REPORT/'methodology_kk.md').write_text('\n'.join(lines[lines.index('## Әдіснама'):lines.index('## Графиктер')]).rstrip()+'\n',encoding='utf-8')
    if sync_results:
        target=ROOT/'results/figures'
        target.mkdir(parents=True,exist_ok=True)
        for path in plots:
            shutil.copy2(path,target/path.name)
        inventory=[{'path':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':sha256(p)} for p in sorted(target.glob('*.png'))]
        write_json(target/'inventory.json',inventory)
        write_json(ROOT/'results/summary.json',{'model':'yamnet_transfer','metrics':all_metrics,'cross_validation':cv})
        (ROOT/'docs/coursework_report_kk.md').write_text(
            (REPORT/'report_kk.md').read_text(encoding='utf-8').replace('](plots/','](../reports/accuracy_v2/plots/'),encoding='utf-8')
        shutil.copy2(REPORT/'methodology_kk.md',ROOT/'docs/methodology_kk.md')
    print(f'YAMNet report: {len(plots)} result plots, {len(all_metrics)} evaluations',flush=True)


if __name__=='__main__':
    generate()
