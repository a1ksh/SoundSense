"""Run real fixed-split training, one frozen test evaluation, then official CV.

PowerShell: python -m experiments.accuracy_v2.run --yamnet-python PATH
Use --models cnn_v2 efficientnet_b0_mel if TensorFlow is unavailable.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from soundsense.common import ROOT, config, identity, read_json, seed_everything, sha256, write_json
from soundsense.preprocess import load_audio, logmel
from .core import (CACHE, MODELS, REPORT, RECIPE_PATH, YAMNET_URL, build_candidate,
                   cache_environment, cache_key, efficientnet_backbone,
                   efficientnet_input, load_candidate, manifest, raw_mels, split_masks)


def prepare_embeddings(df, name, mels, device, yamnet_python=None):
    start = time.perf_counter()
    recipe = read_json(RECIPE_PATH)[name]
    key = cache_key(df, name, recipe)
    path = CACHE / f'{name}_embeddings.npz'
    info = path.with_suffix('.json')
    if path.exists() and info.exists() and read_json(info)['key'] == key and read_json(info)['sha256'] == sha256(path):
        with np.load(path, allow_pickle=False) as z:
            if z['filenames'].tolist() != df.filename.tolist():
                raise ValueError('Embedding cache order mismatch')
            return z['embeddings'].copy(), read_json(info)
    if name == 'efficientnet_b0_mel':
        backbone = efficientnet_backbone(pretrained=True).to(device)
        backbone_path = MODELS / 'efficientnet_backbone.pt'
        torch.save(backbone.cpu().state_dict(), backbone_path)
        backbone.to(device)
        values = []
        with torch.inference_mode():
            for i in range(0, len(mels), 8):
                x = efficientnet_input(mels[i:i+8]).to(device)
                values.append(backbone(x).cpu().numpy())
                if (i + 8) % 80 == 0:
                    print(f'EfficientNet embeddings {i+8}/{len(mels)}', flush=True)
        embeddings = np.concatenate(values).astype(np.float32)
        np.savez_compressed(path, embeddings=embeddings, filenames=df.filename.to_numpy(dtype=str), key=np.asarray(key))
        provenance = {'backbone_sha256': sha256(backbone_path), 'pretrained': 'IMAGENET1K_V1',
                      'weights_url': 'https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth'}
        del backbone
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    else:
        if not yamnet_python:
            raise RuntimeError('YAMNet extraction needs --yamnet-python pointing to isolated TensorFlow runtime')
        wave_path = CACHE / 'waveforms_16k.npy'
        cfg = config()['preprocessing']
        waves = np.stack([load_audio(ROOT / 'data/raw' / filename, cfg) for filename in df.filename])
        np.save(wave_path, waves, allow_pickle=False)
        request = CACHE / 'yamnet_request.json'
        write_json(request, {'waveforms': str(wave_path), 'filenames': df.filename.tolist(), 'key': key, 'model_url': YAMNET_URL})
        command = [str(yamnet_python), str(Path(__file__).with_name('extract_yamnet.py')), '--request', str(request), '--output', str(path)]
        subprocess.run(command, check=True, timeout=1200, env=os.environ.copy())
        with np.load(path, allow_pickle=False) as z:
            embeddings = z['embeddings'].copy()
        provenance = {'pretrained': YAMNET_URL, 'pooling': 'mean over frames; 1024 features',
                      'extraction': 'CPU frozen model, no labels passed to subprocess'}
    if embeddings.shape[0] != 400 or not np.isfinite(embeddings).all():
        raise ValueError('Nonfinite or incomplete frozen embeddings')
    provenance.update({'key': key, 'sha256': sha256(path), 'seconds': time.perf_counter() - start,
                       'dimension': int(embeddings.shape[1]), 'labels_used': False})
    write_json(info, provenance)
    return embeddings, provenance


def augmented_train_mels(df, train_mask, recipe, mels):
    subset = df.loc[train_mask]
    key = cache_key(subset, 'cnn_v2_wave_augmentation', recipe)
    path = CACHE / f'augmented_{key[:16]}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            return z['variants'].copy()
    rng = np.random.default_rng(read_json(RECIPE_PATH)['seed'])
    cfg = config()['preprocessing']
    variants = [mels[train_mask]]
    for variant in range(recipe['waveform_variants']):
        result = []
        for row in subset.itertuples():
            y = load_audio(ROOT / 'data/raw' / row.filename, cfg)
            shift = int(rng.uniform(-recipe['shift_seconds'], recipe['shift_seconds']) * cfg['sample_rate'])
            # Shift with zero fill; no artificial circular splice.
            if shift > 0:
                y = np.pad(y[:-shift], (shift, 0))
            elif shift < 0:
                y = np.pad(y[-shift:], (0, -shift))
            y = y * 10 ** (rng.uniform(-recipe['gain_db'], recipe['gain_db']) / 20)
            snr = rng.uniform(*recipe['noise_snr_db'])
            noise = rng.normal(0, max(float(np.sqrt(np.mean(y*y))), 1e-8) / 10 ** (snr / 20), len(y))
            result.append(logmel((y + noise).astype(np.float32), cfg))
        variants.append(np.stack(result))
        print(f'CNN train-only waveform variant {variant+1} complete', flush=True)
    values = np.stack(variants).astype(np.float32)
    np.savez_compressed(path, variants=values)
    return values


def specaugment(x, recipe):
    x = x.clone()
    for i in range(len(x)):
        fw = int(torch.randint(0, recipe['frequency_mask'] + 1, ()).item())
        tw = int(torch.randint(0, recipe['time_mask'] + 1, ()).item())
        fs = int(torch.randint(0, x.shape[-2] - fw + 1, ()).item())
        ts = int(torch.randint(0, x.shape[-1] - tw + 1, ()).item())
        x[i, :, fs:fs+fw, :] = 0
        x[i, :, :, ts:ts+tw] = 0
    return x


def train_candidate(name, features, df, test_fold, device, mels, provenance):
    recipes = read_json(RECIPE_PATH)
    recipe = recipes[name]
    train, val, test = split_masks(df, test_fold)
    run_id = f'{name}_fixed' if test_fold == 5 else f'{name}_fold{test_fold}'
    path = MODELS / f'{run_id}.pt'
    history_path = REPORT / 'training_history' / f'{run_id}.csv'
    meta_path = history_path.with_suffix('.json')
    code = {p.name: hashlib.sha256(p.read_text(encoding='utf-8').encode('utf-8')).hexdigest()
            for p in Path(__file__).parent.glob('*.py') if p.name in ('run.py', 'core.py', 'extract_yamnet.py')}
    stable_provenance = {key: value for key, value in provenance.items() if key != 'seconds'}
    signature = identity({'recipe': recipe, 'seed': recipes['seed'], 'test_fold': test_fold,
                          'files': df[['filename', 'sha256']].to_dict('records'), 'code': code,
                          'feature_provenance': stable_provenance, 'torch': str(torch.__version__)})
    if path.exists() and meta_path.exists():
        saved = read_json(meta_path)
        if saved['signature'] != signature:
            raise ValueError(f'{run_id}: completed artifact recipe/code differs; archive previous run instead of overwriting')
        if sha256(path) != saved['checkpoint_sha256'] or sha256(history_path) != saved['history_sha256']:
            raise ValueError(f'{run_id}: artifact checksum mismatch')
        print(f'Reusing completed {run_id}', flush=True)
        return saved
    seed_everything(recipes['seed'])
    dimension = None if name == 'cnn_v2' else int(features.shape[1])
    model = build_candidate(name, dimension).to(device)
    mean = features[train].mean(axis=0, dtype=np.float64) if dimension else float(features[train].mean(dtype=np.float64))
    std = np.maximum(features[train].std(axis=0, dtype=np.float64), 1e-6) if dimension else max(float(features[train].std(dtype=np.float64)), 1e-6)
    normalized = ((features - mean) / std).astype(np.float32)
    variants = None
    if name == 'cnn_v2':
        variants = ((augmented_train_mels(df, train, recipe, mels) - mean) / std).astype(np.float32)[:, :, None]
        normalized = normalized[:, None]
    labels = torch.tensor(df.class_index.to_numpy(), dtype=torch.long)
    xt = torch.from_numpy(normalized[train])
    yt = labels[train]
    vl = DataLoader(TensorDataset(torch.from_numpy(normalized[val]), labels[val]), batch_size=16, shuffle=False)
    generator = torch.Generator().manual_seed(recipes['seed'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['learning_rate'], weight_decay=recipe['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.5)
    criterion = nn.CrossEntropyLoss()
    best, stale, history = float('inf'), 0, []
    start = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, recipe['epochs'] + 1):
        if variants is not None:
            selected = np.random.randint(len(variants), size=len(xt))
            xt = torch.from_numpy(variants[selected, np.arange(len(xt))])
        loader = DataLoader(TensorDataset(xt, yt), batch_size=recipe['batch_size'], shuffle=True, generator=generator)
        row = {'epoch': epoch, 'learning_rate': optimizer.param_groups[0]['lr']}
        for phase, batches in [('train', loader), ('validation', vl)]:
            training = phase == 'train'
            model.train(training)
            loss_sum, correct, total = 0., 0, 0
            with torch.set_grad_enabled(training):
                for x, y in batches:
                    x, y = x.to(device), y.to(device)
                    if training and variants is not None:
                        x = specaugment(x, recipe)
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = criterion(logits, y)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss')
                    if training:
                        loss.backward()
                        optimizer.step()
                    loss_sum += float(loss.detach()) * len(y)
                    correct += int((logits.argmax(1) == y).sum())
                    total += len(y)
            row[f'{phase}_loss'] = loss_sum / total
            row[f'{phase}_accuracy'] = correct / total
        history.append(row)
        scheduler.step(row['validation_loss'])
        if row['validation_loss'] < best:
            best, stale = row['validation_loss'], 0
            payload = {'schema': 'accuracy_v2_1', 'name': name, 'state_dict': model.state_dict(),
                       'dimension': dimension, 'mean': np.asarray(mean).tolist(), 'std': np.asarray(std).tolist(),
                       'preprocessing': config()['preprocessing'], 'class_mapping': read_json(ROOT / 'artifacts/class_mapping.json'),
                       'seed': recipes['seed'], 'epoch': epoch, 'validation_loss': best,
                       'validation_accuracy': row['validation_accuracy'], 'test_fold': test_fold,
                       'train_folds': sorted(df.loc[train, 'fold'].unique().tolist()),
                       'validation_fold': test_fold - 1 or 5, 'fit_count': 240, 'signature': signature,
                       'recipe': recipe, 'provenance': provenance, 'deployment_trained_all_data': False}
            torch.save(payload, path)
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f'{run_id} epoch {epoch}: train {row["train_accuracy"]:.3f}, validation {row["validation_accuracy"]:.3f}, val loss {row["validation_loss"]:.4f}', flush=True)
        if stale >= recipe['patience']:
            break
    pd.DataFrame(history).to_csv(history_path, index=False)
    _, checkpoint = load_candidate(path)
    meta = {'name': name, 'run_id': run_id, 'signature': signature, 'checkpoint': str(path.relative_to(ROOT)),
            'checkpoint_sha256': sha256(path), 'history_sha256': sha256(history_path), 'epochs_completed': len(history),
            'best_epoch': checkpoint['epoch'], 'validation_loss': best, 'validation_accuracy': checkpoint['validation_accuracy'],
            'training_seconds': time.perf_counter() - start, 'feature_extraction_seconds': provenance.get('seconds', 0),
            'parameter_count': sum(p.numel() for p in model.parameters()), 'model_bytes': path.stat().st_size,
            'device': str(device), 'peak_gpu_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
            'test_fold': test_fold, 'complete': True}
    write_json(meta_path, meta)
    return meta


def evaluate_candidate(meta, features, df, device):
    run_id = meta['run_id']
    path = REPORT / 'predictions' / f'{run_id}.csv'
    receipt = path.with_suffix('.json')
    if path.exists() and receipt.exists():
        saved = read_json(receipt)
        if saved['checkpoint_sha256'] != meta['checkpoint_sha256'] or saved['csv_sha256'] != sha256(path):
            raise ValueError('Frozen prediction provenance mismatch')
        return pd.read_csv(path)
    model, payload = load_candidate(ROOT / meta['checkpoint'], device)
    _, _, test = split_masks(df, payload['test_fold'])
    x = ((features[test] - np.asarray(payload['mean'])) / np.asarray(payload['std'])).astype(np.float32)
    if payload['name'] == 'cnn_v2':
        x = x[:, None]
    probabilities = []
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in torch.from_numpy(x).split(16):
            probabilities.append(model(batch.to(device)).softmax(1).cpu().numpy())
    elapsed = time.perf_counter() - start
    p = np.concatenate(probabilities)
    result = df.loc[test, ['filename', 'fold', 'src_file', 'class_index', 'category']].rename(columns={'class_index': 'true_index', 'category': 'true_label'}).reset_index(drop=True)
    labels = [r['label'] for r in payload['class_mapping']]
    result['predicted_index'] = p.argmax(1)
    result['predicted_label'] = [labels[i] for i in p.argmax(1)]
    result['predicted_probability'] = p.max(1)
    for i, label in enumerate(labels):
        result[f'prob_{label}'] = p[:, i]
    result.to_csv(path, index=False)
    write_json(receipt, {'checkpoint_sha256': meta['checkpoint_sha256'], 'csv_sha256': sha256(path),
                         'test_fold': payload['test_fold'], 'count': len(result), 'probability_order': labels,
                         'head_or_cnn_seconds_per_clip': elapsed / len(result),
                         'timing_excludes_frozen_feature_extraction': payload['name'] != 'cnn_v2',
                         'exploratory': True, 'evaluation_calls': 1})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', choices=['cnn_v2', 'efficientnet_b0_mel', 'yamnet_transfer'], default=['cnn_v2', 'efficientnet_b0_mel', 'yamnet_transfer'])
    parser.add_argument('--yamnet-python', type=Path)
    parser.add_argument('--cv-top', type=int, default=2)
    parser.add_argument('--train-only', action='store_true', help='Prepare and train on folds 1-3; do not select or evaluate test clips yet')
    args = parser.parse_args()
    cache_environment()
    for directory in [MODELS, REPORT / 'training_history', REPORT / 'predictions', REPORT / 'plots']:
        directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    seed_everything(42)
    df = manifest()
    df.to_csv(REPORT / 'split_manifest.csv', index=False)
    shutil.copy2(RECIPE_PATH, REPORT / 'config.json')
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    write_json(REPORT / 'environment.json', {'python': sys.version, 'executable': sys.executable, 'packages': packages,
               'torch_cuda': torch.cuda.is_available(), 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
               'command': subprocess.list2cmdline([sys.executable] + sys.argv), 'seed': 42,
               'config_sha256': sha256(RECIPE_PATH), 'source_manifest_sha256': sha256(ROOT / 'data/metadata/manifest.csv')})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mels = raw_mels(df)
    feature_sets, provenance, trained, blockers = {}, {}, [], {}
    # All fixed-split training and validation completes BEFORE any new test access.
    for name in args.models:
        try:
            if name == 'cnn_v2':
                feature_sets[name], provenance[name] = mels, {'pretrained': False, 'feature_digest': hashlib.sha256(mels.tobytes()).hexdigest()}
            else:
                feature_sets[name], provenance[name] = prepare_embeddings(df, name, mels, device, args.yamnet_python)
            trained.append(train_candidate(name, feature_sets[name], df, 5, device, mels, provenance[name]))
        except Exception:
            blockers[name] = traceback.format_exc()
            print(blockers[name], flush=True)
            write_json(REPORT / 'blockers.json', blockers)
    if not trained:
        raise RuntimeError('No candidate trained successfully; see blockers.json')
    if args.train_only:
        write_json(REPORT / 'training_stage.json', {'trained': trained, 'blockers': blockers, 'test_evaluated': False})
        return
    ranked = sorted(trained, key=lambda row: row['validation_loss'])
    winner = ranked[0]
    cv_names = [row['name'] for row in ranked[:max(0, args.cv_top)]]
    selection = {'name': winner['name'], 'checkpoint': winner['checkpoint'], 'checkpoint_sha256': winner['checkpoint_sha256'],
                 'rule': 'minimum fixed-split fold-4 validation loss, frozen before new fold-5 evaluation and CV',
                 'validation_ranking': [{'name': r['name'], 'validation_loss': r['validation_loss'], 'validation_accuracy': r['validation_accuracy']} for r in ranked],
                 'cv_candidates_frozen_before_outer_evaluation': cv_names, 'deployment_trained_all_data': False,
                 'caveat': 'ESC-10 source overlap with external AudioSet/ImageNet pretraining has not been audited'}
    selection_path = REPORT / 'selection.json'
    if selection_path.exists() and read_json(selection_path) != selection:
        raise ValueError('Selection was already frozen; do not change candidates after test access')
    write_json(selection_path, selection)
    shutil.copy2(ROOT / winner['checkpoint'], MODELS / 'best_v2.pt')
    write_json(MODELS / 'best_v2.json', selection)
    for meta in trained:
        evaluate_candidate(meta, feature_sets[meta['name']], df, device)
    for name in cv_names:
        oof = []
        for fold in range(1, 6):
            meta = next(r for r in trained if r['name'] == name) if fold == 5 else train_candidate(name, feature_sets[name], df, fold, device, mels, provenance[name])
            oof.append(evaluate_candidate(meta, feature_sets[name], df, device))
        predictions = pd.concat(oof, ignore_index=True)
        if len(predictions) != 400 or not predictions.filename.is_unique or set(predictions.filename) != set(df.filename):
            raise ValueError('Expected exactly one OOF prediction per original clip')
        predictions.to_csv(REPORT / 'predictions' / f'{name}_oof.csv', index=False)
    write_json(REPORT / 'run_status.json', {'trained_candidates': [r['name'] for r in trained], 'blockers': blockers,
                                          'cv_candidates': cv_names, 'selected': selection['name'], 'training_and_evaluation_complete': True})
    from .report import generate
    generate()


if __name__ == '__main__':
    main()
