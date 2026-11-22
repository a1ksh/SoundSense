"""Run real fixed-split training, one frozen test evaluation, then official CV.

PowerShell: python -m experiments.accuracy_v2.run --yamnet-python PATH
Only YAMNet is supported. Saved checkpoints are verified unless --retrain is explicit.
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from soundsense.common import ROOT, config, identity, read_json, seed_everything, sha256, write_json, code_identity
from soundsense.preprocess import load_audio
from .core import (CACHE, MODELS, REPORT, RECIPE_PATH, YAMNET_URL, MODEL_NAME,
                   build_candidate, cache_environment, cache_key, load_candidate, manifest, split_masks)


def prepare_embeddings(df, yamnet_python=None):
    name = MODEL_NAME
    start = time.perf_counter()
    recipe = read_json(RECIPE_PATH)[name]
    key = cache_key(df, name, recipe)
    # New training never replaces the signed deployment embedding cache.
    directory = CACHE / 'retraining' / key
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{name}_embeddings.npz'
    info = path.with_suffix('.json')
    if path.exists() and info.exists() and read_json(info)['key'] == key:
        if read_json(info)['sha256'] != sha256(path):
            raise ValueError('YAMNet embedding cache checksum mismatch')
        with np.load(path, allow_pickle=False) as z:
            if z['filenames'].tolist() != df.filename.tolist() or z['key'].item() != key:
                raise ValueError('Embedding cache order mismatch')
            values = z['embeddings'].copy()
        if values.shape != (len(df), 1024) or not np.isfinite(values).all():
            raise ValueError('Invalid cached YAMNet embeddings')
        return values, read_json(info)
    if not yamnet_python:
        raise RuntimeError('YAMNet extraction needs --yamnet-python pointing to isolated TensorFlow runtime')
    wave_path = directory / 'waveforms_16k.npy'
    cfg = config()['preprocessing']
    waves = np.stack([load_audio(ROOT / 'data/raw' / filename, cfg) for filename in df.filename])
    np.save(wave_path, waves, allow_pickle=False)
    request = directory / 'yamnet_request.json'
    write_json(request, {'waveforms': str(wave_path), 'filenames': df.filename.tolist(), 'key': key, 'model_url': YAMNET_URL})
    command = [str(yamnet_python), str(Path(__file__).with_name('extract_yamnet.py')), '--request', str(request), '--output', str(path)]
    subprocess.run(command, check=True, timeout=1200, env=os.environ.copy())
    with np.load(path, allow_pickle=False) as z:
        embeddings = z['embeddings'].copy()
    provenance = {'pretrained': YAMNET_URL, 'pooling': 'mean over frames; 1024 features',
                  'extraction': 'CPU frozen model, no labels passed to subprocess'}
    if embeddings.shape != (400, 1024) or not np.isfinite(embeddings).all():
        raise ValueError('Nonfinite or incomplete frozen embeddings')
    provenance.update({'key': key, 'sha256': sha256(path), 'seconds': time.perf_counter() - start,
                       'dimension': int(embeddings.shape[1]), 'labels_used': False})
    write_json(info, provenance)
    return embeddings, provenance


def train_candidate(features, df, test_fold, device, provenance):
    name = MODEL_NAME
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
                          'head_and_waveform_code': code_identity(['models.py', 'preprocess.py']),
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
    dimension = 1024
    model = build_candidate(name, dimension).to(device)
    mean = features[train].mean(axis=0, dtype=np.float64)
    std = np.maximum(features[train].std(axis=0, dtype=np.float64), 1e-6)
    normalized = ((features - mean) / std).astype(np.float32)
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
        loader = DataLoader(TensorDataset(xt, yt), batch_size=recipe['batch_size'], shuffle=True, generator=generator)
        row = {'epoch': epoch, 'learning_rate': optimizer.param_groups[0]['lr']}
        for phase, batches in [('train', loader), ('validation', vl)]:
            training = phase == 'train'
            model.train(training)
            loss_sum, correct, total = 0., 0, 0
            with torch.set_grad_enabled(training):
                for x, y in batches:
                    x, y = x.to(device), y.to(device)
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
                         'head_seconds_per_clip': elapsed / len(result),
                         'timing_excludes_frozen_feature_extraction': True,
                         'exploratory': True, 'evaluation_calls': 1})
    return result



def verify_saved_runs():
    """Reuse the signed historical experiment without pretending code is unchanged."""
    result=[]
    for fold in range(1,6):
        run_id=f'{MODEL_NAME}_fixed' if fold==5 else f'{MODEL_NAME}_fold{fold}'
        meta=read_json(REPORT/'training_history'/f'{run_id}.json')
        path=ROOT/meta['checkpoint']
        history=REPORT/'training_history'/f'{run_id}.csv'
        if sha256(path)!=meta['checkpoint_sha256'] or sha256(history)!=meta['history_sha256']:
            raise ValueError(f'Historical YAMNet artifacts changed: {run_id}')
        _,payload=load_candidate(path)
        if payload['test_fold']!=fold:
            raise ValueError('Wrong historical test fold')
        result.append(meta)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yamnet-python',type=Path,default=Path(os.environ.get('SOUNDSENSE_YAMNET_PYTHON',ROOT/'.runtime/yamnet311/python.exe')))
    parser.add_argument('--retrain',action='store_true',help='Train a new experiment in a separate empty output directory')
    parser.add_argument('--output-dir',type=Path,help='New directory inside this project; required for --retrain')
    parser.add_argument('--train-only',action='store_true')
    args=parser.parse_args(argv)
    global MODELS, REPORT
    if not args.retrain:
        if args.output_dir or args.train_only:
            parser.error('--output-dir and --train-only require --retrain')
        verify_saved_runs()
        from .report import generate
        generate()
        print('Verified and reused five saved YAMNet heads; no retraining',flush=True)
        return
    if args.output_dir is None:
        parser.error('--retrain requires --output-dir to preserve the deployed model')
    output=args.output_dir.resolve()
    if not output.is_relative_to(ROOT) or output==ROOT or (output.exists() and any(output.iterdir())):
        parser.error('--output-dir must be a new or empty directory inside this project')
    MODELS,REPORT=output/'models',output/'reports'
    cache_environment()
    for directory in [MODELS,REPORT/'training_history',REPORT/'predictions',REPORT/'plots']:
        directory.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    seed_everything(42)
    df=manifest()
    df.to_csv(REPORT/'split_manifest.csv',index=False)
    shutil.copy2(RECIPE_PATH,REPORT/'config.json')
    write_json(REPORT/'environment.json',{'python':sys.version,'torch':str(torch.__version__),
               'config_sha256':sha256(RECIPE_PATH),'seed':42,'new_training':True})
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features,provenance=prepare_embeddings(df,args.yamnet_python)
    fixed=train_candidate(features,df,5,device,provenance)
    if args.train_only:
        write_json(REPORT/'training_stage.json',{'model':MODEL_NAME,'trained':[fixed],'test_evaluated':False})
        return
    frames=[]
    for fold in range(1,6):
        meta=fixed if fold==5 else train_candidate(features,df,fold,device,provenance)
        frames.append(evaluate_candidate(meta,features,df,device))
    combined=pd.concat(frames,ignore_index=True)
    if len(combined)!=400 or not combined.filename.is_unique or set(combined.filename)!=set(df.filename):
        raise ValueError('Expected one OOF prediction per original clip')
    combined.to_csv(REPORT/'predictions/yamnet_transfer_oof.csv',index=False)
    write_json(MODELS/'yamnet.json',{'name':MODEL_NAME,'checkpoint':fixed['checkpoint'],
               'checkpoint_sha256':fixed['checkpoint_sha256'],'deployment_trained_all_data':False})
    from . import report
    report.REPORT=REPORT
    report.generate(sync_results=False)
    print(f'New YAMNet experiment saved in {output}; deployed checkpoint preserved',flush=True)


if __name__=='__main__':
    main()
