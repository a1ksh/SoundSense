"""Validation-selected training, deterministic seeds and completed-run reuse."""
import gc
import time

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .common import ROOT, code_identity, identity, read_json, seed_everything, sha256, write_json
from .dataset import split_dataset
from .models import build_model, parameter_count


def experiment_signature(name, cfg, stats, mapping):
    return identity({'name': name, 'seed': cfg['seed'], 'training': cfg['training'],
                     'model': cfg['models'][name], 'stats': stats, 'mapping': mapping,
                     'torch_version': str(torch.__version__),
                     'code': code_identity(['train.py', 'models.py', 'dataset.py', 'preprocess.py'])})


def epoch_pass(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    loss_sum, correct, total = 0., 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training/validation loss')
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += int((logits.argmax(1) == y).sum().item())
            total += len(y)
    return loss_sum / total, correct / total


def _train(name, cfg, features, manifest, stats, mapping, device, batch_size, signature):
    seed_everything(cfg['seed'])
    opts = cfg['training']
    model = build_model(name, cfg['models'][name], cfg['preprocessing']).to(device)
    train = split_dataset(features, manifest, 'train')
    val = split_dataset(features, manifest, 'validation')
    generator = torch.Generator().manual_seed(cfg['seed'])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=opts['num_workers'], generator=generator)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=opts['num_workers'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts['learning_rate'], weight_decay=opts['weight_decay'])
    checkpoint = ROOT / f'models/{name}_best.pt'
    checkpoint.parent.mkdir(exist_ok=True)
    history_path = ROOT / f'results/history/{name}.csv'
    history_path.parent.mkdir(parents=True, exist_ok=True)
    best, stale, best_epoch = float('inf'), 0, 0
    rows = []
    start = time.perf_counter()
    for epoch in range(1, opts['max_epochs'] + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = epoch_pass(model, train_loader, device, optimizer)
        val_loss, val_acc = epoch_pass(model, val_loader, device)
        rows.append({'epoch': epoch, 'train_loss': train_loss, 'validation_loss': val_loss,
                     'train_accuracy': train_acc, 'validation_accuracy': val_acc,
                     'seconds': time.perf_counter() - epoch_start})
        pd.DataFrame(rows).to_csv(history_path, index=False)
        print(f'{name.upper()} {epoch:02d}/{opts["max_epochs"]}: train loss={train_loss:.4f} acc={train_acc:.3f} '
              f'| val loss={val_loss:.4f} acc={val_acc:.3f}', flush=True)
        if val_loss < best:
            best, best_epoch, stale = val_loss, epoch, 0
            payload = {'schema': 1, 'name': name, 'state_dict': model.state_dict(),
                       'model_config': cfg['models'][name], 'preprocessing': cfg['preprocessing'],
                       'statistics': stats, 'class_mapping': mapping, 'seed': cfg['seed'],
                       'epoch': epoch, 'validation_loss': best, 'validation_accuracy': val_acc,
                       'signature': signature}
            tmp = checkpoint.with_suffix('.pt.tmp')
            torch.save(payload, tmp)
            tmp.replace(checkpoint)
        else:
            stale += 1
        if stale >= opts['patience']:
            break
    if device.type == 'cuda':
        torch.cuda.synchronize()
    result = {'model': name, 'signature': signature, 'epochs_completed': len(rows),
              'best_epoch': best_epoch, 'best_validation_loss': best,
              'training_seconds': time.perf_counter() - start, 'batch_size': batch_size,
              'parameter_count': parameter_count(model), 'device': str(device),
              'device_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU',
              'checkpoint_sha256': sha256(checkpoint), 'history_sha256': sha256(history_path),
              'seed': cfg['seed'], 'complete': True}
    result['torch_version'] = str(torch.__version__)
    write_json(ROOT / f'results/history/{name}_training.json', result)
    return result


def train_model(name, cfg, features, manifest, stats, mapping, device, force=False):
    signature = experiment_signature(name, cfg, stats, mapping)
    status = ROOT / f'results/history/{name}_training.json'
    checkpoint = ROOT / f'models/{name}_best.pt'
    if not force and status.exists() and checkpoint.exists():
        previous = read_json(status)
        if previous['signature'] != signature:
            raise ValueError(f'{name}: configuration/code/data changed; use --force-train explicitly')
        if previous['complete'] and previous['checkpoint_sha256'] == sha256(checkpoint):
            if previous['history_sha256'] != sha256(ROOT / f'results/history/{name}.csv'):
                raise ValueError('Training history checksum mismatch')
            print(f'{name}: reusing completed training (best epoch {previous["best_epoch"]})', flush=True)
            return previous
    batch_size = cfg['training']['batch_size']
    while True:
        try:
            return _train(name, cfg, features, manifest, stats, mapping, device, batch_size, signature)
        except torch.cuda.OutOfMemoryError:
            if batch_size <= 1:
                raise
            batch_size = max(1, batch_size // 2)
            gc.collect()
            torch.cuda.empty_cache()
            print(f'{name}: CUDA OOM; restarting from seed with batch size {batch_size}', flush=True)
