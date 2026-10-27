"""Noninteractive local experiment. python -m soundsense.run_all"""
import argparse
import importlib.metadata
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')

import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import pandas as pd
import torch

from .common import ROOT, config, read_json, seed_everything, write_json


def utc():
    return datetime.now(timezone.utc).isoformat()


def git_sha():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unavailable'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config.yaml'))
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument('--force-train', action='store_true')
    args = parser.parse_args()
    cfg = config(args.config)
    status_path = ROOT / 'results/pipeline_status.json'
    runs_path = ROOT / 'results/pipeline_runs.json'
    runs = read_json(runs_path) if runs_path.exists() else []
    if status_path.exists():
        prior = read_json(status_path)
        if prior.get('finished_utc') and not any(run['started_utc'] == prior['started_utc'] for run in runs):
            runs.append(prior)
            write_json(runs_path, runs)
    status = {'success': False, 'started_utc': utc(), 'stages': {}, 'force_train': args.force_train}
    write_json(status_path, status)
    start = time.perf_counter()

    def stage(name, fn):
        print(f'\n--- {name} ---', flush=True)
        t0 = time.perf_counter()
        value = fn()
        status['stages'][name] = {'success': True, 'seconds': time.perf_counter() - t0}
        write_json(status_path, status)
        return value

    try:
        torch.set_num_threads(cfg['training']['cpu_threads'])
        seed_everything(cfg['seed'])
        reason = None
        if torch.cuda.is_available():
            try:
                device = torch.device('cuda')
                torch.ones(4, device=device).sum().item()
            except RuntimeError as exc:
                device, reason = torch.device('cpu'), f'CUDA smoke failed: {exc}'
        else:
            device, reason = torch.device('cpu'), 'torch.cuda.is_available() returned False'
        print(f'Device: {device}; {torch.cuda.get_device_name(0) if device.type == "cuda" else reason}', flush=True)
        from .download_data import prepare
        from .preprocess import prepare_features
        from .train import train_model
        from .evaluate import metrics_from_predictions, test_predictions
        from .visualize import generate
        from .report import write_reports
        manifest, mapping = stage('download_validate', lambda: prepare(cfg, args.skip_download))
        features, stats = stage('preprocess', lambda: prepare_features(manifest, cfg['preprocessing']))
        packages = ['torch', 'numpy', 'pandas', 'librosa', 'soundfile', 'scikit-learn', 'matplotlib', 'seaborn', 'scipy', 'gradio', 'tqdm', 'pyyaml', 'pytest']
        metadata_path = ROOT / 'results/run_metadata.json'
        # Preserve the original training run start when the complete artifacts are reused.
        existing = read_json(metadata_path) if metadata_path.exists() else {}
        metadata = {'started_utc': existing.get('started_utc', status['started_utc']) if not args.force_train else status['started_utc'],
                    'last_pipeline_utc': status['started_utc'], 'python': platform.python_version(), 'platform': platform.platform(),
                    'packages': {p: importlib.metadata.version(p) for p in packages},
                    'device': str(device), 'device_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else platform.processor(),
                    'cpu_fallback_reason': reason, 'cuda_available': torch.cuda.is_available(), 'cuda_runtime': torch.version.cuda,
                    'seed': cfg['seed'], 'deterministic_algorithms': True, 'config': cfg, 'statistics': stats,
                    'git_commit_at_run_start': existing.get('git_commit_at_run_start', git_sha()) if not args.force_train else git_sha(),
                    'dataset_provenance': read_json(ROOT / 'data/metadata/provenance.json'),
                    'study_design': 'single-seed, one held-out original fold; no five-fold replicated study'}
        write_json(metadata_path, metadata)
        freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
        (ROOT / 'requirements-lock.txt').write_text('# Actual executed Windows/Python environment. Install official CUDA wheel first.\n' + freeze, encoding='utf-8')
        trainings = {}
        for name in ['mlp', 'cnn']:
            trainings[name] = stage(f'train_{name}', lambda name=name: train_model(name, cfg, features, manifest, stats, mapping, device, args.force_train))
        models = {}
        comparison = []
        for name in ['mlp', 'cnn']:
            predictions = stage(f'test_inference_{name}', lambda name=name: test_predictions(name, features, manifest, stats, device))
            metrics = stage(f'metrics_{name}', lambda name=name, predictions=predictions: metrics_from_predictions(name, predictions, mapping))
            models[name] = {'training': trainings[name], 'metrics': metrics}
            comparison.append({'model': name, **{k: trainings[name][k] for k in ['parameter_count', 'training_seconds', 'device', 'device_name', 'epochs_completed', 'best_epoch']},
                               **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}})
        summary = {'study_design': metadata['study_design'], 'seed': cfg['seed'], 'test_fold': 5, 'test_count': 80, 'models': models}
        write_json(ROOT / 'results/summary.json', summary)
        pd.DataFrame(comparison).to_csv(ROOT / 'results/model_comparison.csv', index=False)
        metadata['training'] = trainings
        metadata['total_training_seconds'] = sum(t['training_seconds'] for t in trainings.values())
        write_json(metadata_path, metadata)
        if not (ROOT / 'results/delivery.json').exists():
            write_json(ROOT / 'results/delivery.json', {'push_verified': False, 'raw_wav_uploaded': False, 'reason': 'Delivery not yet checked'})
        inventory = stage('figures', generate)
        status['figure_count'] = len(inventory)
        stage('documentation', write_reports)

        def tests():
            env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}
            result = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=ROOT, env=env,
                                    capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=240)
            (ROOT / 'results/pytest.txt').write_text(result.stdout + result.stderr, encoding='utf-8')
            print(result.stdout, flush=True)
            if result.returncode:
                raise RuntimeError(f'pytest failed ({result.returncode}); see results/pytest.txt\n{result.stderr}')
        stage('pytest', tests)
        from .smoke import gradio_smoke
        stage('gradio_http_smoke', gradio_smoke)
        from .verify import verify
        stage('artifact_audit', verify)
        status.update({'success': True, 'finished_utc': utc(), 'seconds': time.perf_counter() - start})
        metadata['pipeline_success'] = True
        metadata['last_pipeline_seconds'] = status['seconds']
        write_json(metadata_path, metadata)
        write_json(status_path, status)
        runs.append(status)
        write_json(runs_path, runs)
        write_reports()
        print('\n' + pd.DataFrame(comparison).to_string(index=False), flush=True)
        print('\nPIPELINE_RESULT=' + __import__('json').dumps(status), flush=True)
        return 0
    except Exception as exc:
        status.update({'success': False, 'finished_utc': utc(), 'seconds': time.perf_counter() - start,
                       'error': str(exc), 'traceback': traceback.format_exc()})
        write_json(status_path, status)
        runs.append(status)
        write_json(runs_path, runs)
        traceback.print_exc()
        print('PIPELINE_RESULT=' + __import__('json').dumps(status), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
