"""Check that Windows checkout preserves the signed artifacts and source identities."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
files = ['results/history/mlp.csv', 'results/history/cnn.csv',
         'results/predictions/mlp_test.csv', 'results/predictions/cnn_test.csv',
         'models/mlp_best.pt', 'models/cnn_best.pt']
for path in files:
    original = (ROOT / path).read_bytes()
    staged = subprocess.check_output(['git', 'show', ':' + path], cwd=ROOT)
    checkout = subprocess.check_output(['git', '-c', 'core.autocrlf=true', 'cat-file', '--filters', '--path=' + path, ':' + path], cwd=ROOT)
    if hashlib.sha256(original).digest() != hashlib.sha256(staged).digest() or original != checkout:
        raise ValueError(f'Git normalization would invalidate recorded checksum: {path}')
for name in ['train.py', 'models.py', 'dataset.py', 'preprocess.py']:
    path = 'soundsense/' + name
    original = (ROOT / path).read_text(encoding='utf-8').encode()
    checkout = subprocess.check_output(['git', '-c', 'core.autocrlf=true', 'cat-file', '--filters', '--path=' + path, ':' + path], cwd=ROOT)
    normalized = checkout.decode('utf-8').replace('\r\n', '\n').encode()
    if original != normalized:
        raise ValueError(f'Source identity changes on Windows checkout: {name}')
from soundsense.common import config, read_json
from soundsense.train import experiment_signature
cfg = config()
stats = read_json(ROOT / 'artifacts/preprocessing_stats.json')
mapping = read_json(ROOT / 'artifacts/class_mapping.json')
for name in ['mlp', 'cnn']:
    expected = read_json(ROOT / f'results/history/{name}_training.json')['signature']
    if experiment_signature(name, cfg, stats, mapping) != expected:
        raise ValueError(f'Training resume signature changed: {name}')
result = {'passed': True, 'byte_exact_artifacts': files, 'normalized_source_identities': 4,
          'training_resume_signatures': 2, 'checkout_test': 'Git filters with core.autocrlf=true'}
(ROOT / 'results/git_checkout_verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result))
