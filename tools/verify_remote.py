"""Verify the remote GitHub tree against local Git blobs, without logging credentials."""
import argparse
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record-delivery', action='store_true')
    args = parser.parse_args()
    remote = subprocess.check_output(['git', 'remote', 'get-url', 'origin'], cwd=ROOT, text=True).strip()
    if remote != 'https://github.com/a1ksh/SoundSense.git':
        raise ValueError('Unexpected remote')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GCM_INTERACTIVE': 'never'}
    result = subprocess.run(['git', 'credential', 'fill'], input='protocol=https\nhost=github.com\npath=a1ksh/SoundSense.git\n\n',
                            text=True, capture_output=True, env=env, cwd=ROOT, timeout=30, check=True)
    credentials = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    auth = base64.b64encode((credentials['username'] + ':' + credentials['password']).encode()).decode()

    def get(path):
        req = urllib.request.Request('https://api.github.com/repos/a1ksh/SoundSense/' + path,
                                     headers={'Authorization': 'Basic ' + auth, 'User-Agent': 'SoundSense-remote-verification',
                                              'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'})
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)

    head = get('git/ref/heads/main')['object']['sha']
    if head != commit:
        raise ValueError('Remote main differs from local HEAD')
    tree = get(f'git/trees/{commit}?recursive=1')
    if tree.get('truncated'):
        raise ValueError('Remote tree response was truncated')
    actual = {item['path']: item['sha'] for item in tree['tree'] if item['type'] == 'blob'}
    lines = subprocess.check_output(['git', 'ls-tree', '-r', '--format=%(objectname) %(path)', 'HEAD'], cwd=ROOT, text=True)
    expected = {line.split(' ', 1)[1]: line.split(' ', 1)[0] for line in lines.splitlines()}
    if actual != expected:
        raise ValueError('Remote file/blob tree differs from the committed project')
    if any(path.startswith(('data/raw/', 'data/cache/', '.venv/')) for path in actual):
        raise ValueError('Excluded local artifacts were unexpectedly uploaded')
    remote_summary = get(f'contents/results/summary.json?ref={commit}')
    summary = json.loads(base64.b64decode(remote_summary['content']))
    receipt = {'push_verified': True, 'repository': 'https://github.com/a1ksh/SoundSense',
               'remote': remote, 'branch': 'main', 'commit': commit,
               'verified_utc': datetime.now(timezone.utc).isoformat(),
               'verification': 'GitHub REST branch ref + recursive tree: every committed path and blob SHA matches local Git; remote summary decoded',
               'verified_files': len(actual), 'remote_png_figures': sum(p.startswith('results/figures/') and p.endswith('.png') for p in actual),
               'remote_test_accuracy': {name: summary['models'][name]['metrics']['accuracy'] for name in ['mlp', 'cnn']},
               'checkpoints': {item['path']: item['size'] for item in tree['tree'] if item['path'] in ['models/mlp_best.pt', 'models/cnn_best.pt']},
               'raw_wav_uploaded': False, 'lfs_used': False}
    (ROOT / 'results/delivery_receipt.local.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    if args.record_delivery:
        path = ROOT / 'results/delivery.json'
        delivery = json.loads(path.read_text(encoding='utf-8'))
        delivery.update({'push_verified': True, 'status': 'Payload pushed and verified through the GitHub API',
                         'verified_payload_commit': commit, 'verified_utc': receipt['verified_utc'],
                         'verified_file_count': len(actual), 'verification': receipt['verification'],
                         'final_head_note': 'The receipt commit may follow this payload commit. Final HEAD is verified separately; see local delivery_receipt.local.json or git log -1.'})
        path.write_text(json.dumps(delivery, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
