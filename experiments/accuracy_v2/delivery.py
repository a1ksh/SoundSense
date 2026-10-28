"""Verify this exact repository and pushed branch using existing Git credentials.

Credentials stay in memory and are never printed or written. The receipt is local
and ignored so it can identify the exact final committed payload.
"""
import argparse
import base64
from datetime import datetime, timezone
import json
import hashlib
import os
import subprocess
import urllib.parse
import urllib.request

from soundsense.common import ROOT, read_json, write_json
from .core import REPORT

REPOSITORY = 'a1ksh/SoundSense'
REMOTE = f'https://github.com/{REPOSITORY}.git'
BRANCH = 'feature/accuracy-v2'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--access-only', action='store_true')
    parser.add_argument('--create-draft-pr', action='store_true')
    args = parser.parse_args()
    if git('remote', 'get-url', 'origin') != REMOTE:
        raise ValueError('Unexpected remote; refusing another repository')
    env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GCM_INTERACTIVE': 'never'}
    result = subprocess.run(['git', 'credential', 'fill'],
                            input=f'protocol=https\nhost=github.com\npath={REPOSITORY}.git\n\n',
                            capture_output=True, text=True, check=True, timeout=30, env=env, cwd=ROOT)
    credential = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    authorization = base64.b64encode((credential['username'] + ':' + credential['password']).encode()).decode()

    def request(path, payload=None):
        # Curl's Windows network handling proved reliable when urllib timed out.
        # Pass the authorization config on stdin, never argv, disk or logs.
        values = [('url', f'https://api.github.com/repos/{REPOSITORY}{path}'),
                  ('header', 'Authorization: Basic ' + authorization),
                  ('header', 'User-Agent: SoundSense-Accuracy-V2'),
                  ('header', 'Accept: application/vnd.github+json'),
                  ('header', 'Content-Type: application/json')]
        if payload is not None:
            values.extend([('request', 'POST'), ('data', json.dumps(payload))])
        configuration = '\n'.join(f'{key} = {json.dumps(value)}' for key, value in values) + '\n'
        completed = subprocess.run(['curl.exe', '--config', '-', '--silent', '--show-error',
                                    '--fail-with-body', '--connect-timeout', '15', '--max-time', '60'],
                                   input=configuration, text=True, capture_output=True, timeout=75,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if completed.returncode:
            raise RuntimeError(f'GitHub API request failed (curl code {completed.returncode}): {completed.stderr[-300:]}')
        return json.loads(completed.stdout)

    repository = request('')
    if repository['full_name'].lower() != REPOSITORY.lower() or not repository.get('permissions', {}).get('push'):
        raise ValueError('Could not verify push permission for this repository')
    if args.access_only:
        receipt = {'repository': REPOSITORY, 'push_permission_verified': True, 'branch': BRANCH,
                   'verified_utc': datetime.now(timezone.utc).isoformat()}
        write_json(REPORT / 'remote_access.json', receipt)
        print(json.dumps(receipt))
        return
    if git('branch', '--show-current') != BRANCH:
        raise ValueError('Unexpected branch')
    commit = git('rev-parse', 'HEAD')
    # CSV files are checksum-bearing experiment artifacts. Git text conversion
    # must not change their bytes when publishing or checking out another OS.
    artifact_checks = 0
    for directory, digest_key in [('predictions', 'csv_sha256'), ('training_history', 'history_sha256')]:
        for receipt_path in (REPORT / directory).glob('*.json'):
            receipt = read_json(receipt_path)
            if digest_key not in receipt:
                continue
            csv_path = receipt_path.with_suffix('.csv')
            relative = csv_path.relative_to(ROOT).as_posix()
            content = subprocess.check_output(['git', 'show', f'HEAD:{relative}'], cwd=ROOT)
            if hashlib.sha256(content).hexdigest() != receipt[digest_key]:
                raise ValueError(f'Committed artifact checksum differs: {relative}')
            artifact_checks += 1
    remote_commit = request('/git/ref/heads/' + BRANCH)['object']['sha']
    if remote_commit != commit:
        raise ValueError('Pushed branch does not match local HEAD')
    tree = request(f'/git/trees/{commit}?recursive=1')
    if tree.get('truncated'):
        raise ValueError('Remote tree truncated')
    actual = {item['path']: item['sha'] for item in tree['tree'] if item['type'] == 'blob'}
    expected = {line.split(' ', 1)[1]: line.split(' ', 1)[0] for line in git('ls-tree', '-r', '--format=%(objectname) %(path)', 'HEAD').splitlines()}
    if actual != expected:
        raise ValueError('Remote/local file trees differ')
    forbidden = ('.runtime/', '.venv/', '.venv-v2/', '.venv-yamnet/', 'data/raw/', 'data/cache/', 'data/accuracy_v2_cache/')
    if any(path.startswith(forbidden) or path == '.env' for path in actual):
        raise ValueError('Local-only environment/data unexpectedly committed')
    receipt = {'repository': REMOTE, 'branch': BRANCH, 'commit': commit, 'push_verified': True,
               'verified_utc': datetime.now(timezone.utc).isoformat(), 'verified_blob_count': len(actual),
               'committed_csv_checksums_verified': artifact_checks,
               'verification': 'GitHub REST branch SHA and all recursive tree blob SHAs match local HEAD',
               'raw_audio_uploaded': False, 'lfs_used': False}
    if args.create_draft_pr:
        existing = request('/pulls?state=open&head=' + urllib.parse.quote('a1ksh:' + BRANCH))
        if existing:
            pull = existing[0]
        else:
            selection = read_json(REPORT / 'selection.json')
            metrics = read_json(REPORT / 'metrics.json')[selection['name'] + '_fixed']
            cv = read_json(REPORT / 'cross_validation_summary.json').get(selection['name'])
            body = (f'Add real ESC-10 transfer-learning experiments and a selectable Best V2 model while preserving the CNN v1 app and checkpoints. '
                    f'The validation-selected {selection["name"]} measured {metrics["correct"]}/{metrics["count"]} '
                    f'({metrics["accuracy"]:.2%}) on exploratory fold 5, macro-F1 {metrics["macro_f1"]:.4f}.\n\n')
            if cv:
                body += (f'Official five-fold OOF evaluation: {cv["aggregate_correct"]}/400, '
                         f'fold accuracy {cv["accuracy_mean"]:.2%} ± {cv["accuracy_std_sample"]:.2%} (sample SD, seed 42).\n\n')
            body += ('Model and checkpoint selection uses validation loss; no test-score tuning. Frozen ImageNet/AudioSet pretraining source overlap is not ruled out. '
                     'CV candidates were screened on the fixed validation split, so this is exploratory rather than a fully nested model search.\n\n'
                     'Validation evidence: reports/accuracy_v2/pytest.txt, inference_verification.json, app_smoke.json, '
                     'persisted clip probabilities, real training histories, 300 dpi figures and Kazakh reports. '
                     'Both model choices retain audio upload/plots/top-3/playback. Runtime, raw WAVs and large caches are excluded. '
                     'YAMNet uses a separately pinned CPU extraction environment; official model download instructions are included.')
            pull = request('/pulls', {'title': 'Improve SoundSense with validated audio transfer learning',
                                     'head': BRANCH, 'base': repository['default_branch'], 'draft': True, 'body': body})
        receipt['pull_request'] = pull['html_url']
        receipt['pull_request_draft'] = pull['draft']
    write_json(REPORT / 'delivery.local.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
