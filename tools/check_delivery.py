"""Read-only GitHub delivery checks. Credentials stay in memory and are never logged."""
import base64
import json
import os
from pathlib import Path
import subprocess
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    remote = subprocess.check_output(['git', 'remote', 'get-url', 'origin'], cwd=ROOT, text=True).strip()
    if remote != 'https://github.com/a1ksh/SoundSense.git':
        raise ValueError('Unexpected remote; refusing credential use')
    env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GCM_INTERACTIVE': 'never'}
    result = subprocess.run(['git', 'credential', 'fill'], input='protocol=https\nhost=github.com\npath=a1ksh/SoundSense.git\n\n',
                            text=True, capture_output=True, env=env, cwd=ROOT, timeout=30)
    if result.returncode:
        print(json.dumps({'repository': 'a1ksh/SoundSense', 'credential_available': False, 'lfs_quota_verified': False}))
        return
    credentials = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    auth = base64.b64encode((credentials['username'] + ':' + credentials['password']).encode()).decode()

    def get(path):
        req = urllib.request.Request('https://api.github.com/' + path,
                                     headers={'Authorization': 'Basic ' + auth, 'User-Agent': 'SoundSense-delivery-check',
                                              'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'})
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            return exc.code, {}

    code, repo = get('repos/a1ksh/SoundSense')
    user_code, user = get('user')
    billing_code, billing = get('users/a1ksh/settings/billing/usage?year=2026&month=10')
    # Usage alone does not prove free remaining quota or a zero spending limit.
    output = {'repository': repo.get('full_name', 'a1ksh/SoundSense'), 'repository_http_status': code,
              'private': repo.get('private'), 'permissions': repo.get('permissions'),
              'authenticated_login': user.get('login'), 'billing_http_status': billing_code,
              'lfs_quota_verified': False,
              'lfs_reason': 'Read-only billing API did not establish remaining free LFS storage/bandwidth and zero overage spending. Raw WAVs remain local.'}
    path = ROOT / 'results/github_access.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
