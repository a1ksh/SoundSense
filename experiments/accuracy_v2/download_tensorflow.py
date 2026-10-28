"""Bounded parallel download of official PyPI wheel, with SHA-256 validation.

Only a project-local cache is written. HTTP ranges must be honored.
"""
import concurrent.futures
import hashlib
import json
import subprocess
import socket
from pathlib import Path
import time
import urllib.request


def main():
    root = Path(__file__).resolve().parents[2]
    with urllib.request.urlopen('https://pypi.org/pypi/tensorflow/2.20.0/json', timeout=30) as response:
        metadata = json.load(response)
    wheel = next(f for f in metadata['urls'] if 'cp311-cp311-win_amd64' in f['filename'])
    target = root / '.runtime' / wheel['filename']
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == wheel['digests']['sha256']:
        print('Verified existing official TensorFlow wheel', flush=True)
        return
    size = wheel['size']
    # Resolve once; repeated Windows DNS lookups failed during the first retry.
    # TLS still verifies the original official hostname on every request.
    download_ip = socket.gethostbyname('files.pythonhosted.org')
    count = 64
    chunk = (size + count - 1) // count

    def fetch(index):
        start, end = index * chunk, min(size - 1, (index + 1) * chunk - 1)
        part = target.with_suffix(f'.part{index}')
        if part.exists() and part.stat().st_size == end - start + 1:
            return part
        for attempt in range(3):
            try:
                headers = part.with_suffix(part.suffix + '.headers')
                subprocess.run(['curl.exe', '-fsSL', '--resolve', f'files.pythonhosted.org:443:{download_ip}',
                                '--connect-timeout', '15', '--max-time', '120',
                                '-r', f'{start}-{end}', '-D', str(headers), wheel['url'], '-o', str(part)],
                               check=True, timeout=140, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                if f'content-range: bytes {start}-{end}/' not in headers.read_text(encoding='utf-8').lower():
                    raise RuntimeError('Official server did not honor range request')
                if part.stat().st_size != end - start + 1:
                    raise RuntimeError('Incomplete wheel range')
                print(f'Official TensorFlow wheel range {index+1}/{count} complete', flush=True)
                return part
            except (OSError, subprocess.SubprocessError):
                if attempt == 2:
                    raise
                time.sleep(2)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        parts = list(pool.map(fetch, range(count)))
    with target.open('wb') as out:
        for part in parts:
            with part.open('rb') as source:
                while block := source.read(1024 * 1024):
                    out.write(block)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != wheel['digests']['sha256']:
        raise RuntimeError('Official TensorFlow wheel checksum mismatch')
    (target.parent / 'tensorflow-wheel-provenance.json').write_text(json.dumps({'url': wheel['url'], 'sha256': digest, 'size': size}, indent=2), encoding='utf-8')
    print(f'Verified {target.name}: {digest}', flush=True)


if __name__ == '__main__':
    main()
