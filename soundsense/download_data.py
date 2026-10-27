"""Download exactly ESC-10, pinned to an upstream commit, without a full archive."""
import argparse
import concurrent.futures
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from .common import ROOT, config, read_json, sha256, write_json

META = ROOT / "data/metadata"
RAW = ROOT / "data/raw"
SPLITS = {1: "train", 2: "train", 3: "train", 4: "validation", 5: "test"}


def fetch(url, path, attempts=4):
    """Atomic bounded retry. A failed partial file is never treated as complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SoundSense-coursework/1.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                tmp = path.with_suffix(path.suffix + ".part")
                with tmp.open("wb") as out:
                    while block := response.read(256 * 1024):
                        out.write(block)
            tmp.replace(path)
            return
        except (OSError, ValueError) as exc:
            if attempt == attempts - 1:
                raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc
            time.sleep(2 ** attempt)


def validate_splits(df):
    if len(df) != 400 or not df.filename.is_unique or df.category.nunique() != 10:
        raise ValueError("Expected 400 unique ESC-10 files in ten classes")
    if set(df.fold) != set(range(1, 6)) or not (df.groupby('fold').size() == 80).all():
        raise ValueError("Invalid original folds")
    if not (df.groupby(['fold', 'category']).size() == 8).all():
        raise ValueError("Expected eight examples per class in each fold")
    if not (df.groupby('src_file').fold.nunique() == 1).all():
        raise ValueError("Source recording leakage between folds")
    if not (df['split'] == df.fold.map(SPLITS)).all():
        raise ValueError("Split must be train=1,2,3 / validation=4 / test=5")
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        for column in ['filename', 'src_file']:
            if set(df.loc[df.split == a, column]) & set(df.loc[df.split == b, column]):
                raise ValueError(f"Leakage in {column}: {a}/{b}")


def valid_audio(path):
    try:
        info = sf.info(path)
        if info.frames != 220500 or info.samplerate != 44100 or info.channels != 1:
            return False
        y, _ = sf.read(path, dtype="float32")
        return bool(np.isfinite(y).all())
    except (OSError, RuntimeError):
        return False


def prepare(cfg, skip_download=False):
    META.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    lock_path = META / "source_lock.json"
    if lock_path.exists():
        lock = read_json(lock_path)
        if lock['repository'] != cfg['dataset']['repository']:
            raise ValueError("Dataset repository differs from pinned source")
        if cfg['dataset']['revision'] not in ('master', lock['commit']):
            raise ValueError("Revision differs from source_lock.json")
    else:
        if skip_download:
            raise FileNotFoundError("--skip-download needs pinned metadata and all 400 WAVs")
        repo = cfg['dataset']['repository']
        revision = cfg['dataset']['revision']
        tmp = META / "upstream_commit.json"
        fetch(f"https://api.github.com/repos/{repo}/commits/{revision}", tmp)
        commit = read_json(tmp)['sha']
        tmp.unlink()
        lock = {"repository": repo, "commit": commit, "resolved_utc": datetime.now(timezone.utc).isoformat()}
        write_json(lock_path, lock)
    base = f"https://raw.githubusercontent.com/{lock['repository']}/{lock['commit']}"
    for remote, local in [('meta/esc50.csv', 'esc50.csv'), ('LICENSE', 'LICENSE'), ('README.md', 'UPSTREAM_README.md')]:
        dest = META / local
        if not dest.exists():
            if skip_download:
                raise FileNotFoundError(dest)
            fetch(f"{base}/{remote}", dest)
    full = pd.read_csv(META / "esc50.csv", dtype={'src_file': str})
    df = full[full.esc10.astype(str).str.lower() == 'true'].copy().reset_index(drop=True)
    df['split'] = df.fold.map(SPLITS)
    classes = df[['target', 'category']].drop_duplicates().sort_values('target')
    mapping = [{'index': i, 'esc50_target': int(row.target), 'label': row.category}
               for i, row in enumerate(classes.itertuples())]
    index = {item['label']: item['index'] for item in mapping}
    df['class_index'] = df.category.map(index)
    validate_splits(df)
    extra = set(p.name for p in RAW.glob('*.wav')) - set(df.filename)
    if extra:
        raise ValueError(f"Unexpected non-ESC-10 WAVs in data/raw: {sorted(extra)[:5]}")
    previous = {}
    manifest_path = META / "manifest.csv"
    if manifest_path.exists():
        old = pd.read_csv(manifest_path)
        if 'sha256' in old:
            previous = dict(zip(old.filename, old.sha256))

    def one(filename):
        path = RAW / filename
        valid = path.exists() and valid_audio(path)
        if valid and filename in previous:
            valid = sha256(path) == previous[filename]
        if not valid:
            if skip_download:
                raise ValueError(f"Missing or invalid WAV: {filename}")
            fetch(f"{base}/audio/{filename}", path)
            if not valid_audio(path):
                raise ValueError(f"Downloaded WAV failed integrity validation: {filename}")
        return filename, sha256(path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        hashes = dict(tqdm(pool.map(one, df.filename), total=len(df), desc="ESC-10 integrity"))
    df['sha256'] = df.filename.map(hashes)
    df['source_url'] = df.filename.map(lambda f: f"{base}/audio/{f}")
    df.to_csv(manifest_path, index=False)
    write_json(ROOT / "artifacts/class_mapping.json", mapping)
    # Upstream LICENSE includes original Freesound URLs, authors and clip licenses.
    license_text = (META / 'LICENSE').read_text(encoding='utf-8')
    attributions = []
    for row in df.itertuples():
        token = f"{row.fold}-{row.src_file}-{row.take}.ogg"
        lines = [line for line in license_text.splitlines() if f'[{token}]' in line]
        if len(lines) != 1:
            raise ValueError(f"Missing/ambiguous clip attribution for {row.filename}")
        attributions.append({'filename': row.filename, 'original_source_id': row.src_file, 'attribution': lines[0]})
    pd.DataFrame(attributions).to_csv(META / "attributions.csv", index=False)
    write_json(META / "provenance.json", {
        **lock, 'dataset': 'ESC-10 subset of ESC-50', 'count': len(df),
        'license': 'CC BY 3.0 (ESC-10 subset); see original LICENSE for individual attributions',
        'metadata_sha256': sha256(META / 'esc50.csv'), 'license_sha256': sha256(META / 'LICENSE'),
        'citation': 'Karol J. Piczak (2015). ESC: Dataset for Environmental Sound Classification. ACM Multimedia, 1015–1018. DOI:10.1145/2733373.2806390',
        'split_counts': {str(k): int(v) for k, v in df.groupby('split').size().items()},
        'source_overlap': False,
    })
    return df, mapping


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-download', action='store_true')
    args = parser.parse_args()
    prepare(config(), args.skip_download)
