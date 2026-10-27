"""Exercise supported formats with real training audio through the local Gradio API."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from gradio_client import Client, handle_file

from app import create_app
from soundsense.audio import ffmpeg_executable
from soundsense.common import sha256, write_json
from soundsense.infer import Predictor

ENCODINGS = {
    '.wav': ('pcm_s16le', 'wav'), '.wave': ('pcm_s16le', 'wav'),
    '.mp3': ('libmp3lame', 'mp3'), '.m4a': ('aac', 'mp4'), '.aac': ('aac', 'adts'),
    '.flac': ('flac', 'flac'), '.ogg': ('libvorbis', 'ogg'), '.oga': ('libvorbis', 'ogg'),
    '.opus': ('libopus', 'opus'), '.aiff': ('pcm_s16be', 'aiff'), '.aif': ('pcm_s16be', 'aiff'),
    '.wma': ('wmav2', 'asf'), '.webm': ('libopus', 'webm'), '.mp4': ('aac', 'mp4'),
    '.caf': ('pcm_s16le', 'caf'), '.au': ('pcm_s16be', 'au'),
}


def main():
    torch.set_num_threads(4)
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'PYTHONIOENCODING': 'utf-8'}
    tests = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=ROOT, env=env,
                           capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=240)
    (ROOT / 'results/audio_formats_pytest.txt').write_text(tests.stdout + tests.stderr, encoding='utf-8')
    print(tests.stdout, flush=True)
    if tests.returncode:
        print(tests.stderr, flush=True)
        return tests.returncode
    manifest = pd.read_csv(ROOT / 'data/metadata/manifest.csv')
    row = manifest[manifest.split == 'train'].iloc[0]
    source = ROOT / 'data/raw' / row.filename
    predictor = Predictor()
    audit_before = sha256(ROOT / 'results/predictions/test_access_audit.json')
    checkpoint_before = {name: sha256(ROOT / f'models/{name}_best.pt') for name in ['mlp', 'cnn']}
    checks = []
    demo = create_app()
    try:
        _, url, _ = demo.launch(server_name='127.0.0.1', share=False, inbrowser=False,
                                prevent_thread_lock=True, quiet=True, max_file_size='100mb')
        client = Client(url, verbose=False)
        with tempfile.TemporaryDirectory(prefix='soundsense-format-check-') as directory:
            for extension, (codec, container) in ENCODINGS.items():
                path = Path(directory) / ('Дыбыс үлгісі' + extension)
                subprocess.run([ffmpeg_executable(), '-nostdin', '-loglevel', 'error', '-i', str(source),
                                '-c:a', codec, '-f', container, str(path)], check=True, capture_output=True, timeout=30,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                result = predictor.predict(path)
                probabilities = np.array(list(result['probabilities'].values()))
                assert len(probabilities) == 10 and np.isfinite(probabilities).all()
                assert abs(probabilities.sum() - 1) < 1e-5
                output = client.predict(handle_file(str(path)), api_name='/predict')
                assert len(output) == 5 and output[0] and output[2] and output[3]
                assert len(output[1]['confidences']) == 3
                preview = sf.info(output[4])
                assert preview.frames == 80000 and preview.samplerate == 16000 and preview.channels == 1
                checks.append({'extension': extension, 'codec': codec, 'passed': True,
                               'api_predicted_label': output[0], 'api_outputs': 5,
                               'preview_sample_rate': preview.samplerate, 'preview_frames': preview.frames})
                print(f'PASS {extension}: decoder, checkpoint inference, API and audio preview', flush=True)
            uppercase = Path(directory) / 'Үлкен әріп.MP3'
            uppercase.write_bytes((Path(directory) / 'Дыбыс үлгісі.mp3').read_bytes())
            assert client.predict(handle_file(str(uppercase)), api_name='/predict')[0]
    finally:
        demo.close()
    assert sha256(ROOT / 'results/predictions/test_access_audit.json') == audit_before
    assert {name: sha256(ROOT / f'models/{name}_best.pt') for name in ['mlp', 'cnn']} == checkpoint_before
    result = {'passed': True, 'formats': checks, 'format_count': len(checks), 'uppercase_extension_api': True,
              'unicode_and_spaces_in_filenames': True, 'source_audio': row.filename, 'source_split': 'train',
              'public_share': False, 'server_closed': True, 'training_repeated': False,
              'held_out_test_inference_repeated': False,
              'checkpoint_sha256': checkpoint_before, 'test_access_audit_sha256': audit_before,
              'imageio_ffmpeg': importlib.metadata.version('imageio-ffmpeg'),
              'ffmpeg_version': subprocess.check_output([ffmpeg_executable(), '-version'], text=True).splitlines()[0],
              'pytest_log': 'results/audio_formats_pytest.txt'}
    write_json(ROOT / 'results/audio_formats_validation.json', result)
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (ROOT / 'requirements-lock.txt').write_text('# Actual executed Windows/Python environment. Install official CUDA wheel first.\n' + freeze, encoding='utf-8')
    print(json.dumps({'passed': True, 'formats': len(checks), 'training_repeated': False}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
