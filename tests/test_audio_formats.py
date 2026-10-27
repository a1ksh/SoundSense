import subprocess

import numpy as np
import pytest
import soundfile as sf

from soundsense.audio import SUPPORTED_EXTENSIONS, ffmpeg_executable, load_uploaded_audio
from soundsense.common import ROOT, config
from soundsense.infer import Predictor
from soundsense.preprocess import load_audio

# These are encoded test fixtures, not replacements for the ESC-10 training data.
ENCODINGS = {
    '.wav': ('pcm_s16le', 'wav'), '.wave': ('pcm_s16le', 'wav'),
    '.mp3': ('libmp3lame', 'mp3'), '.m4a': ('aac', 'mp4'), '.aac': ('aac', 'adts'),
    '.flac': ('flac', 'flac'), '.ogg': ('libvorbis', 'ogg'), '.oga': ('libvorbis', 'ogg'),
    '.opus': ('libopus', 'opus'), '.aiff': ('pcm_s16be', 'aiff'), '.aif': ('pcm_s16be', 'aiff'),
    '.wma': ('wmav2', 'asf'), '.webm': ('libopus', 'webm'), '.mp4': ('aac', 'mp4'),
    '.caf': ('pcm_s16le', 'caf'), '.au': ('pcm_s16be', 'au'),
}


@pytest.fixture(scope='module')
def audio_files(tmp_path_factory):
    directory = tmp_path_factory.mktemp('audio_formats')
    sr = 44100
    t = np.arange(sr * 2) / sr
    y = np.column_stack([.2 * np.sin(2 * np.pi * 440 * t), .1 * np.sin(2 * np.pi * 660 * t)])
    original = directory / 'original.wav'
    sf.write(original, y, sr, subtype='PCM_16')
    files = {}
    for extension, (codec, container) in ENCODINGS.items():
        path = directory / ('Дыбыс үлгісі' + extension)
        subprocess.run([ffmpeg_executable(), '-nostdin', '-loglevel', 'error', '-i', str(original),
                        '-c:a', codec, '-f', container, str(path)], check=True, capture_output=True, timeout=30,
                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        files[extension] = path
    return original, files


@pytest.mark.parametrize('extension', SUPPORTED_EXTENSIONS)
def test_real_encoded_formats_decode_and_pad(extension, audio_files):
    _, files = audio_files
    y = load_uploaded_audio(files[extension], config()['preprocessing'])
    assert y.shape == (80000,) and np.isfinite(y).all()
    assert np.std(y[:16000]) > .01
    assert np.max(np.abs(y[64000:])) < 1e-6


def test_lossless_format_preserves_features_and_checkpoint_prediction(audio_files):
    original, files = audio_files
    cfg = config()['preprocessing']
    np.testing.assert_array_equal(load_uploaded_audio(files['.flac'], cfg), load_audio(original, cfg))
    predictor = Predictor(ROOT / 'models/cnn_best.pt')
    expected = predictor.predict(original)['probabilities']
    actual = predictor.predict(files['.flac'])['probabilities']
    np.testing.assert_allclose(list(actual.values()), list(expected.values()), rtol=1e-6, atol=1e-7)


def test_uppercase_extension_and_unicode_filename(audio_files):
    original, _ = audio_files
    path = original.with_name('Қазақша атауы бар файл.WAV')
    path.write_bytes(original.read_bytes())
    assert load_uploaded_audio(path, config()['preprocessing']).shape == (80000,)


def test_corrupt_file_is_rejected(tmp_path):
    path = tmp_path / 'broken.m4a'
    path.write_bytes(b'This is not audio')
    with pytest.raises(ValueError, match='бүлінген'):
        load_uploaded_audio(path, config()['preprocessing'])


def test_unsupported_file_is_rejected(tmp_path):
    path = tmp_path / 'document.pdf'
    path.write_bytes(b'%PDF')
    with pytest.raises(ValueError, match='форматына'):
        load_uploaded_audio(path, config()['preprocessing'])


def test_decoder_timeout_is_reported(tmp_path, monkeypatch):
    path = tmp_path / 'slow.m4a'
    path.write_bytes(b'undecodable')

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 30)

    monkeypatch.setattr('soundsense.audio.subprocess.run', timeout)
    with pytest.raises(ValueError, match='уақыты'):
        load_uploaded_audio(path, config()['preprocessing'])
