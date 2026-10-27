"""Decode user uploads before applying the original training preprocessing."""
from pathlib import Path
import subprocess
import tempfile

import soundfile as sf

from .preprocess import load_audio

SUPPORTED_EXTENSIONS = (
    '.wav', '.wave', '.mp3', '.m4a', '.aac', '.flac', '.ogg', '.oga',
    '.opus', '.aiff', '.aif', '.wma', '.webm', '.mp4', '.caf', '.au',
)
FORMAT_LABEL = 'WAV/WAVE, MP3, M4A/AAC, FLAC, OGG/OGA, OPUS, AIFF/AIF, WMA, WEBM, MP4, CAF, AU'
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


def ffmpeg_executable():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError('Аудио декодері орнатылмаған. Орындаңыз: pip install -r requirements.txt') from exc


def load_uploaded_audio(path, cfg):
    """Keep native sample rate/channels; mono, soxr and Mel processing stay shared."""
    path = Path(path)
    if not path.is_file():
        raise ValueError('Аудиофайл табылмады. Файлды қайта жүктеңіз.')
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f'Бұл файл форматына қолдау жоқ. Қолдау көрсетіледі: {FORMAT_LABEL}.')
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError('Файл көлемі 100 МБ-тан аспауы тиіс.')
    # Use precisely the existing decoder for readable, reasonably small audio.
    # This preserves the original WAV features and saved-checkpoint predictions.
    try:
        info = sf.info(path)
    except sf.LibsndfileError:
        info = None
    if info is not None and info.frames * info.channels <= 2_000_000:
        return load_audio(path, cfg)
    # AAC/M4A, WMA and media containers need FFmpeg. Decode only the prefix,
    # without downmixing/resampling, then use the same preprocessing as training.
    with tempfile.TemporaryDirectory(prefix='soundsense-audio-') as directory:
        wav = Path(directory) / 'decoded.wav'
        command = [ffmpeg_executable(), '-nostdin', '-hide_banner', '-loglevel', 'error',
                   '-protocol_whitelist', 'file,pipe', '-i', str(path.resolve()),
                   '-map', '0:a:0', '-t', str(cfg['duration_seconds']), '-vn', '-sn', '-dn',
                   '-c:a', 'pcm_f32le', '-f', 'wav', '-fs', '64000000', str(wav)]
        try:
            result = subprocess.run(command, capture_output=True, timeout=30,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except subprocess.TimeoutExpired as exc:
            raise ValueError('Аудионы оқу уақыты аяқталды. Қысқарақ файл жүктеңіз.') from exc
        if result.returncode != 0 or not wav.is_file():
            raise ValueError('Файлда оқылатын аудио жоқ немесе файл бүлінген. Басқа аудиофайл жүктеңіз.')
        return load_audio(wav, cfg)
