"""Serialized access to a persistent YAMNet child with timeouts and cleanup."""
import atexit
import base64
import json
from pathlib import Path
from queue import Empty, Queue
import subprocess
import tempfile
from threading import RLock, Thread

import numpy as np

from soundsense.cli import utf8_environment
from .core import YAMNET_URL


class YamnetWorker:
    def __init__(self, python, timeout=180):
        self.python = Path(python)
        self.timeout = timeout
        self.process = None
        self.lock = RLock()
        self.sequence = 0
        self.stderr = None
        atexit.register(self.close)

    def _read(self, process, messages):
        try:
            for line in process.stdout:
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise ValueError('Expected a JSON object from YAMNet worker')
                messages.put(message)
        except (ValueError, OSError) as exc:
            messages.put({'error': str(exc)})
        finally:
            messages.put({'error': 'YAMNet worker exited unexpectedly'})

    def _receive(self):
        try:
            message = self.messages.get(timeout=self.timeout)
        except Empty as exc:
            raise RuntimeError('YAMNet worker timed out') from exc
        if 'error' in message:
            raise RuntimeError(f"YAMNet: {message['error']}")
        return message

    def start(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return
            self.close()
            self.messages = Queue()
            self.stderr = tempfile.TemporaryFile()
            try:
                self.process = subprocess.Popen(
                    [str(self.python), '-u', str(Path(__file__).with_name('yamnet_worker.py')),
                     '--model-url', YAMNET_URL],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
                    text=True, encoding='utf-8', errors='replace', env=utf8_environment(),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                )
                self.reader = Thread(target=self._read, args=(self.process, self.messages), daemon=True)
                self.reader.start()
                message = self._receive()
                if message.get('ready') is not True:
                    raise RuntimeError('YAMNet worker did not become ready')
                self.ready = message
            except (OSError, RuntimeError) as exc:
                self.close()
                raise RuntimeError(f'YAMNet ортасы немесе weights дайын емес. '
                                   f'SOUNDSENSE_YAMNET_PYTHON және README тексеріңіз. {exc}') from exc

    def _request(self, waveform, include_mel=False):
        wave = np.asarray(waveform, dtype='<f4')
        if wave.ndim != 1 or not 0 < wave.size <= 160000 or not np.isfinite(wave).all():
            raise ValueError('Expected a finite mono waveform up to ten seconds')
        with self.lock:
            self.start()
            self.sequence += 1
            request = {'id': self.sequence, 'audio': base64.b64encode(wave.tobytes()).decode('ascii')}
            try:
                self.process.stdin.write(json.dumps(request) + '\n')
                self.process.stdin.flush()
                message = self._receive()
                result = np.asarray(message.get('embedding'), dtype=np.float32)
                if message.get('id') != self.sequence or result.shape != (1024,) or not np.isfinite(result).all():
                    raise RuntimeError('Invalid YAMNet worker response')
                if include_mel:
                    mel = np.asarray(message.get('log_mel'), dtype=np.float32)
                    if mel.ndim != 2 or mel.shape[0] != 64 or not 1 <= mel.shape[1] <= 2000 or not np.isfinite(mel).all():
                        raise RuntimeError('Invalid YAMNet log-Mel response')
                    return result, mel
                return result
            except (OSError, RuntimeError, ValueError) as exc:
                self.close()
                raise RuntimeError(f'YAMNet талдауы орындалмады: {exc}') from exc

    def predict(self, waveform):
        return self._request(waveform)

    def analyze(self, waveform):
        return self._request(waveform, include_mel=True)

    def close(self):
        with self.lock:
            process, self.process = self.process, None
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                for stream in (process.stdin, process.stdout):
                    if stream is not None:
                        stream.close()
                reader = getattr(self, 'reader', None)
                if reader is not None:
                    reader.join(timeout=2)
            if self.stderr is not None:
                self.stderr.close()
                self.stderr = None
