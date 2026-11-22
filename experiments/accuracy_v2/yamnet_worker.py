"""One CPU TensorFlow model per application, with a bounded request protocol.

Executed directly by the isolated TensorFlow interpreter; never imports Torch.
Only base64 float32 waveforms enter the process, never labels or file paths.
"""
import argparse
import base64
from contextlib import redirect_stdout
import json
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def send(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def serve(model_url):
    import numpy as np
    with redirect_stdout(sys.stderr):
        import tensorflow as tf
        import tensorflow_hub as hub
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        model = hub.load(model_url)

        def embedding(wave):
            _, values, spectrogram = model(tf.convert_to_tensor(wave, dtype=tf.float32))
            result = values.numpy().mean(axis=0)
            if result.shape != (1024,) or not np.isfinite(result).all():
                raise ValueError('Invalid YAMNet embedding')
            return result, spectrogram.numpy().T

        # Validate packages, weights and the first graph execution before READY.
        embedding(np.zeros(80000, dtype=np.float32))
    send({'ready': True, 'pid': os.getpid(), 'tensorflow': tf.__version__})
    for line in sys.stdin:
        if len(line) > 1_000_000:
            raise ValueError('YAMNet request too large')
        request = json.loads(line)
        if set(request) != {'id', 'audio'}:
            raise ValueError('Expected only request id and waveform')
        raw = base64.b64decode(request['audio'], validate=True)
        if not raw or len(raw) > 16000 * 10 * 4 or len(raw) % 4:
            raise ValueError('Expected up to ten seconds of float32 audio')
        wave = np.frombuffer(raw, dtype='<f4')
        if not np.isfinite(wave).all():
            raise ValueError('Nonfinite waveform')
        with redirect_stdout(sys.stderr):
            result, mel = embedding(wave)
        send({'id': request['id'], 'embedding': result.tolist(), 'log_mel': mel.tolist()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-url', choices=['https://tfhub.dev/google/yamnet/1'], required=True)
    args = parser.parse_args()
    try:
        serve(args.model_url)
    except Exception as exc:
        send({'error': f'{type(exc).__name__}: {exc}'})
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
