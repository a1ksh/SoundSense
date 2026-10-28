"""CPU-only official TF Hub YAMNet extraction. No labels enter this process."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding='utf-8'))
    # Strictly only waveforms, filenames, checksum and frozen model metadata.
    if set(request) != {'waveforms', 'filenames', 'key', 'model_url'}:
        raise ValueError('Embedding request must not contain labels')
    import tensorflow as tf
    import tensorflow_hub as hub
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    model = hub.load(request['model_url'])
    waves = np.load(request['waveforms'], allow_pickle=False)
    pooled = []
    for i, waveform in enumerate(waves):
        _, embeddings, _ = model(tf.convert_to_tensor(waveform, dtype=tf.float32))
        pooled.append(embeddings.numpy().mean(axis=0))
        if (i + 1) % 40 == 0:
            print(f'YAMNet CPU embeddings: {i+1}/{len(waves)}', flush=True)
    values = np.stack(pooled).astype(np.float32)
    if values.shape != (len(request['filenames']), 1024) or not np.isfinite(values).all():
        raise ValueError('Invalid YAMNet embeddings')
    np.savez_compressed(args.output, embeddings=values, filenames=np.asarray(request['filenames']), key=np.asarray(request['key']))
    print(json.dumps({'tensorflow': tf.__version__, 'tensorflow_hub': hub.__version__,
                      'model_url': request['model_url'], 'shape': list(values.shape),
                      'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}), flush=True)


if __name__ == '__main__':
    main()
