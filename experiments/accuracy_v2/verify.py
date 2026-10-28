"""Full app inference matches cached training features; local HTTP API smoke."""
import json
import time

import numpy as np
import torch

from soundsense.common import ROOT, write_json
from soundsense.infer import Predictor
from .core import CACHE, MODELS, REPORT, load_candidate, manifest, raw_mels
from .inference import V2Predictor


def main():
    torch.set_num_threads(4)
    df = manifest()
    index = int(df.index[df.fold == 1][0])
    path = ROOT / 'data/raw' / df.iloc[index].filename
    results = {}
    baseline = Predictor()
    start = time.perf_counter()
    baseline.predict(path)
    results['cnn_v1'] = {'passed': True, 'end_to_end_seconds_one_clip': time.perf_counter() - start,
                         'input': path.name, 'split': 'training fold 1', 'latency_note': 'one clip, model initialization excluded'}
    # A training example is used for transform equivalence, not performance claims.
    for name in ['cnn_v2', 'efficientnet_b0_mel', 'yamnet_transfer']:
        checkpoint = MODELS / f'{name}_fixed.pt'
        if not checkpoint.exists():
            continue
        predictor = V2Predictor(checkpoint)
        start = time.perf_counter()
        first = predictor.predict(path)
        elapsed = time.perf_counter() - start
        second = predictor.predict(path)
        p = np.asarray(list(first['probabilities'].values()))
        np.testing.assert_allclose(p, list(second['probabilities'].values()), rtol=1e-6, atol=1e-7)
        model, payload = load_candidate(checkpoint)
        if name == 'cnn_v2':
            x = raw_mels(df)[index]
        else:
            with np.load(CACHE / f'{name}_embeddings.npz', allow_pickle=False) as z:
                x = z['embeddings'][index].copy()
        x = ((x - np.asarray(payload['mean'])) / np.asarray(payload['std'])).astype(np.float32)
        x = x[None, None] if name == 'cnn_v2' else x[None]
        with torch.inference_mode():
            expected = model(torch.from_numpy(x)).softmax(1)[0].numpy()
        np.testing.assert_allclose(p, expected, rtol=2e-4, atol=2e-5)
        assert first['waveform'].shape == (80000,) and first['mel'].shape == (64, 128)
        assert len(first['top3']) == 3
        results[name] = {'passed': True, 'input': path.name, 'split': 'training fold 1',
                         'repeat_deterministic': True, 'training_app_probability_equivalence': True,
                         'end_to_end_seconds_one_clip': elapsed, 'latency_note': 'one clip, model initialization excluded; YAMNet subprocess startup included'}
        print(f'{name}: end-to-end verified, {elapsed:.3f} seconds', flush=True)
    write_json(REPORT / 'inference_verification.json', results)
    from app import create_app
    from gradio_client import Client, handle_file
    app = create_app()
    responses = {}
    try:
        _, url, _ = app.launch(server_name='127.0.0.1', share=False, inbrowser=False,
                               prevent_thread_lock=True, quiet=True, max_file_size='100mb')
        client = Client(url, verbose=False)
        for name in ['CNN v1 (baseline)', 'Best V2']:
            output = client.predict(handle_file(str(path)), name, api_name='/predict')
            assert len(output) == 5 and output[0] and output[1] and output[2] and output[3] and output[4]
            responses[name] = {'passed': True, 'kazakh_label': output[0], 'output_count': len(output)}
        # Preserve API callers that provide only the audio: Gradio fills the
        # selector's documented default rather than silently relabeling a model.
        default_output = client.predict(handle_file(str(path)), api_name='/predict')
        assert len(default_output) == 5
        responses['default_selector'] = {'passed': True, 'kazakh_label': default_output[0]}
    finally:
        app.close()
    write_json(REPORT / 'app_smoke.json', {'passed': True, 'transport': 'local HTTP Gradio API',
                                         'models': responses, 'server_closed': True, 'public_share': False})
    print('Both Gradio models and default API verified; server closed', flush=True)


if __name__ == '__main__':
    main()
