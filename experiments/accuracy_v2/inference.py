"""Checkpoint-driven V2 inference with explicit dependency failures."""
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

from soundsense.audio import load_uploaded_audio
from soundsense.common import ROOT, read_json, sha256, write_json
from soundsense.infer import KAZAKH
from soundsense.preprocess import logmel
from .core import (CACHE, MODELS, YAMNET_URL, cache_environment, efficientnet_backbone,
                   efficientnet_input, load_candidate)


class V2Predictor:
    def __init__(self, checkpoint=None, device='cpu'):
        cache_environment()
        self.device = torch.device(device)
        path = Path(checkpoint or MODELS / 'best_v2.pt')
        if not path.is_file():
            raise RuntimeError('Best V2 моделі табылмады. Алдымен Accuracy V2 тәжірибесін іске қосыңыз.')
        if checkpoint is None:
            selection = read_json(MODELS / 'best_v2.json')
            if sha256(path) != selection['checkpoint_sha256']:
                raise RuntimeError('Best V2 checkpoint SHA-256 сәйкес емес. Модельді қайта тексеріңіз.')
        self.model, self.metadata = load_candidate(path, self.device)
        self.labels = [r['label'] for r in self.metadata['class_mapping']]
        self.name = self.metadata['name']
        self.backbone = None
        if self.name == 'efficientnet_b0_mel':
            backbone = MODELS / 'efficientnet_backbone.pt'
            if not backbone.is_file():
                raise RuntimeError('EfficientNet pretrained backbone табылмады. README бойынша ресми weights-ті жүктеңіз.')
            if sha256(backbone) != self.metadata['provenance']['backbone_sha256']:
                raise RuntimeError('EfficientNet backbone SHA-256 сәйкес емес.')
            self.backbone = efficientnet_backbone(pretrained=False).to(self.device)
            self.backbone.load_state_dict(torch.load(backbone, map_location=self.device, weights_only=True), strict=True)
            self.backbone.eval()
        if self.name == 'yamnet_transfer':
            self.yamnet_python = Path(os.environ.get('SOUNDSENSE_YAMNET_PYTHON', str(ROOT / '.runtime/yamnet311/python.exe')))
            if not self.yamnet_python.is_file():
                raise RuntimeError('YAMNet үшін бөлек TensorFlow ортасы қажет. SOUNDSENSE_YAMNET_PYTHON айнымалысын орнатыңыз; experiments/accuracy_v2/README.md қараңыз.')

    def embedding(self, waveform, mel):
        if self.name == 'efficientnet_b0_mel':
            with torch.inference_mode():
                return self.backbone(efficientnet_input(mel).to(self.device)).cpu().numpy()[0]
        if self.name == 'yamnet_transfer':
            with tempfile.TemporaryDirectory(prefix='soundsense-yamnet-') as directory:
                directory = Path(directory)
                waves, request, output = directory / 'waveforms.npy', directory / 'request.json', directory / 'embedding.npz'
                np.save(waves, waveform[None], allow_pickle=False)
                write_json(request, {'waveforms': str(waves), 'filenames': ['uploaded_clip'], 'key': 'app_waveform_v1', 'model_url': YAMNET_URL})
                command = [str(self.yamnet_python), str(Path(__file__).with_name('extract_yamnet.py')), '--request', str(request), '--output', str(output)]
                try:
                    completed = subprocess.run(command, check=False, timeout=180, capture_output=True, text=True,
                                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                except (OSError, subprocess.TimeoutExpired) as exc:
                    raise RuntimeError(f'YAMNet embedding extraction іске қосылмады: {exc}') from exc
                if completed.returncode or not output.is_file():
                    detail = (completed.stderr or completed.stdout)[-1500:]
                    raise RuntimeError(f'YAMNet TensorFlow немесе pretrained weights жүктелмеді. README бойынша ортаны тексеріңіз. {detail}')
                with np.load(output, allow_pickle=False) as z:
                    return z['embeddings'][0].copy()
        return mel

    def predict(self, path):
        cfg = self.metadata['preprocessing']
        waveform = load_uploaded_audio(path, cfg)
        mel = logmel(waveform, cfg)
        features = self.embedding(waveform, mel)
        x = ((features - np.asarray(self.metadata['mean'])) / np.asarray(self.metadata['std'])).astype(np.float32)
        x = x[None, None] if self.name == 'cnn_v2' else x[None]
        with torch.inference_mode():
            p = self.model(torch.from_numpy(x).to(self.device)).softmax(1)[0].cpu().numpy()
        if p.shape != (10,) or not np.isfinite(p).all():
            raise RuntimeError('V2 моделі жарамды он ықтималдық қайтармады.')
        order = np.argsort(-p)
        return {'label': self.labels[order[0]], 'kazakh_label': KAZAKH.get(self.labels[order[0]], self.labels[order[0]]),
                'probabilities': dict(zip(self.labels, map(float, p))),
                'top3': {self.labels[i]: float(p[i]) for i in order[:3]},
                'waveform': waveform, 'mel': mel, 'sample_rate': cfg['sample_rate'], 'model': self.name}
