"""Checkpoint-only deployment, independent of cached features and raw dataset."""
import numpy as np
import torch

from .common import ROOT
from .audio import load_uploaded_audio
from .evaluate import load_checkpoint
from .preprocess import logmel, standardize

KAZAKH = {'dog': 'Иттің үруі', 'rain': 'Жаңбыр', 'sea_waves': 'Теңіз толқындары',
          'crying_baby': 'Сәбидің жылауы', 'clock_tick': 'Сағат тықылы', 'sneezing': 'Түшкіру',
          'helicopter': 'Тікұшақ', 'chainsaw': 'Моторлы ара', 'rooster': 'Әтеш', 'crackling_fire': 'Оттың шытырлауы'}


class Predictor:
    def __init__(self, checkpoint=None, device='cpu'):
        self.device = torch.device(device)
        self.model, self.metadata = load_checkpoint(checkpoint or ROOT / 'models/cnn_best.pt', self.device)
        self.labels = [item['label'] for item in self.metadata['class_mapping']]

    def predict(self, path):
        cfg = self.metadata['preprocessing']
        y = load_uploaded_audio(path, cfg)
        mel = logmel(y, cfg)
        x = standardize(mel, self.metadata['statistics'])
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(torch.from_numpy(x[None, None]).to(self.device)), dim=1)[0].cpu().numpy()
        order = np.argsort(-probabilities)
        return {'label': self.labels[order[0]], 'kazakh_label': KAZAKH.get(self.labels[order[0]], self.labels[order[0]]),
                'probabilities': {label: float(p) for label, p in zip(self.labels, probabilities)},
                'top3': {self.labels[i]: float(probabilities[i]) for i in order[:3]},
                'waveform': y, 'mel': mel, 'sample_rate': cfg['sample_rate']}
