"""YAMNet-only inference using the official frontend and a persistent worker."""
import os
from pathlib import Path
import numpy as np
import torch
from soundsense.audio import load_uploaded_audio
from soundsense.common import ROOT, read_json, sha256
from soundsense.labels import KAZAKH
from .core import MODELS, MODEL_NAME, cache_environment, load_candidate
from .worker_client import YamnetWorker


class YAMNetPredictor:
    def __init__(self, checkpoint=None, device='cpu'):
        cache_environment()
        self.device = torch.device(device)
        path = Path(checkpoint or MODELS/'yamnet_transfer_fixed.pt')
        if not path.is_file():
            raise RuntimeError('YAMNet моделі табылмады. README бойынша checkpoint-ті қалпына келтіріңіз.')
        if checkpoint is None:
            record=read_json(MODELS/'yamnet.json')
            if record['name'] != MODEL_NAME or sha256(path) != record['checkpoint_sha256']:
                raise RuntimeError('YAMNet checkpoint SHA-256 сәйкес емес.')
        self.model,self.metadata=load_candidate(path,self.device)
        self.labels=[r['label'] for r in self.metadata['class_mapping']]
        self.name=MODEL_NAME
        python=Path(os.environ.get('SOUNDSENSE_YAMNET_PYTHON',str(ROOT/'.runtime/yamnet311/python.exe')))
        if not python.is_file():
            raise RuntimeError('YAMNet үшін бөлек TensorFlow ортасы қажет. SOUNDSENSE_YAMNET_PYTHON айнымалысын орнатыңыз; README қараңыз.')
        self.worker=YamnetWorker(python)

    def warmup(self):
        self.worker.start()

    def close(self):
        self.worker.close()

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()

    def predict(self,path):
        waveform=load_uploaded_audio(path,self.metadata['preprocessing'])
        features,mel=self.worker.analyze(waveform)
        x=((features-np.asarray(self.metadata['mean']))/np.asarray(self.metadata['std'])).astype(np.float32)
        with torch.inference_mode():
            probabilities=self.model(torch.from_numpy(x[None]).to(self.device)).softmax(1)[0].cpu().numpy()
        if probabilities.shape!=(10,) or not np.isfinite(probabilities).all():
            raise RuntimeError('YAMNet жарамды он ықтималдық қайтармады.')
        order=np.argsort(-probabilities)
        return {'label':self.labels[order[0]], 'kazakh_label':KAZAKH[self.labels[order[0]]],
                'probabilities':dict(zip(self.labels,map(float,probabilities))),
                'top3':{self.labels[i]:float(probabilities[i]) for i in order[:3]},
                'waveform':waveform,'mel':mel,'mel_frame_hop':.01,
                'sample_rate':16000,'model':MODEL_NAME}
