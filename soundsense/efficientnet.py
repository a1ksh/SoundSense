"""Deploy the EfficientNet-B0 fixed head trained in the two-model Colab."""
import numpy as np
import librosa
from scipy.ndimage import zoom
import torch
from torch import nn

from .audio import load_uploaded_audio
from .checkpoints import load_weights
from .common import ROOT, read_json, sha256
from .labels import KAZAKH

DEPLOYMENT = ROOT / 'models/transfer'


def logmel(wave):
    power = librosa.feature.melspectrogram(
        y=wave, sr=16000, n_fft=1024, hop_length=512, n_mels=64, fmin=0, fmax=8000,
        power=2., center=True, pad_mode='constant', window='hann', htk=False, norm='slaney')
    db = librosa.power_to_db(power, ref=1., top_db=80)
    result = zoom(db, (1, 128/db.shape[1]), order=1, prefilter=False).astype(np.float32)
    if result.shape != (64, 128) or not np.isfinite(result).all():
        raise ValueError('Invalid EfficientNet Mel input')
    return result


def backbone_input(mel):
    x = torch.as_tensor(mel, dtype=torch.float32)
    if x.ndim == 2:
        x = x[None]
    x = ((x-x.amax(dim=(-2,-1), keepdim=True)+80)/80).clamp(0,1)
    x = nn.functional.interpolate(x[:,None].repeat(1,3,1,1), size=(224,224), mode='bilinear', align_corners=False)
    return (x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]


class EfficientNetHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(.2), nn.Linear(1280,10))

    def forward(self, x):
        return self.net(x)


class EfficientNetPredictor:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.name = 'efficientnet_b0_mel'
        checkpoint = DEPLOYMENT/'efficientnet_b0_fixed.pt'
        backbone = DEPLOYMENT/'efficientnet_b0_backbone.pt'
        record = DEPLOYMENT/'efficientnet_b0.json'
        if not all(path.is_file() for path in [checkpoint,backbone,record]):
            raise RuntimeError('EfficientNet-B0 моделі табылмады. models/transfer файлдарын қалпына келтіріңіз.')
        receipt = read_json(record)
        if receipt['name'] != self.name or sha256(checkpoint) != receipt['checkpoint_sha256'] or sha256(backbone) != receipt['backbone_sha256']:
            raise RuntimeError('EfficientNet-B0 салмақтарының SHA-256 мәні сәйкес емес.')
        payload = load_weights(checkpoint)
        self.labels = [r['label'] for r in read_json(ROOT/'artifacts/class_mapping.json')]
        if payload.get('schema') != 'soundsense_two_models_v1' or payload.get('name') != self.name or payload.get('dimension') != 1280:
            raise ValueError('Unsupported EfficientNet-B0 checkpoint format')
        if payload.get('labels') != self.labels or payload.get('test_fold') != 5 or payload.get('train_folds') != [1,2,3]:
            raise ValueError('EfficientNet-B0 label order or training split differs from the deployed experiment')
        for key in ['mean','std']:
            values = np.asarray(payload.get(key), dtype=np.float64)
            if values.shape != (1280,) or not np.isfinite(values).all() or (key == 'std' and (values < 1e-6).any()):
                raise ValueError(f'Invalid EfficientNet-B0 train {key}')
        self.mean = np.asarray(payload['mean'])
        self.std = np.asarray(payload['std'])
        self.metadata = payload
        try:
            from torchvision.models import efficientnet_b0
            self.backbone = efficientnet_b0(weights=None)
        except (ImportError, OSError, RuntimeError) as exc:
            raise RuntimeError('EfficientNet-B0 үшін torch==2.10.0 және torchvision==0.25.0 сәйкес ортасын орнатыңыз.') from exc
        self.backbone.classifier = nn.Identity()
        self.backbone.load_state_dict(load_weights(backbone), strict=True)
        self.backbone = self.backbone.requires_grad_(False).eval().to(self.device)
        self.head = EfficientNetHead().eval().to(self.device)
        self.head.load_state_dict(payload['state_dict'], strict=True)

    def warmup(self):
        with torch.inference_mode():
            features = self.backbone(torch.zeros(1,3,224,224,device=self.device))
            logits = self.head(features)
        if logits.shape != (1,10) or not torch.isfinite(logits).all():
            raise RuntimeError('EfficientNet-B0 бастапқы тексеруі орындалмады.')

    def close(self):
        pass

    def predict(self, path):
        wave = load_uploaded_audio(path, {'sample_rate':16000,'duration_seconds':5})
        mel = logmel(wave)
        with torch.inference_mode():
            features = self.backbone(backbone_input(mel).to(self.device)).cpu().numpy()[0]
            x = ((features-self.mean)/self.std).astype(np.float32)
            p = self.head(torch.from_numpy(x[None]).to(self.device)).softmax(1)[0].cpu().numpy()
        if p.shape != (10,) or not np.isfinite(p).all():
            raise RuntimeError('EfficientNet-B0 жарамды он ықтималдық қайтармады.')
        order = np.argsort(-p)
        return {'label':self.labels[order[0]], 'kazakh_label':KAZAKH[self.labels[order[0]]],
                'probabilities':dict(zip(self.labels,map(float,p))),
                'top3':{self.labels[i]:float(p[i]) for i in order[:3]},
                'waveform':wave,'mel':mel,'mel_frame_hop':5/128,
                'mel_scale_label':'Қуат (дБ)','display_name':'EfficientNet-B0',
                'sample_rate':16000,'model':self.name}
