"""Capture exact local pretrained artifact digests without changing models."""
from pathlib import Path
import platform
import subprocess

from soundsense.common import ROOT, read_json, sha256, write_json
from .core import CACHE, MODELS, REPORT


def main():
    artifact = {'efficientnet_b0_mel': {
        'source': 'https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth',
        'weights_enum': 'torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1',
        'backbone_checkpoint': 'models/accuracy_v2/efficientnet_backbone.pt',
        'backbone_sha256': sha256(MODELS / 'efficientnet_backbone.pt'),
        'license_files': ['models/accuracy_v2/TORCHVISION_LICENSE', 'models/accuracy_v2/EFFICIENTNET_LICENSE']}}
    tfhub = CACHE / 'tfhub'
    files = {str(path.relative_to(tfhub)).replace('\\', '/'): {'sha256': sha256(path), 'bytes': path.stat().st_size}
             for path in sorted(tfhub.rglob('*')) if path.is_file() and not path.name.endswith('.lock')}
    if files:
        artifact['yamnet_transfer'] = {'source': 'https://tfhub.dev/google/yamnet/1', 'version': 1,
                                      'sample_rate': 16000, 'input': 'original mono waveform', 'embedding_dimension': 1024,
                                      'pooling': 'mean over frames', 'frozen': True, 'extraction_device': 'CPU',
                                      'cached_files': files, 'total_cache_bytes': sum(row['bytes'] for row in files.values())}
    wheel = ROOT / '.runtime/tensorflow-wheel-provenance.json'
    if wheel.exists():
        artifact['tensorflow_wheel'] = read_json(wheel)
    runtime = ROOT / '.runtime/python-3.11.9-embed-amd64.zip'
    if runtime.exists():
        artifact['recovery_runtime'] = {'source': 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip',
                                        'sha256': sha256(runtime), 'baseline_venv_unchanged': True}
    write_json(REPORT / 'pretrained_artifacts.json', artifact)
    yamnet_python = ROOT / '.runtime/yamnet311/python.exe'
    if yamnet_python.exists():
        script = ("import json,sys,importlib.metadata as m; "
                  "names=['tensorflow','tensorflow-hub','tf-keras','keras','numpy','protobuf','setuptools','h5py','ml-dtypes']; "
                  "print(json.dumps({'python':sys.version,'packages':{n:m.version(n) for n in names}}))")
        output = subprocess.check_output([str(yamnet_python), '-c', script], text=True)
        import json
        write_json(REPORT / 'environment_yamnet.json', json.loads(output))
    print('Pretrained artifact hashes and isolated runtime provenance saved', flush=True)


if __name__ == '__main__':
    main()
