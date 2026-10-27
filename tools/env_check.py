"""Print only nonsecret execution-environment details."""
import json
import platform
import shutil

import gradio
import librosa
import torch

disk = shutil.disk_usage('.')
available = torch.cuda.is_available()
print(json.dumps({'python': platform.python_version(), 'platform': platform.platform(),
                  'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'cuda_available': available,
                  'gpu': torch.cuda.get_device_name(0) if available else None,
                  'cuda_tensor_sum': torch.ones(4, device='cuda').sum().item() if available else None,
                  'librosa': librosa.__version__, 'gradio': gradio.__version__, 'free_disk_bytes': disk.free}, indent=2))
