"""Verify that presentation changes preserve model outputs, plots and playback."""
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from app import create_app
from soundsense.common import read_json, sha256, write_json
from soundsense.infer import KAZAKH, Predictor

torch.set_num_threads(4)
manifest = pd.read_csv(ROOT / 'data/metadata/manifest.csv')
path = ROOT / 'data/raw' / manifest[manifest.split == 'train'].iloc[0].filename
expected = Predictor().predict(path)
demo = create_app()
classify = next(fn.fn for fn in demo.fns.values() if fn.fn is not None and fn.fn.__name__ == 'classify')
label, top3, waveform, mel, audio = classify(str(path))
assert label == expected['kazakh_label']
expected_top3 = {KAZAKH[k]: v for k, v in expected['top3'].items()}
assert list(top3) == list(expected_top3)
np.testing.assert_allclose(list(top3.values()), list(expected_top3.values()), rtol=0, atol=0)
np.testing.assert_array_equal(waveform.axes[0].lines[0].get_ydata(), expected['waveform'])
np.testing.assert_array_equal(mel.axes[0].images[0].get_array(), expected['mel'])
assert waveform.axes[0].get_xlabel() == 'Уақыт (с)'
assert waveform.axes[0].get_ylabel() == 'Амплитуда'
assert mel.axes[0].get_ylabel() == 'Мел жолағы'
assert audio[0] == expected['sample_rate']
np.testing.assert_array_equal(audio[1], expected['waveform'])
qa = ROOT / 'results/qa'
waveform.savefig(qa / 'interface_waveform.png', dpi=160)
mel.savefig(qa / 'interface_mel.png', dpi=160)
subprocess.run(['node', '--check', str(ROOT / 'assets/interface.js')], check=True, timeout=30)
formats = read_json(ROOT / 'results/audio_formats_validation.json')
assert formats['passed'] and formats['format_count'] == 16
result = {
    'passed': True, 'scope': 'Kazakh presentation only', 'prediction_matches_original_model': True,
    'top_three_probabilities_unchanged': True, 'waveform_samples_unchanged': True,
    'mel_values_unchanged': True, 'playback_samples_unchanged': True,
    'kazakh_class_names_and_plot_axes': True, 'prediction_api_output_count': 5,
    'regression_tests': (ROOT / 'results/audio_formats_pytest.txt').read_text(encoding='utf-8').strip(),
    'tested_audio_extensions': formats['format_count'], 'javascript_syntax_valid': True,
    'checkpoint_sha256': {name: sha256(ROOT / f'models/{name}_best.pt') for name in ['mlp', 'cnn']},
    'visual_browser_review_completed': False,
    'visual_review_blocker': 'Computer Use stopped because it could not determine the current browser URL on Windows with enough confidence.',
}
write_json(ROOT / 'results/interface_validation.json', result)
print(result)
