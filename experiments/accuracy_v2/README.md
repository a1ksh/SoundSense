# Accuracy V2

Independent experiments; baseline `models/cnn_best.pt`, `models/mlp_best.pt`,
`results/`, `docs/`, raw data and `.venv` are preserved. Results belong in
`reports/accuracy_v2/`; V2 checkpoints in `models/accuracy_v2/`.

## Normal Windows setup

Use a working Python 3.11 installation. Preserve a working baseline environment;
create separate environments for experiments. Match Torchvision to Torch 2.7.1.

```powershell
py -3.11 -m venv .venv-v2
.\.venv-v2\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv-v2\Scripts\python.exe -m pip install -r requirements.txt
py -3.11 -m venv .venv-yamnet
.\.venv-yamnet\Scripts\python.exe -m pip install -r experiments/accuracy_v2/requirements-yamnet.txt
$env:SOUNDSENSE_YAMNET_PYTHON = (Resolve-Path .venv-yamnet/Scripts/python.exe).Path
.\.venv-v2\Scripts\python.exe -m experiments.accuracy_v2.run --yamnet-python $env:SOUNDSENSE_YAMNET_PYTHON --cv-top 2
.\.venv-v2\Scripts\python.exe -m experiments.accuracy_v2.verify
.\.venv-v2\Scripts\python.exe -m pytest -q
.\.venv-v2\Scripts\python.exe app.py
```

For CPU use PyTorch's `https://download.pytorch.org/whl/cpu` wheel index.
CUDA is detected automatically. EfficientNet extraction uses batches of eight;
CNN/head training uses sixteen. Both transfer backbones are frozen; TensorFlow
extraction is CPU-only. A CUDA Toolkit installation is unnecessary when the
matching CUDA wheel and supported NVIDIA driver are available.

If YAMNet cannot be installed, `--models cnn_v2 efficientnet_b0_mel` explicitly
runs only those experiments. Failed candidates are recorded in `blockers.json`;
they are never counted as successful or substituted with random features.

## Restored runtime in this checkout

This checkout's `.venv` base Python path no longer existed. The session downloaded
the official Python 3.11.9 embeddable ZIP into ignored `.runtime/python311/`,
and configured its `python311._pth` to read the existing `.venv/Lib/site-packages`
without modifying those packages. Matching Torchvision lives in
`.runtime/packages/`. The project absolute path is included in that `.pth`;
after moving this checkout update it or use the normal environments above.
TensorFlow is installed separately under `.runtime/yamnet-packages/`, used by
`.runtime/yamnet311/python.exe`. Neither runtime nor downloaded caches are
committed. This is an application-local recovery, not a repair of the old
`.venv/Scripts/python.exe` launcher.

```powershell
$env:SOUNDSENSE_YAMNET_PYTHON = (Resolve-Path .runtime/yamnet311/python.exe).Path
.\.runtime\python311\python.exe -m experiments.accuracy_v2.run --yamnet-python $env:SOUNDSENSE_YAMNET_PYTHON --cv-top 2
.\.runtime\python311\python.exe -m experiments.accuracy_v2.verify
.\.runtime\python311\python.exe -m pytest -q
.\.runtime\python311\python.exe app.py
```

`--train-only` completes candidate training and validation without selecting a
winner or evaluating test labels. It was used while the isolated TensorFlow
runtime downloaded. Rerunning the full command reuses completed training when
recipe, code, data and checkpoint/history SHA-256 match. Predictions are reused
with checksums, rather than re-evaluating frozen test checkpoints. Changed code
or recipes require archiving the old V2 outputs in a separate experiment
directory; existing completed artifacts are deliberately not overwritten.

## Evaluation

`config.json` fixes seed and all three recipes before test evaluation. Original
comparison: train 1–3, validation 4, test 5. Every checkpoint and deployable model
is selected by minimum validation cross-entropy. After all candidates finish,
`selection.json` freezes the ranking and best two CV candidates before any new
fold-5 prediction. Outer CV test K uses validation K−1 (1 wraps to 5), all other
three folds train. Every original clip receives exactly one OOF prediction.
Fixed fold 5 is reused as its corresponding CV checkpoint; no duplicate test
inference is needed. A shared source is never split across partitions.

This remains exploratory: baseline fold 5 was already known; CV candidates were
screened on fixed validation, not selected in a fully nested search for each
outer fold. External AudioSet/ImageNet pretraining source overlap with ESC
recordings has not been ruled out. Single-seed fold standard deviation is not a
confidence interval. No all-data deployment retraining is performed.

Mel settings and label order are inherited from the baseline. CNN augmentation
is applied only to training clips; validation/test transforms are deterministic.
Transfer scaling is fitted on each fold's training embeddings only. YAMNet
receives original mono 16 kHz waveforms: the extraction subprocess gets only
waveforms, filenames, model URL and a cache identity, with no labels. Its mean
frame embeddings feed a newly trained ten-class head; 521 AudioSet classes are
never mistaken for ESC-10 labels.

## Packaged weights and cache recovery

V2 heads/CNN checkpoints, the selected `best_v2.pt` and EfficientNet's 16 MB
frozen backbone are packaged. The backbone remains below GitHub's per-file
limit and enables complete offline EfficientNet inference. To recreate it
from the official ImageNet weights (the download is about 20 MB):

```powershell
.\.runtime\python311\python.exe -c "import torch; from experiments.accuracy_v2.core import cache_environment, efficientnet_backbone, MODELS; cache_environment(); MODELS.mkdir(parents=True, exist_ok=True); torch.save(efficientnet_backbone(pretrained=True).state_dict(), MODELS / 'efficientnet_backbone.pt')"
```

Checkpoint metadata records the backbone SHA-256. Use the packaged backbone
for exact byte reproducibility. Repacked Torch serialization can produce
different bytes for the same tensors; rerun training/extraction in a fresh
output directory together rather than mixing repacked files with older head
metadata. The verified artifact digest is in `pretrained_artifacts.json`.
App loading fails clearly on a mismatch.

Official YAMNet SavedModel is downloaded by `tensorflow_hub.load` from
`https://tfhub.dev/google/yamnet/1`, cached under ignored
`data/accuracy_v2_cache/tfhub`. The same URL/version is used during extraction
and app inference. First use needs internet access; later inference uses the
cached model. To warm the cache in the separate environment:

```powershell
$env:TFHUB_CACHE_DIR = (Join-Path (Get-Location) 'data/accuracy_v2_cache/tfhub')
& $env:SOUNDSENSE_YAMNET_PYTHON -c "import tensorflow_hub as hub; hub.load('https://tfhub.dev/google/yamnet/1'); print('YAMNet cached')"
```

`download_tensorflow.py` is a recovery helper for slow Windows connections:
it downloads only the official PyPI Windows CPython 3.11 TensorFlow 2.20 wheel,
requires valid HTTP byte ranges, retries a bounded number of times and checks
the complete wheel against PyPI SHA-256 before installation.

## Evidence and references

`metrics.json`, `comparison.csv`, `cross_validation_summary.json`, per-fold
prediction receipts, `training_history/`, `split_manifest.csv`, `environment.json`,
`inference_verification.json`, `app_smoke.json` and actual pytest logs provide
the evidence. `report_kk.md` and `methodology_kk.md` explain results in Kazakh.
Graphs are generated by `python -m experiments.accuracy_v2.report` only from
these persisted rows, at 300 dpi; it never generates simulated performance.

- [TensorFlow official YAMNet transfer tutorial](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio).
- [Torchvision EfficientNet-B0](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html).
- [ESC-50 / ESC-10](https://github.com/karolpiczak/ESC-50): Piczak (2015), ESC: Dataset for Environmental Sound Classification. ESC-10 CC BY 3.0; baseline metadata attribution files are preserved.
- ImageNet EfficientNet weights: Torchvision/PyTorch, BSD-3-Clause implementation; upstream model/pretraining dataset terms apply. The frozen EfficientNet backbone is redistributed with its upstream attribution. YAMNet pretrained model: Google TensorFlow Hub model version 1; retain upstream model card/license when redistributing. YAMNet's SavedModel and source WAV files are not committed; reproducible official download instructions are above.
