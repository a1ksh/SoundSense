# Accuracy V2 baseline audit

Audit date: 2026-10-28. Repository root: `C:/Users/nurda/Downloads/SoundSense/SoundSense`.
Initial branch: `main`; clean working tree. Origin: `https://github.com/a1ksh/SoundSense.git`.
Work is isolated on `feature/accuracy-v2`; baseline checkpoints and reports are retained.

## Verified baseline

`results/summary.json`, `results/predictions/{cnn,mlp}_test.csv`, and
`results/history/*_training.json` store the prior experiment evidence. CNN: 67/80
(83.75%), macro-F1 0.8349856353, 35 epochs, best epoch 34. MLP: 48/80
(60%), macro-F1 0.5775787517, 19 epochs, best epoch 12. These are previously
observed fold-5 results, not a new untouched evaluation.

`data/metadata/esc50.csv` contains exactly 400 ESC-10 clips: ten classes of 40,
five official folds of 80. All 400 WAV files are present in `data/raw`.
Grouping `src_file` across official folds found zero overlapping sources.
`soundsense/download_data.py` validates the same separation and fixes
train folds 1–3, validation 4, test 5. `artifacts/class_mapping.json` orders
classes by original ESC-50 target, not alphabetically.

## Pipeline and limitations

`soundsense/preprocess.py`: mono 16 kHz, deterministic first five seconds,
right zero padding; 1024-point Hann FFT, hop 512, 64 Slaney Mel bands, 80 dB
range, time interpolation to 128 frames. Global scalar normalization is
fitted only on training clips. `soundsense/infer.py` uses these same transforms.
`soundsense/models.py`: three Conv/BatchNorm/ReLU/MaxPool blocks with channels
16/32/64, adaptive average pooling, dropout 0.3, ten-class linear head
(24,058 parameters). `soundsense/train.py` uses AdamW, batch 16, LR 0.001,
weight decay 0.0001, maximum 35 epochs and patience 7; checkpoints selected
strictly by minimum validation cross-entropy. No augmentation or pretrained
features are used. Very small data, temporal interpolation and globally pooled
scratch features may limit generalization. Single-seed one-fold results have
substantial sampling uncertainty; each test mistake changes accuracy by 1.25 pp.

## Environment and app

`requirements.txt` pins Torch 2.7.1; `results/run_metadata.json` records previous
Torch 2.7.1+cu126, Python 3.11.9 and RTX 3050 training. Current `nvidia-smi`
detects RTX 3050 Laptop GPU with 4096 MiB. However `.venv/Scripts/python.exe`
fails because its base interpreter in AppData no longer exists. No `python`,
`python3`, `py`, `uv` or `conda` command was found. Preserve `.venv` packages;
attempt a project-local official Python runtime before any model changes.
CUDA availability in Torch remains unverified until that runtime works.

Recovery verified: official project-local Python 3.11.9 can import the unchanged
baseline packages; Torch reports `2.7.1+cu126`, CUDA available, RTX 3050 Laptop
GPU. Matching Torchvision `0.22.1+cu126` was added only to `.runtime/packages`.
The first embedded-runtime pytest collection failed because its `.pth` project
path resolved incorrectly; an explicit absolute project path corrected that.
All original **32 tests passed** in 60.23 seconds; the real new run is stored
in `reports/accuracy_v2/baseline_pytest.txt`. The old `.venv` launcher remains
untouched; use the local runtime or recreate an independent Python environment.

`app.py`, `soundsense/audio.py`, `assets/interface.*` provide Gradio upload,
16 extensions, 100 MB limit, top three probabilities, waveform/Mel plots and
processed audio. Existing tests live in `tests/`; prior logs in `results/pytest.txt`
and `results/audio_formats_pytest.txt` are historical evidence only.

## V2 evaluation contract

Freeze candidate configurations before evaluation. Compare fixed training
1–3 / validation 4, choose deployable candidate by validation loss only, then
evaluate each frozen candidate on fold 5 once. Treat this as exploratory because
fold 5 was already observed. For five-fold CV, outer test K, validation K−1
(wrapping 1→5), other three folds train. Fit scaling only on each training
partition. Store one prediction per original clip, not per frame or augmentation.
Frozen AudioSet/ImageNet pretraining may overlap semantically or potentially in
source with ESC: external pretraining source overlap has not been ruled out.
