# Installation and reproducibility

The [main README](../README.md) provides installation commands for the two-model application. Use Python 3.11 with PyTorch 2.10.0 / Torchvision 0.25.0 in the main environment, and TensorFlow 2.20.0 in a separate YAMNet environment. Set `SOUNDSENSE_YAMNET_PYTHON` to that environment's Python executable before running the application or live inference tests.

The recorded experiment ran locally on Windows with CUDA. The notebook outputs are real saved outputs from that environment, not verified Google-hosted Colab execution. The exact recorded versions are in [colab_two_models/run/environment.json](../colab_two_models/run/environment.json); training metadata remains with each checkpoint.

## Required files

| Task | Required assets |
| --- | --- |
| Run the application | Repository model files; separate TensorFlow environment; official YAMNet weights downloaded on first use |
| Run the complete test suite | Application assets, the 400 WAVs in `data/raw/`, and the original YAMNet verification feature cache |
| Reproduce the two-model notebook | Full trained Colab bundle, or internet access to download data and pretrained backbones |
| Inspect results | Committed CSV/JSON tables, histories, predictions, and PNG/SVG/PDF figures |

## Verify the local source and application

1. Install both environments following the [README](../README.md).
2. Download the pinned audio using `python -m soundsense.download_data`, or copy `assets/audio/*.wav` from the full release bundle into `data/raw/`.
3. Download `SoundSense_YAMNet_Verification_Cache.zip` from the [latest release](https://github.com/a1ksh/SoundSense/releases/latest) and extract it **into the repository root**. It supplies `data/accuracy_v2_cache/yamnet_transfer_embeddings.npz` and its checksum metadata. This small cache reproduces the original YAMNet experiment and is separate from the two-model notebook's feature-cache format.
4. Run:

```bash
python -m soundsense.download_data --skip-download
python -m experiments.accuracy_v2.check
```

The comprehensive check verifies five saved YAMNet heads and their history hashes, regenerates the YAMNet report, runs pytest, reconstructs all 400 saved OOF predictions and train-only scalers, exercises a persistent YAMNet worker, and tests both deployed models through the actual local Gradio HTTP API. It writes [reports/accuracy_v2/quality_checks.json](../reports/accuracy_v2/quality_checks.json). It does not retrain models.

Run only tests with `python -m pytest -q`. To verify a separately running app:

```bash
python app.py --port 7869
# In a second terminal with the same environment:
python -m tools.check_website_models --url http://127.0.0.1:7869
```

The HTTP check compares each model's top-three predictions with its saved test scores and checks all three page routes, five API outputs, and the 16 kHz five-second audio preview.

## Recreate reports without retraining

```bash
python -m experiments.accuracy_v2.report
```

This reads the preserved YAMNet predictions and epoch histories and produces 16 YAMNet plots in `reports/accuracy_v2/plots/` and `results/figures/`. The latter also contains two dataset plots. It does not recreate the two-model comparison figures or change the README.

For both models and all 58 figures, run the [Colab notebook](../colab_two_models/SoundSense_YAMNet_EfficientNet_Full_Colab.ipynb) with `RUN_MODE = 'use_trained'` and the full bundle. Its checks cover all ten heads, train-only scalers, dataset/backbone/frontend feature contracts, fold coverage, real audio inference, and exports in PNG/SVG/PDF. See the [Colab guide](../colab_two_models/README.md).

## Train a new experiment

Use the notebook with `RUN_MODE = 'retrain'` to train both models. It uses frozen backbone features; it does not fine-tune either backbone.

For a separate YAMNet-only experiment, point the command at a new empty output directory:

```powershell
python -m experiments.accuracy_v2.run --retrain --output-dir .runtime/new_yamnet_run --yamnet-python $env:SOUNDSENSE_YAMNET_PYTHON
```

Saved deployment checkpoints are not overwritten. The default `python -m experiments.accuracy_v2.run` validates existing YAMNet artifacts and recreates their report.

| Head training setting | YAMNet | EfficientNet-B0 |
| --- | ---: | ---: |
| Seed | 42 | 42 |
| Optimizer | AdamW | AdamW |
| Learning rate | 0.001 | 0.002 |
| Weight decay | 0.001 | 0.001 |
| Batch size | 16 | 16 |
| Maximum epochs | 160 | 120 |
| Early stopping patience | 25 | 20 |
| Dropout | 0.2 | 0.2 |
| Checkpoint selection | Minimum validation cross-entropy | Minimum validation cross-entropy |

The scheduler reduces learning rate on a validation plateau. Feature means and standard deviations are fitted on the training partition only. The fixed split is train folds 1–3, validation fold 4, test fold 5. Each of five evaluation folds uses three training folds, the preceding fold for validation, and one test fold.

## Artifact integrity and interpretation

Release downloads include `SHA256SUMS.txt`. The full Colab bundle contains `package_manifest.json`, which records original immutable asset and feature-cache hashes. The notebook checks immutable assets before use and checks feature contracts against data, backbone, and frontend details. Git attributes preserve checksum-bearing dataset metadata, checkpoints, and result files byte for byte across platforms.

The application verifies deployed checkpoint hashes using [models/accuracy_v2/yamnet.json](../models/accuracy_v2/yamnet.json) and [models/transfer/efficientnet_b0.json](../models/transfer/efficientnet_b0.json). The two-model research heads use a separate checkpoint schema from the original YAMNet-only experiment; the EfficientNet deployment metadata records its source checkpoint.

The historical source signatures belong to the original training run and are preserved. Documentation updates and code cleanup are not presented as new training. Existing saved results have one seed, a previously observed fixed test, and historical fixed-validation model selection; they are not a fully nested or externally validated estimate. See the [README evaluation limits](../README.md#evaluation-limits).
