# SoundSense: YAMNet and EfficientNet-B0 in Google Colab

This folder contains the complete two-model ESC-10 experiment: an executed notebook, its Python workflow, ten trained classification heads, predictions, epoch histories, metrics, and 58 figures in PNG, SVG, and PDF.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a1ksh/SoundSense/blob/main/colab_two_models/SoundSense_YAMNet_EfficientNet_Full_Colab.ipynb)

## Run the notebook

1. Download `SoundSense_2Models_Trained_Colab.zip` from the [latest GitHub release](https://github.com/a1ksh/SoundSense/releases/latest). Access requires permission to this private repository.
2. Open [the notebook](SoundSense_YAMNet_EfficientNet_Full_Colab.ipynb). For a private repository, downloading it and choosing **Colab → File → Upload notebook** is the most direct option.
3. Select a GPU runtime. The notebook setup supports Python 3.10–3.12 and originally targets Colab's 2026.04 / Python 3.12 runtime; availability of older runtimes can change. See [Colab runtime versions](https://research.google.com/colaboratory/runtime-version-faq.html).
4. In the first code cell, choose `RUN_MODE = 'use_trained'` to verify the supplied heads and regenerate results, or `'retrain'` to train new heads. **The default is `'retrain'`.**
5. Leave `USE_BUNDLE = True`, run all cells, and upload the full trained bundle when requested.
6. Download `SoundSense_2Models_Results.zip` at the end.

PyTorch runs in the notebook environment. The notebook creates a separate TensorFlow environment for YAMNet and performs real waveform-to-prediction inference for both systems. GPU acceleration is optional for the classification heads; YAMNet feature extraction uses a separate CPU process.

## Notebook controls

| Setting | Meaning |
| --- | --- |
| `RUN_MODE = 'retrain'` | Train both models' heads using frozen features |
| `RUN_MODE = 'use_trained'` | Validate and reuse the supplied checkpoints |
| `USE_BUNDLE = True` | Extract the supplied dataset, backbones, feature caches, and results |
| `USE_BUNDLE = False` | Download the pinned original dataset and official pretrained backbones |
| `BUNDLE_PATH` | Optional path to a bundle already uploaded to the runtime |
| `RECOMPUTE_FEATURES = True` | Re-extract features; requires `RUN_MODE = 'retrain'` |
| `DOWNLOAD_RESULTS = True` | Download the results archive when execution finishes |

For a new experiment, use a fresh runtime or output folder. Notebook cells can replace results under their working `run/` directory; keep the original release bundle unchanged for comparison.

## Full bundle contents

| Path inside ZIP | Contents |
| --- | --- |
| `assets/audio/` | All 400 original ESC-10 WAV files |
| `assets/metadata/` | Pinned manifest, clip hashes, upstream license, and attribution |
| `assets/backbones/` | Official frozen YAMNet and EfficientNet-B0 weights |
| `assets/features/` | 1024/1280-dimensional embeddings with dataset/backbone/frontend contracts |
| `assets/licenses/` | Third-party licenses and notices |
| `run/models/` | Five heads per model; ten total |
| `run/history/` | Real epoch loss/accuracy histories and checkpoint metadata |
| `run/predictions/` | Fixed-test, fold, and OOF predictions, class scores, and errors |
| `run/metrics/` | Classification reports, confusion matrices, ROC and PR data |
| `run/architecture/` | Backbone stage/channel/stride/repeat tables |
| `run/figures/` | 58 figures, each as 300 dpi PNG, SVG, and PDF |
| `run/tables/` | Model comparison, splits, and figure inventory |
| `package_manifest.json` | Immutable asset SHA-256 hashes and feature cache hashes |

The source repository keeps metadata, trained heads, and research outputs. Large audio, backbone, and feature-cache copies for this notebook are supplied through the release bundle. The main application's EfficientNet backbone is also stored under `models/transfer/`.

The smaller `SoundSense_2Models_Results.zip` contains the research outputs and trained heads. It does not contain everything required for waveform inference or a complete rerun; use the full bundle for those tasks.

## Method and results

Both systems are deep CNN transfer-learning models with frozen backbones. Only `Dropout(0.2) → Linear(d, 10)` is trained after train-only feature standardization. YAMNet uses AudioSet-pretrained MobileNetV1 embeddings (`d=1024`, 10,250 head parameters); EfficientNet-B0 uses ImageNet-pretrained features (`d=1280`, 12,810 head parameters).

Seed: 42. Fixed train/validation/test: folds 1–3 / 4 / 5 (240 / 80 / 80 clips). For five-fold evaluation, test fold K uses validation fold K−1, with fold 5 validating test fold 1. The scaler is fitted to each training partition, and minimum validation cross-entropy selects the checkpoint.

| Model | Fixed accuracy | Fixed macro F1 | OOF accuracy | OOF macro F1 |
| --- | ---: | ---: | ---: | ---: |
| YAMNet transfer | 97.50% | 0.9739 | 97.75% | 0.9775 |
| EfficientNet-B0 transfer | 91.25% | 0.9114 | 88.50% | 0.8849 |

See [the complete results and figure gallery](../docs/results.md), [the dataset guide](../docs/dataset.md), and the supplementary [Kazakh Colab guide](README_kk.md).

## Execution provenance and limits

All saved code-cell outputs come from an actual local GPU execution. They were not produced on Google's hosted Colab servers. This is recorded in notebook metadata and [run/environment.json](run/environment.json). [run/checks.json](run/checks.json) records the original automated checks; it is not a claim that hosted Colab execution has been verified.

There is one training seed and a small dataset. Earlier model selection used fixed validation results, and the fixed test was already known; this is not fully nested model selection. Pretraining domains and frontends differ. Overlapping training folds, unexamined pretraining overlap, absence of external testing, and uncalibrated scores limit interpretation. Fold standard deviation is not a confidence interval.
