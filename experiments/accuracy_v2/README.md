# Preserved YAMNet experiment

This package trains and verifies the original YAMNet transfer-learning experiment. The current application supports **YAMNet and EfficientNet-B0**; the separate [two-model Colab workflow](../../colab_two_models/README.md) contains the full comparison.

YAMNet uses the official AudioSet-pretrained frozen backbone, mean-pooled 1024-dimensional embeddings, training-partition standardization, and a `Dropout(0.2) → Linear(1024, 10)` ESC-10 head. Five saved checkpoints correspond to five outer test folds. The deployed checkpoint is `models/accuracy_v2/yamnet_transfer_fixed.pt` (test fold 5).

## Verify saved results

Follow the [installation and asset instructions](../../docs/reproducibility.md), including the isolated TensorFlow environment, original audio, and verification feature cache.

```bash
python -m experiments.accuracy_v2.check
```

This verifies saved checkpoint/history hashes, regenerates the report, runs pytest, reconstructs 400 OOF predictions, tests live YAMNet worker reuse and cleanup, and checks both application models over the local HTTP API. It does not retrain models.

```bash
python -m experiments.accuracy_v2.report  # Regenerate the YAMNet report and 16 plots
python -m experiments.accuracy_v2.run     # Verify saved heads and regenerate results
```

For a new YAMNet-only experiment, use a new empty output directory and provide the isolated TensorFlow interpreter:

```powershell
python -m experiments.accuracy_v2.run --retrain --output-dir .runtime/new_yamnet_run --yamnet-python $env:SOUNDSENSE_YAMNET_PYTHON
```

Saved source signatures describe the original training execution; cleanup does not rewrite them or claim new training.

## Files

| File | Responsibility |
| --- | --- |
| `core.py` | Official splits, feature contracts, and YAMNet checkpoint validation |
| `run.py` | Head training, reuse validation, and evaluation |
| `extract_yamnet.py` | Label-free frozen embedding extraction |
| `inference.py` | YAMNet predictor with the native frontend |
| `worker_client.py`, `yamnet_worker.py` | Persistent isolated TensorFlow process with timeouts and cleanup |
| `verify.py` | Saved-score reconstruction, live inference, and application API checks |
| `report.py` | YAMNet-only metrics, reports, and plots |
| `check.py` | Comprehensive verification entry point |

This package accepts only the YAMNet checkpoint schema. The deployed EfficientNet predictor is [soundsense/efficientnet.py](../../soundsense/efficientnet.py), and the two-model research checkpoint schema belongs to [colab_two_models/workflow.py](../../colab_two_models/workflow.py).

Fixed train/validation/test folds: 1–3 / 4 / 5. Each outer fold uses the preceding fold for validation and the remaining three for training. Seed: 42. See [results](../../docs/results.md) and [evaluation limits](../../README.md#evaluation-limits).
