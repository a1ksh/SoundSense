# Executed session and recovery notes

All training and evaluation ran locally in this Windows checkout on 2026-10-28;
no Colab or simulated metrics were used. The original 20 figures, checkpoints,
histories and reports were preserved. All 400 original WAV SHA-256 checks passed.

## Commands actually executed

```powershell
git rev-parse --show-toplevel
git status --short
git switch -c feature/accuracy-v2
nvidia-smi
.\.runtime\python311\python.exe -m pytest -q -p no:cacheprovider
.\.runtime\python311\python.exe -u -m experiments.accuracy_v2.run --models cnn_v2 efficientnet_b0_mel --train-only
.\.runtime\python311\python.exe -u -m experiments.accuracy_v2.download_tensorflow
.\.runtime\python311\python.exe -m pip install --timeout 60 --retries 2 --no-deps --target .runtime/yamnet-packages -r experiments/accuracy_v2/requirements-yamnet-components.txt
.\.runtime\python311\python.exe -m pip install --no-deps --target .runtime/yamnet-packages .runtime/tensorflow-2.20.0-cp311-cp311-win_amd64.whl
.\.runtime\python311\python.exe -u -m experiments.accuracy_v2.run --models yamnet_transfer --yamnet-python .runtime/yamnet311/python.exe --train-only
.\.runtime\python311\python.exe -m experiments.accuracy_v2.provenance
.\.runtime\python311\python.exe -u -m experiments.accuracy_v2.run --yamnet-python .runtime/yamnet311/python.exe --cv-top 2
.\.runtime\python311\python.exe -u -m experiments.accuracy_v2.check
.\.runtime\python311\python.exe -m experiments.accuracy_v2.report
.\.runtime\python311\python.exe -m experiments.accuracy_v2.delivery --access-only
```

Final `check` executed pytest and `experiments.accuracy_v2.verify` as actual
subprocesses. The latter launched and closed a local Gradio server, called both
model selections and the default API over HTTP, and checked all five outputs.
The unit test result is **48 passed, zero skipped**. Original tests separately
passed **32/32** before implementation. The preflight log's one skip predates
YAMNet training and is superseded by the final complete test run.

## Concrete recovery events

- Original `.venv/Scripts/python.exe` could not find its Python 3.11 base
  installation. The old environment was preserved. Official Python 3.11.9
  embeddable runtime plus existing compatible packages restored local execution;
  its `.pth` project path needed an absolute path after initial collection errors.
- Default sandbox writes to `.git` and network downloads required tool
  escalation; the feature branch and authorized downloads succeeded.
- TensorFlow's initial pip/single-connection download stalled. The ranged
  download encountered DNS failures, retained completed ranges, then resumed
  with one DNS lookup and verified the complete wheel against official PyPI
  SHA-256. Initial failures and successful retry logs are retained separately.
- Small dependency download initially timed out; its bounded retry succeeded.
  Pip warned that the shared target `bin` directory existed; TensorFlow imports
  and actual extraction/inference verified the installed libraries. No original
  `.venv` packages were replaced. TF/Hub emitted upstream deprecation warnings;
  those do not indicate failed training.
- Before test access, experiment signatures were corrected to ignore elapsed
  extraction time and Windows source line endings. Preliminary E1/E2 artifacts
  were archived, then identical recipes rerun. No test tuning was performed.
- GitHub connector returned 404 for this repository. Existing Git credentials
  verified the configured remote; curl-based GitHub API access verified push
  permission after a urllib request timed out. Credentials stayed in memory.

## Results and limits

Validation selected YAMNet before new test access. Fixed fold 5: CNN v1
67/80, CNN v2 60/80, EfficientNet 73/80, YAMNet 78/80. CNN augmentation did not
improve this run. OOF YAMNet: 391/400; EfficientNet: 354/400. CV used fixed
recipes, seed 42, official source-separated folds and training-only scalers.

YAMNet end-to-end CPU inference measured 41.885 seconds for one training clip,
including a fresh TensorFlow subprocess. Each app inference currently restarts
that process. The baseline selector is retained. These sequential one-clip
timings include differing framework/JIT startup and are not a controlled speed
benchmark. See the Kazakh report for the small-data, exploratory-validation and
external-pretraining overlap limitations.

No unresolved training, evaluation, test or app-smoke blocker remains. The old
`.venv` launcher still needs its missing base installation if used directly;
the documented `.runtime/python311/python.exe` command works in this checkout.
Runtime folders, raw WAVs and caches are excluded from Git. Final remote commit
and draft-PR verification are recorded in ignored `delivery.local.json`.
