# ESC-10 dataset

SoundSense uses the official ESC-10 subset of [ESC-50](https://github.com/karolpiczak/ESC-50), pinned to upstream commit `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`.

## Composition and labels

There are **400 original five-second clips**, stored as mono 44.1 kHz WAV files. Each of ten classes has 40 clips, with eight clips per class in each of the five official folds. The complete ESC-50 metadata is retained for provenance; only rows marked `esc10=true` enter this project.

| Model index | ESC-50 target | Label | Meaning | Clips |
| ---: | ---: | --- | --- | ---: |
| 0 | 0 | `dog` | Dog barking | 40 |
| 1 | 1 | `rooster` | Rooster crowing | 40 |
| 2 | 10 | `rain` | Rain | 40 |
| 3 | 11 | `sea_waves` | Sea waves | 40 |
| 4 | 12 | `crackling_fire` | Crackling fire | 40 |
| 5 | 20 | `crying_baby` | Baby crying | 40 |
| 6 | 21 | `sneezing` | Sneezing | 40 |
| 7 | 38 | `clock_tick` | Clock ticking | 40 |
| 8 | 40 | `helicopter` | Helicopter | 40 |
| 9 | 41 | `chainsaw` | Chainsaw | 40 |

The authoritative class order is [artifacts/class_mapping.json](../artifacts/class_mapping.json). Kazakh display labels are defined in [soundsense/labels.py](../soundsense/labels.py).

## Splits and leakage checks

| Fixed partition | Official folds | Clips | Clips per class |
| --- | --- | ---: | ---: |
| Training | 1, 2, 3 | 240 | 24 |
| Validation | 4 | 80 | 8 |
| Test | 5 | 80 | 8 |

For five-fold evaluation, fold K is the test partition, fold K−1 is validation (fold 5 when K=1), and the remaining three folds are training. Each clip receives one out-of-fold prediction from a head that did not train on it. Fold 5 reuses the fixed checkpoint.

The downloader checks filename uniqueness, class/fold balance, and that the same `src_file` source recording never crosses fold boundaries. Feature standardization uses each training partition only. The explicit per-fold roles are in [cv_splits.csv](../colab_two_models/run/tables/cv_splits.csv).

## Download or use the supplied bundle

From the repository root, with the main Python environment activated:

```bash
python -m soundsense.download_data
```

This downloads only the 400 ESC-10 WAVs from the pinned upstream revision into `data/raw/`. It preserves the manifest and attribution, checks sample rate, channels, duration, finite samples, and SHA-256 hashes, and rejects unexpected non-ESC-10 WAVs in that folder. To validate a complete local copy without downloading:

```bash
python -m soundsense.download_data --skip-download
```

Alternatively, the [full trained Colab release bundle](https://github.com/a1ksh/SoundSense/releases/latest) contains all 400 WAVs under `assets/audio/`. For local application tests, copy those WAV files into `data/raw/`. Keep the bundle's metadata and original filenames alongside the audio when redistributing it.

## Provenance files

| File | Purpose |
| --- | --- |
| [source_lock.json](../data/metadata/source_lock.json) | Upstream repository and pinned commit |
| [esc50.csv](../data/metadata/esc50.csv) | Original upstream metadata |
| [manifest.csv](../data/metadata/manifest.csv) | Selected 400 clips, original folds, model indices, split roles, source URLs, and SHA-256 hashes |
| [provenance.json](../data/metadata/provenance.json) | Dataset identity, checksums, citation, and partition counts |
| [attributions.csv](../data/metadata/attributions.csv) | Original source identifiers and per-clip attribution |
| [LICENSE](../data/metadata/LICENSE) | Preserved upstream license and source attribution |
| [UPSTREAM_README.md](../data/metadata/UPSTREAM_README.md) | Preserved upstream dataset description |

## Model preprocessing

Both systems decode to mono, resample to 16 kHz using `soxr_hq`, and use 80,000 samples. Longer uploads use the first five seconds; shorter uploads are zero-padded. The stored original WAV files are unchanged.

YAMNet uses its official 64-band frontend (125–7500 Hz, 25 ms analysis window, 10 ms hop, natural logarithm of magnitude plus 0.001). It averages frame embeddings over time. The interface displays the native frontend spectrogram without resizing it to 128 frames.

EfficientNet-B0 uses the project's 64 × 128 power-dB Mel representation. Per-clip scaling maps its 80 dB range into an image, repeats it over three channels, resizes it to 224 × 224, and applies ImageNet normalization. This differs from Torchvision's standard photograph resize/crop recipe.

## License and citation

The ESC-10 subset is licensed under **Creative Commons Attribution 3.0**. Preserve the upstream license and per-clip credits. This statement applies to ESC-10; consult upstream licensing before using the complete ESC-50 collection.

Karol J. Piczak (2015). *ESC: Dataset for Environmental Sound Classification*. Proceedings of the 23rd ACM International Conference on Multimedia, pp. 1015–1018. [DOI: 10.1145/2733373.2806390](https://doi.org/10.1145/2733373.2806390).

```bibtex
@inproceedings{piczak2015esc,
  author = {Piczak, Karol J.},
  title = {ESC: Dataset for Environmental Sound Classification},
  booktitle = {Proceedings of the 23rd ACM International Conference on Multimedia},
  year = {2015},
  pages = {1015--1018},
  doi = {10.1145/2733373.2806390}
}
```
