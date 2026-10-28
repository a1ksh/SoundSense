# SoundSense

GitHub: [a1ksh/SoundSense](https://github.com/a1ksh/SoundSense).

**Accuracy V2:** бастапқы модельдер сақталған; жаңа CNN augmentation және
ImageNet/YAMNet transfer тәжірибелері бөлек орындалады. Нақты нәтижелер:
[`reports/accuracy_v2/report_kk.md`](reports/accuracy_v2/report_kk.md),
[`methodology_kk.md`](reports/accuracy_v2/methodology_kk.md),
[`comparison.csv`](reports/accuracy_v2/comparison.csv),
[`5-fold CV`](reports/accuracy_v2/cross_validation_summary.json).
Орнату, бөлек TensorFlow ортасы, CUDA/CPU және толық қайталау командалары:
[`experiments/accuracy_v2/README.md`](experiments/accuracy_v2/README.md).
Қолданбада `CNN v1 (baseline)` және validation бойынша таңдалған `Best V2` бар.

| Модель | Fixed test дұрыс / барлығы | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| CNN v1 baseline | 67/80 | 83.75% | 0.8350 |
| CNN v2 augmentation | 60/80 | 75.00% | 0.7279 |
| EfficientNet-B0 Mel | 73/80 | 91.25% | 0.9114 |
| **YAMNet transfer — Best V2** | **78/80** | **97.50%** | **0.9739** |

Ресми 5-fold OOF: **YAMNet 391/400 = 97.75%**, fold accuracy
**97.75% ± 1.63 пайыздық тармақ**, aggregate macro-F1 **0.9775**;
EfficientNet 354/400 = 88.50% ± 2.40 пайыздық тармақ, macro-F1 0.8849.
± — бес fold бойынша sample SD, сенімділік интервалы емес. Fold 5 бұрын
белгілі болғандықтан fixed split exploratory; CV кандидаттары validation
арқылы таңдалған, толық nested model search емес. AudioSet pretraining
көздерінің ESC-пен ықтимал қабаттасуы тексерілген жоқ. Нәтиже ESC-10-ға
қатысты, белгісіз open-world дыбыстарға жалпылау дәлелденбеген.

Жергілікті CPU тексеруінде Best V2 бір клипті шамамен **42 секундта** өңдеді:
бөлек TensorFlow process-і әр талдауда іске қосылады. Бұл дәлдікке басымдық
берілген нұсқа; бастапқы CNN v1 таңдауы да сақталған. Уақыт журналы:
[`inference_verification.json`](reports/accuracy_v2/inference_verification.json).

Осы checkout-та `.venv` base Python жоғалғандықтан қалпына келтірілген runtime:

```powershell
.\.runtime\python311\python.exe app.py
.\.runtime\python311\python.exe -m pytest -q
.\.runtime\python311\python.exe -m experiments.accuracy_v2.run --yamnet-python .runtime/yamnet311/python.exe --cv-top 2
```

Төмендегі MLP/CNN кестесі — сақталған **V1 baseline** нәтижелері.

**Конволюциялық нейрондық желілер негізінде қоршаған орта дыбыстарын автоматты жіктеу жүйесін әзірлеу.** ESC-10 дыбыстарын 64×128 log-Mel арқылы жіктейтін MLP және ықшам CNN, нақты жергілікті оқыту, бағалау және Gradio қолданбасы.

## Нақты орындалған нәтиже

Құрылғы: **NVIDIA GeForce RTX 3050 Laptop GPU** (`cuda`), seed=42. Бастапқы fold: train=1–3 (240), validation=4 (80), test=5 (80). Бұл бір seed, бір held-out fold тәжірибесі; 5-fold CV емес.

| Модель | Параметр | Эпоха / үздік | Уақыт, с | Accuracy | Balanced acc. | Macro P | Macro R | Macro F1 | Macro ROC-AUC | Macro AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP | 1,057,610 | 19 / 12 | 6.55 | 0.6000 | 0.6000 | 0.6342 | 0.6000 | 0.5776 | 0.9351 | 0.6942 |
| CNN | 24,058 | 35 / 34 | 13.03 | 0.8375 | 0.8375 | 0.8494 | 0.8375 | 0.8350 | 0.9845 | 0.9296 |

Толық есеп: [курстық жұмыс](docs/coursework_report_kk.md), [әдіснама](docs/methodology_kk.md), [қайта өндіру](docs/reproducibility.md), [JSON нәтижелер](results/summary.json). Нақты pipeline/check мәртебесі: [pipeline_status.json](results/pipeline_status.json). GitHub және raw WAV мәртебесі: [delivery.json](results/delivery.json).

## Windows-та орнату

Python 3.11 ұсынылады. Барлық кітапхана осы жоба `.venv` ортасына орнатылады. RTX 3050 үшін тек ресми PyTorch CUDA wheel жеткілікті; толық CUDA Toolkit орнатылмайды.

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.\.venv\Scripts\python.exe -m soundsense.run_all
```

CPU үшін wheel индексін `https://download.pytorch.org/whl/cpu` деп ауыстырыңыз; құрылғы автоматты анықталады. Виртуалды ортаны қосқаннан кейін негізгі пәрмен: **`python -m soundsense.run_all`**. Pipeline деректі алады/тексереді → preprocessing → MLP → CNN → test inference → метрикалар → 20 сурет → есеп → pytest → қысқа жергілікті Gradio/API smoke test орындайды. `results/pipeline_status.json` success/failure жазады; қате болса process exit code=1.

## Қайта іске қосу және қолданба

```powershell
.\.venv\Scripts\python.exe -m soundsense.run_all --skip-download
.\.venv\Scripts\python.exe -m soundsense.run_all --force-train
.\.venv\Scripts\python.exe -m soundsense.visualize
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe app.py
```

Қалыпты іске қосу аяқталған жұмыстарды қайта қолданады; тест болжамы қайта есептелмейді. `--force-train` — саналы жаңа тәжірибе; бұрын көрілген fold-5 туралы audit сақталады. Қолданба `http://127.0.0.1:7860` адресінде ашылады (порт бос болмаса Gradio келесі бос портты таңдайды). **WAV/WAVE, MP3, M4A/AAC, FLAC, OGG/OGA, OPUS, AIFF/AIF, WMA, WEBM, MP4, CAF, AU** файлдарын жүктеуге болады; қазақша атау, top-3, waveform, Mel және өңделген дыбысты тыңдау бар. MP4/WEBM файлынан алғашқы аудиожол алынады. Алғашқы 5 секунд өңделеді, қысқа файл нөлмен толықтырылады; файл көлемі 100 МБ-қа дейін. FFmpeg `.venv` ішіндегі `imageio-ffmpeg` пакетімен орнатылады. [Форматтар және тексеру](docs/audio_formats_kk.md). Тек 10 санат; белгісіз дыбысты қате жіктейді; softmax калибрленген сенімділік емес.

Checkpoint файлдары: [`models/mlp_best.pt`](models/mlp_best.pt), [`models/cnn_best.pt`](models/cnn_best.pt). Қолданба dataset/cache талап етпейді, тек CNN checkpoint қажет. Тек сенімді жергілікті checkpoint жүктеңіз.

Қазақша интерфейс ашық фон, күлгін акценттер және қара-күлгін талдау карточкасы бар бақылау тақтасы түрінде жасалған. Мәзірден талдауға, графиктерге және нұсқаулыққа өтуге болады. [Интерфейс туралы](docs/interface_kk.md).

## Деректердің шығуы және лицензиясы

[Ресми ESC-50](https://github.com/karolpiczak/ESC-50) метадерегінен **тек 400 ESC-10 WAV** алынды; барлық 2000 аудио жүктелмейді. Upstream commit `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6` бекітілген. ESC-10 — **CC BY 3.0**; ESC-50 толық жиыны — CC BY-NC. Түпнұсқа [LICENSE](data/metadata/LICENSE), [400 файлдың авторлары мен Freesound сілтемелері](data/metadata/attributions.csv), [SHA-256 manifest](data/metadata/manifest.csv), [provenance](data/metadata/provenance.json) сақталған. Қайта таратқанда авторлар мен лицензия сақталуы тиіс. Raw файлдар LFS квотасы расталмаса Git-ке енгізілмейді; downloader оларды дәл қайта алады.

Дәйексөз: Piczak, K. J. (2015). *ESC: Dataset for Environmental Sound Classification*. ACM Multimedia, 1015–1018. https://doi.org/10.1145/2733373.2806390 .

## Графиктер (300 dpi)

- [audio_waveform_examples.png](results/figures/audio_waveform_examples.png)
- [class_distribution.png](results/figures/class_distribution.png)
- [cnn_accuracy_curve.png](results/figures/cnn_accuracy_curve.png)
- [cnn_confusion_matrix.png](results/figures/cnn_confusion_matrix.png)
- [cnn_confusion_matrix_normalized.png](results/figures/cnn_confusion_matrix_normalized.png)
- [cnn_loss_curve.png](results/figures/cnn_loss_curve.png)
- [cnn_prediction_confidence.png](results/figures/cnn_prediction_confidence.png)
- [error_analysis_examples.png](results/figures/error_analysis_examples.png)
- [mel_spectrogram_examples.png](results/figures/mel_spectrogram_examples.png)
- [mlp_accuracy_curve.png](results/figures/mlp_accuracy_curve.png)
- [mlp_confusion_matrix.png](results/figures/mlp_confusion_matrix.png)
- [mlp_confusion_matrix_normalized.png](results/figures/mlp_confusion_matrix_normalized.png)
- [mlp_loss_curve.png](results/figures/mlp_loss_curve.png)
- [mlp_precision_recall_per_class.png](results/figures/mlp_precision_recall_per_class.png)
- [mlp_roc_curves_per_class.png](results/figures/mlp_roc_curves_per_class.png)
- [model_metrics_comparison.png](results/figures/model_metrics_comparison.png)
- [models_training_comparison.png](results/figures/models_training_comparison.png)
- [per_class_f1_comparison.png](results/figures/per_class_f1_comparison.png)
- [precision_recall_per_class.png](results/figures/precision_recall_per_class.png)
- [roc_curves_per_class.png](results/figures/roc_curves_per_class.png)

Барлық қисық нақты history/болжамнан есептелген; сандық дерек `results/figure_data`, `results/metrics`, `results/history`, `results/predictions` ішінде. Әр тест санатына 8 мысал ғана; бір қате жалпы accuracy-ге 1.25 пайыздық тармақ әсер етеді. Басқа ортаға жалпылау, калибрлеу және сыртқы тест тексерілмеді. Модельді өмірге/қауіпсіздікке жауапты жүйе ретінде қолдану бағаланбаған.
