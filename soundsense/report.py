"""Generate Kazakh result sections directly from measured machine-readable artifacts."""
from .common import ROOT, read_json


def write_reports():
    summary = read_json(ROOT / 'results/summary.json')
    metadata = read_json(ROOT / 'results/run_metadata.json')
    inventory = read_json(ROOT / 'results/figures/inventory.json')
    docs = ROOT / 'docs'
    docs.mkdir(exist_ok=True)
    rows = []
    for name in ['mlp', 'cnn']:
        model = summary['models'][name]
        metrics, training = model['metrics'], model['training']
        rows.append(f'| {name.upper()} | {training["parameter_count"]:,} | {training["epochs_completed"]} / {training["best_epoch"]} | '
                    f'{training["training_seconds"]:.2f} | {metrics["accuracy"]:.4f} | {metrics["balanced_accuracy"]:.4f} | '
                    f'{metrics["macro_precision"]:.4f} | {metrics["macro_recall"]:.4f} | {metrics["macro_f1"]:.4f} | '
                    f'{metrics["macro_roc_auc_ovr"]:.4f} | {metrics["macro_average_precision"]:.4f} |')
    table = ('| Модель | Параметр | Эпоха / үздік | Уақыт, с | Accuracy | Balanced acc. | Macro P | Macro R | Macro F1 | Macro ROC-AUC | Macro AP |\n'
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n' + '\n'.join(rows))
    cnn, mlp = summary['models']['cnn'], summary['models']['mlp']
    errors = round((1 - cnn['metrics']['accuracy']) * 80)
    difference = (cnn['metrics']['accuracy'] - mlp['metrics']['accuracy']) * 100
    provenance = read_json(ROOT / 'data/metadata/provenance.json')
    class_report = read_json(ROOT / 'results/metrics/cnn_classification_report.json')
    mapping = read_json(ROOT / 'artifacts/class_mapping.json')
    ranking = sorted((class_report[c['label']]['f1-score'], c['label']) for c in mapping)
    low_f1 = ', '.join(f'{label}: {score:.3f}' for score, label in ranking[:3])
    high_f1 = ', '.join(f'{label}: {score:.3f}' for score, label in ranking[-3:][::-1])
    figure_links = '\n'.join(f'- [{item["file"].split("/")[-1]}](../{item["file"]})' for item in inventory)
    citations = '''1. Piczak, K. J. (2015). *ESC: Dataset for Environmental Sound Classification*. Proceedings of the 23rd ACM International Conference on Multimedia, 1015–1018. https://doi.org/10.1145/2733373.2806390
2. ESC-50 ресми репозиторийі және ESC-10 ішкі жиыны: https://github.com/karolpiczak/ESC-50 . Бастапқы жазбалардың авторлары, Freesound сілтемелері және лицензиялары: [LICENSE](../data/metadata/LICENSE), [400 файлдың атрибуциясы](../data/metadata/attributions.csv).
3. PyTorch құжаттамасы, нейрондық желілер мен оңтайландыру: https://docs.pytorch.org/docs/stable/index.html . Ресми CUDA дистрибутивтері: https://pytorch.org/get-started/previous-versions/ .
4. librosa, Mel-спектрограмма: https://librosa.org/doc/0.11.0/generated/librosa.feature.melspectrogram.html .
5. scikit-learn, жіктеу метрикалары: https://scikit-learn.org/stable/modules/model_evaluation.html .
6. Matplotlib құжаттамасы: https://matplotlib.org/stable/ ; Gradio құжаттамасы: https://www.gradio.app/docs .'''
    report = f'''# Конволюциялық нейрондық желілер негізінде қоршаған орта дыбыстарын автоматты жіктеу жүйесін әзірлеу

## Кіріспе және өзектілік

Қоршаған орта дыбыстары оқиға туралы көрініссіз ақпарат береді. Иттің үруі, жаңбыр немесе тікұшақ дыбысын ажырату акустикалық бақылау мен қолжетімді интерфейстерге негіз бола алады. Алайда шағын оқу жиынындағы нәтиже өндірістік сенімділікті дәлелдемейді. Осы жұмыстың мақсаты — ашық дереккөзге сүйеніп, дыбыс жіктеудің тексерілетін оқу тәжірибесін және шағын демонстрациялық қолданбасын әзірлеу.

Міндеттер: ESC-10 деректерін лицензиясымен алу; дыбысты бірдей log-Mel кескініне түрлендіру; MLP және CNN модельдерін оқыту; бөлек бақылау жиынында салыстыру; нақты нәтижелерден графиктер мен есеп құру. Бұл жұмыс жаңа архитектура немесе бұрын жарияланған көрсеткішті қайталау туралы мәлімдеме жасамайды.

## Теориялық негіз

Дыбыс уақыт бойынша амплитудамен беріледі. Қысқа уақыттық Фурье түрлендіруі оның жергілікті жиілік құрамын ашады. Mel сүзгілері спектрді есту сезімталдығына жақын жолақтарға біріктіреді, ал логарифмдік шкала қуат ауқымын ықшамдайды. MLP барлық ұяшықты жалпақ вектор ретінде қабылдайды. CNN жергілікті уақыт-жиілік өрнектерін ортақ сүзгілермен іздейді. Негізгі ұғымдар [әдіснамада](methodology_kk.md) толық түсіндірілген.

## Деректер және адал бөлу

Ресми ESC-50 метадеректеріндегі `esc10=True` жолдары ғана таңдалды: 400 бірегей, бес секундтық, 44.1 кГц моно WAV; он санат, әр санатта 40 жазба. Бүкіл ESC-50 аудиосы жүктелмейді. Бекітілген upstream commit: `{provenance['commit']}`. ESC-10 үшін CC BY 3.0 қолданылады; ESC-50 толық жиынының лицензиясы бөлек, CC BY-NC. Түпнұсқа LICENSE пен жеке авторлар сақталды.

Оқыту — бастапқы 1, 2, 3 fold (240); валидация — 4-fold (80); қорытынды тест — 5-fold (80). Әр тест санатында сегіз мысал бар. Файл атаулары да, `src_file` бастапқы жазба идентификаторлары да бөліктер арасында қиылыспайды. Кездейсоқ аудиофайл бөлу қолданылған жоқ. Модель нұсқасы тек валидация шығыны бойынша таңдалды. Тест болжамдары таңдау аяқталғаннан кейін сақталады; есепті қайта құру осы болжамдарды пайдаланады.

## Алдын ала өңдеу

Моноға келтіру → `soxr_hq` арқылы 16 кГц-ке қайта дискреттеу → алғашқы 5 секундты алу немесе оң жақтан нөлмен толықтыру (80 000 үлгі) → Hann терезесі, FFT=1024, қадам=512, `center=True`, қуат дәрежесі 2 → 0–8000 Гц, 64 Slaney Mel жолағы → `power_to_db(ref=1.0, top_db=80)` → бүкіл уақыт осін сызықтық интерполяциямен 128 кадрға келтіру. Нәтиже 64×128. Жиілік осі қайта өлшемделмейді. Жеке файлға максимум бойынша амплитуда қалыптау жүргізілмейді.

Стандарттау үшін тек 240 оқу мысалының барлық ұяшықтары бойынша бір орташа және стандарттық ауытқу есептелді: mean={metadata['statistics']['mean']:.8f}, std={metadata['statistics']['std']:.8f}. Валидация мен тест бұл статистикаға кірмейді. Қолданба checkpoint ішіндегі осы параметрлерді және дәл осы функцияларды қолданады. Аугментация және алдын ала оқытылған салмақтар қолданылмады.

## Архитектура және тәжірибе

MLP: Flatten(8192) → Linear(128) → ReLU → Dropout(0.4) → Linear(64) → ReLU → Dropout(0.4) → Linear(10).

CNN: үш Conv2D(3×3, padding=1, bias=False) → BatchNorm → ReLU → MaxPool(2) блогы; арналар 16, 32, 64; соңында global average pooling → Dropout(0.3) → Linear(10). CrossEntropyLoss кірісіне softmax емес, logits беріледі; softmax тек болжам ықтималдығын шығару үшін қолданылады.

Нақты құрылғы: **{metadata['device_name']}** (`{metadata['device']}`). Python {metadata['python']}, PyTorch {metadata['packages']['torch']}. Seed=42, AdamW, learning rate=0.001, weight decay=0.0001, бастапқы batch=16, ең көбі 35 эпоха, validation loss жақсармайтын 7 эпохадан кейін тоқтау. Детерминдік алгоритмдер қосылған. Нақты batch пен әр эпоханың өлшемдері `results/history/` ішінде. Екі модель де кездейсоқ бастапқы салмақтан жеке seed=42 арқылы оқытылды.

## Нақты қорытынды нәтижелер

{table}

CNN тестте {80-errors}/80 жазбаны дұрыс жіктеді, {errors} қате жіберді. CNN мен MLP accuracy айырмасы {difference:+.2f} пайыздық тармақ. Бұл бір seed пен бір бақылау fold бойынша сипаттамалық салыстыру; статистикалық басымдықтың дәлелі емес. Balanced accuracy — санаттар бойынша recall орташа мәні; тең санды сыныптарда оның accuracy-мен сәйкес болуы күтіледі. Толық macro/weighted precision, recall, F1 және әр санаттың support мәні [сандық метрикаларда](../results/metrics/) сақталған.

CNN үшін F1 ең төмен үш санат: **{low_f1}**. F1 ең жоғары үш санат: **{high_f1}**. Бұл сегіз мысалдан алынған санаттық бағалар, сондықтан бір қосымша қате олардың ретін өзгерте алады.

ROC әр санатты қалған тоғызына қарсы бағалайды. Әр санатта 8 оң, 72 теріс мысал болғандықтан осы тәжірибедегі ROC-AUC/AP анықталған. Macro ROC-AUC — он жеке AUC орташа мәні. AP — non-interpolated average precision; оны трапециялық PR-AUC деп атауға болмайды. Precision бөлімінде нөлдік бөлгіш болса, 0 мәні және тиісті саясат метадеректе жазылады.

## Суреттерді талдау

Санаттар диаграммасы әр санатқа train/validation/test=24/8/8 бөлінгенін көрсетеді. Толқын және Mel мысалдары тек оқу бөлігінен алынған, спектрограмманы суреттеу жіктеу дәлдігін өздігінен дәлелдемейді. Оқыту қисықтарындағы тік сызық ең төмен валидация шығыны бар эпоханы белгілейді; соңғы эпоха міндетті түрде ең үздік емес.

Training метрикалары салмақ жаңару барысында, Dropout белсенді `train()` режимінде есептеледі. Validation бір эпохадан кейін тұрақты салмақпен `eval()` режимінде есептеледі; кей эпохада validation accuracy training accuracy-ден жоғары болуы осы режим айырмасымен үйлеседі және өздігінен leakage дәлелі емес. MLP-де үздік эпохадан кейін validation loss төмендемей, patience бойынша тоқтады; CNN-де ең жақсы validation loss 34-эпохада болды. Жақсы AUC пен төменірек argmax accuracy қатар кездесуі рейтингтің сапасы мен соңғы санат шешімінің бөлек өлшем екенін көрсетеді.

Confusion matrix ұяшықтары — нақты fold-5 болжамдарының саны. Жол бойынша қалыпталған нұсқада диагональ әр санаттың recall мәніне тең. Per-class F1 диаграммасы екі модельдің санатқа тәуелді айырмасын көрсетеді. ROC және PR графиктері ықтималдықтардың рейтингін бағалайды; accuracy мен F1 argmax шешімін бағалайды. Қате талдауы manifest ретіндегі алғашқы алты CNN қатесін көрсетеді; олар қолмен ойдан таңдалған мысалдар емес. Confidence гистограммасы — ең үлкен softmax мәндерінің таралуы, калибрленген сенім емес.

Барлық сурет 300 dpi PNG түрінде жасалды. Олардың сандық көздері — `results/history/`, `results/metrics/`, `results/predictions/`, `results/figure_data/`. Әр санаттың нақты жазбасы мен есептелген Mel ұяшықтары да CSV түрінде берілген.

{figure_links}

## Қолданба

`python app.py` тек сақталған CNN checkpoint жүктейді. Пайдаланушы аудиофайл таңдап, қазақша/ағылшынша санатты, үздік үш санат ықтималдығын, толқын пішінін және Mel-спектрограмманы көреді. Ұзақ файлдың алғашқы бес секунды ғана өңделеді. «Unknown» деген оқытылмаған он бірінші санат енгізілмеді. Интерфейс 10 санат шегін және softmax калибрленбегенін көрсетеді.

## Шектеулер

Бұл **single-seed, one held-out fold** зерттеуі; 5-fold қайталанған бағалау емес. 80 тест мысалы ғана бар: бір қате жалпы accuracy-ді 1.25 пайыздық тармаққа, бір санат recall-ын 12.5 тармаққа өзгертеді. Тест үлгілері шағын және бір бастапқы жазбадан бірнеше үзінді болуы мүмкін, сондықтан оларды тәуелсіз бақылау деп алып тар сенім аралығын есептеу дұрыс емес. Басқа құрылғы, бөлме, шу деңгейі және белгісіз санаттар тексерілген жоқ. ESC-50 авторлары бастапқы файлдардың санатқа тәуелді өңделуі (мысалы, жолақ шектеуі) қосымша ақпарат беріп қоюы мүмкін екенін ескертеді. Бір компьютердегі детерминдік баптау өзге кітапхана немесе GPU-де биттік бірдей нәтижені кепілдемейді. Бұл модельді қауіпсіздікке жауапты дыбыстық ескерту жүйесі ретінде қабылдауға болмайды.

## Қорытынды

Түпнұсқа fold бөлінуі сақталған жергілікті эксперимент орындалды. Екі архитектураның нәтижесі нақты checkpoint болжамдарынан есептелді. Салыстыру, CSV/JSON метрикалар, 300 dpi графиктер және оқытусыз жұмыс істейтін қолданба бір pipeline арқылы қайта құрылады. Жұмыстың нәтижесі — өлшенген оқу тәжірибесі; кең ортаға жалпылау туралы тұжырым үшін қосымша fold, seed және сыртқы дерек қажет.

## Әдебиеттер

{citations}
'''
    (docs / 'coursework_report_kk.md').write_text(report, encoding='utf-8')
    versions = '\n'.join(f'- {k}: `{v}`' for k, v in metadata['packages'].items())
    reproducibility = f'''# Қайта өндіру / Reproducibility

## Нақты орындалған тәжірибе

- UTC басталу уақыты: `{metadata['started_utc']}`.
- Құрылғы: `{metadata['device_name']}` / `{metadata['device']}`; CUDA runtime: `{metadata['cuda_runtime']}`.
- Python: `{metadata['python']}`; платформа: `{metadata['platform']}`.
- Seed: 42; детерминдік алгоритмдер қосылған; Windows num_workers=0.
- Жұмыс басталған Git SHA: `{metadata['git_commit_at_run_start']}` (implementation жұмыс ағашында болды; final commit үшін `git log -1`).
- Нақты бастапқы конфигурация: [run metadata](../results/run_metadata.json); файл/код қолтаңбалары checkpoint пен training JSON ішінде.
- ESC-10 upstream SHA: `{provenance['commit']}`; 400 SHA-256: [manifest](../data/metadata/manifest.csv).

{versions}

## Windows PowerShell

```powershell
& "$env:LOCALAPPDATA\\Programs\\Python\\Python311\\python.exe" -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe -m soundsense.run_all
```

CPU баламасы: алғашқы pip жолында `https://download.pytorch.org/whl/cpu` қолданыңыз. CUDA Toolkit бөлек орнату қажет емес. Нақты орындалған орта [requirements-lock.txt](../requirements-lock.txt) ішінде; сол нұсқаларды қалпына келтіру үшін CUDA wheel орнатқан соң `pip install -r requirements-lock.txt` қолданыңыз. Басқа платформада `+cu126` wheel сәйкес келмеуі мүмкін, сондықтан `requirements.txt` пен құрылғыға сәйкес ресми wheel пайдаланылады.

```powershell
.\\.venv\\Scripts\\python.exe -m soundsense.run_all --skip-download
.\\.venv\\Scripts\\python.exe -m soundsense.visualize
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\python.exe -m soundsense.smoke
.\\.venv\\Scripts\\python.exe app.py
```

Қалыпты қайта іске қосу дайын download, feature cache, аяқталған training және test predictions қайта қолданады. Код/баптау/дерек өзгерсе checkpoint автоматты үнсіз қайта оқытылмайды; `--force-train` керек. Жаңа тәжірибе үшін `python -m soundsense.run_all --force-train` барлық модельді seed-тен бастайды. Осыдан кейін fold-5 қайта қаралатыны audit файлында жазылады; бұл тәуелсіз жаңа тест болмайды. Үзілген training аяқталған деп саналмайды, сол модель seed-тен қайта басталады; epoch деңгейіндегі optimizer resume қолданылмайды. Жүктеу жеке файл бойынша қалпына келеді: жарамды файл сақталады, толық түспегені `.part` арқылы қайта жүктеледі.

Windows-та Git жол соңын LF/CRLF түріне ауыстыруы мүмкін. Код fingerprint-тері LF түріне келтіріліп есептеледі; сақталған history/prediction CSV және checkpoint файлдары Git-ке байттары өзгермей енгізілген. `python tools/check_git_artifacts.py` Git индексіндегі файлдарды Windows checkout сүзгісі арқылы тексеріп, екі training resume қолтаңбасын салыстырады. Нақты тексеру: `results/git_checkout_verification.json`.

## Артефактілер

- `models/mlp_best.pt`, `models/cnn_best.pt`: ең төмен validation loss салмақтары, preprocessing, mapping, статистика.
- `artifacts/preprocessing_stats.json`: тек train статистикасы.
- `results/history/*.csv`: нақты эпохалар; `*_training.json`: duration, batch, параметр саны, checksum.
- `results/predictions/*_test.csv`: 80 fold-5 жазбасы, 10 ықтималдық; `test_access_audit.json`: тест inference журналы.
- `results/metrics/`: per-class, macro/weighted, ROC/PR нүктелері, confusion matrix.
- `results/summary.json`, `results/model_comparison.csv`: модельдерді салыстыру.
- `results/figures/inventory.json`: әр PNG өлшемі, 300 dpi және SHA-256.
- `results/pipeline_status.json`: кезеңдер, exit нәтижесі; `results/pytest.txt`, `results/gradio_smoke.json`: нақты тексерулер.

Raw WAV Git LFS-пен тек квота/рұқсат расталса жіберіледі. Әйтпесе local қалады; downloader, manifest және атрибуция репозиторийге кіреді. Нақты push күйі [delivery.json](../results/delivery.json) файлында көрсетіледі.
'''
    (docs / 'reproducibility.md').write_text(reproducibility, encoding='utf-8')
    figure_readme = '\n'.join(f'- [{item["file"].split("/")[-1]}]({item["file"]})' for item in inventory)
    readme = f'''# SoundSense

**Конволюциялық нейрондық желілер негізінде қоршаған орта дыбыстарын автоматты жіктеу жүйесін әзірлеу.** ESC-10 дыбыстарын 64×128 log-Mel арқылы жіктейтін MLP және ықшам CNN, нақты жергілікті оқыту, бағалау және Gradio қолданбасы.

## Нақты орындалған нәтиже

Құрылғы: **{metadata['device_name']}** (`{metadata['device']}`), seed=42. Бастапқы fold: train=1–3 (240), validation=4 (80), test=5 (80). Бұл бір seed, бір held-out fold тәжірибесі; 5-fold CV емес.

{table}

Толық есеп: [курстық жұмыс](docs/coursework_report_kk.md), [әдіснама](docs/methodology_kk.md), [қайта өндіру](docs/reproducibility.md), [JSON нәтижелер](results/summary.json). Нақты pipeline/check мәртебесі: [pipeline_status.json](results/pipeline_status.json). GitHub және raw WAV мәртебесі: [delivery.json](results/delivery.json).

## Windows-та орнату

Python 3.11 ұсынылады. Барлық кітапхана осы жоба `.venv` ортасына орнатылады. RTX 3050 үшін тек ресми PyTorch CUDA wheel жеткілікті; толық CUDA Toolkit орнатылмайды.

```powershell
& "$env:LOCALAPPDATA\\Programs\\Python\\Python311\\python.exe" -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.\\.venv\\Scripts\\python.exe -m soundsense.run_all
```

CPU үшін wheel индексін `https://download.pytorch.org/whl/cpu` деп ауыстырыңыз; құрылғы автоматты анықталады. Виртуалды ортаны қосқаннан кейін негізгі пәрмен: **`python -m soundsense.run_all`**. Pipeline деректі алады/тексереді → preprocessing → MLP → CNN → test inference → метрикалар → 20 сурет → есеп → pytest → қысқа жергілікті Gradio/API smoke test орындайды. `results/pipeline_status.json` success/failure жазады; қате болса process exit code=1.

## Қайта іске қосу және қолданба

```powershell
.\\.venv\\Scripts\\python.exe -m soundsense.run_all --skip-download
.\\.venv\\Scripts\\python.exe -m soundsense.run_all --force-train
.\\.venv\\Scripts\\python.exe -m soundsense.visualize
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\python.exe app.py
```

Қалыпты іске қосу аяқталған жұмыстарды қайта қолданады; тест болжамы қайта есептелмейді. `--force-train` — саналы жаңа тәжірибе; бұрын көрілген fold-5 туралы audit сақталады. Қолданба `http://127.0.0.1:7860` адресінде ашылады (порт бос болмаса Gradio келесі бос портты таңдайды). WAV/қолдау көрсетілетін аудио жүктеу; қазақша атау, top-3, waveform, Mel. Тек 10 санат; белгісіз дыбысты қате жіктейді; softmax калибрленген сенімділік емес. Алғашқы 5 секунд алынады, қысқа файл нөлмен толықтырылады.

Checkpoint файлдары: [`models/mlp_best.pt`](models/mlp_best.pt), [`models/cnn_best.pt`](models/cnn_best.pt). Қолданба dataset/cache талап етпейді, тек CNN checkpoint қажет. Тек сенімді жергілікті checkpoint жүктеңіз.

## Деректердің шығуы және лицензиясы

[Ресми ESC-50](https://github.com/karolpiczak/ESC-50) метадерегінен **тек 400 ESC-10 WAV** алынды; барлық 2000 аудио жүктелмейді. Upstream commit `{provenance['commit']}` бекітілген. ESC-10 — **CC BY 3.0**; ESC-50 толық жиыны — CC BY-NC. Түпнұсқа [LICENSE](data/metadata/LICENSE), [400 файлдың авторлары мен Freesound сілтемелері](data/metadata/attributions.csv), [SHA-256 manifest](data/metadata/manifest.csv), [provenance](data/metadata/provenance.json) сақталған. Қайта таратқанда авторлар мен лицензия сақталуы тиіс. Raw файлдар LFS квотасы расталмаса Git-ке енгізілмейді; downloader оларды дәл қайта алады.

Дәйексөз: Piczak, K. J. (2015). *ESC: Dataset for Environmental Sound Classification*. ACM Multimedia, 1015–1018. https://doi.org/10.1145/2733373.2806390 .

## Графиктер (300 dpi)

{figure_readme}

Барлық қисық нақты history/болжамнан есептелген; сандық дерек `results/figure_data`, `results/metrics`, `results/history`, `results/predictions` ішінде. Әр тест санатына 8 мысал ғана; бір қате жалпы accuracy-ге 1.25 пайыздық тармақ әсер етеді. Басқа ортаға жалпылау, калибрлеу және сыртқы тест тексерілмеді. Модельді өмірге/қауіпсіздікке жауапты жүйе ретінде қолдану бағаланбаған.
'''
    (ROOT / 'README.md').write_text(readme, encoding='utf-8')
