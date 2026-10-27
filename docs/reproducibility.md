# Қайта өндіру / Reproducibility

## Нақты орындалған тәжірибе

- UTC басталу уақыты: `2026-10-27T15:12:33.020356+00:00`.
- Құрылғы: `NVIDIA GeForce RTX 3050 Laptop GPU` / `cuda`; CUDA runtime: `12.6`.
- Python: `3.11.9`; платформа: `Windows-10-10.0.26300-SP0`.
- Seed: 42; детерминдік алгоритмдер қосылған; Windows num_workers=0.
- Жұмыс басталған Git SHA: `091105bbdd577d23a3d1e7f7f255b4bdbab21d76` (implementation жұмыс ағашында болды; final commit үшін `git log -1`).
- Нақты бастапқы конфигурация: [run metadata](../results/run_metadata.json); файл/код қолтаңбалары checkpoint пен training JSON ішінде.
- ESC-10 upstream SHA: `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`; 400 SHA-256: [manifest](../data/metadata/manifest.csv).

- torch: `2.7.1+cu126`
- numpy: `2.2.6`
- pandas: `2.3.3`
- librosa: `0.11.0`
- soundfile: `0.13.1`
- scikit-learn: `1.7.2`
- matplotlib: `3.10.7`
- seaborn: `0.13.2`
- scipy: `1.16.3`
- gradio: `5.49.1`
- tqdm: `4.67.1`
- pyyaml: `6.0.3`
- pytest: `8.4.2`

## Windows PowerShell

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m soundsense.run_all
```

CPU баламасы: алғашқы pip жолында `https://download.pytorch.org/whl/cpu` қолданыңыз. CUDA Toolkit бөлек орнату қажет емес. Нақты орындалған орта [requirements-lock.txt](../requirements-lock.txt) ішінде; сол нұсқаларды қалпына келтіру үшін CUDA wheel орнатқан соң `pip install -r requirements-lock.txt` қолданыңыз. Басқа платформада `+cu126` wheel сәйкес келмеуі мүмкін, сондықтан `requirements.txt` пен құрылғыға сәйкес ресми wheel пайдаланылады.

```powershell
.\.venv\Scripts\python.exe -m soundsense.run_all --skip-download
.\.venv\Scripts\python.exe -m soundsense.visualize
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m soundsense.smoke
.\.venv\Scripts\python.exe app.py
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
