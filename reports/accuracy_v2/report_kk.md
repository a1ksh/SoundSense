# SoundSense Accuracy V2 — нақты нәтижелер

Барлық көрсеткіш сақталған нақты болжамдардан есептелді. Seed=42; ресми ESC-10 fold-тары сақталды.

| Модель / бағалау | Дұрыс / барлығы | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| cnn_v1_fixed | 67/80 | 83.75% | 0.8350 |
| cnn_v2_fixed | 60/80 | 75.00% | 0.7279 |
| efficientnet_b0_mel_fixed | 73/80 | 91.25% | 0.9114 |
| efficientnet_b0_mel_oof | 354/400 | 88.50% | 0.8849 |
| mlp_v1_fixed | 48/80 | 60.00% | 0.5776 |
| yamnet_transfer_fixed | 78/80 | 97.50% | 0.9739 |
| yamnet_transfer_oof | 391/400 | 97.75% | 0.9775 |

Қолданбаға таңдалған модель: **yamnet_transfer**. Таңдау fold-4 validation cross-entropy арқылы, жаңа fold-5 бағалауынан бұрын бекітілді.

## Бес fold бойынша бағалау

- efficientnet_b0_mel: accuracy 88.50% ± 2.40%; macro-F1 0.8837 ± 0.0253. Жиынтық 354/400, OOF macro-F1=0.8849.
- yamnet_transfer: accuracy 97.75% ± 1.63%; macro-F1 0.9772 ± 0.0165. Жиынтық 391/400, OOF macro-F1=0.9775.

Стандарттық ауытқу бес fold бойынша ddof=1 арқылы есептелді; бұл сенімділік интервалы емес.

## Өзгерістер және дәлелдер

- CNN V2: тек train дыбысына уақыт ығысуы, gain, әлсіз шу және SpecAugment; AdamW және validation scheduler.
- EfficientNet: нақты ImageNet pretrained backbone бекітілген; 3 арналы 224×224 Mel, ImageNet normalization; жаңа 10-класты head.
- YAMNet орындалса: ресми AudioSet pretrained модельдің 1024-өлшемді frame embedding-терінің орташа мәні; ESC-10 үшін бөлек 10-класты head. 521 AudioSet label-і ESC-10 label-і ретінде қолданылмайды.
- Нақты орындалған модельдер мен қателер: `run_status.json`, `blockers.json`; ортамен командалар: `environment.json`.
- `predictions/`: бастапқы файл, нақты/болжанған класс, 10 ықтималдық, қателер тізімі. `training_history/`: нақты epoch журналдары.
- `plots/`: 300 dpi confusion matrix, normalized matrix, F1, ROC, PR, оқыту және fold графиктері. `metrics/`: сандық негізі.

## Шектеулер

ESC-10 шағын: барлығы 400 дыбыс, әр test fold-та 80 дыбыс, әр класта 8 мысал. Бір fold-та бір қате accuracy-ді 1.25 пайыздық тармаққа өзгертеді. Бір seed қолданылды; fold нәтижелері өзгереді. Fold 5 бұрын белгілі болғандықтан fixed-split тәжірибе exploratory. CV кандидаттары fixed validation бойынша таңдалды: бұл тәуелсіз жаңа сыртқы dataset емес, кандидат іздеудің толық nested бағалауы емес. AudioSet/ImageNet pretraining көздерінің ESC аудиосымен ықтимал қабаттасуы тексерілген жоқ. 90% алдын ала кепілдендірілген нәтиже емес; кестеде тек өлшенген сандар берілген. Open-world белгісіз санаттар қамтылмайды; softmax сенімділігі калибрленбеген. Барлық 400 дыбыспен deployment retraining жасалмады; app validation арқылы таңдалған fixed checkpoint-ті қолданады.

## Іске қосу

```powershell
.\.runtime\python311\python.exe app.py
.\.runtime\python311\python.exe -m pytest -q
```

Негізгі орнату және қайталау: `experiments/accuracy_v2/README.md`.

## Сапаны тексеру

Нақты pytest: **48 passed**, 0 skipped; exit code=0.
Бастапқы checkpoint-тер SHA-256 арқылы сақталғаны тексерілді. Train/app ықтималдықтары сәйкес, қайталама inference детерминді. Екі модельдің жергілікті Gradio HTTP API, бес output және default selector тексерілді.
Барлық 52 нәтиже графигінің resolution метадерегі 300 dpi. Толық журнал: `pytest.txt`, `verification.log`, `quality_checks.json`.

## Inference уақыты

Осы компьютерде YAMNet толық CPU inference бір train клипке **41.89 секунд** алды. Бұл бөлек TensorFlow process-інің іске қосылуын қамтиды; әр талдауда осы process қайта іске қосылады. CNN v1 және Best V2 таңдауын пайдалануға болады. `comparison.csv` пен `inference_verification.json` нақты өлшемдерді сақтайды. Барлық latency бір клипке тізбектей өлшенді; бастапқы librosa/JIT және framework startup әр модельде әртүрлі. Бұл бақыланатын көп-қайталаулы жылдамдық benchmark-і емес.
