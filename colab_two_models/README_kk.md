# YAMNet және EfficientNet B0 толық Colab

Екі жүйе де терең конволюциялық желіге негізделген. YAMNet AudioSet дыбыстарында
алдын ала оқытылған MobileNetV1 белгілерін, EfficientNet-B0 ImageNet-та оқытылған
MBConv және SE белгілерін қолданады. ESC-10 үшін әрқайсына бөлек жаңа head оқытылады.
Backbone frozen, тек head жаңартылады. Бұл — transfer learning.

## Colab-та ашу

1. `SoundSense_YAMNet_EfficientNet_Full_Colab.ipynb` файлын [Colab](https://colab.research.google.com/) → File → Upload notebook арқылы ашыңыз.
2. Runtime → Change runtime type → T4 GPU, Runtime Version → 2026.04 таңдаңыз. Python 3.12 ортасы қажет.
3. Runtime → Run all басыңыз. Сұралғанда `SoundSense_2Models_Trained_Colab.zip` файлын жүктеңіз.
4. `RUN_MODE='retrain'` жаңа head-тарды оқытады. `RUN_MODE='use_trained'` дайын салмақтарды тексеріп қолданады.

Ноутбукте сақталған орындалу нәтижелері осы компьютердегі нақты GPU іске қосылуынан алынған.
Google серверінде орындау бөлек тексерілген жоқ. Ноутбук ішіндегі қадамдар Colab-та қайта орындалуға арналған.
Ноутбук ішіндегі суреттер — ықшамдалған алдын ала көрініс; архивтерде 300 dpi PNG және толық SVG/PDF нұсқалары сақталады.
Алғашқы орнату кезінде негізгі PyTorch және бөлек TensorFlow ортасының пакеттері жүктеледі.

## Пакет ішінде

- `assets/audio`: ресми ESC-10 400 WAV, әр санатта 40.
- `assets/metadata`: pinned manifest, бастапқы лицензия мен атрибуция.
- `assets/backbones`: ресми frozen YAMNet және EfficientNet-B0 салмақтары.
- `assets/features`: SHA-256 және data/backbone/frontend contract арқылы тексерілетін 1024/1280 embedding кэштері.
- `run/models`: екі архитектураның бес fold үшін оқытылған 10 head-ы.
- `run/history`: нақты epoch loss/accuracy және ең үздік checkpoint метадерегі.
- `run/predictions`: fixed test және 400 OOF болжамдары, он score, барлық қателер.
- `run/metrics`: санаттық метрикалар, confusion matrix, ROC/PR сандық негізі.
- `run/architecture`: екі желінің stage/channel/stride/repeat кестелері.
- `run/figures`: 58 суреттің PNG 300 dpi, SVG және PDF нұсқалары.
- `run/tables/model_comparison.csv`: осы орындалудың негізгі нәтижелері.
- `run/report_kk.md`: нәтижелер мен барлық суреттер тізімі бар қазақша есеп.
- `run/checks.json`: нақты автоматты тексерудің қорытындысы.

Суреттер: әр модельге 16 негізгі нәтиже графигі; 4 дерек/frontend графигі;
5 архитектура/әдіснама сызбасы; 12 тікелей салыстыру графигі; 5 confidence,
reliability, ресурс және қате мысал графигі. Барлығы 58 сурет.

`SoundSense_2Models_Results.zip` архиві тек coursework нәтижелері мен оқытылған head-тарды қамтиды.
Толық қайта орындау мен өз аудиоңызға inference үшін үлкен дайын Colab пакетін пайдаланыңыз.

## Әдіснама

Seed 42. Fixed train folds 1–3 (240), validation 4 (80), test 5 (80).
Бес fold үшін test K, validation K−1 (K=1 үшін 5), қалған үшеуі train.
Scaler mean/std әр train бөлігіне ғана fit жасалады. Checkpoint minimum validation
cross-entropy арқылы таңдалады. Әр клипке оны train ретінде көрмеген модель бір OOF болжам береді.

Trainable head: YAMNet 1024×10+10 = 10 250; EfficientNet 1280×10+10 = 12 810 параметр.
Бұл бүкіл backbone саны емес. Frozen белгілер label қолданбай шығарылады.

Екі модельдің pretraining domain-і және frontend-і бөлек. Сондықтан нәтиже айырмасы
архитектураның жеке әсерін дәлелдемейді. Бір seed, шағын дерек, бұрын белгілі fixed test,
тарихи fixed-validation модель іріктеуі және overlapping train folds бар. Бұл толық nested
model selection немесе сыртқы dataset тексеруі емес. Fold SD confidence interval емес.
Белгісіз санаттар, multi-label және confidence calibration бағаланбаған.

## Ресми дереккөздер

- [ESC-10](https://github.com/karolpiczak/ESC-50), Piczak 2015, CC BY 3.0 subset.
- [YAMNet transfer](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio).
- [MobileNet](https://arxiv.org/abs/1704.04861).
- [EfficientNet](https://arxiv.org/abs/1905.11946).
- [Torchvision B0](https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.efficientnet_b0.html).
- [Colab runtime versions](https://research.google.com/colaboratory/runtime-version-faq.html).
