# Accuracy V2 әдіснамасы

Дерек: Piczak, K. J. (2015), ESC: Dataset for Environmental Sound Classification.
https://github.com/karolpiczak/ESC-50 ; ESC-10 CC BY 3.0. Түпнұсқа лицензия мен
авторлар `data/metadata/LICENSE`, `attributions.csv` ішінде сақталған.

Fixed split: train fold 1–3 (240), validation fold 4 (80), test fold 5 (80).
Seed 42. Checkpoint ең төмен validation cross-entropy бойынша сақталады.
Кандидат та осы validation бойынша таңдалады; test accuracy таңдау үшін
қолданылмайды. Әр аяқталған checkpoint үшін сақталған test болжамы қайта
қолданылады, оның SHA-256 тексеріледі. Бұрын белгілі fold-5 нәтижесі ашық көрсетіледі.

5-fold CV: test K, validation K−1 (1 үшін 5), қалған 3 fold train. Әр бастапқы
клиптің OOF болжамы бір рет қана кездеседі. Бастапқы source recording бір fold-та
сақталады. Train-ға арналған augmentation validation/test-ке қолданылмайды.
Scaler mean/std тек train partition-нан есептеледі; checkpoint ішінде сақталады.
Frozen embedding extraction label-дерді қолданбайды. YAMNet subprocess-іне
label немесе class index жіберілмейді; тек 16 кГц mono waveform мен файл атаулары.

Accuracy = дұрыс болжам / бастапқы клип саны. Macro-F1 — 10 класс F1 орташа мәні.
ROC әр классқа one-vs-rest есептеледі; macro ROC-AUC — класс AUC орташа мәні.
PR көрсеткіші — sklearn non-interpolated average precision, трапециялық PR-AUC
емес; macro AP — он AP орташа мәні. Қисықтарды көрсету үшін ғана interpolation
қолданылады. Confusion matrix жолы — нақты класс, бағаны — болжанған класс.
Fold standard deviation ddof=1; aggregate OOF F1 fold F1 орташа мәнінен өзгеше.

EfficientNet тек frozen backbone режимінде; head optimizer шағын batch-пен
оқытылады, backbone BatchNorm eval режимінде қалады. YAMNet те frozen, CPU
арқылы extraction; жаңа head PyTorch CUDA қолжетімді болса GPU-да оқиды.
CNN V2 train-only shift ±0.35 с, gain ±4 dB, SNR 25–35 dB және modest
frequency/time mask қолданады. Толық recipe `config.json` ішінде бекітілген.

Training time transfer head үшін backbone extraction уақытын қамтымайды;
`comparison.csv` ішінде extraction бөлек көрсетіледі. Prediction receipt-теріндегі
head timing толық audio decoding/backbone inference latency емес. App толық
inference уақыты бөлек end-to-end тексеру журналында өлшенеді.

Ресми implementation деректері:
https://www.tensorflow.org/tutorials/audio/transfer_learning_audio
https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html
