# SoundSense YAMNet нәтижелері

Бұл есеп AudioSet-те алдын ала оқытылған frozen YAMNet және ESC-10 үшін жаңа head нәтижелерін сипаттайды. Қолданба YAMNet пен EfficientNet-B0 модельдерін қолдайды.
MobileNetV1 терең CNN backbone 1024 embedding береді. Train-only mean/std → Dropout 0.2 → Linear 1024→10.
Оқытылатын head параметрлері 10 250. Backbone қайта оқытылмаған.

| Бағалау | Дұрыс / барлығы | Accuracy | Macro F1 | ROC AUC | Macro AP |
|---|---:|---:|---:|---:|---:|
| fixed test | 78/80 | 97.50% | 0.9739 | 0.9988 | 0.9924 |
| OOF | 391/400 | 97.75% | 0.9775 | 0.9984 | 0.9921 |

## Әдіснама

ESC-10: 400 бастапқы клип, он санат, ресми бес fold. Fixed train 1–3 (240), validation 4 (80), test 5 (80).
Seed 42. AdamW lr=0.001 weight_decay=0.001 batch=16; max 160 epoch patience=25. Checkpoint ең төмен validation cross-entropy арқылы таңдалады.
YAMNet frontend: 16 кГц mono; STFT 25 мс және 10 мс; 64 Mel жолағы 125–7500 Гц; log offset 0.001; патч 96×64.
Бес секунд қысқа болса нөлмен толықтырылады, ұзын болса алғашқы бөлігі алынады. Frame embedding уақыт бойынша орташаланады.
CV: test K, validation K−1 (1 үшін 5), қалған үш fold train. Әр клип бір OOF болжам алады. Fold 5 fixed checkpoint-ін қолданады.
Scaler тек train арқылы fit жасалады. Backbone frozen, waveform augmentation және SpecAugment қолданылмаған.

## Графиктер

- [yamnet_transfer_fixed_confusion](../reports/accuracy_v2/plots/yamnet_transfer_fixed_confusion.png)
- [yamnet_transfer_fixed_confusion_normalized](../reports/accuracy_v2/plots/yamnet_transfer_fixed_confusion_normalized.png)
- [yamnet_transfer_fixed_per_class_f1](../reports/accuracy_v2/plots/yamnet_transfer_fixed_per_class_f1.png)
- [yamnet_transfer_fixed_precision_recall](../reports/accuracy_v2/plots/yamnet_transfer_fixed_precision_recall.png)
- [yamnet_transfer_fixed_roc](../reports/accuracy_v2/plots/yamnet_transfer_fixed_roc.png)
- [yamnet_transfer_fixed_training](../reports/accuracy_v2/plots/yamnet_transfer_fixed_training.png)
- [yamnet_transfer_fold1_training](../reports/accuracy_v2/plots/yamnet_transfer_fold1_training.png)
- [yamnet_transfer_fold2_training](../reports/accuracy_v2/plots/yamnet_transfer_fold2_training.png)
- [yamnet_transfer_fold3_training](../reports/accuracy_v2/plots/yamnet_transfer_fold3_training.png)
- [yamnet_transfer_fold4_training](../reports/accuracy_v2/plots/yamnet_transfer_fold4_training.png)
- [yamnet_transfer_oof_confusion](../reports/accuracy_v2/plots/yamnet_transfer_oof_confusion.png)
- [yamnet_transfer_oof_confusion_normalized](../reports/accuracy_v2/plots/yamnet_transfer_oof_confusion_normalized.png)
- [yamnet_transfer_oof_fold_metrics](../reports/accuracy_v2/plots/yamnet_transfer_oof_fold_metrics.png)
- [yamnet_transfer_oof_per_class_f1](../reports/accuracy_v2/plots/yamnet_transfer_oof_per_class_f1.png)
- [yamnet_transfer_oof_precision_recall](../reports/accuracy_v2/plots/yamnet_transfer_oof_precision_recall.png)
- [yamnet_transfer_oof_roc](../reports/accuracy_v2/plots/yamnet_transfer_oof_roc.png)

## Шектеулер

400 клип және бір seed. Fixed test бұрын белгілі болған; бастапқы модель іріктеуі fixed validation арқылы жүргізілген. Бұл толық nested model selection емес.
AudioSet пен ESC жазбаларының ықтимал қабаттасуы жеке тексерілмеген. Fold SD confidence interval емес.
Белгісіз дыбыстар да он санаттың біріне жіктеледі. Softmax калибрленбеген. Сыртқы test, multi-label және оқиғаны уақыт бойынша локализациялау бағаланбаған.
Барлық 400 клиппен соңғы deployment retraining жасалмаған. Бастапқы бес YAMNet checkpoint пен тарих сақталған.

Ресми YAMNet: https://www.tensorflow.org/tutorials/audio/transfer_learning_audio
ESC-50 және ESC-10: https://github.com/karolpiczak/ESC-50
