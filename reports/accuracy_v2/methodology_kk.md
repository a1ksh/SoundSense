## Әдіснама

ESC-10: 400 бастапқы клип, он санат, ресми бес fold. Fixed train 1–3 (240), validation 4 (80), test 5 (80).
Seed 42. AdamW lr=0.001 weight_decay=0.001 batch=16; max 160 epoch patience=25. Checkpoint ең төмен validation cross-entropy арқылы таңдалады.
YAMNet frontend: 16 кГц mono; STFT 25 мс және 10 мс; 64 Mel жолағы 125–7500 Гц; log offset 0.001; патч 96×64.
Бес секунд қысқа болса нөлмен толықтырылады, ұзын болса алғашқы бөлігі алынады. Frame embedding уақыт бойынша орташаланады.
CV: test K, validation K−1 (1 үшін 5), қалған үш fold train. Әр клип бір OOF болжам алады. Fold 5 fixed checkpoint-ін қолданады.
Scaler тек train арқылы fit жасалады. Backbone frozen, waveform augmentation және SpecAugment қолданылмаған.
