# YAMNet және EfficientNet-B0 салыстыруы

Екеуі де терең CNN негізіндегі transfer learning моделі. Frozen backbone жаңартылмайды; ESC-10 head оқытылады.
YAMNet: AudioSet → MobileNetV1 → 1024 embedding → mean → train scaler → Dropout 0.2 → Linear 10.
EfficientNet-B0: ImageNet → MBConv/SE → 1280 embedding → train scaler → Dropout 0.2 → Linear 10.
Бірдей 400 ESC-10 клип, seed 42, ресми folds, train 240 / validation 80 / test 80.
Екі model-дің head optimizer рецепттері алдын ала бекітілген; hyperparameter бюджетін теңестіру жүргізілмеген.

| Модель | Бағалау | Дұрыс | Accuracy | Macro F1 | Macro ROC AUC | Macro AP |
|---|---|---:|---:|---:|---:|---:|
| YAMNet transfer | fixed | 78/80 | 97.50% | 0.9739 | 0.9988 | 0.9924 |
| YAMNet transfer | oof | 391/400 | 97.75% | 0.9775 | 0.9984 | 0.9921 |
| EfficientNet-B0 transfer | fixed | 73/80 | 91.25% | 0.9114 | 0.9922 | 0.9544 |
| EfficientNet-B0 transfer | oof | 354/400 | 88.50% | 0.8849 | 0.9915 | 0.9445 |

## Барлық графиктер

1. [architecture_yamnet](figures/png/architecture_yamnet.png) — YAMNet frozen MobileNetV1 backbone және ESC-10 үшін жаңа head.
2. [architecture_efficientnet](figures/png/architecture_efficientnet.png) — EfficientNet-B0 үшін жобада қолданылатын Mel preprocessing және frozen backbone.
3. [architecture_yamnet_block](figures/png/architecture_yamnet_block.png) — Depthwise уақыт-жиілік өрнектерін өңдейді pointwise арналарды біріктіреді.
4. [architecture_mbconv](figures/png/architecture_mbconv.png) — MBConv expansion ratio бірден үлкен болса expansion қолданылады residual тек stride 1 және кіріс шығыс арналары тең болғанда қосылады.
5. [architecture_protocol](figures/png/architecture_protocol.png) — Екі модель үшін бірдей ресми folds және train-only статистика.
6. [data_class_distribution](figures/png/data_class_distribution.png) — Он санаттың train validation test бөлінісі.
7. [data_waveforms](figures/png/data_waveforms.png) — Train бөлігіндегі он санаттың дыбыс толқындары.
8. [yamnet_transfer_frontend_examples](figures/png/yamnet_transfer_frontend_examples.png) — YAMNet transfer үшін нақты frontend мысалдары.
9. [efficientnet_b0_mel_frontend_examples](figures/png/efficientnet_b0_mel_frontend_examples.png) — EfficientNet-B0 transfer үшін нақты frontend мысалдары.
10. [yamnet_transfer_fold1_training](figures/png/yamnet_transfer_fold1_training.png) — Train және validation қисықтары test метрикасы емес.
11. [yamnet_transfer_fold2_training](figures/png/yamnet_transfer_fold2_training.png) — Train және validation қисықтары test метрикасы емес.
12. [yamnet_transfer_fold3_training](figures/png/yamnet_transfer_fold3_training.png) — Train және validation қисықтары test метрикасы емес.
13. [yamnet_transfer_fold4_training](figures/png/yamnet_transfer_fold4_training.png) — Train және validation қисықтары test метрикасы емес.
14. [yamnet_transfer_fixed_training](figures/png/yamnet_transfer_fixed_training.png) — Train және validation қисықтары test метрикасы емес.
15. [yamnet_transfer_fixed_confusion](figures/png/yamnet_transfer_fixed_confusion.png) — Қателер саны көрсетілген шатасу матрицасы.
16. [yamnet_transfer_fixed_confusion_normalized](figures/png/yamnet_transfer_fixed_confusion_normalized.png) — Қалыпталған шатасу матрицасы.
17. [yamnet_transfer_fixed_per_class_f1](figures/png/yamnet_transfer_fixed_per_class_f1.png) — Әр санаттың precision және recall теңгерімін көрсететін F1.
18. [yamnet_transfer_fixed_roc](figures/png/yamnet_transfer_fixed_roc.png) — One versus rest қисықтары macro AP трапециялық PR AUC емес.
19. [yamnet_transfer_fixed_precision_recall](figures/png/yamnet_transfer_fixed_precision_recall.png) — One versus rest қисықтары macro AP трапециялық PR AUC емес.
20. [yamnet_transfer_oof_confusion](figures/png/yamnet_transfer_oof_confusion.png) — Қателер саны көрсетілген шатасу матрицасы.
21. [yamnet_transfer_oof_confusion_normalized](figures/png/yamnet_transfer_oof_confusion_normalized.png) — Қалыпталған шатасу матрицасы.
22. [yamnet_transfer_oof_per_class_f1](figures/png/yamnet_transfer_oof_per_class_f1.png) — Әр санаттың precision және recall теңгерімін көрсететін F1.
23. [yamnet_transfer_oof_roc](figures/png/yamnet_transfer_oof_roc.png) — One versus rest қисықтары macro AP трапециялық PR AUC емес.
24. [yamnet_transfer_oof_precision_recall](figures/png/yamnet_transfer_oof_precision_recall.png) — One versus rest қисықтары macro AP трапециялық PR AUC емес.
25. [yamnet_transfer_oof_fold_metrics](figures/png/yamnet_transfer_oof_fold_metrics.png) — Бес fold нәтижесі SD сенімділік интервалы емес.
26. [efficientnet_b0_mel_fold1_training](figures/png/efficientnet_b0_mel_fold1_training.png) — Train және validation қисықтары test метрикасы емес.
27. [efficientnet_b0_mel_fold2_training](figures/png/efficientnet_b0_mel_fold2_training.png) — Train және validation қисықтары test метрикасы емес.
28. [efficientnet_b0_mel_fold3_training](figures/png/efficientnet_b0_mel_fold3_training.png) — Train және validation қисықтары test метрикасы емес.
29. [efficientnet_b0_mel_fold4_training](figures/png/efficientnet_b0_mel_fold4_training.png) — Train және validation қисықтары test метрикасы емес.
30. [efficientnet_b0_mel_fixed_training](figures/png/efficientnet_b0_mel_fixed_training.png) — Train және validation қисықтары test метрикасы емес.
31. [efficientnet_b0_mel_fixed_confusion](figures/png/efficientnet_b0_mel_fixed_confusion.png) — Қателер саны көрсетілген шатасу матрицасы.
32. [efficientnet_b0_mel_fixed_confusion_normalized](figures/png/efficientnet_b0_mel_fixed_confusion_normalized.png) — Қалыпталған шатасу матрицасы.
33. [efficientnet_b0_mel_fixed_per_class_f1](figures/png/efficientnet_b0_mel_fixed_per_class_f1.png) — Әр санаттың precision және recall теңгерімін көрсететін F1.
34. [efficientnet_b0_mel_fixed_roc](figures/png/efficientnet_b0_mel_fixed_roc.png) — One versus rest қисықтары macro AP трапециялық PR AUC емес.
35. [efficientnet_b0_mel_fixed_precision_recall](figures/png/efficientnet_b0_mel_fixed_precision_recall.png) — One versus rest қисықтары macro AP трапециялық PR AUC емес.
36. [efficientnet_b0_mel_oof_confusion](figures/png/efficientnet_b0_mel_oof_confusion.png) — Қателер саны көрсетілген шатасу матрицасы.
37. [efficientnet_b0_mel_oof_confusion_normalized](figures/png/efficientnet_b0_mel_oof_confusion_normalized.png) — Қалыпталған шатасу матрицасы.
38. [efficientnet_b0_mel_oof_per_class_f1](figures/png/efficientnet_b0_mel_oof_per_class_f1.png) — Әр санаттың precision және recall теңгерімін көрсететін F1.
39. [efficientnet_b0_mel_oof_roc](figures/png/efficientnet_b0_mel_oof_roc.png) — One versus rest қисықтары macro AP трапециялық PR AUC емес.
40. [efficientnet_b0_mel_oof_precision_recall](figures/png/efficientnet_b0_mel_oof_precision_recall.png) — One versus rest қисықтары macro AP трапециялық PR AUC емес.
41. [efficientnet_b0_mel_oof_fold_metrics](figures/png/efficientnet_b0_mel_oof_fold_metrics.png) — Бес fold нәтижесі SD сенімділік интервалы емес.
42. [comparison_fixed_metrics](figures/png/comparison_fixed_metrics.png) — Екі модельдің бірдей клиптердегі негізгі метрикалары.
43. [comparison_fixed_per_class_f1](figures/png/comparison_fixed_per_class_f1.png) — Санаттар бойынша F1 салыстыруы.
44. [comparison_fixed_roc](figures/png/comparison_fixed_roc.png) — Санаттық қисықтардың көрсетілім үшін интерполяцияланған орташа мәндері.
45. [comparison_fixed_precision_recall](figures/png/comparison_fixed_precision_recall.png) — Санаттық қисықтардың көрсетілім үшін интерполяцияланған орташа мәндері.
46. [comparison_oof_metrics](figures/png/comparison_oof_metrics.png) — Екі модельдің бірдей клиптердегі негізгі метрикалары.
47. [comparison_oof_per_class_f1](figures/png/comparison_oof_per_class_f1.png) — Санаттар бойынша F1 салыстыруы.
48. [comparison_oof_roc](figures/png/comparison_oof_roc.png) — Санаттық қисықтардың көрсетілім үшін интерполяцияланған орташа мәндері.
49. [comparison_oof_precision_recall](figures/png/comparison_oof_precision_recall.png) — Санаттық қисықтардың көрсетілім үшін интерполяцияланған орташа мәндері.
50. [comparison_folds](figures/png/comparison_folds.png) — Бірдей бес сыртқы fold бойынша нәтижелер.
51. [comparison_fold_difference](figures/png/comparison_fold_difference.png) — Accuracy айырмасы пайыздық тармақпен берілген.
52. [comparison_fixed_training](figures/png/comparison_fixed_training.png) — Fixed split train және validation динамикасы.
53. [comparison_oof_agreement](figures/png/comparison_oof_agreement.png) — Екі модельдің бірдей OOF клиптердегі дұрыс және қате жауаптары.
54. [comparison_fixed_confidence](figures/png/comparison_fixed_confidence.png) — Softmax үлесі қате жауапта да жоғары болуы мүмкін.
55. [comparison_fixed_reliability](figures/png/comparison_fixed_reliability.png) — Калибрленбеген score үшін бес interval бойынша сипаттамалық reliability диаграммасы.
56. [comparison_head_resources](figures/png/comparison_head_resources.png) — Жаңа head параметрлері және сақталған head оқыту уақыты.
57. [yamnet_transfer_fixed_error_examples](figures/png/yamnet_transfer_fixed_error_examples.png) — Файл атауы бойынша алғашқы алты test қатесінің нақты waveform мысалдары.
58. [efficientnet_b0_mel_fixed_error_examples](figures/png/efficientnet_b0_mel_fixed_error_examples.png) — Файл атауы бойынша алғашқы алты test қатесінің нақты waveform мысалдары.

## Әдіснамалық шектеулер

Бұл жаңа тәуелсіз dataset емес. Fixed fold-5 бұрын белгілі. Екі нұсқа тарихи fixed validation тәжірибелерімен таңдалғандықтан, OOF толық nested architecture selection болып саналмайды.
Бір seed және шағын дерек; fold train жиындары қабаттасады. Fold SD confidence interval емес. AudioSet/ImageNet pretraining пен ESC жазбаларының ықтимал қабаттасуы жеке тексерілмеген.
YAMNet аудиода, EfficientNet суретте алдын ала оқытылған. Бұл pretraining domain және frontend-тері бөлек екі толық жүйенің салыстыруы; айырмашылықты архитектураның таза әсеріне жатқызуға болмайды.
Score calibration оқытылмаған; reliability тек сипаттамалық. Белгісіз санаттар мен multi-label оқиғалар бағаланбаған. Ең үлкен softmax дұрыстыққа кепілдік бермейді.

## Дереккөздер

- ESC-10: https://github.com/karolpiczak/ESC-50
- YAMNet: https://www.tensorflow.org/tutorials/audio/transfer_learning_audio
- MobileNet: https://arxiv.org/abs/1704.04861
- EfficientNet: https://arxiv.org/abs/1905.11946
- Torchvision implementation: https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.efficientnet_b0.html
- Metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
