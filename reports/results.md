# VeriScript evaluation results

Operating points (tuned on `val`): **fast** [0.45, 0.50], **balanced** [0.38, 0.50], **thorough** [0.19, 0.84]

## Split `test` (n=4000)

| System | Accuracy | Precision | Recall | F1 | AUROC | FPR | ECE | Escalation |
|---|---|---|---|---|---|---|---|---|
| heuristic | 0.9465 | 0.9524 | 0.9400 | 0.9461 | 0.9878 | 0.0470 | 0.0149 | 0.000 |
| transformer | 0.9740 | 0.9523 | 0.9980 | 0.9746 | 0.9992 | 0.0500 | 0.0264 | 1.000 |
| cascade_fast | 0.9467 | 0.9524 | 0.9405 | 0.9464 | 0.9878 | 0.0470 | 0.0148 | 0.000 |
| cascade_balanced | 0.9487 | 0.9517 | 0.9455 | 0.9486 | 0.9881 | 0.0480 | 0.0150 | 0.005 |
| cascade_thorough | 0.9712 | 0.9753 | 0.9670 | 0.9711 | 0.9913 | 0.0245 | 0.0147 | 0.080 |

95% bootstrap CIs (1000 resamples): heuristic: F1 [0.9389, 0.9535], AUROC [0.9849, 0.9904]; transformer: F1 [0.9694, 0.9795], AUROC [0.9987, 0.9995]; cascade_fast: F1 [0.9392, 0.9538], AUROC [0.9850, 0.9904]; cascade_balanced: F1 [0.9416, 0.9559], AUROC [0.9853, 0.9907]; cascade_thorough: F1 [0.9658, 0.9763], AUROC [0.9886, 0.9936]

- Paired bootstrap cascade_balanced - transformer: dF1 = -0.0261 (95% CI [-0.0338, -0.0186], p = 0.000)
- Paired bootstrap cascade_balanced - heuristic: dF1 = +0.0024 (95% CI [+0.0007, +0.0043], p = 0.006)

## Split `heldout_generator` (n=744)

| System | Accuracy | Precision | Recall | F1 | AUROC | FPR | ECE | Escalation |
|---|---|---|---|---|---|---|---|---|
| heuristic | 0.9113 | 0.9581 | 0.8602 | 0.9065 | 0.9641 | 0.0376 | 0.0491 | 0.000 |
| transformer | 0.9570 | 0.9450 | 0.9704 | 0.9576 | 0.9950 | 0.0565 | 0.0434 | 1.000 |
| cascade_fast | 0.9113 | 0.9581 | 0.8602 | 0.9065 | 0.9641 | 0.0376 | 0.0491 | 0.000 |
| cascade_balanced | 0.9167 | 0.9559 | 0.8737 | 0.9129 | 0.9650 | 0.0403 | 0.0493 | 0.013 |
| cascade_thorough | 0.9315 | 0.9707 | 0.8898 | 0.9285 | 0.9649 | 0.0269 | 0.0523 | 0.101 |

95% bootstrap CIs (1000 resamples): heuristic: F1 [0.8822, 0.9282], AUROC [0.9509, 0.9755]; transformer: F1 [0.9429, 0.9716], AUROC [0.9918, 0.9975]; cascade_fast: F1 [0.8822, 0.9282], AUROC [0.9509, 0.9755]; cascade_balanced: F1 [0.8899, 0.9337], AUROC [0.9516, 0.9768]; cascade_thorough: F1 [0.9083, 0.9474], AUROC [0.9518, 0.9768]

- Paired bootstrap cascade_balanced - transformer: dF1 = -0.0449 (95% CI [-0.0678, -0.0226], p = 0.000)
- Paired bootstrap cascade_balanced - heuristic: dF1 = +0.0064 (95% CI [-0.0007, +0.0139], p = 0.068)

## Split `mage_ood` (n=1433)

| System | Accuracy | Precision | Recall | F1 | AUROC | FPR | ECE | Escalation |
|---|---|---|---|---|---|---|---|---|
| heuristic | 0.8521 | 0.8081 | 0.9125 | 0.8571 | 0.9283 | 0.2052 | 0.0742 | 0.000 |
| transformer | 0.6441 | 0.5775 | 1.0000 | 0.7321 | 0.9377 | 0.6929 | 0.3510 | 1.000 |
| cascade_fast | 0.8514 | 0.8071 | 0.9125 | 0.8566 | 0.9278 | 0.2065 | 0.0746 | 0.001 |
| cascade_balanced | 0.8500 | 0.7968 | 0.9283 | 0.8575 | 0.9254 | 0.2242 | 0.0824 | 0.021 |
| cascade_thorough | 0.8207 | 0.7472 | 0.9541 | 0.8381 | 0.9059 | 0.3057 | 0.1516 | 0.220 |

## Split `mage_ood_para` (n=1500)

| System | Accuracy | Precision | Recall | F1 | AUROC | FPR | ECE | Escalation |
|---|---|---|---|---|---|---|---|---|
| heuristic | 0.6767 | 0.7310 | 0.5194 | 0.6073 | 0.7499 | 0.1774 | 0.1560 | 0.000 |
| transformer | 0.5193 | 0.5003 | 0.9958 | 0.6660 | 0.8297 | 0.9229 | 0.4716 | 1.000 |
| cascade_fast | 0.6773 | 0.7315 | 0.5208 | 0.6084 | 0.7502 | 0.1774 | 0.1556 | 0.001 |
| cascade_balanced | 0.6780 | 0.7093 | 0.5609 | 0.6265 | 0.7406 | 0.2134 | 0.1716 | 0.039 |
| cascade_thorough | 0.6847 | 0.6502 | 0.7465 | 0.6950 | 0.7581 | 0.3728 | 0.2729 | 0.437 |

## Split `raid_attack` (n=690)

| System | Accuracy | Precision | Recall | F1 | AUROC | FPR | ECE | Escalation |
|---|---|---|---|---|---|---|---|---|
| heuristic | 0.8435 | 1.0000 | 0.8435 | 0.9151 | - | - | 0.1851 | 0.000 |
| transformer | 0.9870 | 1.0000 | 0.9870 | 0.9934 | - | - | 0.0135 | 1.000 |
| cascade_fast | 0.8449 | 1.0000 | 0.8449 | 0.9159 | - | - | 0.1843 | 0.001 |
| cascade_balanced | 0.8580 | 1.0000 | 0.8580 | 0.9236 | - | - | 0.1771 | 0.014 |
| cascade_thorough | 0.9130 | 1.0000 | 0.9130 | 0.9545 | - | - | 0.0947 | 0.214 |

## Split `val` (n=3996)

| System | Accuracy | Precision | Recall | F1 | AUROC | FPR | ECE | Escalation |
|---|---|---|---|---|---|---|---|---|
| heuristic | 0.9610 | 0.9685 | 0.9530 | 0.9606 | 0.9912 | 0.0310 | 0.0085 | 0.000 |
| transformer | 0.9825 | 0.9662 | 1.0000 | 0.9828 | 0.9997 | 0.0350 | 0.0189 | 1.000 |
| cascade_fast | 0.9610 | 0.9685 | 0.9530 | 0.9606 | 0.9912 | 0.0310 | 0.0085 | 0.000 |
| cascade_balanced | 0.9632 | 0.9686 | 0.9575 | 0.9630 | 0.9915 | 0.0310 | 0.0089 | 0.005 |
| cascade_thorough | 0.9785 | 0.9823 | 0.9745 | 0.9784 | 0.9941 | 0.0175 | 0.0106 | 0.071 |

## Fairness (`fair_test`: human essays only)

| System | FPR native | FPR non-native | Delta-FPR |
|---|---|---|---|
| heuristic | 0.025 | 0.170 | 0.146 |
| transformer | 0.519 | 0.638 | 0.120 |
| cascade_fast | 0.025 | 0.170 | 0.146 |
| cascade_balanced | 0.025 | 0.170 | 0.146 |
| cascade_thorough | 0.025 | 0.255 | 0.231 |

## Per generator family (test split): detection rate (recall) on that family's AI texts

| Family | n AI | heuristic | transformer | cascade_fast | cascade_balanced | cascade_thorough |
|---|---|---|---|---|---|---|
| bloomz | 122 | 0.9426 | 1.0000 | 0.9426 | 0.9508 | 0.9918 |
| cohere | 248 | 0.9194 | 1.0000 | 0.9194 | 0.9355 | 0.9637 |
| dolly | 218 | 0.8073 | 1.0000 | 0.8119 | 0.8211 | 0.8991 |
| flant5 | 41 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| gpt | 1195 | 0.9715 | 0.9983 | 0.9715 | 0.9732 | 0.9824 |
| llama | 89 | 0.8876 | 1.0000 | 0.8876 | 0.8876 | 0.9101 |
| mpt | 87 | 0.9195 | 0.9770 | 0.9195 | 0.9310 | 0.9425 |

## Leave-one-generator-out (heuristic stage retrained without each family)

Recall on the held-out family's AI texts in `test`; F1 on those texts + all test humans.

| Held-out family | n AI | Recall (LOGO) | F1 (LOGO) | F1 (trained with family) | Transformer F1 (LOGO) | Cascade thorough F1 (LOGO) |
|---|---|---|---|---|---|---|
| bloomz | 122 | 0.3197 | 0.3662 | 0.6949 | - | - |
| cohere | 248 | 0.8911 | 0.8185 | 0.8000 | - | - |
| dolly | 218 | 0.7156 | 0.6857 | 0.7213 | - | - |
| flant5 | 41 | 1.0000 | 0.5290 | 0.4659 | - | - |
| gpt | 1195 | 0.9397 | 0.9493 | 0.9478 | - | - |
| llama | 89 | 0.8090 | 0.6316 | 0.6031 | 0.6980 | 0.7536 |
| mpt | 87 | 0.8621 | 0.6608 | 0.6130 | - | - |

## Calibration ablation (RQ2, val split): cascade F1 at matched escalation rate

| Escalation rate | Calibrated | Raw score |
|---|---|---|
| 0.05 | 0.9724 | 0.9751 |
| 0.10 | 0.9819 | 0.9821 |
| 0.20 | 0.9863 | 0.9873 |
| 0.30 | 0.9871 | 0.9878 |
| 0.40 | 0.9871 | 0.9879 |
| 0.50 | 0.9871 | 0.9879 |

## Latency (end to end from clean text, after 20 warm-up docs)

| Device | System | Mean ms | p95 ms | Docs/s | Batch 8 docs/s | Batch 32 docs/s | Escalation |
|---|---|---|---|---|---|---|---|
| cpu | heuristic | 353.9 | 788.1 | 2.83 | 2.83 | 2.86 | 0.00 |
| cpu | transformer | 174.7 | 477.3 | 5.72 | 4.15 | 5.36 | 1.00 |
| cpu | cascade_fast | 358.8 | 828.7 | 2.79 | 2.78 | 2.79 | 0.00 |
| cpu | cascade_balanced | 360.0 | 821.9 | 2.78 | 2.88 | 2.91 | 0.00 |
| cpu | cascade_thorough | 362.5 | 759.1 | 2.76 | 2.75 | 2.75 | 0.07 |
| cuda | heuristic | 73.0 | 133.0 | 13.69 | 13.73 | 14.07 | 0.00 |
| cuda | transformer | 10.0 | 14.3 | 100.48 | 203.55 | 252.35 | 1.00 |
| cuda | cascade_fast | 71.8 | 128.7 | 13.93 | 13.77 | 13.71 | 0.00 |
| cuda | cascade_balanced | 72.6 | 140.0 | 13.78 | 13.96 | 14.01 | 0.00 |
| cuda | cascade_thorough | 73.6 | 134.5 | 13.59 | 13.82 | 13.90 | 0.07 |

## Figures of merit (Eq. 4.8)

- cpu fast: Recovery = 0.009, Speed-up = 0.49x
- cpu balanced: Recovery = 0.085, Speed-up = 0.49x
- cpu thorough: Recovery = 0.878, Speed-up = 0.48x
- cuda fast: Recovery = 0.009, Speed-up = 0.14x
- cuda balanced: Recovery = 0.085, Speed-up = 0.14x
- cuda thorough: Recovery = 0.878, Speed-up = 0.14x

## Model footprint

- heuristic head: 1.62 MB on disk
- reference LM (gpt2): 548.0 MB on disk, 124.4M parameters
- transformer: 480.12 MB on disk, 124.6M parameters
