# F10: Vision strengthening (Feba)

1. **What ran:** `notebooks/Feba_F10_strengthen.ipynb` on Colab (T4), using `src/vision/leakage_cv.py`, `f10_stats.py`, `quantize_static.py` and the new `--cache`, `--freeze-bn` and `--eval-only` options in `train_detector.py`.
   - First I re-scored the F5 seed-42 checkpoints through the new cached input. They match `results/F5/cnn_vit.csv` exactly (`rescore_check.csv`, all differences below 1e-16). That makes every run below directly comparable with F5.
   - Leakage test: 5-fold cross-validation over all 218 VERITE photos instead of the 36 fixed test pairs. It reproduces F3 exactly on the fixed split (`leakage_f3_check.csv`, same verdict on every row).
   - ResNet-50 and ViT-B/16 retrained with seeds 43 and 44, all other settings as in F5.
   - ResNet-50 seed 42 retrained with its frozen BatchNorm layers kept in eval mode (`--freeze-bn`). I also re-scored the F5 checkpoint with the ImageNet BatchNorm statistics put back in the frozen layers, with no retraining.
   - Static per-channel INT8 (ONNX Runtime, QDQ), calibrated on 200 validation images and checked on the same 500 test images as F5.

2. **Settings:** same data, input, model, optimiser and augmentation as F5 (see `results/F5/README.md`). The leakage CV uses folds grouped by photo, so a photo's two captions are always in the same fold, and seeds 42/43/44 change only the fold split. Bootstrap CIs use 1,000 resamples over photos (leakage) or test images (AUROC). McNemar is the exact test. The only cut-off that was tuned was picked on validation; test was never used to choose anything.

3. **W&B:** not used (team decision). Every number is in the CSV and JSON files here. The new training logs are in `results/F5/train_*_s43.json`, `train_*_s44.json` and `results/F10/train_resnet50_s42_freezebn.json`.

4. **Key numbers:**

   Leakage, all 218 photos (seed 42; mean ± sd over three fold splits in brackets):

   | Model | Changed verdicts | ALR | 95% CI |
   |---|---:|---:|---:|
   | Decoupled | 29 | 13.3% (13.3 ± 0.0) | 9.2–18.3 |
   | Fused | 40 | 18.3% (17.9 ± 0.8) | 13.3–23.4 |
   | Fused + image embedding | 37 | 17.0% (16.8 ± 0.7) | 12.4–22.0 |

   - Decoupled vs fused, exact McNemar: 2 vs 13 changed only by one model, p = 0.007 (seed 42). The other two fold splits give p = 0.057 and 0.013.
   - The authenticity score barely moves for most photos (median shift +0.005; 12% move by more than 0.1). Of the 13 photos only the fused model flips, 8 (62%) moved by more than 0.1. Flipped pairs have tiny caption gaps (median 0.004 vs 0.014 overall).
   - The decoupled model gives the same answer for every fold split: its verdict only depends on which caption is more similar to the photo.

   Detectors over three seeds (mean ± sd):

   | Model | Val. accuracy | Val. AUROC | Test accuracy | Test AUROC | Test recall |
   |---|---:|---:|---:|---:|---:|
   | ResNet-50 | 0.840 ± 0.015 | 0.983 ± 0.001 | 0.505 ± 0.001 | 0.598 ± 0.018 | 0.014 ± 0.001 |
   | ViT-B/16 | 0.991 ± 0.001 | 0.999 ± 0.000 | 0.562 ± 0.008 | 0.761 ± 0.010 | 0.141 ± 0.019 |

   - ViT-B/16 beats ResNet-50 on test AUROC for every seed: +0.144, +0.166 and +0.180, each with a 95% CI above 0.11 and bootstrap p < 0.001.

   ResNet-50 BatchNorm (seed 42, cut-off 0.5):

   | Run | Val. accuracy | BigGAN found | Test AUROC | Test recall |
   |---|---:|---:|---:|---:|
   | F5 as trained | 0.843 | 68.7% | 0.610 | 0.013 |
   | F5 + ImageNet BatchNorm statistics (no retraining) | 0.899 | 80.2% | 0.655 | 0.044 |
   | Retrained with frozen BatchNorm | 0.983 | 98.5% | 0.714 | 0.149 |

   - The weak F5 ResNet-50 was mostly caused by the frozen BatchNorm layers updating their statistics during training.
   - With the fix, the cut-off chosen on validation is 0.51, which gives the same test result as 0.5. On the F5 model it was 0.04.

   Static INT8 for ResNet-50 (500 test images, CPU, 2 threads):

   | Checkpoint | FP32 AUROC | INT8 AUROC | Same decision | Score correlation | Size |
   |---|---:|---:|---:|---:|---:|
   | F5 seed 42 | 0.626 | 0.460 | 98.8% | 0.22 | 94 → 24 MB |
   | Frozen BatchNorm | 0.702 | 0.666 | 94.2% | 0.80 | 94 → 24 MB |

   - Static INT8 doesn't give a faithful ResNet-50. Keep FP32 ONNX for ResNet-50. ViT-B/16 dynamic INT8 from F5 is still fine.

5. **Files:** everything is in this folder. The `preds_*.csv` files hold one score per image (md5, split, generator, P(AI)). The checkpoints stay on Drive (`MyDrive/TwinCheck/models/F10/`); they are too big for git.
