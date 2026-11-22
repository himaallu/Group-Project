# F6: Vision error analysis, 8 cases with Grad-CAM (Feba)

1. **What ran:**
   - `notebooks/F6_error_cases_colab.ipynb` (Colab, T4) loads the two F5 detectors (`resnet50_s42.pt`, `vit_b16_s42.pt`) and scores all 4,000 validation and test images → `predictions.csv`.
   - It checks these scores against `results/F5/cnn_vit.csv` → `check_against_f5.csv`.
   - It counts errors per model and group, compares the two models on the unseen generator, and groups false positives on real photos by original size → `error_summary.csv`, `error_overlap.csv`, `real_errors_by_original_size.csv`.
   - It records the original format and size of every class → `data_audit.csv`.
   - It picks 8 test cases by fixed rules and runs Grad-CAM with both models → `cases.csv`, `reports/figures/fig_f6_error_cases.png`.
   - The notebook uses `src/vision/detectors.py`, `train_detector.py` (`metrics`) and `gradcam.py` unchanged.

2. **Settings:**
   - Data: the F1 GenImage split. Validation is 1,000 real + 1,000 BigGAN; test is 1,000 real + 1,000 Stable Diffusion v1.4 (the unseen generator).
   - Input: the same 96 px images as F5 (F3 preprocessing: centre crop, 96 px, JPEG with real-photo settings, greyscale share matched), read from the F5 cache on Drive; resize to 256, centre crop to 224, fp16.
   - Models: F5 ResNet-50 and ViT-B/16, seed 42. Threshold 0.5 on P(AI).
   - Case rules (test split, fixed before viewing the images):

     | Cases | Rule |
     |---|---|
     | 1–2 | Stable Diffusion images missed by both models, lowest mean P(AI) |
     | 3–4 | Missed by both models, highest of the two scores |
     | 5–6 | Missed by ResNet-50, caught by ViT-B/16, highest ViT-B/16 score |
     | 7–8 | Real images with the highest score from either model |

3. **W&B:** not used. Every number is in the CSV files below.

4. **Key numbers:**
   - The scores reproduce F5: the largest difference in accuracy, precision, recall, F1 or AUROC is below 1e-16.

   Errors:

   | Split | Group | ResNet-50 | ViT-B/16 |
   |---|---|---:|---:|
   | Validation | BigGAN missed | 313 of 1,000 | 9 of 1,000 |
   | Validation | Real flagged as AI | 2 of 1,000 | 8 of 1,000 |
   | Test | Stable Diffusion missed | 987 of 1,000 | 874 of 1,000 |
   | Test | Real flagged as AI | 4 of 1,000 | 16 of 1,000 |

   - Stable Diffusion test images: missed by both 872, caught by ViT-B/16 only 115, by ResNet-50 only 2, by both 11.

   Real photos flagged as AI, by the shorter side of the original file:

   | Split | Shorter side | Real photos | ResNet-50 | ViT-B/16 |
   |---|---|---:|---:|---:|
   | Validation | under 128 px | 10 | 2 | 0 |
   | Validation | 128–223 px | 42 | 0 | 1 |
   | Validation | 224–511 px | 897 | 0 | 7 |
   | Validation | 512 px and over | 51 | 0 | 0 |
   | Test | under 128 px | 15 | 1 | 1 |
   | Test | 128–223 px | 49 | 0 | 1 |
   | Test | 224–511 px | 893 | 2 | 12 |
   | Test | 512 px and over | 43 | 1 | 2 |

   The 8 cases (P(AI)):

   | Case | Truth | Original file | ResNet-50 | ViT-B/16 |
   |---|---|---|---:|---:|
   | 1 | Stable Diffusion | 512×512 PNG | 0.0000 | 0.0002 |
   | 2 | Stable Diffusion | 512×512 PNG | 0.0000 | 0.0003 |
   | 3 | Stable Diffusion | 512×512 PNG | 0.0778 | 0.4910 |
   | 4 | Stable Diffusion | 512×512 PNG | 0.0012 | 0.4831 |
   | 5 | Stable Diffusion | 512×512 PNG | 0.0687 | 0.9996 |
   | 6 | Stable Diffusion | 512×512 PNG | 0.0344 | 0.9995 |
   | 7 | real | 500×375 JPEG | 0.0127 | 0.9983 |
   | 8 | real | 500×333 JPEG | 0.0066 | 0.9948 |

   - Grad-CAM recomputes P(AI) in full precision; it differs from the scoring run by less than 0.001.
   - Data audit (original files): AI images are PNG with one size per generator (BigGAN 128×128, Stable Diffusion 512×512); real images are JPEG of many sizes (51 to 4,272 px), most of them not square. The detectors never see these files, only the 96 px input.

5. **Where files are:**
   - `results/F6/`: `predictions.csv`, `check_against_f5.csv`, `error_summary.csv`, `error_overlap.csv`, `real_errors_by_original_size.csv`, `data_audit.csv`, `cases.csv`.
   - `reports/figures/fig_f6_error_cases.png`: the 8 images, with a ResNet-50 and a ViT-B/16 Grad-CAM row.
   - The 8 original images are on Drive (`MyDrive/TwinCheck/results/F6/cases/`), not in git (GenImage licence CC BY-NC-SA 4.0).

**Limitations:**
- One seed and one unseen generator.
- The 8 cases show types of error; they do not estimate error rates.
- The size groups under 128 px are small (10 and 15 photos).
- Grad-CAM shows where a decision's evidence comes from, not why the model uses it.
