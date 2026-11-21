# F6: Vision error analysis, 8 cases with Grad-CAM (Feba)

1. **What ran:**
   - `notebooks/F6_error_cases_colab.ipynb` (Colab, NVIDIA L4) loads the two F5 detectors (`resnet50_s42.pt`, `vit_b16_s42.pt`) and scores all 4,000 validation and test images → `predictions.csv`.
   - It checks these scores against `results/F5/cnn_vit.csv` → `check_against_f5.csv`.
   - It picks 8 test cases by fixed rules and runs Grad-CAM on them with both models → `cases.csv`, `reports/figures/fig_f6_error_cases.png`.
   - The notebook uses `src/vision/detectors.py`, `train_detector.py` (`metrics`) and `gradcam.py` unchanged.

2. **Settings:**
   - Data: the F1 GenImage split. Validation is 1,000 real + 1,000 BigGAN; test is 1,000 real + 1,000 Stable Diffusion v1.4 (the unseen generator).
   - Models: F5 ResNet-50 and ViT-B/16, seed 42. Threshold 0.5 on P(AI).
   - Preprocessing: resize to 256, centre crop to 224, mixed precision. This is the same as `predict` in F5.
   - Grad-CAM: the F5 target layers. Grad-CAM recomputes P(AI) in full precision, so it differs from the scoring run only in the third or fourth decimal.
   - Case rules (test split, fixed before viewing the images):

     | Cases | Rule |
     |---|---|
     | 1–2 | Stable Diffusion images missed by both models, lowest mean P(AI) |
     | 3–4 | Missed by both models, highest of the two scores |
     | 5–6 | Missed by ResNet-50, caught by ViT-B/16, highest ViT-B/16 score |
     | 7–8 | Real images with the highest score from either model |

3. **W&B:** not logged. Every number is in the CSV files below.

4. **Key numbers:**
   - The scores reproduce F5 exactly: the maximum difference is 0.0 for accuracy, precision, recall, F1 and AUROC, for both models and both splits.

   Errors on the test split:

   | | ResNet-50 | ViT-B/16 |
   |---|---|---|
   | Real images flagged as AI | 2 of 1,000 | 4 of 1,000 |
   | Stable Diffusion images missed | 988 of 1,000 | 920 of 1,000 |

   - Missed by both models: 919. Caught by ViT-B/16 only: 69. Caught by ResNet-50 only: 1. Caught by both: 11.

   The 8 cases (P(AI)):

   | Case | Truth | Size, format | ResNet-50 | ViT-B/16 |
   |---|---|---|---|---|
   | 1 | Stable Diffusion | 512×512 png | 0.0002 | 0.0001 |
   | 2 | Stable Diffusion | 512×512 png | 0.0001 | 0.0002 |
   | 3 | Stable Diffusion | 512×512 png | 0.0015 | 0.4969 |
   | 4 | Stable Diffusion | 512×512 png | 0.0124 | 0.4906 |
   | 5 | Stable Diffusion | 512×512 png | 0.0110 | 0.9995 |
   | 6 | Stable Diffusion | 512×512 png | 0.0976 | 0.9993 |
   | 7 | real | 560×457 jpeg | 0.1182 | 0.9765 |
   | 8 | real | 151×100 jpeg | 0.9301 | 0.0121 |

   - Image format and size, from `predictions.csv` and the manifests:
     - In every split, the AI images are PNG with one size per generator (BigGAN 128×128, Stable Diffusion 512×512).
     - All real images are JPEG of varying size.

5. **Where files are:**
   - `results/F6/`: `predictions.csv` (P(AI) of both models, size and format for every validation and test image), `cases.csv` (the 8 cases, with both P(AI) values and the Grad-CAM P(AI)), `check_against_f5.csv`.
   - `reports/figures/fig_f6_error_cases.png`: the 8 images, with a ResNet-50 and a ViT-B/16 Grad-CAM row.
   - The 8 original images are on Drive (`MyDrive/twincheck/F6/cases/`), not in git (GenImage licence CC BY-NC-SA 4.0).

**Limitations:**
- One seed and one unseen generator.
- The 8 cases show types of error; they do not estimate error rates.
- Grad-CAM shows where a decision's evidence comes from, not why the model uses it.
- Whether the detectors rely on image size or resampling has not been tested.
