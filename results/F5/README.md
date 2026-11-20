# F5: Vision experiments (Feba)

1. **What ran:**
   - ResNet-50 and ViT-B/16 detectors (`src/vision/train_detector.py`) on GenImage in three settings, seed 42; the seen setting also with seeds 43 and 44.
   - Evaluation of every detector and a CLIP ViT-L/14 probe, clean and under JPEG 50, JPEG 70 and a 20% centre crop (`src/vision/evaluate_detectors.py`) → `cnn_vit.csv`.
   - Grad-CAM on 8 GenImage validation-folder images, 3 real, 3 SD v1.4, 2 BigGAN, for the seen seed-42 models (`src/vision/gradcam.py`) → `gradcam_*.csv`, `reports/figures/fig_gradcam_resnet50.png`.
   - ONNX export and latency of the two seen seed-42 models (`src/vision/export_onnx.py`) → `onnx_latency.csv`.
   - Run with `notebooks/F5_vision_experiments_colab.ipynb` on an NVIDIA L4.
2. **Settings:**
   - Images come from the GenImage train folder (training) and validation folder (testing); no image is in both (MD5 check on all 18,000).

     | Setting | Train (train folder) | Test (validation folder) |
     |---|---|---|
     | Seen | real + BigGAN + SD v1.4 (15,000) | real + BigGAN + SD v1.4 (3,000) |
     | Unseen BigGAN | real + SD v1.4 (10,000) | real + BigGAN (2,000) |
     | Unseen SD | real + BigGAN (10,000) | real + SD v1.4 (2,000) |

   - Unseen SD follows the project split (BigGAN seen, Stable Diffusion v1.4 unseen). Its images differ from the F1 manifests: it trains on all 5,000 train-folder real images and tests on the 1,000 validation-folder real images, so its numbers are not directly comparable with F3. The seen setting trains on Stable Diffusion and is an in-distribution reference only.
   - Detectors: ImageNet weights, last 4 blocks + head trainable, 3 epochs, AdamW lr 1e-4, weight decay 0.05, cosine schedule, batch 64, mixed precision; augmentation: random resized crop (0.6–1), horizontal flip, JPEG 60–95, blur (p 0.3). There is no separate validation split: training runs a fixed 3 epochs, the test images are scored after each epoch (logged as `val_*` in W&B), and the last epoch is reported.
   - CLIP probe: `src/vision/probe.py` (logistic regression, C by 5-fold CV, balanced classes) on the cached F2 embeddings of the training images; test images are re-embedded after each corruption.
   - Accuracy uses a 0.5 threshold for all models.
3. **W&B:** project `twincheck`, runs `F5-<arch>_<setting>_s<seed>`, logged online. Each run's link is in its `train_*.json`.
4. **Key numbers** (`cnn_vit.csv`, accuracy / AUROC):

   | Setting | ResNet-50 | ViT-B/16 | CLIP probe |
   |---|---|---|---|
   | Seen, clean (seeds 42/43/44) | 0.974 ± 0.002 / 0.996 ± 0.000 | 0.984 ± 0.002 / 0.998 ± 0.000 | 0.988 / 0.999 |
   | Unseen BigGAN, clean | 0.725 / 0.890 | 0.785 / 0.902 | 0.860 / 0.981 |
   | Unseen SD, clean | 0.501 / 0.708 | 0.534 / 0.881 | 0.507 / 0.870 |
   | Seen, JPEG 50 (seed 42) | 0.978 / 0.996 | 0.985 / 0.998 | 0.820 / 0.992 |
   | Unseen BigGAN, JPEG 50 | 0.655 / 0.864 | 0.750 / 0.904 | 0.515 / 0.848 |
   | Seen, crop 20% (seed 42) | 0.972 / 0.995 | 0.984 / 0.998 | 0.988 / 0.999 |

   - Unseen SD: at the 0.5 threshold almost no SD image is flagged (accuracy on SD images: ResNet-50 0.003, ViT-B/16 0.069, CLIP probe 0.015), although AUROC is 0.708, 0.881 and 0.870. ViT-B/16 transfers better than ResNet-50 in both unseen settings.
   - Robustness: the fine-tuned detectors, trained with JPEG augmentation, change by less than 1 point under JPEG 50/70 and crop in the seen setting. The CLIP probe drops under JPEG 50 (seen accuracy 0.988 → 0.820; BigGAN images 0.998 → 0.544).
   - Grad-CAM (`gradcam_resnet50.csv`): the seen ResNet-50 gives P(AI) ≤ 0.02 for the 3 real images and ≥ 0.99 for the 5 AI images.
   - ONNX (`onnx_latency.csv`, batch 1, 224×224): ResNet-50 94.0 MB, 27.96 ms on CPU (ONNX Runtime), 6.68 ms on GPU (PyTorch); ViT-B/16 343.6 MB, 92.63 ms CPU, 7.10 ms GPU. Max difference from PyTorch: 2.4e-6 and 1.9e-6.
5. **Where files are:** this folder holds the tables and per-run JSON. Checkpoints (`<arch>_<setting>_s<seed>.pt`) and ONNX files are in Drive `MyDrive/twincheck/models/F5/` (too big for git).

**Limitations:**
- The unseen settings use seed 42 only.
- The ViT-B/16 Grad-CAM figure was not produced: the heatmaps from this run were empty because of the target layer, which `target_layer` in `src/vision/detectors.py` now sets to the last block's input LayerNorm. `gradcam_vit_b16.csv` (P(AI)) is valid; the figure needs a rerun of `python -m src.vision.gradcam`.
- Grad-CAM explains the detector's P(AI), and ONNX covers the two detectors; the CLIP image encoder + probe was not exported. Robustness is reported as accuracy and AUROC; ALR under the corruptions was not computed.
