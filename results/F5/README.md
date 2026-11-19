# F5: Vision experiments (Feba)

1. **What ran:**
   - ResNet-50 and ViT-B/16 detectors (`src/vision/train_detector.py`) on GenImage for the seen, unseen-BigGAN and unseen-SD settings, seed 42; seen also with seeds 43 and 44.
   - Evaluation of every detector and the F3 CLIP probe, clean and under JPEG 50, JPEG 70 and a 20% crop (`src/vision/evaluate_detectors.py`) → `cnn_vit.csv`.
   - Grad-CAM on 8 validation images, 3 real, 3 SD v1.4, 2 BigGAN (`src/vision/gradcam.py`) → `gradcam_*.csv`, `reports/figures/fig_gradcam_resnet50.png`. The ViT-B/16 heatmaps from this run were empty (wrong target layer, fixed in `src/vision/detectors.py`); `fig_gradcam_vit_b16.png` is added after the rerun.
   - ONNX export and latency of the two seen seed-42 models (`src/vision/export_onnx.py`) → `onnx_latency.csv`.
   - Run with `notebooks/F5_vision_experiments_colab.ipynb` on an NVIDIA L4.
2. **Settings:** ImageNet weights, last 4 blocks + head trainable, 3 epochs, AdamW lr 1e-4, weight decay 0.05, cosine schedule, batch 64, mixed precision; augmentation: random resized crop (0.6–1), horizontal flip, JPEG 60–95, blur (p 0.3). Train 15,000 images (seen) or 10,000 (unseen); test is the GenImage validation split, 3,000 or 2,000 images.
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

   Unseen SD, accuracy on SD images (clean): ResNet-50 0.003, ViT-B/16 0.069, CLIP probe 0.015.

   ONNX (`onnx_latency.csv`, batch 1, 224×224): ResNet-50 94.0 MB, 27.96 ms on CPU (ONNX Runtime), 6.68 ms on GPU (PyTorch); ViT-B/16 343.6 MB, 92.63 ms CPU, 7.10 ms GPU. Max difference from PyTorch: 2.4e-6 and 1.9e-6.
5. **Where files are:** this folder holds the tables and per-run JSON. Checkpoints (`<arch>_<setting>_s<seed>.pt`) and ONNX files are in Drive `MyDrive/twincheck/models/F5/` (too big for git).
