# F5: Vision experiments (Feba)

1. **What ran:** `notebooks/F5_vision_experiments_colab.ipynb` on Colab (T4). It uses the models, augmentation, corruptions, metrics and Grad-CAM code in `src/vision/` (`detectors.py`, `train_detector.py`, `gradcam.py`).
   - ResNet-50 and ViT-B/16 AI-image detectors, trained and saved per epoch.
   - Both detectors and the F3 CLIP probe on validation and test, clean and under JPEG 50, JPEG 70 and a 20% centre crop → `cnn_vit.csv`.
   - Grad-CAM on 8 images (3 real and 3 Stable Diffusion v1.4 from test, 2 BigGAN from validation) → `gradcam_<arch>.csv`, `reports/figures/fig_gradcam_<arch>.png`.
   - ONNX export, FP32 and INT8 (dynamic quantization), checked against PyTorch, with latency → `onnx_latency.csv`, `onnx_int8.csv`.

2. **Settings:**
   - Data: the F1 GenImage split (train 4,000 real + 5,000 BigGAN, validation 1,000 + 1,000 BigGAN, test 1,000 real + 1,000 Stable Diffusion v1.4).
   - Input: the F3 preprocessing for every image (centre crop, 96 px, JPEG with real-photo settings, greyscale share matched).
   - Models: ImageNet ResNet-50 and ViT-B/16, one output (P(AI)), last 4 blocks + head trained.
   - Training: 3 epochs, AdamW (lr 1e-4, weight decay 0.05), cosine schedule, batch 64, fp16, seed 42.
   - Augmentation: random resized crop (0.6–1), horizontal flip, JPEG 60–95, Gaussian blur (p 0.3).
   - CLIP probe: logistic regression on the F3 CLIP ViT-L/14 embeddings, same images.
   - Metrics: threshold 0.5; precision, recall and F1 are for the AI class.

3. **W&B:** not used; every metric is in the files below.

4. **Key numbers:**

   Validation, clean (real + BigGAN):

   | Model | Accuracy | Precision | Recall | F1 | AUROC |
   |---|---:|---:|---:|---:|---:|
   | ResNet-50 | 0.842 | 0.997 | 0.687 | 0.813 | 0.982 |
   | ViT-B/16 | 0.992 | 0.992 | 0.991 | 0.991 | 1.000 |
   | CLIP probe (F3) | 0.958 | 0.940 | 0.980 | 0.959 | 0.992 |

   Test, clean (real + Stable Diffusion v1.4, unseen generator):

   | Model | Accuracy | Precision | Recall | F1 | AUROC |
   |---|---:|---:|---:|---:|---:|
   | ResNet-50 | 0.504 | 0.765 | 0.013 | 0.026 | 0.610 |
   | ViT-B/16 | 0.555 | 0.887 | 0.126 | 0.221 | 0.754 |
   | CLIP probe (F3) | 0.542 | 0.684 | 0.156 | 0.254 | 0.620 |

   Training (loss / validation accuracy per epoch):

   | Model | Epoch 1 | Epoch 2 | Epoch 3 |
   |---|---|---|---|
   | ResNet-50 | 0.301 / 0.802 | 0.097 / 0.820 | 0.072 / 0.842 |
   | ViT-B/16 | 0.142 / 0.981 | 0.027 / 0.990 | 0.007 / 0.992 |

   Robustness, test (accuracy / AUROC):

   | Model | Clean | JPEG 50 | JPEG 70 | Crop 20% |
   |---|---|---|---|---|
   | ResNet-50 | 0.504 / 0.610 | 0.506 / 0.599 | 0.511 / 0.626 | 0.506 / 0.582 |
   | ViT-B/16 | 0.555 / 0.754 | 0.566 / 0.724 | 0.563 / 0.745 | 0.559 / 0.736 |
   | CLIP probe (F3) | 0.542 / 0.620 | 0.525 / 0.578 | 0.526 / 0.577 | 0.547 / 0.598 |

   Robustness, validation (accuracy / AUROC):

   | Model | Clean | JPEG 50 | JPEG 70 | Crop 20% |
   |---|---|---|---|---|
   | ResNet-50 | 0.842 / 0.982 | 0.822 / 0.965 | 0.846 / 0.982 | 0.874 / 0.985 |
   | ViT-B/16 | 0.992 / 1.000 | 0.977 / 0.997 | 0.985 / 0.998 | 0.988 / 0.999 |
   | CLIP probe (F3) | 0.958 / 0.992 | 0.908 / 0.962 | 0.910 / 0.971 | 0.946 / 0.985 |

   ONNX, batch 1, 224×224 (Colab CPU, 2 threads; GPU T4):

   | Model | Size | CPU ms | GPU ms (PyTorch) | Same decision as PyTorch | Test AUROC (500 images) |
   |---|---:|---:|---:|---:|---:|
   | ResNet-50 FP32 | 94.0 MB | 96.2 | 8.0 | 100% | 0.626 |
   | ResNet-50 INT8 | 23.7 MB | 141.4 | – | 99.0% | 0.826 |
   | ViT-B/16 FP32 | 343.6 MB | 481.7 | 15.1 | 100% | 0.771 |
   | ViT-B/16 INT8 | 87.3 MB | 432.7 | – | 98.2% | 0.770 |

   Grad-CAM P(AI): real 0.000–0.002; Stable Diffusion 0.001–0.010 (ResNet-50) and 0.000–0.198 (ViT-B/16); BigGAN 0.959 and 0.070 (ResNet-50), 1.000 and 0.995 (ViT-B/16).

5. **Findings:**
   - ViT-B/16 separates real from BigGAN almost perfectly; ResNet-50 misses 31% of BigGAN images.
   - On the unseen generator all three models call most images real. ViT-B/16 ranks them best (AUROC 0.754).
   - JPEG and cropping change test accuracy by at most 0.017.
   - INT8 cuts file size by about 4 times. It keeps ViT-B/16 scores, but changes ResNet-50 scores a lot and makes it slower.

6. **Limitations:**
   - One seed (42) and one unseen generator.
   - 96 px input gives the models less detail.
   - The INT8 check uses 500 of the 2,000 test images (250 real, 250 SD, seed 42).

7. **Where files are:**
   - `results/F5/`: `cnn_vit.csv`, `train_resnet50_s42.json`, `train_vit_b16_s42.json`, `gradcam_resnet50.csv`, `gradcam_vit_b16.csv`, `onnx_latency.csv`, `onnx_int8.csv`.
   - `reports/figures/fig_gradcam_resnet50.png`, `reports/figures/fig_gradcam_vit_b16.png`.
   - On Drive (not in git): checkpoints and ONNX files in `MyDrive/TwinCheck/models/F5/`, prepared images in `MyDrive/TwinCheck/data/processed/vision/genimage_debiased96_jpeg.pkl`.
