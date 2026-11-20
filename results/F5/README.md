# F5: Vision experiments (Feba)

1. **What ran:**
   - `src/vision/train_detector.py`: ResNet-50 and ViT-B/16 AI-image detectors, seed 42.
   - `src/vision/evaluate_detectors.py`: both detectors on the validation and test splits, clean and under JPEG 50, JPEG 70 and a 20% centre crop → `cnn_vit.csv`.
   - `src/vision/gradcam.py`: Grad-CAM on 8 images (3 real and 3 Stable Diffusion v1.4 from test, 2 BigGAN from validation) → `gradcam_<arch>.csv`, `reports/figures/fig_gradcam_<arch>.png`.
   - `src/vision/export_onnx.py`: ONNX export, check against PyTorch and latency → `onnx_latency.csv`.
   - All steps ran on Colab (NVIDIA L4) with `notebooks/F5_vision_experiments_colab.ipynb`.

2. **Settings:**
   - Data: the F1 GenImage split (`data/processed/vision/genimage_{train,validation,test}_manifest.csv`), the same split as F3.

     | Split | Real | AI-generated | Total |
     | :---- | ----: | :---- | ----: |
     | Train | 4,000 | 5,000 BigGAN | 9,000 |
     | Validation | 1,000 | 1,000 BigGAN | 2,000 |
     | Test (unseen generator) | 1,000 | 1,000 Stable Diffusion v1.4 | 2,000 |

   - Models: ImageNet-pretrained ResNet-50 and ViT-B/16 with one output (P(AI)); only the last 4 blocks and the head are trained.
   - Training: 3 epochs, AdamW (lr 1e-4, weight decay 0.05), cosine schedule, batch 64, mixed precision.
   - Augmentation: random resized crop (scale 0.6–1), horizontal flip, JPEG quality 60–95, Gaussian blur (p 0.3).
   - Metrics: threshold 0.5 on P(AI); precision, recall and F1 are for the AI class.

3. **W&B:** not logged (`--no-wandb`); every metric is in the JSON and CSV files below.

4. **Key numbers:**

   Validation, clean (real + BigGAN):

   | Model | Accuracy | Precision | Recall | F1 | AUROC |
   |---|---|---|---|---|---|
   | ResNet-50 | 0.991 | 1.000 | 0.981 | 0.990 | 1.000 |
   | ViT-B/16 | 0.997 | 0.999 | 0.994 | 0.996 | 1.000 |

   Training per epoch (training loss / validation accuracy, from the notebook output):

   | Model | Epoch 1 | Epoch 2 | Epoch 3 |
   |---|---|---|---|
   | ResNet-50 | 0.161 / 0.985 | 0.030 / 0.990 | 0.019 / 0.991 |
   | ViT-B/16 | 0.066 / 0.996 | 0.007 / 0.995 | 0.003 / 0.997 |

   Test, clean (real + Stable Diffusion v1.4, unseen generator):

   | Model | Accuracy | Precision | Recall | F1 | AUROC |
   |---|---|---|---|---|---|
   | ResNet-50 | 0.505 | 0.857 | 0.012 | 0.024 | 0.713 |
   | ViT-B/16 | 0.538 | 0.952 | 0.080 | 0.148 | 0.885 |

   Robustness, test split (accuracy / AUROC / recall):

   | Model | Clean | JPEG 50 | JPEG 70 | Crop 20% |
   |---|---|---|---|---|
   | ResNet-50 | 0.505 / 0.713 / 0.012 | 0.512 / 0.716 / 0.027 | 0.512 / 0.721 / 0.026 | 0.507 / 0.662 / 0.014 |
   | ViT-B/16 | 0.538 / 0.885 / 0.080 | 0.548 / 0.888 / 0.102 | 0.545 / 0.892 / 0.095 | 0.534 / 0.871 / 0.075 |

   Validation accuracy stays between 0.985 and 0.999 under every corruption (`cnn_vit.csv`).

   ONNX, batch 1, 224×224:

   | Model | ONNX size | Max difference vs PyTorch | CPU, ONNX Runtime | GPU, PyTorch (L4) |
   |---|---|---|---|---|
   | ResNet-50 | 94.0 MB | 6.0e-07 | 15.51 ms | 6.66 ms |
   | ViT-B/16 | 343.6 MB | 2.4e-06 | 85.73 ms | 7.10 ms |

   Grad-CAM P(AI): BigGAN 0.982–1.000; real 0.000–0.007; Stable Diffusion v1.4 0.000–0.055, so all 3 SD images are missed by both models.

5. **Findings:**
   - Both detectors separate real from BigGAN almost perfectly.
   - On the unseen generator, almost every Stable Diffusion image is called real (recall 0.012 and 0.080), so accuracy is close to 0.5.
   - ViT-B/16 ranks the unseen images better than ResNet-50 (AUROC 0.885 vs 0.713), but the 0.5 threshold chosen on BigGAN does not transfer.
   - JPEG and cropping change test accuracy by less than 0.01, so they do not explain the drop; cropping lowers ResNet-50 test AUROC from 0.713 to 0.662.

6. **Limitations:**
   - One seed (42) and one unseen generator.
   - The CLIP probe was not rerun here (`--skip-clip`); its result on the same split is in `results/F3/`.
   - In `gradcam_<arch>.csv`, `image_id` is the image path. The code in `src/vision/` now uses the MD5, but it selects the same 8 images.

7. **Where files are:**
   - `results/F5/`: `cnn_vit.csv`, `train_resnet50_s42.json`, `train_vit_b16_s42.json`, `gradcam_resnet50.csv`, `gradcam_vit_b16.csv`, `onnx_latency.csv`.
   - `reports/figures/fig_gradcam_resnet50.png`, `reports/figures/fig_gradcam_vit_b16.png`.
   - Checkpoints and ONNX files (`resnet50_s42.pt`, `vit_b16_s42.pt`, `resnet50_s42.onnx`, `vit_b16_s42.onnx`) are on Drive in `MyDrive/twincheck/models/F5_corrected_split/` (not in git).
