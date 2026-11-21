# F2: Vision features and image twins (Feba)

1. **What ran:** `notebooks/Feba_F2_vision_features.ipynb` on Colab (T4). It extracts SigLIP 2 and CLIP ViT-L/14 embeddings for GenImage, VERITE and M4FC, makes VAE twins of VERITE and M4FC true images and SDXL img2img twins of GenImage test images, and keeps the twins that stay close to the original.
2. **Settings:**
   - SigLIP 2 `google/siglip2-so400m-patch14-384`, CLIP `openai/clip-vit-large-patch14`, FP16 inference.
   - VAE `stabilityai/sd-vae-ft-mse`, longest side 512.
   - SDXL `stabilityai/stable-diffusion-xl-base-1.0`, strength 0.25, 20 steps, guidance 5.0, 512 px.
   - Kept if SigLIP 2 cosine ≥ 0.92 (VERITE), CLIP and SigLIP 2 ≥ 0.92 (M4FC), SigLIP 2 ≥ 0.70 (SDXL). Faces removed by manual review of OpenCV-flagged images.
3. **W&B:** not used (no training).
4. **Key numbers:**

   | Embeddings | SigLIP 2 | CLIP |
   |---|---:|---:|
   | GenImage (F1 split) | 13,000 × 1152 | 13,000 × 768 |
   | VERITE images | 664 × 1152 | 664 × 768 |
   | VERITE captions | 1,001 × 1152 | – |
   | M4FC images | 603 × 1152 | 603 × 768 |

   | Twins | Candidates | Face exclusions | Kept |
   |---|---:|---:|---:|
   | VERITE VAE | 338 | 39 | 218 |
   | M4FC VAE | 155 | 11 | 21 |
   | GenImage SDXL | 500 | – | 440 (191 real, 249 SD) |

   - `vae.jsonl`: 436 `_V` records from the 218 VERITE twins (train 300, val 64, test 72).
5. **Where files are:**
   - `features/vision/`: embedding row manifests. The `.npy` arrays are on Drive (`MyDrive/TwinCheck/features/vision/`).
   - `data/copies/vision/`: twin tables, similarity tables, `manifests/vae_manifest.csv`, `vae.jsonl`.
   - `data/processed/vision/m4fc_f2_clean_manifest.csv`.
   - `results/F2/summaries/`: step summaries (JSON).
   - Images (originals and twins) stay on Drive, not in git.

**Limitations:**
- GenImage embeddings come from the 256×256 copies, where BigGAN images have a grey border.
- SDXL twins use a lower threshold (0.70) and include AI-image sources.
- M4FC twins have no `_V` records: Y1 labels their claims contradicted, in the test split.
- Only 30 VERITE pairs were checked by eye.
