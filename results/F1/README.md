# F1: Vision data (Feba)

1. **What ran:** `notebooks/Feba_F1_vision_data.ipynb` downloads and checks VERITE and GenImage and exports the GenImage images to Drive. `src/vision/genimage_splits.py` builds the GenImage experiment splits and the leakage report.
2. **Settings:** seed 42. VERITE split by image ID. GenImage: train and validation use real + BigGAN; Stable Diffusion v1.4 is the unseen-generator test set.
3. **W&B:** no training in F1.
4. **Key numbers:** VERITE 1,001 records / 663 images (train 701, validation 149, test 151). GenImage train 9,000 (4,000 real, 5,000 BigGAN), validation 2,000 (1,000 real, 1,000 BigGAN), test 2,000 (1,000 real, 1,000 SD v1.4). No MD5 overlap between splits.
5. **Where files are:** manifests and reports in `data/processed/vision/`. Images are on Drive in `data/processed/vision/genimage/` (not in git, licence CC-BY-NC-SA-4.0).
