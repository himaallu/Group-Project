# F1: Vision data (Feba)

1. **What ran:** `notebooks/Feba_F1_vision_data.ipynb` on Colab. It downloads VERITE and GenImage, checks every file, splits both datasets and saves each GenImage image as the original file and as a 256×256 JPEG copy.
2. **Settings:** seed 42. VERITE is split by image (70 / 15 / 15). GenImage: train and validation are real + BigGAN; the test set is real + Stable Diffusion v1.4, a generator never seen in training.
3. **W&B:** no training in F1.
4. **Key numbers:**

   | Data | Train | Validation | Test |
   |---|---:|---:|---:|
   | VERITE records (images) | 704 (463) | 151 (100) | 146 (100) |
   | GenImage real | 4,000 | 1,000 | 1,000 |
   | GenImage AI | 5,000 BigGAN | 1,000 BigGAN | 1,000 SD v1.4 |

   - GenImage: 13,000 images, 13,000 unique MD5 hashes.
   - RED-DOT evidence: 1,352 files, 1,104 non-empty.
   - Original files: AI images are PNG at one size per generator (BigGAN 128×128, SD 512×512); real photos are JPEG of varying size.
5. **Where files are:**
   - `data/processed/vision/`: `genimage_{train,validation,test}_manifest.csv`, `verite_manifest.csv`, `verite.jsonl`, `red_dot_evidence_report.json`.
   - `results/F1/`: `f1_verification_results.json`, `final_dataset_distribution.csv`.
   - Images are on Drive (`MyDrive/TwinCheck/data/processed/vision/`), not in git (GenImage licence CC BY-NC-SA 4.0; VERITE is gated).

**Limitations:**
- Exact-duplicate check only (MD5), and one held-out generator.
- In the 256×256 copies, BigGAN images (128 px) sit inside a grey border because small images are not enlarged.
- Format and size differ by class in the original files, so detectors may learn them instead of signs of generation.
