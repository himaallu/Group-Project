# F3: Authenticity detector, image leakage test and baselines (Feba)

1. **What ran:**
   - `notebooks/Feba_F3_authenticity_leakage.ipynb` on Colab (T4). Every GenImage image gets the same preprocessing, CLIP embeds it, and a linear detector learns real vs AI. The notebook checks how much the file format, size and other simple cues could give away, then scores camera, AI and edited images and runs the VERITE leakage test.
   - `src/eval/baselines/{claim_only,similarity}.py` on the MultiFC and AVeriTeC test originals.
2. **Settings:**
   - GenImage split from F1: train 4,000 real + 5,000 BigGAN, validation 1,000 real + 1,000 BigGAN, test 1,000 real + 1,000 Stable Diffusion v1.4.
   - Preprocessing for every image: pixels only, centre square crop, 96×96 bicubic, JPEG with the compression settings of one of 500 real training photos, greyscale share matched between real and AI in each split.
   - CLIP `openai/clip-vit-large-patch14` (image tower, FP16, 768-d). Logistic regression and linear SVM (C = 1), seed 42. Tables use the default decision rule (probability 0.5, margin 0); `F3_probe_results.csv` also has a validation-tuned threshold.
   - VERITE: 218 photos with a VAE twin from F2 (train 150, validation 32, test 36). Decoupled, fused and fused + image embedding models, logistic regression, trained on original training photos only.
   - Claim-only: TF-IDF + logistic regression. Similarity: 12 features + LightGBM, seeds 42/43/44.
3. **W&B:** not used. The notebook runs the baselines with `--no-wandb`.
4. **Key numbers:**

   | Input | Model | BigGAN acc. | SD v1.4 acc. | SD v1.4 AUROC | SD v1.4 recall |
   |---|---|---:|---:|---:|---:|
   | File format and size only | LR | 1.000 | 1.000 | 1.000 | 1.000 |
   | Simple cues of the 96 px input | LR | 0.546 | 0.520 | 0.486 | 0.773 |
   | F2 embeddings | LR | 0.995 | 0.496 | 0.536 | 0.002 |
   | Detector | LR | 0.958 | 0.542 | 0.620 | 0.156 |
   | Detector | SVM | 0.981 | 0.556 | 0.666 | 0.152 |

   | VERITE test (36 pairs) | Original acc. | Twin acc. | ALR |
   |---|---:|---:|---:|
   | Decoupled | 52.8% | 52.8% | 11.1% |
   | Fused | 55.6% | 55.6% | 22.2% |
   | Fused + image embedding | 55.6% | 52.8% | 25.0% |

   - Flagged as AI: real test photos 7.2%, BigGAN 98.0%, SD v1.4 15.6%, SDXL edits 24.6%, VERITE photos 1.8%, VERITE twins 3.2%.
   - Baselines, MultiFC macro-F1: claim-only 0.482, similarity 0.385 ± 0.005.
5. **Where files are:**
   - `F3_probe_results.csv`: every input, model, split and threshold rule.
   - `F3_camera_ai_edited.csv`, `F3_verite_authenticity_scores.csv`: detector scores by group and per VERITE photo.
   - `F3_leakage_ALR.csv`, `F3_pair_verdicts.csv`, `F3_caption_metrics.csv`: VERITE leakage test.
   - `F3_summary.csv`: main numbers in one table.
   - `F3_FINAL_SUMMARY.csv`, `F3_corrected_pair_ALR_summary.csv`: the same detector and leakage numbers in the layout read by `src/eval/figures.py`.
   - `baselines_originals.csv`, `baselines/`: baseline scores and predictions.
   - Detector embeddings are on Drive: `MyDrive/TwinCheck/features/vision/genimage/clip_vitl14_debiased96/` (`embeddings_normalized.npy`, `embedding_manifest.csv`).

**Limitations:**
- One unseen generator only (SD v1.4), and the detector still misses most of its images.
- Content and style differences between real and AI images are not removed.
- 34 real photos smaller than 96 px are enlarged.
- VERITE test has 36 pairs, so one changed verdict moves ALR by 2.8 points.
