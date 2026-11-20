# F3: Authenticity probe, image leakage test and baselines on English originals (Feba)

1. **What ran:**
   - `notebooks/Feba_F3_authenticity_leakage.ipynb`: a camera vs AI probe on cached CLIP ViT-L/14 GenImage embeddings (F2), then a VERITE image–caption test on original photos and their VAE twins, comparing a decoupled model with a fused model that also sees the authenticity score.
   - `src/eval/baselines/{claim_only,similarity}.py` on the MultiFC and AVeriTeC test originals.
2. **Settings:**
   - GenImage splits from F1: train 4,000 real + 5,000 BigGAN, validation 1,000 real + 1,000 BigGAN, test 1,000 real + 1,000 Stable Diffusion v1.4. Stable Diffusion is never seen in training or validation, so the test measures transfer to an unseen generator.
   - Probe: logistic regression and linear SVM (C = 1) on the 768-d CLIP embeddings, seed 42. The decision threshold is chosen on validation (best F1).
   - VERITE: true and miscaptioned captions of the same photo (out-of-context excluded). 323 photos with an accepted VAE twin (SigLIP cosine ≥ 0.92): train 223, validation 49, test 51.
   - Decoupled model: SigLIP 2 image–caption similarity only. Fused model: similarity, authenticity score and their product. Both are logistic regression with standard scaling, trained on original training photos only; twins are only evaluated.
   - A pair is correct when the model gives the true caption a higher probability than the miscaptioned one. ALR is the share of pairs whose verdict changes between the original and the twin.
   - Claim-only: TF-IDF + logistic regression. Similarity: 12 features + LightGBM, seeds 42/43/44.
3. **W&B:** the notebook does not log to W&B. The baseline runs `F4-claim_only-s42` and `F4-similarity-s4*` were logged offline on Feba's laptop and still need `wandb sync wandb/offline-run-*`.
4. **Key numbers** (from the CSVs):

   Authenticity probe, unseen-generator test (`F3_FINAL_SUMMARY.csv`):

   | Model | AUROC | Accuracy | Precision | Recall | F1 |
   |---|---:|---:|---:|---:|---:|
   | Logistic regression | 0.860 | 0.510 | 0.857 | 0.024 | 0.047 |
   | Linear SVM | 0.903 | 0.507 | 1.000 | 0.014 | 0.028 |

   The probe ranks Stable Diffusion images above real ones, but at the validation threshold it flags almost none of them, so it does not transfer well to the unseen generator. On VERITE, mean authenticity score goes from 0.025 (original) to 0.031 (twin), and 0/323 binary predictions change (`F3_VERITE_authenticity_probe_original_twin_scores.csv`).

   VERITE pair verdicts (`F3_corrected_pair_ALR_summary.csv`; all-pairs row from `F3_decoupled_similarity_baseline_summary.json`):

   | Split | Model | Original acc. | Twin acc. | Changed | ALR |
   |---|---|---:|---:|---:|---:|
   | All 323 pairs | Decoupled | 56.7% | 54.2% | 30 | 9.3% |
   | Validation (49) | Decoupled | 53.1% | 49.0% | 6 | 12.2% |
   | Validation (49) | Fused | 53.1% | 49.0% | 6 | 12.2% |
   | Test (51) | Decoupled | 60.8% | 60.8% | 4 | 7.8% |
   | Test (51) | Fused | 60.8% | 60.8% | 4 | 7.8% |

   Caption-level test accuracy, original → twin: decoupled 52.9% → 51.0%, fused 52.0% → 52.0% (`F3_corrected_caption_metrics.csv`). Adding the authenticity score does not change pair verdicts or ALR, so there is no evidence here that fusion helps or hurts robustness to VAE twins.

   Baselines on English originals: macro-F1 on MultiFC test, claim-only 0.481, similarity 0.385 ± 0.005 (`baselines_originals.csv`).
5. **Where files are:**
   - `results/F3/`: `F3_FINAL_SUMMARY.csv` (all main numbers), `F3_VERITE_authenticity_probe_original_twin_scores.csv`, `F3_decoupled_similarity_baseline_{results.csv,summary.json}`, `F3_corrected_{caption_metrics,pair_ALR_summary,pair_verdicts}.csv` (decoupled vs fused).
   - Baseline predictions are in `results/F3/baselines/`.
   - The GenImage manifest the notebook builds (`data/processed/vision/f3_authenticity_manifest.csv`) and the embeddings it reads stay on Drive.

**Limitations:**
- VERITE is small (51 test pairs), so a single changed verdict moves ALR by about 2 points.
- The VERITE set includes 12 twins that F2 later removed in its face screen (8 train, 2 validation, 2 test). F2 keeps 311 VERITE twins; these results use 323.
- Probe recall on the unseen generator is very low at the chosen threshold, and only one unseen generator (Stable Diffusion v1.4) is tested.
- The fusion features are simple (three numbers per caption).
- ALR measures verdict stability, not correctness.
