# F3: Detector + leakage + baselines on English originals (Feba)

1. **What ran:**
   - `src/vision/probe.py`: a CLIP ViT-L/14 logistic probe for camera vs AI on GenImage, in three settings (seen, unseen BigGAN, unseen SD v1.4). The same probe scored every VERITE, M4FC and JPEG-85 twin image.
   - `src/vision/jpeg_control.py`: 300 validation images per generator re-saved as JPEG 85.
   - `src/vision/leakage.py`: decoupled, fused and fused-without-authenticity verdict models on VERITE → ALR on the `_V` twins.
   - `src/eval/baselines/{claim_only,similarity}.py` on the MultiFC and AVeriTeC test originals.
2. **Settings:**
   - Probe: C from {0.1, 1, 10} by 5-fold CV on train.
   - Leakage: SigLIP 2 features, C from {0.01, 0.1, 1} on VERITE val. ALR is on val + test twins, with a bootstrap 95% CI (1,000 resamples).
   - Claim-only: TF-IDF + logistic regression. Similarity: 12 features + LightGBM, seeds 42/43/44.
   - Logistic models are deterministic, so they were run with seed 42 only.
3. **W&B:** project `twincheck`, runs `F3-probe-*`, `F3-leakage-*`, `F4-claim_only-s42`, `F4-similarity-s4*`. They were logged **offline** on Feba's laptop and still need `wandb sync wandb/offline-run-*` (link to be added after sync).
4. **Key numbers** (copied from the CSVs):
   - Probe AUROC: seen 0.999, unseen BigGAN 0.981, unseen SD 0.870; seen probe on the JPEG-85 re-saves 0.999 (`probe_metrics.csv`).
   - Mean `p_ai`, original → twin: VERITE VAE 0.486 → 0.554, M4FC VAE 0.592 → 0.652, SDXL 0.598 → 0.667 (`twin_authenticity.csv`).
   - ALR on VAE twins: decoupled 0.109, fused 0.062, fused without authenticity 0.052 (`alr.csv`).
   - VERITE test macro-F1: 0.505 / 0.531 / 0.536 (same order).
   - Macro-F1 on MultiFC test: claim-only 0.481, similarity 0.385 ± 0.005 (`baselines_originals.csv`).
5. **Where files are:**
   - Tables are in `results/F3/*.csv`. Predictions are in `results/F3/leakage/` and `results/F3/baselines/`, in subfolders so `src/eval/report.py` does not mix them into the team tables.
   - `p_ai` per image is in `features/vision/authenticity/clip_probe_p_ai.csv`. Probe weights are in `models/F3/clip_probe_seen.npz`. The JPEG control embeddings are in `features/vision/genimage/jpeg85_control_*`.
   - The MAAI901 skeleton is in `reports/maai901_report/main.tex`.
