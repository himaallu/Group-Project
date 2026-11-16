# Y1 · Arabic data (X-FACT Arabic + M4FC Arabic)
1. What ran: `src/data/load_xfact.py`, `src/data/load_m4fc.py`, `src/data/download_m4fc_images.py` (notebook `Y1_Arabic_Data_Colab.ipynb`, Colab Account B, CPU).
2. Settings: X-FACT own splits (dev -> val); M4FC all 614 Arabic rows as test only, plus M4FC true captions as supported; label map in `data/label_map.csv`; no training, so no W&B run.
3. W&B: n/a (data preparation only).
4. Key numbers (before clean.py): X-FACT AR train 687 / val 96 / test 136 (contradicted 838, supported 55, unverified 26; 1,171 dropped); M4FC AR test 770 (613 contradicted, 157 supported); images: see `m4fc_images/manifest.csv`.
5. Big files: Drive `TwinCheck/data/processed/ar/` and `TwinCheck/data/processed/vision/m4fc_images/`.
