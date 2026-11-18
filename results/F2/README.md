# F2: Vision features and image twins (Feba)

1. **What ran:** I cached SigLIP 2 (so400m) and CLIP ViT-L/14 embeddings for VERITE, M4FC and GenImage, then made `_V` image twins two ways: an SD VAE round-trip for all true images and SDXL img2img for the approved test/val sources. Twins were kept if SigLIP cosine was at least 0.92, faces were screened, and everything was re-saved as JPEG 85.
2. **Settings:** SDXL strength 0.15, seed 42. VERITE split by image (70/15/15, seed 42). Haar face screen with a 10% area threshold. I checked 30 SDXL pairs by eye.
3. **W&B:** no training in F2 (embedding extraction only).
4. **Key numbers:** VAE twins 452 images → 903 records (VERITE 311 → 622, M4FC 141 → 281). SDXL twins 148 images → 295 records. JPEG85 embeddings: 1,200 images, SigLIP (1200, 1152) and CLIP (1200, 768). See `features/vision/F2_completion_report.md`.
5. **Where files are:** twin records are in `data/copies/vision/{vae,sdxl}.jsonl`, `sources.csv` and `manifests/`. VERITE records are in `data/processed/vision/verite.jsonl`. Embeddings are in `features/vision/`. Twin images are on Feba's Drive only (`data/copies/vision/jpeg85_final/`), not in git.
