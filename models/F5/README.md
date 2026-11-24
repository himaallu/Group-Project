# F5 detector checkpoint (Feba)

- `vit_b16_s42.pt`: ViT-B/16 AI-image detector from F5, seed 42, after epoch 3 (stored as `epoch` 2, counted from 0). Validation AUROC 1.000, unseen Stable Diffusion v1.4 test AUROC 0.754 (`results/F5/README.md`).
- Keys: `model` (state dict for `src.vision.detectors.build_model("vit_b16")`), `epoch`, `history`. The optimizer and scheduler state were removed to save space; the model weights are unchanged.
- Stored with Git LFS (343 MB). Run `git lfs pull` after cloning.
- Load: `model.load_state_dict(torch.load(path, map_location="cpu")["model"])`, then feed images prepared like `data/processed/vision/genimage_debiased96_jpeg.pkl` and `eval_transform()`.
