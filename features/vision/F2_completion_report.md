# F2 — Vision Features and Image Twins

## Status

**PARTIALLY COMPLETE.** The final JPEG-quality-85 image-twin pairs, their
SigLIP 2 and CLIP ViT-L/14 embeddings, and the requested PR metadata files
have been generated and validated. Remaining dataset-availability and
review limitations are documented below.

## 1. Image embeddings

| Dataset | SigLIP 2 image embeddings | SigLIP 2 text embeddings | CLIP ViT-L/14 image embeddings |
|---|---:|---:|---:|
| VERITE | 663 | 1,001 captions | 663 |
| GenImage | 18,000 | Not generated | 18,000 |
| M4FC | 603 | 770 Arabic claim embeddings | 603 |

All checked image embeddings passed the recorded embedding audit.

The GenImage float16 embedding copies are available:
- `features/vision/genimage/genimage_siglip2_image_projected_embeddings_float16.npy`
- `features/vision/genimage/genimage_clip_vit_l14_image_projected_embeddings_float16.npy`

Only the float16 copies are in git. The float32 originals (55 MB and 83 MB)
stay on Drive.

**Row manifests (which row is which image or claim).** Every `.npy` has a
`*_manifest.csv` next to it with the same row order. Paths are repo-relative.

| Embedding file | Manifest key | Notes |
|---|---|---|
| VERITE SigLIP 2 / CLIP images (663) | `image_id` (e.g. `true_0.jpg`) | Same order in both; the split matches `verite.jsonl` for all 663 |
| VERITE SigLIP 2 captions (1,001) | `caption_id` | Row *i* = record `verite_{i:06d}` in `verite.jsonl` (checked) |
| M4FC SigLIP 2 / CLIP images (603) | `id` (image id) | `split` here is M4FC's own split; our M4FC records are all `test` |
| M4FC SigLIP 2 Arabic claims (770) | `id`, `embedding_row` | One extra row, `m4fc_000027`, is no longer in `m4fc_ar_test.jsonl` (769). Join by `id`. |
| GenImage (18,000) | `image_id` | 5,000 / 1,000 per class (real, SD v1.4, BigGAN) in train / validation |

The final JPEG85 image embeddings cover 1,200 images (600 original/twin
pairs):
- SigLIP 2: shape `(1200, 1152)`.
- CLIP ViT-L/14: shape `(1200, 768)`.

Both embedding sets were checked for finite, normalized vectors.

**Important distinctions:**
- M4FC's 770 text embeddings represent Arabic claims, not image captions.
- No GenImage captions were found in the inspected metadata, so GenImage
  caption embeddings were not generated.
- GenImage Midjourney test images were not available in the inspected Drive
  dataset. Their embeddings are not included in the completed counts.

## 2. VAE image twins

The corrected VAE encode-decode pipeline generated 494 twins.

| Dataset | Generated | Accepted before final face screen | Final retained |
|---|---:|---:|---:|
| VERITE | 338 | 311 | 311 |
| M4FC | 156 | 142 | 141 |
| **Total** | **494** | **453** | **452** |

The final M4FC count excludes `m4fc_000478`, which was identified as
containing a large face during the additional automatic-screening and
manual-review process.

## 3. SDXL image twins

The SDXL img2img run used strength 0.15 and seed 42.

| Stage | Count |
|---|---:|
| Approved sources processed | 249 |
| Similarity accepted before manual face review | 157 |
| Similarity rejected | 92 |
| Manually excluded for faces | 9 |
| **Final retained** | **148** |

The approved source set contains 152 M4FC and 97 VERITE images:
201 test and 48 validation sources. No VERITE training sources were used.

Manual visual inspection covered 30 pairs; it was not an exhaustive review
of every generated pair.

## 4. JPEG quality 85 conversion and final face screening

The original and twin images were re-saved from decoded RGB pixels as JPEG
quality 85. The conversion process recorded the quality setting and
produced the final manifest.

| Dataset/method | Pairs retained |
|---|---:|
| VERITE VAE | 311 |
| M4FC VAE | 141 |
| SDXL | 148 |
| **Total** | **600** |

The final manifest contains 1,200 files: 600 originals and 600 twins.
One additional pair, `m4fc_000478`, was excluded after face screening.

An OpenCV Haar frontal-face cascade screened all 601 twins before that
exclusion. Four detections exceeded the 10% image-area threshold. Manual
inspection identified one genuine large face and three false positives.
The Haar cascade is not a comprehensive detector for all face sizes,
orientations, or poses.

The final JPEG85 embeddings were extracted after conversion and screening.

## 5. PR metadata deliverables

The twin files are in the shared record format (CLAUDE.md §4). Each line is
the source record from `data/processed/vision/verite.jsonl` or
`data/processed/ar/m4fc_ar_test.jsonl` (same claim, label, evidence, language
and split) with `id = <source id>_V` and `image_path` = the twin image.
`method`, `similarity` and `orig_image_path` are extra fields. A twin image
gets one record per claim that uses the original photo. For example, a
VERITE true photo has a true caption and a miscaptioned one.

| File | Twin images | Records |
|---|---:|---:|
| `data/copies/vision/vae.jsonl` | 452 | 903 (VERITE 622, M4FC 281) |
| `data/copies/vision/sdxl.jsonl` | 148 | 295 (VERITE 106, M4FC 189) |
| `data/copies/vision/sources.csv` | 249 approved SDXL sources (148 with a kept twin) | — |

All three pass `python -m src.eval.loaders --check`. Built by
`python -m src.vision.make_twin_records` from the F2 notebook manifests in
`data/copies/vision/manifests/`. `data/processed/vision/verite.jsonl` is built by
`python -m src.vision.load_verite` (ids `verite_000000`…, labels true →
supported, miscaptioned / out-of-context → contradicted, F1 split by image:
701 / 149 / 151 records). Paths are repo-relative. The image files themselves
are not in git (the repo is public and these are news photos).

IDs can occur in both JSONL files when the same source has both a VAE twin
and an SDXL twin; uniqueness is checked within each file.

## 6. Remaining limitations

1. No GenImage captions were found in the inspected metadata, so GenImage
   caption embeddings were not generated.
2. GenImage Midjourney test images were unavailable in the inspected Drive
   dataset; embeddings for those images are not claimed as completed.
3. M4FC Arabic claim embeddings are claim embeddings, not image-caption
   embeddings.
4. SDXL manual visual inspection covered 30 pairs, not all generated pairs.
5. Haar frontal-face detection is limited and does not establish exhaustive
   face removal.

## 7. Final assessment

The final JPEG85 twin set contains 600 retained pairs across the two
generation methods. The corresponding 1,200 images have SigLIP 2 and
CLIP ViT-L/14 embeddings. The JSONL and CSV PR metadata files have been
created and validated.

F2 remains **partially complete** because the GenImage caption and
Midjourney test-image requirements could not be completed with the
available dataset materials, and face screening has documented limitations.
