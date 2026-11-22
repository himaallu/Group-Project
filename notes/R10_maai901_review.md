# R10 · Technical review of the MAAI901 report (Rathi → Feba)

Reviewed: `reports/maai901_report/main.tex` at commit 0c261bc (20 Nov, still the 12-section
skeleton with TODOs), checked against `results/F1`–`F5`, `notes/F1`–`F5.md` and
`references.bib`. A second pass follows once the full draft (F7, Fri 21:00) is pushed.

**Every file named in the TODOs exists** (`F3_FINAL_SUMMARY.csv`, `F5/cnn_vit.csv`,
`F3_corrected_*`, `F3_VERITE_authenticity_probe_original_twin_scores.csv`,
`features/vision/F2_completion_report.md`, the F3 notebook).

## Must fix (numbers or claims that would be wrong)

1. **Seeds (§Experimental Setup).** The TODO says seeds 42/43/44, but the detectors (F5) and
   the CLIP probe (F3) ran once with seed 42. Only the F3/F4 text baselines have three seeds.
   Write "one run, seed 42" for the vision models and list it as a limitation.
2. **Mixed levels in `F3_FINAL_SUMMARY.csv`.** In the "VERITE Corrected Model" rows,
   `Accuracy` is caption-level (e.g. decoupled test 0.529), but `ALR_percent` is
   pair-level (7.84). The pair accuracy is 0.608 (`F3_corrected_pair_ALR_summary.csv`). Take
   the leakage table from the pair file and the caption numbers from
   `F3_corrected_caption_metrics.csv`. Say which level each column is.
3. **What the decoupled ALR measures.** The decoupled model never sees authenticity, yet its
   ALR is 7.8% (test) and 12.2% (validation). Those changes come from the SigLIP embedding
   moving after the VAE round trip, not from authenticity. Name this (e.g. "embedding drift")
   so a non-zero ALR does not look like it breaks the decoupling rule.
4. **Fused = decoupled.** Both models give the same pair verdicts and ALR on both splits. The
   probe barely reacts to the twins: mean score 0.025 → 0.031, and 0/323 binary flips. So this
   setup cannot show the harm the introduction motivates. State it plainly: "on VERITE with
   VAE twins our probe does not react to the twins, so there is no leakage to remove". Do not
   claim the decoupled design reduced leakage. (Results are frozen, so this is framing only.)
5. **Face screening.** F3 uses 323 VERITE twins, including 12 that F2's face screen later
   removed (8 train / 2 val / 2 test; `results/F3/README.md`). Say so in Ethics and
   Limitations, and say the numbers include them.

## Should fix (clarity and fairness)

6. **RQ1 framing.** All three detectors rank the unseen Stable Diffusion images well:
   AUROC CLIP-LR 0.860, CLIP-SVM 0.903, ViT-B/16 0.885, ResNet-50 0.713. But at the default
   threshold almost none are flagged (recall 0.014–0.080), so accuracy is about 0.5. Make
   AUROC the headline metric and the threshold failure the finding. Do not pick a threshold
   on the test split.
7. **CNN vs ViT vs CLIP table.** Compare like with like. The probe is linear on frozen CLIP
   features with no augmentation; the CNN and ViT fine-tune their last 4 blocks with
   augmentation. Put the F3 probe rows next to the F5 "test, clean" rows and note the
   difference in a footnote. ViT-B/16 vs ResNet-50 on the unseen test: AUROC 0.885 vs 0.713.
8. **VERITE counts.** §Data gives 1,001 records (701/149/151), while the results use 323
   photos (223/49/51) and 51 test pairs. Explain the steps in between: true + miscaptioned
   pair of the same photo, out-of-context excluded, accepted twin with SigLIP ≥ 0.92.
   Otherwise "151 test" and "51 test pairs" look inconsistent.
9. **Robustness wording.** JPEG 50/70 and 20% crop change test accuracy by less than 0.01, but
   crop lowers ResNet-50 test AUROC from 0.713 to 0.662 (`cnn_vit.csv`). Mention both.
10. **ONNX.** CPU latency (15.5 ms ResNet-50, 85.7 ms ViT-B/16) is on a Colab x86_64 CPU; GPU
    latency is PyTorch on an L4, not ONNX. Label the columns that way. ViT's ONNX file is
    343.6 MB: worth a sentence on deployment size.
11. **W&B.** The F3 notebook and F5 did not log to W&B (`--no-wandb`), and the F4 baseline
    runs are still offline on your laptop. Run `wandb sync`, or say in the setup that every
    metric comes from the CSV/JSON files in `results/`.

## Before submission

12. **References.** CLIP, SigLIP 2, UnivFD, GenImage, VERITE, Grad-CAM, ResNet, ViT and Stable
    Diffusion/SDXL are not in `references.bib` yet (it has 6 entries). Add them with checked
    details. GenImage is CC BY-NC-SA: cite the licence in Ethics.
13. **Image twins.** F2 used SDXL img2img at strength 0.15 (the plan said 0.15–0.25). Report
    what was actually run, with counts from `results/F2/README.md`: 600 originals + 600
    twins in the final JPEG 85 set.
14. **Code availability.** The repository is tagged `v1.0` (R11). Cite the tag and
    `docs/implementation.md` for settings.
15. **Turnitin.** Write the data and twin sections fresh. The MAAI902 report describes the
    same twins, so do not reuse its wording (CLAUDE.md §10).

Good: the F3 and F5 READMEs already state the small-sample and single-generator limits
honestly, and the Grad-CAM finding (all 3 Stable Diffusion images missed by both models)
is a strong error-analysis case.

## Update, Sun 22 Nov (after F1–F6 were re-run with the 96 px preprocessing)

The full draft is still not in `reports/maai901_report/main.tex` (34 TODOs), so this pass
checks the new results only:

- **Items 3–4 no longer apply as written.** On the new VERITE split (36 test pairs), the fused
  model's ALR is double the decoupled model's: 22.2% vs 11.1%, and 25.0% with the image
  embedding too (`tab_leakage.csv`). This is now evidence *for* the decoupling rule. Say it with
  care: 4 vs 8 changed verdicts out of 36, and on validation it is 9.4% vs 15.6%.
- **Item 5 is fixed:** F3 now uses the 218 face-screened F2 twins.
- **Item 6 changes:** with the format shortcut removed, AUROC falls to 0.61–0.75 (ViT-B/16 best,
  0.754), so the detectors no longer "rank well". The format-only result (AUROC 1.000) is a
  strong finding for the Data section: the original GenImage files give the answer away.
- **New, INT8 ONNX** (`tab_onnx_int8.csv`): ResNet-50 INT8 has a max output difference of 10.7
  vs FP32, yet its AUROC rises from 0.626 to 0.826. That suggests the quantised model is
  broken rather than better. Check it before reporting, or report only ViT-B/16 INT8 (0.770 vs 0.771).
- The README vision numbers and every table in `reports/tables/tab_{detectors,robustness,leakage,onnx,onnx_int8}` are updated.
