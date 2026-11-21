# TwinCheck — project summary (status Sat 21 Nov 2026)

One page per question: what we built, which models and why, what came out and why, and what it means. Every number has its source file. For report wording use your own words (Turnitin).

## 1. The problem and our idea
- Fact-checkers score well but often use shortcuts (claim–evidence word overlap, how plausible the claim sounds) instead of reading the evidence (Papadopoulos et al., WACV 2025). They are rarely tested on Arabic or mixed Arabic–English, and on images an AI-looking photo can turn a true post "fake" (FALSDO, IJCAI 2026).
- **TwinCheck**: a bilingual fact-checker that gives Supported / Contradicted / Unverified, a calibrated confidence and a cited note, plus **twins** — copies of a real claim that change exactly one thing (`_P` paraphrase, `_A` Arabic, `_M` Arabic claim + English evidence, `_B` swapped entity in the evidence → contradicted, `_C` unrelated evidence → unverified, `_V` AI-fingerprinted image → same verdict). Twins test whether the verdict changes for the right reason.

## 2. What each person built
| Person | Built (tasks) |
|---|---|
| **Rathi** (code) | English data from MultiFC + AVeriTeC with cleaning and an own split (R1); fine-tuned reranker + spaCy entities (R2); English twins `_P _B _C` (R3); NLI fine-tuning v1/v2 + root-cause analysis (R4); cited explanations + faithfulness + offline BM25/BGE-M3 index (R5); Gradio demo (R6); `run_all.sh`, README (R7); figures, tables, T1 sheets, fresh-clone test (R8); technical appendix (R9); demo polish, slides, video script, MAAI901 review (R10); repo v1.0 (R11) |
| **Youssef** (Arabic + claims) | X-FACT Arabic + M4FC data (Y1); 2,658 Arabic NLI pairs + 150 Arabic gold pairs (Y2); 300 gold messages + Qwen claim extractor (Y3); 642 Arabic / mixed twins (Y4); Arabic experiments E7 + bias diagnostics (Y5); Arabic error analysis (Y6); 20 Arabic T1 ratings; translation hand check |
| **Himasri** (integration, evaluation) | repo, metrics, report tables, tests (H1); 150 English gold NLI pairs (H2); verdict head + twin loss (H3); λ grid, calibration, pipeline (H4); ablations, AVeriTeC, Arabic calibration, bootstrap (H5); error cases, figures, fact sheet, kappas (H6); model selection incl. Qwen3-4B (H7); T1 ratings + twin hand-check |
| **Feba** (vision + baselines) | VERITE + GenImage data (F1); SigLIP 2 / CLIP embeddings + 600 image twins (F2); authenticity probe + VERITE leakage test (F3); four baselines incl. LLM judge (F4); ResNet-50 vs ViT-B/16 detectors, robustness, Grad-CAM, ONNX (F5); vision error analysis (F6); T1 ratings + twin hand-check |

## 3. The steps (pipeline) and why each model
| Step | Model | Why this one (evidence) |
|---|---|---|
| Claim extraction | Qwen3-8B few-shot (4-bit) | F1 0.984 vs 0.786 rule-based (`results/Y3`); Qwen3-4B does the same at half the size (0.977, `results/H7/qwen4b`) |
| Evidence retrieval | BM25 + BGE-M3 hybrid, offline | recall@5 EN 0.945 vs BM25 0.91 / dense 0.87; AR 0.92 (`results/R5`) |
| Relevance | `bge-reranker-v2-m3`, fine-tuned 1 epoch | multilingual; AUROC 0.980 → 0.998 (`results/R2`); removing it costs −5 F1, −21 ESR (`results/H5`) |
| Entities | spaCy `en_core_web_trf` | who / where / when for `_B` swaps and entity-match features |
| NLI | mDeBERTa-v3-base XNLI, zero-shot | one model for EN + AR, smallest and fastest; larger XLM-R (+1.5 EN, +6 AR) and DeBERTa-large (EN only, +6) within overlapping CIs; fine-tuned v1/v2 were worse or tied (`results/H7/nli_selection.csv`) |
| Verdict head | MLP 10 → 64 → 3 + twin loss | the KL "keep the answer" term needs a differentiable model; beats LR / LightGBM with twins-as-data by +9 ESR (p < 0.001); LightGBM is more accurate but flips on 1/3 of paraphrases (`results/H7/verdict_head_selection.csv`) |
| Calibration | temperature + Unverified threshold | standard, fitted on validation only |
| Explanation | Qwen3-8B (4-bit), cites [1]–[3] | Arabic support; Qwen3-4B is slightly more faithful (0.352 vs 0.320) |
| Image (MAAI901) | SigLIP 2, CLIP ViT-L/14 probe; ResNet-50 vs ViT-B/16 | frozen features generalise; CNN vs ViT is the course comparison |

## 4. Data
MultiFC 16,847 / 2,104 / 2,106 claims (own split, fact-check snippets removed); AVeriTeC 2,309 / 252 / 429; X-FACT Arabic (135 test); M4FC Arabic; VERITE 1,001 pairs; GenImage real + BigGAN (train/val) and Stable Diffusion v1.4 as the unseen test generator (`results/R1`, `results/Y1`, `results/F1`). Twins: 955 English test + 5,705 train; 642 Arabic/mixed; 600 image twins.

## 5. Results (main numbers)
- **Twin loss works:** ESR 52 → 90%, unrelated 42 → 100%, paraphrase 91% kept, −1 macro-F1 (p < 0.001, 3 seeds) (`results/H4`).
- **Accuracy vs evidence-reading:** claim-only (no evidence) 0.481 macro-F1 but unrelated 0.01; fine-tuned Qwen3-4B 0.500 but ESR 0.31 / unrelated 0.15; with twins it reaches ESR 0.67 / unrelated 0.99 at 0.442; our MLP + twin loss 0.352 with ESR 0.89 / unrelated 1.00 / paraphrase 0.91 (`results/F4`, `results/H7/qwen4b`).
- **Smaller LLM is better:** Qwen3-4B vs 8B — judge 0.391 vs 0.233 on MultiFC, 0.613 vs 0.494 on AVeriTeC (best AVeriTeC score of any model).
- **Arabic:** fair system (no English entity features) Arabic F1 0.597 vs English 0.605, cross-lingual consistency 74% (`_A`) / 80% (`_M`); X-FACT Arabic ≈ chance for all systems (`results/Y5`).
- **Ablations:** reranker matters most (−5 F1, −21 ESR); entity match small (`results/H5`).
- **Vision:** detectors near-perfect on BigGAN but miss most unseen Stable Diffusion images (recall 0.01 / 0.08); ViT ranks better (AUROC 0.885 vs 0.713); image twins change ~8–9% of verdicts and fusing the authenticity score changes nothing (`results/F3`, `results/F5`).
- **Human evaluation:** twin hand-check kappa 0.57 (`_B` only 22.5% valid); T1 kappa 0.35–0.49; gold verdict supported by the snippets in 38–48% of EN notes, 10% AR.

## 6. Why the results came out this way
- **MultiFC rewards guessing from the claim:** labels follow the claim's style/source, so claim-only and a fine-tuned LLM score highest without reading evidence; T1 shows the snippets often do not settle the claim.
- **Snippets repeat false claims:** zero-shot NLI "entails" 34% of false claims, so "supported" is the hardest verdict for every evidence model (`results/H5/zeroshot_signal_by_label.json`).
- **Weak labels teach shortcuts:** NLI fine-tuned on verdict-derived labels learned "same topic → copy the verdict" (gold 0.37 → 0.12).
- **Twins add the missing signal:** `_B` / `_C` examples tell a model that evidence must matter, which is why twin training raises ESR and the unrelated rate for both the MLP and Qwen3-4B; part of the ESR gain is a bias towards "contradicted" (paired ESR 76% vs 33%).
- **`_B` twins are noisy:** automatic swaps often produce nonsense or do not touch the claim (only 22.5% valid), so ESR measures sensitivity to corrupted evidence more than to true contradictions.
- **Arabic gap:** mostly the same head errors as English (85%); the rest is translation (only 54% of checked translations fully correct) and English-only entity features.
- **Vision shortcut:** AI images are fixed-size PNGs and real ones varied JPEGs, so detectors may learn format cues that do not transfer to a new generator (`results/F6`).

## 7. Conclusion
Accuracy alone would have picked the model that reads the evidence least. Twins reveal this and, used in training, make both a small head and a 4B LLM follow the evidence for a small accuracy cost; the small head with the twin loss does it best and is interpretable. The quality of the evidence (search snippets), the `_B` twin generator and Arabic translation are the main limits; future work: validated claim-entity swaps, human-labelled on-topic NLI data, multilingual entity matching, real forwarded messages and dialects.

## 8. Left before / during writing (Sat 21 Nov)
- Himasri: 10 error-analysis causes (`results/H6/error_cases.csv`); MAAI902 report + paper; slide deck rebuild.
- Feba: MAAI901 report + paper (F7, F9).
- Youssef: Arabic sections (prose) for both reports; numbers check of drafts (Y8).
- Rathi: rehearsal; review drafts.
- Everyone: contribution statement from `notes/contributions.md`; Turnitin < 10% each; final upload to `Final/`.
