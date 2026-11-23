# TwinCheck (MAAI902): slide content pack

Slide-by-slide text to build the deck yourself in PowerPoint or any slide tool.
For each slide: **Title** (plain English), **On the slide** (what goes on it), **Visual**, **Say** (one or two spoken lines).
Every project number comes from `results/`. Every outside fact has its source. Keep the source line on the slide in small text.

**Rules that keep it from looking generated**
- Use real material: screenshots of real rumours, a real Google result, real twins from our data (given below), our own figures in `reports/figures/`.
- Titles are plain statements of the point, e.g. "Removing the reranker hurts most", not clever phrases.
- One message per slide. A table beats bullets when you compare numbers.
- Say "to our knowledge" whenever you claim something is new.

---

## 1. Rationale, problem statement and objectives

### Slide 1: Title
- **On the slide:** TwinCheck: Do bilingual fact-checking models follow the evidence? · Himasri, Rathi, Youssef, Feba · University of Wollongong in Dubai · MAAI902 · November 2026

### Slide 2: Hook
- **Title:** False news spreads faster than the truth
- **On the slide:**
  - On Twitter, false stories reached 1,500 people about **six times faster** than true ones, and were **70% more likely** to be retweeted. *(Vosoughi, Roy & Aral, Science, 2018)*
  - During Dubai's record floods in April 2024, misleading claims that cloud seeding caused the flooding spread widely online. *(BBC News, 17 Apr 2024)*
- **Visual:** a screenshot of a real rumour post or headline from the April 2024 floods (blur personal details).
- **Say:** "A false message can reach thousands before anyone checks it. We want to know whether automated fact-checkers can help, and whether their verdicts can be trusted."

### Slide 3: Why this matters (rationale)
- **Title:** Misinformation is a top global risk, and checking cannot keep up
- **On the slide:**
  - Misinformation and disinformation were ranked the **#1 short-term global risk** in the World Economic Forum's Global Risks Reports 2024 and 2025, and **#2** in 2026.
  - **58%** of people worldwide worry about telling real from fake news online. *(Reuters Institute Digital News Report 2025)*
  - Only about **440 fact-checking projects** are active worldwide. More than 30 stopped in 2025 and only 10 started. *(Duke Reporters' Lab census, 2025 and 2026)*
  - The UAE is **99% online**. Spreading false news is a crime under Federal Decree-Law No. 34 of 2021 (Art. 52): at least 1 year in prison and AED 100,000, doubled during crises. *(DataReportal 2025; UAE Legislation portal)*
- **Say:** "People worry, fact-checkers are few, and in the UAE forwarding a rumour has legal consequences. Automated tools can help, but only if their verdicts rest on evidence people can check."

### Slide 4: Problem statement
- **Title:** Fact-checking models can get the right answer without reading the evidence
- **On the slide:**
  - **The task:** given a claim and evidence snippets, decide whether the claim is supported, contradicted or unverified.
  - **The problem:** high scores do not prove the model used the evidence. It may guess from the wording of the claim.
  - **Evidence for this:**
    - On MultiFC, models scored as well from the evidence alone as from claim plus evidence. *(Hansen et al., ACL 2021)*
    - 70% of MultiFC's false claims had a snippet from a fact-checking article that already stated the verdict. Models scored 59.7 macro-F1 with those snippets and 25.6 without. *(Glockner et al., EMNLP 2022)*
    - On AVeriTeC, GPT-4o scored 71.0% with **no evidence** and 72.4% with its full evidence pipeline. *(Rothermel et al., FEVER 2024)*
  - Arabic and mixed Arabic–English claims, common in the UAE, are rarely tested.
- **Say:** "So the question is not only whether the verdict is right, but whether it is right for the right reason."

### Slide 5: Literature review (1 of 2): benchmarks and shortcut studies

| Work | Data | Method | Key finding | What it leaves open |
|---|---|---|---|---|
| Augenstein et al., EMNLP 2019 (MultiFC) | 34,918 real claims, 26 fact-check sites, top-10 Google snippets | BiLSTM, joint evidence ranking + verdict | Macro-F1 0.492 vs 0.253 claim-only | Leaked fact-check snippets; test labels never released |
| Hansen et al., ACL 2021 | MultiFC (PolitiFact, Snopes) | BERT: claim only, evidence only, both | Evidence-only 0.351 beat claim + evidence 0.285 on PolitiFact | Shows the problem on average, not per claim |
| Glockner et al., EMNLP 2022 | MultiFC (PolitiFact, Snopes) | URL and phrase patterns find leaked snippets; BERT on leaked vs clean | 69.7% of false claims leaked; F1 59.7 → 25.6 without leaks | Diagnosis only; no training fix |
| Schlichtkrull et al., NeurIPS 2023 (AVeriTeC) + FEVER shared tasks 2024–25 | 4,568 claims, human Q&A evidence dated before each claim | Retrieve → question generation → LLM verdict | 2024 winner 0.63 (GPT-4o); 2025 winner Qwen3-14B on one GPU; no-evidence GPT-4o nearly ties | Clean evidence, but still no test of whether the verdict uses it |
| Gupta & Srikumar, ACL 2021 (X-FACT) | 31,189 claims, 25 languages (Arabic from misbar.com) | mBERT with top-5 snippets | 41.9 macro-F1 vs 38.2 claim-only, 37.5 with random snippets | No cross-language consistency test; Arabic labels skewed |

- **Say:** "Each benchmark moved the field forward, and each paper also found a shortcut. We build on their datasets and their diagnoses."

### Slide 6: Literature review (2 of 2): testing evidence use, and our building blocks

| Work | Method | Key finding | How we build on it |
|---|---|---|---|
| Schuster et al., NAACL 2021 (VitaminC) | Contrastive pairs from real Wikipedia edits; a small evidence change flips the label | Training on such pairs makes models more sensitive to evidence | Same idea on messy web evidence, plus unrelated, paraphrase and language twins |
| Abdelnabi et al., CVPR 2022 (CCN) | Image–caption checking with real web evidence | Compared with vs without evidence | One controlled change at a time instead |
| Papadopoulos et al., WACV 2025 | Audit of image–caption fact-checkers on VERITE | Similarity alone scores 70.9%; best model caught 13% of miscaptions | `_B` twins keep similarity but change one fact (we extend the idea to text) |
| Chrysidis et al., 2024 (CREDULE) | Classifier for leaked or unreliable evidence | 35% of VERITE's evidence is from fact-check sites | We filter fact-check domains before training |
| Building blocks | NLI (mDeBERTa-v3 XNLI), hybrid retrieval (BM25 + BGE-M3), RAG, QLoRA, temperature scaling (Guo et al., 2017) | Standard, tested components | We combine them and add the twin loss |

### Slide 7: Gaps and research questions
- **Title:** Three gaps lead to three research questions
- **On the slide:**

| Gap in prior work | Our research question |
|---|---|
| Shortcut use was shown on averages, not tested for each claim | **RQ1:** Do current models change their verdict when the evidence changes, and keep it when only the wording changes? |
| Evidence-change tests used clean Wikipedia; nobody trained models with them on real web claims | **RQ2:** Can training with controlled examples make a model follow the evidence, at a cost of at most 1 macro-F1 point? |
| No test of whether a claim keeps its verdict across English, Arabic and mixed text | **RQ3:** Does this hold for Arabic and mixed Arabic–English claims? |

### Slide 8: Our approach
- **Title:** Our approach: audit, train, test in three languages
- **On the slide (four steps):**
  1. **Build twins:** copies of real claims that change exactly one thing, so we know what the right verdict should be.
  2. **Audit:** run baselines and LLMs on the twins to see which ones follow the evidence.
  3. **Train:** add twin terms to the training loss of our verdict model.
  4. **Test:** English, Arabic and mixed; 3 seeds; bootstrap significance tests; ablations; human evaluation.
- Plus a full working pipeline: message → claims → evidence → verdict → cited explanation (demo).

### Slide 8a: Research methodology: design
- **Title:** Research design: a controlled experiment on real claims
- **On the slide:**
  - **Type of study:** quantitative and experimental. Like a lab test, we change one variable at a time and measure the effect.
  - **Independent variables (what we change):**
    - the evidence: original vs swapped (`_B`) vs unrelated (`_C`)
    - the wording and language: original vs paraphrase (`_P`), Arabic (`_A`), mixed (`_M`)
    - the training: with vs without twin loss; the model type (baselines, MLP, LLM)
  - **Dependent variables (what we measure):** macro-F1, AUROC and ECE on originals; ESR, unrelated rate, paraphrase and cross-lingual consistency on twins.
  - **Controls:**
    - same data splits for every model, made once with seed 42
    - twins keep the split of their source claim, and train data never touches test data
    - all settings chosen on validation
    - the same metric code for everyone (`src/eval/metrics.py`)
  - **Hypotheses:**
    - **H1:** models that score well on accuracy can still ignore the evidence (low unrelated rate)
    - **H2:** the twin loss raises ESR and the unrelated rate with ≤ 1 point macro-F1 loss
    - **H3:** a system without English-only features performs similarly in Arabic and English
- **Visual:** a flow of the research process: Data → Clean → Twins → Hand-check → Audit → Train → Compare → Arabic → Human evaluation → Error analysis.

### Slide 8b: Research methodology: how we evaluate
- **Title:** How every result is measured and tested

| Metric | Definition | Target |
|---|---|---|
| Macro-F1 | average of the F1 scores of the 3 classes, each counted equally | higher |
| AUROC | how well the probabilities rank the right class above the others | higher |
| ECE | calibration error over 15 confidence bins: gap between confidence and accuracy | lower |
| ESR (evidence sensitivity) | % of `_B` twins predicted contradicted | 100% |
| Unrelated rate | % of `_C` twins predicted unverified | 100% |
| Paraphrase consistency | % of `_P` twins with the same verdict as the original | 100% |
| Cross-lingual consistency | % of `_A` / `_M` twins with the same verdict as the English original | 100% |

- **Statistics:**
  - 3 seeds (42, 43, 44), reported as mean ± sd.
  - Paired bootstrap, 1,000 resamples grouped by source claim, giving 95% CIs and p-values.
  - Cohen's κ for agreement between people.
- **Comparisons:**
  - 4 baselines: claim-only, similarity + LightGBM, zero-shot NLI rule, LLM judge.
  - 3 ablations: no twin loss, no reranker, no entity match.
  - NLI comparison: zero-shot vs fine-tuned, on the 300 gold pairs.
- **Human evaluation:** 60 explanations rated for verdict correctness, support from the cited snippet, and readability (1–5); 120 twins hand-checked.

### Slide 9: What a twin looks like (a real example from our test set)
- **Title:** A twin changes one thing, so the right verdict is known
- **On the slide:**
  - **Original claim (supported):** "North Carolina is one of 'very few states' that give the governor sole discretion to pick judges."
  - **Original evidence:** "Very few states allow a governor unilateral and sole discretion ... Judges in **North Carolina** are selected through partisan elections."

| Twin | What changes | Real example | Right verdict |
|---|---|---|---|
| `_P` paraphrase | claim reworded | "... that **allow** the governor sole discretion ..." | same (supported) |
| `_B` evidence swap | one place, person or date swapped in the evidence | evidence now says "Judges in **New Hampshire** ..." | contradicted |
| `_C` unrelated | evidence from an unrelated claim | evidence about the TPP trade deal | unverified |
| `_A` Arabic | claim and evidence translated | Arabic version | same |
| `_M` mixed | Arabic claim, English evidence | Arabic claim + English snippet | same |

---

## 2. Datasets and pre-processing

### Slide 10: Datasets

| Dataset | Source and size | Evidence | Labels | How we use it | Licence |
|---|---|---|---|---|---|
| **MultiFC** (Augenstein et al., 2019) | 34,918 claims from 26 fact-check sites; 21,057 kept after mapping and cleaning | up to 10 Google snippets per claim, saved in 2019 | 165 site-specific labels (116 in train + val) | train 16,847 · val 2,104 · test 2,106; English twins | research use only |
| **AVeriTeC** (Schlichtkrull et al., 2023) | 4,568 claims from 50 fact-check organisations; 2,990 used | human-written Q&A from pages dated before the claim | Supported, Refuted, Not Enough Evidence, Conflicting | clean English test (429 claims) | CC BY-NC 4.0 |
| **X-FACT Arabic** (Gupta & Srikumar, 2021) | 2,090 Arabic claims, all from misbar.com; 919 kept | top-5 Google snippets | 7 labels | Arabic NLI training pairs; test (135 claims: 123 false) | research |
| **M4FC Arabic** (Geng et al., 2025) | 614 Arabic claims (613 false) + 156 true captions | none in the main file | true / false | real Arabic test only (authors: evaluation only) | research |

- **Gold sets we labelled by hand (testing only):** 300 NLI pairs (150 English by Himasri, 150 Arabic by Youssef); 300 WhatsApp-style messages with gold claims (102 English, 99 Arabic, 99 mixed).

### Slide 11: Pre-processing, step by step

| Step | What we did | Result |
|---|---|---|
| 1. Get the text | Used the authors' MultiFC release (the Hugging Face mirror has no snippet text) | snippets for 34,836 claims |
| 2. Map labels | Listed all 116 MultiFC labels by hand → supported / contradicted / unverified; dropped unclear ones (half true, mixture, misleading, satire, promise ratings, GossipCop 0–10 scores) | 9,877 of 31,433 claims dropped |
| 3. Remove leaked evidence | Dropped snippets from fact-checking sites (PolitiFact, Snopes, FactCheck.org, FullFact, Misbar, Fatabyyano and others) | −25,993 snippets (PolitiFact 13,585, Snopes 4,002) |
| 4. Remove copies of the claim | Dropped snippets that overlap the claim by more than 90% | −16,769 snippets |
| 5. Remove duplicates, cap evidence | Dropped repeated snippets and duplicate claims; kept at most 10 snippets per claim | −219 snippets, −73 claims |
| 6. Set aside empty claims | Claims with no snippets left go to a separate file | 426 claims |
| 7. Split | Our own 80 / 10 / 10 split, stratified by label, seed 42; no claim in two splits | 16,847 / 2,104 / 2,106 |
| 8. AVeriTeC format | Each Q&A pair → one snippet "question? answer"; Archive.org links unwrapped | 7,671 snippets |

- **Class balance:** 34% supported, 63% contradicted, 3.5% unverified → we report **macro-F1** (each class counts equally).

### Slide 12: Augmentation: how each twin is made

| Twin | How it is generated | Automatic check |
|---|---|---|
| Sources | supported claims with ≥ 2 snippets of relevance > 0.3; 500 test + 3,000 train; fixed with seed 42 | — |
| `_P` | Qwen3-8B rewrites the claim: "same meaning, same names, dates and numbers" | kept only if spaCy finds the same entities |
| `_B` | one place (GPE/LOC), person or date in the claim is swapped in every snippet that mentions it; values drawn from our own data; dates shifted 2–5 years | the swapped entity must appear in the evidence |
| `_C` | evidence replaced with snippets from a claim that shares no entities and has reranker relevance < 0.1 | relevance check |
| `_A`, `_M` | 200 test sources translated by Qwen3-8B with a 3,044-name glossary; swaps made in English before translation | 50 translations checked by Youssef |

| Counts | Test | Train |
|---|---|---|
| English twins | 955 (174 `_P`, 281 `_B`, 500 `_C`) | 5,705 |
| Arabic and mixed twins | 642 (147 `_A`, 192 `_M`, 145 `_B`, 158 `_C`) | — |

- **Hand check (Himasri + Feba, 120 twins, 30 shared):** Cohen's κ = 0.57, 83% agreement. Valid: `_P` 97.5%, `_C` 100%, `_B` 22.5%. Translations: 27 of 50 (54%) fully correct.

---

## 3. Solution architecture

### Slide 13: Pipeline

| Step | Model | Input → output |
|---|---|---|
| 1. Claim extraction | Qwen3-8B, 4-bit, 6 examples in the prompt (2 per language) | message → JSON list of claims |
| 2. Evidence retrieval (demo only) | BM25 (keywords) + BGE-M3 dense embeddings in FAISS (meaning); top 20 from each | claim → candidates from 163,244 snippets |
| 3. Relevance | `bge-reranker-v2-m3`, fine-tuned (reads claim + snippet together) | each snippet → relevance 0–1; demo keeps the best 5 |
| 4. Entities | spaCy `en_core_web_trf` | who / where / when in the claim |
| 5. NLI | `mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, zero-shot | each (snippet, claim) → P(entail), P(contradict), P(neutral) |
| 6. Verdict head | MLP 10 → 64 → 3, trained with the twin loss | 10 features → supported / contradicted / unverified |
| 7. Calibration | temperature scaling + "unverified" threshold | honest confidence |
| 8. Explanation | Qwen3-8B, 4-bit, may cite only snippets [1]–[3] | 1–2 sentences in the claim's language |

- Steps 2–3 (retrieve) + step 8 (generate from what was retrieved) = **retrieval-augmented generation (RAG)**.
- **Experiments vs demo:** each dataset claim already has its own snippets, so experiments skip step 2. This keeps the evidence under our control, which the twins require. Retrieval is tested separately (recall@5).
- **Design rule:** the verdict never sees the image "AI-generated?" score.

### Slide 14: Why each model (tested, not assumed)

| Step | Chosen | Compared with | Result |
|---|---|---|---|
| Claim extraction | Qwen3-8B few-shot | rule-based splitter; Qwen3-4B | F1 0.984 vs 0.786; 4B 0.977 |
| Retrieval | BM25 + BGE-M3 | BM25 only; dense only | recall@5 EN 0.945 vs 0.91 / 0.87; AR 0.92 vs 0.86 / 0.80 |
| Reranker | fine-tuned | off the shelf | accuracy 0.719 → 0.980; AUROC 0.980 → 0.998 |
| NLI | mDeBERTa-v3-base (279M, 90 ms/pair) | XLM-R-large (560M, 178 ms); DeBERTa-v3-large (EN only); fine-tuned v1, v2 | gold macro-F1 EN 0.367 / 0.382 / 0.427, AR 0.443 / 0.502; CIs overlap → smallest model covering both languages |
| Verdict head | MLP + twin loss | logistic regression; LightGBM (with and without twins as data) | +8.5 to +9.3 ESR over twins-as-data (p < 0.001); LightGBM keeps only 66.7% of paraphrases |
| Explanation | Qwen3-8B | Qwen3-4B | faithfulness 0.320 vs 0.352 |

### Slide 15: The verdict head, exactly
- **Title:** The verdict head turns 10 evidence features into one verdict
- **The 10 features (computed per claim from its snippets):**

| # | Feature | Meaning |
|---|---|---|
| 1 | w_entail | average P(entail) over snippets, weighted by reranker relevance |
| 2 | w_contradict | the same for P(contradict) |
| 3 | w_neutral | the same for P(neutral) |
| 4 | max_entail | highest P(entail) among relevant snippets (relevance ≥ 0.5) |
| 5 | max_contradict | highest P(contradict) among relevant snippets |
| 6 | share_relevant | share of snippets with relevance ≥ 0.5 |
| 7 | match_place | share of the claim's places (spaCy GPE/LOC) found in the evidence text |
| 8 | match_date | share of the claim's dates found (years must match) |
| 9 | match_person | share of the claim's people found (surname is enough) |
| 10 | n_snippets | number of snippets ÷ 10 |

- **The network:** 10 inputs → **64 hidden units** (one hidden layer with ReLU; each unit learns one combination of the features, e.g. "strong contradiction AND relevant snippets AND the place is missing") → dropout 0.2 → **3 outputs** (one score per verdict), turned into probabilities by softmax. About 900 weights.
- **Why an MLP:** the twin loss compares the probabilities of an original and its twin, so the model must be trained with gradients. Tree models such as LightGBM cannot use that loss.
- **Note:** entity matching is string matching in English, so it is switched off for the fair Arabic comparison.

### Slide 16: Calibration, in plain words
- **Temperature scaling:** before softmax, divide the 3 output scores by one number T. With T > 1 the model becomes less confident, with T < 1 more confident. T is fitted on the validation set so that the confidence matches how often it is right (minimum log-loss).
  - Our value: **T = 0.89** (the head was slightly under-confident). Validation calibration error (ECE) fell from 8.7% to 7.7%. Accuracy does not change; only the confidence does.
- **"Unverified" threshold τ:** if the highest probability is below τ, the answer becomes "unverified". τ was chosen on validation to maximise macro-F1. It stayed at **0** in 5 of 6 fits because the head already says "unverified" often enough.

---

## 4. Training and fine-tuning

### Slide 17: The loss function we trained with
- **On the slide:** **Loss = weighted CE (originals) + λ_inv × KL(original ‖ twin) for `_P/_A/_M` + λ_sens × CE(`_B` → contradicted, `_C` → unverified)**
  1. **Class-weighted cross-entropy on the original claims.** Weights are n / (3 × class count), so the rare "unverified" class counts.
  2. **KL divergence between an original and its `_P`, `_A`, `_M` twins.** Their probability distributions should match. The original's side is fixed (detached), so only the twin is pushed towards the original.
  3. **Cross-entropy on `_B` and `_C` twins with their known answers** (contradicted, unverified).
- **Settings:**
  - Adam, learning rate 1e-3, up to 30 epochs, early stopping on validation macro-F1 (patience 5).
  - Each batch holds 64 originals together with all their twins.
  - λ grid: λ_inv ∈ {0, 0.5, 1, 2} × λ_sens ∈ {0, 0.5, 1}. We picked the best twin score with validation macro-F1 within 1 point of λ = 0 → **λ_inv = 1, λ_sens = 0.5**.
  - Seeds 42, 43, 44.

### Slide 18: What we fine-tuned

| Model | Why | How | Data | Result | Kept? |
|---|---|---|---|---|---|
| Reranker `bge-reranker-v2-m3` | off the shelf it mixed up relevant and irrelevant snippets | 1 epoch, lr 2e-5 | 20,000 pairs: own snippet = relevant, random snippet from another claim = irrelevant | accuracy 0.719 → 0.980; AUROC 0.980 → 0.998 | yes (on Hugging Face) |
| NLI mDeBERTa, v1 | adapt NLI to fact-check snippets | lr 2e-5, batch 16, max length 256, 2 epochs, fp16 | 32,658 weak pairs, labels copied from the claim's verdict | gold EN F1 0.367 → **0.124**; predicted "neutral" for 0.7% of pairs (gold: 88%) | no |
| NLI mDeBERTa, v2 | fix v1's shortcut | same | 18,525 pairs kept only where zero-shot agreed + hard neutrals + 3,226 `_B` pairs | tie (mean 0.403 vs 0.407) | no |
| Qwen3-4B as verdict model (QLoRA) | test whether an LLM reading the snippets beats our head | 4-bit NF4 + double quantization; LoRA rank 16, α 32, dropout 0.05, on all attention + MLP layers; lr 2e-4; effective batch 16; 1 epoch; max length 384 | prompt = claim + top-3 reranked snippets; 4,500 balanced originals (66 min on T4), then + 1,800 twins (93 min) | see results | comparison |

- **Lesson from NLI v1:** labels copied from the claim taught "same topic → copy the verdict". Automatic labels teach shortcuts.

### Slide 19: Quantization
- **What:** model weights stored in **4 bits** instead of 16 (bitsandbytes library). Calculations still run in 16-bit (fp16 / bf16).
- **Why:** Qwen3-8B needs about 16 GB in 16-bit, more than a free Colab T4 offers once activations are counted. In 4-bit it needs about 5–6 GB.
- **Which 4-bit format:**
  - **NF4** ("NormalFloat 4", designed for model weights, from QLoRA) for paraphrasing, explanations, the LLM judge, and QLoRA training (with double quantization, which also compresses the scaling constants).
  - The library's default **FP4** for claim extraction and translation.
- **Did it hurt?** Claim extraction still reached F1 0.984. We did not run a 16-bit comparison, because the 8B model does not fit a T4 at 16 bits.
- **Compute:** free Colab T4 GPUs for almost all runs; one L4 for NLI fine-tuning.

---

## 5a. Results

### Slide 20: RQ2: the twin loss
- **Title:** With the twin loss, the verdict follows the evidence
- MultiFC test, mean ± sd over 3 seeds (`results/H4`):

| | Macro-F1 | ESR (`_B` → contradicted) | Unrelated rate (`_C` → unverified) | Paraphrase consistency | ECE |
|---|---|---|---|---|---|
| No twin loss | 36.3 ± 0.5 | 52.2 ± 3.0 | 42.4 ± 24.4 | 91.2 | 5.4 |
| Twin loss | 35.2 ± 0.7 | **89.7 ± 0.9** | **100.0 ± 0.0** | 91.4 | 8.2 (7.0 calibrated) |

- Paired bootstrap (1,000 resamples by source claim): ESR +35 to +42 points and unrelated +32 to +81 points, **p < 0.001 in every seed**. Macro-F1 −0.1 to −2.3, significant in one seed only.
- AVeriTeC test: 38.3 vs 38.0 macro-F1 (not significant).

### Slide 21: RQ1: audit of all systems
- **Title:** The most accurate systems read the evidence least
- MultiFC test + twins; AVeriTeC macro-F1 (`results/F4`, `H4`, `H7`):

| System | Macro-F1 | ESR | Unrelated | Paraphrase | AVeriTeC F1 |
|---|---|---|---|---|---|
| Claim-only TF-IDF (no evidence) | 48.1 | 44.5 | 1.2 | 84.5 | 33.3 |
| Similarity + LightGBM | 38.5 | 59.2 | 1.4 | 73.0 | 28.1 |
| Zero-shot NLI rule | 34.6 | 62.3 | 53.6 | 87.4 | 44.0 |
| Qwen3-8B judge (prompted) | 23.3 | 10.0 | 95.0 | 94.8 | 49.4 |
| Qwen3-4B judge (prompted) | 39.1 | 58.0 | 13.6 | 96.0 | **61.3** |
| Qwen3-4B QLoRA, originals | **50.0** | 30.6 | 14.8 | 78.7 | 57.4 |
| Qwen3-4B QLoRA, + twins | 44.2 | 67.3 | 99.2 | 87.4 | 51.0 |
| MLP, no twin loss | 36.3 | 52.2 | 42.4 | 91.2 | 38.0 |
| **MLP + twin loss (ours)** | 35.2 | **89.7** | **100.0** | 91.4 | 38.3 |

- **Visual (recommended):** a scatter plot, macro-F1 on x and unrelated rate on y.

### Slide 22: Ablations
- **Title:** Removing the reranker hurts most
- MultiFC test, mean of 3 seeds (`results/H5`):

| Variant | Macro-F1 | ESR | Unrelated | Paraphrase |
|---|---|---|---|---|
| Full model (twin loss) | 35.2 | 89.7 | 100.0 | 91.4 |
| − reranker | 30.1 | 68.7 | 81.0 | 87.4 |
| − entity match | 33.8 | 86.8 | — | — |
| − twin loss | 36.3 | 52.2 | 42.4 | 91.2 |

- Reranker removal: p < 0.001 in every seed. Entity match: significant in one seed only.

### Slide 23: Components
- **Title:** Each component, measured on its own

| Component | Test | Result |
|---|---|---|
| Claim extraction | 200 held-out messages; a match = BGE-M3 cosine ≥ 0.8 | F1 0.984 (EN 0.989, AR 0.966, mixed 1.000) vs rule-based 0.786 |
| Retrieval | recall@5 on 250 claims, 163,244-snippet index | EN 0.945, AR 0.92 |
| Reranker | own vs random snippets | accuracy 0.980, AUROC 0.998 |
| NLI | 300 gold pairs | macro-F1 EN 0.367, AR 0.443 |
| Explanations | 250 claims; NLI checks each sentence against its cited snippet | faithfulness 0.32; citation accuracy 0.31 |

### Slide 24: RQ3: Arabic and mixed
- **Title:** A fair system scores almost the same in Arabic as in English
- Arabic twin set, 192 sources, 3 seeds (`results/Y5`):

| System | Macro-F1 EN | Macro-F1 AR | Unrelated EN / AR | Same verdict as English: `_A` | `_M` |
|---|---|---|---|---|---|
| Twin loss, no entity features (fair) | 0.605 | **0.597** | 100 / 100 | **73.9%** | **79.5%** |
| Twin loss, with entity features | 0.643 | 0.559 | 100 / 100 | 63.0% | 77.3% |
| No twin loss | 0.356 | 0.390 | 23.1 / 24.7 | 62.4% | — |
| Zero-shot NLI rule | 0.541 | 0.494 | 54.7 / 45.6 | 64.6% | — |

- Real X-FACT Arabic test (135 claims, 123 false): every system scores 0.19–0.33 macro-F1, close to chance.

### Slide 25: LLM size and human evaluation
- **Qwen3-4B vs Qwen3-8B** (same prompts and scripts, both 4-bit):

| Role | 8B | 4B |
|---|---|---|
| Claim extraction F1 | 0.984 | 0.977 |
| Explanation faithfulness | 0.320 | 0.352 |
| Judge, MultiFC macro-F1 | 23.3 | **39.1** (p < 0.001) |
| Judge, AVeriTeC macro-F1 | 49.4 | **61.3** (p < 0.001) |

- **Human evaluation** (60 explanations: 40 English rated by Himasri and Feba, 20 Arabic by Youssef):
  - Agreement between raters: κ 0.49 (verdict), 0.35 (support), 0.48 (readability, weighted).
  - Gold verdict supported by the cited snippets: 37.5% / 47.5% (English, two raters), 10% (Arabic).
  - Readability: 3.9–4.1 out of 5 (English), 3.7 (Arabic).

---

## 5b. Analysis and discussion

### Slide 26: Why the results look this way

| Finding | Evidence |
|---|---|
| MultiFC rewards guessing from the claim | claim-only scores 48.1; raters found the snippets settle the verdict in only 38–48% of English notes |
| Search snippets repeat claims, true or false | a snippet with entail ≥ 0.9 exists for 39% of true and 34% of false claims (figure `fig_entail_shortcut.png`) |
| Automatic labels teach shortcuts | NLI v1 fell from 0.37 to 0.12 on gold pairs |
| Twins add the missing signal | twin training raises ESR and unrelated rate for both the MLP and Qwen3-4B |
| Smaller can be better | Qwen3-4B beat 8B as a judge on both datasets |

### Slide 27: Limitations (stated before anyone asks)

| Limitation | Number | What it means | Next step |
|---|---|---|---|
| `_B` twins are noisy | 22.5% valid | ESR measures reaction to changed evidence more than true contradiction | swap only claim entities; validate by hand |
| Bias towards "contradicted" | twin head calls 61% of supported sources contradicted (40% without) | part of the ESR gain is bias; paired ESR still 76.4% vs 32.7% | report paired ESR |
| Unrelated twins are easy | `_C` relevance < 0.1 by construction | 100% is partly easy | near-topic unrelated twins |
| Translation quality | 54% fully correct | limits the Arabic results | human-checked Arabic, dialects |
| Explanations | faithfulness 0.32 | notes go beyond their cited snippet | constrained generation |
| Small Arabic test | 135 claims, 91% false | X-FACT results only indicative | balanced native Arabic data |
| One seed for QLoRA | seed 42 | LLM results less certain | seeds 43, 44 |

- **Real noisy `_B` example (good for this slide):** "Iraq" was swapped to "Neosho" (a small US town): "the recent events in Neosho". The result is nonsense, not a contradiction.

---

## 6. Ethics

### Slide 28: Ethical issues

| Issue | Risk | What we did |
|---|---|---|
| Harm and over-reliance | a wrong "false" can damage a person; a wrong "true" spreads a rumour | "unverified" option, calibrated confidence, cited snippets; an assistant, not a judge |
| Bias | US-heavy MultiFC sources; English-only entity features; no dialects | results reported per language; a fair system without English-only features |
| Privacy | real forwarded messages contain personal data | no real messages used; demo examples labelled "illustrative"; models run locally, nothing sent to paid APIs |
| Licences | dataset terms | MultiFC research use; AVeriTeC CC BY-NC 4.0; all cited, non-commercial |
| Misuse | labelling opinions as false; learning how to evade checks | verdicts only from retrieved evidence; open code for scrutiny |
| AI-looking images | a real but AI-looking photo could make a true post look fake | the verdict never sees the authenticity score |
| Environment | large LLMs are costly to run | 4-bit models on free T4 GPUs; 4B kept where it matched 8B |

---

## 7. Conclusion and lessons learned

### Slide 29: Conclusion
- **RQ1:** On MultiFC, the most accurate systems read the evidence least. Claim-only scores 48.1 and catches 1.2% of unrelated evidence.
- **RQ2:** The twin loss raises ESR from 52% to 90% and the unrelated rate from 42% to 100%, for about 1 macro-F1 point. Twins help an LLM too: Qwen3-4B goes from 14.8% to 99.2% on unrelated evidence.
- **RQ3:** A fair system scores 0.597 in Arabic vs 0.605 in English and keeps 74% (Arabic) and 80% (mixed) of English verdicts.

### Slide 30: Lessons learned
1. Check the data before the model: leaked snippets and label noise shape every score.
2. Automatic labels teach shortcuts; small hand-labelled gold sets catch this.
3. Bigger is not always better: 4-bit Qwen3-4B matched or beat 8B.
4. Measure what you want: accuracy alone would have picked the model that reads the evidence least.

### Slide 31: Links
- **Code:** github.com/himaallu/Group-Project (tag v1.0.1); `run_all.sh` reproduces the main tables.
- **Demo:** Gradio app, `notebooks/R10_demo_colab.ipynb`.
- **Reranker:** huggingface.co/RathiAbeth/twincheck-reranker-v1
- **Data:** MultiFC (authors' release), AVeriTeC, X-FACT (github.com/utahnlp/x-fact), M4FC (github.com/UKPLab/M4FC).

---

## Sections a thesis or paper presentation usually has that the current deck is missing

| Add | Where | Why markers expect it |
|---|---|---|
| **Outline / agenda** | after the title | tells the audience the structure (the 7 sections) |
| **Research methodology** (slides 8a, 8b above) | after "Our approach" | shows the study is designed, with variables, controls, hypotheses and statistics; the most common thing examiners ask about |
| **Contributions** | end of section 1, or before the conclusion | an explicit list of what is new, "to our knowledge": (1) one-change twins on real web claims, (2) the twin loss, (3) an EN/AR/mixed consistency test, (4) evidence that accuracy and evidence-reading diverge |
| **Scope and assumptions** | after the research questions | what we do *not* claim: no live web search in experiments, no dialects, text verdicts only (images are MAAI901) |
| **Error analysis** | in 5b | 2–3 real failure cases with their cause. Your 10 cases are in `results/H6/error_cases.csv`, but the `cause` column is still empty; fill it first |
| **Live demo** | before the conclusion | the rubric rewards a working system; show one English and one Arabic message |
| **Team contributions** | near the end | the rubric has a team-collaboration criterion; one line per person |
| **References** | last content slide | the full citations for every paper and fact you showed |
| **Backup slides** | after "Thank you" | ready for questions: hyperparameters, full metric tables, λ grid (`fig_lambda_grid.png`), calibration (`fig_reliability.png`), NLI selection, Qwen 4B vs 8B details |

**Suggested final order:**
1. Title
2. Outline
3. Hook
4. Rationale
5. Problem
6. Literature review (2 slides)
7. Gaps and research questions
8. Scope
9. Contributions
10. Approach
11. Research methodology (2 slides)
12. Twin example
13. Datasets
14. Pre-processing
15. Augmentation
16. Architecture
17. Model choices
18. Verdict head
19. Calibration
20. Loss
21. Fine-tuning
22. Quantization
23. Results (6 slides)
24. Analysis
25. Error analysis
26. Limitations
27. Ethics
28. Demo
29. Conclusion
30. Lessons
31. Team
32. Links and references
33. Thank you / Q&A
34. Backup slides

That is too many for a short talk. For a 10–15 minute slot, keep the main thread (hook → problem → research questions → methodology → twins → pipeline → loss → the three key results tables → limitations → ethics → conclusion → demo) and move the details into backup slides.

---

## Sources for the hook slides
- Vosoughi, S., Roy, D. & Aral, S. (2018). The spread of true and false news online. *Science* 359(6380).
- BBC News (17 Apr 2024). "What is cloud seeding and did it cause Dubai flooding?" bbc.com/news/science-environment-68839043
- World Economic Forum. Global Risks Report 2025 (press release, 15 Jan 2025) and Global Risks Report 2026 (weforum.org/publications/global-risks-report-2026).
- Reuters Institute (2025). Digital News Report 2025, executive summary.
- Duke Reporters' Lab (2025). "2025 census: Fact-checkers persevere ..." (443 active projects); Poynter / Reporters' Lab 2026 census (10 new vs 30+ stopped in 2025; 437 by June 2026).
- DataReportal (2025). Digital 2025: The United Arab Emirates (99.0% internet penetration).
- UAE Federal Decree-Law No. 34 of 2021 on Countering Rumors and Cybercrimes, Article 52 (uaelegislation.gov.ae).
