# VERITE benchmark and related work (Papadopoulos et al., CERTH-ITI / MeVer group)

Scope: (a) the VERITE paper and CHASMA; (b) later papers evaluating on VERITE (RED-DOT, MUSE/AITR, LAMAR, DEFAME, MAD-Sherlock, others); (c) the "Similarity over Factuality" (WACV 2025) shortcut finding; (d) work on image authenticity / AI-generated images in relation to verdicts.

Method note: full-text numbers were taken from the arXiv HTML versions of each paper, using an automated extraction tool that pulls text from the page. Abstract-level claims were read directly. Where an extracted number looked inconsistent, it is flagged. Before any number goes into a TwinCheck report, check it against the PDF table it came from (CLAUDE.md §10: "every number must come from a file").

Key papers covered (title, authors, venue, year, URL):
1. **VERITE**: "VERITE: A Robust Benchmark for Multimodal Misinformation Detection Accounting for Unimodal Bias". S.-I. Papadopoulos, C. Koutlis, S. Papadopoulos, P. C. Petrantonakis. *International Journal of Multimedia Information Retrieval* 13(1):4, 2024 (arXiv 2304.14133, 2023). https://arxiv.org/abs/2304.14133 ; https://link.springer.com/article/10.1007/s13735-023-00312-6
2. **RED-DOT**: "RED-DOT: Multimodal Fact-checking via Relevant Evidence Detection". Same four authors. arXiv 2311.09939 (2023/2024). Published in an IEEE journal in 2025: the IEEE Xplore record (doc 10948326) is in journal number 6570650, which appears to be *IEEE Transactions on Computational Social Systems*; confirm this before citing. https://arxiv.org/abs/2311.09939 ; https://ieeexplore.ieee.org/iel8/6570650/6780646/10948326.pdf
3. **Similarity over Factuality (MUSE / AITR)**: "Similarity over Factuality: Are we Making Progress on Multimodal Out-of-Context Misinformation Detection?" Same four authors. WACV 2025 (arXiv 2407.13488). https://arxiv.org/abs/2407.13488 ; https://openaccess.thecvf.com/content/WACV2025/papers/Papadopoulos_Similarity_over_Factuality_Are_we_Making_Progress_on_Multimodal_Out-of-Context_WACV_2025_paper.pdf
4. **LAMAR / "MisCaption This!"**: "Latent Reconstruction from Generated Data for Multimodal Misinformation Detection". Same four authors. arXiv 2504.06010 (2025; revised Jan 2026). Venue not confirmed. https://arxiv.org/abs/2504.06010
5. **DEFAME**: "DEFAME: Dynamic Evidence-based FAct-checking with Multimodal Experts". T. Braun, M. Rothermel, M. Rohrbach, A. Rohrbach. arXiv 2412.10510. The arXiv page metadata lists the keyword "ICML" (ICML 2025); confirm the venue before citing. https://arxiv.org/abs/2412.10510
6. **MAD-Sherlock**: "MAD-Sherlock: Multi-Agent Debate for Visual Misinformation Detection". K. Lakara, G. Channing, C. Rupprecht, J. Sock, P. Torr, J. Collomosse, C. Schroeder de Witt. arXiv 2410.20140 (2024, revised Oct 2025). Venue not stated on the page. https://arxiv.org/abs/2410.20140
7. **CREDULE / EVVER-Net** (leaked evidence in VERITE): "Credible, Unreliable or Leaked?: Evidence Verification for Enhanced Automated Fact-checking". Z. Chrysidis, S.-I. Papadopoulos, S. Papadopoulos, P. C. Petrantonakis. arXiv 2404.18971 (2024). https://arxiv.org/abs/2404.18971
8. Related to image authenticity and verdicts: "Fact or Fake? Assessing the Role of Deepfake Detectors in Multimodal Misinformation Detection" (Sagar et al., arXiv 2602.01854, 2026) https://arxiv.org/abs/2602.01854 ; and "TrustLift: When Images Make False Claims Look True" (Borodulin et al., ACM IMPACT '26 workshop, 2026) https://dl.acm.org/doi/10.1145/3841453.3841505

---

## How is VERITE built and evaluated (pairwise accuracy, splits, size ~1,000 pairs)?

### Takeaway
VERITE is a small, test-only benchmark of **1,000 real-world image–caption pairs**: 338 True, 338 Miscaptioned (MC) and 324 Out-of-Context (OOC). The pairs come from 260 Snopes and 78 Reuters fact-checks. It uses "modality balancing": every image appears in both a true pair and an MC pair, and every caption appears in both a true pair and an OOC pair. This is meant to stop a model from scoring well by looking at only one modality. Models are trained elsewhere (usually NewsCLIPpings or synthetic data) and evaluated on VERITE as binary "True vs OOC" and "True vs MC" accuracy, plus 3-class multiclass accuracy.

### Cited Findings
- **Motivation (unimodal bias):** the authors show that in widely used multimodal misinformation detection (MMD) benchmarks, biased or unimodal models can outperform multimodal ones on what should be a multimodal task. They identify this in **VMU-Twitter** and **COSMOS**. — [VERITE, arXiv 2304.14133](https://arxiv.org/abs/2304.14133)
- **VMU-Twitter evidence:** an image-only model D⁻(I) reaches **83.7%** accuracy, the highest of any model. The multimodal D⁻(I;C) reaches 80.5% and D(I,C) 79.7%; prior multimodal methods reach Intra+Inter 83.1% and BCMF 81.5%. — [VERITE HTML v3](https://arxiv.org/html/2304.14133v3)
- **COSMOS evidence:** a text-only model D⁻(C) trained on CHASMA-D reaches **72.6%**, beating the multimodal D⁻(I;C) (66.9%) and D(I,C) (61.8%). This is text-side unimodal bias. — [VERITE HTML v3](https://arxiv.org/html/2304.14133v3)
- **Design principles:** VERITE "incorporates real-world data, excludes 'asymmetric multimodal misinformation' and utilizes 'modality balancing'". — [VERITE abstract](https://arxiv.org/abs/2304.14133)
- **Asymmetric multimodal misinformation** means cases where one dominant modality carries the misinformation and the other has little or no influence, for example a false claim in the text alone or a manipulated image alone. VERITE excludes such cases. — [VERITE HTML v3](https://arxiv.org/html/2304.14133v3)
- **Modality balancing:** "each image is present once in its truthful pair and once in the MC pair while each caption is present once in its truthful pair and once in the OOC pair". — [VERITE HTML v3](https://arxiv.org/html/2304.14133v3)
- **Size and sources:** "We collected 260 articles from Snopes and 78 from Reuters that met our criteria which translates to 338 Truthful, 338 Miscaptioned and 324 Out-of-Context pairs". The final `VERITE.csv` has 1,000 rows with columns `caption, image_path, label`. The source `VERITE_articles.csv` has 337 rows with columns `id, true_url, false_caption, true_caption, false_url, query, snopes_url`. — [GitHub stevejpapad/image-text-verification](https://github.com/stevejpapad/image-text-verification)
- **How the pairs are made (from the repo example):**
  - True pair: the real image with its correct caption.
  - Miscaptioned (MC) pair: the same image with the false caption from the fact-check. In the repo example, a June 2020 Murmansk bridge collapse is re-captioned as happening in "2022 during the Russia-Ukraine war". So MC changes date, place or event in the text.
  - Out-of-context (OOC) pair: the true caption with a *different* image (`false_<id>.jpg`). The images come from the Snopes/Reuters articles and from Google Images.
  — [GitHub repo](https://github.com/stevejpapad/image-text-verification)
- **Time span and topics:** the Snopes articles date from January 2001 to January 2023 and the Reuters articles from January 2021 to January 2023. Topics: world news 29.04%, politics 27.94%, culture and arts 8.82%, entertainment 7.72%, sports 3.67%, environment 3.66%, others smaller. — [VERITE HTML v3](https://arxiv.org/html/2304.14133v3)
- **Distribution and licence:** the repo provides image URLs, not images, plus precomputed CLIP ViT-L/14 features in `/VERITE`. July 2026 update: "Because many of the original image URLs are no longer accessible, the raw images are provided on Hugging Face (stefpapad/VERITE). Access requires an official academic or research institution email address." The code is Apache-2.0. The dataset is "intended solely for research purposes". — [GitHub repo](https://github.com/stevejpapad/image-text-verification); also mirrored on [Zenodo record 10474450](https://zenodo.org/records/10474450)
- **Evaluation protocol:** results are reported as "True vs OOC" accuracy, "True vs MC" accuracy, and multiclass (3-way) accuracy. — [VERITE HTML v3](https://arxiv.org/html/2304.14133v3)
- In later work (WACV 2025), models are trained on NewsCLIPpings and evaluated on VERITE under an "OOD-CV" protocol: k=3 folds of VERITE, validating and checkpointing on one fold and testing on the others. Results are reported as mean (standard deviation). — [Similarity over Factuality HTML](https://arxiv.org/html/2407.13488v1)
- **Evidence for VERITE:** the original VERITE has no web evidence. External evidence (text and images collected with the Google API, following Abdelnabi et al.'s CCN procedure) was added later by RED-DOT and reused by the WACV 2025 paper. — [RED-DOT HTML](https://arxiv.org/html/2311.09939); [Similarity over Factuality HTML](https://arxiv.org/html/2407.13488v1)
- **Leaked evidence in that evidence set:** EVVER-Net classifies VERITE's evidence (all 3 classes) as **35.3% fact-checked (leaked)**, 22.1% unreliable and 42.6% credible. Broken down:
  - True + Miscaptioned classes: 45.2% leaked, 19.6% unreliable, 35.2% credible.
  - OOC class: 14.5% leaked, 27.3% unreliable, 58.2% credible.
  - For comparison, NewsCLIPpings+: 14.1% leaked, 21.4% unreliable, 64.5% credible.
  The authors explain that Google searches for the claims "often return the original or similar fact-checking articles". — [CREDULE, arXiv 2404.18971](https://arxiv.org/html/2404.18971)
- **VERITE 24/25:** a newer extension of 50 samples from Snopes/Reuters about events from October 2024 to July 2025, balanced between truthful and miscaptioned, built with the same criteria. It was introduced with LAMAR. — [LAMAR HTML](https://arxiv.org/html/2504.06010)

### Inferences
- VERITE's MC class is structurally close to TwinCheck's `_B` evidence-swap twin: the same image with one entity (date, place, event) changed in the text. VERITE's OOC class is close to `_C` (mismatched pairing). The difference is that VERITE changes the *claim/caption* against a fixed image. TwinCheck changes the *evidence* against a fixed claim, or (with `_V`) changes the image's *authenticity fingerprint* while keeping its content.
- VERITE's modality balancing (the same item appears with both labels) is the same idea as TwinCheck's "change exactly one thing" twins. It is worth citing as the closest precedent on the vision side, using "to our knowledge" wording.
- Because 35–45% of the collected VERITE evidence is from fact-checking sites, any evidence-based VERITE run in TwinCheck (F3 leakage test) should apply the same fact-check-domain filter as `src/data/clean.py`.
- Many original image URLs are dead, so Feba's F1 image-download failure count will likely be high. The Hugging Face mirror (academic email required) is the official fallback.

### Gaps
- The exact filtering criteria used to exclude asymmetric cases (for example, whether edited or AI-generated images were explicitly excluded) were not extracted verbatim. Read Section 3 of the PDF.
- The number of VERITE pairs with no evidence, or with broken images, as of 2026 is not reported.

---

## What results do the original and subsequent methods report on VERITE?

### Takeaway
"True vs OOC" accuracy has risen from about 72–75% (CHASMA-era DT-Transformer) to about 83% (AITR, trained on NewsCLIPpings with web evidence). **"True vs Miscaptioned" stays near chance (about 51–66%) for every CLIP-based detector.** Only retrieval-augmented GPT-4o agents (DEFAME, MAD-Sherlock) report about 80% or more on True vs MC. Multiclass accuracy tops out around 52–55% for trained detectors.

### Cited Findings
**VERITE paper (CHASMA, DT-Transformer), IJMIR 2024**
- **CHASMA** (Crossmodal HArd Synthetic MisAlignment) creates synthetic training data. It pairs legitimate images with misleading human-written captions, using CLIP to keep crossmodal relations, so that the training pairs are hard and plausible. — [VERITE abstract](https://arxiv.org/abs/2304.14133)
- **Model:** a "DT-Transformer" on frozen CLIP ViT-L/14 image and text features. D(I,C) uses multi-head self-attention over both modalities; D⁻ variants are used for the unimodal tests. Repo default: 4 layers, 8 heads, dimension 1024, lr 5e-5, batch 512, 30 epochs, seed 0, run on a single RTX 3060. — [GitHub repo](https://github.com/stevejpapad/image-text-verification); [VERITE HTML v3](https://arxiv.org/html/2304.14133v3)
- **Training sets compared:** VMU-Twitter, Fakeddit, VisualNews, RSt, NC-t2t (NewsCLIPpings text-to-text), CSt, MEIR, CLIP-NESt, R-NESt (named-entity swaps), CHASMA and CHASMA-D. — [VERITE HTML v3](https://arxiv.org/html/2304.14133v3)
- **Best multiclass result:** "CLIP-NESt + Misalign [CHASMA] + NC-t2t achieves the highest overall multiclass accuracy of 52.1%, representing a 9.22% improvement over CLIP-NESt + NC-t2t". This is the source of the headline "9.2%" figure. CHASMA + NC-t2t reaches 48.7% and CHASMA-D + NC-t2t 49.0%, versus 47.7% for both CLIP-NESt + NC-t2t and R-NESt + NC-t2t. — [VERITE HTML v3](https://arxiv.org/html/2304.14133v3)
- **No unimodal bias on VERITE:** on average, D(I,C) scores 27.94% higher than text-only D⁻(C) and 43.27% higher than image-only D⁻(I). "A model biased towards one modality can not achieve satisfactory performance on VERITE." When modality balancing was removed, patterns similar to COSMOS (unimodal bias) appeared, at a smaller scale. — [VERITE HTML v3](https://arxiv.org/html/2304.14133v3)
- In a later comparison table (DEFAME), CHASMA is listed with VERITE True/OOC 74.4, True/MC 59.3 and True/False 52.1. — [DEFAME HTML](https://arxiv.org/html/2412.10510)
- RED-DOT reports a DT-Transformer trained on R-NESt + CHASMA-D + NC-t2t at **72.7%** True vs OOC. — [RED-DOT HTML](https://arxiv.org/html/2311.09939)
- A vera.ai project blog summary (search snippet only, not read in full) says the method reached "up to 72.7% accuracy for 'out-of-context' pairs, 63.9% for 'miscaptioned images'". — [vera.ai blog](https://www.veraai.eu/posts/overcoming-unimodal-bias-in-multimodal-misinformation-detection-by-certh)

**RED-DOT (Relevant Evidence Detection Directed Transformer)**
- **Method:** a Transformer over the claim pair plus web evidence, with a "Relevant Evidence Detection" (RED) module. RED re-ranks the evidence, fuses modalities, and predicts for each evidence item whether it is relevant to supporting or refuting the claim; the relevant evidence is then used for the verdict. Variants: RED-DOT-Baseline, -DSL and -SSL. — [RED-DOT HTML](https://arxiv.org/html/2311.09939)
- **Setup:** trained on NewsCLIPpings+ and evaluated only on VERITE's OOC pairs ("True vs OOC"). — [RED-DOT IEEE PDF snippet](https://ieeexplore.ieee.org/iel8/6570650/6780646/10948326.pdf)
- **VERITE True vs OOC results:**
  - RED-DOT-SSL (CLIP ViT-L/14): **76.9 (5.4)**
  - RED-DOT-DSL (CLIP B/32): 73.9 (0.5)
  - RED-DOT-Baseline without evidence (L/14): 75.5 (3.7)
  - RED-DOT-Baseline with evidence (L/14): 70.6 (3.1)
  - DT (L/14) without evidence: 57.5
  — [RED-DOT HTML](https://arxiv.org/html/2311.09939)
- **NewsCLIPpings+ results:** RED-DOT (L/14) 90.3, versus CCN 84.7, SEN 87.1 and ECENet 87.7. — [RED-DOT HTML](https://arxiv.org/html/2311.09939)
- **Claimed improvement:** "significant improvements over the state-of-the-art on the VERITE benchmark by up to 33.7%". This is relative to the DT baseline without evidence. — [RED-DOT HTML](https://arxiv.org/html/2311.09939)
- **Leaked evidence:** RED-DOT notes that FACTIFY2's "Refute" class uses fact-check articles ("leaked evidence"), which lets simple baselines reach 99–100% on that class. — [RED-DOT HTML](https://arxiv.org/html/2311.09939)

**MUSE / AITR (WACV 2025)**: see the next section for details.
- **VERITE True vs OOC results:**
  - AITR (L/14): **82.7 (6.1)**
  - MUSE-MLP (L/14): 80.6 (4.1)
  - AITR (B/32): 76.5 (2.4)
  - MUSE-MLP (B/32): 74.8 (3.8)
  - RED-DOT (L/14): 76.9 (5.4)
- **NewsCLIPpings results:** AITR 93.3, MUSE-MLP 90.0, RED-DOT 90.3, SNIFFER 88.4, ECENet 87.7, SEN 87.1, CCN 84.7. SNIFFER, ECENet, SEN and CCN report no VERITE numbers ("–").
- **Per-class on VERITE:**
  - True vs MC: MUSE-MLP 51.18, AITR 51.78.
  - Recall on the MC class: 12.43 (MUSE-MLP) and 13.02 (AITR).
  — [Similarity over Factuality HTML](https://arxiv.org/html/2407.13488v1)

**LAMAR + "MisCaption This!"**
- **Method:** "MisCaption This!" generates synthetic miscaptioned training data by adversarial prompting of open VLMs (LLaVA-1.6-Mistral, Molmo-7B-D, Llama-3.2-11B-Vision). LAMAR is a Transformer trained to reconstruct the embedding of the truthful caption, used as an auxiliary signal for detection. — [LAMAR HTML](https://arxiv.org/html/2504.06010)
- **VERITE True vs MC:**
  - LAMAR [E2E, Gate]: **66.0**
  - Other detectors: RED-DOT 63.3, AITR 62.9, DT-Transformer 61.8, MUSE 57.7
  - Same LAMAR trained on CHASMA: 59.8; trained on NESt: 61.2
  - Zero-shot LVLMs: LLaVA 1.6 66.9, Molmo 63.0, Llama 3.2 59.3
  — [LAMAR HTML](https://arxiv.org/html/2504.06010)
- **VERITE True vs OOC:** LAMAR 76.3; AITR 74.1; RED-DOT 73.5; MUSE 72.0. These differ from the WACV numbers because the training data differ. — [LAMAR HTML](https://arxiv.org/html/2504.06010)
- **VERITE multiclass:** LAMAR 54.9 (Molmo-generated data), 54.2 (LLaVA data), 53.2 (CHASMA). AITR 51.4–52.0; RED-DOT 48.5–49.1. — [LAMAR HTML](https://arxiv.org/html/2504.06010)
- **VERITE 24/25:** LAMAR 70.0 (Attention) and 68.0 (Gate), versus zero-shot LLaVA 1.6 62.0, Molmo 58.0 and Llama 3.2 53.1. — [LAMAR HTML](https://arxiv.org/html/2504.06010)

**DEFAME (zero-shot MLLM agent with web tools)**
- **Method:** a six-stage zero-shot MLLM pipeline with web search, image search, reverse image search (Google Vision) and geolocation (GeoCLIP). Backbones: GPT-4o, GPT-4o mini, LLaVA-OneVision 7B and Llama 4 Scout. It excludes fact-check domains (snopes.com, politifact.com, factcheck.org, fullfact.org, reuters.com/fact-check, factcheck.afp.com, etc.) and restricts results to sources published before the claim date, to prevent leakage. — [DEFAME HTML](https://arxiv.org/html/2412.10510)
- **VERITE results (accuracy):**

  | Method | True vs OOC | True vs MC | True vs False |
  |---|---|---|---|
  | DEFAME | 78.4 | 83.3 | 83.9 |
  | GPT-4o CoT | 74.1 ± 0.9 | 76.5 ± 0.4 | 80.0 ± 0.6 |
  | GPT-4o | 70.4 ± 1.4 | 72.0 ± 0.8 | 78.7 ± 0.8 |
  | AITR | 82.7 | 51.8 | 58.0 |

  — [DEFAME HTML](https://arxiv.org/html/2412.10510)
- The extracted text describes the VERITE sample count as "1,001 samples (after removing 13 incomplete instances)". This does not match VERITE's 1,000 pairs and is probably an extraction or wording error; check it in the PDF. — [DEFAME HTML](https://arxiv.org/html/2412.10510)

**MAD-Sherlock (multi-agent debate, GPT-4o, external retrieval, no fine-tuning)**
- **Claim:** "Evaluated on NewsCLIPpings, VERITE, and MMFakeBench, it outperforms prior methods by 2%, 3%, and 5%, respectively." — [MAD-Sherlock abstract](https://arxiv.org/abs/2410.20140)
- **VERITE use:** all 1,000 samples (338 true, 324 OOC, 338 MC). The headline number is on the combined misinformation set: MAD-Sherlock **85.2** ("VERITE all"), versus CRAVE 82.0. — [MAD-Sherlock HTML](https://arxiv.org/html/2410.20140)
- **Ablation rows:** with external retrieval, 85.2 (all) / 79.5 (True vs OOC) / 80.7 (True vs MC); without retrieval, 84.8 / 78.6 / 82.6. The extracted text labels these rows "GPT-4o", so it is unclear whether they are plain GPT-4o or MAD-Sherlock configurations; check the appendix table. — [MAD-Sherlock HTML](https://arxiv.org/html/2410.20140)
- **NewsCLIPpings:** MAD-Sherlock 90.8, versus SNIFFER 88.4 (84.5 without fine-tuning), GPT-4o with retrieval 86.0, and GPT-4o 70.7. — [MAD-Sherlock HTML](https://arxiv.org/html/2410.20140)

**Other papers reporting VERITE results (abstract-level only; numbers not extracted)**
- DPOD (Bhattacharya et al., arXiv 2311.16496) claims state-of-the-art on NewsCLIPpings and VERITE. — [arXiv 2311.16496](https://arxiv.org/abs/2311.16496)
- LADLE-MM (Cardullo, Teglia, Amerini, CVIU 2026, arXiv 2512.20257) claims it outperforms LVLM-based state of the art on VERITE. — [arXiv 2512.20257](https://arxiv.org/abs/2512.20257)
- E2LVLM (Wu et al., arXiv 2502.10455) is an evidence-enhanced LVLM for OOC detection. It cites VERITE; whether it reports VERITE numbers was not checked. — [arXiv 2502.10455](https://arxiv.org/abs/2502.10455)

### Inferences
- Across all papers, the **miscaptioned class is the hard case** for CLIP-similarity detectors (about 51–66% binary; MC-class recall about 13% for AITR and MUSE). Only systems that retrieve and *read* external evidence (DEFAME, MAD-Sherlock) get past about 80% on True vs MC. This supports TwinCheck's argument that evidence must actually be read; it matches the `_B` twin test.
- Results are **not directly comparable** across papers. Training data differ (NewsCLIPpings vs CHASMA vs VLM-generated), protocols differ (OOD-CV folds vs full set), class subsets differ (OOC only vs all), and the evidence differs (RED-DOT's Google evidence with leakage vs DEFAME's filtered live search). A TwinCheck report should present numbers with their protocol attached.
- GPT-4o-based results on VERITE may be affected by memorisation, since VERITE fact-checks date from 2001–2023, before the GPT-4o cutoff. DEFAME and VeriTaS explicitly raise this "data leakage into pretraining" concern. This motivates VERITE 24/25 and ClaimReview2024+. — [DEFAME abstract](https://arxiv.org/abs/2412.10510); [VeriTaS, arXiv 2601.08611](https://arxiv.org/abs/2601.08611)

### Gaps
- SNIFFER (Qi et al., CVPR 2024) appears to report no VERITE numbers: it is listed as "–" for VERITE in the WACV 2025 table. I found no paper reporting SNIFFER on VERITE.
- Exact VERITE numbers for DPOD, LADLE-MM, E2LVLM, CRAVE and TRUST-VL were not extracted (tool budget).
- DEFAME's venue (ICML 2025) is inferred from page metadata, not from a proceedings page. RED-DOT's final journal name needs confirmation.

---

## What does the WACV 2025 paper show about shortcuts (similarity vs factuality) and which baselines expose it?

### Takeaway
A deliberately simple baseline, **MUSE**, matches or beats complex evidence-based state-of-the-art models on NewsCLIPpings and VERITE OOC. MUSE is just six CLIP cosine similarities between the image–caption pair and the retrieved image/text evidence, fed to a Decision Tree, Random Forest or MLP. The single strongest feature is the image–caption similarity itself, which uses no evidence at all. This means current models detect *mismatch*, not *falsehood*. They fail completely on miscaptioned pairs, where image and text are related but a key fact is wrong.

### Cited Findings
- **MUSE (MUltimodal SimilaritiEs)** is six CLIP similarities:
  - claim image ↔ claim text, s(I^v,T^v)
  - claim image ↔ evidence images, s(I^v,I^e)
  - claim text ↔ evidence images, s(T^v,I^e)
  - claim image ↔ evidence text, s(I^v,T^e)
  - claim text ↔ evidence text, s(T^v,T^e)
  - evidence image ↔ evidence text, s(I^e,T^e)
  They are fed to a Decision Tree, Random Forest or MLP. — [Similarity over Factuality HTML](https://arxiv.org/html/2407.13488v1)
- **Main claim:** "MUSE, when used with conventional classifiers like Decision Tree, Random Forest, and Multilayer Perceptron, can compete with and even surpass the state-of-the-art on the NewsCLIPpings and VERITE datasets." Adding MUSE to their AITR Transformer improved it "by 3.3% and 7.5% on NewsCLIPpings and VERITE". — [WACV 2025 abstract](https://arxiv.org/abs/2407.13488)
- **AITR ("Attentive Intermediate Transformer Representations"):** a stack of Transformer encoder layers with varying head counts (e.g. [1,2,4,8] or [8,4,2,1]). Attention pooling is applied over each layer's CLS token, as an analogy for fact-checkers making repeated passes over the evidence. — [Similarity over Factuality HTML](https://arxiv.org/html/2407.13488v1)
- **AITR ablation with MUSE:** without MUSE, 88.28 (NewsCLIPpings) and 69.98 (VERITE). With MUSE and attention pooling, 93.31 and 81.00. — [Similarity over Factuality HTML](https://arxiv.org/html/2407.13488v1)
- **Shortcut evidence 1, no evidence needed:** MUSE-MLP using only the image–caption similarity s(I^v,T^v) reaches **80.69% on NewsCLIPpings and 70.89% on VERITE**. All six features give 89.96 / 80.54. — [Similarity over Factuality HTML](https://arxiv.org/html/2407.13488v1)
- **Shortcut evidence 2, feature importance:** s(I^v,T^v) gets the largest weight: 0.3424 in the Random Forest and 0.5918 in the Decision Tree. Next is image↔evidence-image (0.2426 RF, 0.2149 DT). MUSE-RF reaches 90.16 / 79.93 and MUSE-DT 88.44 / 77.37 (NewsCLIPpings / VERITE). — [Similarity over Factuality HTML](https://arxiv.org/html/2407.13488v1)
- **Shortcut evidence 3, retrieval artefacts:** NewsCLIPpings sources (USA Today, Washington Post, BBC, Guardian) are popular, so search engines often return the same or near-identical image and text. Truthful pairs therefore get high image↔evidence-image (0.907) and text↔evidence-text (0.597) similarity, while OOC images retrieve unrelated text. — [Similarity over Factuality HTML](https://arxiv.org/html/2407.13488v1)
- **Shortcut evidence 4, tiny data suffices:** MUSE-RF keeps high performance when trained on 1% of the training set (710 samples). Even with 0.1% and 0.05% (71 and 36 samples), performance "does not completely deteriorate". — [Similarity over Factuality HTML](https://arxiv.org/html/2407.13488v1)
- **Failure on factuality, which exposes the shortcut:** on VERITE True vs MC, MUSE-MLP gets 51.18% and AITR 51.78%. MC-class recall is 12.43% and 13.02%. Miscaptioned pairs are highly related in image and text but misrepresent a person, date or location, and "CLIP similarities do not capture" this. — [Similarity over Factuality HTML](https://arxiv.org/html/2407.13488v1)
- **Recommendations to the field:**
  1. Do not train and evaluate only on algorithmically created OOC datasets.
  2. Use annotated evaluation benchmarks.
  3. Broaden "OOC" to include miscaptioned images, named-entity manipulations and other de-contextualisation.
  4. Expand training sets (Fakeddit, MEIR, TamperedNews, CHASMA) with external evidence.
  5. Handle "leaked evidence".
  6. Design tests showing that models rely on factuality and logical inconsistency rather than shortcuts.
  — [Similarity over Factuality HTML](https://arxiv.org/html/2407.13488v1)
- **Code:** https://github.com/stevejpapad/outcontext-misinfo-progress — [WACV 2025 abstract](https://arxiv.org/abs/2407.13488)
- **Data:** VERITE (1,000 annotated samples: 338 truthful, 338 miscaptioned, 324 OOC) with the external evidence from RED-DOT. — [WACV 2025 CVF PDF snippet](https://openaccess.thecvf.com/content/WACV2025/papers/Papadopoulos_Similarity_over_Factuality_Are_we_Making_Progress_on_Multimodal_Out-of-Context_WACV_2025_paper.pdf); [HTML](https://arxiv.org/html/2407.13488v1)

### Inferences
- MUSE is the model for TwinCheck's "similarity + LightGBM (MUSE-style)" baseline (Feba F4). The text version would use claim↔snippet and snippet↔snippet similarities. TwinCheck's `_B` twins play the role of VERITE's MC class: high similarity but a wrong entity. So a MUSE-style baseline should score a *low* ESR, which would be a clean replication of the WACV finding on text and in Arabic.
- Recommendation 6 ("evaluation tests … that demonstrate models' reliance on factuality") is essentially what TwinCheck's twin metrics (ESR, unrelated rate, consistency) operationalise. TwinCheck can position itself as responding to it, with "to our knowledge" wording.
- The WACV paper is about multimodal OOC (image + caption + evidence). CLAUDE.md §2 cites it for text-claim shortcuts (word overlap, plausibility). The report should describe it accurately: the shortcut is *crossmodal and evidence similarity*, and the text analogue is an extrapolation.

### Gaps
- The WACV paper does not test text-only fact-checking (MultiFC, AVeriTeC) or non-English data. No Arabic or bilingual results exist in this line of work.
- Bootstrap CIs and significance tests are not reported; VERITE results come with standard deviation over folds only.

---

## Does any work study AI-generated or manipulated images on VERITE, or the effect of image authenticity on verdicts?

### Takeaway
I found **no published paper that applies AI-generation or VAE-style perturbations to VERITE images and measures whether verdicts change.** VERITE was designed to *exclude* asymmetric cases such as standalone manipulated images. Closely related 2025–2026 work shows that (a) adding deepfake-detector outputs to fact-checking pipelines *hurts* F1, because of "non-causal authenticity assumptions" (on MMFakeBench and DGM4, not VERITE), and (b) on VERITE, image swaps and plausible images cause harmful "false reassurance" errors (TrustLift). Both support TwinCheck's decoupling rule and the novelty of the `_V` / ALR test, with "to our knowledge" wording.

### Cited Findings
- **VERITE design:** VERITE excludes "asymmetric multimodal misinformation", where one modality alone (for example a manipulated image or a false text claim) carries the misinformation. Its images are therefore meant to be authentic, and the misinformation is in the image–text relation. — [VERITE abstract](https://arxiv.org/abs/2304.14133); [VERITE HTML v3](https://arxiv.org/html/2304.14133v3)
- **"Fact or Fake?" (Sagar et al., arXiv 2602.01854, 2026):**
  - It describes itself as "the first systematic analysis of deepfake detectors in the context of multimodal misinformation detection", on MMFakeBench and DGM4.
  - Image-only deepfake detectors reach F1 0.26–0.53 (MMFakeBench) and 0.33–0.49 (DGM4).
  - "Incorporating their predictions into fact-checking pipelines consistently reduces performance by 0.04–0.08 F1 due to non-causal authenticity assumptions".
  - An evidence-centric system (MCTS retrieval + multi-agent debate) reaches F1 of about 0.81 on MMFakeBench and 0.55 on DGM4.
  — [arXiv 2602.01854](https://arxiv.org/abs/2602.01854)
- **TrustLift (Borodulin, Masich, Tynchenko, Nelyub, Malashin; ACM IMPACT '26 workshop, 2026):**
  - Studies "false reassurance" errors, where a plausible image makes a false claim look true. Uses VERITE and AVerImaTeC with random and "semantically hard" image replacements.
  - On VERITE 3-way, a risk gate raises harmful-error AUROC from 0.628 (confidence-based) to 0.930 and cuts false reassurance at 90% coverage from 0.181 to 0.134.
  - Hard counterfactual augmentation raises hard-swap detection from 0.402 to 0.737 (0.752 with random + hard combined).
  — [ACM DL 10.1145/3841453.3841505](https://dl.acm.org/doi/10.1145/3841453.3841505)
- **DEFAME** includes reverse image search and geolocation tools for VERITE, but the extracted text mentions no AI-image or manipulation detector (TruFor is not mentioned). — [DEFAME HTML](https://arxiv.org/html/2412.10510)
- **GenAI-polluted evidence:** polluted web evidence "can degrade the performance of state-of-the-art detectors by more than 9 percentage points". This concerns the *evidence* being AI-generated, not the claim image. The paper proposes cross-modal evidence reranking and claim–evidence reasoning to mitigate it. — [Yan et al., arXiv 2501.14728](https://arxiv.org/abs/2501.14728)
- **DriftBench:** GenAI-driven stylistic diversity of news lowers LVLM-based detectors by an average F1 of −14.8%, with worse failures under adversarial evidence injection. — [Li et al., arXiv 2508.12711](https://arxiv.org/abs/2508.12711)
- **Same group, newer work on synthetic media:**
  - S-HArM: intent-aware classification of AI-generated images as humour, art or misinformation; 9,576 pairs. — [arXiv 2508.20670](https://arxiv.org/abs/2508.20670)
  - CONVEX: 150K Community Notes posts with miscaptioned, edited and AI-generated visuals; detectors and VLMs decline over time at telling synthetic from authentic images. — [arXiv 2604.15372](https://arxiv.org/abs/2604.15372)
- **MIC-Bench** (UKP, arXiv 2609.33441): pairs each claim with an authentic image and an AI-generated counterpart that introduces a controlled *contextual inconsistency*; 8,812 instances from 4,406 claims. This is a paired-image design, but the AI image changes the content, unlike TwinCheck's `_V`, which keeps the content. — [arXiv 2609.33441](https://arxiv.org/abs/2609.33441)
- **MiRAGeNews:** 12,500 real vs AI-generated image–caption pairs. MLLMs score below 24% F1 and humans 60% F1 at detecting AI-generated news. — [arXiv 2410.09045](https://arxiv.org/abs/2410.09045)

### Inferences
- TwinCheck's `_V` twin (an SD-VAE round-trip of a *true* image: same content, AI fingerprint) combined with the ALR metric appears untested on VERITE in the literature found. "Fact or Fake?" is the closest result: it shows that authenticity priors hurt verdicts. It supports the decoupled design, but uses different benchmarks and no content-preserving twins.
- TrustLift's image-swap counterfactuals on VERITE are the nearest "change one thing on the image side" precedent and should be cited next to the `_V` design.
- Because VERITE excludes manipulated images by construction, a `_V` twin of a VERITE true pair should keep the label "true". That makes VERITE a clean base for the ALR test.

### Gaps
- I found no paper reporting results of AI-image detectors (UnivFD, CLIP probes) on VERITE images, and none measuring verdict flips after diffusion or VAE re-encoding. Absence in a limited search (about 15 tool calls) is not proof of absence. A Semantic Scholar citation sweep of VERITE (110+ citations) and FALSDO (IJCAI 2026, cited in CLAUDE.md but not covered here) is recommended.
- The full text of "Fact or Fake?" and TrustLift (method details, which VERITE subset was used) was not read beyond the abstract and summary level.
