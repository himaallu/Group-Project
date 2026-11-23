# Prior work on AVeriTeC: the dataset, the FEVER shared tasks (2024, 2025) and later papers

Papers read in full or in large part (primary sources):
- [P1] Schlichtkrull, Guo, Vlachos. "AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web." NeurIPS 2023 Datasets & Benchmarks. https://proceedings.neurips.cc/paper_files/paper/2023/hash/cd86a30526cd1aff61d6f89f107634e4-Abstract-Datasets_and_Benchmarks.html (arXiv: https://arxiv.org/abs/2305.13117)
- [P2] Schlichtkrull, Chen, Whitehouse, Deng, Akhtar, Aly, Guo, Christodoulopoulos, Cocarascu, Mittal, Thorne, Vlachos. "The Automated Verification of Textual Claims (AVeriTeC) Shared Task." FEVER 2024 workshop (ACL Anthology 2024.fever-1.1). https://arxiv.org/abs/2410.23850
- [P3] Akhtar, Aly, Chen, Deng, Schlichtkrull, Whitehouse, Vlachos. "The 2nd Automated Verification of Textual Claims (AVeriTeC) Shared Task: Open-weights, Reproducible and Efficient Systems." FEVER 2025, pp. 201–223. https://aclanthology.org/2025.fever-1.15/
- [P4] Rothermel, Braun, Rohrbach, Rohrbach. "InFact: A Strong Baseline for Automated Fact-Checking." FEVER 2024, pp. 108–112. https://aclanthology.org/2024.fever-1.12/
- [P5] Yoon, Jung, Yoon, Park. "HerO at AVeriTeC: The Herd of Open Large Language Models for Verifying Real-World Claims." FEVER 2024, pp. 130–136. https://aclanthology.org/2024.fever-1.15/
- [P6] Ullrich, Mlynář, Drchal. "AIC CTU system at AVeriTeC: Re-framing automated fact-checking as a simple RAG task." FEVER 2024, pp. 137–150. https://aclanthology.org/2024.fever-1.16/
- [P7] Ullrich, Drchal. "AIC CTU@FEVER 8: On-premise fact checking through long context RAG." FEVER 2025, pp. 274–280. https://aclanthology.org/2025.fever-1.22/
- [P8] Yoon, Jung, Yoon, Park. "Team HUMANE at AVeriTeC 2025: HerO 2 for Efficient Fact Verification." FEVER 2025, pp. 224–228. https://aclanthology.org/2025.fever-1.16/
- [P9] Park, Lee, Kim, Park, Park. "Dunamu-ml's Submissions on AVERITEC Shared Task." FEVER 2024, pp. 71–76. https://aclanthology.org/2024.fever-1.7/
- [P10] Upravitelev et al. "Exploring Semantic Filtering Heuristics For Efficient Claim Verification" (team EFC). FEVER 2025, pp. 229–237. https://aclanthology.org/2025.fever-1.17/
- [P11] Braun, Rothermel, Rohrbach, Rohrbach. "DEFAME: Dynamic Evidence-based FAct-checking with Multimodal Experts." ICML 2025 (PMLR v267). https://proceedings.mlr.press/v267/braun25b.html (arXiv: https://arxiv.org/abs/2412.10510)
- [P12] Wu Lu, Setiawan et al. "No Hindsight for LLM Fact-Checkers: Measuring Leakage Channels in Misinformation Detection." arXiv preprint 2610.04888 (Oct 2026, not peer-reviewed). https://arxiv.org/abs/2610.04888
- [P13] UHH team. "UHH at AVeriTeC: RAG for Fact-Checking with Real-World Claims." FEVER 2024, pp. 55–63. https://aclanthology.org/2024.fever-1.5/

## Q1. What does AVeriTeC contain, and how is it evaluated?

### Takeaway
AVeriTeC has 4,568 real-world English claims from 50 fact-checking organisations. Each claim comes with metadata, question–answer (QA) evidence backed by web URLs from before the claim date, one of 4 verdict labels, and a justification. The splits are temporal (train 3,068 / dev 500 / test 1,000), and the shared tasks added hidden test sets of 1,215 claims (2024) and 1,000 claims (2025). Evidence is scored with Hungarian METEOR over QA pairs (Ev2R recall in 2025). The headline "AVeriTeC score" counts a verdict as correct only when the evidence score passes a threshold: λ = 0.25 for METEOR, t = 0.5 for Ev2R recall.

### Cited Findings
**Data construction**
- 4,568 real-world claims from fact-checks by 50 organisations, collected with the Google FactCheck Claim Search API (ClaimReview). — [P1](https://arxiv.org/abs/2305.13117)
- The team started from 8,000 fact-checking articles and discarded 537 duplicates and 802 paywalled or dead articles. The rest went through a 5-phase pipeline:
  1. Claim extraction and normalisation: decontextualise the claim and add metadata.
  2. QA generation from the web, plus a provisional verdict.
  3. Evidence-sufficiency check: a *different* annotator sees only the QA pairs and gives a verdict and justification.
  4–5. If the two verdicts disagree, phases 2–3 are repeated with new annotators, and the claim is discarded if they still disagree.
  No annotator saw the same claim twice. The annotation was done by Appen. — [P1](https://arxiv.org/abs/2305.13117)
- Annotators could not use the fact-checking article itself as evidence. A custom Google search bar showed only documents published before the claim date. The authors call the protection against temporal leakage "approximate": about 6% of dev answers come from a fact-checking domain, mostly earlier articles by a different fact-checker. — [P1](https://arxiv.org/abs/2305.13117)
- Speculative claims (about the future or opinions) and multimodal claims were discarded, so every claim is textual. — [P1](https://arxiv.org/abs/2305.13117); [P2](https://arxiv.org/abs/2410.23850)

**Evidence format**
- Evidence is a set of QA pairs. A question may have several answers, and later questions may refer to earlier ones (multi-hop). Every answer other than "No answer could be found." must link to a source URL, and pages are cached in the Internet Archive. — [P1](https://arxiv.org/abs/2305.13117)
- On average a claim has 2.60 questions and a question has 1.07 answers. Answers are 53% extractive, 26% abstractive, 17% boolean, and 4% of questions are unanswerable. — [P1](https://arxiv.org/abs/2305.13117)
- Metadata includes speaker, publisher, claim date and location. Claim type and fact-checking strategy are also annotated, but are meant for analysis only, not as model input. — [P1](https://arxiv.org/abs/2305.13117)

**Labels**
- There are 4 labels: Supported, Refuted, Not Enough Evidence (NEE), and Conflicting Evidence/Cherrypicking. The 4th class covers both genuinely conflicting sources and technically true claims that mislead by leaving out context. — [P1](https://arxiv.org/abs/2305.13117)
- Splits are temporal, with label percentages (S/R/C/N):

  | Split | Claims | Ends | S/R/C/N (%) |
  |---|---|---|---|
  | Train | 3,068 | 25-08-2020 | 27.6/56.8/6.4/9.2 |
  | Dev | 500 | 31-10-2020 | 24.4/61.0/7.6/7.0 |
  | Test | 1,000 | 22-12-2021 | 25.5/62.0/6.3/6.2 |

  — [P1](https://arxiv.org/abs/2305.13117)
- In counts, the dev set has 305 Refuted, 122 Supported, 35 NEI and 38 C/CP. — [P4](https://aclanthology.org/2024.fever-1.12/)
- Refuted dominates because journalists choose false or misleading claims. The dataset includes every ClaimReview claim labelled supported (or a variant such as "true") within the date limits. — [P1](https://arxiv.org/abs/2305.13117)
- Inter-annotator agreement was measured on 100 re-annotated claims: Randolph's free-marginal κ = 0.619 and Fleiss' κ = 0.503. — [P1](https://arxiv.org/abs/2305.13117)
- The shared-task test sets are hidden:

  | Test set | Claims | Ends | S/R/C/N (%) | Questions/claim |
  |---|---|---|---|---|
  | 2024 new | 1,215 | 13-08-2023 | 17.3/66.5/4.1/12.1 | 2.89 |
  | 2025 | 1,000 | 19-12-2024 (covers Jan–Dec 2024) | 22.2/71.9/1.7/4.2 | 2.79 |

  — [P2](https://arxiv.org/abs/2410.23850); [P3](https://aclanthology.org/2025.fever-1.15/)
- Licence: dataset and baseline are released under CC-BY-NC-4.0 at https://fever.ai/dataset/averitec.html. — [P2](https://arxiv.org/abs/2410.23850)

**Knowledge store (added for the shared task)**
- For each claim there is a set of documents found by Google search with queries generated by gpt-3.5-turbo-0125 from the claim and the *gold* questions and answers. It also contains distractor queries (changed entities, dates and events) and the annotator-chosen gold documents, deduplicated and shuffled. Text was scraped with trafilatura.
- A claim has on average 955 documents of about 6,095 tokens each.
- The most common domains are NCBI, Wikipedia, Quora, NYT and CNN.

  — [P2](https://arxiv.org/abs/2410.23850)
- The 2025 knowledge store has 1,018,800 URLs. A claim has on average 1,019 URLs, of which 593 have valid scraped text. — [P3](https://aclanthology.org/2025.fever-1.15/)
- Known problems:
  - Scraping: in 297 of 500 dev examples, at least one gold document was not scraped correctly (finding of team Papelo).
  - Date leakage in 2024: a date-format bug let post-claim pages, including fact-checking articles, into the store for the 1,215 new test claims. The organisers re-released a corrected store.

  — [P2](https://arxiv.org/abs/2410.23850)

**Evaluation**
- Hungarian METEOR works in 3 steps:
  1. The Hungarian algorithm finds the best one-to-one matching between the generated and reference sequences, using NLTK METEOR as the pairwise score f.
  2. The matched scores are summed and divided by |Y|, the number of reference pairs.
  3. With exact match as f, this reduces to FEVER evidence recall.

  The score is computed twice: on questions only (Q) and on question+answer concatenations (Q+A). It is recall-only, with no precision term, and only the first k = 10 QA pairs count. — [P1](https://arxiv.org/abs/2305.13117)
- AVeriTeC score = verdict accuracy, but a claim gets 0 unless its Q+A Hungarian METEOR is ≥ λ. The recommended λ is 0.25, chosen from human-vs-human agreement: two annotators' QA sets scored only 0.28 (Q) and 0.22 (Q+A) against each other. — [P1](https://arxiv.org/abs/2305.13117)
- In the 2024 shared task, justifications were not required, and each evidence item had to include a URL and the scraped text. — [P2](https://arxiv.org/abs/2410.23850)
- Metric gaming: because there is no brevity penalty, submitting more questions helps. Duplicating each question twice raised the baseline by 2 points of Q+A and 0.5 points of AVeriTeC score. — [P2](https://arxiv.org/abs/2410.23850)
- In the 2024 human evaluation, about 15% of evidence got coverage rating 1 ("covers none of the reference") and about 21% got rating 2. — [P2](https://arxiv.org/abs/2410.23850)
- In 2025 the organisers replaced METEOR with Ev2R, an LLM-as-judge (Akhtar et al. 2024, arXiv 2411.05375). It splits retrieved and reference evidence into atomic facts and computes precision and recall. Only recall is used. AVeriTeC score = accuracy where Ev2R recall > t = 0.5. — [P3](https://aclanthology.org/2025.fever-1.15/)
- Correlation of the automatic evidence score with human ratings:
  - 2024 (METEOR): close to zero.
  - 2025 (Ev2R): coverage ρ = .404 / r = .406, relevance ρ = .244 / r = .242.

  — [P3](https://aclanthology.org/2025.fever-1.15/)

### Inferences
- For TwinCheck, AVeriTeC labels map most naturally as Supported→`supported`, Refuted→`contradicted`, NEE→`unverified`. Conflicting/Cherrypicking matches the "misleading/mixture" group that CLAUDE.md §4 says to drop, so dropping it is consistent with the plan and the count should be reported (about 6–8% of each split, per [P1]).
- Each AVeriTeC QA answer becomes one snippet in our format (R1 task), which fits the "evidence = list of short texts" design. Because the gold QA evidence was checked for sufficiency, it is cleaner than MultiFC's Google snippets.
- The CC-BY-NC-4.0 licence allows research use. Pushing processed AVeriTeC to a public repo needs attribution and a non-commercial notice.

### Gaps
- I did not find a published per-split breakdown of how many claims have each number of QA pairs. Only the averages are reported.
- The exact share of claims with "No answer could be found" answers per label was not reported.

## Q2. What did the baseline achieve, and what did the top shared-task systems achieve?

### Takeaway
The original baseline (BLOOM-7b + BERT, Google search) scored only 0.15 AVeriTeC score on dev at λ = 0.25, and the knowledge-store baseline scored 0.11 on the 2024 test set. In 2024, GPT-4o pipelines led: InFact/TUDA_MAI won with 0.63, followed by HerO (open Llama, 0.57) and CTU AIC (0.50); 18 of 21 teams beat the baseline. In 2025, the rules required open-weight models on one 23 GB GPU in ≤ 60 s per claim, and the metric changed to Ev2R. Scores dropped: CTU AIC (Qwen3-14B) won with 0.3317 against a HerO-based baseline of 0.2023.

### Cited Findings
**Original baseline [P1]**
- The pipeline:
  1. Google Search API with a reduced claim (verbs, nouns and adjectives only), limited to pages from before the claim date; top 30 results.
  2. Query expansion with questions generated by BLOOM-7b, with 10 BM25-retrieved training claims as in-context examples.
  3. BM25 keeps the top 100 sentences, BLOOM writes a question for each, and a fine-tuned BERT-large reranks the QA pairs and keeps the top 3.
  4. BERT-large stance per QA pair (support / refute / irrelevant), then a deterministic rule:
     - both supporting and refuting evidence → Conflicting;
     - only supporting → Supported, only refuting → Refuted;
     - otherwise → NEE.
  5. BART-large writes the justification.

  Everything runs on 1 A100. — [P1](https://arxiv.org/abs/2305.13117)
- Dev results (Table 3 of P1). Veracity is shown at λ = 0.2 / 0.25 / 0.3:

  | System | Q | Q+A | Veracity @ .2/.25/.3 |
  |---|---|---|---|
  | No search | 0.19 | 0.11 | 0.03 / 0.02 / 0.01 |
  | Gold evidence | 1.00 | 1.00 | 0.49 / 0.49 / 0.49 |
  | AVeriTeC-BLOOM-7b | 0.26 | 0.21 | 0.23 / 0.15 / 0.00 |
  | gpt-3.5-turbo | 0.29 | 0.16 | 0.17 / 0.10 / 0.06 |

  — [P1](https://arxiv.org/abs/2305.13117)
- Per-class veracity F1 (S/R/C/N/macro):

  | System | S | R | C | N | Macro |
  |---|---|---|---|---|---|
  | No evidence | .30 | .22 | .00 | .16 | .17 |
  | Gold evidence | .48 | .74 | .15 | .59 | .49 |
  | AVeriTeC pipeline | .41 | .69 | .10 | .16 | .23 |
  | gpt-3.5-turbo | .62 | .71 | .02 | .20 | .39 |

  With gold evidence, BERT-large, bloom-7b1 and Vicuna-13b reached macro-F1 .49, .43 and .48. — [P1](https://arxiv.org/abs/2305.13117)
- Retrieval is the bottleneck. In 20 dev claims, Google results from the claim plus generated questions held useful evidence in only 9/20 cases, against 16/20 when the gold questions were used as queries. — [P1](https://arxiv.org/abs/2305.13117)

**2024 shared task [P2]**
- There were 21 submissions, 18 above baseline, and 15 system papers. Evaluation covered 2,214 hidden test claims: the old 1,000 plus the new 1,215. The 2024 baseline (knowledge store, BM25 top 100 sentences → BLOOM-7b1 QA generation → BERT rerank → BERT veracity on the top 3) scored Q 0.24, Q+A 0.20, AVeriTeC 0.11. — [P2](https://arxiv.org/abs/2410.23850)
- Leaderboard (Q / Q+A / AVeriTeC @ .25) and main components:

  | Rank | Team | Q | Q+A | AVeriTeC | Evidence source | QG | Retrieval | Veracity |
  |---|---|---|---|---|---|---|---|---|
  | 1 | TUDA_MAI (InFact) | 0.45 | 0.34 | 0.63 | KS | GPT-4o | gte-base-en-v1.5 | GPT-4o |
  | 2 | HUMANE (HerO) | 0.48 | 0.35 | 0.57 | KS | Llama-3-8b | BM25 + SFR-embedding-2 | Llama-3.1-70b |
  | 3 | CTU AIC | 0.46 | 0.32 | 0.50 | KS | GPT-4o | mxbai-large-v1 | GPT-4o |
  | 4 | Dunamu-ml | 0.49 | 0.35 | 0.50 | KS | GPT-4 | BM25 | GPT-4 |
  | 5 | Papelo | 0.44 | 0.30 | 0.48 | Google | T5-large + GPT-4o | — | GPT-4o |
  | 6 | UHH | 0.48 | 0.32 | 0.45 | KS | GPT-4o-mini | BM25 + gte | Mixtral-8x7B |
  | 7 | SynApSe | 0.41 | 0.30 | 0.42 | Google | GPT-4o | — | — |
  | 8 | arioriAveri | 0.38 | 0.29 | 0.39 | — | — | — | — |
  | 9 | Data-Wizards | 0.35 | 0.27 | 0.33 | — | Phi-3-medium | — | Mixtral-8x22B |
  | 19 | Baseline | 0.24 | 0.20 | 0.11 | KS | BLOOM-7b1 | BM25 | BERT |

  KS = organisers' knowledge store; — = not recorded in these notes. — [P2](https://arxiv.org/abs/2410.23850)
- Dunamu-ml had the best retrieval overall, using a custom scraper for PDFs and YouTube transcripts. — [P2](https://arxiv.org/abs/2410.23850)
- InFact's exact test score is 62.6%, and its dev accuracy is 72.4% with GPT-4o. GPT-4o costs about $0.46 per claim, GPT-4o-mini about $0.01, and Llama 3 (70B) was about 8× slower. — [P4](https://aclanthology.org/2024.fever-1.12/)
- HerO dev ablation of veracity prediction (accuracy / AVeriTeC score):

  | Method | Model | Accuracy | AVeriTeC |
  |---|---|---|---|
  | In-context | Llama-3-70b | 0.628 | 0.494 |
  | In-context | Llama-3.1-70b | 0.54 | 0.422 |
  | In-context | GPT-4o-mini | 0.488 | 0.382 |
  | LoRA | Llama-3-70b | 0.724 | 0.556 |
  | LoRA | Llama-3.1-70b | 0.704 | 0.55 |
  | Full fine-tune | Llama-3-70b | 0.746 | 0.57 |
  | Full fine-tune | Llama-3.1-70b | 0.752 | 0.578 |

  HyDE-FC query expansion raised the dev answer score from 0.26 (claim only, BM25 + SFR) to 0.2801. Adding the claim to the question-generation prompt raised Llama-3-8b's Q score from 0.4210 to 0.4938. Inference on 500 dev claims took about 6.6 h on 2 H100s, 6 h of it for retrieval. — [P5](https://aclanthology.org/2024.fever-1.15/)
- CTU AIC (3rd, 50.4%):
  - One LLM call produces QA evidence and Likert-scale (1–5) confidence for each of the 4 labels, turned into probabilities with a softmax.
  - With retrieval fixed, GPT-4o reached 72% dev accuracy.
  - A fine-tuned DeBERTa-v3-large 4-way classifier on gold evidence reached 0.71 dev macro-F1, but adding it to the LLM did not improve the full pipeline.

  — [P6](https://aclanthology.org/2024.fever-1.16/)
- Dunamu-ml reports a gain of 0.33 over the baseline in AVeriTeC score. — [P9](https://aclanthology.org/2024.fever-1.7/)

**2025 shared task [P3]**
- Rules: open-weight models only at inference; one AWS g5.2xlarge (A10G 23 GB, 32 GB RAM); ≤ 1 min per claim on average (16 h 40 min for 1,000 claims); runs were done by the organisers on their VM; no external API calls. There were 8 submissions, 7 reproducible, and 6 above the baseline. — [P3](https://aclanthology.org/2025.fever-1.15/)
- Leaderboard. Scores use Ev2R recall; "time" is seconds per claim:

  | Rank | Team | Time | Q | Q+A | AVeriTeC | Main models |
  |---|---|---|---|---|---|---|
  | 1 | CTU AIC | 53.67 | 0.2003 | 0.4774 | 0.3317 | Qwen3-14B via Ollama, not quantised; mxbai-embed-large-v1 dense index |
  | 2 | HUMANE (HerO 2) | 29.19 | 0.1933 | 0.4299 | 0.2707 | Qwen3-8B QG/QA; fully fine-tuned Qwen3-32B with AWQ for veracity |
  | 3 | Yellow Flash | 31.71 | — | — | 0.2527 | Qwen2.5-7B QG, Phi-4-14B veracity, GPTQ-int4 |
  | 4 | FZIGOT | 18.50 | 0.3622 (best Q) | — | 0.2440 | Qwen2.5-14B + LoRA, Graph-of-Thoughts iterative QG |
  | 5 | EFC | 7.01 (fastest) | — | — | 0.2047 | Phi-4-14B GGUF/llama.cpp, semantic filtering |
  | 6 | Checkmate | — | — | — | 0.2043 | — |
  | 7 | Baseline (HerO with Llama-3.1-8B) | 33.88 | — | — | 0.2023 | Llama-3.1-8B |
  | 8 | OldJoe | — | — | — | 0.1517 | Qwen3-14B |

  — [P3](https://aclanthology.org/2025.fever-1.15/)
- Swapping GPT-4o or GPT-4o-mini into the CTU pipeline *lowered* the AVeriTeC score (0.2690 and 0.3176, against 0.3317 for open Qwen3-14B), although the Q score rose. — [P3](https://aclanthology.org/2025.fever-1.15/)
- CTU 2025 uses chunks of up to 2,048 characters, retrieves k = 40, reranks with MMR to l = 10 (λ = 0.75), and passes up to about 60K characters of sources to the LLM. The authors credit the long context. They also note that Qwen3 may have seen 2024 claims during training. — [P7](https://aclanthology.org/2025.fever-1.22/)
- HerO 2 adds document summarisation, answer reformulation, post-training quantisation of the veracity model and newer backbones. It was 2nd, with the shortest runtime of the top 3. — [P8](https://aclanthology.org/2025.fever-1.16/)

**Later results on AVeriTeC dev (accuracy only, no evidence threshold)**
- DEFAME (ICML 2025), a zero-shot GPT-4o pipeline over the knowledge base: 70.5 ± 0.6 accuracy, against plain GPT-4o at 62.1 ± 0.4, GPT-4o CoT at 61.0 ± 0.4, and a prior DeBERTa SOTA (Cao et al., 2023) at 65.6. Its InFact adaptation is the system that won the 2024 task. — [P11](https://arxiv.org/abs/2412.10510)

### Inferences
- Under the 2024 metric, the jump from 0.11 to 0.63 came mostly from better retrieval (LLM query/question generation plus dense embeddings) and from very large LLMs for the verdict. The 2025 restrictions roughly halved scores, but part of that drop comes from the stricter Ev2R threshold (t = 0.5), so 2024 and 2025 numbers are not directly comparable.
- HerO's ablation is a useful reference for a small-compute project. Fine-tuning the verdict model added about 0.08 AVeriTeC score over in-context learning with the same 70B model.

### Gaps
- I did not record per-team label accuracy (without the evidence threshold) for the test sets. The overview papers give AVeriTeC score and per-label/per-type AVeriTeC scores, not raw accuracy.
- I did not read the full Papelo, Dunamu-ml or UHH papers for their dev numbers. Their test numbers above come from the overview.

## Q3. What methodological patterns do the winning systems use?

### Takeaway
The winning recipe is retrieve-then-LLM:
- generate questions, queries or hypothetical documents with an LLM to expand the claim;
- retrieve from the knowledge store with dense embeddings (often gte/Stella/mxbai), sometimes after BM25 coarse filtering;
- generate answers grounded in retrieved text;
- predict the 4-way label with a large LLM, via in-context learning (with BM25-retrieved training examples) or fine-tuning.

In 2025, the open-weight constraint pushed teams to 7–32B models with quantisation (AWQ, GPTQ, GGUF), vLLM, and precomputed dense indices.

### Cited Findings
- Generating questions, not only searching with the claim, was "essential for good retrieval performance" according to many top teams. Small models (Llama-3-8b, GPT-4o-mini, Phi-3-medium, T5) can be competitive at question generation. — [P2](https://arxiv.org/abs/2410.23850)
- Multi-hop or iterative retrieval helped Papelo, SynApSe and IKR3. Papelo used T5 for the first question and GPT-4o for the following ones. — [P2](https://arxiv.org/abs/2410.23850)
- Hybrid retrieval (BM25 coarse, then dense reranking) was used by HUMANE, UHH and SK_DU. gte-family embedders (including Stella) were popular because they led MTEB at the time. — [P2](https://arxiv.org/abs/2410.23850)
- Top systems used very large models for veracity (GPT-4o, Llama-3.1-70b, Mixtral-8x7b). The overview says reasoning ability seems most critical at this stage, more than for question generation. HUMANE and SynApSe report that fine-tuning (full or LoRA) was necessary for good veracity prediction. — [P2](https://arxiv.org/abs/2410.23850)
- InFact runs 6 GPT-4o prompts:
  1. interpret the claim with its metadata;
  2. pose 10 questions;
  3. generate search queries per question and retrieve 5 kNN results per query with gte-base-en-v1.5;
  4. answer each question from results in order, or return NONE;
  5. give a verdict;
  6. write a justification.

  The authors say this semantic search was "qualitatively superior" to BM25. — [P4](https://aclanthology.org/2024.fever-1.12/)
- HerO's HyDE-FC: Llama-3.1-70b writes N = 8 hypothetical fact-checking passages. These are concatenated with the claim for BM25 (top 10,000 sentences), then reranked by SFR-embedding-2 using the averaged embedding of the claim and passages (top 10). — [P5](https://aclanthology.org/2024.fever-1.15/)
- CTU AIC treats the task as plain RAG:
  - chunks with surrounding context, a FAISS index of mxbai embeddings, and MMR diversity reranking;
  - one LLM call returns QA pairs and Likert confidences per label;
  - a separate NLI classifier stage was dropped as unnecessary.

  — [P6](https://aclanthology.org/2024.fever-1.16/); [P7](https://aclanthology.org/2025.fever-1.22/)
- Evidence grounding was checked with a DeBERTa-v3-large NLI model (MoritzLaurer mnli-fever-anli-ling-wanli), answer as hypothesis and document as premise. It labelled about 80% of answers neutral, because documents are long and out of distribution. Manual checks of 20 neutral or contradiction cases each found answers entailed for HUMANE 19/20, CTU AIC 17/20 and Dunamu-ml 12/20. All TUDA_MAI answers were extractive, so all were entailed. — [P2](https://arxiv.org/abs/2410.23850)
- 2025 patterns:
  - 5 of 7 teams used vLLM.
  - Quantisation: AWQ (HUMANE, FZIGOT, OldJoe), GPTQ/OPTQ (Yellow Flash, Checkmate), GGUF (EFC).
  - Most teams did not fine-tune. The exceptions are HUMANE (full fine-tune of Qwen3-32B) and FZIGOT (LoRA on Qwen2.5-14B with weakly supervised data).
  - CTU AIC added few-shot examples selected by BM25 from the training set.

  — [P3](https://aclanthology.org/2025.fever-1.15/)
- EFC's semantic filtering predicts NEI or Conflicting from cosine-similarity statistics of the retrieved evidence alone: thresholds on relevance and on semantic variance. This avoids LLM calls and gives 7.01 s per claim. — [P10](https://aclanthology.org/2025.fever-1.17/)
- Search with the claim alone found complete evidence in 6/20 claims, generated questions in 9/20, gold questions in 16/20, and the full knowledge-store query process in 19/20. — [P3](https://aclanthology.org/2025.fever-1.15/)

### Inferences
- These patterns support TwinCheck's design: BM25 + BGE-M3 hybrid retrieval followed by reranking (our R2/R5) mirrors the HUMANE/UHH/SK_DU hybrid approach, and Qwen3-8B in 4-bit is close to the 2025 open-weight regime.
- Our NLI-per-snippet + MLP verdict head differs from the winners, who use an LLM for the verdict. Evidence for the head: CTU found a fine-tuned DeBERTa 4-way classifier competitive on gold evidence (0.71 macro-F1). The 2024 organisers' NLI-based groundedness check is a precedent for our faithfulness metric, with a caveat: zero-shot NLI on long documents defaults to "neutral", which supports keeping premises short (snippet-level).

### Gaps
- No winning system reports a twin-style or counterfactual-evidence test (changing evidence while keeping the claim fixed). I found no AVeriTeC paper that evaluates verdict sensitivity to swapped entities in evidence.

## Q4. Which classes are hard, and do models rely on claim-only cues?

### Takeaway
Conflicting/Cherrypicking and NEE are hard for every system:
- In 2025 no system scored above 0.1 AVeriTeC on either class.
- In 2024 the winner scored 0.12 (NEE) and 0.19 (CE/C), against 0.64–0.73 on Supported and Refuted.
- Several teams simply never predict these labels.

There is also strong evidence that LLMs score well on AVeriTeC *without real evidence*:
- InFact's "No Evidence" GPT-4o variant reached 71.0% dev accuracy, against 72.4% for the full system.
- Naive Llama-3-70B with no retrieval (67.0%) beat full InFact with Llama 3 (61.8%).
- An Oct 2026 preprint shows memorised outcomes and post-claim evidence inflate scores.

### Cited Findings
**Difficult classes**
- 2024 AVeriTeC score per gold label (S / R / NEE / CE/C):

  | Team | S | R | NEE | CE/C |
  |---|---|---|---|---|
  | TUDA_MAI | 0.64 | 0.73 | 0.12 | 0.19 |
  | HUMANE | 0.76 | 0.62 | 0.01 | 0.12 |
  | CTU AIC | 0.58 | 0.58 | 0.1 | 0.01 |
  | Dunamu-ml | 0.69 | 0.5 | 0.31 | 0.12 |
  | Papelo | 0.45 | 0.59 | 0.0 | 0.0 |
  | Average (all teams) | 0.36 | 0.32 | 0.06 | 0.06 |

  — [P2](https://arxiv.org/abs/2410.23850)
- 2025: no system above 0.1 on NEE or CE/C. The averages are S 0.22, R 0.28, NEE 0.03, CE/C 0.02, and CTU AIC scored NEE 0.1 and CE/C 0.06. — [P3](https://aclanthology.org/2025.fever-1.15/)
- Papelo blocked its model from predicting NEE and Conflicting because prompted LLMs over-predict these rare labels. The organisers note this was a common strategy. Because Refuted is about two-thirds of the data, systems calibrated towards Refuted gain an advantage. — [P2](https://arxiv.org/abs/2410.23850); [P3](https://aclanthology.org/2025.fever-1.15/)
- Human verdicts on system evidence (2024, rows = human, columns = system prediction): humans agreed with system "Refuted" 85.4% of the time and "Supported" 76.0%, but only 22.1% for NEE and 35.7% for CE/C. Human-NEE evidence was most often system-labelled Refuted (60.3%). — [P2](https://arxiv.org/abs/2410.23850)
- The original baseline's Conflicting F1 is .10 (pipeline) and .15 (gold evidence), with precision of about 10% on gold evidence. The rule "any evidence with a different stance → conflicting" over-fires, because context-adding QA pairs get labelled refuting. — [P1](https://arxiv.org/abs/2305.13117)
- CTU could not train a reliable binary cherrypicking detector (DeBERTa-v3-large, weighted cross-entropy). The class is under 7% of train. — [P6](https://aclanthology.org/2024.fever-1.16/)
- By claim type, the hardest were position statements in 2024, which often need abductive reasoning, and numerical claims in 2025, where the gap grew with the Ev2R metric. — [P2](https://arxiv.org/abs/2410.23850); [P3](https://aclanthology.org/2025.fever-1.15/)

**Claim-only cues, parametric knowledge and leakage**
- Original paper: a "no search" BERT-large model (claim only, answers set to "No answer could be found") gets macro-F1 .17 and near-zero AVeriTeC score, because it fails the evidence threshold. ChatGPT without retrieval gets macro-F1 .39 but only 0.10 AVeriTeC score, because its evidence is often hallucinated. Example: an invented USDA statement given as "evidence" for a true dandelion-greens claim. — [P1](https://arxiv.org/abs/2305.13117)
- Original paper: augmenting stance training with ChatGPT *paraphrases* of claims (same evidence) slightly lowered gold-evidence macro-F1, from .49 to .46, mainly by confusing Refuted and NEE. — [P1](https://arxiv.org/abs/2305.13117)
- InFact dev ablations (accuracy; columns ordered Naive / No Interpret. / No Evidence / No Q&A / No Query Gen. / InFact):

  | Backbone | Naive | No Interpret. | No Evidence | No Q&A | No Query Gen. | InFact |
  |---|---|---|---|---|---|---|
  | Llama 3 | 67.0 | 63.2 | 67.0 | 52.9 | 65.0 | 61.8 |
  | GPT-4o-mini | 36.2 | 61.6 | 56.8 | 54.8 | 59.6 | 60.2 |
  | GPT-4o | 52.6 | 71.8 | 71.0 | 68.8 | 70.2 | 72.4 |

  The authors' comments:
  - "Leaving out all evidence does not lead to a drastic decline of the METEOR scores."
  - In the No Evidence variant, GPT models "achieve a higher accuracy and predict NEI much less, while still having no access to any external information."
  - Llama 3 in the Naive setting is "self-confident" and rarely predicts NEI.
  - The LLMs were pre-trained on data into 2023, "likely covering many of AVeriTeC's claims and evidence."

  — [P4](https://aclanthology.org/2024.fever-1.12/) (table columns confirmed against a structured parse of the same PDF)
- InFact's own analysis of 20 failure cases: in more than half, the gold label was debatable or InFact gave a valid alternative fact-check. Only 6/20 errors were clearly the system's fault. — [P4](https://aclanthology.org/2024.fever-1.12/)
- DEFAME (ICML 2025): plain GPT-4o reaches 62.1% on AVeriTeC dev but only 35.2% on ClaimReview2024+, which contains claims after GPT-4o's cutoff. DEFAME with retrieval stays at 69.7%. The authors read this as temporal dependence of parametric knowledge. — [P11](https://arxiv.org/abs/2412.10510)
- The 2025 overview notes that Qwen3 (training cutoff around March 2025) "has likely seen data" from the 2025 test period, and that open-data models would help measure this temporal leakage. — [P3](https://aclanthology.org/2025.fever-1.15/)
- "No Hindsight" preprint (Oct 2026, not peer-reviewed): on 2,554 entity-bearing binary AVeriTeC claims with *no evidence*, Qwen3-14B reaches accuracy 0.710 and macro-F1 0.640, against a majority baseline (always Refuted) of 0.679 / 0.405. Masking named entities lowers accuracy by only 0.032, but Supported recall collapses from 0.421 to 0.131, which the authors interpret as parametric (memorised-outcome) leakage hidden inside aggregate accuracy. Allowing post-claim evidence inflates zero-shot accuracy by 6.3 points (0.782 point-in-time vs 0.844 post-claim), of which 3.1 points are strict hindsight. — [P12](https://arxiv.org/abs/2610.04888)

### Inferences
- AVeriTeC is a strong "clean" English test set for TwinCheck. However, plain verdict accuracy on it can be high with little evidence use (InFact No-Evidence ≈ full system with GPT-4o), so the evidence threshold, or our evidence-swap (`_B`) and unrelated-evidence (`_C`) twins, is needed to show evidence reliance. This directly motivates TwinCheck's twin audit on AVeriTeC as well as MultiFC. "To our knowledge", no AVeriTeC paper reports counterfactual evidence-swap tests.
- [P1]'s paraphrase-augmentation result (slightly lower macro-F1) is a relevant caution for our `_P` twin loss: paraphrase invariance alone may not help accuracy. TwinCheck's goal is consistency/ESR with ≤ 1 point macro-F1 loss, which fits that evidence.
- The entity-masking finding in [P12] parallels our entity-swap twins. Both treat named entities as the anchor of memorised shortcuts.
- Because NEE ≈ our `unverified` is rare in AVeriTeC (about 6–12%) and poorly handled by LLMs, macro-F1 and the unverified rate on AVeriTeC will likely be dominated by that class. It is worth reporting per-class F1.

### Gaps
- I found no published claim-only *fine-tuned classifier* (e.g. claim-only RoBERTa) accuracy on the AVeriTeC test sets, beyond the original paper's "no search" BERT (dev macro-F1 .17).
- I found no per-label accuracy for InFact's No-Evidence or Naive variants. Only overall accuracy and confusion matrices for the full system are given.
- [P12] is a very recent arXiv preprint (Oct 2026) and has not been peer-reviewed. Its numbers should be cited as preliminary.
- 2025 FEVER also ran an image–text successor task (AVerImaTeC, leaderboard at https://fever.ai/task.html). It is out of scope here and was not analysed.
