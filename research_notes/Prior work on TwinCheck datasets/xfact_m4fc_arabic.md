# Prior work on X-FACT, M4FC and Arabic fact-checking

Scope: the original X-FACT and M4FC papers, later papers that use them, and Arabic claim-verification work (AraFacts, ANS, CheckThat!). Numbers are copied exactly from the papers named. Where a number comes only from an abstract, or could not be checked, that is said.

## X-FACT: construction, labels, evidence, splits, baselines and the evidence finding

### Takeaway
X-FACT (Gupta & Srikumar, ACL 2021 short paper) has 31,189 non-English claims in 25 languages, from 85 fact-checkers before filtering. The Arabic claims come from misbar.com. Claims have 7 labels, and each has the top-5 Google search snippets as evidence. There are three test sets: in-domain α1, out-of-domain α2 and zero-shot α3. The best mBERT model reaches only 41.9 macro-F1 on α1 and about 15–17 on α2 and α3. Evidence snippets add only 2.5 points in-domain, do not help on α3, and a claim-only model is almost as good. The authors read this as a sign of claim-only bias in the dataset.

### Cited Findings
- **Paper:** "X-FACT: A New Benchmark Dataset for Multilingual Fact Checking", Ashim Gupta & Vivek Srikumar (University of Utah), *Proceedings of ACL-IJCNLP 2021 (Short Papers)*, pp. 675–682. Data and code: https://github.com/utahnlp/x-fact — [ACL Anthology PDF](https://aclanthology.org/2021.acl-short.86.pdf); [arXiv 2106.09248](https://arxiv.org/abs/2106.09248)
- **Size and languages:** "31,189 general domain non-English claims from 25 languages", 11 language families. The languages are ar, az, bn, de, es, fa, fr, gu, hi, id, it, ka, mr, no, nl, pa, pl, pt, ro, ru, si, sr, sq, ta, tr. The authors call it "an order of magnitude larger than any other multilingual dataset available for fact checking" at the time. — [Gupta & Srikumar 2021](https://aclanthology.org/2021.acl-short.86.pdf)
- **Construction:**
  - Sources are non-partisan fact-checkers from the IFCN and Duke Reporter's Lab lists. English-language sites already covered by LIAR or MultiFC were excluded.
  - Claims came from Google Fact Check Explorer plus direct crawling. Duplicates and claims with the label written inside the claim were removed. This left 85 fact-checkers.
  - The 54 sites where one label was more than 80% of the data were removed, and so were sites with fewer than 60 examples.
  - Metadata kept: claimant, URL, claim date, review date.
  - — [Gupta & Srikumar 2021](https://aclanthology.org/2021.acl-short.86.pdf)
- **Arabic source:** In the appendix (Table 6), the only Arabic fact-checker listed is misbar.com (Afro-Asiatic family). — [Gupta & Srikumar 2021](https://aclanthology.org/2021.acl-short.86.pdf)
- **Arabic size:**
  - A later paper reports Arabic as 8.2% of the X-FACT training data. — [Shcharbakova et al. 2025](https://arxiv.org/html/2507.20700v1)
  - Another took the Train, Dev and α1 splits, kept only claims labelled true or false, and got 771 Arabic claims: 730 false and 41 true. So Arabic is one of the 12 training languages, not a zero-shot language. — [Gupta et al., AbjadNLP 2025](https://aclanthology.org/2025.abjadnlp-1.12.pdf)
- **Labels (7):**
  - The original site ratings were manually translated to English and merged with help from Factly, an IFCN signatory.
  - Five truth levels: True, Mostly-True, Partly-True, Mostly-False, False.
  - Plus "Unverifiable" (lack of evidence or subjective) and "Other". The majority label is False.
  - — [Gupta & Srikumar 2021](https://aclanthology.org/2021.acl-short.86.pdf)
- **Splits (Table 2):**

  | Split | Claims | Languages |
  |---|---|---|
  | Train | 19,079 | 13 |
  | Dev | 2,535 | 12 |
  | α1 in-domain test (same languages and sources as train) | 3,826 | 12 |
  | α2 out-of-domain test (same languages, different sources) | 2,368 | 4 |
  | α3 zero-shot test (languages not in train) | 3,381 | 12 |

  The top 12 languages are split 75/10/15 into train, dev and α1. The text says "13 languages" are left for α3, while Table 2 lists 12; this is an inconsistency inside the paper. — [Gupta & Srikumar 2021](https://aclanthology.org/2021.acl-short.86.pdf)
- **Evidence:** Each claim was searched on Google and the top-5 result snippets kept. Snippets were used instead of full pages because of mBERT's 512-token limit, following MultiFC. — [Gupta & Srikumar 2021](https://aclanthology.org/2021.acl-short.86.pdf)
- **Models:** All use mBERT-base, trained once on all languages together; monolingual models were unstable.
  - Claim-Only.
  - Attn-EA: the claim and each of the 5 snippets are encoded separately, combined with dot-product attention, and concatenated.
  - +Meta: language, website, claimant, claim date and review date added as "Key : Value" text.
  - Training: batch 8, 10 epochs, macro-F1, mean of 4 seeds.
  - — [Gupta & Srikumar 2021](https://aclanthology.org/2021.acl-short.86.pdf)
- **Results (macro-F1, mean with sd in brackets, Table 3):**

  | Model | α1 | α2 | α3 |
  |---|---|---|---|
  | Majority | 6.9 | 10.6 | 7.6 |
  | Claim-Only | 38.2 (0.9) | 16.2 (0.9) | 14.7 (0.6) |
  | Claim-Only + Meta | 39.4 (0.9) | 15.4 (0.8) | 16.7 (1.1) |
  | Attn-EA (Random snippets) | 37.5 (0.8) | 16.3 (0.5) | 14.9 (1.2) |
  | Attn-EA | 38.9 (0.2) | 15.7 (0.1) | 16.5 (0.7) |
  | Attn-EA + Meta | 41.9 (1.2) | 15.4 (1.5) | 16.0 (0.3) |

  — [Gupta & Srikumar 2021](https://aclanthology.org/2021.acl-short.86.pdf)
- **What they found about evidence:**
  - Snippets plus metadata give +2.5 points on α1.
  - Pairing each claim with random snippets from the same language scores about the same as claim-only. The authors take this as confirming "the dataset exhibits some claim-only bias".
  - On α3 the claim-only model beats the evidence model by a small margin, possibly because the evidence model overfits in-domain data.
  - In a manual check of 100 Hindi α1 claims, only 45% of snippets gave enough information to decide the label, against 83% for the full web pages.
  - Scores on α2 and α3 are much lower, so models overfit to source-specific and language-specific patterns.
  - — [Gupta & Srikumar 2021](https://aclanthology.org/2021.acl-short.86.pdf)
- **Adding English data hurts:** Adding 12,311 PolitiFact claims from MultiFC to training lowered scores. Claim-Only+Meta fell to 37.1 / 14.5 / 14.4 and Attn-EA+Meta to 38.0 / 14.7 / 14.3 (α1/α2/α3). The authors suggest it is because PolitiFact is political while X-FACT is general-domain. — [Gupta & Srikumar 2021](https://aclanthology.org/2021.acl-short.86.pdf)

### Inferences
- X-FACT's snippets are Google results, so they can contain fact-checker verdict words; CONCRETE (below) shows this is a leak. If TwinCheck uses X-FACT Arabic, `clean.py` should drop fact-checking domains (misbar.com, fatabyyano.net and similar) from the snippets. This is already in the CLAUDE.md rules and is strongly supported by this literature.
- The random-snippet result in X-FACT is close to our `_C` (unrelated) twin. The original paper already found that the model barely changes when evidence is replaced. That is direct prior evidence for the "evidence is not read" problem TwinCheck audits, in a multilingual setting.
- Arabic in X-FACT is very skewed toward false. In the true/false-only subset, 94.7% of claims are false (730/771). After our label map (True/Mostly-True → supported; False/Mostly-False → contradicted; Unverifiable → unverified; Partly-True and Other dropped), expect very few `supported` Arabic claims. That matters for the twin sources rule, which needs supported claims, and for macro-F1.

### Gaps
- The per-language split counts in Table 6 (the Train/Dev/α1/α2/α3 columns) did not extract from the PDF. So the exact Arabic count per split is not confirmed here. The 8.2% share and the 771-claim filtered subset are from later papers. Check by counting `language == "ar"` in the GitHub release.

## Later papers using X-FACT: tasks, models, results

### Takeaway
Later work falls into three groups:
1. Cross-lingual retrieval (CONCRETE, COLING 2022) raised zero-shot α3 to 19.83 macro-F1.
2. Fine-tuned XLM-R (FEVER workshop 2025) raised it to 57.7 / 47.6 / 43.2. In the same study, 7–12B LLMs with few-shot prompting or LoRA scored only 8–17 macro-F1 and did not use evidence well.
3. Small LLM-prompting studies on X-FACT subsets, including an Arabic-only one, report accuracy on binary true/false labels, which hides heavy class imbalance.

### Cited Findings
- **CONCRETE: setup.** "CONCRETE: Improving Cross-lingual Fact-checking with Cross-lingual Retrieval", Kung-Hsiang Huang, ChengXiang Zhai, Heng Ji, COLING 2022. — [arXiv 2209.02071](https://arxiv.org/abs/2209.02071)
  - A bi-encoder retriever trained with a self-supervised Cross-lingual Inverse Cloze Task (X-ICT).
  - It retrieves from 347,557 BBC passages of at most 100 tokens, in Arabic, Russian, Indonesian, Persian, French and Portuguese.
  - The reader is mBERT. Main focus: the zero-shot split.
- **CONCRETE: results (macro-F1, Table 2, zero-shot / in-domain).** — [Huang et al. 2022](https://arxiv.org/pdf/2209.02071)

  | Reader | Retrieval | Zero-shot | In-domain |
  |---|---|---|---|
  | Prior mBERT | none | 16.7 | 39.4 |
  | Prior mBERT | Google | 16.0 | 41.9 |
  | Reimplemented mBERT | none | 17.25 | 36.91 |
  | Reimplemented mBERT | Google | 16.02 | 42.61 |
  | mBERT | MT+DPR | 15.01 | 35.29 |
  | mBERT | BM25 | 17.43 | 38.29 |
  | mBERT | mDPR | 17.60 | 36.79 |
  | mBERT | CONCRETE | 19.83 (p < .05) | 40.53 |

  This is a new zero-shot state of the art, +2.23 absolute.
- **CONCRETE: Google snippets leak verdicts.** Google results for well-resourced training languages often contain fact-checkers' verdict words. Example: among Indonesian claims whose Google results contain "SALAH" (WRONG), 50% are PARTLY TRUE and 45% FALSE. "Such patterns can also be found in the in-domain split, but not in the zero-shot split." This explains why Google evidence helps in-domain but not zero-shot. — [Huang et al. 2022](https://arxiv.org/pdf/2209.02071)
- **CONCRETE: Arabic findings.**
  - Retrieved passages were translated into Indonesian, Portuguese and Arabic. The drop in performance grows with the distance between the claim's language and the passage's language (Kendall's τ = −0.490).
  - Cross-lingual retrieval beats monolingual retrieval for both mDPR and CONCRETE (trained on pt, id, ar; tested on fr, fa, ru).
  - — [Huang et al. 2022](https://arxiv.org/pdf/2209.02071)
- **Shcharbakova et al. 2025: setup.** "When Scale Meets Diversity: Evaluating Language Models on Fine-Grained Multilingual Claim Verification", Hanna Shcharbakova, Tatiana Anikina, Natalia Skachkova, Josef van Genabith, FEVER Workshop at ACL 2025. — [arXiv 2507.20700](https://arxiv.org/abs/2507.20700)
- **Shcharbakova et al. 2025: encoder results (macro-F1, claim + evidence, Table 3; Test / OOD / Zero-shot).** — [HTML](https://arxiv.org/html/2507.20700v1)

  | Model | Test | OOD | Zero-shot |
  |---|---|---|---|
  | mBERT baseline (from Gupta & Srikumar) | 41.9 | 16.2 | 16.7 |
  | XLM-R frozen (head only) | 51.4 | 40.8 | 41.3 |
  | XLM-R full fine-tuning | 57.7 | 47.6 | 43.2 |
  | mT5 | 47.6 | 22.2 | 19.2 |

  - The paper describes 57.7 as a gain of "15.8%" over the previous SOTA of 41.9 (absolute points).
  - Note: the mBERT baseline row mixes columns from different rows of the original Table 3 (41.9 is Attn-EA+Meta α1; 16.2 and 16.7 are claim-only figures).
- **Shcharbakova et al. 2025: LLM results (Table 4, macro-F1).**

  | Model | Mode | Test | OOD | Zero-shot |
  |---|---|---|---|---|
  | Qwen 2.5 7B | 7-shot, claim + evidence | 12.7 | 13.0 | 10.9 |
  | Qwen 2.5 7B | LoRA, claim + evidence | 15.9 | 15.1 | 15.4 |
  | Qwen 2.5 7B | LoRA, claim only | 16.9 | 11.1 | 11.7 |
  | Mistral Nemo 12B | 7-shot, claim + evidence | 14.8 | 16.1 | 15.1 |
  | Llama 3.1 8B | 7-shot, claim + evidence | 14.0 | 13.3 | 12.9 |
  | Llama 3.1 8B | LoRA, claim only | 15.5 | 13.5 | 12.1 |

  - The best LLM score is 16.9 (Qwen, LoRA, claim-only).
  - The abstract says LLMs show "systematic difficulty leveraging evidence" and a strong bias toward frequent classes.
  - LoRA: rank 16, alpha 32, 4-bit via Unsloth.
  - XLM-R hyperparameters: lr 1.82e-5, batch 6.
  - No per-language or Arabic breakdown is given.
  - — [Shcharbakova et al. 2025](https://arxiv.org/html/2507.20700v1)
- **Shcharbakova et al., RANLP 2025.** "Cross-Lingual Fact Verification: Analyzing LLM Performance Patterns across Languages", same authors, RANLP 2025, pp. 1137–1147. Llama 3.1, Qwen 2.5 and Mistral Nemo on X-FACT. Findings:
  - "Large performance gaps tied to writing script": Latin-script languages do best.
  - Systematic cross-lingual instruction-following failures, especially for non-Latin scripts.
  - Indonesian and Polish beat German and Spanish, so resource level does not fully predict performance.
  - The abstract gives no numbers. — [ACL Anthology](https://aclanthology.org/2025.ranlp-1.131/)
- **Singhal et al., NLP4PI 2024.** "Multilingual Fact-Checking using LLMs", Aryan Singhal, Thomas Law, Coby Kassner, Ayushman Gupta, Evan Duan, Aviral Damle, Ryan Luo Li, NLP4PI 2024 (EMNLP workshop), pp. 13–31.
  - Data: an X-FACT subset in Spanish, Italian, Portuguese, Turkish and Tamil. Five LLMs.
  - Prompting: zero-shot, English chain-of-thought (CoT), cross-lingual prompting, each with greedy decoding or self-consistency.
  - GPT-4o was best. Zero-shot with self-consistency was most effective. CoT and cross-lingual prompting did not reliably help.
  - Accuracy correlated negatively with the amount of internet content in a language.
  - The landing page gives no exact numbers. — [ACL Anthology](https://aclanthology.org/2024.nlp4pi-1.2/)
- **Gupta et al., AbjadNLP 2025: setup.** "Can LLMs Verify Arabic Claims? Evaluating the Arabic Fact-Checking Abilities of Multilingual LLMs", Ayushman Gupta, Aryan Singhal, Thomas Law, Veekshith Rao, Evan Duan, Ryan Luo Li, AbjadNLP 2025 (Abu Dhabi), pp. 104–113. — [PDF](https://aclanthology.org/2025.abjadnlp-1.12.pdf)
  - Data: 771 X-FACT Arabic claims labelled true or false (730 false, 41 true).
  - Claim only, no evidence; the model's own knowledge is tested. Metric is accuracy.
- **Gupta et al., AbjadNLP 2025: accuracy (Table 1).**

  | Model | Zero-shot | English CoT | Self-consistency | Cross-lingual |
  |---|---|---|---|---|
  | Llama 3 8B | 59.01 | 66.93 | 68.61 | 86.55 |
  | Llama 3 70B | 40.21 | 61.22 | 59.66 | 80.42 |
  | Gemini 1.0 Pro | 30.60 | 49.68 | 52.53 | 49.41 |
  | GPT-3.5-turbo | 60.94 | 59.79 | 63.68 | 78.21 |

  The authors conclude that cross-lingual prompting works best. — [PDF](https://aclanthology.org/2025.abjadnlp-1.12.pdf)

### Inferences
- The AbjadNLP Arabic accuracies are hard to interpret. Always answering "false" would score 94.7% (730/771), above every model in Table 1. Macro-F1 is not reported. If TwinCheck cites this paper, it should note this caveat. It also supports our choice of macro-F1 and ESR over accuracy.
- The gap between the 2021 mBERT scores and the 2025 XLM-R scores (41.9 → 57.7 in-domain; 16.7 → 43.2 zero-shot) is large. It may partly reflect much more hyperparameter search (Bayesian sweeps, 60–90 runs) and a different backbone. No twin-style test checks whether XLM-R actually uses the evidence. To our knowledge, none of these X-FACT papers runs evidence-swap or unrelated-evidence tests beyond the original random-snippet ablation.
- Together, CONCRETE and the original random-snippet ablation show that good in-domain X-FACT scores can come from verdict leakage in snippets and claim-only cues. This is a multilingual parallel to *Similarity over Factuality*, the main motivation for TwinCheck.

### Gaps
- Exact accuracies for Singhal et al. 2024 and numbers for RANLP 2025 were not read; only abstracts were seen.
- The arXiv 2502.15419 ("Beyond Translation: LLM-Based Data Generation for Multilingual Fact-Checking") and FactNet (arXiv 2602.03417) papers appeared in search. One search summary reported FactNet zero-shot X-FACT macro-F1 of 0.462 against 0.387 for machine-translation-projected training data, over 7 languages. This was not checked against the paper, so leave it out unless it is checked.

## M4FC: paper, tasks, languages (incl. Arabic), models and results

### Takeaway
M4FC (Geng, Tonglet & Gurevych; arXiv 2510.23508, Oct 2025, v2 Jan 2026) is a real-world multimodal fact-checking dataset. It has 4,982 images and 6,980 claims in 10 languages from 22 IFCN fact-checkers in 17 countries; Arabic is 12.4% of claims. It defines six tasks. Verdict prediction is binary true/false with macro-F1, and no model passes 80 F1 in the balanced setting. Web evidence and image contextualization are the biggest helps.

### Cited Findings
- **Paper:** "M4FC: a Multimodal, Multilingual, Multicultural, Multitask Real-World Fact-Checking Dataset", Jiahui Geng (MBZUAI) and Jonathan Tonglet (TU Darmstadt / KU Leuven), equal contribution, with Iryna Gurevych (MBZUAI / TU Darmstadt). arXiv 2510.23508 (v2, January 2026). Code: github.com/UKPLab/M4FC; project page ukplab.github.io/M4FC. — [arXiv](https://arxiv.org/abs/2510.23508); [HTML v2](https://arxiv.org/html/2510.23508v2); [project page](https://ukplab.github.io/M4FC)
- **Venue (unconfirmed):** The arXiv page names no venue. A co-author's LinkedIn post says Findings of EMNLP 2026; no official listing was found. — [web search result](https://se.linkedin.com/in/johnnysharkhead)
- **Size:**
  - 4,982 images and 6,980 claims. The project page says 6,985 claims, a mismatch between sources.
  - From 22 IFCN fact-checking organisations in 17 countries. Images cover 159 countries.
  - — [arXiv](https://arxiv.org/html/2510.23508v2); [project page](https://ukplab.github.io/M4FC)
- **Languages (10):** Arabic, Dutch, English, French, German, Portuguese, Spanish, Tamil, Telugu, Turkish. — [Geng et al.](https://arxiv.org/html/2510.23508v2)
  - Claim shares: English 63.8%, Arabic 12.4%, Portuguese 5.0%, French 4.6%, Spanish 4.0%, German 3.6%, Telugu 3.0%, Dutch 1.6%, Tamil 1.0%, Turkish 1.0%.
  - Arabic sources: fatabyyano.net (Jordan, 658 articles) and verify-sy.com (42 articles).
- **Labels and checks:**
  - True claims are 5.9% of the imbalanced setting and 26.0% of the balanced setting.
  - 72% of false claims are out-of-context.
  - Labels were extracted by GPT-4o, and Prolific annotators checked them. Verdict was 99.3% correct (lenient bound) and claimed date 86.2%, the lowest.
  - — [Geng et al.](https://arxiv.org/html/2510.23508v2)
- **Six tasks (metrics):**
  1. Visual claim extraction from screenshots (ROUGE-L, METEOR, BERTScore, G-Eval). New task.
  2. Claimant intent prediction (same metrics).
  3. Fake image detection (macro-F1).
  4. Image contextualization: source, date, location, people, things, event, motivation (METEOR, F1, Δ-years, CO·Δ-km).
  5. Location verification against map or satellite imagery (macro-F1). New task, 195 instances.
  6. Verdict prediction, true/false (macro-F1).
  - — [Geng et al.](https://arxiv.org/html/2510.23508v2)
- **Splits:** Temporal split, about 60/10/30. Verdict prediction has 4,955 instances (2,979 / 498 / 1,478); fake detection 4,954; image contextualization 4,720. — [Geng et al.](https://arxiv.org/html/2510.23508v2)
- **Models:**
  - Open: InternVL2.5-8B, Qwen2.5-VL-7B, Llama-3.2-11B-Vision.
  - Proprietary: GPT-4o-mini, Gemini-1.5-Flash.
  - Fake-image detectors (pretrained and fine-tuned).
  - COVE for image contextualization and DEFAME for verdict prediction.
  - — [Geng et al.](https://arxiv.org/html/2510.23508v2)
- **Results by task:** — [Geng et al.](https://arxiv.org/html/2510.23508v2)
  - Fake detection: pretrained Deep-Fake Detector 70.31 F1; AI-Image Detector 6.41 F1. Fine-tuning on M4FC gives large gains.
  - Location verification: Gemini-1.5-Flash is best with text (80.83), dropping to 67.52 with satellite and 65.62 with map + satellite. Llama-3.2-11B scores 40.00 against a random baseline of 49.18.
  - Verdict prediction, English-only imbalanced pipeline: web evidence has the largest single effect. Ground-truth image contextualization adds about 17 F1. The best reported is about 89.2 F1 (GPT-4o-mini with ground-truth intermediate outputs plus web evidence). DEFAME (GPT-4o-mini) gets 86.63. Predicted rather than gold claims lower F1 substantially. (The table was dense; small row mismatches are possible.)
  - Verdict prediction, balanced setting: no model exceeds 80 F1. Proprietary models are similar on English and multilingual claims. Open models drop more on multilingual input, Llama-3.2-11B most; Qwen2.5-VL-7B improves slightly. Weak spots: GPT-4o-mini on Turkish, Gemini on Spanish and Tamil. The Arabic result is only in a figure (Fig. 7) and not stated in the text.
- **Data use and licence:**
  - Web evidence was found for only 48.62% of images, about 4 pages per image, mostly in English.
  - The authors say the balanced and multilingual verdict settings carry style or language biases. They should be used for evaluation only; train on the English-only imbalanced setting.
  - Annotations are CC BY-SA 4.0 and code Apache 2.0. Images are not redistributed; they are fetched with the download scripts.
  - — [Geng et al.](https://arxiv.org/html/2510.23508v2)

### Inferences
- About 12.4% of 6,980 claims is roughly 860 Arabic claims, which matches the "~860 Arabic claims" in CLAUDE.md. But the paper's share is for the multilingual setting, so the exact Arabic count for verdict prediction needs checking in the release.
- The authors themselves recommend evaluation-only use of the multilingual split. That fits CLAUDE.md's "M4FC = test only".
- M4FC evidence is mostly English while 36% of claims are not English. That is a natural "mixed" setting, close to our `_M` twin (Arabic claim, English evidence).
- The FD result (one off-the-shelf AI-image detector scores 6.41 F1 on real fact-checked images) supports the FALSDO motivation and our rule that the verdict head never sees the authenticity output.

### Gaps
- Exact Arabic verdict-prediction F1 per model was not extracted; it is only in Fig. 7.
- No later papers using M4FC were found. It is recent (Oct 2025).
- The EMNLP 2026 Findings venue is unconfirmed.

## Arabic fact-checking: datasets, performance and difficulties

### Takeaway
Arabic claim-verification resources are small and use different labels:
- AraFacts: about 6.1–6.2k claims from 5 Arabic fact-checking sites.
- ANS: about 4k news-title claims written by annotators.
- CheckThat!: binary Arabic tasks.

Reported difficulties are scarce Arabic evidence, retrieval quality, label imbalance (true claims are rare), dialects, and LLMs failing to follow instructions in non-Latin scripts. Very high Arabic scores (e.g., 96.15 macro-F1 at CheckThat! 2025) come from binary tasks without a "conflicting" class.

### Cited Findings
- **AraFacts.** "AraFacts: The First Large Arabic Dataset of Naturally Occurring Claims", Zien Sheikh Ali, Watheq Mansour, Tamer Elsayed, Abdulaziz Al-Ali, WANLP 2021, pp. 231–236.
  - Claims from 5 Arabic fact-checking sites including Fatabyyano and Misbar, from 2016 onward.
  - Metadata: factual labels unified across sites, article text, topic category, and links to the posts that spread each claim.
  - The ACL Anthology abstract gives 6,121 claims; a search snippet (QSpace record) said 6,222. Treat 6,121 (Anthology) as the citable figure.
  - No baselines on the landing page. — [ACL Anthology](https://aclanthology.org/2021.wanlp-1.26)
- **ANS.** "Stance Prediction and Claim Verification: An Arabic Perspective", Jude Khouja, FEVER Workshop 2020.
  - Claims built from Arabic news titles (BBC, CNN, Al Arabiya and others in the ANT corpus). Annotators rewrote them into true or false claims.
  - Stance labels: agree / disagree / other. The Hugging Face mirror lists 2,652 / 755 / 379 stance instances.
  - Best model (BERT): 76.7 F1 on stance, 64.3 F1 on claim verification.
  - The author concludes pretrained representations "are insufficient for verifying claims without access to context or evidence".
  - — [ACL Anthology PDF](https://aclanthology.org/2020.fever-1.2.pdf); [arXiv 2005.10410](https://arxiv.org/abs/2005.10410); [HF mirror](https://huggingface.co/datasets/strombergnlp/ans-stance/blob/main/README.md)
- **CheckThat! 2025 Task 3: setup.** "Overview of the CLEF-2025 CheckThat! Lab Task 3 on Fact-Checking Numerical Claims", Venktesh V, Vinay Setty, Avishek Anand, Boushra Bendou, Maram Hasanain, Houda Bouamor, Gabriel Iturra-Bocaz, Petra Galuščáková, Firoj Alam, CLEF 2025 Working Notes (CEUR Vol-4038). — [CEUR PDF](https://ceur-ws.org/Vol-4038/paper_53.pdf)
  - Arabic is True/False only, "as real-world distribution of conflicting claims for Arabic is too low".
  - Arabic claims come from news, social media and fact-checkers, and include AraFacts.
  - Arabic split (True / False): train 975 / 1,216 (2,191); dev 274 / 313 (587); test 206 / 276 (482).
  - Evidence was collected by search engine with fact-checking sites excluded "to avoid leakage of fact-checker justification and verdict". The top-100 BM25 evidence is provided.
- **CheckThat! 2025 Task 3: results (macro-F1).**

  | Team | Arabic | Spanish | English |
  |---|---|---|---|
  | LIS | 96.15 | 50.34 | 59.54 |
  | NGU_Research | 63.52 | 24.41 | not entered |
  | TIFIN | 55.36 | not entered | 55.70 |
  | JU_NLP | 36.38 | not entered | 48.83 |

  - Four teams entered Arabic.
  - The organisers explain the high Arabic score by the binary task. LIS's advantage is "improvement of retrieval using dense retrieval approaches like Linq-Embed-Mistral and fine-tuning LLMs instead of prompting".
  - — [CEUR PDF](https://ceur-ws.org/Vol-4038/paper_53.pdf)
- **Earlier CheckThat! Arabic tasks.**
  - 2019: claims verified against the top-100 Google pages per claim.
  - 2021: the Arabic track was verified-claim retrieval (858 tweets; 30,329 previously fact-checked claims).
  - 2026: the Arabic data has 3,260 claims.
  - These come from search snippets of the overview papers; the official numbers were not read. A secondary aggregator gave 2020 Arabic evidence-retrieval P@10 = 0.0456 and best claim-verification macro-F1 = 0.5524; not checked against the [CheckThat! 2020 overview](https://arxiv.org/html/2007.07997v1).
  - — [CLEF-2019 overview](https://arxiv.org/pdf/2109.15118); [CLEF-2021 overview](https://arxiv.org/pdf/2109.12987v1); [CLEF-2026 lab](https://arxiv.org/pdf/2602.09516)
- **Difficulties found by X-FACT-based work:**
  - Cross-lingual prompting (reasoning in English) raised claim-only Arabic accuracy for 3 of 4 LLMs (e.g., Llama 3 8B 59.01 → 86.55). — [Gupta et al. 2025](https://aclanthology.org/2025.abjadnlp-1.12.pdf)
  - Non-Latin-script languages show instruction-following failures. — [Shcharbakova et al. RANLP 2025](https://aclanthology.org/2025.ranlp-1.131/)
  - Passages in a language far from the claim's language are harder for an mBERT reader, which included Arabic in its translation tests. — [Huang et al. 2022](https://arxiv.org/pdf/2209.02071)
- **Other recent Arabic work (search snippets only, not read):**
  - Arafa, an LLM-generated Arabic fact-checking dataset (arXiv 2609.25833): all four models it tested struggled most with "not enough info".
  - An AbjadNLP 2025 study of GPT-4 on health claims found that changing between Saudi, Egyptian, Lebanese and Moroccan dialects affected answer consistency.
  - — [Arafa](https://arxiv.org/html/2609.25833v2); [AbjadNLP 2025 health claims](https://aclanthology.org/2025.abjadnlp-1.11.pdf)

### Inferences
- Across X-FACT Arabic (94.7% false in the true/false subset), CheckThat! (no conflicting class) and M4FC (5.9% true overall), the Arabic "supported" and "unverified" classes are rare. Report macro-F1 per class and use twins (`_A`, `_M` from English sources) to get balanced Arabic tests. This supports TwinCheck's design of building Arabic twins from English supported sources.
- Several sources agree that retrieval quality is the bottleneck for Arabic, while snippets from fact-checking sites leak the verdict: CheckThat! 2025 LIS, CONCRETE, and the X-FACT Hindi snippet audit. Using a reranker plus removing fact-check domains is consistent with this literature.
- The literature says little about dialects for fact verification specifically. Most Arabic resources (AraFacts, X-FACT/Misbar, ANS) are mostly Modern Standard Arabic news or fact-check text. This is our inference; no source states the MSA/dialect share.

### Gaps
- No source gave the share of dialectal versus MSA claims in AraFacts, X-FACT Arabic or M4FC Arabic.
- No paper reports Arabic results on X-FACT per split (α1) with macro-F1. The only Arabic-only X-FACT paper uses accuracy on a skewed binary subset.
- CheckThat! 2020–2024 official Arabic scores were not verified from the overview papers.
- AraStance (Alhindi et al., 2021) and the Arabic subsets of multilingual claim-matching datasets (e.g., MultiClaim) were not covered in this pass.
