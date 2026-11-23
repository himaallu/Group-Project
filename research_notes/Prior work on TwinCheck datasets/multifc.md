# Prior work using MultiFC (Augenstein et al., EMNLP 2019)

## (a) What did the original MultiFC paper build, and what models/results did it report?

### Takeaway
MultiFC is 34,918 English claims from 26 fact-checking sites, with 165 site-specific labels kept as they are (2 to 27 labels per site). Each claim comes with the top-10 Google search snippets and rich metadata. The best model, a joint evidence-ranking and veracity model with multi-task label embeddings plus metadata, reaches **Micro-F1 0.625 / Macro-F1 0.492**. The claim-only baseline in the same paper gets Micro-F1 0.469 / Macro-F1 0.253.

### Cited Findings
- **Paper:** Isabelle Augenstein, Christina Lioma, Dongsheng Wang, Lucas Chaves Lima, Casper Hansen, Christian Hansen, Jakob Grue Simonsen. "MultiFC: A Real-World Multi-Domain Dataset for Evidence-Based Fact Checking of Claims." EMNLP-IJCNLP 2019 — [ACL Anthology D19-1475](https://aclanthology.org/D19-1475.pdf); arXiv [1909.03242](https://arxiv.org/abs/1909.03242)
- **Collection:** 43,837 claims crawled from 38 sources, giving 36,534 claims from 26 fact-checking sites. Instances whose label occurs fewer than 5 times were then dropped, leaving **34,918 claims** with **165 unique labels**. Each site ("domain") has 2 to 27 labels — [Augenstein et al. 2019](https://aclanthology.org/D19-1475.pdf)
- **Metadata per claim:** ID, claim, label, URL, reason for label, categories, speaker, checker, tags, article title, publication date, claim date, full text, outlinks — [Augenstein et al. 2019](https://aclanthology.org/D19-1475.pdf)
- **Evidence:** each claim's text was sent verbatim (no quotes) to the Google Search API. The top-10 results were saved with title, rank, URL, timestamp, snippet and full page. The models use only the **search snippets** — [Augenstein et al. 2019](https://aclanthology.org/D19-1475.pdf). The release README says that fewer than 10 snippets may appear because "we have excluded the claimURL from the snippets". Only the claim's own fact-check article URL was removed, not other fact-checking sites — [HF mirror pszemraj/multi_fc, original README](https://huggingface.co/datasets/pszemraj/multi_fc)
- **Labels:** no common label set. The task is framed as **multi-task learning, one task per domain**, with a **Label Embedding Layer**: labels are projected into a shared embedding space, the prediction is the dot product of the claim+evidence representation with the label embeddings, and per-domain masks restrict the output to that site's labels — [Augenstein et al. 2019](https://aclanthology.org/D19-1475.pdf)
- **Evidence ranking:** the "Joint Veracity Prediction and Evidence Ranking" model (BiLSTM encoders) learns to rank snippets by how useful they are for veracity, with no gold ranking labels — [Augenstein et al. 2019](https://aclanthology.org/D19-1475.pdf)
- **Splits:** label-stratified 80/10/10 train/dev/test — [Augenstein et al. 2019](https://aclanthology.org/D19-1475.pdf). **Test labels were removed from the public release** ("we have removed the label"). The HF mirror warns that the test `label` column holds **dummy values**. HF split sizes: train 27,871 / validation 3,484 / test 3,487 — [HF mirror card](https://huggingface.co/datasets/pszemraj/multi_fc)
- **Metric:** Micro-F1 and Macro-F1 computed per domain, then averaged over domains — [Augenstein et al. 2019](https://aclanthology.org/D19-1475.pdf)
- **Results (test, average over domains)** — [Augenstein et al. 2019, results table](https://aclanthology.org/D19-1475.pdf):

| Model | Micro-F1 | Macro-F1 |
|---|---|---|
| claim-only | 0.469 | 0.253 |
| claim-only_embavg | 0.384 | 0.302 |
| crawled-docavg | 0.438 | 0.248 |
| crawled_ranked | 0.613 | 0.441 |
| claim-only + meta | 0.494 | 0.324 |
| claim-only_embavg + meta | 0.418 | 0.333 |
| crawled-docavg + meta | 0.483 | 0.286 |
| **crawled_ranked + meta** | **0.625** | **0.492** |

- **Authors' analysis:** evidence-based models clearly beat claim-only ones, and metadata (speaker, tags) helps further. Longer claims are harder. High token overlap between claim and evidence leads to a high evidence rank. Very specific tags (e.g. "brisbane-4000", "hong-kong") co-occur with correct predictions and general tags with errors — [Augenstein et al. 2019](https://aclanthology.org/D19-1475.pdf)

### Inferences
- The Micro/Macro numbers are averages over 26 heterogeneous label spaces. They cannot be compared with 3-class (supported/contradicted/unverified) scores such as TwinCheck's.
- The paper itself points to shortcuts: token overlap drives evidence ranking, and specific tags or topics predict veracity. This is the kind of "similarity over factuality" shortcut TwinCheck's twins are designed to test.
- Removing only the claim's own URL leaves other fact-check articles in the snippets. This is the leakage that Glockner et al. (2022) later measured. TwinCheck's cleaning rule 1 (drop fact-check-site snippets) targets exactly this.

### Gaps
- Exact per-site label counts and the breakdown of claims per site were not pulled from the paper's appendix tables.
- The HF split sizes (27,871 + 3,484 + 3,487 = 34,842) do not exactly match the paper's 34,918. I did not find the reason. It may be related to the 202 duplicate claims mentioned in CLAUDE.md, but that is not verified.
- The paper reports test-set numbers although the public test labels are withheld. The authors had the labels; I found no public leaderboard for the hidden test set.

## (b) Which later works use MultiFC, and with what task, model, method and results?

### Takeaway
Later work falls into four groups: (1) temporal evidence-ranking extensions of the original model on the full 26-domain task, reaching up to Micro-F1 0.6952–0.6973 / Macro-F1 0.5744 with DistilRoBERTa; (2) shortcut and ablation studies on the Snopes and PolitiFact subsets; (3) domain-adaptation studies that binarise the labels; (4) surveys and dataset papers that cite MultiFC and point out its flaws.

### Cited Findings

**1. Allein, Augenstein, Moens. "Time-aware evidence ranking for fact-checking." Journal of Web Semantics, 2021** — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S157082682100038X); [PDF](https://researchprofiles.ku.dk/files/284630787/Time_aware_evidence_ranking_for_fact_checking.pdf)
- Task: full 26-domain MultiFC veracity prediction. Splits: train 27,940 / dev 3,493 / test 3,491 (34,924 claims), label-stratified. Per-domain label masks as in the original paper. Up to 10 Google snippets per claim. The base model is the MultiFC joint ranking model. Evidence is re-ranked by four temporal schemes (evidence date, claim date, claim-centred distance, evidence-centred distance). The original Google order is treated as "search ranking".
- Results (Micro / Macro-F1): BiLSTM base .5521 / .3185 → time-aware (best temporal ranking per domain) **.6265 / .3673**. RNN base .5053 / .2691 → .5899 / .3453. DistilBERT base .5558 / .2909 → .5921 / .3135. Best single scheme (evidence-based date ranking, BiLSTM): .5794 / .3227.
- Per-domain scores vary hugely: e.g. pomt (PolitiFact) Micro .1813 / Macro .0594; snes (Snopes) .5662 / .1026; tron .4767 / .0366; and some tiny domains (bove, fani, ranz) at 1.0 / 1.0.

**2. Allein, Saelens, Cartuyvels, Moens. "Implicit Temporal Reasoning for Evidence-Based Fact-Checking." Findings of EACL 2023** — [ACL Anthology](https://aclanthology.org/2023.findings-eacl.13.pdf)
- Full MultiFC (34,924 claims), label-stratified 80/10/10. The base model is again the joint ranking model of Augenstein et al. The new models reason implicitly over shared timelines at document level and content level (DL/CL, local/global).
- Results (Micro / Macro-F1, mean with sd): BiLSTM base .5520 / .3239. BiLSTM DL+CLglob (best BiLSTM) **.6417 / .4743**. DistilRoBERTa base .6952 / .5532. RoBERTa CLglob .6882 / .5744. RoBERTa DLglob .6973 / .5608. RoBERTa DL+CLglob .6947 / .5739. They also report that the 2021 explicit time-aware ranking gives a Transformer .5921 / .3135.
- So a plain transformer base already beats the original paper's best reported Macro-F1 (0.492).

**3. Hansen, Hansen, Chaves Lima. "Automatic Fake News Detection: Are Models Learning to Reason?" ACL-IJCNLP 2021 (short paper)** — [arXiv 2105.07698](https://arxiv.org/abs/2105.07698)
- Uses only the **PolitiFact and Snopes** subsets extracted from MultiFC. Own label-stratified **70/10/20** train/val/test split, since there are no public test labels. Non-veracity labels are removed (PolitiFact: full flop, half flip, no flip; Snopes: unproven, miscaptioned, legend, outdated, misattributed, scam, correct attribution). For cross-dataset tests, "pants on fire!" is merged with "false".
- Models: Random Forest (bag-of-words), GloVe-LSTM, BERT. Each is given claim only, evidence (snippets) only, or claim+evidence.
- Within-dataset results (F1-micro / F1-macro):
  - Snopes, BERT: claim 0.533 / 0.312; evidence 0.531 / 0.321; claim+evidence 0.556 / 0.313.
  - PolitiFact, BERT: claim 0.275 / 0.282; **evidence 0.351 / 0.359**; claim+evidence 0.285 / 0.292.
  - Snopes, RF: claim 0.473 / 0.231; evidence 0.504 / 0.280; claim+evidence 0.550 / 0.271.
  - PolitiFact, LSTM: claim 0.237 / 0.237; evidence 0.290 / 0.295; claim+evidence 0.288 / 0.294.
- Conclusion: "the best performance can most often be obtained using only the evidence"; models using claim+evidence "are inherently not learning to reason".

**4. Glockner, Hou, Gurevych. "Missing Counter-Evidence Renders NLP Fact-Checking Unrealistic for Misinformation." EMNLP 2022** — [arXiv 2210.13865](https://arxiv.org/abs/2210.13865); [HTML](https://arxiv.org/html/2210.13865v1)
- Defines **leaked evidence** as snippets that come from a fact-checking article or that discuss the claim's veracity. These are found with URL patterns and phrase patterns.
- On MultiFC misinformation claims, leaked evidence is found for **69.7%** of claims (URL patterns 55.6% = 8,999 claims; phrase patterns 59.7% = 9,656; URL or phrase 11,267 = 69.7%).
- Experiment: bert-base-uncased fine-tuned on Hansen et al.'s (2021) Snopes and PolitiFact splits. Inputs: snippet text, snippet title, full snippets, and the full input with the claim. The test set is split into claims with and without leaked evidence (F1-macro / F1-micro):
  - PolitiFact, Full: All 52.6 / 51.9; **Leaked 59.7 / 59.3; Unleaked 25.6 / 25.9** (2,111 leaked vs 606 unleaked test claims).
  - Snopes, Full: All 32.7 / 62.7; Leaked 30.6 / 68.7; Unleaked 33.0 / 57.2.
  - Same claims scored with leaked vs unleaked evidence: PolitiFact F1-micro 57.9 vs 25.8. Snopes 66.0 vs 61.6.
  - Claim-only baseline: Snopes F1-micro 67.4 / F1-macro 29.8; PolitiFact 28.9 / 29.5.
- Conclusion: models "rely on leaked evidence, which makes them unsuitable in real-world scenarios". Every surveyed dataset fails to meet both requirements: evidence sufficient to refute, and not leaked.

**5. Karisani & Ji. "Fact Checking Beyond Training Set." NAACL 2024** — [ACL Anthology](https://aclanthology.org/2024.naacl-long.124.pdf); [arXiv 2403.18671](https://arxiv.org/html/2403.18671v1)
- Cross-domain fact checking. Five MultiFC domains are built by topic (Arts, Business, Misc, Politics, Sensitive). Misc and Politics are the source domains; the others are targets whose labels are used only for evaluation. **Labels are binarised**: "True" → Support, everything else → Refute. Two random evidence documents are kept per claim.
- Model: bi-encoder dense retriever + RoBERTa reader, with adversarial domain adaptation on unlabeled target data. Baselines include fine-tuned NLI readers and GPT-3 / ChatGPT / GPT-4 in-context learning.
- Results (Macro-F1, pipeline): ours average **0.618**, best baseline 0.605. Reader alone: ours average 0.637, nli-ft 0.625, zero-shot nli 0.447. GPT-3 per target: M→A 0.456, M→B 0.536, M→S 0.530, against ours 0.595 / 0.605 / 0.648. Retriever NDCG@10 average: ours 0.784, BM25 0.711.

**6. Pan et al. "Investigating Zero- and Few-shot Generalization in Fact Verification." IJCNLP-AACL 2023** — [arXiv 2309.09444](https://arxiv.org/html/2309.09444v1)
- Lists MultiFC as a candidate dataset but **leaves it out of the benchmark**. They say evidence in some candidate datasets is coarse (whole pages) or noisy (e.g. the original fact-check webpage). The extraction did not show which of these reasons they attach to MultiFC specifically.

**7. Schlichtkrull, Guo, Vlachos. "AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web." NeurIPS 2023 Datasets & Benchmarks** — [arXiv 2305.13117](https://arxiv.org/html/2305.13117v3)
- Names MultiFC (with FEVER) as a common benchmark. Cites Glockner et al. (2022) for **temporal leaks** in MultiFC: "a claim from January might be annotated with evidence from March that year". Leaks can also cross splits. AVeriTeC was built partly to avoid this.

**8. Guo, Schlichtkrull, Vlachos. "A Survey on Automated Fact-Checking." TACL 2022** — [ACL Anthology](https://aclanthology.org/2022.tacl-1.11.pdf)
- Dataset table row: MultiFC, input = statement, 36,534 inputs, evidence = text/metadata, verdict = 2–27 classes, source = fact-check sites, English.

### Inferences
- On the full 26-domain task, the strongest numbers I found are DistilRoBERTa-based (Allein et al. 2023): about 0.69–0.70 Micro-F1 and up to 0.5744 Macro-F1.
- Nearly all careful follow-up work avoids the 165-label problem. Some keep the per-domain masks (the Allein line of work). Some restrict to PolitiFact and Snopes and drop non-veracity labels (Hansen; Glockner). Some binarise (Karisani & Ji). TwinCheck's 3-way mapping with "drop mixture/half-true/misleading/satire" matches the Hansen/Glockner practice of removing labels that are not about veracity.

### Gaps
- I found no peer-reviewed paper that runs a modern instruction-tuned LLM (e.g. Llama/Qwen/GPT-4) zero-shot on the full 26-domain MultiFC. Karisani & Ji evaluate GPT models only on their binarised 5-domain setup.
- I found no MultiFC-based paper on generating explanations. Explanation work (e.g. LIAR-PLUS, PUBHEALTH) uses other datasets. Not verified exhaustively.
- I did not verify whether the Allein et al. papers scored on the official hidden test set (Augenstein co-authored the 2021 paper) or on a re-split.

## (c) Can MultiFC be solved from the claim alone, or do its snippets leak the answer?

### Takeaway
Yes on both counts. In MultiFC's Snopes subset, a claim-only BERT matches or beats evidence-based models: Glockner et al. report claim-only F1-micro 67.4 against 62.7 with full evidence. Snippets leak the verdict for 69.7% of misinformation claims. On PolitiFact, F1 drops from 59.7 to 25.6 (macro) when leaked snippets are absent.

### Cited Findings
- Original paper: claim-only 0.469 / 0.253 vs best evidence+meta 0.625 / 0.492. On the full task, claim-only is clearly worse — [Augenstein et al. 2019](https://aclanthology.org/D19-1475.pdf)
- Snopes subset, BERT: claim-only 0.533 / 0.312 vs claim+evidence 0.556 / 0.313. The gap is tiny, and macro-F1 is basically equal — [Hansen et al. 2021](https://arxiv.org/abs/2105.07698)
- Evidence-only beats claim+evidence on PolitiFact for BERT (0.351 / 0.359 vs 0.285 / 0.292): the models do not reason over the claim–evidence pair — [Hansen et al. 2021](https://arxiv.org/abs/2105.07698)
- Leakage affects 69.7% of misinformation claims (URL or phrase patterns). PolitiFact Full input: leaked 59.7 / 59.3 vs unleaked 25.6 / 25.9 (F1-macro / micro). Snopes claim-only 67.4 F1-micro is above Snopes Full 62.7 — [Glockner et al. 2022](https://arxiv.org/html/2210.13865v1)
- Temporal leakage: evidence published after the claim, and evidence shared across splits — [Schlichtkrull et al. 2023 (AVeriTeC), citing Glockner et al. 2022](https://arxiv.org/html/2305.13117v3)
- The original authors themselves found that specific tags or topics co-occur with correct veracity predictions, a metadata or topic shortcut — [Augenstein et al. 2019](https://aclanthology.org/D19-1475.pdf)

### Inferences
- For TwinCheck: the claim-only baseline (Feba F4) and the leakage cleaning (drop fact-check domains, drop >90% overlap snippets) are directly justified by Hansen 2021 and Glockner 2022. The `_B` (evidence swap) and `_C` (unrelated evidence) twins test exactly what these papers suggest: whether verdicts really depend on the evidence.
- Expect lower MultiFC scores after cleaning than in pre-2022 papers, because leaked snippets inflated those numbers. This is worth saying in the report when comparing with earlier work.

### Gaps
- Glockner et al.'s full list of URL and phrase patterns (their Appendix B.2) was not extracted. It would be a useful reference for our `clean.py` domain list.
- I found no published fraction of leaked snippets for MultiFC *true* claims; Glockner's 69.7% is for misinformation claims.

## (d) How do later papers handle the hidden test labels and the per-site label sets?

### Takeaway
Because the test labels were never released, independent groups make their own stratified splits from the labelled train+dev data (Hansen 70/10/20; Glockner reuses Hansen's splits; Karisani & Ji use domain-transfer splits). For labels, they either keep per-domain masks or reduce to one or two sites and drop labels that are not about veracity. Others binarise.

### Cited Findings
- Test file released without labels; HF mirror test labels are dummy values — [HF mirror card / original README](https://huggingface.co/datasets/pszemraj/multi_fc)
- Hansen et al.: PolitiFact + Snopes only, own label-stratified 70/10/20 split, non-veracity labels removed, "pants on fire!" merged with "false" for cross-dataset tests — [Hansen et al. 2021](https://arxiv.org/html/2105.07698v1)
- Glockner et al.: reuse Hansen's train/dev/test splits. Snopes labels used: true, mostly true, mixture, mostly false, false. PolitiFact: true, mostly true, half-true, mostly false, false, pants on fire! — [Glockner et al. 2022](https://arxiv.org/html/2210.13865v1)
- Allein et al. 2021/2023: full 26 domains, per-domain label masks, label-stratified split of about 34.9k claims (27,940 / 3,493 / 3,491) — [Allein et al. 2021](https://researchprofiles.ku.dk/files/284630787/Time_aware_evidence_ranking_for_fact_checking.pdf); [Allein et al. 2023](https://aclanthology.org/2023.findings-eacl.13.pdf)
- Karisani & Ji: binary labels ("True" → Support, rest → Refute); topic-domain transfer split — [Karisani & Ji 2024](https://aclanthology.org/2024.naacl-long.124.pdf)
- Survey view: verdicts in MultiFC range over "2–27 classes" per source — [Guo et al. 2022](https://aclanthology.org/2022.tacl-1.11.pdf)

### Inferences
- TwinCheck's plan (train + validation only, own 80/10/10 split stratified by label with seed 42, 3-way mapping, drop mixture/half-true labels and report the count) is consistent with standard practice. Cite Hansen et al. (2021) as precedent for re-splitting and for removing non-veracity labels.
- Our numbers will not be directly comparable with Augenstein 2019 or Allein 2021/2023, which use per-domain label spaces and averaged Micro/Macro-F1. The report should say this explicitly.

### Gaps
- I found no paper that publishes a full mapping of all 165 MultiFC labels to a 3-way scheme. TwinCheck's `data/label_map.csv` would be new work, or at least I could not find a prior public version.
