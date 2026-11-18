# Y3 · Claim extraction (Youssef)

`src/claims/extract.py` → `extract_claims(message) -> list[str]` (used by `src/pipeline.py`).

## Data
`data/claims/gold.jsonl`: 300 WhatsApp-style messages (102 EN / 99 AR / 99 mixed), each with 1-3 real
fact-checked claims (MultiFC train for English, X-FACT Arabic train for Arabic) wrapped in chat text
("Forwarded as received", "انشروها", emojis, "they're saying ..."). Drafted with `src/claims/draft_gold.py`
(seed 42, no claim used twice), checked by Youssef.
- `claims_0001-0100`: the 6 few-shot examples (2 per language) + LoRA training set (Wed).
- `claims_0101-0300`: **200 held-out test messages** (68 EN / 66 AR / 66 mixed).

## Results (200 held-out messages; a predicted claim matches a gold claim when BGE-M3 cosine >= 0.8)

| Extractor | EN F1 | AR F1 | Mixed F1 | All P | All R | All F1 |
|---|---|---|---|---|---|---|
| Rule-based split (CPU baseline) | 0.772 | 0.751 | 0.845 | 0.812 | 0.762 | 0.786 |
| Qwen3-8B few-shot (4-bit, greedy) | **0.989** | **0.966** | **1.000** | 0.984 | 0.984 | **0.984** |

Files: `f1_{split,qwen}.json`, `preds_{split,qwen}.jsonl`. Run on a Colab T4.

## Errors (Qwen, 8 of 200 messages with a different number of claims)
- Chat opener kept as a claim, mostly Arabic ("ما أدري إذا صحيح بس", "Not sure if true but") or a framing
  word left at the start ("الخبر يقول إن ...", "والأغرب ...").
- Long claims split in two ("..., والكورونا مجرد تغطية على الأمر") or two sentences of one claim split.
- A short title-like claim missed ("Inspirational story titled “The Room").
Arabic is the weakest language: the chat framing is less predictable there.

## Limitation
The messages are template-based and the gold claims appear word-for-word in them, so the task mainly
tests removing chat text and splitting claims, not rewriting paraphrased claims. Scores are an upper
bound for real forwarded messages.

## Re-run
    python -m src.claims.evaluate --backend split
    python -m src.claims.evaluate --backend qwen      # GPU
