# R10 · Demo video script (2–3 min)

**Setup.** Open `notebooks/R10_demo_colab.ipynb` in Colab (T4) and keep the defaults:
`VERDICT = "head"` (Himasri's trained twin-loss verdict head) and `FAST = True` (rule-based
claim split + template notes, ready in ~5 min). Run cells 1, 2 and 4 (cell 3 is optional).
Open the `gradio.live` link and **click "Twin 1" once before recording**, so the models are
already loaded. Browser at 100% zoom, recorder at 1080p, notifications off.

The blue banner above each result says "Verdict by: trained verdict head". If you switch to
`VERDICT = "rule"`, a red banner says "Rule stand-in, not the trained model". Mention it in
the voice-over.

**Expected with the trained head** (tested by Himasri in `notebooks/demo_quick_colab.ipynb`):
Twin 1 → Contradicted ~83%, Twin 2 → Contradicted ~83%, Twin 3 → Unverified ~97%. Check the
numbers on screen before recording. Do not use the six "Ready examples" with the head: only 2
of 6 come out as expected (the head's "supported" weakness, `results/R6/README.md`).

| Time | Screen | Say |
|---|---|---|
| 0:00–0:20 | App start page | "This is TwinCheck, a bilingual fact-checking assistant. For each claim it gives a verdict, a confidence and a short note that cites the evidence. The question we test is: does the verdict really come from the evidence?" |
| 0:20–0:50 | Click **Twin 1 · original (EN)** | "A made-up claim: Dubai Metro suspended all lines because of flooding. The evidence says the lines ran normally, so our trained verdict head says Contradicted, about 83%. The banner shows the trained model made this verdict." |
| 0:50–1:20 | Click **Twin 2 · Arabic claim** | "This is a twin: the same claim in Arabic, with the same English evidence. Only the language changed, so the verdict should not change, and it does not: Contradicted, about 83%. The Arabic claim is shown right to left." |
| 1:20–1:50 | Click **Twin 3 · unrelated evidence** | "Second twin: the same English claim, but the evidence is about a museum and a bakery. A model that only matches words would still guess. Ours says Unverified, about 97%, because nothing in the evidence talks about the metro." |
| 1:50–2:15 | Back to Twin 1, point at the note and the numbered evidence | "Each note cites the snippet it uses, so a user can check the source. With the full model, Qwen3-8B writes the note in the claim's language." |
| 2:15–2:40 | Scroll to the footer | "The footer lists every component. The code is in our GitHub repository, tagged v1.0.1, and the models are on Hugging Face." |

**After recording.** Trim the start and end, keep it under 3 minutes, export MP4 (1080p),
and put it in `Final/` (or link it there if it is over 100 MB). Post the link in the group.
