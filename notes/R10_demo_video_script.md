# R10 · Demo video script (2–3 min)

**Setup.** Open `notebooks/R10_demo_colab.ipynb` in Colab (T4). In step 1 set `VERDICT`
to what the group agreed: `"rule"` (5 of 6 examples right in R6) or `"head"` (2 of 6).
Run all; step 4 prints a public `gradio.live` link. Browser at 100% zoom, recorder at
1080p, notifications off. Speak slowly; about 2 min 30 s in total.

**Before recording.** Check `results/R10/examples_<VERDICT>.json` (step 3). Only show
examples whose verdict matches `expected`. The shots below are the ones the rule got
right in R6. With the head, swap in examples it gets right, or say honestly on screen
that the verdict is wrong and why.

| Time | Screen | Say |
|---|---|---|
| 0:00–0:15 | App start page | "This is TwinCheck, a bilingual fact-checking assistant. It splits a message into claims, finds evidence, and gives each claim a verdict, a confidence and a short note that cites the evidence." |
| 0:15–0:45 | Click **Schools closed for a month (EN)** | "The yellow banner says this is a made-up example. The evidence says schools moved online for one day only, so the claim is Contradicted. The bar shows the confidence, and each number in the note points to the snippet it uses." |
| 0:45–1:10 | Click **New metro station (AR)** | "The same works in Arabic. The claim, the note and the evidence are shown right to left. The evidence confirms the new station, so the claim is Supported." |
| 1:10–1:35 | Click **Metro stopped by floods (mixed)** | "UAE messages often mix Arabic and English. Here the claim is mixed and the evidence is English. The note is written in Arabic and still cites snippet [1]." |
| 1:35–1:55 | Click **Crewed Mars trip (AR)** | "When the evidence is about something else, TwinCheck says Unverified instead of guessing." |
| 1:55–2:20 | Clear the evidence box, paste one claim from the MultiFC test set, click **Check** | "With no evidence pasted, it searches its offline library of 163 thousand snippets and keeps the best five. The citations now link to the original web pages." |
| 2:20–2:40 | Scroll to the footer | "The footer shows which verdict step is running and that every component is real. All the code is in our GitHub repository, and the models are on Hugging Face." |

**After recording.** Trim the start and end, keep it under 3 minutes, export MP4 (1080p),
and put it in `Final/` (or link it there if it is over 100 MB). Post the link in the group.
