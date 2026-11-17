# R10 · Demo video script (2–3 min)

Record in Colab with the R6 notebook (step 4 public link), browser at 100% zoom,
screen recorder at 1080p. Speak slowly; total about 2 min 30 s.

| Time | Screen | Say |
|---|---|---|
| 0:00–0:15 | App start page | "This is TwinCheck, a bilingual fact-checking assistant. It splits a message into claims, finds evidence in an offline library, and gives each claim a verdict, a confidence and a short note that cites the evidence." |
| 0:15–0:45 | Click **Metro line reopens (EN)** | "The banner says this is an illustrative, made-up example. The claim is Supported; the bar shows the confidence, and the note cites snippet [1], shown below with its source." |
| 0:45–1:15 | Click **Schools closed for a month (EN)** | "Here the evidence says schools moved online for one day only, so the claim is Contradicted, and the note quotes that sentence." |
| 1:15–1:45 | Click **Metro stopped by floods (mixed)** | "Messages in the UAE often mix Arabic and English. The claim is in Arabic with English words, the evidence is English, and the verdict and note still work; the note is written in Arabic, right to left." |
| 1:45–2:05 | Click **Crewed Mars trip (AR)** | "When the evidence does not talk about the claim, TwinCheck says Unverified instead of guessing." |
| 2:05–2:25 | Clear the evidence box, type a real claim from the MultiFC test set, click **Check** | "With no evidence pasted, it searches its offline library of about 160 thousand snippets and picks the best five." |
| 2:25–2:40 | Scroll to the components line | "Every part, from claim extraction to the cited note, is in our GitHub repository, with the models on Hugging Face." |

Before recording: run step 3 of the R6 notebook and check `results/R6/examples.json`;
only show examples whose verdict matches the expected one, or say honestly when it does not.
