# Labelling guide: NLI gold pairs (H2 English, Y2 Arabic)

**Purpose.** These pairs are the answer key for the NLI model (pipeline step 5). We
label the relationship between ONE snippet and the claim, not whether the claim is
true in the world. The claim's verdict is hidden on purpose.

**Labels** (type `e` / `c` / `n` in the `label` column)

| Label | When |
|---|---|
| `e` entail | if the snippet is true, the claim is true |
| `c` contradict | if the snippet is true, the claim is false (states something incompatible) |
| `n` neutral | unrelated, same topic only, too vague, or cut off before the deciding part |

**Procedure per row.** (1) Find the claim's core: who / what / where / when / how
many. (2) Read the snippet. (3) Ask: "if everything in this snippet is true, what
does it say about the claim?" (4) Unsure after ~20 s: best guess + note `unsure`.

**Rules**
- Snippet only repeats or asks the claim ("Did X…?", "Claim: X") → `n`.
- Snippet reports the claim as fact → `e`; hedged ("reportedly", "may") → `n`.
- Claim "Y happened", snippet "Z said Y happened" → `n`; claim "Z said Y" → `e`.
- Two-part claim, snippet covers one part → `n`, note `partial`.
- Rounded numbers match → `e`; clearly different number/date/place → `c`.
- Different event at a different time is not a contradiction → `n`.
- Leaked fact-check verdict ("FALSE: …") → label what it says, note `factcheck`.
- Junk (`<S> <UNK> …`), bare titles, link lists → `n`, note `junk`.
- Satire / opinion → `n` unless it plainly states a checkable fact.
- Never use outside knowledge: if the snippet does not say it, it is `n`.

**Notes codes:** `unsure`, `partial`, `factcheck`, `junk`.
**Expect** mostly `n` (often 50–70%); do not force balance. Re-check only `unsure`
rows at the end. Record time taken and number of `unsure` rows for the methods section.

**Example** (illustrative). Claim: "The Burj Khalifa was opened in 2010."
"...officially opened tonight [Jan 4, 2010]" → `e`; "...opened in 2008" → `c`;
"828 metres tall" → `n`; "When did it open? Find out…" → `n`.
