"""Draft WhatsApp-style claim-extraction examples from real claims (task Y3, part 1).

What it does
    Builds messages that look like forwarded WhatsApp posts, each hiding 1-3 REAL
    claims taken from our data, and records those claims as the gold answer.
    - English claims: MultiFC train claims.
    - Arabic claims: X-FACT Arabic train claims.
    - Mixed messages: Arabic chat around English claims, English chat around an
      Arabic claim, or one Arabic and one English claim in the same message.
    The chat text around the claims (greetings, "please share", emojis, "they say
    that ...") is template-based with random choices (seed 42). Gold claims are the
    original claim sentences, in their own language. These are DRAFTS: Youssef
    checks every one (CLAUDE.md: "LLM can draft, you check every one").

Inputs
    --multifc   data/processed/en/multifc_train.jsonl
    --xfact_ar  data/processed/ar/xfact_ar_train.jsonl

Outputs
    data/claims/gold_draft_<start>.jsonl   {"id", "language", "message", "claims", "source_ids"}
    results/Y3/review_<start>.csv          one row per example, columns ok / fix for checking

How to run (repo root)
    python -m src.claims.draft_gold --n 10              # look at 10 first
    python -m src.claims.draft_gold --n 100 --start 1   # claims_0001 .. claims_0100
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re

from src.data.common import SEED, read_jsonl, write_jsonl

WORD = re.compile(r"\w+", re.UNICODE)

EN = {
    "open": ["Forwarded as received:", "Guys did you see this??", "BREAKING 🚨", "My cousin just sent this 👇",
             "Not sure if true but", "Share before they delete it!!", "Good morning everyone 🌸", "",
             "Saw this on Facebook", "URGENT please read"],
    "frame": ["{c}", "{c}", "Apparently: {c}", "They're saying: {c}", "{c} 😳", "News says: {c}"],
    "join": ["Also:", "And apparently:", "On top of that:", "Plus:", "", "Another thing:"],
    "close": ["Please share 🙏", "Is this true?", "Forward to all groups", "😱😱", "", "Stay safe everyone",
              "Can someone confirm?"],
}
AR = {
    "open": ["منقول كما وصلني:", "يا جماعة شفتوا هذا؟", "عاجل 🚨", "أرسلها لي أخوي 👇", "ما أدري إذا صحيح بس",
             "انشروا قبل الحذف!!", "صباح الخير 🌸", "", "شفت هذا في الفيسبوك", "مهم جداً اقرأوا"],
    "frame": ["{c}", "{c}", "يقولون إن {c}", "الخبر يقول إن {c}", "{c} 😳", "انتشر أن {c}"],
    "join": ["وأيضاً", "وكمان يقولون", "وفوق هذا", "", "وشي ثاني:", "والأغرب"],
    "close": ["انشروها 🙏", "هل هذا صحيح؟", "أرسلوها لكل القروبات", "😱😱", "", "الله يستر", "أحد يأكد؟"],
}


def usable(text: str, lo: int = 5, hi: int = 25) -> bool:
    """True for a clean, short, single statement (5-25 words; no links, questions,
    'Says ...', 'FACT CHECK' headlines or unbalanced quotes)."""
    n = len(WORD.findall(text))
    return (lo <= n <= hi and "http" not in text and "not enough information" not in text
            and not text.lower().startswith(("say", "claim")) and "\n" not in text
            and "?" not in text and "؟" not in text and "fact check" not in text.lower()
            and text.count('"') % 2 == 0)


def tidy(text: str) -> str:
    """Strip spaces and wrapping quotes from a claim."""
    return text.strip().strip('"“”«»\'').strip()


def load_pools(multifc_path: str, xfact_path: str) -> dict:
    """Return {"en": [(id, claim)], "ar": [(id, claim)]} of usable real claims.

    Only fact-checked claims (MultiFC, X-FACT) are used: they are full sentences,
    unlike M4FC's image captions, which are often fragments.
    """
    en = [(r["id"], tidy(r["claim"])) for r in read_jsonl(multifc_path) if usable(tidy(r["claim"]))]
    ar = [(r["id"], tidy(r["claim"])) for r in read_jsonl(xfact_path) if usable(tidy(r["claim"]))]
    return {"en": en, "ar": ar}


def framed(claim: str, words: dict, rng: random.Random) -> str:
    """Put one claim into chat wording ('Apparently: ...'); the gold claim stays unchanged."""
    c = claim.rstrip(".")
    return rng.choice(words["frame"]).format(c=c)


def build_message(parts: list[tuple[str, dict]], outer: dict, rng: random.Random) -> str:
    """Join opener, framed claims (each with its own language's wording) and closer."""
    lines = [rng.choice(outer["open"])]
    for k, (claim, words) in enumerate(parts):
        if k > 0:  # joined claims get a connector instead of a second frame
            piece = f"{rng.choice(words['join'])} {claim.rstrip('.')}".strip()
        else:
            piece = framed(claim, words, rng)
        lines.append(piece)
    lines.append(rng.choice(outer["close"]))
    sep = rng.choice(["\n", " ", "\n\n"])
    return sep.join(x for x in lines if x).strip()


def make_example(number: int, language: str, pools: dict, rng: random.Random, used: set) -> dict:
    """Draft one example in the given language ('en', 'ar' or 'mixed')."""
    k = rng.choices([1, 2, 3], weights=[35, 40, 25])[0]

    def take(lang):
        while True:
            sid, claim = rng.choice(pools[lang])
            if sid not in used:
                used.add(sid)
                return sid, claim

    if language == "mixed":
        style = rng.choice(["ar_chat_en_claims", "en_chat_ar_claim", "one_each"])
        if style == "ar_chat_en_claims":
            picks, outer = [take("en") for _ in range(k)], AR
            words = [EN] * len(picks)  # English claims keep English framing ("apparently ...")
        elif style == "en_chat_ar_claim":
            picks, outer, words = [take("ar")], EN, [AR]
        else:
            picks, outer, words = [take("ar"), take("en")], rng.choice([AR, EN]), [AR, EN]
    else:
        picks = [take(language) for _ in range(k)]
        outer = EN if language == "en" else AR
        words = [outer] * len(picks)
    message = build_message([(c, w) for (_, c), w in zip(picks, words)], outer, rng)
    return {"id": f"claims_{number:04d}", "language": language, "message": message,
            "claims": [c for _, c in picks], "source_ids": [s for s, _ in picks]}


def main() -> None:
    """Draft n examples (balanced en / ar / mixed) and write the draft file + review sheet."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--multifc", default="data/processed/en/multifc_train.jsonl")
    ap.add_argument("--xfact_ar", default="data/processed/ar/xfact_ar_train.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rng = random.Random(args.seed + args.start)  # different parts get different claims
    pools = load_pools(args.multifc, args.xfact_ar)
    langs = (["en", "ar", "mixed"] * args.n)[: args.n]
    used: set = set()
    rows = [make_example(args.start + i, lang, pools, rng, used) for i, lang in enumerate(langs)]

    out = f"data/claims/gold_draft_{args.start:04d}.jsonl"
    write_jsonl(rows, out)
    sheet = f"results/Y3/review_{args.start:04d}.csv"
    os.makedirs(os.path.dirname(sheet), exist_ok=True)
    with open(sheet, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "language", "message", "claim_1", "claim_2", "claim_3", "ok", "fix"])
        for r in rows:
            cl = r["claims"] + [""] * (3 - len(r["claims"]))
            w.writerow([r["id"], r["language"], r["message"], *cl, "", ""])
    print(f"wrote {len(rows)} drafts -> {out}, review sheet -> {sheet}")


if __name__ == "__main__":
    main()
