"""Make the Arabic twins _A, _M, _B_A, _C_A for 200 English test sources (task Y4).

What it does
    1. Sources: 200 of Rathi's 500 test sources, picked with seed 42 among those that
       have an `_B` twin (so every source gets all four Arabic twins).
    2. Glossary: names (people, places, organisations, groups) found by spaCy in the
       source claims, their snippets and the `_C` donor snippets, plus the swapped-in
       names of `_B`, are translated once each. Every later translation is told to
       use these spellings, so a name is written the same way in all twins.
    3. Translation EN -> Modern Standard Arabic with Qwen3-8B (4-bit, greedy, thinking
       off), the same prompt as src/translate/translate_pairs.py. Each distinct text
       is translated once and cached (resumable after a Colab disconnect). Outputs
       that drift into another script or stay mostly English are retried once with
       sampling; twins whose text still fails are dropped and counted.
    4. Twins (CLAUDE.md section 4; swaps were done in English before translation):
         {id}_A    claim + snippets in Arabic             label = original (supported)
         {id}_M    Arabic claim, English snippets         label = original
         {id}_B_A  Arabic version of Rathi's _B           label = contradicted
         {id}_C_A  Arabic version of Rathi's _C           label = unverified
    5. A sheet of 50 random translations (25 claims, 25 snippets, seed 42) for
       Youssef to check by hand.

Inputs
    data/copies/en/test.jsonl, data/copies/en/twin_log.csv (R3),
    data/processed/en/multifc_test.jsonl, features/entities_en.jsonl

Outputs
    data/copies/ar/test.jsonl            800 Arabic / mixed twins, shared format
    data/translate/glossary_en_ar.csv    name glossary (en,ar)
    data/copies/ar/translation_cache.jsonl
    results/Y4/report.json               counts, dropped twins, timings
    results/Y4/check_50.csv              50 translations to check (ok / note columns)

How to run (repo root, GPU runtime)
    python -m src.copies.arabic_twins --n 5      # quick test on 5 sources
    python -m src.copies.arabic_twins            # all 200 (~1.5 h on a T4)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from collections import Counter

from src.data.common import SEED, read_jsonl, write_jsonl
from src.translate.translate_pairs import PROMPT, good_translation

N_SOURCES = 200
NAME_LABELS = {"PERSON", "GPE", "LOC", "ORG", "NORP", "FAC", "EVENT"}
MAX_GLOSSARY_LINES = 15
CACHE = "data/copies/ar/translation_cache.jsonl"
GLOSSARY = "data/translate/glossary_en_ar.csv"
NAME_PROMPT = ("Write this name in Arabic, as Arabic news media write it. "
               "Output only the Arabic name.\nName: {text}")


# ---------------------------------------------------------------- selection

def pick_sources(twins: list[dict], n: int, seed: int = SEED) -> list[str]:
    """Return n source ids (sorted) among the test sources that have an _B twin."""
    with_b = sorted(t["id"][:-2] for t in twins if t["id"].endswith("_B"))
    return sorted(random.Random(seed).sample(with_b, min(n, len(with_b))))


def donor_ids(log_path: str) -> dict:
    """{source id: donor id} for the _C twins, from R3's twin_log.csv."""
    with open(log_path, encoding="utf-8") as f:
        return {r["id"][:-2]: r["donor_id"] for r in csv.DictReader(f) if r["type"] == "C"}


def swapped_names(log_path: str, sources: list[str]) -> list[str]:
    """Names that the sources' _B twins swapped in (they need a glossary entry too)."""
    want = {s + "_B" for s in sources}
    with open(log_path, encoding="utf-8") as f:
        return [r["new"] for r in csv.DictReader(f)
                if r["id"] in want and r["kind"] in ("place", "person") and r["new"]]


def collect_names(ids: list[str], entities: dict) -> list[str]:
    """Distinct names (spaCy NAME_LABELS) in the claims and snippets of the given ids."""
    names = set()
    for i in ids:
        e = entities.get(i)
        if not e:
            continue
        for ent in e["claim_entities"] + [x for s in e["snippet_entities"] for x in s]:
            if ent["label"] in NAME_LABELS:
                name = clean_name(ent["text"])
                if name:
                    names.add(name)
    return sorted(names)


def clean_name(text: str) -> str:
    """Tidy a spaCy name ('... County', '.Oklahoma', 'Cramer’s'); '' if it is not a usable name."""
    name = re.sub(r"^[^\w]+|[^\w]+$", "", text.strip())
    name = re.sub(r"[’']s$", "", name)
    if len(name) < 2 or '"' in name or "..." in name or not name[0].isalpha():
        return ""
    return name


# ---------------------------------------------------------------- translation

def glossary_for(text: str, glossary: dict) -> str:
    """Prompt lines for the glossary names that occur in text (longest first)."""
    hits = [en for en in glossary if re.search(rf"(?<!\w){re.escape(en)}(?!\w)", text)]
    hits = sorted(hits, key=len, reverse=True)[:MAX_GLOSSARY_LINES]
    return ("Use these spellings:\n" + "\n".join(f"{en} -> {glossary[en]}" for en in hits) + "\n") if hits else ""


def load_cache(path: str) -> dict:
    """{english text: arabic text} already translated."""
    if not os.path.exists(path):
        return {}
    return {r["en"]: r["ar"] for r in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())}


class Translator:
    """Qwen3-8B 4-bit translator (loaded once). generate(prompts, sample) -> list[str]."""

    def __init__(self, model_name: str = "Qwen/Qwen3-8B"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16))

    def generate(self, prompts: list[str], sample: bool = False, max_new_tokens: int = 400) -> list[str]:
        chats = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False) for p in prompts]
        enc = self.tok(chats, return_tensors="pt", padding=True).to(self.model.device)
        kw = dict(do_sample=True, temperature=0.7, top_p=0.9) if sample else dict(do_sample=False)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, **kw)
        return [re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()
                for t in self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]


def is_good(en: str, ar: str) -> bool:
    """Same check as translate_pairs.merge, applied to one text."""
    return bool(ar) and good_translation({"claim": ar, "snippet": ar})


def translate_names(names: list[str], gen, batch: int, path: str) -> dict:
    """Translate each name once; reuse the glossary file if it exists (resumable)."""
    gloss = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            gloss = {r["en"]: r["ar"] for r in csv.DictReader(f)}
    todo = [n for n in names if n not in gloss]
    print(f"glossary: {len(names)} names, {len(todo)} to translate")
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        for en, ar in zip(chunk, gen([NAME_PROMPT.format(text=n) for n in chunk])):
            ar = ar.splitlines()[0].strip() if ar else ""
            if ar and len(ar) < 4 * len(en) + 10 and is_good(en, ar):
                gloss[en] = ar
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["en", "ar"])
            w.writerows(sorted(gloss.items()))
        print(f"  names {min(i + batch, len(todo))}/{len(todo)}")
    return gloss


def translate_texts(texts: list[str], glossary: dict, gen, batch: int, cache_path: str) -> tuple[dict, list]:
    """Translate distinct texts (cached, resumable). Returns ({en: ar}, failed texts)."""
    cache = load_cache(cache_path)
    todo = sorted({t for t in texts if t not in cache}, key=len)  # similar lengths batch well
    print(f"texts: {len(set(texts))} distinct, {len(todo)} to translate")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    start = time.time()
    with open(cache_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            prompts = [PROMPT.format(glossary=glossary_for(t, glossary), text=t) for t in chunk]
            outs = gen(prompts)
            retry = [k for k, (en, ar) in enumerate(zip(chunk, outs)) if not is_good(en, ar)]
            if retry:  # one more try with sampling for drifted outputs
                again = gen([prompts[k] for k in retry], sample=True)
                for k, ar in zip(retry, again):
                    if is_good(chunk[k], ar):
                        outs[k] = ar
            for en, ar in zip(chunk, outs):
                cache[en] = ar
                f.write(json.dumps({"en": en, "ar": ar}, ensure_ascii=False) + "\n")
            f.flush()
            done = min(i + batch, len(todo))
            print(f"  texts {done}/{len(todo)}  ({(time.time() - start) / 60:.1f} min)")
    failed = [t for t in set(texts) if not is_good(t, cache.get(t, ""))]
    return cache, failed


# ---------------------------------------------------------------- twins

def texts_of(rec: dict) -> list[str]:
    """Claim + snippet texts of a record."""
    return [rec["claim"]] + [s["text"] for s in rec.get("evidence") or []]


def arabic(rec: dict, new_id: str, tr: dict, language: str, translate_evidence: bool) -> dict:
    """Copy of rec with an Arabic claim (and Arabic snippets if translate_evidence)."""
    ev = [{**s, "text": tr[s["text"]]} if translate_evidence else dict(s) for s in rec.get("evidence") or []]
    return {**rec, "id": new_id, "language": language, "claim": tr[rec["claim"]], "evidence": ev}


def build_twins(sources: list[str], originals: dict, en_twins: dict, tr: dict, failed: set) -> tuple[list, Counter]:
    """Make _A, _M, _B_A, _C_A for each source; drop a twin if any of its texts failed."""
    out, dropped = [], Counter()
    for sid in sources:
        orig, b, c = originals[sid], en_twins[sid + "_B"], en_twins[sid + "_C"]
        plan = [("A", orig, "ar", True), ("M", orig, "mixed", False),
                ("B_A", b, "ar", True), ("C_A", c, "ar", True)]
        for suffix, rec, lang, full in plan:
            needed = texts_of(rec) if full else [rec["claim"]]
            if any(t in failed for t in needed):
                dropped[suffix] += 1
                continue
            out.append(arabic(rec, f"{sid}_{suffix}", tr, lang, full))
    return out, dropped


def write_check_sheet(sources: list[str], originals: dict, tr: dict, path: str, seed: int = SEED) -> None:
    """50 random translations (25 claims, 25 snippets) with empty ok / note columns."""
    rng = random.Random(seed)
    claims = [originals[s]["claim"] for s in sources]
    snippets = sorted({s["text"] for sid in sources for s in originals[sid]["evidence"]})
    rows = [("claim", t) for t in rng.sample(claims, min(25, len(claims)))] + \
           [("snippet", t) for t in rng.sample(snippets, min(25, len(snippets)))]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "english", "arabic", "ok", "note"])
        for kind, en in rows:
            w.writerow([kind, en, tr.get(en, ""), "", ""])


def main() -> None:
    """Select sources, build the glossary, translate, write twins, report and check sheet."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--twins", default="data/copies/en/test.jsonl")
    ap.add_argument("--log", default="data/copies/en/twin_log.csv")
    ap.add_argument("--originals", default="data/processed/en/multifc_test.jsonl")
    ap.add_argument("--entities", default="features/entities_en.jsonl")
    ap.add_argument("--out", default="data/copies/ar/test.jsonl")
    ap.add_argument("--n", type=int, default=N_SOURCES)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    t0 = time.time()
    en_twins = {t["id"]: t for t in read_jsonl(args.twins)}
    sources = pick_sources(list(en_twins.values()), args.n)
    originals = {r["id"]: r for r in read_jsonl(args.originals)}
    donors = donor_ids(args.log)
    entities = {r["id"]: r for r in read_jsonl(args.entities)}
    names = collect_names(sources + [donors[s] for s in sources], entities)
    names = sorted(set(names) | set(swapped_names(args.log, sources)))

    model = Translator()
    glossary = translate_names(names, model.generate, args.batch * 2, GLOSSARY)

    texts = []
    for sid in sources:
        for rec in (originals[sid], en_twins[sid + "_B"], en_twins[sid + "_C"]):
            texts += texts_of(rec)
    tr, failed = translate_texts(texts, glossary, model.generate, args.batch, CACHE)

    twins, dropped = build_twins(sources, originals, en_twins, tr, set(failed))
    write_jsonl(twins, args.out)
    write_check_sheet(sources, originals, tr, "results/Y4/check_50.csv")
    report = {"sources": len(sources), "twins": len(twins),
              "by_type": dict(Counter(t["id"].split("_", 2)[2] for t in twins)),
              "by_label": dict(Counter(t["label"] for t in twins)),
              "dropped_twins": dict(dropped), "failed_texts": len(failed),
              "distinct_texts": len(set(texts)), "glossary_names": len(glossary),
              "minutes": round((time.time() - t0) / 60, 1), "model": "Qwen/Qwen3-8B 4-bit", "seed": SEED}
    os.makedirs("results/Y4", exist_ok=True)
    json.dump(report, open("results/Y4/report.json", "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
