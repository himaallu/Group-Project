"""Faithfulness and citation accuracy of the cited notes (R5, CLAUDE.md §5).

What it does
    For every note in results/R5/notes.jsonl:
      1. splits it into sentences and reads each sentence's citations [k];
      2. runs the fine-tuned NLI model (R4) with premise = cited snippet k and
         hypothesis = the sentence without its citation marks;
      3. a citation "supports" its sentence if p_entail > 0.5; a sentence is
         faithful if at least one of its citations supports it (a sentence with no
         citation is not faithful).
    Then calls Himasri's ``metrics.faithfulness`` and ``metrics.citation_accuracy``
    (per language and overall), and saves 5 good notes (every sentence faithful) and
    5 bad notes (at least one sentence not faithful), seed 42.

Inputs
    --notes results/R5/notes.jsonl (from src/explain/generate.py)
    --model NLI model (default: src.nli.score.model_name(), zero-shot until R4 v2 beats it)

Outputs
    results/R5/explain_eval.json, results/R5/notes_scored.jsonl,
    results/R5/notes_good.jsonl, results/R5/notes_bad.jsonl

How to run (Colab GPU, repo root)
    python -m src.explain.faithfulness --notes results/R5/test10/notes.jsonl \
        --out-dir results/R5/test10                         # 10-row check
    python -m src.explain.faithfulness
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.eval.metrics import citation_accuracy, faithfulness
from src.explain.generate import CITATION_RE, split_sentences

ENTAIL_MIN = 0.5


def sentence_citations(sentence: str) -> tuple[str, list[int]]:
    """(sentence without citation marks, cited snippet numbers in order, no repeats)."""
    cited = list(dict.fromkeys(int(k) for k in CITATION_RE.findall(sentence)))
    text = CITATION_RE.sub("", sentence)
    return " ".join(text.split()).replace(" .", "."), cited


def judge_note(note: str, snippets: list[dict], score_pairs) -> dict:
    """Check every sentence of one note against the snippets it cites.

    Args:
        note: the explanation text.
        snippets: the numbered snippets ([1] = snippets[0]).
        score_pairs: function(claims, snippets) -> [{"p_entail", ...}] (premise = snippet).

    Returns:
        {"sentences": [{"text", "cited", "p_entail": [...], "faithful"}],
         "sentence_entailed": [bool], "citation_supports": [bool]}
    """
    rows, hyps, prems = [], [], []
    for sentence in split_sentences(note):
        text, cited = sentence_citations(sentence)
        valid = [k for k in cited if 1 <= k <= len(snippets)]
        rows.append({"text": text, "cited": valid})
        for k in valid:
            hyps.append(text)
            prems.append(snippets[k - 1]["text"])
    scores = iter(score_pairs(hyps, prems) if hyps else [])
    for row in rows:
        row["p_entail"] = [round(next(scores)["p_entail"], 4) for _ in row["cited"]]
        row["faithful"] = any(p > ENTAIL_MIN for p in row["p_entail"])
    return {"sentences": rows,
            "sentence_entailed": [r["faithful"] for r in rows],
            "citation_supports": [p > ENTAIL_MIN for r in rows for p in r["p_entail"]]}


def summarise(scored: list[dict]) -> dict:
    """Faithfulness and citation accuracy overall and per language."""
    groups = {"all": scored}
    for row in scored:
        groups.setdefault(row["language"], []).append(row)
    out = {}
    for name, rows in groups.items():
        out[name] = {
            "n_notes": len(rows),
            "n_sentences": sum(len(r["sentence_entailed"]) for r in rows),
            "n_citations": sum(len(r["citation_supports"]) for r in rows),
            "faithfulness": faithfulness([r["sentence_entailed"] for r in rows]),
            "citation_accuracy": citation_accuracy(
                [c for r in rows for c in r["citation_supports"]]),
            "share_notes_fully_faithful": sum(all(r["sentence_entailed"]) and
                                              bool(r["sentence_entailed"]) for r in rows)
            / len(rows) if rows else None,
        }
    return out


def pick_examples(scored: list[dict], n: int = 5, seed: int = 42) -> tuple[list, list]:
    """``n`` good notes (all sentences faithful) and ``n`` bad ones, chosen with ``seed``."""
    rng = random.Random(seed)
    good = [r for r in scored if r["sentence_entailed"] and all(r["sentence_entailed"])]
    bad = [r for r in scored if not (r["sentence_entailed"] and all(r["sentence_entailed"]))]
    rng.shuffle(good)
    rng.shuffle(bad)
    return good[:n], bad[:n]


def write_jsonl(rows: list[dict], path: Path) -> None:
    """Write rows as UTF-8 JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


def main() -> None:
    """CLI: score every note and write the evaluation files."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--notes", default="results/R5/notes.jsonl")
    ap.add_argument("--model", help="NLI model (default: fine-tuned R4 model)")
    ap.add_argument("--out-dir", default="results/R5")
    args = ap.parse_args()

    from src.eval.loaders import load_jsonl
    from src.nli.score import model_name, score_pairs

    model = args.model or model_name()
    scored = []
    for row in load_jsonl(args.notes):
        result = judge_note(row["note"], row["snippets"],
                            lambda c, s: score_pairs(c, s, model=model))
        scored.append({**row, **result})
    report = {"nli_model": model, "entail_min": ENTAIL_MIN,
              "verdict_explained": "gold label", **summarise(scored)}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "explain_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_jsonl(scored, out / "notes_scored.jsonl")
    good, bad = pick_examples(scored)
    write_jsonl(good, out / "notes_good.jsonl")
    write_jsonl(bad, out / "notes_bad.jsonl")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
