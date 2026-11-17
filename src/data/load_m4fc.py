"""Convert the Arabic claims of M4FC to the TwinCheck shared record format (task Y1).

What it does
    Reads M4FC.json (GitHub UKPLab/M4FC, CC-BY-SA-4.0), keeps rows whose
    claim_language is Arabic, and writes them as TEST data only (agreed: nobody
    trains on M4FC, so all 614 Arabic rows are used for testing; the official M4FC
    split is not needed).
    - The Arabic claim (`multilingual_claim`) becomes `claim`; the fact-checker's
      verdict gives the label.
    - M4FC is almost all false claims. Following M4FC's own balanced setting, rows
      with use_true_caption == True also give a SUPPORTED claim: the Arabic true
      caption written for the same image (id m4fc_tc_000123).
    - M4FC has no evidence text, so the main file has evidence = [].
      A second file adds M4FC's context items (source, date, location, event ...)
      as one English snippet with domain "m4fc_context". This context comes from
      the fact-check article, so it is near-gold evidence and must be reported that way.

Inputs
    --json        path to M4FC.json
    --image_dir   folder used by download_m4fc_images.py (for image_path)
    --limit       only the first N Arabic rows (use 10 for the test run)

Outputs (in --out_dir)
    m4fc_ar_test.jsonl        main file, evidence = []
    m4fc_ar_test_ctx.jsonl    same rows, evidence = M4FC context snippet
    m4fc_ar_claims_en.csv     id, English claim (for cross-lingual checks in Y4/Y5)

How to run
    git clone --depth 1 https://github.com/UKPLab/M4FC data/raw/M4FC
    python src/data/load_m4fc.py --json data/raw/M4FC/data/M4FC.json \
        --out_dir data/processed/ar/uncleaned --limit 10
"""

import argparse
import csv
import json
import os
from collections import Counter

MISSING = "not enough information"  # M4FC's value for empty fields
LABEL_MAP = {  # M4FC fine verdict -> team label (same rows go in data/label_map.csv)
    "true": "supported",
    "false (out-of-context)": "contradicted",
    "false (manipulated/fake)": "contradicted",
    "false (ai-generated)": "contradicted",
    "not enough information": "unverified",
}
CONTEXT_FIELDS = ["source", "date", "location", "event", "motivation"]


def read_m4fc_arabic(json_path, limit=None):
    """Load M4FC.json and return (idx, row) pairs for Arabic claims.

    idx is the row's position in M4FC.json; it is used in the id so ids never change
    and match the image file names from download_m4fc_images.py.
    """
    rows = json.load(open(json_path, encoding="utf-8"))
    arabic = [(i, r) for i, r in enumerate(rows) if r.get("claim_language") == "Arabic"]
    return arabic[:limit] if limit else arabic


def image_path_for(idx, image_dir):
    """Return the image path Feba's folder uses for M4FC row idx."""
    return f"{image_dir.rstrip('/')}/m4fc_{idx:06d}.jpg"


def context_evidence(row):
    """Build one evidence snippet from M4FC's context items, or [] if all are missing."""
    parts = [f"{k.capitalize()}: {row[k]}" for k in CONTEXT_FIELDS
             if isinstance(row.get(k), str) and row[k].strip() and row[k] != MISSING]
    if not parts:
        return []
    return [{"text": ". ".join(parts) + ".", "url": row.get("article_url"), "domain": "m4fc_context"}]


def map_label(verdict):
    """Map an M4FC fine verdict to supported/contradicted/unverified (None if unknown)."""
    verdict = verdict.strip().lower()
    if verdict in LABEL_MAP:
        return LABEL_MAP[verdict]
    return "contradicted" if verdict.startswith("false") else None


def row_to_record(idx, row, image_dir):
    """Convert one Arabic M4FC row to a shared-format record (or None if the label is unknown)."""
    label = map_label(row["verdict"])
    if label is None:
        return None
    return {
        "id": f"m4fc_{idx:06d}",
        "source": "m4fc",
        "split": "test",
        "language": "ar",
        "claim": row["multilingual_claim"].strip(),
        "evidence": [],
        "image_path": image_path_for(idx, image_dir),
        "label": label,
        "orig_label": row["verdict"].strip().lower(),
    }


def true_caption_record(idx, row, image_dir):
    """Make a SUPPORTED record from the Arabic true caption (M4FC balanced setting), or None."""
    caption = row.get("multilingual_true_caption", MISSING)
    if not row.get("use_true_caption") or not caption or caption == MISSING:
        return None
    return {
        "id": f"m4fc_tc_{idx:06d}",
        "source": "m4fc",
        "split": "test",
        "language": "ar",
        "claim": caption.strip(),
        "evidence": [],
        "image_path": image_path_for(idx, image_dir),
        "label": "supported",
        "orig_label": "true_caption",
    }


def write_jsonl(path, records):
    """Write records as JSON Lines with Arabic kept readable (ensure_ascii=False)."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    """Convert Arabic M4FC rows, write the three output files and print counts."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True)
    ap.add_argument("--image_dir", default="data/processed/vision/m4fc_images")
    ap.add_argument("--out_dir", default="data/processed/ar/uncleaned")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    main_recs, ctx_recs, en_rows, skipped = [], [], [], 0
    for idx, row in read_m4fc_arabic(args.json, args.limit):
        if map_label(row["verdict"]) is None:
            skipped += 1
        # (record, English version of the same claim) for the claim and its true caption
        pairs = [(row_to_record(idx, row, args.image_dir), row["claim"]),
                 (true_caption_record(idx, row, args.image_dir), row.get("true_caption", MISSING))]
        for rec, claim_en in pairs:
            if rec is None:
                continue
            main_recs.append(rec)
            ctx_recs.append({**rec, "evidence": context_evidence(row)})
            en_rows.append({"id": rec["id"], "claim_en": claim_en})

    write_jsonl(os.path.join(args.out_dir, "m4fc_ar_test.jsonl"), main_recs)
    write_jsonl(os.path.join(args.out_dir, "m4fc_ar_test_ctx.jsonl"), ctx_recs)
    with open(os.path.join(args.out_dir, "m4fc_ar_claims_en.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "claim_en"])
        w.writeheader()
        w.writerows(en_rows)

    labels = Counter(r["label"] for r in main_recs)
    no_ctx = sum(1 for r in ctx_recs if not r["evidence"])
    print(f"m4fc arabic: records={len(main_recs)} {dict(labels)} skipped_unknown_label={skipped} "
          f"no_context_evidence={no_ctx} -> {args.out_dir}")


if __name__ == "__main__":
    main()
