# src/data: datasets in the shared record format

| File | Owner | What it does |
|---|---|---|
| `download_en.py` | Rathi (R1) | downloads MultiFC (authors' release with snippets) and AVeriTeC to `data/raw/` |
| `load_multifc.py`, `load_averitec.py` | Rathi (R1) | raw files → shared JSONL format, labels via `data/label_map.csv`; own 80/10/10 MultiFC split (seed 42) |
| `clean.py` | Rathi (R1) | the 5 cleaning rules of CLAUDE.md §4 for any dataset in the shared format; per-rule counts in a report |
| `common.py` | Rathi (R1) | shared helpers: JSONL I/O, building a record, reading the label map, URL domain, stratified split |
| `entities.py` | Rathi (R2) | spaCy entities (PERSON, GPE, LOC, ORG, DATE, EVENT) with character offsets |
| `run_r1.sh` | Rathi (R1) | download → load → clean → tests, in one command |
| `load_xfact.py`, `load_m4fc.py`, `download_m4fc_images.py` | Youssef (Y1) | Arabic data (X-FACT Arabic, M4FC Arabic + images) |

Run: `bash src/data/run_r1.sh` (English). Outputs: `data/processed/{en,ar}/`. Tests: `tests/data/`.
