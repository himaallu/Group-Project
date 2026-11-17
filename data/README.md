# data/

Data is shared through git (there is no shared Drive). Pull the branch that produced a
file, or `main` once its PR is merged.

| Path | Contents | Owner | In git? |
|---|---|---|---|
| `data/raw/` | original downloads (never edited) | Rathi, Youssef, Feba | no: rebuild with `src/data/download_*.py` |
| `data/processed/{en,ar,vision}/` | cleaned data in the shared record format (`*_{train,val,test}.jsonl`) | Rathi (en), Youssef (ar), Feba (vision) | yes |
| `data/copies/{en,ar,vision}/` | twins (`_P _A _M _B _C _V`), `sources.csv` | Rathi, Youssef, Feba | yes |
| `data/nli/` | NLI training pairs + `gold_val.jsonl` | Rathi, Himasri, Youssef | yes |
| `data/claims/` | claim-extraction gold (`gold.jsonl`) | Youssef | yes |
| `data/label_map.csv` | site label → supported / contradicted / unverified | Rathi (+ Youssef for Arabic) | yes |

GitHub refuses single files over 100 MB: split large files or track them with Git LFS.

Check a file against the shared format:

```bash
python -m src.eval.loaders --check data/processed/en/multifc_train.jsonl
```
