# data/ (not in git)

All data lives in the shared Drive folder `TwinCheck/` with the same paths.
In Colab: `DATA = '/content/drive/MyDrive/TwinCheck'`.

| Path | Contents | Owner |
|---|---|---|
| `data/raw/` | original downloads (never edited) | Rathi, Youssef, Feba |
| `data/processed/{en,ar,vision}/` | cleaned data in the shared record format (`*_{train,val,test}.jsonl`) | Rathi (en), Youssef (ar), Feba (vision) |
| `data/copies/{en,ar,vision}/` | twins (`_P _A _M _B _C _V`), `sources.csv` | Rathi, Youssef, Feba |
| `data/nli/` | NLI training pairs + `gold_val.jsonl` | Rathi, Himasri, Youssef |
| `data/claims/` | claim-extraction gold (`gold.jsonl`) | Youssef |
| `data/label_map.csv` | site label → supported / contradicted / unverified | Rathi (+ Youssef for Arabic) |

Check a file against the shared format:

```bash
python -m src.eval.loaders --check data/processed/en/multifc_train.jsonl
```
