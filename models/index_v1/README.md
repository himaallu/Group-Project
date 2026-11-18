# index_v1 (R5)

Offline evidence library on Hugging Face Hub: https://huggingface.co/datasets/RathiAbeth/twincheck-index-v1 (embeddings too big for git).

Get it: `python -m src.retrieval.build_index download --repo RathiAbeth/twincheck-index-v1`; then `from src.retrieval.index import search`.

163244 unique snippets from 24945 claims, encoder `BAAI/bge-m3`.
