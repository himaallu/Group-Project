# reranker_v1 (R2)

Weights are on Hugging Face Hub: https://huggingface.co/RathiAbeth/twincheck-reranker-v1 (too big for git).

Load: `from src.retrieval.rerank import Reranker; rr = Reranker('RathiAbeth/twincheck-reranker-v1')`

Base `BAAI/bge-reranker-v2-m3`, 1 epoch, 20000 pairs, lr 2e-05, seed 42.
Val AUROC 0.980 -> 0.998. W&B: https://forge.coreweave.com/wandb/rathiabeth-uowd/twincheck/runs/yftu2n5c
