# nli_v1 (R4)

Weights are on Hugging Face Hub: https://huggingface.co/RathiAbeth/twincheck-nli-v1 (too big for git).

Load: `from src.nli.score import score_pairs` (finds the Hub copy by itself), or `python -m src.nli.score FILE --model RathiAbeth/twincheck-nli-v1`.

Base `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, 2 epochs, 32658 pairs, lr 2e-05, seed 42, fp32 (tf32). Gold macro-F1 EN 0.3711 -> 0.1236, AR 0.4427 -> 0.1617. W&B: https://forge.coreweave.com/wandb/rathiabeth-uowd/twincheck/runs/gepd51e6
