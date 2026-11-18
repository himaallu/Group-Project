# nli_v2 (R4)

Weights are on Hugging Face Hub: https://huggingface.co/RathiAbeth/twincheck-nli-v2 (too big for git).

Load: `python -m src.nli.score FILE --model RathiAbeth/twincheck-nli-v2`.

Base `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, 2 epochs, 18525 v2 pairs, lr 2e-05, seed 42, fp32 (tf32). Gold macro-F1 EN 0.3711 -> 0.3335, AR 0.4427 -> 0.4734. W&B: https://forge.coreweave.com/wandb/rathiabeth-uowd/twincheck/runs/hmtd5olh
