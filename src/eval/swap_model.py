"""Run a teammate's script with a different LLM, without editing their file (H7).

What it does
    Imports the given module, replaces its module-level ``MODEL`` name (and, for the
    claim extractor, the ``TWINCHECK_EXTRACT_MODEL`` setting it reads) with the
    requested checkpoint, then calls the module's ``main()`` with the remaining
    arguments, exactly as ``python -m <module> <args>`` would. Used to compare
    Qwen3-4B (4-bit) with the Qwen3-8B the team used in:
        src.claims.evaluate          claim extraction (Youssef, Y3)
        src.explain.generate         cited explanations (Rathi, R5)
        src.eval.baselines.llm_judge LLM-judge baseline (Feba, F4)
    The scripts already load their model in 4-bit NF4, so the swap keeps that.
    Point their outputs somewhere new (``--out`` / ``--out-dir``) so the 8B results
    are not overwritten.

Inputs / outputs
    Whatever the wrapped script takes and writes.

How to run
    python -m src.eval.swap_model --module src.eval.baselines.llm_judge --model Qwen/Qwen3-4B -- \
        --predict data/processed/en/multifc_test.jsonl data/copies/en/test.jsonl \
        --out results/H7/qwen4b/preds_llm_judge4b.jsonl --no-wandb
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys


def run(module_name: str, model: str, args: list[str]) -> None:
    """Swap ``MODEL`` in ``module_name`` (and the extractor setting) and call its main()."""
    os.environ["TWINCHECK_EXTRACT_MODEL"] = model       # read by src.claims.extract at import
    module = importlib.import_module(module_name)
    if hasattr(module, "MODEL"):
        module.MODEL = model
    # The claim-extraction evaluator imports src.claims.extract, which holds the model name.
    extract = sys.modules.get("src.claims.extract")
    if extract is not None:
        extract.MODEL = model
    if not hasattr(module, "main"):
        raise SystemExit(f"{module_name} has no main()")
    sys.argv = [module_name] + args
    print(f"running {module_name} with MODEL = {model}", flush=True)
    module.main()


def main() -> None:
    """CLI: wrapper options, then ``--``, then the wrapped script's own arguments."""
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--module", required=True, help="e.g. src.eval.baselines.llm_judge")
    ap.add_argument("--model", required=True, help="e.g. Qwen/Qwen3-4B")
    opts = ap.parse_args(argv[:split])
    run(opts.module, opts.model, argv[split + 1:])


if __name__ == "__main__":
    main()
