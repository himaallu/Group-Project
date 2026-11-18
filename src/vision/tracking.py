"""Small Weights & Biases helper for Feba's scripts (F3, F4).

What it does
    Starts a W&B run in project ``twincheck``. With ``WANDB_API_KEY`` set (Colab
    Secrets) the run is online; without a key it is written **offline** to
    ``wandb/`` (git-ignored) and can be uploaded later with ``wandb sync wandb/offline-run-*``.
    If wandb is not installed, nothing is logged and the script still runs.

Inputs / outputs
    ``start_run(name, config)`` -> run or None; ``log_summary(run, metrics)``; ``finish(run)``.
"""
from __future__ import annotations

import os


def start_run(name: str, config: dict, enabled: bool = True):
    """Start a W&B run (online with a key, offline without), or return None.

    Args:
        name: run name, e.g. ``"F3-probe-seen-s42"``.
        config: settings to record (must include the seed).
        enabled: False skips W&B entirely (e.g. in unit tests).
    """
    if not enabled:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed: run is not logged")
        return None
    mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
    return wandb.init(project="twincheck", name=name, config=config, mode=mode, reinit=True)


def log_summary(run, metrics: dict) -> None:
    """Write final numbers to the run summary (skips None values)."""
    if run is None:
        return
    for key, value in metrics.items():
        if value is not None:
            run.summary[key] = value


def finish(run) -> None:
    """Close the run if there is one."""
    if run is not None:
        run.finish()
