"""Lightweight WandB helpers that are safe if wandb is missing."""
from typing import Optional


class WandbStub:
    """No-op stub when wandb is unavailable."""

    def __getattr__(self, _name):
        def _noop(*args, **kwargs):
            return None

        return _noop


def _get_wandb():
    try:
        import wandb

        return wandb
    except ImportError:
        return None


def init_wandb(config: dict, project: Optional[str], run_name: Optional[str], mode: str = "disabled"):
    """Init wandb run or return stub."""
    wandb = _get_wandb()
    if wandb is None or project is None:
        return WandbStub()
    return wandb.init(project=project, name=run_name, config=config, mode=mode)


def log_metrics(wandb_run, metrics: dict, step: int):
    """Log scalar metrics."""
    if hasattr(wandb_run, "log"):
        wandb_run.log(metrics, step=step)


def finish(wandb_run):
    """Finish run if supported."""
    if hasattr(wandb_run, "finish"):
        wandb_run.finish()


def log_histograms(wandb_run, params, step: int):
    """Log weight and grad histograms."""
    wandb = _get_wandb()
    if wandb is None or not hasattr(wandb_run, "log"):
        return
    payload = {}
    for i, (p, g, _l2) in enumerate(params):
        payload[f"hist/weights_{i}"] = wandb.Histogram(p)
        payload[f"hist/grads_{i}"] = wandb.Histogram(g)
    wandb_run.log(payload, step=step)
