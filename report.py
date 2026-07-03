"""
report.py -- the charts: how to LOOK at t2's training.

Nothing in this file affects learning. It exists because ML fails silently: a run
that "completes" proves nothing, and the pictures are where a human catches what
the raw numbers hide. Each function draws one chart and writes one PNG (they land
in data/, next to the dataset and checkpoints -- artifacts, not source).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

FloatArr = npt.NDArray[np.float64]


def plot_lr_finder(lrs: FloatArr, losses: FloatArr, chosen: float, out: Path) -> None:
    """Loss after ONE fresh-net step at each (exponentially growing) learning rate.

    HOW TO READ IT: flat at tiny lr (steps too small to matter), then a falling
    valley (healthy step sizes), then a cliff upward (steps overshoot the minimum
    and training diverges). Pick from the valley. The dashed line marks what
    TrainConfig actually uses -- the plot is the evidence for that choice.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(lrs, losses, lw=1.6)
    ax.axvline(chosen, ls="--", color="tab:red", lw=1.2, label=f"chosen lr = {chosen:g}")
    ax.set_xscale("log")
    # clamp the VIEW (not the data) so the post-cliff explosion can't squash the valley
    ax.set_ylim(0.0, min(4.0, float(np.nanmax(losses)) * 1.05))
    ax.set_xlabel("learning rate (log scale)")
    ax.set_ylabel("minibatch loss after one step")
    ax.set_title("lr finder: valley = usable, cliff = divergence")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_onestep_card(channels: list[str], series: dict[str, FloatArr], out: Path) -> None:
    """Grouped bars, one cluster per channel: the do-nothing baseline vs each model.

    HOW TO READ IT: log scale, so every visible step DOWN from the persistence bar
    is a multiple of skill on that channel. A model bar at persistence height =
    that channel learned nothing. The gap between full and twin bars = what
    feeling the air is worth there.
    """
    x = np.arange(len(channels), dtype=np.float64)
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + (i + 0.5) * width - 0.4, vals, width, label=name)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=30, ha="right")
    ax.set_ylabel("one-step RMSE, physical units (log scale)")
    ax.set_title("one-step report card: lower = better, persistence = learned nothing")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_loss_curves(curves: dict[str, tuple[list[float], list[float]]], out: Path) -> None:
    """Train (solid) vs val (dashed) MSE per epoch, one color per model.

    HOW TO READ IT: both falling together = learning. Val flattening while train
    keeps falling = memorizing (the gap IS the overfit). Nothing moving off 1.0 =
    learned nothing: 1.0 is the predict-the-mean line, where an untrained net
    starts (tripwire 1). Log scale because the interesting story is in the ORDERS
    OF MAGNITUDE below 1.0, invisible on a linear axis.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    palette = ("tab:blue", "tab:orange", "tab:green", "tab:purple")
    for (name, (train_loss, val_loss)), color in zip(curves.items(), palette, strict=False):
        epochs = np.arange(1, len(train_loss) + 1)
        ax.plot(epochs, train_loss, color=color, lw=1.6, label=f"{name} train")
        ax.plot(epochs, val_loss, color=color, lw=1.6, ls="--", label=f"{name} val")
    ax.axhline(1.0, color="gray", ls=":", lw=1.2, label="predict-the-mean = 1.0")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE on z-scored deltas (log scale)")
    ax.set_title("training curves: gap = memorization, plateau = done")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
