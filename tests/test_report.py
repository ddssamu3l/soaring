"""
Smoke tests for the chart writers: each must produce a non-empty PNG without
blowing up on representative inputs. Visual QUALITY is judged by human eyes
(that is the whole point of report.py); these only pin that the plumbing works.
"""

from pathlib import Path

import numpy as np

from report import plot_loss_curves, plot_lr_finder, plot_onestep_card


def test_lr_finder_plot_writes_a_file(tmp_path: Path) -> None:
    lrs = np.geomspace(1e-4, 1.0, 30)
    losses = np.concatenate([np.linspace(1.0, 0.4, 20), np.linspace(0.5, 40.0, 10)])
    out = tmp_path / "lr_finder.png"
    plot_lr_finder(lrs, losses, chosen=1e-3, out=out)
    assert out.stat().st_size > 0


def test_onestep_card_plot_writes_a_file(tmp_path: Path) -> None:
    out = tmp_path / "onestep_card.png"
    plot_onestep_card(
        ["z", "vario"],
        {"persistence": np.array([0.5, 2.3]), "full": np.array([0.01, 0.9])},
        out,
    )
    assert out.stat().st_size > 0


def test_loss_curves_plot_writes_a_file(tmp_path: Path) -> None:
    out = tmp_path / "loss_curves.png"
    plot_loss_curves(
        {"full": ([1.0, 0.5, 0.3], [1.1, 0.6, 0.4]), "twin": ([1.0, 0.7], [1.1, 0.8])},
        out,
    )
    assert out.stat().st_size > 0
