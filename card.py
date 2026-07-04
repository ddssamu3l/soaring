"""
card.py -- the one-step report card: did training actually WORK, channel by channel?

Training's single loss number can hide anything (0.134 of what, exactly?). This
card decomposes it on the 20 TEST flights the model has never seen -- one row per
panel channel, in that channel's own physical units, three verdicts per row:

  model RMSE    -- predict each next tick from the TRUE current tick; how far off?
  persist RMSE  -- the do-nothing baseline ("nothing changes"). TRIPWIRE 3: the
                   model must beat this channel by channel, or it learned nothing
                   that matters there.
  spread ratio  -- std(predicted deltas) / std(true deltas). TRIPWIRE 2, collapse:
                   a net that gave up and predicts "the average change, always"
                   still posts a passable RMSE on quiet channels -- but its spread
                   is ~0 and this column exposes it.

The full-vs-twin comparison at the end decomposes WHERE feeling the air helps:
expectation (pinned before running it): the twin trails mainly on the three lift
channels and roughly ties on kinematics.

Run:  .venv/bin/python card.py   (needs data/dataset.npz + data/model_*.pt from train.py)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import report
from train import (
    Checkpoint,
    FloatArr,
    IntArr,
    Panels,
    load_checkpoint,
    load_panels,
    pair_indices,
    predict_delta,
)

# display-only unit labels; unknown channel names simply print without a unit
UNITS = {
    "x": "m",
    "y": "m",
    "z": "m",
    "heading": "rad",
    "airspeed": "m/s",
    "bank": "rad",
    "vario": "m/s",
    "vario_te": "m/s",
    "lift_asym": "m/s",
}


@dataclass(frozen=True)
class Card:
    """One model's report card on the test flights (all arrays are per-channel)."""

    channels: tuple[str, ...]
    model_rmse: FloatArr  # physical units per dt step
    persist_rmse: FloatArr  # same, for the do-nothing predictor
    spread_ratio: FloatArr  # pred-delta std / true-delta std (~1 healthy, ~0 collapsed)
    n_pairs: int
    dt: float


def held_out_pair_rows(ck: Checkpoint, data: Panels) -> IntArr:
    """Pair rows whose episode is in the checkpoint's own TEST split -- the split
    travels inside the checkpoint precisely so eval can't accidentally re-split."""
    pairs = pair_indices(data.episode)
    rows: IntArr = pairs[np.isin(data.episode[pairs], ck.split.test)]
    return rows


def one_step_card(ck: Checkpoint, data: Panels) -> Card:
    """Grade one checkpoint on its test flights, one step at a time.

    Every prediction starts from the TRUE current panel (teacher-forced) -- this
    card isolates raw prediction skill; compounding is keystone.py's question.
    """
    idx = held_out_pair_rows(ck, data)
    # physical units straight from THE shared model call (train.predict_delta):
    # the card must speak meters and m/s, not z-scores
    pred_delta = predict_delta(ck, data.sensors[idx], data.actions[idx])
    true_delta = data.sensors[idx + 1] - data.sensors[idx]
    err = pred_delta - true_delta
    return Card(
        channels=ck.sensor_names,
        model_rmse=np.sqrt(np.mean(err**2, axis=0)),
        persist_rmse=np.sqrt(np.mean(true_delta**2, axis=0)),  # predicting zero change
        spread_ratio=pred_delta.std(axis=0) / np.maximum(true_delta.std(axis=0), 1e-12),
        n_pairs=len(idx),
        dt=data.dt,
    )


def print_card(title: str, card: Card) -> None:
    """The table itself. `better` = persist/model (how many times better than doing
    nothing); `spread` flags collapse when the model's output barely varies."""
    print(
        f"one-step report card -- {title} model "
        f"({card.n_pairs:,} test pairs, physical units per {card.dt:g} s step)"
    )
    head = ("channel", "unit", "model RMSE", "persist RMSE", "better", "spread")
    print(f"  {head[0]:>10} {head[1]:>5} {head[2]:>12} {head[3]:>13} {head[4]:>8} {head[5]:>7}")
    for i, ch in enumerate(card.channels):
        better = card.persist_rmse[i] / card.model_rmse[i]
        flag = "  <-- COLLAPSE?" if card.spread_ratio[i] < 0.1 else ""
        print(
            f"  {ch:>10} {UNITS.get(ch, ''):>5} {card.model_rmse[i]:12.6f}"
            f" {card.persist_rmse[i]:13.6f} {better:7.1f}x {card.spread_ratio[i]:7.2f}{flag}"
        )


def main() -> None:
    here = Path(__file__).resolve().parent
    data = load_panels(here / "data" / "dataset.npz")
    cards: dict[str, Card] = {}
    for name in ("full", "twin"):
        ck = load_checkpoint(here / "data" / f"model_{name}.pt")
        cards[name] = one_step_card(ck, data)
        print_card(name, cards[name])
        print()

    full, twin = cards["full"], cards["twin"]
    print("where feeling the air matters (twin RMSE / full RMSE; ~1 = senses don't help):")
    for i, ch in enumerate(full.channels):
        print(f"  {ch:>10}: {twin.model_rmse[i] / full.model_rmse[i]:5.2f}x")

    report.plot_onestep_card(
        list(full.channels),
        {
            "persistence": full.persist_rmse,
            "full": full.model_rmse,
            "twin": twin.model_rmse,
        },
        here / "data" / "onestep_card.png",
    )
    print("\nchart: data/onestep_card.png")


if __name__ == "__main__":
    main()
