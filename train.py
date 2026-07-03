"""
train.py -- rung-1 world model: an MLP that predicts the glider's next panel.

THE CONTRACT (one model, one loss, two consumers -- the signed-off design):

    eats    : panel_t  -- the 9 numbers the glider can honestly FEEL
                          (x, y, z, heading, airspeed, bank, vario, vario_te, lift_asym)
              action_t -- the 2 stick commands (bank_cmd, pitch_cmd)
    answers : delta-panel -- how much each panel channel CHANGES over the next dt,
              predicted as per-channel z-scored numbers (undo_delta() -> physical units)
    used by : TRAINING    -- graded with MSE against the logged next row (this file)
              IMAGINATION -- prediction added onto the panel and fed back in as the
                             next input, step after step (keystone.py -- the go/no-go plot)

WHY DELTAS, NOT ABSOLUTES: in 0.1 s the glider barely moves, so predicting the next
value would spend the whole net on copying inputs to outputs. Predicting the CHANGE
aims every parameter at the dynamics. Bonus: "output zero" is exactly the
persistence baseline ("nothing changes"), so beating zero = beating persistence.

WHY z-SCORE EVERYTHING: channel scales span ~5000x (x wanders over +-1500 m while
lift_asym lives inside +-0.3 m/s). Raw, the big channels would bully both the
activations and the loss. Every channel is normalized with TRAIN-split statistics:
inputs so the net sees O(1) numbers, targets so one unit of loss means "one typical
step of that channel's own motion" -- that is what lets all 9 channels count equally.

WHY sin/cos(heading): the sim never wraps heading -- a circling glider accumulates
radians without bound (the real dataset reaches 13.7 rad). Physics is periodic in
heading, so the model sees (sin, cos) and every winding looks identical. The TARGET
side stays a raw small delta (max |dheading| ~ 0.06 rad per step in the data).

THE SENSOR FIREWALL: model food is sense() + actions ONLY. Thermal truth
(x0, y0, w_peak, radius) never appears in this file; `true_*` arrays are eval-only.

TRIPWIRES (pinned BEFORE any result existed -- ML fails silently, so we decided in
advance what healthy looks like):
  1. step-0 loss ~= 1.0 -- a fresh net outputs ~0 against targets of std 1, so the
     untrained MSE is computable in advance (makemore's -ln(1/27) = 3.29 moment).
     Running this file prints it; anything far from 1.0 = broken pipeline.
  2. collapse check -- predicted-delta spread ~= 0 means the net learned the dataset
     mean, not dynamics.                       (arrives with the one-step report card)
  3. beat persistence -- per channel, physical units, or it learned nothing useful.
                                               (arrives with the one-step report card)

Run:  .venv/bin/python train.py    (needs data/dataset.npz; rebuild via data_gen.py)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn
from torch.nn import functional as F

FloatArr = npt.NDArray[np.float64]
IntArr = npt.NDArray[np.int64]

# The three "feel the air" channels. The ablation twin trains with these zeroed --
# on a fixed field x,y alone can memorize where the lift lives, and the twin
# measures exactly how much of the full model's skill is map-memory vs felt-air.
LIFT_CHANNELS = ("vario", "vario_te", "lift_asym")


# --- the data, as the model is allowed to see it ---------------------------------
@dataclass(frozen=True)
class Panels:
    """The slice of dataset.npz that is MODEL FOOD (+ episode ids for bookkeeping).

    The `true_*` arrays stay behind in the file on purpose: they are the omniscient
    answer key for evaluation, never inputs (the sensor firewall).
    """

    sensors: FloatArr  # (N, 9) what the glider felt at each tick
    actions: FloatArr  # (N, 2) what the stick asked at that same tick
    episode: IntArr  # (N,)   rollout id; rows chain only WITHIN an episode
    sensor_names: tuple[str, ...]
    action_names: tuple[str, ...]
    dt: float


def load_panels(path: Path) -> Panels:
    """Read the data factory's npz. The file is self-describing (channel names are
    stored in-band), so dims come from the FILE -- nothing hardcodes counts/orders."""
    with np.load(path, allow_pickle=False) as d:
        return Panels(
            sensors=np.asarray(d["sensors"], dtype=np.float64),
            actions=np.asarray(d["actions"], dtype=np.float64),
            episode=np.asarray(d["episode"], dtype=np.int64),
            sensor_names=tuple(str(n) for n in d["sensor_names"]),
            action_names=tuple(str(n) for n in d["action_names"]),
            dt=float(d["dt"]),
        )


# --- split + pairs ----------------------------------------------------------------
@dataclass(frozen=True)
class Split:
    """Episode ids per role. Split BY EPISODE, never by row: consecutive rows are
    near-duplicates (0.1 s apart), so a row-level split would leak test into train."""

    train: IntArr
    val: IntArr
    test: IntArr


def split_episodes(episode: IntArr, seed: int = 0) -> Split:
    """Deterministic 80/10/10 shuffle-split of the episode ids."""
    eps = np.unique(episode)
    order = np.random.default_rng(seed).permutation(eps)
    n_train = int(round(0.8 * len(eps)))
    n_val = int(round(0.1 * len(eps)))
    return Split(
        train=np.sort(order[:n_train]),
        val=np.sort(order[n_train : n_train + n_val]),
        test=np.sort(order[n_train + n_val :]),
    )


def pair_indices(episode: IntArr) -> IntArr:
    """Rows i whose NEXT row continues the same episode. A training pair is
    (sensors[i], actions[i]) -> sensors[i+1]; each episode's final row has no
    successor and is dropped (200 rows out of 120k on the real dataset)."""
    idx: IntArr = np.nonzero(episode[:-1] == episode[1:])[0].astype(np.int64)
    return idx


# --- normalization ------------------------------------------------------------------
@dataclass(frozen=True)
class Stats:
    """Per-channel z-scoring constants, fitted on TRAIN pairs only -- val/test must
    not leak into the normalization, same discipline as the episode split."""

    panel_mean: FloatArr  # (9,)
    panel_std: FloatArr  # (9,)
    action_mean: FloatArr  # (2,)
    action_std: FloatArr  # (2,)
    delta_mean: FloatArr  # (9,)
    delta_std: FloatArr  # (9,)


def fit_stats(data: Panels, train_pairs: IntArr) -> Stats:
    """Means/stds over the train pairs; stds floored so a dead channel can't div-by-0."""
    panel = data.sensors[train_pairs]
    action = data.actions[train_pairs]
    delta = data.sensors[train_pairs + 1] - panel
    floor = 1e-8
    return Stats(
        panel_mean=panel.mean(axis=0),
        panel_std=np.maximum(panel.std(axis=0), floor),
        action_mean=action.mean(axis=0),
        action_std=np.maximum(action.std(axis=0), floor),
        delta_mean=delta.mean(axis=0),
        delta_std=np.maximum(delta.std(axis=0), floor),
    )


# --- featurization -------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureSpec:
    """Column bookkeeping, resolved BY NAME from the dataset -- dims-from-data, so
    nothing downstream ever hardcodes "heading is column 3"."""

    heading: int  # panel column that enters as (sin, cos) instead of z-scored
    lift: tuple[int, ...]  # panel columns the ablation twin never feels
    n_features: int  # model input width: 9 - heading + (sin, cos) + 2 actions = 12


def make_feature_spec(data: Panels) -> FeatureSpec:
    names = data.sensor_names
    return FeatureSpec(
        heading=names.index("heading"),
        lift=tuple(names.index(c) for c in LIFT_CHANNELS),  # raises if a cue is missing
        n_features=len(names) + 1 + len(data.action_names),
    )


def featurize(
    panel: FloatArr,
    action: FloatArr,
    stats: Stats,
    spec: FeatureSpec,
    ablate: bool = False,
) -> FloatArr:
    """(B, 9) panel + (B, 2) action -> (B, 12) model input, every number O(1).

    Layout mirrors the panel: z-scored channels in panel order, except heading which
    enters as (sin, cos) in place; z-scored actions appended at the end.

    `ablate=True` zeroes the lift columns AFTER z-scoring: a constant input carries
    zero information whatever its value, and 0 is the cleanest constant in z-space
    (it IS the train mean -- the twin flies believing the air is always average).
    """
    z = (panel - stats.panel_mean) / stats.panel_std
    cols: list[FloatArr] = []
    for i in range(panel.shape[1]):
        if i == spec.heading:
            cols.append(np.sin(panel[:, i]))
            cols.append(np.cos(panel[:, i]))
        elif ablate and i in spec.lift:
            cols.append(np.zeros(len(panel)))
        else:
            cols.append(z[:, i])
    z_action = (action - stats.action_mean) / stats.action_std
    out: FloatArr = np.column_stack([*cols, z_action])
    return out


# --- targets (and their inverse: the imagination step) --------------------------------
def delta_targets(panel: FloatArr, next_panel: FloatArr, stats: Stats) -> FloatArr:
    """(B, 9) z-scored true deltas -- what the model is graded against.

    Raw heading differences are safe ONLY because the sim never wraps heading;
    a wrap would show up as a ~2pi delta, and a test pins that this never happens.
    """
    return (next_panel - panel - stats.delta_mean) / stats.delta_std


def undo_delta(z_delta: FloatArr, panel: FloatArr, stats: Stats) -> FloatArr:
    """Model output back to a physical panel: panel + de-normalized delta.
    This single line is the imagination step keystone.py loops 150 times."""
    return panel + z_delta * stats.delta_std + stats.delta_mean


# --- the model -------------------------------------------------------------------------
class PanelMLP(nn.Module):
    """The rung-1 world model: featurized (panel, action) -> z-scored delta-panel.

    Deliberately dumb: a Linear/GELU stack. No encoder, no latent, no EMA target --
    that machinery is rung 2 (t6). Width 2x256 (~71k params vs 119,800 training
    pairs) is the data-budget starting guess; the width sweep ratifies or amends it.
    """

    def __init__(self, n_in: int, n_out: int, hidden: tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        widths = (n_in, *hidden)
        for a, b in zip(widths[:-1], widths[1:], strict=True):
            layers.append(nn.Linear(a, b))
            layers.append(nn.GELU())
        layers.append(nn.Linear(widths[-1], n_out))  # no activation: deltas are unbounded
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.net(x)
        return out


def param_count(model: nn.Module) -> int:
    """Data-budget sanity: parameters should sit comfortably below training pairs."""
    return sum(p.numel() for p in model.parameters())


# --- step-0 demo: tripwire 1 fires before any training exists ----------------------------
def main() -> None:
    here = Path(__file__).resolve().parent
    data = load_panels(here / "data" / "dataset.npz")
    split = split_episodes(data.episode)
    pairs = pair_indices(data.episode)
    train_pairs = pairs[np.isin(data.episode[pairs], split.train)]
    stats = fit_stats(data, train_pairs)
    spec = make_feature_spec(data)

    hidden = (256, 256)
    torch.manual_seed(0)  # seeded so this demo prints the same numbers on every run
    model = PanelMLP(spec.n_features, len(data.sensor_names), hidden)

    # TRIPWIRE 1: the untrained loss is computable IN ADVANCE. Targets are z-scored
    # (std 1 per channel); a fresh net outputs ~0; so MSE must print ~1.0. Any other
    # number means the pipeline is broken -- known before training even exists.
    batch = train_pairs[:4096]
    x = torch.from_numpy(
        featurize(data.sensors[batch], data.actions[batch], stats, spec).astype(np.float32)
    )
    y = torch.from_numpy(
        delta_targets(data.sensors[batch], data.sensors[batch + 1], stats).astype(np.float32)
    )
    with torch.no_grad():
        step0 = float(F.mse_loss(model(x), y))

    arrow = " -> ".join(str(w) for w in (spec.n_features, *hidden, len(data.sensor_names)))
    print(
        f"episodes       : {len(split.train)} train / {len(split.val)} val / {len(split.test)} test"
    )
    print(f"training pairs : {len(train_pairs):,} of {len(pairs):,}")
    print(f"model          : {arrow}   ({param_count(model):,} params)")
    print(f"step-0 loss    : {step0:.3f}   (tripwire: must be ~1.0 BEFORE any training)")


if __name__ == "__main__":
    main()
