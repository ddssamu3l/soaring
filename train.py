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
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn
from torch.nn import functional as F

import report

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
    panel_lo: FloatArr  # (9,) train-range floor -- the imagination clamp (clamp_panel)
    panel_hi: FloatArr  # (9,) train-range ceiling


def fit_stats(data: Panels, train_pairs: IntArr) -> Stats:
    """Means/stds over the train pairs; stds floored so a dead channel can't div-by-0.
    Also the per-channel train RANGE (both pair endpoints), which bounds what the
    model has ever been graded on -- imagination is clamped to it (clamp_panel)."""
    panel = data.sensors[train_pairs]
    next_panel = data.sensors[train_pairs + 1]
    action = data.actions[train_pairs]
    delta = next_panel - panel
    floor = 1e-8
    return Stats(
        panel_mean=panel.mean(axis=0),
        panel_std=np.maximum(panel.std(axis=0), floor),
        action_mean=action.mean(axis=0),
        action_std=np.maximum(action.std(axis=0), floor),
        delta_mean=delta.mean(axis=0),
        delta_std=np.maximum(delta.std(axis=0), floor),
        panel_lo=np.minimum(panel.min(axis=0), next_panel.min(axis=0)),
        panel_hi=np.maximum(panel.max(axis=0), next_panel.max(axis=0)),
    )


# --- featurization -------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureSpec:
    """Column bookkeeping, resolved BY NAME from the dataset -- dims-from-data, so
    nothing downstream ever hardcodes "heading is column 3"."""

    heading: int  # panel column that enters as (sin, cos) instead of z-scored
    lift: tuple[int, ...]  # panel columns the ablation twin never feels
    n_features: int  # model input width: 9 - heading + (sin, cos) + 2 actions = 12


def make_feature_spec(sensor_names: tuple[str, ...], action_names: tuple[str, ...]) -> FeatureSpec:
    """Takes NAMES (not a Panels) so a reloaded checkpoint can rebuild its spec
    from what was saved inside it, without needing the dataset around."""
    return FeatureSpec(
        heading=sensor_names.index("heading"),
        lift=tuple(sensor_names.index(c) for c in LIFT_CHANNELS),  # raises if a cue is missing
        n_features=len(sensor_names) + 1 + len(action_names),
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


def clamp_panel(panel: FloatArr, stats: Stats) -> FloatArr:
    """Pin an IMAGINED panel to the training data's per-channel range.

    Free-running feedback can drift a panel off the training manifold, where
    the MLP extrapolates arbitrarily -- on the t3 world ~1% of keystone
    rollouts ran away to 1e9 through exactly this loop (vario error -> wilder
    input -> wilder output). Physically: no instrument can read outside its
    gauge, so values beyond anything the world ever produced are not states,
    they're numerical debris. Applied ONLY where predictions are fed back
    (keystone free-run, planner imagination) -- never to real sensor data and
    never during training. It is a stability crutch and is reported when it
    engages; the principled cure (multi-step rollout training loss) is logged
    for rung 2.
    """
    return np.clip(panel, stats.panel_lo, stats.panel_hi)


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


# --- training ----------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainConfig:
    """The dials. Defaults are the signed-off starting point, not sacred truths:
    lr is Adam's classic (the finder plot is the evidence), epochs from the curves."""

    hidden: tuple[int, ...] = (256, 256)
    batch: int = 1024  # noisy-but-frequent gradients beat exact-but-rare ones
    epochs: int = 30  # ~93 minibatch steps per epoch on the real dataset
    lr: float = 1e-3
    drop_frac: float = 0.8  # final 20% of epochs run at lr/10: settle, don't rattle
    seed: int = 0


@dataclass
class History:
    """Per-epoch loss curves -- what the train-vs-val chart is drawn from."""

    train_loss: list[float]
    val_loss: list[float]


def tensor_pairs(
    data: Panels, idx: IntArr, stats: Stats, spec: FeatureSpec, ablate: bool = False
) -> tuple[Tensor, Tensor]:
    """Materialize (inputs, targets) for the given pair rows as float32 tensors.
    Precomputing every pair up front is fine: 95,840 x 12 floats is ~4 MB."""
    x = featurize(data.sensors[idx], data.actions[idx], stats, spec, ablate)
    y = delta_targets(data.sensors[idx], data.sensors[idx + 1], stats)
    return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))


def train_model(
    x_train: Tensor,
    y_train: Tensor,
    x_val: Tensor,
    y_val: Tensor,
    cfg: TrainConfig,
    verbose: bool = True,
) -> tuple[PanelMLP, History]:
    """The loop: shuffled minibatches -> forward -> MSE -> backward -> Adam step,
    with the val set measured after every epoch. Two guards live here:

      - LAST-STRETCH LR DROP (drop_frac): finish at lr/10 so the weights settle
        into the minimum instead of rattling around it (makemore's 0.1 -> 0.01 move);
      - KEEP-BEST-ON-VAL: the returned model is the epoch snapshot with the lowest
        val loss, not whatever the last step happened to leave behind.
    """
    torch.manual_seed(cfg.seed)
    model = PanelMLP(x_train.shape[1], y_train.shape[1], cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed)  # our own shuffle stream, decoupled from torch
    hist = History([], [])
    best_val = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    n = len(x_train)
    for epoch in range(cfg.epochs):
        lr = cfg.lr if epoch < int(cfg.drop_frac * cfg.epochs) else cfg.lr / 10
        for group in opt.param_groups:
            group["lr"] = lr
        order = torch.from_numpy(rng.permutation(n))  # fresh shuffle every epoch
        model.train()
        total, count = 0.0, 0
        for start in range(0, n - cfg.batch + 1, cfg.batch):  # tail < 1 batch is skipped
            b = order[start : start + cfg.batch]
            loss = F.mse_loss(model(x_train[b]), y_train[b])
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]  # torch stubs omit Tensor.backward
            opt.step()
            total += float(loss.detach()) * len(b)
            count += len(b)
        model.eval()
        with torch.no_grad():
            val = float(F.mse_loss(model(x_val), y_val))
        hist.train_loss.append(total / count)
        hist.val_loss.append(val)
        if val < best_val:
            best_val = val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose:
            print(
                f"  epoch {epoch + 1:>3}/{cfg.epochs}  lr {lr:.0e}"
                f"  train {hist.train_loss[-1]:.5f}  val {val:.5f}"
            )
    model.load_state_dict(best_state)
    model.eval()
    return model, hist


def lr_find(
    x: Tensor,
    y: Tensor,
    hidden: tuple[int, ...] = (256, 256),
    lo: float = 1e-4,
    hi: float = 1.0,
    steps: int = 120,
    batch: int = 1024,
    seed: int = 0,
) -> tuple[FloatArr, FloatArr]:
    """Karpathy's finder: a FRESH net takes one minibatch step per learning rate,
    climbing exponentially lo -> hi; each step's loss is recorded. Plotted, the
    curve falls into a valley then explodes off a cliff -- you pick from the valley.
    Used as EVIDENCE for TrainConfig.lr, not as automation: the plot is for eyes."""
    torch.manual_seed(seed)
    model = PanelMLP(x.shape[1], y.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lo)
    rng = np.random.default_rng(seed)
    lrs = np.geomspace(lo, hi, steps)
    losses = np.empty(steps)
    for i in range(steps):
        for group in opt.param_groups:
            group["lr"] = float(lrs[i])
        b = torch.from_numpy(rng.integers(0, len(x), batch))
        loss = F.mse_loss(model(x[b]), y[b])
        opt.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]  # torch stubs omit Tensor.backward
        opt.step()
        losses[i] = float(loss.detach())
    return lrs, losses


# --- checkpoints (what keystone.py reloads) ------------------------------------------------
@dataclass(frozen=True)
class Checkpoint:
    """Everything imagination needs, reloaded: the net plus the EXACT normalization
    it was trained with -- mismatched stats would silently wreck every rollout."""

    model: PanelMLP
    stats: Stats
    spec: FeatureSpec
    split: Split
    ablate: bool
    sensor_names: tuple[str, ...]
    action_names: tuple[str, ...]
    dt: float


_STATS = (
    "panel_mean",
    "panel_std",
    "action_mean",
    "action_std",
    "delta_mean",
    "delta_std",
    "panel_lo",
    "panel_hi",
)


def save_checkpoint(
    path: Path,
    model: PanelMLP,
    cfg: TrainConfig,
    stats: Stats,
    split: Split,
    data: Panels,
    ablate: bool,
) -> None:
    """One torch file holding weights + stats + split + provenance. Tensor-only
    payload so it reloads under torch's safe `weights_only=True`."""
    payload: dict[str, object] = {
        "model_state": model.state_dict(),
        "hidden": list(cfg.hidden),
        "ablate": ablate,
        "sensor_names": list(data.sensor_names),
        "action_names": list(data.action_names),
        "dt": data.dt,
        "stats": {k: torch.from_numpy(getattr(stats, k)) for k in _STATS},
        "split": {k: torch.from_numpy(getattr(split, k)) for k in ("train", "val", "test")},
    }
    torch.save(payload, path)


def load_checkpoint(path: Path) -> Checkpoint:
    raw: dict[str, Any] = torch.load(path, weights_only=True)
    sensor_names = tuple(str(s) for s in raw["sensor_names"])
    action_names = tuple(str(s) for s in raw["action_names"])
    spec = make_feature_spec(sensor_names, action_names)
    st = {k: np.asarray(raw["stats"][k].numpy(), dtype=np.float64) for k in _STATS}
    stats = Stats(
        panel_mean=st["panel_mean"],
        panel_std=st["panel_std"],
        action_mean=st["action_mean"],
        action_std=st["action_std"],
        delta_mean=st["delta_mean"],
        delta_std=st["delta_std"],
        panel_lo=st["panel_lo"],
        panel_hi=st["panel_hi"],
    )
    sp = {k: np.asarray(raw["split"][k].numpy(), dtype=np.int64) for k in ("train", "val", "test")}
    split = Split(train=sp["train"], val=sp["val"], test=sp["test"])
    model = PanelMLP(spec.n_features, len(sensor_names), tuple(int(w) for w in raw["hidden"]))
    model.load_state_dict(raw["model_state"])
    model.eval()
    return Checkpoint(
        model=model,
        stats=stats,
        spec=spec,
        split=split,
        ablate=bool(raw["ablate"]),
        sensor_names=sensor_names,
        action_names=action_names,
        dt=float(raw["dt"]),
    )


def predict_delta(ck: Checkpoint, panel: FloatArr, action: FloatArr) -> FloatArr:
    """(B, 9) panel + (B, 2) action -> (B, 9) predicted panel CHANGE, physical units.

    THE model call -- featurize -> net -> de-normalize, in one place. One-step
    prediction is `panel + predict_delta(...)`; imagination is that same line fed
    back into itself. card.py grades this call, keystone.py loops it, and the
    planner scores candidate futures through it -- one code path, one behavior.
    """
    x = torch.from_numpy(featurize(panel, action, ck.stats, ck.spec, ck.ablate).astype(np.float32))
    with torch.no_grad():
        z = np.asarray(ck.model(x).numpy(), dtype=np.float64)
    out: FloatArr = z * ck.stats.delta_std + ck.stats.delta_mean
    return out


# --- the run: tripwire -> lr finder -> train full + twin -> curves + checkpoints -----------
def main() -> None:
    here = Path(__file__).resolve().parent
    data = load_panels(here / "data" / "dataset.npz")
    split = split_episodes(data.episode)
    pairs = pair_indices(data.episode)
    role_pairs = {  # pair rows whose episode belongs to each role
        role: pairs[np.isin(data.episode[pairs], getattr(split, role))]
        for role in ("train", "val", "test")
    }
    stats = fit_stats(data, role_pairs["train"])
    spec = make_feature_spec(data.sensor_names, data.action_names)
    cfg = TrainConfig()
    x_tr, y_tr = tensor_pairs(data, role_pairs["train"], stats, spec)

    # TRIPWIRE 1: the untrained loss is computable IN ADVANCE. Targets are z-scored
    # (std 1 per channel); a fresh net outputs ~0; so MSE must print ~1.0. Any other
    # number means the pipeline is broken -- known before training even starts.
    torch.manual_seed(0)
    probe = PanelMLP(spec.n_features, len(data.sensor_names), cfg.hidden)
    with torch.no_grad():
        step0 = float(F.mse_loss(probe(x_tr[:4096]), y_tr[:4096]))
    arrow = " -> ".join(str(w) for w in (spec.n_features, *cfg.hidden, len(data.sensor_names)))
    print(
        f"episodes       : {len(split.train)} train / {len(split.val)} val / {len(split.test)} test"
    )
    print(f"training pairs : {len(role_pairs['train']):,} of {len(pairs):,}")
    print(f"model          : {arrow}   ({param_count(probe):,} params)")
    print(f"step-0 loss    : {step0:.3f}   (tripwire: must be ~1.0 BEFORE any training)")

    print("\nlr finder (one fresh-net step per lr; evidence for the chosen lr) ...")
    lrs, losses = lr_find(x_tr, y_tr, cfg.hidden)
    report.plot_lr_finder(lrs, losses, cfg.lr, here / "data" / "lr_finder.png")

    # Train the full model and the blindfolded twin IDENTICALLY -- same data rows,
    # same seed, same schedule; the only difference is the zeroed lift columns.
    curves: dict[str, tuple[list[float], list[float]]] = {}
    for name, ablate in (("full", False), ("twin", True)):
        print(f"\ntraining the {name} model {'(lift senses zeroed)' if ablate else ''}...")
        xt, yt = tensor_pairs(data, role_pairs["train"], stats, spec, ablate)
        xv, yv = tensor_pairs(data, role_pairs["val"], stats, spec, ablate)
        model, hist = train_model(xt, yt, xv, yv, cfg)
        save_checkpoint(here / "data" / f"model_{name}.pt", model, cfg, stats, split, data, ablate)
        curves[name] = (hist.train_loss, hist.val_loss)
        print(f"  best val {min(hist.val_loss):.5f}  ->  data/model_{name}.pt")
    report.plot_loss_curves(curves, here / "data" / "loss_curves.png")
    print("\ncharts: data/lr_finder.png, data/loss_curves.png")


if __name__ == "__main__":
    main()
