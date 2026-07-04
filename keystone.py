"""
keystone.py -- THE KEYSTONE PLOT: can the model be trusted AHEAD OF TIME?

This is t2's actual deliverable, and the go/no-go gate for the whole roadmap.

THE QUESTION. A planner never gets the true panel mid-imagination: to evaluate
"what if I bank left for 5 s", it must feed the model's OWN predictions back in
as inputs, fifty times. One-step skill (card.py) says nothing about what happens
then -- tiny biases amplify, and each small error nudges the imagined panel
off the training distribution where the net is worse, which grows the next
error. This file measures exactly that regime.

THE EXPERIMENT. From every test-episode start point: roll the model freely for
150 steps (15 s) -- state fed back from its own beliefs, actions replayed from
the log -- and measure error against the true flight at every horizon. Three
reference curves frame it:

  persistence     -- panel frozen at t0. MUST rise (the glider flies away from
                     a snapshot); if it doesn't, the experiment itself is broken.
  teacher-forced  -- one-step predictions along the same trajectory: the error
                     floor. The gap between it and free-running IS compounding.
  the twin        -- same rollout, blindfolded model: how much of free-running
                     skill is felt-air vs map-memory.

HOW TO READ THE PLOT (pinned before it existed): the model curve hugging the
floor out to 15 s = FLAT = errors do not compound = you can plan through this
model (LeCun Mode-2; t3 planner is GO). The model curve rejoining persistence
within a couple of seconds = CLIFF = the model is a monitor, not an imagination
(react-only; the roadmap's premise fails here, honestly). In between: the
usable planning horizon is wherever the curve is still well below persistence.

Run:  .venv/bin/python keystone.py   (needs data/dataset.npz + data/model_*.pt)
"""

from pathlib import Path
from typing import Any

import numpy as np

import report
from data_gen import make_world
from train import (
    Checkpoint,
    FloatArr,
    IntArr,
    Panels,
    load_checkpoint,
    load_panels,
    predict_delta,
)

HORIZON = 150  # 15 s at dt=0.1 -- generous vs the 2-10 s a planner needs
STRIDE = 50  # a rollout start every 5 s within each test episode

# persistence is a plot baseline, not an imagination -- pointless to replay
# in the viewport (it is literally the start frame, frozen), so it stays out
# of the saved file.
UNSAVED_RUNS = ("persistence",)


def rollout_starts(data: Panels, test_eps: IntArr, horizon: int, stride: int) -> IntArr:
    """Absolute row indices to roll from: every `stride` rows within each TEST
    episode, keeping only starts with `horizon` true rows still ahead of them
    (crash-shortened episodes simply contribute fewer starts)."""
    starts: list[int] = []
    for ep in test_eps:
        rows = np.nonzero(data.episode == ep)[0]
        # clamp at 0: a NEGATIVE stop would wrap the slice around and emit
        # starts with no full horizon ahead (episodes shorter than `horizon`
        # must contribute zero starts, not garbage ones)
        stop = max(len(rows) - horizon, 0)
        starts.extend(rows[:stop:stride])
    return np.asarray(starts, dtype=np.int64)


def plot_bounds(
    truth: FloatArr, xc: int, yc: int, margin: float = 100.0
) -> tuple[float, float, float, float]:
    """(xlo, xhi, ylo, yhi) covering every true flight path plus a margin --
    the ghost chart's field grid follows the DATA, not a hardcoded arena size
    (the t1 world fit in +/-250 m; the t3 decision corridor does not)."""
    return (
        float(truth[:, :, xc].min() - margin),
        float(truth[:, :, xc].max() + margin),
        float(truth[:, :, yc].min() - margin),
        float(truth[:, :, yc].max() + margin),
    )


def check_shared_test_split(full: Checkpoint, twin: Checkpoint) -> None:
    """Refuse to compare checkpoints that hold different TEST episodes. If the
    twin were ever retrained with another split seed, its 'held-out' rollouts
    could include the full model's TRAINING flights -- a silent leak that would
    flatter every curve on the plot. Hard stop, not a warning."""
    if not np.array_equal(full.split.test, twin.split.test):
        raise ValueError("full/twin checkpoints hold different test splits -- retrain the pair")


def true_panels(data: Panels, starts: IntArr, horizon: int) -> FloatArr:
    """(n, H+1, 9) what the glider actually felt -- the answer key."""
    grid = starts[:, None] + np.arange(horizon + 1)
    out: FloatArr = data.sensors[grid]
    return out


def free_run(ck: Checkpoint, data: Panels, starts: IntArr, horizon: int) -> FloatArr:
    """(n, H+1, 9) panels IMAGINED by the model: h=0 is the true starting panel,
    every later row is built from the model's own previous output -- the exact
    feedback loop a planner would run. Actions are replayed from the log, so
    this isolates STATE compounding (the planner picks its own actions; here we
    grade prediction, not policy). All n rollouts advance together as one batch."""
    out = np.empty((len(starts), horizon + 1, len(ck.sensor_names)))
    panel: FloatArr = data.sensors[starts].copy()
    out[:, 0] = panel
    for h in range(1, horizon + 1):
        action = data.actions[starts + h - 1]  # the command driving step h-1 -> h
        panel = panel + predict_delta(ck, panel, action)  # the imagination step
        out[:, h] = panel
    return out


def teacher_forced(ck: Checkpoint, data: Panels, starts: IntArr, horizon: int) -> FloatArr:
    """(n, H+1, 9) one-step predictions along the true trajectory: at every h the
    model predicts from the TRUE previous panel. No feedback, so no compounding --
    this is the error floor; free-running minus this floor = pure compounding."""
    out = np.empty((len(starts), horizon + 1, len(ck.sensor_names)))
    out[:, 0] = data.sensors[starts]
    for h in range(1, horizon + 1):
        prev = data.sensors[starts + h - 1]
        action = data.actions[starts + h - 1]
        out[:, h] = prev + predict_delta(ck, prev, action)
    return out


def persistence_run(data: Panels, starts: IntArr, horizon: int) -> FloatArr:
    """(n, H+1, 9) the do-nothing imagination: the panel frozen at t0 forever."""
    panel0 = data.sensors[starts]
    out: FloatArr = np.repeat(panel0[:, None, :], horizon + 1, axis=1)
    return out


def sigma_error(pred: FloatArr, true: FloatArr, panel_std: FloatArr) -> FloatArr:
    """(H+1,) whole-panel error per horizon, in units of each channel's natural
    spread (z-scored by TRAIN panel std, RMS over rollouts and channels).
    1.0 sigma ~ 'as wrong as a randomly chosen panel' -- the ceiling of useless."""
    z = (pred - true) / panel_std
    out: FloatArr = np.sqrt(np.mean(z**2, axis=(0, 2)))
    return out


def channel_error(pred: FloatArr, true: FloatArr, col: int) -> FloatArr:
    """(H+1,) RMSE of one channel per horizon, physical units."""
    out: FloatArr = np.sqrt(np.mean((pred[:, :, col] - true[:, :, col]) ** 2, axis=0))
    return out


def position_error(pred: FloatArr, true: FloatArr, x_col: int, y_col: int) -> FloatArr:
    """(H+1,) RMS straight-line position error per horizon, meters."""
    d2 = (pred[:, :, x_col] - true[:, :, x_col]) ** 2 + (pred[:, :, y_col] - true[:, :, y_col]) ** 2
    out: FloatArr = np.sqrt(np.mean(d2, axis=0))
    return out


def save_rollouts(
    path: Path,
    data: Panels,
    starts: IntArr,
    truth: FloatArr,
    runs: dict[str, FloatArr],
    horizon: int,
) -> None:
    """Persist the rollouts so the viewport can replay imagination against
    reality (t27). Self-describing like dataset.npz: channel names, dt and
    alignment metadata travel IN the file, one `rollouts_<name>` array per
    predictor. Alignment contract (what the viewport leans on): rollout row h
    of start i is dataset row starts[i]+h -- same dt, and h=0 IS the true
    starting panel. Regenerate anytime by rerunning this script."""
    # one flat payload dict (typed Any because numpy's savez stub can't take
    # a **dict alongside its allow_pickle: bool parameter)
    payload: dict[str, Any] = {
        "sensor_names": np.array(data.sensor_names),
        "dt": np.array(data.dt, dtype=np.float64),
        "horizon": np.array(horizon, dtype=np.int64),
        "starts": starts,
        "episode": data.episode[starts],
        "true": truth,
    }
    for k, v in runs.items():
        if k not in UNSAVED_RUNS:
            payload["rollouts_" + k.replace("-", "_")] = v
    np.savez_compressed(path, **payload)


def main() -> None:
    here = Path(__file__).resolve().parent
    data = load_panels(here / "data" / "dataset.npz")
    full = load_checkpoint(here / "data" / "model_full.pt")
    twin = load_checkpoint(here / "data" / "model_twin.pt")
    check_shared_test_split(full, twin)  # twin graded on the SAME held-out flights

    starts = rollout_starts(data, full.split.test, HORIZON, STRIDE)
    truth = true_panels(data, starts, HORIZON)
    print(f"free-running {len(starts)} rollouts x {HORIZON} steps ({HORIZON * data.dt:g} s) ...")
    runs = {
        "full": free_run(full, data, starts, HORIZON),
        "twin": free_run(twin, data, starts, HORIZON),
        "persistence": persistence_run(data, starts, HORIZON),
        "teacher-forced": teacher_forced(full, data, starts, HORIZON),
    }

    names = full.sensor_names
    xc, yc, vc = names.index("x"), names.index("y"), names.index("vario")
    seconds = np.arange(HORIZON + 1, dtype=np.float64) * data.dt
    panels = {
        "whole panel (sigma)": {
            k: sigma_error(v, truth, full.stats.panel_std) for k, v in runs.items()
        },
        "position (m)": {k: position_error(v, truth, xc, yc) for k, v in runs.items()},
        "vario (m/s)": {k: channel_error(v, truth, vc) for k, v in runs.items()},
    }

    # the headline table: THE numbers of t2
    marks = [int(s / data.dt) for s in (1, 5, 10, 15)]
    for metric, curves in panels.items():
        print(f"\n{metric}, by horizon:")
        print(f"  {'predictor':>15} {'1s':>9} {'5s':>9} {'10s':>9} {'15s':>9}")
        for name, curve in curves.items():
            cells = " ".join(f"{curve[m]:9.3f}" for m in marks)
            print(f"  {name:>15} {cells}")

    report.plot_keystone(seconds, panels, here / "data" / "keystone.png")

    # persist the rollouts for the viewport's ghost-compare (t27)
    save_rollouts(here / "data" / "rollouts.npz", data, starts, truth, runs, HORIZON)
    print("rollouts: data/rollouts.npz  (viewport ghosts: .venv/bin/python -m viewport.app)")

    # ghost paths: the keystone as a picture of flying (3 example rollouts).
    # Thermal truth for HUMAN EYES only -- the firewall forbids it as model input.
    _, air = make_world()
    xlo, xhi, ylo, yhi = plot_bounds(truth, xc, yc)
    gx, gy = np.meshgrid(np.linspace(xlo, xhi, 200), np.linspace(ylo, yhi, 200))
    ghosts = []
    for i in np.linspace(0, len(starts) - 1, 3, dtype=int):
        ghosts.append(
            {
                "true": (truth[i, :, xc], truth[i, :, yc]),
                "imagined": (runs["full"][i, :, xc], runs["full"][i, :, yc]),
            }
        )
    lift = np.asarray(air.updraft(gx, gy), dtype=np.float64)  # updraft() may return scalar; pin
    report.plot_ghosts((gx, gy, lift), ghosts, here / "data" / "ghost_paths.png")
    print("\ncharts: data/keystone.png, data/ghost_paths.png")


if __name__ == "__main__":
    main()
