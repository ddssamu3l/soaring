"""
experiment.py -- t4: DOES PREDICTING BEAT REACTING? The organizing question,
run as a paired experiment and answered as a number.

TWO AGENTS, ONE MISSING ABILITY. planner.py flies by imagining futures through
the frozen t2-certified model (predict); reactive.py flies by reflex on the
current panel (react). Same world, same sensors, same goal knowledge, same
own-body arithmetic, same pilot constants (reserve, margin) -- the ONE thing
the reactor lacks is imagination. Whatever gap this file measures is the value
of prediction on this task, which is the number the whole roadmap exists to
produce (flat keystone -> you can plan through the model; THIS experiment asks
whether planning through it actually beats not planning at all).

THE DESIGN, and why each piece is load-bearing for honesty:
  * PAIRED trials -- every agent flies the IDENTICAL (start, goal) task, so
    task difficulty cancels out of the comparison; random start AND goal, so
    neither agent gets a task hand-picked for its strengths. The field stays
    the t3 fixed world: that is the fixed-field phase's premise (the planner's
    model has it memorized in weights -- part of what "predicting" means here;
    randomized worlds are t5's experiment, not this one).
  * The reactor ladder (b0 blind glide / b1 pure reflex / b2 reflex + the
    planner's own final-glide arithmetic) shows WHERE reaction fails, not just
    that it fails. b2 is the fair headline opponent.
  * The reactor is TUNED (--tune: grid search) before the eval -- beating a
    strawman proves nothing. Tuning runs on its OWN trial seed; the eval seed
    is never touched during tuning (no test-set tuning).
  * Trials are stratified by the start's own-body arithmetic (deficit > 0 =
    "must climb"): the strata separate "any glider could have glided it" tasks
    from the decision-forcing ones the experiment is really about.

Run:  .venv/bin/python experiment.py --tune       (grid-search the reactors)
      .venv/bin/python experiment.py --n 24       (the paired eval; needs
                                                   data/model_full*.pt)
"""

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from data_gen import make_world, random_start
from glider_sim import SENSOR_NAMES, GliderState, Simulation
from planner import GlidePolar, Goal, PlannerConfig, best_glide, fly_to_goal
from reactive import ReactiveConfig, ReactivePilot, Variant, fly_reactive
from train import Checkpoint, load_checkpoint

# the goal box: inside the sky the dataset actually flew (starts span x -250..
# 1250, y +-350 and flights wander outward), so no task routes the planner
# through un-flown air where its imagination was never graded. Wide enough to
# put goals on both sides of the sink wall.
GOAL_X = (-200.0, 1500.0)
GOAL_Y = (-300.0, 300.0)
MIN_DIST = 400.0  # closer goals are trivial for everyone -- no signal


@dataclass(frozen=True)
class Trial:
    """One task: spawn here, reach that circle. Given to every agent verbatim."""

    start: GliderState
    goal: Goal


@dataclass(frozen=True)
class Result:
    """One agent's attempt: its fate, when, and where it ended up (the death
    map -- WHERE an agent dies is the diagnostic the rates alone hide)."""

    outcome: str  # "arrived" | "crashed" | "timeout"
    seconds: float
    x: float
    y: float
    z: float


ReactorRun = Callable[[Trial, ReactiveConfig], Result]


def sample_trials(rng: np.random.Generator, n: int) -> list[Trial]:
    """n random (start, goal) tasks. Starts reuse data_gen.random_start (the
    training envelope -- same sky the model learned); goals are uniform in the
    goal box, rejection-sampled to be at least MIN_DIST out."""
    trials: list[Trial] = []
    while len(trials) < n:
        start = random_start(rng)
        goal = Goal(x=float(rng.uniform(*GOAL_X)), y=float(rng.uniform(*GOAL_Y)))
        if math.hypot(start.x - goal.x, start.y - goal.y) < MIN_DIST:
            continue
        trials.append(Trial(start=start, goal=goal))
    return trials


def start_deficit(trial: Trial, polar: GlidePolar, pcfg: PlannerConfig) -> float:
    """The task's own-body difficulty flag: the reactor's deficit arithmetic
    (identical to the planner scorer's) evaluated at the spawn. > 0 = the goal
    is NOT in margined still-air glide range = the task forces a climb (or a
    death); 0 = a straight final glide could work if the route allows it."""
    pilot = ReactivePilot(
        ReactiveConfig(reserve_height=pcfg.reserve_height, glide_margin=pcfg.glide_margin),
        polar,
        trial.goal,
        dt=0.1,
    )
    s = trial.start
    return pilot.deficit({"x": s.x, "y": s.y, "z": s.z, "airspeed": s.airspeed})


# --- running one trial -------------------------------------------------------------------
def run_reactor(trial: Trial, cfg: ReactiveConfig, max_seconds: float) -> Result:
    """One reactor flight on a fresh copy of the fixed world."""
    glider, air = make_world()
    sim = Simulation(glider, air, replace(trial.start))
    pilot = ReactivePilot(cfg, best_glide(glider), trial.goal, sim.dt)
    flight = fly_reactive(sim, pilot, trial.goal, max_seconds)
    s = sim.state
    return Result(outcome=flight.outcome, seconds=flight.seconds, x=s.x, y=s.y, z=s.z)


def run_planner(
    trial: Trial,
    ensemble: list[Checkpoint],
    pcfg: PlannerConfig,
    seed: int,
    max_seconds: float,
) -> Result:
    """One planner flight, CEM seeded per-trial so the whole eval is one
    reproducible run."""
    glider, air = make_world()
    sim = Simulation(glider, air, replace(trial.start))
    flight = fly_to_goal(
        sim, ensemble, trial.goal, pcfg, np.random.default_rng(seed), max_seconds=max_seconds
    )
    s = sim.state
    return Result(outcome=flight.outcome, seconds=flight.seconds, x=s.x, y=s.y, z=s.z)


# --- tuning the reactor (the anti-strawman step) ------------------------------------------
def reactor_grid(variant: Variant) -> list[ReactiveConfig]:
    """The honest search space for each reactor's dials. reserve/margin are
    NOT in the grid -- they are pinned to the planner's constants (fairness).
    v_dash None-vs-28 lets the tuner decide whether classic speed-to-fly
    (dash through sink) earns its keep, rather than us legislating it."""
    if variant == "b0":
        return [ReactiveConfig(variant="b0", v_dash=dash) for dash in (None, 28.0)]
    return [
        ReactiveConfig(
            variant=variant,
            vario_enter=enter,
            exit_patience=patience,
            k_asym=k_asym,
            v_dash=dash,
        )
        for enter in (0.3, 0.6, 1.0)
        for patience in (4.0, 8.0)
        for k_asym in (0.5, 1.5)
        for dash in (None, 28.0)
    ]


def tune_reactor(
    grid: list[ReactiveConfig], trials: list[Trial], run: ReactorRun
) -> tuple[ReactiveConfig, list[tuple[int, float]]]:
    """Grid search: fly every config over the tuning trials; winner = most
    arrivals, ties broken by faster mean arrival time, then grid order
    (deterministic). Returns (winner, per-config (arrivals, mean_time))."""
    rows: list[tuple[int, float]] = []
    best_i = 0
    best_key = (1, math.inf)  # (-arrivals, mean_time): smaller is better
    for i, cfg in enumerate(grid):
        results = [run(t, cfg) for t in trials]
        arrivals = sum(r.outcome == "arrived" for r in results)
        times = [r.seconds for r in results if r.outcome == "arrived"]
        mean_time = float(np.mean(times)) if times else math.inf
        rows.append((arrivals, mean_time))
        key = (-arrivals, mean_time)
        if key < best_key:
            best_key, best_i = key, i
    return grid[best_i], rows


# Winners of `experiment.py --tune` (the grid above, 12 tuning trials on seed
# 7, 240 s cap) -- pinned so the eval is reproducible without re-tuning.
# Re-run --tune and re-pin after any world or reactor change.
TUNED: dict[str, ReactiveConfig] = {
    "b0": ReactiveConfig(variant="b0"),
    "b1": ReactiveConfig(variant="b1"),
    "b2": ReactiveConfig(variant="b2"),
}


# --- the paired eval ----------------------------------------------------------------------
def outcome_counts(results: list[Result]) -> dict[str, int]:
    """How many arrived / crashed / timed out."""
    return {
        fate: sum(r.outcome == fate for r in results) for fate in ("arrived", "crashed", "timeout")
    }


def paired_counts(a: list[Result], b: list[Result]) -> dict[str, int]:
    """The 2x2 paired table: on the SAME tasks, who arrived? This is the
    experiment's headline shape -- 'a arrived where b died' is the value of
    what a has and b lacks."""
    a_in = [r.outcome == "arrived" for r in a]
    b_in = [r.outcome == "arrived" for r in b]
    return {
        "both": sum(x and y for x, y in zip(a_in, b_in, strict=True)),
        "only_a": sum(x and not y for x, y in zip(a_in, b_in, strict=True)),
        "only_b": sum(y and not x for x, y in zip(a_in, b_in, strict=True)),
        "neither": sum(not x and not y for x, y in zip(a_in, b_in, strict=True)),
    }


def save_results(path: Path, trials: list[Trial], results: dict[str, list[Result]]) -> None:
    """Persist the whole eval, self-describing (same discipline as the
    dataset): trials + per-agent fates, so any later analysis re-reads the
    experiment instead of trusting a summary."""
    n = len(trials)
    arrays: dict[str, Any] = {
        "starts": np.array(
            [
                [t.start.x, t.start.y, t.start.z, t.start.heading, t.start.airspeed, t.start.bank]
                for t in trials
            ]
        ),
        "goals": np.array([[t.goal.x, t.goal.y, t.goal.radius] for t in trials]),
        "agents": np.array(sorted(results)),
    }
    for name, res in results.items():
        assert len(res) == n
        arrays[f"{name}_outcome"] = np.array([r.outcome for r in res])
        arrays[f"{name}_seconds"] = np.array([r.seconds for r in res])
        arrays[f"{name}_final"] = np.array([[r.x, r.y, r.z] for r in res])
    np.savez(path, **arrays)


def plot_results(trials: list[Trial], results: dict[str, list[Result]], out: Path) -> None:
    """The death map + the scoreboard. Drawing the TRUE field is legal here:
    this is eval-side visualization for humans; no model input is involved."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(13.5, 6.5), gridspec_kw={"width_ratios": [2.2, 1.0]}
    )
    _, air = make_world()
    for th in air.thermals:
        face = "tab:green" if th.w_peak > 0 else "tab:red"
        ax.add_patch(Circle((th.x0, th.y0), th.radius, color=face, alpha=0.14, zorder=0))
    for t in trials:
        ax.plot([t.start.x, t.goal.x], [t.start.y, t.goal.y], color="0.88", lw=0.7, zorder=1)
        ax.plot(t.start.x, t.start.y, ".", color="k", ms=4, zorder=2)
        ax.plot(t.goal.x, t.goal.y, "*", mec="k", mfc="none", ms=9, zorder=2)
    colors = {"planner": "tab:blue", "b2": "tab:orange", "b1": "tab:purple", "b0": "0.5"}
    markers = {"arrived": "o", "crashed": "x", "timeout": "s"}
    for name, res in results.items():
        c = colors.get(name, "tab:brown")
        for r in res:
            ax.plot(r.x, r.y, markers[r.outcome], color=c, ms=5, alpha=0.85, zorder=3)
        ax.plot([], [], "o", color=c, label=name)  # legend proxy
    ax.plot([], [], "x", color="k", label="crashed")
    ax.plot([], [], "s", color="k", label="timeout")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("where each agent ended up (dots=starts, stars=goals)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)

    names = list(results)
    arrived = [outcome_counts(results[n])["arrived"] for n in names]
    axb.bar(names, arrived, color=[colors.get(n, "tab:brown") for n in names])
    axb.set_ylabel(f"arrivals out of {len(trials)}")
    axb.set_title("the headline")
    axb.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def summarize(
    trials: list[Trial],
    results: dict[str, list[Result]],
    polar: GlidePolar,
    pcfg: PlannerConfig,
) -> None:
    """Print the whole verdict: rates, the paired 2x2, and the must-climb
    stratum (the decision-forcing tasks the experiment is really about)."""
    n = len(trials)
    must = [start_deficit(t, polar, pcfg) > 0.0 for t in trials]
    print(f"\n=== DOES PREDICTING BEAT REACTING?  ({n} paired trials) ===")
    print(f"must-climb tasks (start deficit > 0): {sum(must)}/{n}")
    for name, res in results.items():
        c = outcome_counts(res)
        times = sorted(r.seconds for r in res if r.outcome == "arrived")
        med = f"{times[len(times) // 2]:6.1f} s" if times else "     --"
        hard = sum(1 for r, m in zip(res, must, strict=True) if m and r.outcome == "arrived")
        print(
            f"  {name:8s} arrived {c['arrived']:2d}/{n}  crashed {c['crashed']:2d}  "
            f"timeout {c['timeout']:2d}   median arrival {med}   "
            f"must-climb arrivals {hard}/{sum(must)}"
        )
    if "planner" in results and "b2" in results:
        p = paired_counts(results["planner"], results["b2"])
        print(
            f"  paired (planner vs b2): both {p['both']}, planner-only {p['only_a']}, "
            f"b2-only {p['only_b']}, neither {p['neither']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=24, help="paired eval trials")
    parser.add_argument("--tune", action="store_true", help="grid-search the reactors and exit")
    parser.add_argument("--tune-n", type=int, default=12, help="tuning trials per config")
    parser.add_argument("--max-seconds", type=float, default=240.0, help="per-flight time cap")
    parser.add_argument("--eval-seed", type=int, default=42, help="eval trial sampler seed")
    parser.add_argument("--tune-seed", type=int, default=7, help="tuning trial sampler seed")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    glider, _ = make_world()
    polar = best_glide(glider)
    pcfg = PlannerConfig()

    if args.tune:
        # tuning trials come from their OWN seed: the eval set stays untouched
        trials = sample_trials(np.random.default_rng(args.tune_seed), args.tune_n)
        for variant in ("b0", "b1", "b2"):
            grid = reactor_grid(variant)

            def run(t: Trial, c: ReactiveConfig) -> Result:
                return run_reactor(t, c, args.max_seconds)

            best, rows = tune_reactor(grid, trials, run)
            print(f"\n--- {variant}: {len(grid)} configs x {len(trials)} trials ---")
            for cfg, (arrivals, mean_time) in zip(grid, rows, strict=True):
                mark = " <== winner" if cfg == best else ""
                print(
                    f"  enter={cfg.vario_enter:.1f} patience={cfg.exit_patience:.0f} "
                    f"k_asym={cfg.k_asym:.1f} dash={cfg.v_dash}  ->  "
                    f"{arrivals}/{len(trials)} arrived, mean {mean_time:.1f} s{mark}"
                )
            print(f"  winner: {best}")
        print("\npin the winners into TUNED before running the eval.")
        return

    ensemble = [load_checkpoint(p) for p in sorted((here / "data").glob("model_full*.pt"))]
    assert ensemble, "no data/model_full*.pt -- run train.py first"
    assert tuple(ensemble[0].sensor_names) == SENSOR_NAMES
    trials = sample_trials(np.random.default_rng(args.eval_seed), args.n)

    results: dict[str, list[Result]] = {"planner": [], "b0": [], "b1": [], "b2": []}
    for i, trial in enumerate(trials):
        results["planner"].append(run_planner(trial, ensemble, pcfg, 1000 + i, args.max_seconds))
        for name in ("b0", "b1", "b2"):
            results[name].append(run_reactor(trial, TUNED[name], args.max_seconds))
        parts = [
            f"{name}:{res[-1].outcome[:3]}@{res[-1].seconds:5.1f}s" for name, res in results.items()
        ]
        print(f"trial {i:02d}  " + "  ".join(parts), flush=True)

    summarize(trials, results, polar, pcfg)
    save_results(here / "data" / "experiment.npz", trials, results)
    plot_results(trials, results, here / "data" / "experiment.png")
    print("\nsaved -> data/experiment.npz, data/experiment.png")


if __name__ == "__main__":
    main()
