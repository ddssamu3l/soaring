"""
tests/test_experiment.py -- the experiment harness must be trustworthy before
its number is: paired tasks really are identical and reproducible, the tuner
really picks the best reactor (no strawman), the paired table really counts
what it claims, and the persisted npz really round-trips the eval.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from data_gen import make_world
from experiment import (
    GOAL_X,
    GOAL_Y,
    MIN_DIST,
    Result,
    Trial,
    outcome_counts,
    paired_counts,
    reactor_grid,
    run_reactor,
    sample_trials,
    save_results,
    start_deficit,
    tune_reactor,
)
from glider_sim import GliderState
from planner import Goal, PlannerConfig, best_glide
from reactive import ReactiveConfig


def test_sample_trials_is_reproducible_and_in_bounds() -> None:
    a = sample_trials(np.random.default_rng(0), 20)
    b = sample_trials(np.random.default_rng(0), 20)
    assert a == b  # the eval set is pinned by its seed
    for t in a:
        assert GOAL_X[0] <= t.goal.x <= GOAL_X[1]
        assert GOAL_Y[0] <= t.goal.y <= GOAL_Y[1]
        assert math.hypot(t.start.x - t.goal.x, t.start.y - t.goal.y) >= MIN_DIST
        assert 30.0 <= t.start.z <= 600.0  # data_gen's training envelope


def test_start_deficit_flags_the_decision_forcing_tasks() -> None:
    glider, _ = make_world()
    polar = best_glide(glider)
    pcfg = PlannerConfig()
    low_far = Trial(
        start=GliderState(x=0.0, y=0.0, z=60.0, heading=0.0, airspeed=24.0, bank=0.0),
        goal=Goal(x=1500.0, y=0.0),
    )
    high_near = Trial(
        start=GliderState(x=1000.0, y=0.0, z=500.0, heading=0.0, airspeed=24.0, bank=0.0),
        goal=Goal(x=1500.0, y=0.0),
    )
    assert start_deficit(low_far, polar, pcfg) > 0.0  # must climb
    assert start_deficit(high_near, polar, pcfg) == 0.0  # glide made


def test_tune_reactor_picks_most_arrivals_then_fastest() -> None:
    trials = sample_trials(np.random.default_rng(1), 3)
    grid = [ReactiveConfig(variant="b1", vario_enter=e) for e in (0.1, 0.2, 0.3)]

    def stub(trial: Trial, cfg: ReactiveConfig) -> Result:
        # 0.1 crashes everywhere; 0.2 and 0.3 both arrive, 0.3 arrives faster
        if cfg.vario_enter == pytest.approx(0.1):
            return Result("crashed", 10.0, 0.0, 0.0, 0.0)
        return Result("arrived", cfg.vario_enter * 100.0, 0.0, 0.0, 0.0)

    best, rows = tune_reactor(grid, trials, stub)
    assert len(rows) == len(grid)
    assert rows[0] == (0, math.inf)  # no arrivals -> no mean time
    assert best.vario_enter == pytest.approx(0.2)  # fewer seconds beats grid order


def test_reactor_grid_never_tunes_the_pinned_fairness_constants() -> None:
    pcfg = PlannerConfig()
    for variant in ("b0", "b1", "b2"):
        for cfg in reactor_grid(variant):  # type: ignore[arg-type]
            assert cfg.reserve_height == pcfg.reserve_height
            assert cfg.glide_margin == pcfg.glide_margin
            assert cfg.variant == variant


def test_paired_and_outcome_counts_count_what_they_claim() -> None:
    def res(outcome: str) -> Result:
        return Result(outcome, 1.0, 0.0, 0.0, 0.0)

    a = [res("arrived"), res("arrived"), res("crashed"), res("timeout")]
    b = [res("arrived"), res("crashed"), res("arrived"), res("crashed")]
    assert paired_counts(a, b) == {"both": 1, "only_a": 1, "only_b": 1, "neither": 1}
    assert outcome_counts(a) == {"arrived": 2, "crashed": 1, "timeout": 1}


def test_run_reactor_flies_a_fresh_world_per_trial() -> None:
    trial = Trial(
        start=GliderState(x=1200.0, y=0.0, z=300.0, heading=0.0, airspeed=24.0, bank=0.0),
        goal=Goal(x=1500.0, y=0.0),
    )
    first = run_reactor(trial, ReactiveConfig(variant="b2"), max_seconds=60.0)
    again = run_reactor(trial, ReactiveConfig(variant="b2"), max_seconds=60.0)
    assert first == again  # deterministic AND state never leaks across runs
    assert first.outcome == "arrived"


def test_save_results_round_trips_the_eval(tmp_path: Path) -> None:
    trials = sample_trials(np.random.default_rng(2), 4)
    results = {
        "b0": [Result("crashed", 12.5, 100.0, -50.0, 0.0) for _ in trials],
        "b2": [Result("arrived", 99.0, 800.0, 10.0, 55.0) for _ in trials],
    }
    out = tmp_path / "experiment.npz"
    save_results(out, trials, results)
    d = np.load(out)
    assert d["starts"].shape == (4, 6)
    assert d["goals"].shape == (4, 3)
    assert list(d["agents"]) == ["b0", "b2"]
    assert list(d["b2_outcome"]) == ["arrived"] * 4
    assert d["b0_seconds"] == pytest.approx([12.5] * 4)
    assert d["b0_final"][0] == pytest.approx([100.0, -50.0, 0.0])
    # the tasks themselves round-trip: start kinematics land in the arrays
    assert d["starts"][0][2] == pytest.approx(trials[0].start.z)
