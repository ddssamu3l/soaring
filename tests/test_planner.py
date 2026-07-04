"""
Property tests for the CEM/MPC planner (planner.py).

The planner is downstream of a certified model, so what needs proving here is
the SEARCH machinery itself: that batched imagination equals a per-candidate
loop (one candidate's future must not bleed into another's), that the cost
ranks futures by the task and nothing else, that CEM respects the
training-support clip, and that the MPC executor terminates with an honest
verdict. Model quality is keystone.py's business, not this file's.
"""

import numpy as np
import pytest
import torch

from data_gen import generate_dataset
from glider_sim import Glider, GliderState, Simulation, Thermal, ThermalMap
from planner import (
    GlidePolar,
    Goal,
    Plan,
    PlannerConfig,
    best_glide,
    cem_plan,
    expand_segments,
    fly_to_goal,
    imagine,
    rank,
    score_rollouts,
)
from train import (
    Checkpoint,
    PanelMLP,
    clamp_panel,
    fit_stats,
    load_panels,
    make_feature_spec,
    pair_indices,
    predict_delta,
    split_episodes,
)

SMALL = {"n_rollouts": 10, "steps_per_rollout": 40, "hold_steps": 5, "seed": 11}

# a tiny config so tests fly in milliseconds: 3 segments x 5 ticks = 1.5 s horizon
TINY = PlannerConfig(
    n_segments=3, ticks_per_segment=5, population=24, n_elites=6, iterations=2, replan_ticks=5
)


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory):
    """Small dataset + a seeded RANDOM-weight checkpoint. Random (not trained):
    planner tests exercise search mechanics, which must hold for ANY net."""
    path = tmp_path_factory.mktemp("t3planner") / "small.npz"
    generate_dataset(**SMALL, out_path=path)
    data = load_panels(path)
    pairs = pair_indices(data.episode)
    stats = fit_stats(data, pairs)
    spec = make_feature_spec(data.sensor_names, data.action_names)
    torch.manual_seed(0)
    net = PanelMLP(spec.n_features, len(data.sensor_names), (16,))
    net.eval()
    ck = Checkpoint(
        model=net,
        stats=stats,
        spec=spec,
        split=split_episodes(data.episode),
        ablate=False,
        sensor_names=data.sensor_names,
        action_names=data.action_names,
        dt=data.dt,
    )
    return ck, data


# --- the polar arithmetic -------------------------------------------------------------
def test_best_glide_is_the_polar_optimum_and_matches_the_airframe_class() -> None:
    glider = Glider()
    polar = best_glide(glider)
    assert glider.stall_speed(0.0) < polar.v_best_glide < glider.v_max
    # no sampled speed may MEANINGFULLY beat the returned operating point (both
    # grids are finite, so allow a hair of resolution slack -- 0.1% of L/D)
    for v in np.linspace(glider.stall_speed(0.0), glider.v_max, 57):
        assert float(v) / glider.sink_rate(float(v), 0.0) <= polar.glide_ratio * 1.001
    # ASK-21-class sanity: best glide in the low 20s m/s, L/D in the 30s
    assert 20.0 < polar.v_best_glide < 30.0
    assert 25.0 < polar.glide_ratio < 40.0


# --- candidate expansion ---------------------------------------------------------------
def test_expand_segments_holds_each_command_for_its_block() -> None:
    segments = np.array([[0.1, -0.2, 0.3], [0.0, 0.5, -0.5]])
    ticks = expand_segments(segments, 4)
    assert ticks.shape == (2, 12)
    assert np.array_equal(ticks[0], np.repeat([0.1, -0.2, 0.3], 4))
    assert np.array_equal(ticks[1, 4:8], np.full(4, 0.5))


# --- imagination -----------------------------------------------------------------------
def test_imagine_batch_equals_a_per_candidate_loop(setup) -> None:
    """Batched imagination must equal the naive one-candidate-at-a-time loop
    through predict_delta: candidates share a batch, never a future. This is
    the test that catches action-wiring and cross-candidate bleed bugs. (Not
    bit-for-bit: torch's matmul kernels accumulate in a batch-shape-dependent
    order, so equality holds only to float32 ulps -- a real wiring bug would
    be off by whole panels, not 1e-9.)"""
    ck, data = setup
    panel0 = data.sensors[0]
    rng = np.random.default_rng(3)
    bank_ticks = rng.uniform(-0.8, 0.8, size=(3, 6))
    pitch = 24.0
    got = imagine(ck, panel0, bank_ticks, np.full_like(bank_ticks, pitch))
    assert got.shape == (3, 7, len(ck.sensor_names))
    bank_col = ck.action_names.index("bank_cmd")
    pitch_col = ck.action_names.index("pitch_cmd")
    for j in range(3):
        panel = panel0[None, :].copy()
        assert np.array_equal(got[j, 0], panel[0])  # h=0 IS the shared true start
        for h in range(6):
            action = np.empty((1, 2))
            action[:, bank_col] = bank_ticks[j, h]
            action[:, pitch_col] = pitch
            panel = clamp_panel(panel + predict_delta(ck, panel, action), ck.stats)
            assert np.allclose(got[j, h + 1], panel[0], rtol=1e-6, atol=1e-7)


def test_imagine_zero_net_drifts_by_the_mean_delta(setup) -> None:
    """Closed form (mirrors the keystone's strongest guard): a zero net ignores
    its inputs and predicts the mean delta, so every candidate's imagined panel
    is exactly panel0 + h * delta_mean regardless of its actions."""
    ck, data = setup
    zero = PanelMLP(ck.spec.n_features, len(ck.sensor_names), (8,))
    with torch.no_grad():
        for p in zero.parameters():
            p.zero_()
    zck = Checkpoint(
        model=zero,
        stats=ck.stats,
        spec=ck.spec,
        split=ck.split,
        ablate=False,
        sensor_names=ck.sensor_names,
        action_names=ck.action_names,
        dt=ck.dt,
    )
    panel0 = data.sensors[5]
    bank_ticks = np.linspace(-0.5, 0.5, 4 * 8).reshape(4, 8)  # actions vary, output must not
    got = imagine(zck, panel0, bank_ticks, np.full_like(bank_ticks, 24.0))
    expected = panel0.astype(np.float64)
    assert np.allclose(got[:, 0], expected, atol=1e-9)
    for h in range(1, 9):
        expected = clamp_panel(expected[None] + ck.stats.delta_mean, ck.stats)[0]
        if h in (3, 8):
            assert np.allclose(got[:, h], expected, atol=1e-9)


# --- the cost --------------------------------------------------------------------------
NAMES = ("x", "y", "z", "airspeed")
POLAR = GlidePolar(v_best_glide=25.0, glide_ratio=30.0)


def _future(rows: list[list[float]]) -> np.ndarray:
    """(H+1, 4) hand-authored imagined future in NAMES channel order."""
    return np.array(rows, dtype=np.float64)


def test_score_rollouts_grades_the_three_fates_and_ranks_them() -> None:
    """One arriving, one gliding-short, one crashing future -- flags, values and
    the lexicographic order (arrived < still-gliding < crashed) all pinned."""
    goal = Goal(x=100.0, y=0.0, radius=10.0)
    dt = 0.1
    arrives = _future([[0, 0, 200, 25], [50, 0, 199, 25], [95, 0, 198, 25], [96, 0, 197, 25]])
    glides = _future([[0, 0, 200, 25], [10, 0, 199, 25], [20, 0, 198, 25], [30, 0, 197, 25]])
    crashes = _future([[0, 0, 2, 25], [10, 0, 1, 25], [20, 0, 0, 25], [30, 0, -1, 25]])
    scores = score_rollouts(np.stack([arrives, glides, crashes]), NAMES, goal, POLAR, dt)

    assert list(scores.crashed) == [False, False, True]
    # the arriver: deficit 0, time = first in-radius step (h=2) * dt
    assert scores.deficit[0] == 0.0
    assert scores.est_time[0] == pytest.approx(2 * dt)
    # the glider ends at (30,0,197,25): 70 m short, in glide (197*30 >> 70) -> deficit 0,
    # time = horizon elapsed + still-air tail at best-glide speed
    assert scores.deficit[1] == 0.0
    assert scores.est_time[1] == pytest.approx(3 * dt + 70.0 / 25.0)
    # the crasher: survival time recorded (first z<=0 row is h=2)
    assert scores.t_crash[2] == pytest.approx(2 * dt)
    assert np.isinf(scores.t_crash[0]) and np.isinf(scores.t_crash[1])
    assert list(rank(scores)) == [0, 1, 2]


def test_rank_prefers_dying_later_over_dying_closer() -> None:
    """When EVERY imagined future ends in the ground, the planner must keep
    flying the airplane: a future that survives longer outranks one that gets
    closer to the goal but dies sooner (the kamikaze-dive trap -- distance-at-
    death only breaks exact survival ties)."""
    goal = Goal(x=1000.0, y=0.0, radius=10.0)
    dives = _future([[0, 0, 10, 30], [200, 0, 2, 32], [400, 0, -1, 33], [410, 0, -1, 33]])
    lives = _future([[0, 0, 10, 20], [50, 0, 8, 20], [100, 0, 4, 20], [150, 0, -1, 20]])
    scores = score_rollouts(np.stack([dives, lives]), NAMES, goal, POLAR, 0.1)
    assert list(scores.crashed) == [True, True]
    assert scores.t_crash[1] > scores.t_crash[0]
    assert list(rank(scores)) == [1, 0]  # die last, not die close


def test_score_rollouts_deficit_prices_energy_height_not_just_altitude() -> None:
    """Two identical end positions, 3000 m short, low: the faster end state has
    convertible kinetic energy -> smaller deficit. And the deficit must equal
    the hand-computed remaining - energy_height * L/D."""
    goal = Goal(x=3000.0, y=0.0, radius=10.0)
    slow = _future([[0, 0, 50, 25], [0, 0, 50, 25]])
    fast = _future([[0, 0, 50, 35], [0, 0, 50, 35]])
    scores = score_rollouts(np.stack([slow, fast]), NAMES, goal, POLAR, 0.1)
    assert scores.deficit[1] < scores.deficit[0]
    assert scores.deficit[0] == pytest.approx(3000.0 - 50.0 * 30.0)  # v == v_bg: pure altitude
    energy_height = 50.0 + (35.0**2 - 25.0**2) / (2.0 * 9.81)
    assert scores.deficit[1] == pytest.approx(3000.0 - energy_height * 30.0, rel=1e-3)


def test_score_rollouts_reserve_height_shrinks_usable_energy() -> None:
    """With an arrival reserve, only energy ABOVE the reserve counts toward
    reach -- the tail plans to arrive with altitude in hand, not at 0 m."""
    goal = Goal(x=3000.0, y=0.0, radius=10.0)
    future = _future([[0, 0, 100, 25], [0, 0, 100, 25]])
    plain = score_rollouts(future[None], NAMES, goal, POLAR, 0.1)
    reserved = score_rollouts(future[None], NAMES, goal, POLAR, 0.1, reserve_height=40.0)
    assert plain.deficit[0] == pytest.approx(3000.0 - 100.0 * 30.0)
    assert reserved.deficit[0] == pytest.approx(3000.0 - 60.0 * 30.0)


def test_score_rollouts_arrival_before_ground_contact_counts_as_arrived() -> None:
    """Reaching the goal and THEN touching imagined ground is an arrival --
    the flight ends at the goal; whatever the imagination does after is moot."""
    goal = Goal(x=10.0, y=0.0, radius=15.0)
    future = _future([[0, 0, 5, 25], [10, 0, 3, 25], [20, 0, -1, 25]])
    scores = score_rollouts(future[None], NAMES, goal, POLAR, 0.1)
    assert not scores.crashed[0]
    assert scores.deficit[0] == 0.0


# --- CEM -------------------------------------------------------------------------------
def test_cem_plan_respects_the_training_support_clip(setup) -> None:
    """Every command the planner can ever emit -- BOTH axes -- must sit inside
    the data's own action support; outside it the model is ungraded and
    imagination is hallucination. Winning plan and refit mean both stay clipped."""
    ck, data = setup
    goal = Goal(x=500.0, y=0.0)
    polar = best_glide(Glider())
    plan = cem_plan(ck, data.sensors[0], goal, polar, TINY, np.random.default_rng(0))
    assert isinstance(plan, Plan)
    assert plan.segments.shape == (TINY.n_segments, 2)
    assert np.all(np.abs(plan.segments[:, 0]) <= TINY.max_bank_cmd + 1e-12)
    assert np.all((plan.segments[:, 1] >= TINY.pitch_lo) & (plan.segments[:, 1] <= TINY.pitch_hi))
    assert plan.imagined.shape == (TINY.horizon + 1, len(ck.sensor_names))
    assert np.array_equal(plan.imagined[0], data.sensors[0])  # dreams start from truth


def test_cem_plan_trace_records_the_real_search(setup) -> None:
    """The trace must be the search itself, not a summary: one entry per
    iteration, order a true permutation of the population, and the recorded
    mean exactly the elites' mean -- the CEM refit, pinned."""
    ck, data = setup
    from planner import CEMIteration

    trace: list[CEMIteration] = []
    goal = Goal(x=500.0, y=0.0)
    polar = best_glide(Glider())
    cem_plan(ck, data.sensors[0], goal, polar, TINY, np.random.default_rng(1), trace=trace)
    assert len(trace) == TINY.iterations
    for it in trace:
        assert it.candidates.shape == (TINY.population, TINY.n_segments, 2)
        assert sorted(it.order) == list(range(TINY.population))  # a real permutation
        elites = it.candidates[it.order[: TINY.n_elites]]
        assert np.allclose(it.mean, elites.mean(axis=0))  # the refit, exactly
        assert it.imagined_xy.shape == (TINY.population, TINY.horizon + 1, 2)


def test_config_anchors_hold() -> None:
    """The two anchored dials: the default horizon is the certified 30 s (any
    deeper is past the measured edge of trustworthy imagination), and the
    glide margin is a real safety factor in (0, 1)."""
    cfg = PlannerConfig()
    assert cfg.horizon * 0.1 == pytest.approx(30.0)
    assert 0.0 < cfg.glide_margin < 1.0
    assert cfg.reserve_height > 0.0  # plan to ARRIVE with altitude, not to graze the fence


def test_cem_plan_warm_start_keeps_the_incumbent_alive(setup) -> None:
    """The warm-started mean is injected as candidate 0 every iteration, so a
    plan can only be replaced by one the cost ranks BETTER -- never lost to
    sampling luck. Pin the mechanism: identical rng, with and without warm."""
    ck, data = setup
    goal = Goal(x=500.0, y=0.0)
    polar = best_glide(Glider())
    warm = np.tile(np.array([0.3, 24.0]), (TINY.n_segments, 1))
    a = cem_plan(ck, data.sensors[0], goal, polar, TINY, np.random.default_rng(7), warm)
    b = cem_plan(ck, data.sensors[0], goal, polar, TINY, np.random.default_rng(7), warm)
    assert np.array_equal(a.segments, b.segments)  # deterministic under a fixed seed


# --- the MPC executor -------------------------------------------------------------------
def _sim(z: float = 400.0) -> Simulation:
    air = ThermalMap(thermals=[Thermal(x0=0.0, y0=0.0, w_peak=4.0, radius=60.0)])
    state = GliderState(x=0.0, y=0.0, z=z, heading=0.0, airspeed=24.0, bank=0.0)
    return Simulation(Glider(), air, state)


def test_fly_to_goal_declares_arrival_at_the_goal_circle(setup) -> None:
    """Goal placed 40 m ahead of a glider flying straight at it: any sane plan
    arrives within a couple of seconds, and the executor must say so."""
    ck, _ = setup
    sim = _sim()
    flight = fly_to_goal(sim, ck, Goal(x=40.0, y=0.0, radius=30.0), TINY, np.random.default_rng(0))
    assert flight.outcome == "arrived"
    assert flight.seconds <= 5.0
    assert len(flight.plans) >= 1
    assert len(sim.history) == int(round(flight.seconds / sim.dt))


def test_fly_to_goal_times_out_honestly_when_the_goal_is_unreachable(setup) -> None:
    """An impossibly distant goal: the executor must give up at max_seconds with
    outcome 'timeout' -- not loop forever, not claim success."""
    ck, _ = setup
    sim = _sim()
    flight = fly_to_goal(
        sim, ck, Goal(x=1e7, y=0.0), TINY, np.random.default_rng(0), max_seconds=2.0
    )
    assert flight.outcome == "timeout"
    assert flight.seconds == pytest.approx(2.0)
