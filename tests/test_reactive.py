"""
tests/test_reactive.py -- what must be TRUE of any correct reflex pilot.

The load-bearing one is the fairness pin: the reactor's deficit arithmetic
must be EXACTLY the planner scorer's (same formula, same constants), because
t4's claim -- "the only difference between the agents is imagination" -- is
only as true as that equality.
"""

import math

import numpy as np
import pytest

from data_gen import make_world
from glider_sim import SENSOR_NAMES, GliderState, Simulation, ThermalMap
from planner import GlidePolar, Goal, PlannerConfig, best_glide, score_rollouts
from reactive import ReactiveConfig, ReactivePilot, fly_reactive


def _panel(**overrides: float) -> dict[str, float]:
    base: dict[str, float] = {
        "x": 0.0,
        "y": 0.0,
        "z": 300.0,
        "heading": 0.0,
        "airspeed": 24.0,
        "bank": 0.0,
        "vario": 0.0,
        "vario_te": 0.0,
        "lift_asym": 0.0,
    }
    base.update(overrides)
    return base


def _pilot(cfg: ReactiveConfig, goal: Goal) -> ReactivePilot:
    glider, _ = make_world()
    return ReactivePilot(cfg, best_glide(glider), goal, dt=0.1)


def test_deficit_matches_the_planner_scorers_arithmetic() -> None:
    """THE FAIRNESS PIN: pilot.deficit == score_rollouts' deficit on the same
    panel, with the same reserve/margin -- the two agents run one arithmetic."""
    glider, _ = make_world()
    polar = best_glide(glider)
    pcfg = PlannerConfig()
    goal = Goal(x=800.0, y=-120.0)
    # the scorer takes the margin pre-folded into the polar (planner.cem_plan
    # builds this exact scored polar); the pilot folds it in itself
    scored = GlidePolar(polar.v_best_glide, polar.glide_ratio * pcfg.glide_margin)
    pilot = ReactivePilot(
        ReactiveConfig(reserve_height=pcfg.reserve_height, glide_margin=pcfg.glide_margin),
        polar,
        goal,
        dt=0.1,
    )
    for z, v in [(50.0, 20.0), (200.0, 24.3), (400.0, 30.0), (10.0, 18.0)]:
        panel = _panel(z=z, airspeed=v)
        # a 2-row "rollout" that just sits at this panel: both solvency
        # checkpoints see the identical state, so its deficit IS the formula
        row = [panel[name] for name in SENSOR_NAMES]
        imagined = np.array([[row, row]])
        scores = score_rollouts(
            imagined,
            SENSOR_NAMES,
            goal,
            scored,
            dt=0.1,
            ground_z=-1.0,
            reserve_height=pcfg.reserve_height,
        )
        assert scores.deficit[0] == pytest.approx(pilot.deficit(panel))


def test_glide_steering_closes_the_heading_error_through_the_real_physics() -> None:
    """Sign-proof by construction: aim at a goal 90 degrees off the nose, fly
    the commanded banks through the REAL sim, and the heading error must
    shrink -- if the bank sign convention were flipped it would grow."""
    glider, _ = make_world()
    still_air = ThermalMap(thermals=[])
    start = GliderState(x=0.0, y=0.0, z=400.0, heading=0.0, airspeed=24.0, bank=0.0)
    goal = Goal(x=0.0, y=2000.0)  # due north; nose starts due east
    sim = Simulation(glider, still_air, start)
    pilot = ReactivePilot(ReactiveConfig(variant="b0"), best_glide(glider), goal, sim.dt)
    err0 = abs(math.pi / 2 - start.heading)
    for _ in range(80):
        bank_cmd, pitch_cmd = pilot.act(sim.sense())
        sim.step(bank_cmd, pitch_cmd)
    s = sim.state
    bearing = math.atan2(goal.y - s.y, goal.x - s.x)
    err = abs(math.remainder(bearing - s.heading, 2.0 * math.pi))
    assert err < err0 / 4  # decisively turning the right way


def test_b0_never_circles_even_in_strong_lift() -> None:
    pilot = _pilot(ReactiveConfig(variant="b0"), Goal(x=2000.0, y=0.0))
    for _ in range(50):
        pilot.act(_panel(z=50.0, vario_te=4.0, lift_asym=0.5))
    assert pilot.mode == "glide"


def test_b1_circles_toward_the_lifting_wing_and_gives_up_when_lift_dies() -> None:
    cfg = ReactiveConfig(variant="b1", vario_enter=0.5, vario_exit=0.0, exit_patience=1.0)
    pilot = _pilot(cfg, Goal(x=2000.0, y=0.0))
    # lift on the RIGHT wing (asym < 0) -> circle right (negative bank)
    bank, speed = pilot.act(_panel(vario_te=1.5, lift_asym=-0.2))
    assert pilot.mode == "climb"
    assert pilot.turn_sign == -1.0
    assert bank < 0.0
    assert speed == pytest.approx(cfg.v_climb)
    # b1 has no arithmetic: strong lift holds it in the circle indefinitely
    for _ in range(100):
        pilot.act(_panel(vario_te=1.5, lift_asym=-0.2))
    assert pilot.mode == "climb"
    # ...until the lift dies for longer than the patience window
    for _ in range(11):
        pilot.act(_panel(vario_te=-0.5))
    assert pilot.mode == "glide"


def test_b2_only_stops_for_lift_while_the_glide_is_not_made() -> None:
    goal = Goal(x=2000.0, y=0.0)
    pilot = _pilot(ReactiveConfig(variant="b2"), goal)
    # glide already made (high, close in energy terms): lift is a distraction
    pilot.act(_panel(z=500.0, x=1000.0, vario_te=3.0, lift_asym=0.3))
    assert pilot.mode == "glide"
    # same lift with a real deficit: stop and climb
    pilot.act(_panel(z=60.0, x=0.0, vario_te=3.0, lift_asym=0.3))
    assert pilot.mode == "climb"
    # feed a rising panel until the arithmetic says "glide made" -> commit
    z = 60.0
    while pilot.mode == "climb" and z < 1000.0:
        z += 2.0
        pilot.act(_panel(z=z, vario_te=3.0, lift_asym=0.3))
    assert pilot.mode == "glide"
    assert z < 1000.0  # it left BECAUSE the deficit closed, not the loop cap


def test_centering_tightens_onto_a_core_inside_the_circle() -> None:
    cfg = ReactiveConfig(variant="b1", vario_enter=0.5)
    pilot = _pilot(cfg, Goal(x=2000.0, y=0.0))
    # enter circling LEFT (lift on the left wing)
    pilot.act(_panel(vario_te=1.5, lift_asym=0.3))
    assert pilot.turn_sign == 1.0
    bank_in, _ = pilot.act(_panel(vario_te=1.5, lift_asym=0.3))  # core inside
    bank_out, _ = pilot.act(_panel(vario_te=1.5, lift_asym=-0.3))  # core outside
    assert bank_in > cfg.bank_circle  # tighten toward the core
    assert bank_out < cfg.bank_circle  # flatten to drift back out
    assert bank_out >= cfg.bank_flat  # ...but never past the clips


def test_reactor_flies_an_easy_final_glide_to_arrival() -> None:
    """Integration: on a task its arithmetic says is made, b2 just goes."""
    glider, air = make_world()
    start = GliderState(x=1200.0, y=0.0, z=300.0, heading=0.0, airspeed=24.0, bank=0.0)
    goal = Goal(x=1500.0, y=0.0)
    sim = Simulation(glider, air, start)
    pilot = ReactivePilot(ReactiveConfig(variant="b2"), best_glide(glider), goal, sim.dt)
    flight = fly_reactive(sim, pilot, goal, max_seconds=60.0)
    assert flight.outcome == "arrived"
    assert flight.seconds < 30.0


def test_b1_parks_in_the_home_thermal_it_never_needed_to_leave() -> None:
    """The predicted b1 failure, verified in the real world: no arithmetic
    means no reason to leave good lift -- it circles A until the clock runs
    out instead of flying to a goal its altitude already made."""
    glider, air = make_world()
    start = GliderState(x=0.0, y=0.0, z=400.0, heading=0.0, airspeed=20.0, bank=0.0)
    goal = Goal(x=-800.0, y=0.0)  # behind it, no wall in the way, easily made
    sim = Simulation(glider, air, start)
    pilot = ReactivePilot(ReactiveConfig(variant="b1"), best_glide(glider), goal, sim.dt)
    flight = fly_reactive(sim, pilot, goal, max_seconds=60.0)
    assert flight.outcome == "timeout"
    assert math.hypot(sim.state.x, sim.state.y) < 400.0  # still orbiting A
