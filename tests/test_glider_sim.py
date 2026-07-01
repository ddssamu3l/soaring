"""
Property tests for the glider sim.

These assert *physics that must be true for any correct implementation* — not a
replay of what the code currently returns. That's the point: a test that just
echoes the implementation passes even when the implementation is wrong. Each test
below would fail if the underlying physics broke, which is what makes it a real
guard for agents editing glider_sim.py.
"""

import numpy as np
import pytest

from glider_sim import Glider, GliderState, Simulation, Thermal, ThermalMap


# --- Thermal: the updraft field ------------------------------------------
def test_updraft_is_strongest_at_the_center() -> None:
    t = Thermal(x0=0.0, y0=0.0, w_peak=4.0, radius=60.0)
    center = t.updraft(0.0, 0.0)
    edge = t.updraft(60.0, 0.0)
    far = t.updraft(400.0, 0.0)
    assert center == pytest.approx(4.0)  # peak at the core
    assert center > edge > far  # fades monotonically outward
    assert far == pytest.approx(0.0, abs=1e-3)


def test_updraft_works_elementwise_on_arrays() -> None:
    # the heatmap in fly.py relies on this; a scalar-only impl would break it.
    t = Thermal()
    grid = np.linspace(-100, 100, 50)
    out = t.updraft(grid, np.zeros_like(grid))
    assert isinstance(out, np.ndarray)
    assert out.shape == grid.shape


def test_thermalmap_sums_thermals_and_empty_is_zero() -> None:
    a = Thermal(x0=0.0, y0=0.0, w_peak=3.0)
    b = Thermal(x0=0.0, y0=0.0, w_peak=2.0)
    both = ThermalMap(thermals=[a, b])
    assert both.updraft(0.0, 0.0) == pytest.approx(5.0)
    assert ThermalMap(thermals=[]).updraft(0.0, 0.0) == pytest.approx(0.0)


# --- Glider: the airframe trade-off --------------------------------------
def test_level_flight_sinks_at_base_rate() -> None:
    g = Glider(airspeed=15.0, base_sink=0.7)
    assert g.sink_rate(0.0) == pytest.approx(0.7)  # wings level == base_sink
    assert g.turn_rate(0.0) == pytest.approx(0.0)  # no bank == no turn


def test_steeper_bank_sinks_faster_and_turns_faster() -> None:
    # THE core tension of soaring: tighter turn buys tighter circle at more sink.
    g = Glider()
    shallow, steep = np.radians(20.0), np.radians(50.0)
    assert g.sink_rate(steep) > g.sink_rate(shallow) > g.sink_rate(0.0)
    assert g.turn_rate(steep) > g.turn_rate(shallow) > g.turn_rate(0.0)


# --- Simulation: the world + its invariants ------------------------------
def test_sense_exposes_only_local_vario_not_thermal_location() -> None:
    # sensor firewall (design invariant): the onboard reading is a LOCAL scalar,
    # never the thermal's coordinates. If a future edit leaks (x0,y0) through
    # sense(), this test screams.
    sim = _sim_on_core()
    reading = sim.sense()
    assert set(reading.keys()) == {"vario"}
    assert reading["vario"] == pytest.approx(4.0)  # sitting on a w_peak=4 core


def test_history_records_one_row_per_step() -> None:
    sim = _sim_on_core()
    for _ in range(10):
        sim.step(np.radians(40.0))
    assert len(sim.history) == 10
    state0, bank0, vario0 = sim.history[0]
    assert isinstance(state0, GliderState)
    assert bank0 == pytest.approx(np.radians(40.0))
    assert vario0 == pytest.approx(4.0)  # felt the core on the first tick


def test_on_core_climbs_and_dead_air_sinks() -> None:
    # the headline result, pinned as a regression: a glider circling the core
    # gains energy; the same glider circling far away loses it.
    bank = np.radians(40.0)
    core = _run(_sim_on_core(), bank, steps=1200)
    dead = _run(_sim_far_away(), bank, steps=1200)
    assert core.z > 500.0  # climbed above the 500 m start
    assert dead.z < 500.0  # sank below it
    assert core.z > dead.z


# --- helpers -------------------------------------------------------------
def _sim_on_core() -> Simulation:
    air = ThermalMap(thermals=[Thermal(x0=0.0, y0=0.0, w_peak=4.0, radius=60.0)])
    start = GliderState(x=0.0, y=0.0, z=500.0, heading=0.0)
    return Simulation(Glider(), air, start, dt=0.1)


def _sim_far_away() -> Simulation:
    air = ThermalMap(thermals=[Thermal(x0=0.0, y0=0.0, w_peak=4.0, radius=60.0)])
    start = GliderState(x=-100.0, y=0.0, z=500.0, heading=0.0)
    return Simulation(Glider(), air, start, dt=0.1)


def _run(sim: Simulation, bank: float, steps: int) -> GliderState:
    state = sim.state
    for _ in range(steps):
        state = sim.step(bank)
    return state
