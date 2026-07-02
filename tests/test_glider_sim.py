"""
Property tests for the glider sim.

These assert *physics that must be true for any correct implementation* — not a
replay of what the code currently returns. Each test would fail if the physics
broke, which is what makes it a real guard for agents editing glider_sim.py.

The two guards that matter most here:
  - ENERGY EXACTNESS: total energy changes ONLY through the polar (drain) and
    rising air (gain); the speed<->height exchange itself is lossless. The TE
    vario is defined as the energy-rate instrument, so vario_te * g * dt must
    equal the per-tick energy change EXACTLY. If the bookkeeping drifts, this
    screams.
  - NO FAKE LIFT: pulling up makes the raw vario show climb (your own zoom),
    but vario_te must stay pinned to (air - sink). A reactive policy that
    chases its own zooms would be chasing this exact bug.
"""

import numpy as np
import pytest

from glider_sim import (
    SENSOR_NAMES,
    Glider,
    GliderState,
    Simulation,
    Thermal,
    ThermalMap,
)


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


# --- Glider: the polar + the airframe trade-offs --------------------------
def test_polar_has_min_sink_at_the_right_speed() -> None:
    g = Glider()
    at_min = g.sink_rate(g.v_min_sink, 0.0)
    assert at_min == pytest.approx(g.min_sink)
    assert g.sink_rate(g.v_min_sink - 3.0, 0.0) > at_min  # slower sinks more
    assert g.sink_rate(g.v_min_sink + 6.0, 0.0) > at_min  # faster sinks more


def test_best_glide_is_faster_than_min_sink() -> None:
    # min-sink = stay up longest; best-glide = go farthest per meter lost.
    # Best-glide is ALWAYS faster on a real polar — the fact behind speed-to-fly.
    g = Glider()
    speeds = np.linspace(g.v_stall, 40.0, 200)
    glide_ratio = [v / g.sink_rate(float(v), 0.0) for v in speeds]
    v_best = float(speeds[int(np.argmax(glide_ratio))])
    assert v_best > g.v_min_sink
    assert max(glide_ratio) > 25.0  # trainer-class performance, not a brick


def test_steeper_bank_sinks_faster_and_turns_tighter() -> None:
    # THE core tension of soaring, now speed-aware.
    g = Glider()
    v = 20.0
    shallow, steep = np.radians(20.0), np.radians(50.0)
    assert g.sink_rate(v, steep) > g.sink_rate(v, shallow) > g.sink_rate(v, 0.0)
    assert g.turn_rate(v, steep) > g.turn_rate(v, shallow)
    # slower flight turns a smaller circle at the same bank (radius = V/omega)
    r_slow = 18.0 / g.turn_rate(18.0, steep)
    r_fast = 30.0 / g.turn_rate(30.0, steep)
    assert r_slow < r_fast


def test_stall_speed_rises_with_bank_and_loading() -> None:
    g = Glider()
    assert g.stall_speed(np.radians(60.0)) == pytest.approx(g.v_stall * np.sqrt(2.0), rel=1e-6)
    heavy = Glider(mass=g.mass_ref * 1.44)  # 44% heavier -> k = 1.2
    assert heavy.stall_speed(0.0) == pytest.approx(g.v_stall * 1.2, rel=1e-6)


# --- Simulation: command lag ----------------------------------------------
def test_bank_chases_command_at_the_roll_rate_never_teleports() -> None:
    sim = _sim(x=0.0, airspeed=25.0)
    sim.step(bank_cmd=np.radians(45.0), pitch_cmd=25.0)
    expected = sim.glider.roll_rate * sim.dt  # one tick's worth of roll
    assert sim.state.bank == pytest.approx(expected)
    assert sim.state.bank < np.radians(10.0)  # nowhere near 45 deg yet


def test_airspeed_chases_pitch_command_at_the_accel_limit() -> None:
    sim = _sim(x=1e6, airspeed=20.0)  # dead air
    sim.step(bank_cmd=0.0, pitch_cmd=30.0)
    assert sim.state.airspeed == pytest.approx(20.0 + sim.glider.accel * sim.dt)


# --- Simulation: the energy game -------------------------------------------
def test_te_vario_is_exactly_the_energy_rate_instrument() -> None:
    # THE bookkeeping guard, valid in ANY air (here: inside a thermal, while
    # maneuvering): per tick, d(E/m) == vario_te * g * dt. The exchange itself
    # must be lossless; only air and polar move total energy.
    sim = _sim(x=20.0, airspeed=24.0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        e_before = G_E(sim.state)
        sim.step(
            bank_cmd=float(rng.uniform(-0.9, 0.9)),
            pitch_cmd=float(rng.uniform(14.0, 35.0)),
        )
        de = G_E(sim.state) - e_before
        assert de == pytest.approx(sim.sense()["vario_te"] * 9.81 * sim.dt, abs=1e-9)


def test_zoom_climbs_but_te_vario_shows_no_fake_lift() -> None:
    # dead air, flying fast, command slow -> the glider zooms: raw vario reads
    # positive (you ARE going up) while vario_te stays negative (the air gave
    # you nothing; you're spending your own speed). The anti-self-deception test.
    sim = _sim(x=1e6, airspeed=32.0)
    sim.step(bank_cmd=0.0, pitch_cmd=15.0)
    panel = sim.sense()
    assert panel["vario"] > 0.5  # the needle shows a healthy climb...
    assert panel["vario_te"] < 0.0  # ...but no energy is being gained
    assert sim.state.z > 500.0  # genuinely higher (bought with speed)


def test_pitch_cannot_create_energy_in_dead_air() -> None:
    # whatever the stick does, total energy in still air only ever drains.
    sim = _sim(x=1e6, airspeed=25.0)
    e = G_E(sim.state)
    rng = np.random.default_rng(1)
    for _ in range(300):
        sim.step(float(rng.uniform(-1.0, 1.0)), float(rng.uniform(14.0, 40.0)))
        e_now = G_E(sim.state)
        assert e_now < e  # strictly downhill, every single tick
        e = e_now


# --- Simulation: the stall --------------------------------------------------
def test_flying_too_slow_stalls_then_recovers() -> None:
    # command a speed below stall: the wing quits (big sink), the nose drops
    # (speed rebuilds regardless of the stick), and once commanded back to a
    # sane speed the glider flies again.
    sim = _sim(x=1e6, airspeed=17.0)
    worst_te = 0.0
    for _ in range(100):  # hold the stick back: mush into the stall
        sim.step(bank_cmd=0.0, pitch_cmd=13.0)
        worst_te = min(worst_te, sim.sense()["vario_te"])
    assert worst_te < -2.5  # the stall_sink penalty clearly bit
    for _ in range(200):  # push the nose down: fly out of it
        sim.step(bank_cmd=0.0, pitch_cmd=25.0)
    assert sim.state.airspeed > sim.glider.stall_speed(sim.state.bank)
    assert sim.sense()["vario_te"] > -1.5  # back to ordinary polar sink


# --- Simulation: the ground -------------------------------------------------
def test_hitting_the_ground_ends_the_flight() -> None:
    sim = _sim(x=1e6, airspeed=20.0, z=0.5)
    for _ in range(50):
        sim.step(bank_cmd=0.0, pitch_cmd=20.0)
    assert sim.crashed
    assert sim.state.z == 0.0
    frozen = sim.state
    n_rows = len(sim.history)
    sim.step(bank_cmd=0.5, pitch_cmd=30.0)  # the world stops responding
    assert sim.state is frozen
    assert len(sim.history) == n_rows


# --- Simulation: the panel (sensor firewall) --------------------------------
def test_panel_exposes_instruments_only_never_thermal_truth() -> None:
    # design invariant: sense() = what a physical instrument could read.
    # If a future edit leaks thermal coordinates through it, this screams.
    sim = _sim(x=0.0, airspeed=20.0)
    panel = sim.sense()
    assert set(panel.keys()) == set(SENSOR_NAMES)
    assert panel["vario"] == 0.0  # parked needle before launch
    sim.step(0.4, 20.0)
    assert sim.sense()["vario_te"] == pytest.approx(
        4.0 - sim.glider.sink_rate(20.0, sim.state.bank), abs=0.1
    )  # sitting on a w_peak=4 core: TE vario reads (air - sink)


def test_history_records_raw_commands_and_the_panel() -> None:
    sim = _sim(x=0.0, airspeed=20.0)
    sim.step(bank_cmd=9.9, pitch_cmd=99.0)  # wild asks, way past the envelope
    state0, (bank_cmd, pitch_cmd), panel0 = sim.history[0]
    assert isinstance(state0, GliderState)
    assert (bank_cmd, pitch_cmd) == (9.9, 99.0)  # RAW ask recorded...
    assert abs(sim.state.bank) <= sim.glider.max_bank  # ...clamped in the flying
    assert sim.state.airspeed <= sim.glider.v_max
    assert set(panel0.keys()) == set(SENSOR_NAMES)


# --- the bird cue: wingtip lift asymmetry -----------------------------------
def test_lift_asym_points_toward_the_thermal() -> None:
    # south of the core flying east: the LEFT wingtip (north side) sits in
    # stronger lift -> positive cue -> "turn left" is toward the thermal.
    # Mirrored start -> mirrored cue. THE bird rule, as a sign convention.
    air = ThermalMap(thermals=[Thermal(x0=0.0, y0=0.0, w_peak=4.0, radius=60.0)])
    south = Simulation(
        Glider(), air, GliderState(x=0.0, y=-50.0, z=500.0, heading=0.0, airspeed=20.0, bank=0.0)
    )
    north = Simulation(
        Glider(), air, GliderState(x=0.0, y=50.0, z=500.0, heading=0.0, airspeed=20.0, bank=0.0)
    )
    assert south.sense()["lift_asym"] > 0.1  # thermal to the left
    assert north.sense()["lift_asym"] < -0.1  # thermal to the right
    assert south.sense()["lift_asym"] == pytest.approx(-north.sense()["lift_asym"])


def test_lift_asym_is_zero_when_lift_is_symmetric() -> None:
    # dead centered and heading through the core: both tips feel the same air.
    sim = _sim(x=0.0, airspeed=20.0)  # at the core, heading east: tips at +/-y
    assert sim.sense()["lift_asym"] == pytest.approx(0.0, abs=1e-12)


def test_banking_shrinks_the_lift_asym_baseline() -> None:
    # the span's horizontal footprint scales with cos(bank): a banked glider
    # samples a narrower slice of air; knife-edge would feel no cue at all.
    air = ThermalMap(thermals=[Thermal(x0=0.0, y0=0.0, w_peak=4.0, radius=60.0)])
    level = Simulation(
        Glider(), air, GliderState(x=0.0, y=-50.0, z=500.0, heading=0.0, airspeed=20.0, bank=0.0)
    )
    banked = Simulation(
        Glider(), air, GliderState(x=0.0, y=-50.0, z=500.0, heading=0.0, airspeed=20.0, bank=1.0)
    )
    assert 0.0 < banked.sense()["lift_asym"] < level.sense()["lift_asym"]


# --- the headline regression -------------------------------------------------
def test_on_core_climbs_and_dead_air_sinks() -> None:
    # circling the core at thermalling speed gains altitude; the same commands
    # in dead air lose it. The +206/-91 of v1, surviving real physics.
    core = _sim(x=0.0, airspeed=19.0)
    dead = _sim(x=1e6, airspeed=19.0)
    for _ in range(1200):  # 120 s
        core.step(bank_cmd=np.radians(40.0), pitch_cmd=19.0)
        dead.step(bank_cmd=np.radians(40.0), pitch_cmd=19.0)
    assert core.state.z > 500.0
    assert dead.state.z < 500.0
    assert core.state.z > dead.state.z


# --- helpers -------------------------------------------------------------
def _sim(x: float, airspeed: float, z: float = 500.0) -> Simulation:
    """A standard world: one w_peak=4 thermal at the origin. Spawn at x to be
    on the core (x=0) or in effectively dead air (x=1e6)."""
    air = ThermalMap(thermals=[Thermal(x0=0.0, y0=0.0, w_peak=4.0, radius=60.0)])
    start = GliderState(x=x, y=0.0, z=z, heading=0.0, airspeed=airspeed, bank=0.0)
    return Simulation(Glider(), air, start, dt=0.1)


def G_E(s: GliderState) -> float:
    """Total mechanical energy per unit mass: g*h + V^2/2."""
    return 9.81 * s.z + 0.5 * s.airspeed**2
