"""
Property tests for the data factory (data_gen.py).

Same philosophy as test_glider_sim.py: assert what must be TRUE of any correct
dataset, not what the code happens to output. The two guards that matter most:

  - REPLAY: every logged (true_state, action, true_next_state) row must
    reproduce exactly through the real physics. A dataset that drifts from
    step() would train the MLP on a world that doesn't exist -- and ML fails
    silently, so nothing else would catch it.
  - FIREWALL: the saved file must contain zero thermal truth. If (x0, y0,
    w_peak, radius) ever leak into the dataset, every downstream "planning
    beats reacting" result is fake. `true_*` arrays are the answer key for
    EVALUATION; model food is sensors + actions only.
"""

from pathlib import Path

import numpy as np

from data_gen import (
    fly_rollout,
    generate_dataset,
    make_world,
    random_start,
    sample_commands,
)
from glider_sim import ACTION_NAMES, SENSOR_NAMES, STATE_NAMES, GliderState, Simulation

# small-but-real settings so the suite stays fast (3 x 40 = 120 rows)
SMALL = {"n_rollouts": 3, "steps_per_rollout": 40, "hold_steps": 5, "seed": 7}


# --- shape + basic sanity --------------------------------------------------
def test_shapes_line_up_and_values_are_finite() -> None:
    d = generate_dataset(**SMALL, out_path=None)
    n = len(d["episode"])
    assert n > 0
    assert d["true_states"].shape == (n, len(STATE_NAMES))
    assert d["true_next_states"].shape == (n, len(STATE_NAMES))
    assert d["actions"].shape == (n, len(ACTION_NAMES))
    assert d["sensors"].shape == (n, len(SENSOR_NAMES))
    for key in ("true_states", "actions", "sensors", "true_next_states"):
        assert np.all(np.isfinite(d[key])), f"non-finite values in {key}"
    # channel-name arrays ship inside the dataset (self-describing file)
    assert tuple(d["state_names"]) == STATE_NAMES
    assert tuple(d["action_names"]) == ACTION_NAMES
    assert tuple(d["sensor_names"]) == SENSOR_NAMES


def test_commands_are_piecewise_constant_and_bounded() -> None:
    # held blocks (so consequences complete), both turn directions, bounded.
    rng = np.random.default_rng(0)
    banks, speeds = sample_commands(rng, n_steps=20, hold_steps=5)
    assert banks.shape == speeds.shape == (20,)
    assert np.all(np.abs(banks) <= np.radians(50.0))
    for block in banks.reshape(4, 5):
        assert np.all(block == block[0])  # constant within each block
    for block in speeds.reshape(4, 5):
        assert np.all(block == block[0])


# --- determinism (a dataset you can't reproduce is one you can't debug) -----
def test_same_seed_reproduces_identical_data() -> None:
    d1 = generate_dataset(**SMALL, out_path=None)
    d2 = generate_dataset(**SMALL, out_path=None)
    for key in d1:
        assert np.array_equal(d1[key], d2[key]), f"seed did not pin {key}"


def test_different_seed_gives_different_data() -> None:
    d1 = generate_dataset(**SMALL, out_path=None)
    d2 = generate_dataset(**{**SMALL, "seed": 8}, out_path=None)
    assert not np.array_equal(d1["true_states"], d2["true_states"])


# --- the REPLAY guard: logged rows == the real physics ----------------------
def test_rows_chain_within_an_episode() -> None:
    # true_next_states[i] must BE true_states[i+1] inside one rollout -- the
    # rows of a trajectory chain together.
    d = generate_dataset(**SMALL, out_path=None)
    ep0 = d["episode"] == 0
    states, nexts = d["true_states"][ep0], d["true_next_states"][ep0]
    assert np.array_equal(nexts[:-1], states[1:])


def test_sensors_mirror_true_kinematics_in_the_fixed_field() -> None:
    # self is fully observable through the panel: the first six sensor
    # channels (GPS/altimeter/compass/ASI/attitude) equal the true kinematics
    # exactly. (At step 5 the WORLD goes hidden -- not the self.)
    assert SENSOR_NAMES[: len(STATE_NAMES)] == STATE_NAMES
    d = generate_dataset(**SMALL, out_path=None)
    assert np.array_equal(d["sensors"][:, : len(STATE_NAMES)], d["true_states"])


def test_logged_rows_replay_exactly_through_the_real_physics() -> None:
    # rebuild the same world, re-fly episode 0 from its logged start using the
    # logged RAW commands, and demand bit-identical next states. This pins the
    # dataset to step() -- the factory cannot silently drift from the sim.
    d = generate_dataset(**SMALL, out_path=None)
    ep0 = d["episode"] == 0
    states, actions, nexts = d["true_states"][ep0], d["actions"][ep0], d["true_next_states"][ep0]

    glider, air = make_world()
    start = GliderState(*(float(v) for v in states[0]))
    sim = Simulation(glider, air, start, dt=0.1)
    for i in range(len(actions)):
        nxt = sim.step(float(actions[i, 0]), float(actions[i, 1]))
        replayed = np.array([getattr(nxt, n) for n in STATE_NAMES])
        assert np.array_equal(replayed, nexts[i]), f"replay diverged at row {i}"


# --- crashes end episodes early ----------------------------------------------
def test_ground_impact_ends_the_episode_early() -> None:
    glider, air = make_world()
    start = GliderState(x=1e6, y=0.0, z=2.0, heading=0.0, airspeed=20.0, bank=0.0)
    sim = Simulation(glider, air, start, dt=0.1)
    rng = np.random.default_rng(0)
    banks, speeds = sample_commands(rng, n_steps=100, hold_steps=5)
    fly_rollout(sim, banks, speeds)
    assert sim.crashed
    assert len(sim.history) < 100  # episode cut short
    assert sim.state.z == 0.0


def test_random_starts_are_airworthy() -> None:
    # spawns must be flying (above stall, wings level, off the ground)
    rng = np.random.default_rng(3)
    glider, _ = make_world()
    for _ in range(50):
        s = random_start(rng)
        assert s.airspeed > glider.stall_speed(s.bank)
        assert s.z > 0.0
        assert s.bank == 0.0


def test_world_is_decision_forcing() -> None:
    """Pin the t3 world's SHAPE (not its exact numbers): real lift at home
    thermal A and far thermal B, real SINK all across the band between them,
    calm air off-corridor. If a tweak ever un-walls the corridor, the 'reach
    the goal' task silently degenerates back into 'just glide straight'."""
    _, air = make_world()
    assert float(air.updraft(0.0, 0.0)) > 3.0  # A: a strong home climb
    # a NORMAL thermalling circle must fit inside A: at best-glide speed and
    # ~40 deg bank the turn radius is ~70 m, and the lift out there must still
    # beat the banked sink (~1 m/s) -- t3 flying proved a core you can only
    # work with a perfect min-speed circle is a core the planner cannot use
    assert float(air.updraft(70.0, 0.0)) > 2.0
    assert float(air.updraft(1000.0, 0.0)) > 2.5  # B: a real top-up
    for y in (-500.0, -250.0, 0.0, 250.0, 500.0):
        assert float(air.updraft(500.0, y)) < -2.5  # the WALL sinks hard all the way across
    assert abs(float(air.updraft(500.0, 1200.0))) < 0.3  # ...but ends; air calms outside


def test_random_starts_cover_the_task_corridor() -> None:
    """Spawns must span the whole corridor (A, the band, past B) and reach down
    to near-ground: the planner may only be routed through air the model has
    actually flown, and the decision task starts LOW (crashes included)."""
    rng = np.random.default_rng(7)
    starts = [random_start(rng) for _ in range(300)]
    xs = np.array([s.x for s in starts])
    zs = np.array([s.z for s in starts])
    assert xs.min() < -100.0 and xs.max() > 1100.0  # both thermals and beyond
    assert zs.min() < 80.0  # near-ground spawns exist...
    assert zs.max() > 450.0  # ...and high cruise too


def test_dataset_carries_a_live_wingtip_cue() -> None:
    # the bird cue must actually vary in real rollouts (a dead column would
    # mean the factory or the sensor silently broke) and stay bounded by the
    # field itself (|left - right| can never exceed the peak updraft).
    d = generate_dataset(**SMALL, out_path=None)
    cue = d["sensors"][:, list(SENSOR_NAMES).index("lift_asym")]
    assert np.any(np.abs(cue) > 0.01)  # alive: rollouts felt the gradient
    assert np.all(np.abs(cue) < 4.0)  # bounded by w_peak


# --- the FIREWALL guard: no thermal truth in the artifact --------------------
def test_saved_file_contains_no_thermal_truth(tmp_path: Path) -> None:
    out = tmp_path / "d.npz"
    generate_dataset(**SMALL, out_path=out)
    with np.load(out) as f:
        keys = set(f.keys())
        # exactly the allowed channels: answer key (true_*), model food
        # (sensors/actions), bookkeeping, and airframe/sim constants.
        assert keys == {
            "true_states",
            "actions",
            "sensors",
            "true_next_states",
            "episode",
            "state_names",
            "action_names",
            "sensor_names",
            "glider_params",
            "glider_param_names",
            "dt",
            "seed",
        }
        # belt-and-suspenders: name every forbidden field explicitly, so the
        # intent survives even if the allowed set above is ever loosened.
        forbidden = {"x0", "y0", "w_peak", "radius", "thermals", "wind"}
        assert not (keys & forbidden)
        for name_key in ("sensor_names", "glider_param_names"):
            assert not (set(f[name_key]) & forbidden)
        # model-visible feature widths: 8 sensors + 2 actions. A widened
        # sensors array would be the classic silent leak.
        assert f["sensors"].shape[1] == len(SENSOR_NAMES)
        assert f["actions"].shape[1] == len(ACTION_NAMES)
