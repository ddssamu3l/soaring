"""
Property tests for the data factory (data_gen.py).

Same philosophy as test_glider_sim.py: assert what must be TRUE of any correct
dataset, not what the code happens to output. The two guards that matter most:

  - REPLAY: every logged (state, action, next_state) row must reproduce exactly
    through the real physics. A dataset that drifts from step() would train the
    MLP on a world that doesn't exist -- and ML fails silently, so nothing else
    would catch it.
  - FIREWALL: the saved file must contain zero thermal truth. If (x0, y0,
    w_peak, radius) ever leak into the dataset, every downstream "planning
    beats reacting" result is fake.
"""

from pathlib import Path

import numpy as np

from data_gen import generate_dataset, make_world, sample_banks
from glider_sim import GliderState, Simulation

# small-but-real settings so the suite stays fast (3 x 40 = 120 rows)
SMALL = {"n_rollouts": 3, "steps_per_rollout": 40, "hold_steps": 5, "seed": 7}


# --- shape + basic sanity --------------------------------------------------
def test_shapes_line_up_and_values_are_finite() -> None:
    d = generate_dataset(**SMALL, out_path=None)
    n = SMALL["n_rollouts"] * SMALL["steps_per_rollout"]
    assert d["states"].shape == (n, 4)
    assert d["next_states"].shape == (n, 4)
    assert d["actions"].shape == (n,)
    assert d["varios"].shape == (n,)
    assert d["episode"].shape == (n,)
    for key in ("states", "actions", "next_states", "varios"):
        assert np.all(np.isfinite(d[key])), f"non-finite values in {key}"


def test_banks_are_piecewise_constant_and_bounded() -> None:
    # the action schedule must give sustained arcs (constant within a block),
    # symmetric coverage (both signs), and never exceed the requested bank.
    rng = np.random.default_rng(0)
    banks = sample_banks(rng, n_steps=20, hold_steps=5, max_bank=0.8)
    assert banks.shape == (20,)
    assert np.all(np.abs(banks) <= 0.8)
    for block in banks.reshape(4, 5):
        assert np.all(block == block[0])  # constant within each block


# --- determinism (a dataset you can't reproduce is a dataset you can't debug)
def test_same_seed_reproduces_identical_data() -> None:
    d1 = generate_dataset(**SMALL, out_path=None)
    d2 = generate_dataset(**SMALL, out_path=None)
    for key in d1:
        assert np.array_equal(d1[key], d2[key]), f"seed did not pin {key}"


def test_different_seed_gives_different_data() -> None:
    d1 = generate_dataset(**SMALL, out_path=None)
    d2 = generate_dataset(**{**SMALL, "seed": 8}, out_path=None)
    assert not np.array_equal(d1["states"], d2["states"])


# --- the REPLAY guard: logged rows == the real physics ----------------------
def test_rows_chain_within_an_episode() -> None:
    # next_states[i] must BE states[i+1] inside one rollout -- the rows of a
    # trajectory chain together. (Across an episode boundary they must not.)
    d = generate_dataset(**SMALL, out_path=None)
    ep0 = d["episode"] == 0
    states, nexts = d["states"][ep0], d["next_states"][ep0]
    assert np.array_equal(nexts[:-1], states[1:])


def test_logged_rows_replay_exactly_through_the_real_physics() -> None:
    # rebuild the same world, re-fly episode 0 from its logged start using the
    # logged actions, and demand bit-identical next_states. This pins the
    # dataset to step() -- the factory cannot silently drift from the sim.
    d = generate_dataset(**SMALL, out_path=None)
    ep0 = d["episode"] == 0
    states, actions, nexts = d["states"][ep0], d["actions"][ep0], d["next_states"][ep0]

    glider, air = make_world()
    start = GliderState(*(float(v) for v in states[0]))
    sim = Simulation(glider, air, start, dt=0.1)
    for i in range(len(actions)):
        nxt = sim.step(float(actions[i]))
        replayed = np.array([nxt.x, nxt.y, nxt.z, nxt.heading])
        assert np.array_equal(replayed, nexts[i]), f"replay diverged at row {i}"


# --- the FIREWALL guard: no thermal truth in the artifact --------------------
def test_saved_file_contains_no_thermal_truth(tmp_path: Path) -> None:
    out = tmp_path / "d.npz"
    generate_dataset(**SMALL, out_path=out)
    with np.load(out) as f:
        keys = set(f.keys())
        # exactly the allowed channels: own kinematics, action, sensed vario,
        # episode id, and sim/airframe constants. Nothing else.
        assert keys == {
            "states",
            "actions",
            "next_states",
            "varios",
            "episode",
            "dt",
            "airspeed",
            "base_sink",
            "seed",
        }
        # belt-and-suspenders: name every forbidden field explicitly, so the
        # intent survives even if the allowed set above is ever loosened.
        for forbidden in ("x0", "y0", "w_peak", "radius", "thermals", "wind"):
            assert forbidden not in keys
        # the model-visible feature width is state(4) + action(1) + vario(1):
        # a widened states array would be the classic silent leak.
        assert f["states"].shape[1] == 4
        assert f["actions"].ndim == 1
        assert f["varios"].ndim == 1
