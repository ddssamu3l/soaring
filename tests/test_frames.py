"""Tests for viewport/frames.py -- both flight sources against a real (tiny)
dataset produced by the actual data factory, so schema drift can't hide."""

from pathlib import Path

import numpy as np
import pytest

from data_gen import generate_dataset, make_world
from glider_sim import ACTION_NAMES, SENSOR_NAMES
from viewport.frames import FlightLog, LiveFlight, default_start

N_EPISODES = 3


@pytest.fixture(scope="module")
def mini_npz(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "mini.npz"
    generate_dataset(n_rollouts=N_EPISODES, steps_per_rollout=40, out_path=path, seed=7)
    return path


@pytest.fixture(scope="module")
def log(mini_npz: Path) -> FlightLog:
    return FlightLog.load(mini_npz)


# --------------------------------------------------------------- FlightLog
def test_log_is_self_describing(log: FlightLog) -> None:
    """Channel names come from the FILE -- and today they match the sim's."""
    assert log.sensor_names == SENSOR_NAMES
    assert log.action_names == ACTION_NAMES
    assert log.episode_ids == list(range(N_EPISODES))


def test_flight_slices_one_episode(log: FlightLog) -> None:
    f = log.flight(1)
    assert 0 < f.n <= 40
    assert f.sensors.shape == (f.n, len(f.sensor_names))
    assert f.actions.shape == (f.n, len(f.action_names))


def test_frame_is_name_keyed_and_matches_arrays(log: FlightLog) -> None:
    f = log.flight(0)
    fr = f.frame(3)
    assert set(fr.sensors) == set(f.sensor_names)
    assert set(fr.actions) == set(f.action_names)
    for j, name in enumerate(f.sensor_names):
        assert fr.sensors[name] == f.sensors[3, j]


def test_frame_index_clamps(log: FlightLog) -> None:
    f = log.flight(0)
    assert f.frame(-99) == f.frame(0)
    assert f.frame(10**9) == f.frame(f.n - 1)


def test_path_reads_positions_by_name(log: FlightLog) -> None:
    f = log.flight(2)
    assert f.path.shape == (f.n, 3)
    for k, name in enumerate(("x", "y", "z")):
        col = f.true_states[:, f.state_names.index(name)]
        assert np.array_equal(f.path[:, k], col)


def test_climbs_are_altitude_differences(log: FlightLog) -> None:
    f = log.flight(0)
    z = f.path[:, 2]
    assert len(f.climbs) == f.n
    assert f.climbs[0] == pytest.approx((z[1] - z[0]) / f.dt)
    assert f.climbs[-1] == f.climbs[-2]  # last value repeats to match length


# --------------------------------------------------------------- LiveFlight
def _live() -> LiveFlight:
    glider, air = make_world()
    return LiveFlight(glider, air)


def test_live_step_yields_named_frame() -> None:
    live = _live()
    fr = live.step(0.3, 25.0)
    assert set(fr.sensors) == set(SENSOR_NAMES)
    assert fr.actions == {"bank_cmd": 0.3, "pitch_cmd": 25.0}
    assert live.n == 1


def test_live_path_grows_with_flight() -> None:
    live = _live()
    for _ in range(5):
        live.step(0.0, 25.0)
    assert live.path.shape == (6, 3)  # 5 history rows + current state
    assert len(live.climbs) == 6
    start = default_start()
    assert live.path[0][0] == start.x and live.path[0][2] == start.z


def test_live_save_roundtrips_through_flightlog(tmp_path: Path) -> None:
    """A saved manual flight IS a dataset: same keys, loads right back."""
    live = _live()
    for _ in range(8):
        live.step(0.2, 24.0)
    saved = live.save(tmp_path)
    # exact schema parity with the data factory's files
    generate_dataset(n_rollouts=1, steps_per_rollout=5, out_path=tmp_path / "ref.npz")
    with np.load(saved) as d, np.load(tmp_path / "ref.npz") as r:
        assert set(d.keys()) == set(r.keys())
        assert int(d["seed"]) == -1  # marks "human pilot, not rng"
        assert list(d["episode"]) == [0] * 8
    back = FlightLog.load(saved)
    assert back.flight(0).n == 8
    assert back.sensor_names == SENSOR_NAMES


def test_live_save_refuses_empty_flight(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _live().save(tmp_path)
