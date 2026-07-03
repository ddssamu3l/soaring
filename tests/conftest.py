"""Shared test setup.

pygame runs HEADLESS for the whole test session: the SDL "dummy" video
driver renders into plain memory surfaces -- no window, no GPU, CI-safe.
The env var must be set before pygame initializes, hence at import time.
Tests draw onto their own pygame.Surface objects and assert on pixels.

Shared data fixtures live here too (one tiny dataset + one rollouts file
aligned to it), so test_frames/test_app stop growing private copies.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402  (import must follow the env var)

pygame.init()

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from data_gen import generate_dataset  # noqa: E402

# the shared mini dataset's shape -- tests that count episodes/steps assume
# these exact numbers, so they are pinned here next to the fixture.
MINI_EPISODES = 3
MINI_STEPS = 40
MINI_H = 6  # rollout horizon in the shared rollouts file


@pytest.fixture(scope="session")
def mini_npz(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real (tiny) dataset from the actual data factory -- schema drift
    between the factory and the viewport cannot hide behind a hand-built file."""
    path = tmp_path_factory.mktemp("data") / "mini.npz"
    generate_dataset(n_rollouts=MINI_EPISODES, steps_per_rollout=MINI_STEPS, out_path=path, seed=7)
    return path


@pytest.fixture(scope="session")
def mini_rollouts(mini_npz: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A rollouts.npz aligned to mini_npz, in keystone.save_rollouts' exact
    schema (test_keystone pins that schema against save_rollouts itself).
    Two rollouts in episode 1; two predictors with known geometry:
      full -- drifts east by 2 m per step (h=0 exact, then visibly wrong)
      twin -- the perfect predictor (imagined == true everywhere)
    """
    with np.load(mini_npz) as d:
        sensors = d["sensors"].astype(np.float64)
        episode = d["episode"].astype(np.int64)
        sensor_names = d["sensor_names"]
        dt = float(d["dt"])
    rows = np.nonzero(episode == 1)[0]
    # rollout 1 begins INSIDE rollout 0's horizon (keystone's real shape:
    # overlapping dreams) -- the hold-vs-freshest picker tests need that.
    starts = np.array([rows[2], rows[2 + MINI_H - 2]], dtype=np.int64)
    assert int(starts.max()) + MINI_H < int(rows.max()) + 1  # horizon stays inside episode 1

    true = np.stack([sensors[s : s + MINI_H + 1] for s in starts])
    full = true.copy()
    x_col = [str(n) for n in sensor_names].index("x")
    full[:, :, x_col] += 2.0 * np.arange(MINI_H + 1)

    path = tmp_path_factory.mktemp("rollouts") / "mini_rollouts.npz"
    np.savez_compressed(
        path,
        sensor_names=sensor_names,
        dt=np.array(dt, dtype=np.float64),
        horizon=np.array(MINI_H, dtype=np.int64),
        starts=starts,
        episode=episode[starts],
        true=true,
        rollouts_full=full,
        rollouts_twin=true.copy(),
    )
    return path
