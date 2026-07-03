"""
Property tests for the keystone rollout machinery.

The keystone plot is the project's go/no-go gate, so the machinery beneath it
must be provably honest: a broken rollout loop would produce a beautiful,
meaningless curve. The strongest guards are closed-form: a ZERO net (all
weights zero) predicts exactly the mean delta every step, so its free-run has
an exact algebraic answer the code must reproduce to the last bit of float64.
"""

import numpy as np
import pytest
import torch

from data_gen import generate_dataset
from keystone import (
    channel_error,
    free_run,
    persistence_run,
    position_error,
    rollout_starts,
    sigma_error,
    teacher_forced,
    true_panels,
)
from train import (
    Checkpoint,
    PanelMLP,
    fit_stats,
    load_panels,
    make_feature_spec,
    pair_indices,
    split_episodes,
)

SMALL = {"n_rollouts": 10, "steps_per_rollout": 40, "hold_steps": 5, "seed": 11}
H, STRIDE = 12, 10


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory):
    """Dataset + a ZERO-weight checkpoint (its predictions have a closed form)."""
    path = tmp_path_factory.mktemp("t2keystone") / "small.npz"
    generate_dataset(**SMALL, out_path=path)
    data = load_panels(path)
    split = split_episodes(data.episode)
    pairs = pair_indices(data.episode)
    stats = fit_stats(data, pairs[np.isin(data.episode[pairs], split.train)])
    spec = make_feature_spec(data.sensor_names, data.action_names)
    zero_net = PanelMLP(spec.n_features, len(data.sensor_names), (8,))
    with torch.no_grad():
        for p in zero_net.parameters():
            p.zero_()
    ck = Checkpoint(
        model=zero_net,
        stats=stats,
        spec=spec,
        split=split,
        ablate=False,
        sensor_names=data.sensor_names,
        action_names=data.action_names,
        dt=data.dt,
    )
    return ck, data


def test_rollout_starts_stay_inside_test_episodes(setup) -> None:
    ck, data = setup
    starts = rollout_starts(data, ck.split.test, H, STRIDE)
    assert len(starts) > 0
    # the whole horizon must lie within the start's own episode
    assert np.all(data.episode[starts] == data.episode[starts + H])
    assert set(np.unique(data.episode[starts])) <= set(ck.split.test)


def test_true_panels_are_the_actual_rows(setup) -> None:
    ck, data = setup
    starts = rollout_starts(data, ck.split.test, H, STRIDE)
    truth = true_panels(data, starts, H)
    assert truth.shape == (len(starts), H + 1, len(data.sensor_names))
    assert np.array_equal(truth[:, 0], data.sensors[starts])
    assert np.array_equal(truth[:, H], data.sensors[starts + H])


def test_zero_net_free_run_matches_closed_form(setup) -> None:
    """A zero net outputs z-delta = 0, i.e. 'the mean delta, every step'. Its
    imagined panel therefore has the exact closed form panel0 + h*delta_mean.
    This pins the whole recursion: featurize -> model -> undo_delta -> feed back."""
    ck, data = setup
    starts = rollout_starts(data, ck.split.test, H, STRIDE)
    run = free_run(ck, data, starts, H)
    for h in (0, 1, H // 2, H):
        expected = data.sensors[starts] + h * ck.stats.delta_mean
        assert np.allclose(run[:, h], expected, atol=1e-9)


def test_zero_net_teacher_forced_matches_closed_form(setup) -> None:
    """Teacher-forced with a zero net = true previous panel + mean delta, at
    every horizon (no feedback, so no accumulation)."""
    ck, data = setup
    starts = rollout_starts(data, ck.split.test, H, STRIDE)
    tf = teacher_forced(ck, data, starts, H)
    for h in (1, H):
        expected = data.sensors[starts + h - 1] + ck.stats.delta_mean
        assert np.allclose(tf[:, h], expected, atol=1e-9)


def test_free_and_teacher_forced_agree_at_horizon_one(setup) -> None:
    """At h=1 both start from the true panel, so they MUST coincide exactly --
    a cheap consistency proof that the two loops share one prediction step."""
    ck, data = setup
    starts = rollout_starts(data, ck.split.test, H, STRIDE)
    assert np.allclose(
        free_run(ck, data, starts, 1)[:, 1], teacher_forced(ck, data, starts, 1)[:, 1]
    )


def test_persistence_is_frozen(setup) -> None:
    ck, data = setup
    starts = rollout_starts(data, ck.split.test, H, STRIDE)
    per = persistence_run(data, starts, H)
    assert np.array_equal(per[:, 0], per[:, H])
    assert np.array_equal(per[:, 0], data.sensors[starts])


def test_error_metrics_are_zero_for_perfect_prediction(setup) -> None:
    ck, data = setup
    starts = rollout_starts(data, ck.split.test, H, STRIDE)
    truth = true_panels(data, starts, H)
    assert np.all(sigma_error(truth, truth, ck.stats.panel_std) == 0)
    assert np.all(position_error(truth, truth, 0, 1) == 0)
    assert np.all(channel_error(truth, truth, 2) == 0)


def test_persistence_error_grows(setup) -> None:
    """The experiment's own sanity check: a real glider flies away from a frozen
    snapshot, so persistence sigma-error at the far horizon must exceed h=1."""
    ck, data = setup
    starts = rollout_starts(data, ck.split.test, H, STRIDE)
    truth = true_panels(data, starts, H)
    err = sigma_error(persistence_run(data, starts, H), truth, ck.stats.panel_std)
    assert err[H] > err[1] > 0
