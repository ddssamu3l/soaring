"""
Property tests for the keystone rollout machinery.

The keystone plot is the project's go/no-go gate, so the machinery beneath it
must be provably honest: a broken rollout loop would produce a beautiful,
meaningless curve. The strongest guards are closed-form: a ZERO net (all
weights zero) predicts exactly the mean delta every step, so its free-run has
an exact algebraic answer the code must reproduce to the last bit of float64.
"""

from dataclasses import replace

import numpy as np
import pytest
import torch

from data_gen import generate_dataset
from keystone import (
    channel_error,
    check_shared_test_split,
    free_run,
    persistence_run,
    plot_bounds,
    position_error,
    rollout_starts,
    save_rollouts,
    sigma_error,
    teacher_forced,
    true_panels,
)
from train import (
    Checkpoint,
    PanelMLP,
    Panels,
    Split,
    clamp_panel,
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


def test_rollout_starts_skip_episodes_shorter_than_the_horizon() -> None:
    """REGRESSION (t2 judge): an episode shorter than the horizon used to make the
    slice stop NEGATIVE, wrapping around and emitting starts with no full horizon
    ahead (an index-out-of-bounds crash, or worse, silently misaligned rollouts).
    Crash-shortened episodes must simply contribute zero starts."""
    n_long, n_short = 30, 8  # 8 sits in the hazard zone: shorter than H, longer than H-stop=0
    episode = np.concatenate([np.zeros(n_long, np.int64), np.ones(n_short, np.int64)])
    data = Panels(
        sensors=np.zeros((len(episode), 2)),
        actions=np.zeros((len(episode), 1)),
        episode=episode,
        sensor_names=("a", "b"),
        action_names=("u",),
        dt=0.1,
    )
    starts = rollout_starts(data, np.array([0, 1], dtype=np.int64), H, STRIDE)
    assert len(starts) > 0
    # every start still has the whole horizon ahead INSIDE its own episode ...
    assert np.all(data.episode[starts] == data.episode[starts + H])
    # ... which forces the short episode to contribute nothing at all
    assert set(np.unique(data.episode[starts])) == {0}


def test_shared_split_guard_catches_a_mismatched_twin(setup) -> None:
    """The guard behind the full-vs-twin comparison: same test split passes in
    silence; a checkpoint holding different held-out episodes is refused."""
    ck, _ = setup
    check_shared_test_split(ck, ck)  # identical splits: no complaint
    shuffled = replace(ck, split=Split(train=ck.split.test, val=ck.split.val, test=ck.split.train))
    with pytest.raises(ValueError, match="test splits"):
        check_shared_test_split(ck, shuffled)


def test_true_panels_are_the_actual_rows(setup) -> None:
    ck, data = setup
    starts = rollout_starts(data, ck.split.test, H, STRIDE)
    truth = true_panels(data, starts, H)
    assert truth.shape == (len(starts), H + 1, len(data.sensor_names))
    assert np.array_equal(truth[:, 0], data.sensors[starts])
    assert np.array_equal(truth[:, H], data.sensors[starts + H])


def test_zero_net_free_run_matches_closed_form(setup) -> None:
    """A zero net outputs z-delta = 0, i.e. 'the mean delta, every step', and
    the feedback loop clamps each result to the training range -- so the exact
    closed form is the clamp iterated: e_h = clamp(e_{h-1} + delta_mean).
    This pins the whole recursion: featurize -> model -> undo_delta -> clamp."""
    ck, data = setup
    starts = rollout_starts(data, ck.split.test, H, STRIDE)
    run = free_run(ck, data, starts, H)
    expected = data.sensors[starts].astype(np.float64)
    assert np.allclose(run[:, 0], expected, atol=1e-9)
    for h in range(1, H + 1):
        expected = clamp_panel(expected + ck.stats.delta_mean, ck.stats)
        if h in (1, H // 2, H):
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
    a cheap consistency proof that the two loops share one prediction step
    (free-run additionally clamps -- teacher-forced never feeds back, so it
    never needs to)."""
    ck, data = setup
    starts = rollout_starts(data, ck.split.test, H, STRIDE)
    tf = teacher_forced(ck, data, starts, 1)[:, 1]
    assert np.allclose(free_run(ck, data, starts, 1)[:, 1], clamp_panel(tf, ck.stats))


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


def test_save_rollouts_roundtrip_pins_the_alignment(setup, tmp_path) -> None:
    """The persisted file's contract (what the viewport scrubs by): row h of
    rollout i aligns with dataset row starts[i]+h, h=0 IS the true start row,
    per-predictor arrays are stored verbatim under rollouts_<name>, and the
    persistence baseline is deliberately NOT saved."""
    ck, data = setup
    starts = rollout_starts(data, ck.split.test, H, STRIDE)
    truth = true_panels(data, starts, H)
    runs = {
        "full": free_run(ck, data, starts, H),
        "persistence": persistence_run(data, starts, H),
        "teacher-forced": teacher_forced(ck, data, starts, H),
    }
    save_rollouts(tmp_path / "r.npz", data, starts, truth, runs, H)
    with np.load(tmp_path / "r.npz") as d:
        assert tuple(str(n) for n in d["sensor_names"]) == data.sensor_names
        assert float(d["dt"]) == data.dt and int(d["horizon"]) == H
        assert np.array_equal(d["starts"], starts)
        assert np.array_equal(d["episode"], data.episode[starts])
        # hyphenated run name -> underscored key; baseline stays out
        assert {k for k in d.files if k.startswith("rollouts_")} == {
            "rollouts_full",
            "rollouts_teacher_forced",
        }
        assert np.array_equal(d["rollouts_full"], runs["full"])
        # the viewport's clock: true row h IS the dataset row starts+h ...
        for h in (0, H):
            assert np.array_equal(d["true"][:, h], data.sensors[starts + h])
        # ... and every imagination starts at the true panel
        assert np.array_equal(d["rollouts_full"][:, 0], d["true"][:, 0])


def test_plot_bounds_contain_every_path_with_margin(setup) -> None:
    """The ghost chart's field grid must follow the data: every true path point
    strictly inside the bounds, with the margin actually applied."""
    ck, data = setup
    starts = rollout_starts(data, ck.split.test, H, STRIDE)
    truth = true_panels(data, starts, H)
    xlo, xhi, ylo, yhi = plot_bounds(truth, 0, 1, margin=50.0)
    assert xlo <= truth[:, :, 0].min() - 50.0 + 1e-9
    assert xhi >= truth[:, :, 0].max() + 50.0 - 1e-9
    assert ylo <= truth[:, :, 1].min() - 50.0 + 1e-9
    assert yhi >= truth[:, :, 1].max() + 50.0 - 1e-9


def test_persistence_error_grows(setup) -> None:
    """The experiment's own sanity check: a real glider flies away from a frozen
    snapshot, so persistence sigma-error at the far horizon must exceed h=1."""
    ck, data = setup
    starts = rollout_starts(data, ck.split.test, H, STRIDE)
    truth = true_panels(data, starts, H)
    err = sigma_error(persistence_run(data, starts, H), truth, ck.stats.panel_std)
    assert err[H] > err[1] > 0
