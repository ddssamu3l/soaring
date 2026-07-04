"""
Property tests for the model piece of train.py (installment 1: split / pairs /
normalize / featurize / MLP -- no training loop yet).

Same philosophy as the sim tests: assert what must be TRUE of any correct
implementation, because ML fails silently -- a leaking split, a broken target, or
an un-normalized feature would all train "successfully" and produce garbage.

Tests mint their own small dataset through the real data factory (never the real
data/dataset.npz): fast, hermetic, and exercises load_panels() on a real file.
"""

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from data_gen import generate_dataset
from train import (
    Checkpoint,
    PanelMLP,
    Panels,
    TrainConfig,
    clamp_panel,
    delta_targets,
    featurize,
    fit_stats,
    load_checkpoint,
    load_panels,
    lr_find,
    make_feature_spec,
    pair_indices,
    param_count,
    predict_delta,
    save_checkpoint,
    split_episodes,
    tensor_pairs,
    train_model,
    undo_delta,
)

# 10 episodes so the 80/10/10 split gives every role at least one episode.
SMALL = {"n_rollouts": 10, "steps_per_rollout": 40, "hold_steps": 5, "seed": 11}


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> Panels:
    path = tmp_path_factory.mktemp("t2") / "small.npz"
    generate_dataset(**SMALL, out_path=path)
    return load_panels(path)


def _fitted(data: Panels):  # stats + spec over ALL pairs -- fine for property tests
    idx = pair_indices(data.episode)
    spec = make_feature_spec(data.sensor_names, data.action_names)
    return idx, fit_stats(data, idx), spec


# --- split ------------------------------------------------------------------------
def test_split_covers_all_episodes_disjointly_and_deterministically(data: Panels) -> None:
    s1 = split_episodes(data.episode, seed=0)
    s2 = split_episodes(data.episode, seed=0)
    got = np.concatenate([s1.train, s1.val, s1.test])
    assert sorted(got) == sorted(np.unique(data.episode))  # covers everything, no dupes
    assert len(s1.val) >= 1 and len(s1.test) >= 1  # every role is populated
    for a, b in ((s1.train, s2.train), (s1.val, s2.val), (s1.test, s2.test)):
        assert np.array_equal(a, b)  # same seed -> same split, forever


# --- pairs ------------------------------------------------------------------------
def test_pairs_stay_within_episode_and_drop_only_final_rows(data: Panels) -> None:
    idx = pair_indices(data.episode)
    assert np.all(data.episode[idx] == data.episode[idx + 1])
    n_eps = len(np.unique(data.episode))
    assert len(idx) == len(data.episode) - n_eps  # exactly one dropped row per episode


def test_heading_deltas_are_small_so_raw_differences_are_safe(data: Panels) -> None:
    """delta_targets() uses raw heading differences. That is only valid while the
    sim never wraps heading -- a wrap would appear as a ~2pi one-step delta."""
    idx = pair_indices(data.episode)
    h = data.sensor_names.index("heading")
    dh = data.sensors[idx + 1, h] - data.sensors[idx, h]
    assert np.abs(dh).max() < 0.5  # real per-step turns are ~0.06 rad; 2pi would scream


# --- featurize ---------------------------------------------------------------------
def test_features_are_O1_with_sincos_on_the_unit_circle(data: Panels) -> None:
    idx, stats, spec = _fitted(data)
    x = featurize(data.sensors[idx], data.actions[idx], stats, spec)
    assert x.shape == (len(idx), spec.n_features)
    assert np.abs(x).max() < 20.0  # everything z-scored/bounded; raw x would be ~100s
    sin_col, cos_col = spec.heading, spec.heading + 1  # heading expands in place
    assert np.allclose(x[:, sin_col] ** 2 + x[:, cos_col] ** 2, 1.0)


def test_ablation_zeroes_exactly_the_lift_columns(data: Panels) -> None:
    idx, stats, spec = _fitted(data)
    full = featurize(data.sensors[idx], data.actions[idx], stats, spec)
    twin = featurize(data.sensors[idx], data.actions[idx], stats, spec, ablate=True)
    # feature column of panel channel c: +1 past heading (it expanded into two cols)
    lift_cols = [c + 1 if c > spec.heading else c for c in spec.lift]
    assert np.all(twin[:, lift_cols] == 0.0)  # the twin never feels the air...
    others = [j for j in range(spec.n_features) if j not in lift_cols]
    assert np.array_equal(twin[:, others], full[:, others])  # ...and nothing else moved


# --- targets -----------------------------------------------------------------------
def test_delta_targets_roundtrip_exactly_to_the_next_panel(data: Panels) -> None:
    """undo_delta(delta_targets(...)) must reproduce the next panel: this identity is
    the imagination step of the rollout loop, so it has to be exact, not approximate."""
    idx, stats, _ = _fitted(data)
    panel, next_panel = data.sensors[idx], data.sensors[idx + 1]
    z = delta_targets(panel, next_panel, stats)
    assert np.allclose(undo_delta(z, panel, stats), next_panel, atol=1e-9)
    # z-scored on the same rows the stats were fitted on => standardized by construction
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(z.std(axis=0), 1.0, atol=1e-6)


def test_fit_stats_range_covers_exactly_the_train_pairs(data: Panels) -> None:
    """panel_lo/panel_hi are the imagination clamp: they must be the true
    per-channel extremes over BOTH pair endpoints -- every training row inside,
    and the bounds attained (not padded)."""
    idx, stats, _ = _fitted(data)
    rows = np.concatenate([data.sensors[idx], data.sensors[idx + 1]])
    assert np.all(rows >= stats.panel_lo) and np.all(rows <= stats.panel_hi)
    assert np.array_equal(stats.panel_lo, rows.min(axis=0))
    assert np.array_equal(stats.panel_hi, rows.max(axis=0))


def test_clamp_panel_pins_only_out_of_range_values(data: Panels) -> None:
    """In-range panels pass through untouched (the clamp must be inert on the
    manifold); runaway values pin to the gauge limits."""
    idx, stats, _ = _fitted(data)
    real = data.sensors[idx[:50]]
    assert np.array_equal(clamp_panel(real, stats), real)  # inert on real data
    wild = real.copy()
    wild[:, 0] = 1e9
    wild[:, 6] = -1e9
    pinned = clamp_panel(wild, stats)
    assert np.all(pinned[:, 0] == stats.panel_hi[0])
    assert np.all(pinned[:, 6] == stats.panel_lo[6])
    assert np.array_equal(pinned[:, 1:6], wild[:, 1:6])  # untouched channels unchanged


# --- model -------------------------------------------------------------------------
def test_mlp_shape_and_param_count_match_the_design() -> None:
    torch.manual_seed(0)
    model = PanelMLP(12, 9, hidden=(256, 256))
    assert model(torch.zeros(7, 12)).shape == (7, 9)
    expected = (12 * 256 + 256) + (256 * 256 + 256) + (256 * 9 + 9)  # 71,433
    assert param_count(model) == expected


def test_step0_loss_is_near_one(data: Panels) -> None:
    """TRIPWIRE 1: with z-scored targets (std 1) and a fresh net (outputs ~0), the
    untrained MSE is ~1.0 by arithmetic -- computable before training exists."""
    idx, stats, spec = _fitted(data)
    x, y = tensor_pairs(data, idx, stats, spec)
    torch.manual_seed(0)
    model = PanelMLP(spec.n_features, len(data.sensor_names))
    with torch.no_grad():
        loss = float(F.mse_loss(model(x), y))
    assert 0.5 < loss < 2.0


# --- training loop -----------------------------------------------------------------
def test_training_learns_and_keeps_the_best_val_weights(data: Panels) -> None:
    """The loop must (a) actually drive loss well below the predict-the-mean 1.0
    line, and (b) hand back the BEST epoch's weights, not the last step's."""
    idx, stats, spec = _fitted(data)
    x, y = tensor_pairs(data, idx, stats, spec)
    cfg = TrainConfig(hidden=(32,), batch=64, epochs=20, seed=0)
    model, hist = train_model(x, y, x, y, cfg, verbose=False)  # train==val: mechanics test
    # history must hold plain floats -- a Tensor here would drag its whole autograd
    # graph into the lists (memory leak) and break plotting/serialization
    assert all(type(v) is float for v in hist.train_loss + hist.val_loss)
    assert hist.val_loss[-1] < 0.7
    assert hist.val_loss[-1] < hist.val_loss[0]
    with torch.no_grad():
        final = float(F.mse_loss(model(x), y))
    assert final == pytest.approx(min(hist.val_loss), abs=1e-6)  # best snapshot was kept


def test_different_seeds_train_different_members(data: Panels) -> None:
    """The ensemble's whole value is diversity: two members trained on the same
    rows with different seeds must NOT be the same function -- if seeding ever
    stopped reaching init/shuffle, worst-member voting would silently become
    single-model planning."""
    idx, stats, spec = _fitted(data)
    x, y = tensor_pairs(data, idx, stats, spec)
    cfg = TrainConfig(hidden=(8,), batch=64, epochs=2, seed=0)
    m0, _ = train_model(x, y, x, y, cfg, verbose=False)
    m1, _ = train_model(
        x, y, x, y, TrainConfig(hidden=(8,), batch=64, epochs=2, seed=1), verbose=False
    )
    with torch.no_grad():
        assert not torch.equal(m0(x[:64]), m1(x[:64]))


def test_lr_finder_runs_and_starts_near_one(data: Panels) -> None:
    idx, stats, spec = _fitted(data)
    x, y = tensor_pairs(data, idx, stats, spec)
    lrs, losses = lr_find(x, y, hidden=(32,), steps=20, batch=64)
    assert len(lrs) == len(losses) == 20
    assert 0.5 < losses[0] < 2.0  # the first step is still basically untrained
    assert np.all(np.isfinite(losses[:5]))  # sane in the safe zone (MAY explode later)


# --- predict_delta: THE shared model call --------------------------------------------
def _checkpoint_around(model: PanelMLP, data: Panels, ablate: bool = False) -> Checkpoint:
    idx, stats, spec = _fitted(data)
    model.eval()
    return Checkpoint(
        model=model,
        stats=stats,
        spec=spec,
        split=split_episodes(data.episode),
        ablate=ablate,
        sensor_names=data.sensor_names,
        action_names=data.action_names,
        dt=data.dt,
    )


def test_predict_delta_matches_the_pipeline_it_replaced(data: Panels) -> None:
    """predict_delta must equal the hand-inlined featurize -> net -> de-normalize
    it deduplicated (card.py / keystone.py / the planner all lean on this one call),
    and `panel + delta` must agree with undo_delta -- the imagination-step identity."""
    idx, stats, spec = _fitted(data)
    torch.manual_seed(0)
    ck = _checkpoint_around(PanelMLP(spec.n_features, len(data.sensor_names), (16,)), data)
    panel, action = data.sensors[idx], data.actions[idx]
    delta = predict_delta(ck, panel, action)
    x = torch.from_numpy(featurize(panel, action, stats, spec).astype(np.float32))
    with torch.no_grad():
        z = np.asarray(ck.model(x).numpy(), dtype=np.float64)
    assert np.array_equal(delta, z * stats.delta_std + stats.delta_mean)  # same ops, bit-for-bit
    assert np.allclose(panel + delta, undo_delta(z, panel, stats), atol=1e-9)


def test_predict_delta_zero_net_returns_the_mean_delta(data: Panels) -> None:
    """Closed form: a zero net outputs z-delta 0, so the physical delta is exactly
    the train-mean delta for every row -- pinned before any planner trusts it."""
    _, stats, spec = _fitted(data)
    zero = PanelMLP(spec.n_features, len(data.sensor_names), (8,))
    with torch.no_grad():
        for p in zero.parameters():
            p.zero_()
    ck = _checkpoint_around(zero, data)
    delta = predict_delta(ck, data.sensors[:50], data.actions[:50])
    assert np.array_equal(delta, np.broadcast_to(stats.delta_mean, delta.shape))


def test_predict_delta_honors_the_checkpoints_own_ablate_flag(data: Panels) -> None:
    """The twin's blindfold travels INSIDE its checkpoint: the same net predicts
    differently once ablate=True, with no caller-side flag to forget."""
    idx, _, spec = _fitted(data)
    torch.manual_seed(0)
    net = PanelMLP(spec.n_features, len(data.sensor_names), (16,))
    seeing = _checkpoint_around(net, data, ablate=False)
    blind = _checkpoint_around(net, data, ablate=True)
    panel, action = data.sensors[idx], data.actions[idx]
    assert not np.array_equal(
        predict_delta(seeing, panel, action), predict_delta(blind, panel, action)
    )


# --- checkpoints ---------------------------------------------------------------------
def test_checkpoint_roundtrip_reproduces_predictions(tmp_path, data: Panels) -> None:
    """keystone.py trusts checkpoints completely: reloaded (model, stats, split)
    must reproduce the original's predictions bit-for-bit, flags included."""
    idx, stats, spec = _fitted(data)
    split = split_episodes(data.episode)
    x, y = tensor_pairs(data, idx, stats, spec)
    cfg = TrainConfig(hidden=(16,), batch=64, epochs=2, seed=0)
    model, _ = train_model(x, y, x, y, cfg, verbose=False)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, cfg, stats, split, data, ablate=True)
    ck = load_checkpoint(path)
    assert ck.ablate is True
    assert ck.sensor_names == data.sensor_names
    assert np.array_equal(ck.split.test, split.test)
    assert np.allclose(ck.stats.delta_std, stats.delta_std)
    assert np.array_equal(ck.stats.panel_lo, stats.panel_lo)  # the clamp travels too
    with torch.no_grad():
        assert torch.equal(ck.model(x), model(x))  # identical weights => identical output
