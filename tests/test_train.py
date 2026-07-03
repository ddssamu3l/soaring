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
    PanelMLP,
    Panels,
    delta_targets,
    featurize,
    fit_stats,
    load_panels,
    make_feature_spec,
    pair_indices,
    param_count,
    split_episodes,
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
    return idx, fit_stats(data, idx), make_feature_spec(data)


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
    x = torch.from_numpy(
        featurize(data.sensors[idx], data.actions[idx], stats, spec).astype(np.float32)
    )
    y = torch.from_numpy(
        delta_targets(data.sensors[idx], data.sensors[idx + 1], stats).astype(np.float32)
    )
    torch.manual_seed(0)
    model = PanelMLP(spec.n_features, len(data.sensor_names))
    with torch.no_grad():
        loss = float(F.mse_loss(model(x), y))
    assert 0.5 < loss < 2.0
