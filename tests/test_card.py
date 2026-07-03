"""
Property tests for the one-step report card (card.py).

The card is itself a measuring instrument, so these tests calibrate the
instrument: persistence RMSE must equal the hand-computed value, and the
collapse detector must actually fire on a deliberately dead net -- a tripwire
nobody has tested is just a decoration.
"""

import numpy as np
import pytest
import torch

from card import held_out_pair_rows, one_step_card, print_card
from data_gen import generate_dataset
from train import (
    Checkpoint,
    PanelMLP,
    TrainConfig,
    fit_stats,
    load_panels,
    make_feature_spec,
    pair_indices,
    save_checkpoint,
    split_episodes,
    tensor_pairs,
    train_model,
)

SMALL = {"n_rollouts": 10, "steps_per_rollout": 40, "hold_steps": 5, "seed": 11}


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory):
    """A tiny trained checkpoint + its dataset, built once for the whole module."""
    path = tmp_path_factory.mktemp("t2card") / "small.npz"
    generate_dataset(**SMALL, out_path=path)
    data = load_panels(path)
    split = split_episodes(data.episode)
    pairs = pair_indices(data.episode)
    train_rows = pairs[np.isin(data.episode[pairs], split.train)]
    stats = fit_stats(data, train_rows)
    spec = make_feature_spec(data.sensor_names, data.action_names)
    x, y = tensor_pairs(data, train_rows, stats, spec)
    cfg = TrainConfig(hidden=(16,), batch=64, epochs=3, seed=0)
    model, _ = train_model(x, y, x, y, cfg, verbose=False)
    ck_path = tmp_path_factory.mktemp("t2card") / "model.pt"
    save_checkpoint(ck_path, model, cfg, stats, split, data, ablate=False)
    from train import load_checkpoint

    return load_checkpoint(ck_path), data


def test_card_uses_only_test_episodes(setup) -> None:
    ck, data = setup
    rows = held_out_pair_rows(ck, data)
    assert len(rows) > 0
    assert set(np.unique(data.episode[rows])) <= set(ck.split.test)


def test_persistence_rmse_matches_hand_computation(setup) -> None:
    """The baseline column is the card's yardstick -- calibrate it exactly."""
    ck, data = setup
    card = one_step_card(ck, data)
    rows = held_out_pair_rows(ck, data)
    true_delta = data.sensors[rows + 1] - data.sensors[rows]
    assert np.allclose(card.persist_rmse, np.sqrt(np.mean(true_delta**2, axis=0)))
    assert card.n_pairs == len(rows)


def test_card_shapes_and_sanity(setup) -> None:
    ck, data = setup
    card = one_step_card(ck, data)
    n = len(data.sensor_names)
    assert card.channels == data.sensor_names
    for arr in (card.model_rmse, card.persist_rmse, card.spread_ratio):
        assert arr.shape == (n,)
        assert np.all(np.isfinite(arr)) and np.all(arr >= 0)


def test_collapse_detector_fires_on_a_dead_net(setup, capsys) -> None:
    """Zero every weight: the net outputs a constant, the classic silent failure.
    Its RMSE can still look passable -- the spread ratio MUST expose it."""
    ck, data = setup
    dead = PanelMLP(ck.spec.n_features, len(ck.sensor_names), (16,))
    with torch.no_grad():
        for p in dead.parameters():
            p.zero_()
    dead_ck = Checkpoint(
        model=dead,
        stats=ck.stats,
        spec=ck.spec,
        split=ck.split,
        ablate=False,
        sensor_names=ck.sensor_names,
        action_names=ck.action_names,
        dt=ck.dt,
    )
    card = one_step_card(dead_ck, data)
    assert np.all(card.spread_ratio < 0.1)  # every channel is spread-dead
    print_card("dead", card)
    assert "COLLAPSE" in capsys.readouterr().out  # ...and the table SAYS so


def test_print_card_renders_every_channel(setup, capsys) -> None:
    ck, data = setup
    print_card("full", one_step_card(ck, data))
    out = capsys.readouterr().out
    for ch in data.sensor_names:
        assert ch in out  # (a 3-epoch toy net MAY legitimately semi-collapse; no flag assert)
