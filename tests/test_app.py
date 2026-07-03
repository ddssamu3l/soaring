"""Tests for viewport/app.py -- the mode/input state machine, driven
deterministically through _tick(dt)/_input(key)/_render(surface). Headless:
no window, no clock. (mini_npz / mini_rollouts fixtures live in conftest.)"""

from pathlib import Path

import numpy as np
import pygame
import pytest

from viewport import hud
from viewport.app import MODE_FLY, MODE_REPLAY, ViewportApp
from viewport.camera import Camera
from viewport.colors import GHOST_VIOLET
from viewport.frames import FlightLog, RolloutSet
from viewport.scene import px

N_EPISODES = 3  # pinned in conftest.MINI_EPISODES


@pytest.fixture(scope="module")
def app(mini_npz: Path, tmp_path_factory: pytest.TempPathFactory) -> ViewportApp:
    return ViewportApp(data_path=mini_npz, flights_dir=tmp_path_factory.mktemp("flights"))


@pytest.fixture()
def replaying(app: ViewportApp) -> ViewportApp:
    """Reset the shared app to a known replay state before each test."""
    if app.mode != MODE_REPLAY:
        app._toggle_mode()
    app._load_episode(0)
    app.speed = 1.0
    app.held = dict.fromkeys(app.held, 0)
    return app


# ------------------------------------------------------------------ replay
def test_boots_into_replay_with_gauges(replaying: ViewportApp) -> None:
    assert replaying.mode == MODE_REPLAY
    assert replaying.flight is not None and replaying.flight.n > 0
    assert len(replaying.specs) > 0


def test_playback_advances_with_time(replaying: ViewportApp) -> None:
    replaying._tick(0.5)  # 0.5 s at speed 1 and dt 0.1 = 5 frames
    assert int(replaying.frame_pos) == 5


def test_playback_speed_scales(replaying: ViewportApp) -> None:
    replaying._input("up arrow")  # x2
    replaying._tick(0.5)
    assert int(replaying.frame_pos) == 10
    replaying._input("down arrow")


def test_playback_holds_on_last_frame(replaying: ViewportApp) -> None:
    replaying._tick(1e9)
    assert replaying.frame_pos == replaying.flight.n - 1  # type: ignore[union-attr]
    assert replaying.playing is False
    replaying._input("space")  # resume from the end = restart
    assert replaying.playing is True and replaying.frame_pos == 0.0


def test_single_step_pauses_and_moves_one_frame(replaying: ViewportApp) -> None:
    replaying._input(".")
    assert replaying.playing is False and replaying.frame_pos == 1.0
    replaying._input(",")
    assert replaying.frame_pos == 0.0


def test_arrow_scrub_jumps_one_second(replaying: ViewportApp) -> None:
    replaying._input("right arrow")
    assert replaying.frame_pos == pytest.approx(10.0)  # 1 s / dt 0.1
    replaying._input("left arrow")
    assert replaying.frame_pos == 0.0


def test_episode_switch_wraps(replaying: ViewportApp) -> None:
    for want in range(1, N_EPISODES):
        replaying._input("]")
        assert replaying.ep_pos == want
    replaying._input("]")  # off the end -> wraps to 0
    assert replaying.ep_pos == 0
    replaying._input("[")
    assert replaying.ep_pos == N_EPISODES - 1


def test_timeline_click_seeks(replaying: ViewportApp) -> None:
    surface = pygame.Surface((1280, 780))
    track = hud.timeline_rect(surface)
    replaying._click((track.centerx, track.centery), surface)
    n = replaying.flight.n  # type: ignore[union-attr]
    assert replaying.frame_pos == pytest.approx((n - 1) / 2, abs=1.0)


def test_click_off_timeline_does_nothing(replaying: ViewportApp) -> None:
    surface = pygame.Surface((1280, 780))
    replaying.frame_pos = 3.0
    replaying._click((10, 10), surface)
    assert replaying.frame_pos == 3.0


# --------------------------------------------------------------------- fly
@pytest.fixture()
def flying(replaying: ViewportApp) -> ViewportApp:
    replaying._toggle_mode()
    assert replaying.mode == MODE_FLY
    return replaying


def test_fly_steps_real_sim_ticks(flying: ViewportApp) -> None:
    flying._tick(0.35)  # dt 0.1 -> 3 whole ticks, 0.05 stays in the accumulator
    assert flying.live is not None and flying.live.n == 3
    assert flying._acc == pytest.approx(0.05)


def test_left_arrow_banks_left_sim_convention(flying: ViewportApp) -> None:
    """Held LEFT arrow must push bank_cmd POSITIVE (sim: + = left turn), and
    the flown glider must actually roll that way."""
    flying.held["left arrow"] = 1
    flying._tick(0.5)
    assert flying.bank_cmd > 0.0
    assert flying.live is not None and flying.live.sim.state.bank > 0.0


def test_up_arrow_slows_speed_command(flying: ViewportApp) -> None:
    before = flying.pitch_cmd
    flying.held["up arrow"] = 1
    flying._tick(0.5)
    assert flying.pitch_cmd < before  # pull up = trade speed away


def test_save_writes_loadable_flight_log(flying: ViewportApp) -> None:
    flying._tick(1.0)
    flying._input("s")
    assert "saved" in flying.status
    saved = list(Path(flying.flights_dir).glob("manual-*.npz"))
    assert len(saved) == 1
    assert FlightLog.load(saved[0]).flight(0).n == flying.live.n  # type: ignore[union-attr]
    saved[0].unlink()  # keep the module-scoped fixture reusable


def test_save_with_nothing_flown_is_polite(flying: ViewportApp) -> None:
    flying._input("r")  # fresh flight, zero ticks
    flying._input("s")
    assert flying.status == "nothing flown yet"


def test_crash_shows_banner_and_stops_stepping(flying: ViewportApp) -> None:
    assert flying.live is not None
    flying.live.sim.crashed = True
    n = flying.live.n
    flying._tick(0.5)
    assert flying.live.n == n  # the world stopped responding
    assert "CRASHED" in flying.banner


def test_reset_gives_a_fresh_flight(flying: ViewportApp) -> None:
    flying.live.sim.crashed = True  # type: ignore[union-attr]
    flying._input("r")
    assert flying.live is not None and flying.live.n == 0 and not flying.live.crashed
    assert flying.banner == ""


# --------------------------------------------------------------- rendering
def test_render_every_mode_and_camera(replaying: ViewportApp) -> None:
    """The whole frame draws headless in replay AND fly, in every camera --
    the closest a unit test gets to 'the window works'."""
    surface = pygame.Surface((1280, 780))
    for _ in Camera.MODES:
        replaying._render(surface)
        replaying._input("tab")
    replaying._toggle_mode()  # fly mode
    replaying._tick(0.3)
    for _ in Camera.MODES:
        replaying._render(surface)
        replaying._input("tab")


# ----------------------------------------------------------- ghost-compare
@pytest.fixture(scope="module")
def ghost_app(
    mini_npz: Path, mini_rollouts: Path, tmp_path_factory: pytest.TempPathFactory
) -> ViewportApp:
    return ViewportApp(
        data_path=mini_npz,
        flights_dir=tmp_path_factory.mktemp("flights"),
        rollout_paths=[mini_rollouts],
    )


@pytest.fixture()
def ghosting(ghost_app: ViewportApp) -> ViewportApp:
    """The shared ghost app parked in episode 1 (where the rollouts live),
    ghost channel 'full' selected."""
    ghost_app._load_episode(1)
    ghost_app.ghost_idx = 1
    ghost_app.playing = False
    return ghost_app


def test_rollouts_load_as_channels_and_start_on(ghost_app: ViewportApp) -> None:
    """Passing a rollouts file yields one channel per predictor, already ON --
    whoever loads imaginations wants to see them."""
    assert [c.label for c in ghost_app.ghost_channels] == ["full", "twin"]
    assert ghost_app.ghost_idx == 1


def test_g_cycles_off_and_through_channels(ghosting: ViewportApp) -> None:
    ghosting._input("g")
    assert ghosting.ghost_idx == 2
    ghosting._input("g")
    assert ghosting.ghost_idx == 0  # off
    ghosting._input("g")
    assert ghosting.ghost_idx == 1


def test_g_without_rollouts_says_how_to_get_them(replaying: ViewportApp) -> None:
    replaying._input("g")
    assert "keystone" in replaying.status


def test_ghost_follows_the_playback_cursor(ghosting: ViewportApp) -> None:
    """The shared clock: before the first start there is NO ghost; inside a
    rollout the ghost is the latest start behind the cursor."""
    rset = ghosting.ghost_channels[0].rollouts
    first, second = (int(s) - ghosting.ep_row0 for s in rset.starts)
    ghosting.frame_pos = float(first - 1)
    assert ghosting._ghost() is None
    ghosting.frame_pos = float(first + 1)
    g = ghosting._ghost()
    assert g is not None and g.i == 0 and g.local_start == first
    ghosting.frame_pos = float(second)  # overlaps rollout 0's tail -> latest wins
    g = ghosting._ghost()
    assert g is not None and g.i == 1 and g.local_start == second


def test_ghost_panel_starts_true_then_diverges(ghosting: ViewportApp) -> None:
    """h=0 IS the true panel; conftest's 'full' predictor then drifts east
    2 m per step -- the imagined x must read exactly that ahead of truth."""
    assert ghosting.flight is not None
    rset = ghosting.ghost_channels[0].rollouts
    local = int(rset.starts[0]) - ghosting.ep_row0
    at_h = ghosting.ghost_channels[0].rollouts.panel
    assert at_h("full", 0, 0)["x"] == ghosting.flight.frame(local).sensors["x"]
    true_x3 = ghosting.flight.frame(local + 3).sensors["x"]
    assert at_h("full", 0, 3)["x"] == pytest.approx(true_x3 + 6.0)


def test_ghost_off_between_episodes_without_rollouts(ghosting: ViewportApp) -> None:
    ghosting._load_episode(0)  # rollouts live in episode 1 only
    ghosting.frame_pos = 5.0
    assert ghosting._ghost() is None
    surface = pygame.Surface((640, 400))
    ghosting._render(surface)  # and rendering does not mind


def test_render_paints_the_ghost_violet(ghosting: ViewportApp) -> None:
    """With the cursor inside a rollout, actual GHOST_VIOLET pixels appear
    (path/glider/panel title all wear it; any of them proves the layer ran)."""
    rset = ghosting.ghost_channels[0].rollouts
    ghosting.frame_pos = float(int(rset.starts[0]) - ghosting.ep_row0 + 2)
    ghosting.cam.mode_idx = Camera.MODES.index("topdown")
    surface = pygame.Surface((1280, 780))
    ghosting._render(surface)
    pixels = pygame.surfarray.pixels3d(surface)
    hit = bool((pixels == np.array(px(GHOST_VIOLET))).all(axis=-1).any())
    del pixels
    assert hit
    assert "ghost full" in ghosting._status_line()


def test_foreign_rollouts_are_refused(ghosting: ViewportApp, tmp_path: Path) -> None:
    """Rollouts index absolute rows of ONE dataset; a file whose answer key
    does not match the loaded log must be rejected, not scrubbed as garbage."""
    rset = ghosting.ghost_channels[0].rollouts
    np.savez_compressed(
        tmp_path / "foreign.npz",
        sensor_names=np.array(rset.sensor_names),
        dt=np.array(rset.dt),
        horizon=np.array(rset.horizon, dtype=np.int64),
        starts=rset.starts,
        episode=rset.episode,
        true=rset.true + 1.0,  # answer key that matches no real rows
        rollouts_full=rset.predictors["full"],
    )
    before = len(ghosting.ghost_channels)
    ghosting._bind_rollouts(RolloutSet.load(tmp_path / "foreign.npz"))
    assert len(ghosting.ghost_channels) == before
    assert "not from this dataset" in ghosting.status


# ------------------------------------------------------------------- modes
def test_without_dataset_boots_into_fly(tmp_path: Path) -> None:
    solo = ViewportApp(data_path=tmp_path / "missing.npz", flights_dir=tmp_path / "flights")
    assert solo.mode == MODE_FLY and solo.log is None
    solo._input("f")  # replay unavailable -> stays in fly, says why
    assert solo.mode == MODE_FLY
    assert "replay unavailable" in solo.status
    solo._tick(0.2)  # still flies fine
    assert solo.live is not None and solo.live.n == 2
