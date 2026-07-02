"""Tests for viewport/app.py -- the mode/input state machine, driven
deterministically through _tick(dt)/_input(key)/_render(surface). Headless:
no window, no clock (see conftest)."""

from pathlib import Path

import pygame
import pytest

from data_gen import generate_dataset
from viewport import hud
from viewport.app import MODE_FLY, MODE_REPLAY, ViewportApp
from viewport.camera import Camera
from viewport.frames import FlightLog

EP_LEN = 40


@pytest.fixture(scope="module")
def mini_npz(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "mini.npz"
    generate_dataset(n_rollouts=2, steps_per_rollout=EP_LEN, out_path=path, seed=3)
    return path


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
    replaying._input("]")
    assert replaying.ep_pos == 1
    replaying._input("]")  # 2 episodes -> wraps to 0
    assert replaying.ep_pos == 0
    replaying._input("[")
    assert replaying.ep_pos == 1


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


# ------------------------------------------------------------------- modes
def test_without_dataset_boots_into_fly(tmp_path: Path) -> None:
    solo = ViewportApp(data_path=tmp_path / "missing.npz", flights_dir=tmp_path / "flights")
    assert solo.mode == MODE_FLY and solo.log is None
    solo._input("f")  # replay unavailable -> stays in fly, says why
    assert solo.mode == MODE_FLY
    assert "replay unavailable" in solo.status
    solo._tick(0.2)  # still flies fine
    assert solo.live is not None and solo.live.n == 2
