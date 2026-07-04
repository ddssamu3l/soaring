"""Tests for viewport/scene.py -- the world precomputation and every draw
call, exercised headless onto real surfaces (see conftest: SDL dummy)."""

import numpy as np
import pygame
import pytest

from data_gen import make_world
from viewport.camera import Camera
from viewport.colors import GHOST_VIOLET, PORT_RED, SKY, UPDRAFT_STOPS
from viewport.scene import (
    MAX_RIBBON_POINTS,
    WorldScene,
    build_world,
    draw_glider,
    draw_path,
    draw_world,
    px,
)


@pytest.fixture(scope="module")
def world() -> WorldScene:
    _, air = make_world()
    return build_world(air, half_extent=400.0)


@pytest.fixture()
def surface() -> pygame.Surface:
    return pygame.Surface((640, 400))


def _cam(mode: str, size: tuple[int, int] = (640, 400)) -> Camera:
    cam = Camera(width=size[0], height=size[1])
    cam.mode_idx = Camera.MODES.index(mode)
    cam.update(0.0, 0.0, 400.0, heading=0.0)
    return cam


# ------------------------------------------------------------------- world
def test_world_normalizes_to_its_own_peak(world: WorldScene) -> None:
    """Thermal A (4 m/s) dominates the t3 field; finite grid sampling and the
    sink band's overlapping tails may shave a hair off the exact peak."""
    assert world.w_ref == pytest.approx(4.0, abs=0.05)


def test_heatmap_encodes_the_field(world: WorldScene) -> None:
    """Center pixel (over the core) wears the ramp's TOP color; a far corner
    recedes into the surface color -- the ground really is the chart."""
    w, h = world.heatmap.get_size()
    assert (w, h) == (801, 801)  # 1 px per meter over +/-400 m
    center = world.heatmap.get_at((400, 400))
    assert (center.r, center.g, center.b) == px(UPDRAFT_STOPS[-1][1])
    corner = world.heatmap.get_at((3, 3))
    assert (corner.r, corner.g, corner.b) == px(UPDRAFT_STOPS[0][1])


def test_world_geometry_counts(world: WorldScene) -> None:
    assert len(world.grid_lines) == 18  # 9 north-south + 9 east-west hairlines
    n = len(make_world()[1].thermals)  # one ring+column per thermal, sink included
    assert len(world.rings) == n and len(world.columns) == n
    assert len(world.glow_rings[0]) == 4  # nested perspective glow


def test_draw_world_every_mode(world: WorldScene, surface: pygame.Surface) -> None:
    """Each camera mode draws without error and actually paints something."""
    for mode in Camera.MODES:
        draw_world(surface, _cam(mode), world)
        pixels = pygame.surfarray.pixels3d(surface)
        painted = (pixels != px(SKY)).any()
        del pixels  # release the surface lock
        assert painted, f"nothing painted in {mode}"


def test_topdown_ground_is_the_heatmap(world: WorldScene, surface: pygame.Surface) -> None:
    """Camera over the thermal: the screen must glow ~the top ramp color near
    center. (Sampled a few px off-center -- the 0-meter grid hairline runs
    exactly through the middle.)"""
    draw_world(surface, _cam("topdown"), world)
    got = surface.get_at((326, 194))
    top = px(UPDRAFT_STOPS[-1][1])
    assert all(abs(g - t) < 20 for g, t in zip((got.r, got.g, got.b), top, strict=True))


# ------------------------------------------------------------------- paths
def test_draw_path_climb_colored(world: WorldScene, surface: pygame.Surface) -> None:
    cam = _cam("topdown")
    path = np.array([[-50.0, 0.0, 400.0], [0.0, 0.0, 401.0], [50.0, 0.0, 399.0]])
    draw_path(surface, cam, path, climbs=np.array([2.0, 2.0, -2.0]))


def test_draw_path_ghost_flat_color(world: WorldScene, surface: pygame.Surface) -> None:
    """The t2 ghost-overlay contract: a flat violet polyline."""
    cam = _cam("topdown")
    path = np.array([[0.0, -100.0, 400.0], [0.0, 100.0, 400.0]])
    draw_path(surface, cam, path, flat_color=GHOST_VIOLET)
    assert surface.get_at((320, 150))[:3] == px(GHOST_VIOLET)


def test_draw_path_survives_points_behind_lens(surface: pygame.Surface) -> None:
    cam = _cam("chase")  # eye west of origin, looking east
    path = np.array([[-500.0, 0.0, 400.0], [0.0, 0.0, 400.0], [100.0, 0.0, 400.0]])
    draw_path(surface, cam, path, climbs=np.zeros(3))  # must not crash or smear


def test_draw_path_downsamples_long_flights(surface: pygame.Surface) -> None:
    n = MAX_RIBBON_POINTS * 4
    path = np.zeros((n, 3))
    path[:, 0] = np.linspace(-100, 100, n)
    path[:, 2] = 400.0
    draw_path(surface, _cam("topdown"), path, climbs=np.zeros(n))  # fast + no crash


def test_short_path_is_a_noop(surface: pygame.Surface) -> None:
    before = surface.get_at((320, 200))
    draw_path(surface, _cam("topdown"), np.zeros((1, 3)))
    assert surface.get_at((320, 200)) == before


# ------------------------------------------------------------------ glider
def test_draw_glider_port_light_is_red_and_north(surface: pygame.Surface) -> None:
    """Topdown, heading east: the LEFT (port) tip sits NORTH of center and
    wears the red nav light -- chirality visible in actual pixels."""
    cam = _cam("topdown")
    draw_glider(surface, cam, 0.0, 0.0, 400.0, heading=0.0, bank=0.0, wingspan=17.0)
    tip, _ = cam.project(np.array([[0.0, 8.5, 400.0]]))  # port tip, sim coords
    got = surface.get_at((int(tip[0, 0]), int(tip[0, 1])))
    assert (got.r, got.g, got.b) == px(PORT_RED)


def test_draw_glider_color_override_is_flat_and_unlit(surface: pygame.Surface) -> None:
    """The ghost airframe: one flat violet, and NO nav lights -- a model's
    belief must never be mistakable for the real aircraft."""
    cam = _cam("topdown")
    draw_glider(
        surface, cam, 0.0, 0.0, 400.0, heading=0.0, bank=0.0, wingspan=17.0, color=GHOST_VIOLET
    )
    tip, _ = cam.project(np.array([[0.0, 8.5, 400.0]]))  # port tip, sim coords
    got = surface.get_at((int(tip[0, 0]), int(tip[0, 1])))
    assert (got.r, got.g, got.b) == px(GHOST_VIOLET)  # wing stroke, not PORT_RED
    pixels = pygame.surfarray.pixels3d(surface)
    no_red = not bool((pixels == px(PORT_RED)).all(axis=-1).any())
    del pixels
    assert no_red


def test_draw_glider_skips_when_behind_lens(surface: pygame.Surface) -> None:
    cam = _cam("chase")
    before = pygame.surfarray.array3d(surface).copy()
    draw_glider(surface, cam, -500.0, 0.0, 400.0, 0.0, 0.0, 17.0)  # behind the eye
    assert (pygame.surfarray.array3d(surface) == before).all()
