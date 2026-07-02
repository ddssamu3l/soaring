"""Tests for viewport/hud.py -- the drawing layer over panel.py. Values and
polarity are panel.py's tests; here we pin layout mechanics and that every
element actually draws headless."""

import pygame

from glider_sim import ACTION_NAMES, SENSOR_NAMES
from viewport import hud
from viewport.colors import GRID, SKY
from viewport.panel import build_panel, read
from viewport.scene import px


def _surface() -> pygame.Surface:
    s = pygame.Surface((1280, 780))
    s.fill(px(SKY))
    return s


def test_font_is_cached() -> None:
    assert hud.font(14) is hud.font(14)


def test_draw_panel_paints_every_gauge() -> None:
    surface = _surface()
    specs = build_panel(SENSOR_NAMES, ACTION_NAMES)
    readings = [read(s, 1.0) for s in specs]
    hud.draw_panel(surface, readings)
    # the card region is no longer pure sky
    region = pygame.Rect(hud.PANEL_X, hud.PANEL_X, hud.PANEL_W, 18 + len(readings) * hud.ROW_H)
    sub = surface.subsurface(region)
    pixels = pygame.surfarray.pixels3d(sub)
    assert (pixels != px(SKY)).any()
    del pixels


def test_timeline_rect_fits_surface() -> None:
    surface = _surface()
    track = hud.timeline_rect(surface)
    assert surface.get_rect().contains(track)


def test_draw_timeline_fill_tracks_frac() -> None:
    surface = _surface()
    hud.draw_timeline(surface, 0.5)
    track = hud.timeline_rect(surface)
    quarter = surface.get_at((track.left + track.width // 4, track.centery))
    assert (quarter.r, quarter.g, quarter.b) != px(GRID)  # inside the fill
    right = surface.get_at((track.right - 3, track.centery))
    assert (right.r, right.g, right.b) == px(GRID)  # past the playhead: empty


def test_draw_timeline_clamps_silly_fracs() -> None:
    surface = _surface()
    hud.draw_timeline(surface, 7.0)
    hud.draw_timeline(surface, -3.0)  # no crash, no overflow


def test_draw_chrome_with_everything() -> None:
    surface = _surface()
    hud.draw_chrome(surface, "STATUS", "help line", toast="saved x.npz", banner="CRASHED")
    pixels = pygame.surfarray.pixels3d(surface)
    assert (pixels != px(SKY)).any()
    del pixels
