"""
viewport/hud.py -- draws the instrument panel + chrome onto a pygame surface.

This is the THIN drawing layer over viewport/panel.py: panel.py decides what
every gauge says (values, needle fractions, polarity, colors -- all unit
tested); this file only turns Readings into rectangles and text. Keep it
dumb: anything that could be WRONG belongs in panel.py where tests can grab
it.

Layout (fractions of the surface, so any window size works):
  left edge   -- the panel card: one row per gauge (label, value, and a
                 needle track for bar/center kinds).
  top center  -- status line (mode, episode, time, speed, camera).
  bottom      -- timeline (replay scrubber; click-to-seek hit-tested via
                 timeline_rect) above the key-help line.
  center      -- big red banner (CRASHED) + toast line (saved file, notes).
"""

from __future__ import annotations

from functools import lru_cache

import pygame

from viewport.colors import (
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    PANEL_BG,
    SINK_RED,
    SURFACE,
)
from viewport.panel import Reading
from viewport.scene import px, pxa

# panel geometry (pixels)
PANEL_X = 12
PANEL_W = 190
ROW_H = 34
TRACK_W = 150
TRACK_H = 4
TIMELINE_H = 8
TIMELINE_MARGIN = 120


@lru_cache(maxsize=8)
def font(size: int) -> pygame.font.Font:
    """One shared monospace font per size (tabular digits keep gauge values
    from jittering as they change)."""
    pygame.font.init()
    return pygame.font.SysFont("menlo,monaco,couriernew,monospace", size)


def _text(
    surface: pygame.Surface,
    s: str,
    pos: tuple[int, int],
    size: int = 14,
    color: tuple[float, float, float, float] = INK,
    align: str = "left",
) -> None:
    img = font(size).render(s, True, px(color))
    r = img.get_rect()
    if align == "left":
        r.topleft = pos
    elif align == "right":
        r.topright = pos
    else:
        r.midtop = pos
    surface.blit(img, r)


def draw_panel(
    surface: pygame.Surface,
    readings: list[Reading],
    x: int = PANEL_X,
    title: str = "",
    title_color: tuple[float, float, float, float] = INK,
) -> None:
    """One instrument column: a row per Reading, in panel order. `x` places
    the column (the default is the classic left card; the IMAGINED column
    passes a right-side x) and `title` names whose panel this is -- TRUE vs
    IMAGINED must never be mistakable for each other."""
    title_h = 22 if title else 0
    h = 18 + title_h + len(readings) * ROW_H
    card = pygame.Surface((PANEL_W, h), pygame.SRCALPHA)
    card.fill(pxa(PANEL_BG))
    surface.blit(card, (x, PANEL_X))

    y = PANEL_X + 12
    if title:
        _text(surface, title, (x + 10, y), 13, title_color)
        y += title_h
    for r in readings:
        _text(surface, r.spec.label, (x + 10, y), 12, INK_SECONDARY)
        _text(surface, r.text, (x + PANEL_W - 10, y), 14, r.color, align="right")
        if r.frac is not None:
            ty = y + 19
            track = pygame.Rect(x + 10, ty, TRACK_W, TRACK_H)
            pygame.draw.rect(surface, px(GRID), track)
            if r.spec.kind == "center":  # zero tick
                cx = track.left + TRACK_W // 2
                pygame.draw.line(surface, px(INK_MUTED), (cx, ty - 3), (cx, ty + TRACK_H + 2))
            nx = track.left + int(r.frac * TRACK_W)
            pygame.draw.rect(surface, px(r.color), pygame.Rect(nx - 2, ty - 4, 4, TRACK_H + 8))
        y += ROW_H


def timeline_rect(surface: pygame.Surface) -> pygame.Rect:
    """Where the scrubber lives -- also the click hit-target (padded)."""
    w, h = surface.get_size()
    return pygame.Rect(TIMELINE_MARGIN, h - 46, w - 2 * TIMELINE_MARGIN, TIMELINE_H)


def draw_timeline(
    surface: pygame.Surface,
    frac: float,
    marks: list[float] | None = None,
    mark_color: tuple[float, float, float, float] = INK_MUTED,
) -> None:
    """The scrubber. `marks` are extra ticks at fractional positions -- the
    ghost-compare uses them to show where each saved rollout begins."""
    track = timeline_rect(surface)
    pygame.draw.rect(surface, px(GRID), track)
    fill = track.copy()
    fill.width = max(0, int(track.width * min(1.0, max(0.0, frac))))
    pygame.draw.rect(surface, px(INK_SECONDARY), fill)
    for m in marks or []:
        mx = track.left + int(track.width * min(1.0, max(0.0, m)))
        pygame.draw.rect(surface, px(mark_color), pygame.Rect(mx - 1, track.top - 5, 2, 5))
    knob_x = track.left + fill.width
    pygame.draw.rect(surface, px(INK), pygame.Rect(knob_x - 2, track.top - 3, 4, track.height + 6))


def draw_chrome(
    surface: pygame.Surface,
    status: str,
    help_line: str,
    toast: str = "",
    banner: str = "",
) -> None:
    w, h = surface.get_size()
    _text(surface, status, (w // 2, 10), 14, INK, align="center")
    _text(surface, help_line, (w // 2, h - 26), 12, INK_MUTED, align="center")
    if toast:
        _text(surface, toast, (w // 2, h - 70), 13, INK, align="center")
    if banner:
        img = font(34).render(banner, True, px(SINK_RED))
        r = img.get_rect(center=(w // 2, h // 3))
        pad = r.inflate(24, 12)
        card = pygame.Surface(pad.size, pygame.SRCALPHA)
        card.fill(pxa(SURFACE, 0.85))
        surface.blit(card, pad)
        surface.blit(img, r)
