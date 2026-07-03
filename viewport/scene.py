"""
viewport/scene.py -- the vector 3D renderer: sim world -> pygame strokes.

Everything is drawn as PROJECTED VECTOR GRAPHICS (lines, polygons, circles)
through camera.Camera -- no GPU, no shaders, no assets. That is the scope
guard that keeps "3D viewport" from becoming a graphics project, and it is
immune to the macOS OpenGL mess that killed the ursina attempt (documented
in progress.txt).

What gets drawn, and from what:
  heatmap  -- the updraft field sampled through ThermalMap.updraft on a
              grid, colored via colors.updraft_color (normalized to this
              world's own peak lift). In TOPDOWN it is blitted as a real
              image -- the ground IS the chart. In perspective views the
              same field appears as nested "glow" rings on the ground.
  world    -- a 100 m hairline grid + one translucent column and ground
              ring per thermal, all from the Thermal objects directly
              (legal: pixels for human eyes are outside the sensor
              firewall, which only guards MODEL inputs).
  ribbons  -- draw_path(): a flown trajectory as a polyline, one vertex per
              tick, colored by climb rate (diverging blue/red), or in one
              flat color for overlays -- t2's model-predicted GHOST path is
              just another draw_path call with flat_color=GHOST_VIOLET.
  glider   -- wing + fuselage + fin strokes from camera.glider_points, with
              real nav-light convention: RED = left (port) tip, GREEN =
              right (starboard). One glance tells you the roll even from
              dead astern.

Painter's order (no z-buffer; callers draw in this order): ground, world
chrome, ribbons, glider, HUD on top.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pygame

from glider_sim import ThermalMap
from viewport.camera import Camera, glider_points
from viewport.colors import (
    GRID,
    INK,
    INK_SECONDARY,
    PORT_RED,
    RGBA,
    SKY,
    STARBOARD_GREEN,
    climb_color,
    updraft_color,
)

FloatArray = npt.NDArray[np.float64]

# beyond this many path points, stride-downsample the drawn polyline
# (rendering only -- the recorded log keeps every tick).
MAX_RIBBON_POINTS = 1500

HEATMAP_PX_PER_M = 1.0  # topdown ground image resolution
COLUMN_TOP = 800.0  # thermal column height (m)


def px(c: RGBA) -> tuple[int, int, int]:
    """(r, g, b, a) floats -> pygame's 0..255 ints (alpha handled separately
    by drawing translucent bits onto SRCALPHA overlays)."""
    return (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))


def pxa(c: RGBA, alpha: float | None = None) -> tuple[int, int, int, int]:
    a = c[3] if alpha is None else alpha
    return (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), int(a * 255))


# ---------------------------------------------------------------------------
# The world, precomputed once per ThermalMap.
# ---------------------------------------------------------------------------
@dataclass
class WorldScene:
    """Everything static about the world, ready to draw every frame.
    w_ref is the field's own peak updraft -- what the brightest color means."""

    air: ThermalMap
    half_extent: float
    w_ref: float
    heatmap: pygame.Surface  # topdown ground image (1 px = 1 m)
    grid_lines: list[tuple[FloatArray, FloatArray]]  # 3D segment endpoints
    rings: list[FloatArray]  # per thermal: (n, 3) ground-ring points
    columns: list[list[tuple[FloatArray, FloatArray]]]  # per thermal: 3D wireframe
    glow_rings: list[list[tuple[FloatArray, RGBA]]]  # per thermal: (circle pts, color)


def build_world(air: ThermalMap, half_extent: float = 400.0) -> WorldScene:
    """Sample the field + precompute all static geometry."""
    # --- the heatmap image: one updraft sample per meter, vectorized.
    n = int(2 * half_extent * HEATMAP_PX_PER_M) + 1
    xs = np.linspace(-half_extent, half_extent, n)
    ys = np.linspace(half_extent, -half_extent, n)  # row 0 = NORTH edge (screen-up)
    gx, gy = np.meshgrid(xs, ys)
    w = np.asarray(air.updraft(gx, gy), dtype=np.float64)
    w_ref = max(float(w.max()), 1e-9)

    # vectorized ramp: sample updraft_color at 256 steps, index by w/w_ref.
    lut = np.array(
        [px(updraft_color(w_ref * i / 255.0, w_ref)) for i in range(256)], dtype=np.uint8
    )
    idx = np.clip(w / w_ref * 255.0, 0.0, 255.0).astype(np.uint8)
    rgb = lut[idx]  # (rows, cols, 3)
    # surfarray wants (x, y, rgb) = (cols, rows, 3)
    heatmap = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))

    # --- the 100 m grid, as 3D segments slightly above the ground.
    grid_lines: list[tuple[FloatArray, FloatArray]] = []
    for m in np.arange(-half_extent, half_extent + 1.0, 100.0):
        grid_lines.append((np.array([m, -half_extent, 0.3]), np.array([m, half_extent, 0.3])))
        grid_lines.append((np.array([-half_extent, m, 0.3]), np.array([half_extent, m, 0.3])))

    # --- per-thermal chrome: ground ring, wireframe column, glow rings.
    rings, columns, glow_rings = [], [], []
    for t in air.thermals:
        angles = np.linspace(0.0, 2.0 * math.pi, 49)
        ring = np.stack(
            [t.x0 + t.radius * np.cos(angles), t.y0 + t.radius * np.sin(angles), np.full(49, 0.5)],
            axis=1,
        )
        rings.append(ring)

        wires: list[tuple[FloatArray, FloatArray]] = []
        for a in np.arange(0.0, 2.0 * math.pi, math.pi / 6.0):  # 12 verticals
            base = np.array([t.x0 + t.radius * math.cos(a), t.y0 + t.radius * math.sin(a), 0.0])
            wires.append((base, base + np.array([0.0, 0.0, COLUMN_TOP])))
        for h in (COLUMN_TOP / 3.0, 2.0 * COLUMN_TOP / 3.0, COLUMN_TOP):  # 3 hoops
            hoop = np.stack(
                [
                    t.x0 + t.radius * np.cos(angles),
                    t.y0 + t.radius * np.sin(angles),
                    np.full(49, h),
                ],
                axis=1,
            )
            wires.extend((hoop[i], hoop[i + 1]) for i in range(len(hoop) - 1))
        columns.append(wires)

        # nested ground "glow" for perspective views: brightest small ring
        # out to a faint wide one -- the heatmap's stand-in when the ground
        # image can't be perspective-mapped.
        glows: list[tuple[FloatArray, RGBA]] = []
        for k, r_scale in ((0.35, 1.0), (0.7, 0.55), (1.1, 0.25), (1.6, 0.1)):
            radius = t.radius * k
            circle = np.stack(
                [
                    t.x0 + radius * np.cos(angles),
                    t.y0 + radius * np.sin(angles),
                    np.full(49, 0.2),
                ],
                axis=1,
            )
            glows.append((circle, updraft_color(t.w_peak * r_scale, w_ref)))
        glow_rings.append(glows)

    return WorldScene(
        air=air,
        half_extent=half_extent,
        w_ref=w_ref,
        heatmap=heatmap,
        grid_lines=grid_lines,
        rings=rings,
        columns=columns,
        glow_rings=glow_rings,
    )


# ---------------------------------------------------------------------------
# Per-frame drawing. All functions: (surface, camera, ...) -> None.
# ---------------------------------------------------------------------------
def _segment(
    surface: pygame.Surface,
    cam: Camera,
    a: FloatArray,
    b: FloatArray,
    color: tuple[int, int, int],
    width: int = 1,
) -> None:
    """Draw one 3D segment if both ends are in front of the lens."""
    pts, vis = cam.project(np.stack([a, b]))
    if bool(vis[0]) and bool(vis[1]):
        pygame.draw.line(surface, color, pts[0], pts[1], width)


def _polyline(
    surface: pygame.Surface,
    cam: Camera,
    points: FloatArray,
    color: tuple[int, int, int],
    width: int = 1,
) -> None:
    pts, vis = cam.project(points)
    for i in range(len(pts) - 1):
        if bool(vis[i]) and bool(vis[i + 1]):
            pygame.draw.line(surface, color, pts[i], pts[i + 1], width)


def draw_world(surface: pygame.Surface, cam: Camera, world: WorldScene) -> None:
    """Background + ground for the current camera mode."""
    surface.fill(px(SKY))
    if cam.mode == "topdown":
        # the real heatmap, scaled to ortho pixels and centered on the camera
        s = cam.ortho_scale * (1.0 / HEATMAP_PX_PER_M)
        w, h = world.heatmap.get_size()
        scaled = pygame.transform.smoothscale(world.heatmap, (int(w * s), int(h * s)))
        # world point (-half, +half) [NW corner] is heatmap pixel (0, 0)
        corner, _ = cam.project(np.array([[-world.half_extent, world.half_extent, 0.0]]))
        surface.blit(scaled, (corner[0, 0], corner[0, 1]))
    else:
        # perspective ground: horizon + thermal glow rings
        eye_dir = math.atan2(cam.forward[1], cam.forward[0])
        horizon = np.stack(
            [
                cam.eye[0] + 5000.0 * np.cos(eye_dir + np.linspace(-1.2, 1.2, 25)),
                cam.eye[1] + 5000.0 * np.sin(eye_dir + np.linspace(-1.2, 1.2, 25)),
                np.zeros(25),
            ],
            axis=1,
        )
        _polyline(surface, cam, horizon, px(GRID))
        for glows in world.glow_rings:
            for circle, color in glows:
                pts, vis = cam.project(circle)
                if bool(np.all(vis)):
                    corners = [(float(p[0]), float(p[1])) for p in pts]
                    pygame.draw.polygon(surface, px(color), corners, width=2)

    for a, b in world.grid_lines:
        _segment(surface, cam, a, b, px(GRID))
    for ring in world.rings:
        _polyline(surface, cam, ring, px(INK_SECONDARY), width=2)
    if cam.mode != "topdown":  # columns only read in perspective
        for wires in world.columns:
            for a, b in wires:
                _segment(surface, cam, a, b, px(GRID))


def draw_path(
    surface: pygame.Surface,
    cam: Camera,
    path: FloatArray,
    climbs: FloatArray | None = None,
    flat_color: RGBA | None = None,
    width: int = 2,
) -> None:
    """One trajectory: climb-diverging colors for real flights, flat color
    for overlays (the t2 ghost)."""
    if len(path) < 2:
        return
    stride = max(1, len(path) // MAX_RIBBON_POINTS)
    pts3 = path[::stride]
    pts, vis = cam.project(pts3)
    if flat_color is not None:
        colors = [px(flat_color)] * len(pts)
    elif climbs is not None:
        colors = [px(climb_color(float(c))) for c in climbs[::stride]]
    else:
        colors = [px(INK)] * len(pts)
    for i in range(len(pts) - 1):
        if bool(vis[i]) and bool(vis[i + 1]):
            pygame.draw.line(surface, colors[i], pts[i], pts[i + 1], width)


def draw_glider(
    surface: pygame.Surface,
    cam: Camera,
    x: float,
    y: float,
    z: float,
    heading: float,
    bank: float,
    wingspan: float,
    color: RGBA | None = None,
) -> None:
    """Wing + fuselage + fin strokes, nav lights on the tips. `color` paints
    the whole airframe one flat color with NO nav lights -- the ghost glider
    (a model belief has no hardware); default draws the real, lit aircraft."""
    g = glider_points(x, y, z, heading, bank, wingspan)
    order = ["nose", "tail", "left_tip", "right_tip", "fin_top"]
    pts, vis = cam.project(np.stack([g[k] for k in order]))
    if not bool(np.all(vis)):
        return  # partially behind the lens: skip rather than smear
    p = dict(zip(order, pts, strict=True))
    wing, body = (color, color) if color is not None else (INK, INK_SECONDARY)
    pygame.draw.line(surface, px(wing), p["left_tip"], p["right_tip"], 3)  # wing
    pygame.draw.line(surface, px(body), p["nose"], p["tail"], 2)  # fuselage
    pygame.draw.line(surface, px(body), p["tail"], p["fin_top"], 2)  # fin
    if color is None:
        pygame.draw.circle(surface, px(PORT_RED), p["left_tip"], 4)
        pygame.draw.circle(surface, px(STARBOARD_GREEN), p["right_tip"], 4)
