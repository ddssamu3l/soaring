"""
viewport/colors.py -- the viewport's entire color system. Pure math, no ursina.

Every color the 3D viewport shows is decided here, as (r, g, b, a) floats in
0..1 (what ursina's color.rgba eats). Keeping this file free of any rendering
import means the color logic is trivially unit-testable, and there is exactly
ONE place to retune the look.

Provenance: values come from a dataviz palette validated for the DARK surface
(CVD separation, contrast, lightness-monotone ramps -- checked by script, not
by eye, before being hard-coded here):
  - sequential blue ramp   -> updraft MAGNITUDE painted onto the terrain.
    Zero lift recedes into the surface color; strong lift glows bright.
  - diverging blue <-> red -> climb/sink POLARITY on the flight ribbon
    (blue = gaining, red = losing, neutral gray = holding). When the sim
    grows sinking air (step 5's NASA ring), the terrain switches to this
    same diverging logic: red arm below zero. updraft_color already routes
    negatives there, so that day costs nothing here.
  - violet                 -> reserved for the FUTURE model-ghost trajectory
    (t2's predicted path overlaid on truth). Never used for real flights.
  - ink/gray tokens        -> HUD text and chrome, never data.
"""

from __future__ import annotations

RGBA = tuple[float, float, float, float]

# stops for a piecewise-linear ramp: (position in 0..1, color at it)
Stops = tuple[tuple[float, RGBA], ...]


def _hex(code: str, alpha: float = 1.0) -> RGBA:
    """'#rrggbb' -> (r, g, b, a) floats. The one place hex is parsed."""
    code = code.lstrip("#")
    r, g, b = (int(code[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (r, g, b, alpha)


# --- scene chrome (never carries data) -------------------------------------
SKY = _hex("#0d0d0d")  # the page plane: near-black, so data glows
SURFACE = _hex("#1a1a19")  # dark chart surface: terrain at zero lift, panel bg
INK = _hex("#ffffff")  # primary HUD text
INK_SECONDARY = _hex("#c3c2b7")  # gauge labels
INK_MUTED = _hex("#898781")  # help line, units, ticks
GRID = _hex("#2c2c2a")  # hairlines: ground grid, gauge tracks
PANEL_BG = _hex("#1a1a19", 0.88)  # translucent HUD card over the 3D scene

# --- data colors ------------------------------------------------------------
CLIMB_BLUE = _hex("#3987e5")  # diverging pole: energy coming IN
SINK_RED = _hex("#e66767")  # diverging pole: energy going OUT
NEUTRAL = _hex("#383835")  # diverging midpoint: nothing happening
GHOST_VIOLET = _hex("#9085e9")  # t2 ghost trajectory -- reserved, unused today

# aviation nav lights orient the glider at a glance (real convention):
PORT_RED = _hex("#e66767")  # LEFT wingtip
STARBOARD_GREEN = _hex("#0ca30c")  # RIGHT wingtip

# sequential blue, dark-mode direction: near-zero sinks into the surface,
# maximum glows. Stops are palette steps 600 -> 200 (validated ordinal ramp).
UPDRAFT_STOPS: Stops = (
    (0.0, SURFACE),
    (0.2, _hex("#184f95")),
    (0.4, _hex("#256abf")),
    (0.6, _hex("#3987e5")),
    (0.8, _hex("#6da7ec")),
    (1.0, _hex("#9ec5f4")),
)

# red arm for sinking air (unused until the sim makes any; wired so it isn't
# a refactor later). Same construction, toward the red pole.
SINK_STOPS: Stops = (
    (0.0, SURFACE),
    (0.5, _hex("#a94444")),
    (1.0, SINK_RED),
)


def lerp(a: RGBA, b: RGBA, t: float) -> RGBA:
    """Straight-line blend between two colors, t in 0..1."""
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    )


def ramp(stops: Stops, t: float) -> RGBA:
    """Piecewise-linear color ramp. t is clamped into 0..1, then interpolated
    between the two stops that bracket it."""
    t = min(1.0, max(0.0, t))
    for (p0, c0), (p1, c1) in zip(stops, stops[1:], strict=False):
        if t <= p1:
            span = p1 - p0
            return lerp(c0, c1, (t - p0) / span if span > 0.0 else 0.0)
    return stops[-1][1]


def updraft_color(w: float, w_ref: float) -> RGBA:
    """Terrain color for vertical air speed w (m/s, + = up), scaled so that
    w_ref (the world's own peak lift) hits the brightest stop. Negative w
    (sinking air -- none in today's worlds) routes to the red arm."""
    if w < 0.0:
        return ramp(SINK_STOPS, -w / w_ref if w_ref > 0.0 else 0.0)
    return ramp(UPDRAFT_STOPS, w / w_ref if w_ref > 0.0 else 0.0)


def climb_color(climb: float, scale: float = 3.0) -> RGBA:
    """Flight-ribbon color for a climb rate (m/s): diverging blue (gaining)
    <-> neutral gray <-> red (losing), saturating at +/- scale."""
    t = min(1.0, max(-1.0, climb / scale))
    if t >= 0.0:
        return lerp(NEUTRAL, CLIMB_BLUE, t)
    return lerp(NEUTRAL, SINK_RED, -t)
