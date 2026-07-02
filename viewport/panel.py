"""
viewport/panel.py -- the instrument panel MODEL. Pure logic, no ursina.

Turns a Frame (name-keyed channel values) into gauge Readings (label, text,
needle position). The split matters: everything that could be WRONG about a
gauge -- ranges, units, formatting, needle polarity -- lives here where a unit
test can grab it; the ursina layer just draws Readings.

Robustness contract (same as frames.py): the panel is built FROM the channel
name lists, at runtime. Channels with a hand-tuned spec below get a proper
gauge; any channel this file has never heard of still gets a generic numeric
readout via generic_spec(). So a new sensor added to glider_sim shows up on
the panel by itself -- unstyled, but present and correct.

Gauge kinds:
  readout -- number + unit, no needle (position, altitude).
  bar     -- fill from the LEFT edge, for one-signed magnitudes (airspeed).
  center  -- needle around a CENTER zero, for signed quantities (vario, bank,
             lift_asym). `invert=True` flips display polarity for channels
             where positive means LEFT in the sim's convention (positive bank
             = left turn; positive lift_asym = stronger lift on the LEFT
             wing), so the needle physically points the way the thing leans.
  heading -- special-cased: sim heading is radians CCW from EAST; pilots read
             a compass, degrees CLOCKWISE from NORTH. Converted for display.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from viewport.colors import INK, RGBA, climb_color

_RAD2DEG = 180.0 / math.pi


@dataclass(frozen=True)
class GaugeSpec:
    """How to display one channel. lo/hi bound the needle travel in DISPLAY
    units (after unit_scale); for kind='center' they are symmetric +/-span."""

    name: str  # channel name in the Frame
    label: str  # short text on the panel
    unit: str
    kind: str  # 'readout' | 'bar' | 'center' | 'heading'
    lo: float = 0.0
    hi: float = 1.0
    fmt: str = "{:.1f}"
    unit_scale: float = 1.0  # raw value * this = display value (rad -> deg)
    invert: bool = False  # center gauges: positive value -> needle LEFT


@dataclass(frozen=True)
class Reading:
    """One gauge, ready to draw: formatted text + needle position.
    frac is the needle position in 0..1 across the gauge track (None for
    readouts); color tints the needle/value when the value carries polarity."""

    spec: GaugeSpec
    value: float
    text: str
    frac: float | None
    color: RGBA


# Hand-tuned gauges for the channels we know today. ORDER = panel order.
# Anything not listed still renders via generic_spec() -- see build_panel().
KNOWN_SPECS: tuple[GaugeSpec, ...] = (
    GaugeSpec("z", "ALT", "m", "readout", fmt="{:.0f}"),
    GaugeSpec("airspeed", "IAS", "m/s", "bar", lo=10.0, hi=55.0),
    GaugeSpec("heading", "HDG", "\N{DEGREE SIGN}", "heading", fmt="{:03.0f}"),
    GaugeSpec(
        "bank",
        "BANK",
        "\N{DEGREE SIGN}",
        "center",
        lo=-60.0,
        hi=60.0,
        fmt="{:+.0f}",
        unit_scale=_RAD2DEG,
        invert=True,
    ),
    GaugeSpec("vario", "VARIO", "m/s", "center", lo=-4.0, hi=4.0, fmt="{:+.1f}"),
    GaugeSpec("vario_te", "TE VARIO", "m/s", "center", lo=-4.0, hi=4.0, fmt="{:+.1f}"),
    GaugeSpec("lift_asym", "ASYM", "m/s", "center", lo=-1.0, hi=1.0, fmt="{:+.2f}", invert=True),
    GaugeSpec("x", "EAST", "m", "readout", fmt="{:.0f}"),
    GaugeSpec("y", "NORTH", "m", "readout", fmt="{:.0f}"),
    GaugeSpec(
        "bank_cmd",
        "BANK CMD",
        "\N{DEGREE SIGN}",
        "center",
        lo=-60.0,
        hi=60.0,
        fmt="{:+.0f}",
        unit_scale=_RAD2DEG,
        invert=True,
    ),
    GaugeSpec("pitch_cmd", "SPD CMD", "m/s", "bar", lo=10.0, hi=55.0),
)

# channels whose sign means energy in/out -- their needle wears the diverging
# climb/sink color so polarity is readable across the cockpit.
_ENERGY_CHANNELS = {"vario", "vario_te"}


def generic_spec(name: str) -> GaugeSpec:
    """The fallback gauge for a channel nobody has styled yet: a plain
    readout with a general-purpose number format. Guarantees NEW sensors
    appear on the panel with zero viewport edits."""
    return GaugeSpec(name, name.upper(), "", "readout", fmt="{:+.3g}")


def build_panel(sensor_names: tuple[str, ...], action_names: tuple[str, ...]) -> list[GaugeSpec]:
    """The panel for THIS flight's channels: known specs in their curated
    order first (only those actually present), then generic gauges for any
    channel the spec table doesn't know, in the source's own order."""
    present = list(sensor_names) + list(action_names)
    known = [s for s in KNOWN_SPECS if s.name in present]
    known_names = {s.name for s in known}
    unknown = [generic_spec(n) for n in present if n not in known_names]
    return known + unknown


def read(spec: GaugeSpec, value: float) -> Reading:
    """Evaluate one gauge: format the display value, place the needle."""
    shown = value * spec.unit_scale
    frac: float | None
    if spec.kind == "heading":
        # radians CCW-from-east -> compass degrees CW-from-north
        shown = (90.0 - value * _RAD2DEG) % 360.0
        frac = None
    elif spec.kind == "readout":
        frac = None
    else:  # bar | center: needle position across the track, clamped
        span = spec.hi - spec.lo
        frac = min(1.0, max(0.0, (shown - spec.lo) / span if span > 0.0 else 0.0))
        if spec.kind == "center" and spec.invert:
            frac = 1.0 - frac
    color = climb_color(value) if spec.name in _ENERGY_CHANNELS else INK
    text = f"{spec.fmt.format(shown)}{spec.unit}"
    return Reading(spec=spec, value=value, text=text, frac=frac, color=color)
