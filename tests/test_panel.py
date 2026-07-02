"""Tests for viewport/panel.py -- gauge ranges, polarity, and the
new-sensor-appears-automatically guarantee."""

import math

from glider_sim import ACTION_NAMES, SENSOR_NAMES
from viewport.colors import INK
from viewport.panel import KNOWN_SPECS, GaugeSpec, build_panel, generic_spec, read


def _spec(name: str) -> GaugeSpec:
    return next(s for s in KNOWN_SPECS if s.name == name)


def test_panel_covers_every_channel_exactly_once() -> None:
    specs = build_panel(SENSOR_NAMES, ACTION_NAMES)
    assert sorted(s.name for s in specs) == sorted(SENSOR_NAMES + ACTION_NAMES)


def test_unknown_sensor_still_gets_a_gauge() -> None:
    """THE robustness contract: a sensor this file has never heard of shows
    up as a generic readout -- adding sensors never breaks the viewport."""
    specs = build_panel(SENSOR_NAMES + ("plasma_flux",), ACTION_NAMES)
    plasma = [s for s in specs if s.name == "plasma_flux"]
    assert len(plasma) == 1 and plasma[0].kind == "readout"
    r = read(plasma[0], 0.1234)
    assert "0.123" in r.text


def test_missing_channel_just_drops_off_the_panel() -> None:
    specs = build_panel(("z", "airspeed"), ())
    assert [s.name for s in specs] == ["z", "airspeed"]


def test_generic_spec_is_a_readout() -> None:
    s = generic_spec("mystery")
    assert s.kind == "readout" and s.label == "MYSTERY"


def test_bar_frac_clamps() -> None:
    ias = _spec("airspeed")
    assert read(ias, 10.0).frac == 0.0
    assert read(ias, 55.0).frac == 1.0
    assert read(ias, 500.0).frac == 1.0  # silly values can't break the needle
    assert read(ias, 32.5).frac is not None


def test_center_gauge_zero_sits_center() -> None:
    vario = _spec("vario")
    assert read(vario, 0.0).frac == 0.5
    assert read(vario, 4.0).frac == 1.0
    assert read(vario, -99.0).frac == 0.0


def test_left_positive_channels_render_needle_left() -> None:
    """Sim convention: positive bank / lift_asym mean LEFT. invert=True must
    put the needle physically LEFT of center (frac < 0.5)."""
    bank = _spec("bank")
    frac = read(bank, math.radians(30.0)).frac
    assert frac is not None and frac < 0.5
    asym = _spec("lift_asym")
    frac = read(asym, 0.5).frac
    assert frac is not None and frac < 0.5


def test_bank_displays_degrees() -> None:
    assert read(_spec("bank"), math.radians(45.0)).text.startswith("+45")


def test_heading_reads_as_compass() -> None:
    """Sim heading is radians CCW from EAST; pilots read degrees CW from
    NORTH. h=0 (east) -> 090, h=pi/2 (north) -> 000, h=pi (west) -> 270."""
    hdg = _spec("heading")
    assert read(hdg, 0.0).text.startswith("090")
    assert read(hdg, math.pi / 2).text.startswith("000")
    assert read(hdg, math.pi).text.startswith("270")
    assert read(hdg, 0.0).frac is None


def test_vario_wears_polarity_color() -> None:
    vario = _spec("vario")
    up, down = read(vario, 2.0), read(vario, -2.0)
    assert up.color[2] > up.color[0]  # climbing = blue-ish
    assert down.color[0] > down.color[2]  # sinking = red-ish
    assert read(_spec("airspeed"), 25.0).color == INK  # unsigned stays ink
