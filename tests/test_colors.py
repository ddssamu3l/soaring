"""Tests for viewport/colors.py -- the color math is pure, so test it hard."""

import pytest

from viewport.colors import (
    CLIMB_BLUE,
    NEUTRAL,
    SINK_RED,
    SURFACE,
    UPDRAFT_STOPS,
    climb_color,
    lerp,
    ramp,
    updraft_color,
)


def _luma(c: tuple[float, float, float, float]) -> float:
    """Cheap perceived brightness -- enough to check ramp monotonicity."""
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def test_lerp_endpoints_and_midpoint() -> None:
    a, b = (0.0, 0.0, 0.0, 1.0), (1.0, 0.5, 0.0, 1.0)
    assert lerp(a, b, 0.0) == a
    assert lerp(a, b, 1.0) == b
    mid = lerp(a, b, 0.5)
    assert mid[0] == pytest.approx(0.5) and mid[1] == pytest.approx(0.25)


def test_ramp_hits_its_stops_and_clamps() -> None:
    assert ramp(UPDRAFT_STOPS, 0.0) == UPDRAFT_STOPS[0][1]
    assert ramp(UPDRAFT_STOPS, 1.0) == UPDRAFT_STOPS[-1][1]
    # out-of-range inputs clamp instead of exploding
    assert ramp(UPDRAFT_STOPS, -5.0) == UPDRAFT_STOPS[0][1]
    assert ramp(UPDRAFT_STOPS, 5.0) == UPDRAFT_STOPS[-1][1]


def test_updraft_ramp_brightness_is_monotone() -> None:
    """Stronger lift must never look DARKER -- the magnitude encoding."""
    ws = [i / 20 * 4.0 for i in range(21)]
    lumas = [_luma(updraft_color(w, w_ref=4.0)) for w in ws]
    assert all(b >= a - 1e-9 for a, b in zip(lumas, lumas[1:], strict=False))


def test_updraft_zero_recedes_into_surface() -> None:
    assert updraft_color(0.0, w_ref=4.0) == SURFACE


def test_updraft_negative_routes_to_red_arm() -> None:
    c = updraft_color(-4.0, w_ref=4.0)
    assert c[0] > c[2]  # more red than blue


def test_updraft_degenerate_w_ref_is_safe() -> None:
    assert updraft_color(3.0, w_ref=0.0) == SURFACE


def test_climb_color_poles_and_neutral() -> None:
    assert climb_color(0.0) == NEUTRAL
    assert climb_color(99.0) == CLIMB_BLUE  # saturates at the pole
    assert climb_color(-99.0) == SINK_RED
    # polarity: climbing is bluer than sinking
    up, down = climb_color(2.0), climb_color(-2.0)
    assert up[2] > up[0] and down[0] > down[2]
