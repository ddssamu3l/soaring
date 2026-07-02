"""Tests for viewport/camera.py -- the projection math IS the 3D-ness of the
viewport, so it gets pinned hard: axes, chirality, clipping, both lens types."""

import math

import numpy as np
import pytest

from viewport.camera import TOPDOWN_EXTENT, Camera, basis, glider_points, heading_vec


def test_heading_vec_compass_axes() -> None:
    assert heading_vec(0.0) == pytest.approx([1.0, 0.0, 0.0])  # east
    assert heading_vec(math.pi / 2) == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)  # north


def test_basis_is_orthonormal_and_right_handed() -> None:
    eye = np.array([0.0, 0.0, 100.0])
    right, up, fwd = basis(eye, eye + np.array([1.0, 0.0, 0.0]))  # facing east
    for v in (right, up, fwd):
        assert np.linalg.norm(v) == pytest.approx(1.0)
    assert right @ up == pytest.approx(0.0, abs=1e-12)
    assert right == pytest.approx([0.0, -1.0, 0.0])  # facing east, right = south
    assert up == pytest.approx([0.0, 0.0, 1.0])


def test_straight_down_look_has_stable_basis() -> None:
    eye = np.array([0.0, 0.0, 500.0])
    right, up, fwd = basis(eye, np.zeros(3))  # degenerate: fwd parallel to world up
    assert np.linalg.norm(right) == pytest.approx(1.0)
    assert fwd == pytest.approx([0.0, 0.0, -1.0])


# ------------------------------------------------------------------- chase
def test_chase_look_target_projects_to_screen_center() -> None:
    cam = Camera(width=1000, height=800)
    cam.update(50.0, -20.0, 400.0, heading=0.7)
    target = np.array([50.0, -20.0, 400.0]) + heading_vec(0.7) * 25.0
    pts, vis = cam.project(target[None, :])
    assert bool(vis[0])
    assert pts[0] == pytest.approx([500.0, 400.0], abs=1.0)


def test_points_behind_the_lens_are_invisible() -> None:
    cam = Camera(width=1000, height=800)
    cam.update(0.0, 0.0, 400.0, heading=0.0)  # chase eye is WEST of the glider
    behind = np.array([[-500.0, 0.0, 400.0]])  # far west: behind the camera
    _, vis = cam.project(behind)
    assert not bool(vis[0])


def test_closer_objects_project_bigger() -> None:
    """Perspective sanity: the same 10 m offset spans more pixels up close."""
    cam = Camera(width=1000, height=800)
    cam.update(0.0, 0.0, 400.0, heading=0.0)
    near_pair = np.array([[50.0, -5.0, 400.0], [50.0, 5.0, 400.0]])
    far_pair = np.array([[300.0, -5.0, 400.0], [300.0, 5.0, 400.0]])
    (n1, n2), _ = cam.project(near_pair)
    (f1, f2), _ = cam.project(far_pair)
    assert abs(n2[0] - n1[0]) > abs(f2[0] - f1[0]) * 2


# ----------------------------------------------------------------- topdown
def test_topdown_is_north_up_east_right() -> None:
    cam = Camera(width=1000, height=800)
    cam.mode_idx = cam.MODES.index("topdown")
    cam.update(0.0, 0.0, 400.0, heading=0.3)
    pts, vis = cam.project(np.array([[0.0, 100.0, 0.0], [100.0, 0.0, 0.0]]))
    assert bool(np.all(vis))
    north, east = pts
    assert north[0] == pytest.approx(500.0) and north[1] < 400.0  # up
    assert east[1] == pytest.approx(400.0) and east[0] > 500.0  # right
    # scale: TOPDOWN_EXTENT meters top-to-bottom
    assert 400.0 - north[1] == pytest.approx(100.0 * 800 / TOPDOWN_EXTENT)


def test_camera_mode_cycles() -> None:
    cam = Camera(width=100, height=100)
    seen = [cam.mode] + [cam.next_mode() for _ in range(len(Camera.MODES) - 1)]
    assert sorted(seen) == sorted(Camera.MODES)
    assert cam.next_mode() == seen[0]


# ------------------------------------------------------------------ glider
def test_glider_points_chirality() -> None:
    """Facing EAST, the pilot's LEFT is NORTH. Pins the side the red nav
    light lives on -- a mirror bug would swap port and starboard."""
    g = glider_points(10.0, 20.0, 300.0, heading=0.0, bank=0.0, wingspan=17.0)
    assert g["left_tip"][1] == pytest.approx(20.0 + 8.5)  # north of the fuselage
    assert g["right_tip"][1] == pytest.approx(20.0 - 8.5)
    assert g["nose"][0] > 10.0  # nose points east
    assert g["fin_top"][2] > 300.0  # fin sticks up


def test_positive_bank_drops_left_wing() -> None:
    """Sim: positive bank = LEFT turn = left wing DOWN (same geometry the
    lift_asym sensor samples with)."""
    g = glider_points(0.0, 0.0, 300.0, heading=0.9, bank=math.radians(35.0), wingspan=17.0)
    assert g["left_tip"][2] < 300.0 < g["right_tip"][2]


def test_bank_shrinks_horizontal_span() -> None:
    flat = glider_points(0.0, 0.0, 300.0, 0.0, 0.0, 17.0)
    banked = glider_points(0.0, 0.0, 300.0, 0.0, math.radians(60.0), 17.0)
    dy_flat = flat["left_tip"][1] - flat["right_tip"][1]
    dy_banked = banked["left_tip"][1] - banked["right_tip"][1]
    assert dy_banked == pytest.approx(dy_flat * 0.5, rel=1e-6)  # cos(60) = 0.5
