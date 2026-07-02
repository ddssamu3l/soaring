"""
viewport/camera.py -- ALL the 3D math of the viewport. Pure NumPy, no pygame.

The renderer draws vector graphics: every 3D point is pushed through THIS
file's Camera and comes out as a 2D pixel (or is flagged invisible). Keeping
the projection math free of any drawing library makes the "is the 3D right?"
question a unit test instead of a squint.

Everything speaks SIM coordinates: x east, y north, z up (meters). There is
no axis-swap anywhere -- the camera basis absorbs orientation.

Three modes (one Camera object, cycled with TAB):
  chase   -- perspective, behind and above the glider, looking through it.
            The flying view.
  topdown -- ORTHOGRAPHIC, straight down, north = screen-up, tracking the
            glider. The analysis view (this is where the heatmap ground
            reads like a chart).
  tower   -- perspective from a fixed corner mast, watching the glider
            cross the field. The spectator view.

How projection works (the whole trick, four lines of math):
  1. build an orthonormal camera basis: forward = where it looks,
     right = forward x world-up, up = right x forward.
  2. express (point - eye) in that basis -> camera coords (Xc right,
     Yc up, Zc depth).
  3. perspective: screen_x = cx + f*Xc/Zc, screen_y = cy - f*Yc/Zc, where
     f = (height/2)/tan(fov/2) is the focal length in pixels.
  4. anything with Zc <= near is BEHIND the lens -> invisible (the renderer
     drops those points/segments rather than drawing mirror-world garbage).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

NEAR = 1.0  # clip plane (m): closer than this is "behind the lens"

CHASE_BACK = 42.0  # chase eye: this far behind the glider...
CHASE_UP = 14.0  # ...and this far above it
CHASE_AHEAD = 25.0  # looking at a point this far ahead of it
TOWER_EYE = np.array([-450.0, -450.0, 280.0])  # the fixed corner mast
TOPDOWN_EXTENT = 650.0  # meters of world visible top-to-bottom in topdown


def heading_vec(heading: float) -> FloatArray:
    """Unit forward vector on the ground plane for a sim heading
    (radians CCW from east): 0 -> east, pi/2 -> north."""
    return np.array([math.cos(heading), math.sin(heading), 0.0])


def basis(eye: FloatArray, target: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Right-handed orthonormal camera basis (right, up, forward) looking
    from eye toward target, with world +z as the up reference."""
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    n = np.linalg.norm(right)
    if n < 1e-9:  # looking straight up/down: pick east as a stable right
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / n
    up = np.cross(right, fwd)
    return right, up, fwd


@dataclass
class Camera:
    """The viewport's one camera. update() re-aims it at the glider each
    frame; project() turns (n, 3) sim points into (n, 2) pixels + a
    visibility mask."""

    width: int
    height: int
    fov_deg: float = 70.0
    mode_idx: int = 0

    MODES = ("chase", "topdown", "tower")

    # aim state, refreshed by update()
    eye: FloatArray = field(default_factory=lambda: np.array([0.0, 0.0, 500.0]))
    _right: FloatArray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    _up: FloatArray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    _fwd: FloatArray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    _center: FloatArray = field(default_factory=lambda: np.zeros(3))  # topdown center

    @property
    def mode(self) -> str:
        return self.MODES[self.mode_idx]

    @property
    def forward(self) -> FloatArray:
        """Current look direction (unit, sim coords)."""
        return self._fwd

    def next_mode(self) -> str:
        self.mode_idx = (self.mode_idx + 1) % len(self.MODES)
        return self.mode

    @property
    def focal_px(self) -> float:
        return (self.height / 2.0) / math.tan(math.radians(self.fov_deg) / 2.0)

    @property
    def ortho_scale(self) -> float:
        """topdown pixels-per-meter."""
        return self.height / TOPDOWN_EXTENT

    def update(self, x: float, y: float, z: float, heading: float) -> None:
        """Re-aim for this frame, from the glider's pose."""
        pos = np.array([x, y, z])
        if self.mode == "chase":
            fwd = heading_vec(heading)
            self.eye = pos - fwd * CHASE_BACK + np.array([0.0, 0.0, CHASE_UP])
            self._right, self._up, self._fwd = basis(self.eye, pos + fwd * CHASE_AHEAD)
        elif self.mode == "tower":
            self.eye = TOWER_EYE.copy()
            self._right, self._up, self._fwd = basis(self.eye, pos)
        else:  # topdown: pure 2D affine, stored as a center to track
            self._center = pos

    def project(self, points: FloatArray) -> tuple[FloatArray, BoolArray]:
        """(n, 3) sim points -> ((n, 2) pixel coords, (n,) visible mask).
        Invisible points still get coordinates (their segment partner may be
        visible) but must not be drawn on their own."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        cx, cy = self.width / 2.0, self.height / 2.0

        if self.mode == "topdown":
            s = self.ortho_scale
            sx = cx + (pts[:, 0] - self._center[0]) * s
            sy = cy - (pts[:, 1] - self._center[1]) * s  # north = screen-up
            return np.stack([sx, sy], axis=1), np.ones(len(pts), dtype=bool)

        rel = pts - self.eye
        xc = rel @ self._right
        yc = rel @ self._up
        zc = rel @ self._fwd
        visible = zc > NEAR
        zsafe = np.where(visible, zc, NEAR)  # avoid divide-by-~0 for hidden pts
        f = self.focal_px
        sx = cx + f * xc / zsafe
        sy = cy - f * yc / zsafe
        return np.stack([sx, sy], axis=1), visible


def glider_points(
    x: float, y: float, z: float, heading: float, bank: float, wingspan: float
) -> dict[str, FloatArray]:
    """The glider's key 3D points from its pose -- the renderer connects
    them into wing/fuselage/fin strokes.

    Bank rolls the wing around the fuselage axis. Positive bank = LEFT turn
    in the sim (heading rate g*tan(bank)/V > 0 = CCW), so the LEFT tip drops:
    the same geometry sense() uses for the wingtip lift_asym cue."""
    pos = np.array([x, y, z])
    fwd = heading_vec(heading)
    left = np.array([-math.sin(heading), math.cos(heading), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    half = wingspan / 2.0
    wing_dir = left * math.cos(bank) - up * math.sin(bank)  # rolled span axis
    return {
        "nose": pos + fwd * 3.2,
        "tail": pos - fwd * 3.2,
        "left_tip": pos + wing_dir * half,
        "right_tip": pos - wing_dir * half,
        "fin_top": pos - fwd * 2.9 + (up * math.cos(bank) + left * math.sin(bank)) * 1.8,
    }
