"""
glider_sim.py -- a minimal point-mass glider in a thermal updraft field.

Pure Python + NumPy. There is NO machine learning in this file. This is just
physics: given the glider's current state and a control action (how steeply to
bank), compute the state one small time-step later.

Why this file matters: later, a neural network (the JEPA) will try to PREDICT
what this sim does. So this code is the "ground truth" of our little world --
it needs to be simple and you need to understand it.

The world is built from FOUR objects, each owning one idea:
  - Thermal       : one column of rising air.
  - ThermalMap    : ALL the air -- a list of thermals + a base wind. ("the air")
  - Glider        : the AIRFRAME -- constant params (how it sinks/turns).
  - GliderState   : the glider's DYNAMIC state right now (x, y, z, heading).
  - Simulation    : the WORLD -- holds a Glider + a ThermalMap + the current
                    GliderState, owns step()/sense(), and records history.

The split that matters for the whole project:
  Simulation is OMNISCIENT (it owns the true thermal map).
  A future MLP is NOT -- it will only ever get what `sense()` returns plus the
  glider's own kinematics + airframe params + the action. The thermal's true
  (x0, y0, w_peak, radius) must NEVER reach the model. `sense()` is the one
  legitimate channel the thermal reaches a learner: felt, never told.

The energy game of thermal soaring, in four lines:
  - Warm air rises in columns called "thermals."
  - A glider has no engine, so it always SINKS through the air around it. How
    fast it sinks depends on how hard it is turning.
  - But if the air it sits in is rising FASTER than it sinks through that air,
    the glider gains altitude. Free energy.
  - So: fly circles that stay inside the rising core of a thermal -> climb.
    Drift to the edge -> sink.

The whole tension: turn tighter to stay in the core (good) -- but tighter turns
sink faster (bad). There is a sweet spot. That tension is what makes this a real
control problem worth predicting through.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# updraft() works on a single point (float) OR a whole grid (NumPy array, for the
# heatmap). This alias names "a number or an array of numbers" so the signatures
# stay honest about both.
Field = float | npt.NDArray[np.float64]

G = 9.81  # gravity (m/s^2) -- the one true constant of the universe here


# ---------------------------------------------------------------------------
# 1. The thermal: a single column of rising air.
# ---------------------------------------------------------------------------
@dataclass
class Thermal:
    """A rising-air column, modeled as a 2D Gaussian bump of vertical wind:
    strongest at the center (x0, y0), fading smoothly with horizontal distance.

    Simplest honest model. (NASA Allen 2006 also varies strength with altitude
    and adds a sinking ring at the rim -- bolt on later. Start simple.)
    """

    x0: float = 0.0  # center, east  (meters)
    y0: float = 0.0  # center, north (meters)
    w_peak: float = 4.0  # peak updraft at the center (m/s, positive = up)
    radius: float = 60.0  # how wide the rising core is (meters)

    def updraft(self, x: Field, y: Field) -> Field:
        """Vertical wind speed (m/s, up) at horizontal point (x, y).

        exp(-(r/R)^2) is a bell curve: 1.0 at the center, fading toward 0 far
        away. Works on single numbers OR NumPy arrays (handy for the heatmap).
        """
        r2 = (x - self.x0) ** 2 + (y - self.y0) ** 2  # squared dist from center
        return self.w_peak * np.exp(-r2 / self.radius**2)


# ---------------------------------------------------------------------------
# 2. The thermal map: ALL the air. A bag of thermals + a base wind.
# ---------------------------------------------------------------------------
@dataclass
class ThermalMap:
    """The whole air field the glider flies through.

    `wind` is the air's bulk horizontal motion (east, north) in m/s. DEFAULT
    (0, 0) -- a still day. It does nothing scientifically in the fixed-field
    phase (a uniform wind is just a moving coordinate frame), but it's the
    structural hook for two later steps: thermals that DRIFT with the wind, and
    the wind-SHEAR layer that makes dynamic soaring possible. Wired now so we
    never have to refactor it in; left at 0 so it can't surprise us.
    """

    thermals: list[Thermal]
    wind: tuple[float, float] = (0.0, 0.0)  # base wind (east, north) m/s -- DEFAULT: none

    def updraft(self, x: Field, y: Field) -> Field:
        """Vertical wind at (x, y) = sum of every thermal's contribution.
        (With one thermal this is just that thermal. Arrays work fine.)"""
        total: Field = 0.0
        for t in self.thermals:
            total = total + t.updraft(x, y)
        return total


# ---------------------------------------------------------------------------
# 3. The glider: the AIRFRAME. Constant params + how it sinks/turns.
# ---------------------------------------------------------------------------
@dataclass
class Glider:
    """Everything intrinsic to THIS aircraft -- the stuff you'd tune to "plug a
    different glider." None of it changes during a flight. (Airspeed is held
    constant in v1: the glider trims to a steady speed.)
    """

    airspeed: float = 15.0  # constant forward speed through the air (m/s)
    base_sink: float = 0.7  # sink rate in straight, wings-level flight (m/s)

    def sink_rate(self, bank_angle: float) -> float:
        """How fast the glider sinks THROUGH the surrounding air (m/s, positive
        = sinking), as a function of bank angle.

        Wings level (bank = 0): sinks at base_sink.
        Turning costs extra: in a banked turn the wings must support MORE than
        the glider's weight. Load factor n = 1/cos(bank) measures that extra
        loading (1 level, growing toward a 90-deg bank). More loading -> more
        drag -> more sink; a standard approximation is sink ~ n^1.5. THIS is the
        "tighter turns sink faster" penalty -- the cost side of the trade-off.
        """
        load_factor = 1.0 / np.cos(bank_angle)  # n >= 1
        return float(self.base_sink * load_factor**1.5)

    def turn_rate(self, bank_angle: float) -> float:
        """How fast the heading rotates (rad/s) in a coordinated turn.

        Standard banked-turn physics: heading_dot = g * tan(bank) / airspeed.
        Steeper bank -> faster turn -> tighter circle. (bank = 0 -> 0 -> straight.)
        """
        return float(G * np.tan(bank_angle) / self.airspeed)


# ---------------------------------------------------------------------------
# 4. The glider's dynamic state -- what we know about it RIGHT NOW.
# ---------------------------------------------------------------------------
@dataclass
class GliderState:
    x: float  # east position  (m)
    y: float  # north position (m)
    z: float  # altitude       (m)
    heading: float  # horizontal pointing direction (radians; 0 = east,
    # pi/2 = north). Turning changes this.
    # airspeed lives on the Glider (constant). bank is the CONTROL, passed into
    # step() each tick -- neither is stored here.


# ---------------------------------------------------------------------------
# 5. The simulation: the WORLD. Advances time, simulates sensors, records.
# ---------------------------------------------------------------------------
class Simulation:
    """One virtual world + the glider living in it. Run many independently.

    Holds the omniscient truth (glider, air, current state) and exposes exactly
    two verbs the rest of the project leans on:
        step(bank) -- advance dt seconds, return the new GliderState.
        sense()    -- what a real onboard sensor would feel right now.

    `history` accumulates one row per step -- (state_before, bank, vario) -- so a
    finished Simulation IS a logged rollout (the seed of the training set).
    """

    def __init__(self, glider: Glider, air: ThermalMap, state: GliderState, dt: float = 0.1):
        self.glider = glider
        self.air = air  # the OMNISCIENT truth -- never handed to a model
        self.state = state
        self.dt = dt
        # rows: (GliderState before step, bank, vario)
        self.history: list[tuple[GliderState, float, float]] = []

    def sense(self) -> dict[str, float]:
        """The glider's onboard reading at its CURRENT position.

        For now just the variometer: the air's vertical velocity right here. This
        is a LOCAL measurement (a real vario gives it) -- NOT the thermal's
        location. It is the only legitimate way the thermal's existence reaches a
        learner. More sensors slot in here later; kept a plain dict so adding one
        is a one-line change, not a new class.
        """
        return {"vario": float(self.air.updraft(self.state.x, self.state.y))}

    def step(self, bank_angle: float) -> GliderState:
        """Advance the world one time-step (forward-Euler) and return new state.

        bank_angle is THE ACTION (radians): 0 = wings level (straight); positive
        = bank and turn. ~0.5 rad (~30 deg) is a normal thermalling turn. This is
        what a controller -- dumb rule, planner, or JEPA later -- picks each tick.

        Three independent updates: heading, horizontal position, altitude.
        """
        g, s, dt = self.glider, self.state, self.dt
        wx, wy = self.air.wind

        # record where we are + what we did + what we feel, BEFORE moving, so
        # history rows line up as (state_t, action_t, ...) and self.state ends at
        # state_{t+1}.
        self.history.append((s, bank_angle, self.sense()["vario"]))

        # 1) Rotate heading according to how hard we're banking.
        new_heading = s.heading + g.turn_rate(bank_angle) * dt

        # 2) Slide horizontally: airspeed in the new heading, PLUS the air's bulk
        #    drift (the wind carries the whole glider along). wind=0 -> no drift.
        new_x = s.x + (g.airspeed * np.cos(new_heading) + wx) * dt
        new_y = s.y + (g.airspeed * np.sin(new_heading) + wy) * dt

        # 3) Vertical = (air rising here) minus (our sink through the air).
        #    Positive -> climbing; negative -> sinking. The whole energy game.
        climb_rate = float(self.air.updraft(s.x, s.y)) - g.sink_rate(bank_angle)
        new_z = s.z + climb_rate * dt

        self.state = GliderState(x=new_x, y=new_y, z=new_z, heading=new_heading)
        return self.state
