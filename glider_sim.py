"""
glider_sim.py -- a minimal point-mass glider in a thermal updraft field.

Pure Python + NumPy. There is NO machine learning in this file. This is just
physics: given the glider's current state and a control action (how steeply to
bank), compute the state one small time-step later.

Why this file matters: later, a neural network (the JEPA) will try to PREDICT
what this sim does. So this code is the "ground truth" of our little world --
it needs to be simple and you need to understand it. Read `step()` at the
bottom; everything else just feeds it.

The energy game of thermal soaring, in four lines:
  - Warm air rises in columns called "thermals."
  - A glider has no engine, so it always SINKS through the air around it. How
    fast it sinks depends on how hard it is turning.
  - But if the air it sits in is rising FASTER than it sinks through that air,
    the glider gains altitude. Free energy.
  - So: fly circles that stay inside the rising core of a thermal -> climb.
    Drift to the edge -> sink.

The whole tension is: turn tighter to stay in the core (good) -- but tighter
turns sink faster (bad). There is a sweet spot. That tension is what makes
this a real control problem worth predicting through.
"""

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. The thermal: a column of rising air.
# ---------------------------------------------------------------------------
@dataclass
class Thermal:
    """A rising-air column, modeled as a 2D Gaussian bump of vertical wind:
    strongest at the center (x0, y0), fading smoothly with horizontal distance.

    This is the simplest honest model. (The research doc mentions the NASA
    Allen 2006 model, which also varies the strength with altitude and adds a
    sinking ring around the rim -- we can bolt those on later. Start simple.)
    """
    x0: float = 0.0       # center, east  (meters)
    y0: float = 0.0       # center, north (meters)
    w_peak: float = 4.0   # peak updraft at the center (m/s, positive = up)
    radius: float = 60.0  # how wide the rising core is (meters)

    def updraft(self, x, y):
        """Vertical wind speed (m/s, up) at horizontal point (x, y).

        exp(-(r/R)^2) is a bell curve: 1.0 at the center, fading toward 0 far
        away. Works on single numbers OR NumPy arrays (handy for the heatmap).
        """
        r2 = (x - self.x0) ** 2 + (y - self.y0) ** 2   # squared dist from center
        return self.w_peak * np.exp(-r2 / self.radius ** 2)


# ---------------------------------------------------------------------------
# 2. The glider's state -- everything we need to know about it right now.
# ---------------------------------------------------------------------------
@dataclass
class GliderState:
    x: float        # east position  (m)
    y: float        # north position (m)
    z: float        # altitude       (m)
    heading: float  # which way it points in the horizontal plane (radians;
                    # 0 = east, pi/2 = north). Turning changes this.
    # NOTE: airspeed is held constant in this first version (the glider trims
    # to a steady speed). The bank angle is the CONTROL and is passed into
    # step() each tick, so it isn't stored here.


# ---------------------------------------------------------------------------
# 3. Constants describing THIS particular glider.
# ---------------------------------------------------------------------------
G = 9.81           # gravity (m/s^2)
AIRSPEED = 15.0    # constant forward speed through the air (m/s)
BASE_SINK = 0.7    # sink rate in straight, wings-level flight (m/s)


def sink_rate(bank_angle):
    """How fast the glider sinks through the surrounding air (m/s, positive =
    sinking), as a function of how steeply it banks.

    Wings level (bank = 0): sinks at BASE_SINK.
    Turning costs extra: in a banked turn the wings must support MORE than the
    glider's weight. The "load factor" n = 1/cos(bank) measures that extra
    loading (n = 1 level, grows without bound toward a 90-degree bank). More
    loading -> more drag -> more sink; a standard approximation is sink growing
    with n^1.5. THIS is the "tighter turns sink faster" penalty -- the cost
    side of the soaring trade-off.
    """
    load_factor = 1.0 / np.cos(bank_angle)     # n >= 1
    return BASE_SINK * load_factor ** 1.5


def turn_rate(bank_angle):
    """How fast the heading rotates (radians/sec) in a coordinated turn.

    Standard physics of a banked turn: heading_dot = g * tan(bank) / airspeed.
    Steeper bank -> faster turn -> tighter circle. (At bank = 0 this is 0, so
    the glider flies dead straight.)
    """
    return G * np.tan(bank_angle) / AIRSPEED


# ---------------------------------------------------------------------------
# 4. The ONE function that matters: advance the world by dt seconds.
# ---------------------------------------------------------------------------
def step(state: GliderState, bank_angle: float, thermal: Thermal,
         dt: float = 0.1) -> GliderState:
    """Advance the simulation one small time-step and return the new state.

    INPUT
      state       -- where the glider is now
      bank_angle  -- THE ACTION (radians). 0 = wings level (fly straight);
                     positive = bank and turn. ~0.5 rad (~30 deg) is a normal
                     thermalling turn. (This is what a controller -- dumb rule,
                     planner, or JEPA later -- gets to choose every tick.)
      thermal     -- the air it is flying through
      dt          -- time-step in seconds (smaller = more accurate, slower)

    OUTPUT
      the new GliderState, dt seconds later.

    This is plain forward-Euler integration: new_value = old_value + rate * dt.
    Three independent updates -- heading, horizontal position, altitude.
    """
    # 1) Rotate the heading according to how hard we're banking.
    new_heading = state.heading + turn_rate(bank_angle) * dt

    # 2) Slide horizontally in the direction we now point, at AIRSPEED.
    new_x = state.x + AIRSPEED * np.cos(new_heading) * dt
    new_y = state.y + AIRSPEED * np.sin(new_heading) * dt

    # 3) Vertical motion = (air rising here) minus (our sink through the air).
    #    Positive -> climbing; negative -> sinking. This single line is the
    #    whole energy game.
    climb_rate = thermal.updraft(state.x, state.y) - sink_rate(bank_angle)
    new_z = state.z + climb_rate * dt

    return GliderState(x=new_x, y=new_y, z=new_z, heading=new_heading)
