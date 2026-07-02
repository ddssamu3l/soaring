"""
glider_sim.py -- a point-mass glider with real airframe physics, in a thermal
updraft field.

Pure Python + NumPy. There is NO machine learning in this file. This is just
physics: given the glider's current state and the pilot's two stick commands
(bank + pitch), compute the state one small time-step later.

Why this file matters: later, a neural network (the JEPA) will try to PREDICT
what this sim does. So this code is the "ground truth" of our little world --
it needs to be simple and you need to understand it.

The world is built from these objects, each owning one idea:
  - Thermal       : one column of rising air.
  - ThermalMap    : ALL the air -- a list of thermals + a base wind. ("the air")
  - Glider        : the AIRFRAME -- mass, glide polar, limits. Constant.
  - GliderState   : the glider's DYNAMIC state (x, y, z, heading, airspeed, bank).
  - Simulation    : the WORLD -- holds a Glider + a ThermalMap + the current
                    GliderState, owns step()/sense(), records history.

NAMING CONVENTION (load-bearing -- the firewall lives in the names):
  - `<name>_cmd`  : what the controller ASKED for (the action channel).
  - bare `<name>` : what actually IS (measured reality -- state & sensors).
  - `true_*`      : omniscient ground truth, for evaluation ONLY (dataset files).
  Models eat sense() + actions + Glider params. Anything `true_*` in a model's
  diet is a bug. The thermal's (x0, y0, w_peak, radius) must NEVER reach a
  model: the air is FELT (vario), never TOLD.

THE PHYSICS, in five ideas:
  1. THE POLAR. A glider always sinks through the air around it; how fast
     depends on airspeed. The sink-vs-speed curve (the "glide polar") is the
     airframe's signature: slowest sink at min-sink speed (~19 m/s here),
     best distance-per-height at the faster best-glide speed. Fly slower or
     faster than that and you pay.
  2. ENERGY EXCHANGE. Pitch does not create climb -- it TRADES speed for
     height through E = m*g*h + 1/2*m*V^2. Nose down: h -> V. Pull up: V -> h
     (a "zoom"). The exchange is exact by construction here; total energy only
     DRAINS through the polar and only GROWS from rising air. Gravity sets the
     exchange rate (~5 m of height buys ~1 m/s at these speeds).
  3. COMMAND LAG. Stick commands are not teleports: bank slews toward bank_cmd
     at a max roll rate, and airspeed relaxes toward the pitch set-point at a
     max longitudinal acceleration. The gap between commanded and actual IS the
     vehicle's response dynamics -- part of what a world model must learn.
  4. STALL. Below stall speed the wing cannot hold the glider: the nose drops
     (speed force-rebuilds), sink spikes, then flight resumes. Stall speed
     rises with sqrt(load factor) -- steep slow turns are the biting corner.
  5. THE GROUND. z <= 0 ends the flight. Energy is survival, not a score.

The soaring game: air rising faster than your polar sink -> free altitude.
Circle tight enough to stay in the core, slow enough to sink little, fast
enough not to stall. That three-way tension is the whole sport.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# updraft() works on a single point (float) OR a whole grid (NumPy array, for the
# heatmap). This alias names "a number or an array of numbers" so the signatures
# stay honest about both.
Field = float | npt.NDArray[np.float64]

G = 9.81  # gravity (m/s^2) -- works through the turn, the exchange, the polar

# Canonical channel orderings -- datasets and model code key off these, so they
# live HERE, next to the physics that defines them.
STATE_NAMES = ("x", "y", "z", "heading", "airspeed", "bank")
ACTION_NAMES = ("bank_cmd", "pitch_cmd")
SENSOR_NAMES = ("x", "y", "z", "heading", "airspeed", "bank", "vario", "vario_te", "lift_asym")


# ---------------------------------------------------------------------------
# 1. The thermal: a single column of rising air.
# ---------------------------------------------------------------------------
@dataclass
class Thermal:
    """A rising-air column, modeled as a 2D Gaussian bump of vertical wind:
    strongest at the center (x0, y0), fading smoothly with horizontal distance.

    Simplest honest model. (NASA Allen 2006 also varies strength with altitude
    and adds a sinking ring at the rim -- that's step-5 world realism.)
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
# 3. The glider: the AIRFRAME. Real params + the polar. All constant in flight.
# ---------------------------------------------------------------------------
@dataclass
class Glider:
    """Everything intrinsic to THIS aircraft -- the stuff you'd change to "plug
    in a different glider." Defaults are ASK-21-trainer-class numbers.

    The polar is quadratic around min-sink:  sink(V) = min_sink + c*(V - Vms)^2
    -- the standard first honest approximation. Wing loading scales it: heavier
    (same wing) -> every speed on the polar shifts up by k = sqrt(mass/mass_ref)
    and sink scales with k too (same glide angles at faster speeds). That is
    what water ballast does to real gliders.
    """

    mass: float = 450.0  # all-up mass (kg) -- glider + pilot
    mass_ref: float = 450.0  # mass the polar numbers below were measured at
    min_sink: float = 0.65  # slowest possible sink (m/s), at v_min_sink, wings level
    v_min_sink: float = 19.0  # airspeed of that slowest sink (m/s)
    polar_curv: float = 0.0028  # how fast sink grows off min-sink speed (1/m/s per (m/s)^2)
    v_stall: float = 16.0  # stall speed, wings level, at mass_ref (m/s)
    v_max: float = 55.0  # never-exceed speed (m/s) -- the command clamp
    roll_rate: float = np.radians(45.0)  # max bank change rate (rad/s)
    accel: float = 2.0  # max airspeed change rate from pitch (m/s^2)
    stall_sink: float = 3.0  # EXTRA sink while stalled (m/s) -- the wing giving up
    max_bank: float = np.radians(60.0)  # command clamp; n=2 there, sink already ~2.8x
    wingspan: float = 17.0  # tip-to-tip (m) -- the lateral baseline of the lift-asym cue

    def loading_scale(self) -> float:
        """k = sqrt(mass / mass_ref): the wing-loading shift. 1.0 at reference."""
        return float(np.sqrt(self.mass / self.mass_ref))

    def sink_rate(self, airspeed: float, bank: float) -> float:
        """Sink through the air (m/s, positive = down) at this speed and bank.

        Polar sink first (scaled to wing loading), then the turn penalty: a
        banked wing must lift n = 1/cos(bank) times the weight; more lift ->
        more induced drag -> sink grows ~ n^1.5. Tighter turns sink faster --
        the cost side of thermalling.
        """
        k = self.loading_scale()
        polar = k * (self.min_sink + self.polar_curv * (airspeed / k - self.v_min_sink) ** 2)
        load_factor = 1.0 / np.cos(bank)
        return float(polar * load_factor**1.5)

    def turn_rate(self, airspeed: float, bank: float) -> float:
        """Heading change rate (rad/s) in a coordinated banked turn:
        g * tan(bank) / V. Slower flight -> faster turn AND smaller circle
        (radius = V^2 / (g*tan(bank))) -- why pilots slow down inside thermals.
        """
        return float(G * np.tan(bank) / airspeed)

    def stall_speed(self, bank: float) -> float:
        """The speed below which the wing can't hold the glider up, at this
        bank. Banking raises it by sqrt(n): the wing must make n times the
        lift, and lift goes as V^2. A 60-degree bank stalls ~41% faster than
        level flight -- slow AND steep is where gliders bite.
        """
        load_factor = 1.0 / np.cos(bank)
        return float(self.v_stall * self.loading_scale() * np.sqrt(load_factor))


# ---------------------------------------------------------------------------
# 4. The glider's dynamic state -- what is true of it RIGHT NOW.
# ---------------------------------------------------------------------------
@dataclass
class GliderState:
    x: float  # east position  (m)
    y: float  # north position (m)
    z: float  # altitude       (m)
    heading: float  # horizontal pointing direction (radians; 0 = east, pi/2 = north)
    airspeed: float  # actual speed through the air (m/s) -- chases pitch_cmd
    bank: float  # actual bank angle (radians) -- chases bank_cmd at roll_rate
    # The two commands are NOT state: they're inputs to step() each tick. The
    # lag between command and these actuals is the vehicle's response dynamics.


# ---------------------------------------------------------------------------
# 5. The simulation: the WORLD. Advances time, simulates the panel, records.
# ---------------------------------------------------------------------------
class Simulation:
    """One virtual world + the glider living in it. Run many independently.

    Holds the omniscient truth (glider, air, current state) and exposes exactly
    two verbs the rest of the project leans on:
        step(bank_cmd, pitch_cmd) -- advance dt seconds, return the new state.
        sense()                   -- the instrument panel, right now.

    `history` accumulates one row per step -- (state, (bank_cmd, pitch_cmd),
    panel) BEFORE the move -- so a finished Simulation IS a logged rollout.
    `crashed` latches True when the glider hits the ground; further steps are
    no-ops (the flight is over).
    """

    def __init__(self, glider: Glider, air: ThermalMap, state: GliderState, dt: float = 0.1):
        self.glider = glider
        self.air = air  # the OMNISCIENT truth -- never handed to a model
        self.state = state
        self.dt = dt
        self.crashed = False
        # instrument memories: varios read the just-elapsed tick (a real needle
        # shows the recent past, not the future). Zero = parked, pre-launch.
        self._vario = 0.0
        self._vario_te = 0.0
        # rows: (state_t, (bank_cmd_t, pitch_cmd_t), panel_t)
        self.history: list[tuple[GliderState, tuple[float, float], dict[str, float]]] = []

    def sense(self) -> dict[str, float]:
        """The instrument panel: everything this aircraft can LEGITIMATELY know.

        Self is fully observable (GPS, altimeter, compass, airspeed indicator,
        attitude indicator) -- a real glider always knows its own kinematics.
        The WORLD reaches the panel only as felt lift:
          vario    -- own climb rate over the last tick (raw needle). Beware:
                      it shows your own zooms as "lift" (pull up -> it rises).
          vario_te -- total-energy-compensated vario: subtracts the speed<->
                      height exchange, leaving only (air - polar sink). This is
                      what real pilots center thermals with, and it is exactly
                      the ENERGY RATE instrument: vario_te * g == d(E/m)/dt.
          lift_asym -- THE BIRD CUE: lift at the left wingtip minus lift at the
                      right wingtip. Positive = left wing in stronger lift =
                      the thermal is to the left. This is the rolling-moment
                      cue birds center thermals with (a pilot feels it as "a
                      wing lifts"); the torque cue Reddy et al. (PNAS 2016 /
                      Nature 2018) found essential for learned soaring. It is
                      an instantaneous LATERAL lift gradient -- something the
                      vario trail can never give on a straight path. Felt at
                      aircraft-attached points, so it is firewall-legal; the
                      point-mass dynamics are unchanged (sensed, not a torque).
        This method is the ONLY window a model gets (plus its own actions and
        Glider params). The thermal's true parameters are not here -- ever.
        """
        s = self.state
        # wingtip positions: heading is (cos, sin), so "left" is 90 deg CCW =
        # (-sin, cos). Banking tilts the span out of the horizontal plane,
        # shrinking its horizontal footprint by cos(bank) (knife-edge -> zero
        # lateral baseline -> no cue).
        half = 0.5 * self.glider.wingspan * float(np.cos(s.bank))
        left = (s.x - half * np.sin(s.heading), s.y + half * np.cos(s.heading))
        right = (s.x + half * np.sin(s.heading), s.y - half * np.cos(s.heading))
        return {
            "x": s.x,
            "y": s.y,
            "z": s.z,
            "heading": s.heading,
            "airspeed": s.airspeed,
            "bank": s.bank,
            "vario": self._vario,
            "vario_te": self._vario_te,
            "lift_asym": float(self.air.updraft(*left)) - float(self.air.updraft(*right)),
        }

    def step(self, bank_cmd: float, pitch_cmd: float) -> GliderState:
        """Advance the world one time-step (forward Euler) and return new state.

        THE ACTIONS (what a pilot's stick does, per tick):
          bank_cmd  -- target bank angle (radians). Wings roll toward it at
                       roll_rate. ~0.5 rad (~30 deg) is a normal thermal turn.
          pitch_cmd -- target airspeed (m/s): the stick's fore/aft axis, i.e.
                       the trimmed-speed set-point. Airspeed relaxes toward it
                       at `accel`, PAYING for every change with altitude
                       through the exact energy exchange (dive to speed up,
                       zoom to slow down). Command below stall speed and you
                       will stall -- allowed on purpose; that's how real
                       stalls happen.
        Commands are recorded RAW in history (what was asked); clamps apply to
        what the airframe DOES (the flight envelope).
        """
        if self.crashed:
            return self.state  # flight's over -- the world stops responding

        g, s, dt = self.glider, self.state, self.dt
        wx, wy = self.air.wind

        # record (state, action, panel) BEFORE moving, so rows align at time t
        # and self.state always sits one step ahead of the last history row.
        self.history.append((s, (bank_cmd, pitch_cmd), self.sense()))

        # 1) BANK chases its command at the roll-rate limit (never teleports).
        want = np.clip(bank_cmd, -g.max_bank, g.max_bank)
        bank = s.bank + float(np.clip(want - s.bank, -g.roll_rate * dt, g.roll_rate * dt))

        # 2) AIRSPEED chases its command at the pitch-authority limit -- unless
        #    stalled, in which case the nose drops and speed rebuilds no matter
        #    what the stick asks (the wing has quit; physics is flying now).
        stalled = s.airspeed < g.stall_speed(bank)
        if stalled:
            airspeed = s.airspeed + g.accel * dt
        else:
            want_v = float(np.clip(pitch_cmd, 0.8 * g.stall_speed(0.0), g.v_max))
            airspeed = s.airspeed + float(np.clip(want_v - s.airspeed, -g.accel * dt, g.accel * dt))

        # 3) THE ENERGY EXCHANGE, exact: whatever kinetic energy just changed
        #    is paid for (or refunded) in altitude. 1/2*(V'^2 - V^2) = -g*dz.
        dz_exchange = -(airspeed**2 - s.airspeed**2) / (2.0 * G)

        # 4) VERTICAL: air lifts the whole aircraft (attitude-independent);
        #    the polar drains; a stalled wing drains extra. This line plus the
        #    exchange is the entire energy game.
        sink = g.sink_rate(s.airspeed, bank) + (g.stall_sink if stalled else 0.0)
        climb = float(self.air.updraft(s.x, s.y)) - sink
        z = s.z + climb * dt + dz_exchange

        # 5) TURN + TRANSLATE: coordinated turn at the CURRENT speed; slide in
        #    the new heading; the wind carries the whole glider along.
        heading = s.heading + g.turn_rate(s.airspeed, bank) * dt
        x = s.x + (s.airspeed * np.cos(heading) + wx) * dt
        y = s.y + (s.airspeed * np.sin(heading) + wy) * dt

        # 6) INSTRUMENTS remember this tick (they display the just-elapsed
        #    interval). vario = what the needle shows (includes your own zoom);
        #    vario_te subtracts the exchange EXACTLY -> pure (air - sink), and
        #    equivalently the total-energy rate: d(E/m)/dt / g.
        self._vario = (z - s.z) / dt
        self._vario_te = (
            self._vario + (airspeed + s.airspeed) / 2.0 * ((airspeed - s.airspeed) / dt) / G
        )

        # 7) THE GROUND. Hit it and the flight is over -- energy was survival.
        if z <= 0.0:
            z = 0.0
            self.crashed = True

        self.state = GliderState(x=x, y=y, z=z, heading=heading, airspeed=airspeed, bank=bank)
        return self.state
