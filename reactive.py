"""
reactive.py -- the REACTOR: t4's baseline. Reflexes only, no imagination.

This is the other half of the organizing question. planner.py answers "what can
you do if you can predict?"; this file answers "what can you do if you can only
REACT?" -- map the CURRENT instrument panel straight to a stick command, with
no model, no rollouts, no search. Same sensors (sense()'s nine channels), same
goal knowledge, same own-body arithmetic; the ONE removed ability is imagining
futures. Whatever gap experiment.py measures between the two is therefore the
measured value of prediction itself -- LeCun-vs-Malik as a number.

The reactor is a ladder of three variants, weakest to strongest, so the
experiment shows WHERE reaction fails, not just that it fails:

  b0 -- blind glide: steer at the goal at best-glide speed, never circle.
        The floor. (The 9-route certification says this dies on wall tasks.)
  b1 -- pure reflex: circle when the vario says lift, leave when it says the
        lift died. No arithmetic: b1 has no concept of "enough altitude", so
        it either parks in a thermal it never needed or leaves on a whim.
  b2 -- reflex + pilot arithmetic (the fair opponent): b1's reflexes gated by
        the SAME final-glide deficit the planner's scorer uses -- identical
        formula, identical reserve and margin constants (pinned equal by
        test_reactive). It climbs while the goal is out of margined glide
        range and commits the moment the glide is made. Withholding from the
        reactor a calculator the planner got would rig the experiment.

The climb reflexes are the classic ones -- exactly what birds and pilots do at
the reflex level, no memory beyond the mode latch:
  * enter a circle when total-energy climb beats a threshold, turning toward
    the wing the lift is under (lift_asym > 0 = thermal to the left = bank
    left; positive bank turns CCW/left in this sim's convention);
  * center by the wingtip cue: core inside the circle -> tighten, core
    outside -> flatten (Reddy et al.'s torque cue, used the way birds use it);
  * give up when the climb has been weak for a sustained patience window.

What the reactor legitimately CANNOT do -- and this is the point -- is weigh a
commitment: crossing the sink band costs altitude that only a model of the
NEXT thirty seconds can price. The reactor's deficit arithmetic assumes still
air; it leaves a thermal exactly at "still-air safe + margin" no matter what
lies on the route. If that margin is what the wall eats, only the agent that
imagined the wall keeps the reserve.

Run via experiment.py (this module only defines the pilot and its flight loop).
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from data_gen import MAX_BANK_CMD
from glider_sim import G, Simulation
from planner import Flight, GlidePolar, Goal

Mode = Literal["glide", "climb"]
Variant = Literal["b0", "b1", "b2"]


@dataclass(frozen=True)
class ReactiveConfig:
    """The reactor's dials. The honesty rule of t4: these get TUNED (grid
    search, experiment.py --tune) as hard as the planner's dials were tuned by
    watching it fly -- a strawman baseline would make the headline number
    fake. reserve_height and glide_margin are NOT tuned: they are pinned to
    the planner's values so both agents run the same arithmetic."""

    variant: Variant = "b2"
    vario_enter: float = 0.5  # m/s of TE climb that triggers circling
    vario_exit: float = 0.0  # sustained TE climb below this counts as "weak"
    exit_patience: float = 5.0  # s of weak climb before giving up the thermal
    bank_circle: float = np.radians(38.0)  # base thermalling bank
    k_asym: float = 1.0  # centering gain: rad of bank per m/s of wingtip asymmetry
    bank_flat: float = np.radians(20.0)  # centering clip: never flatter than this...
    bank_steep: float = np.radians(42.0)  # ...never steeper (stall margin at v_climb)
    v_climb: float = 19.5  # circling speed: tight radius, stall-safe at bank_steep
    v_dash: float | None = None  # speed-to-fly: command this in strong sink (None = off;
    #   the tuning grid decides whether dashing through sink earns its keep)
    sink_dash: float = -1.5  # m/s of TE sink that counts as "strong" for v_dash
    heading_gain: float = 4.0  # glide steering: rad of bank per rad of heading error
    reserve_height: float = 50.0  # PINNED = PlannerConfig.reserve_height (fairness)
    glide_margin: float = 0.6  # PINNED = PlannerConfig.glide_margin (fairness)
    max_bank_cmd: float = MAX_BANK_CMD  # same envelope the planner samples within


def _wrap(angle: float) -> float:
    """Fold any angle into (-pi, pi] -- the shortest-way-around heading error."""
    return float(math.remainder(angle, 2.0 * math.pi))


class ReactivePilot:
    """The reflex policy: panel in, (bank_cmd, pitch_cmd) out, once per tick.

    Internal state is reflex-level only -- the current mode, the circle
    direction chosen at entry, and how long the climb has been weak. Nothing
    here looks ahead; nothing remembers the field."""

    def __init__(self, cfg: ReactiveConfig, polar: GlidePolar, goal: Goal, dt: float):
        self.cfg = cfg
        self.polar = polar
        self.goal = goal
        self.dt = dt
        self.mode: Mode = "glide"
        self.turn_sign = 1.0  # +1 = circle CCW/left (positive bank), -1 = CW/right
        self._weak_for = 0.0  # s of climb below vario_exit while circling

    def deficit(self, panel: dict[str, float]) -> float:
        """Meters of glide range still missing -- the planner scorer's exact
        final-glide arithmetic in scalar form (test_reactive pins the two
        equal on the same inputs). 0 = the goal is makeable from here in
        still air, on a margined polar, keeping the reserve."""
        dist = math.hypot(panel["x"] - self.goal.x, panel["y"] - self.goal.y)
        v2 = panel["airspeed"] ** 2 - self.polar.v_best_glide**2
        energy_height = max(panel["z"] + v2 / (2.0 * G), 0.0)
        usable = max(energy_height - self.cfg.reserve_height, 0.0)
        return max(dist - usable * self.polar.glide_ratio * self.cfg.glide_margin, 0.0)

    def act(self, panel: dict[str, float]) -> tuple[float, float]:
        """One reflex tick: update the mode latch, then act by mode."""
        self._transition(panel)
        if self.mode == "climb":
            return self._climb_cmd(panel)
        return self._glide_cmd(panel)

    def _transition(self, panel: dict[str, float]) -> None:
        cfg = self.cfg
        if cfg.variant == "b0":
            return  # blind glide never circles
        te = panel["vario_te"]
        if self.mode == "glide":
            worth_stopping = te >= cfg.vario_enter
            if cfg.variant == "b2":
                # arithmetic gate: once the goal is in margined glide range,
                # lift is a distraction -- press on and cash the final glide.
                worth_stopping = worth_stopping and self.deficit(panel) > 0.0
            if worth_stopping:
                self.mode = "climb"
                self.turn_sign = 1.0 if panel["lift_asym"] >= 0.0 else -1.0
                self._weak_for = 0.0
        else:
            if cfg.variant == "b2" and self.deficit(panel) <= 0.0:
                self.mode = "glide"  # glide made: commit to the goal
                return
            # eviction reflex: a thermal that has stopped paying gets a
            # patience window, then the pilot moves on (without this, b1
            # parks forever and b2 parks under a thermal too weak to ever
            # close its deficit).
            self._weak_for = 0.0 if te >= cfg.vario_exit else self._weak_for + self.dt
            if self._weak_for >= cfg.exit_patience:
                self.mode = "glide"

    def _glide_cmd(self, panel: dict[str, float]) -> tuple[float, float]:
        """Steer the nose at the goal; fly best-glide (or dash through sink)."""
        cfg = self.cfg
        bearing = math.atan2(self.goal.y - panel["y"], self.goal.x - panel["x"])
        err = _wrap(bearing - panel["heading"])
        # positive heading error (goal CCW of the nose) -> positive bank
        # (turn_rate = g*tan(bank)/V: positive bank raises heading)
        bank = float(np.clip(cfg.heading_gain * err, -cfg.max_bank_cmd, cfg.max_bank_cmd))
        speed = self.polar.v_best_glide
        if cfg.v_dash is not None and panel["vario_te"] <= cfg.sink_dash:
            speed = cfg.v_dash
        return bank, speed

    def _climb_cmd(self, panel: dict[str, float]) -> tuple[float, float]:
        """Hold the circle; center on the wingtip cue. With turn_sign s, the
        circle's center is on the side s points to, so s*lift_asym > 0 means
        the core is INSIDE the circle -> tighten toward it; < 0 means it is
        outside -> flatten and drift out."""
        cfg = self.cfg
        inside = self.turn_sign * panel["lift_asym"]
        mag = float(np.clip(cfg.bank_circle + cfg.k_asym * inside, cfg.bank_flat, cfg.bank_steep))
        return self.turn_sign * mag, cfg.v_climb


def fly_reactive(
    sim: Simulation, pilot: ReactivePilot, goal: Goal, max_seconds: float = 240.0
) -> Flight:
    """The reactor's whole control loop: sense -> reflex -> step, every tick.

    Mirrors planner.fly_to_goal's grading exactly (arrival checked before
    crash, same Flight fates) so experiment.py compares like with like --
    the only difference is what happens between sense and step."""
    ticks = 0
    max_ticks = int(max_seconds / sim.dt)
    while ticks < max_ticks:
        bank_cmd, pitch_cmd = pilot.act(sim.sense())
        state = sim.step(bank_cmd, pitch_cmd)
        ticks += 1
        if math.hypot(state.x - goal.x, state.y - goal.y) <= goal.radius:
            return Flight(outcome="arrived", seconds=ticks * sim.dt, goal=goal)
        if sim.crashed:
            return Flight(outcome="crashed", seconds=ticks * sim.dt, goal=goal)
    return Flight(outcome="timeout", seconds=max_ticks * sim.dt, goal=goal)
