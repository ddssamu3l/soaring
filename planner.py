"""
planner.py -- choose actions by IMAGINING futures through the learned model (t3).

This is LeCun's "Mode-2" loop made concrete, the smallest honest version of
V-JEPA-2-AC's planner: nothing here trains. The world model is frozen; planning
is pure SEARCH through its imagination, done fresh at every replan:

  sense the panel -> sample candidate action sequences -> roll each through
  train.predict_delta (the exact call the keystone certified) -> rank the
  imagined futures -> fly the best one's first second -> reality corrects ->
  repeat.

The search is the Cross-Entropy Method (CEM). Each candidate is a short vector
of PIECEWISE-CONSTANT bank commands -- per-tick white noise would be low-pass
filtered by the airframe's roll-rate lag into "fly straight" (every candidate
alike, nothing to rank). Elites are never re-rolled: their statistics refit the
sampling Gaussian, and a fresh population is drawn from the sharpened
distribution -- the dice learn where the good actions live.

Two numbers are anchored, not tuned:
  * candidates hold bank commands inside +/-MAX_BANK_CMD, the exact support of
    the training data -- a planner that samples outside it would be optimizing
    into regions where the model has never been graded (hallucination-chasing);
  * the horizon is 15 s because that is precisely how deep the keystone plot
    certified the imagination (position error ~15 m at 15 s; beyond is faith).

THE COST CONTAINS NO SOARING ADVICE -- that is the experiment. Rollouts are
ranked lexicographically by (crashed, glide-deficit, estimated total time):
first don't die, then be somewhere the goal is makeable, then be fast. The
deficit prices altitude AND airspeed through the glider's own polar (energy
height x glide ratio -- arithmetic any pilot knows about their own aircraft),
never through knowledge of the field. If circling in lift emerges, it emerged
because imagined futures that climb first rank closer to the goal -- not
because we paid for climbing (an energy bonus would bribe the exact behavior
t4 exists to measure; a path-length penalty would bribe against it).

Run:  .venv/bin/python planner.py   (needs data/model_full.pt from train.py)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from data_gen import MAX_BANK_CMD, make_world
from glider_sim import G, Glider, GliderState, Simulation
from train import Checkpoint, FloatArr, IntArr, load_checkpoint, predict_delta

BoolArr = npt.NDArray[np.bool_]

NEVER = 10**9  # sentinel index for "this never happened within the horizon"


# --- what the planner may legitimately know ------------------------------------------
@dataclass(frozen=True)
class GlidePolar:
    """Own-body arithmetic: the glider's best-glide operating point, derived
    numerically from its OWN polar (glider_sim.Glider.sink_rate). A real pilot
    knows these two numbers cold; no world knowledge is involved."""

    v_best_glide: float  # airspeed of flattest glide (m/s)
    glide_ratio: float  # meters forward per meter down at that speed (L/D)


def best_glide(glider: Glider) -> GlidePolar:
    """Scan the wings-level polar for the speed that maximizes V / sink."""
    v = np.linspace(glider.stall_speed(0.0), glider.v_max, 400)
    ratio = np.array([vi / glider.sink_rate(float(vi), 0.0) for vi in v])
    i = int(np.argmax(ratio))
    return GlidePolar(v_best_glide=float(v[i]), glide_ratio=float(ratio[i]))


@dataclass(frozen=True)
class Goal:
    """The task IS its goal: a circle on the map. Being told where to go is the
    problem statement, not a leak (contrast: being told where the LIFT is)."""

    x: float
    y: float
    radius: float = 30.0


@dataclass(frozen=True)
class PlannerConfig:
    """CEM/MPC dials. Anchored: max_bank_cmd (training support) and the total
    horizon n_segments * ticks_per_segment = 150 ticks = the keystone's 15 s.
    Everything else is a starting guess, tuned by watching it fly."""

    n_segments: int = 5  # decision variables per candidate
    ticks_per_segment: int = 30  # 3 s each -- roughly a 50-60 degree arc of turn
    population: int = 512  # candidates per CEM iteration
    n_elites: int = 64  # top-scoring candidates that refit the Gaussian
    iterations: int = 4  # sample -> rank -> refit rounds per replan
    init_std: float = np.radians(30.0)  # first-round spread: cover the bank range
    std_floor: float = np.radians(5.0)  # never let exploration collapse to zero
    replan_ticks: int = 10  # execute 1 s of the plan, then re-sense and replan
    max_bank_cmd: float = MAX_BANK_CMD  # the training-support clip (+/-50 deg)

    @property
    def horizon(self) -> int:
        return self.n_segments * self.ticks_per_segment


# --- imagination: batched free-running rollouts through the frozen model -------------
def expand_segments(segments: FloatArr, ticks_per_segment: int) -> FloatArr:
    """(N, K) per-segment bank commands -> (N, K*ticks) per-tick commands."""
    return np.repeat(segments, ticks_per_segment, axis=1)


def imagine(ck: Checkpoint, panel0: FloatArr, bank_ticks: FloatArr, pitch_cmd: float) -> FloatArr:
    """Roll N candidate command schedules through the model as ONE batch.

    (9,) sensed start panel + (N, H) per-tick bank commands -> (N, H+1, 9)
    imagined panels: row 0 is the shared true start, every later row is built
    from the model's own previous output -- keystone.free_run's exact feedback
    loop, with the planner's candidate actions instead of logged ones.
    """
    n, horizon = bank_ticks.shape
    bank_col = ck.action_names.index("bank_cmd")
    pitch_col = ck.action_names.index("pitch_cmd")
    out = np.empty((n, horizon + 1, len(ck.sensor_names)))
    panel = np.tile(panel0, (n, 1))
    out[:, 0] = panel
    action = np.empty((n, len(ck.action_names)))
    action[:, pitch_col] = pitch_cmd
    for h in range(1, horizon + 1):
        action[:, bank_col] = bank_ticks[:, h - 1]
        panel = panel + predict_delta(ck, panel, action)  # the imagination step
        out[:, h] = panel
    return out


# --- the cost: task arithmetic only, ranked lexicographically -------------------------
@dataclass(frozen=True)
class RolloutScores:
    """Per-candidate verdicts, compared in strict order: first don't crash,
    then minimize the glide-deficit (is the goal makeable from where this
    future ends?), then minimize estimated total time (be fast)."""

    crashed: BoolArr  # imagined ground contact before reaching the goal
    deficit: FloatArr  # m of glide range still missing (0 = goal in glide/reached)
    est_time: FloatArr  # s elapsed + still-air time-to-go estimate


def score_rollouts(
    imagined: FloatArr,
    sensor_names: tuple[str, ...],
    goal: Goal,
    polar: GlidePolar,
    dt: float,
) -> RolloutScores:
    """Grade (N, H+1, 9) imagined futures against the task -- nothing else.

    Time-to-go is deliberately coarse (straight still-air glide at best-glide
    speed): the imagined horizon carries the real dynamics, the tail is
    arithmetic, and every replan converts another second of tail into horizon.
    """
    xc = sensor_names.index("x")
    yc = sensor_names.index("y")
    zc = sensor_names.index("z")
    vc = sensor_names.index("airspeed")
    horizon = imagined.shape[1] - 1

    dist = np.hypot(imagined[:, :, xc] - goal.x, imagined[:, :, yc] - goal.y)  # (N, H+1)
    arrive_mask = dist <= goal.radius
    ground_mask = imagined[:, :, zc] <= 0.0
    t_arrive = np.where(arrive_mask.any(axis=1), arrive_mask.argmax(axis=1), NEVER)
    t_ground = np.where(ground_mask.any(axis=1), ground_mask.argmax(axis=1), NEVER)
    arrived = t_arrive <= t_ground  # touching ground AFTER arriving doesn't count
    crashed = (~arrived) & (t_ground < NEVER)

    # end-state arithmetic for the futures still in the air and short of the goal:
    # kinetic energy above best-glide speed is convertible altitude (the sim's own
    # speed<->height exchange makes it fungible), so reach = energy height * L/D.
    remaining = dist[:, horizon]
    z_end = imagined[:, horizon, zc]
    v_end = imagined[:, horizon, vc]
    energy_height = np.maximum(z_end + (v_end**2 - polar.v_best_glide**2) / (2.0 * G), 0.0)
    deficit = np.maximum(remaining - energy_height * polar.glide_ratio, 0.0)
    est_time = horizon * dt + remaining / polar.v_best_glide

    # arrived futures: nothing missing, their time is the actual arrival time
    deficit = np.where(arrived & (t_arrive < NEVER), 0.0, deficit)
    est_time = np.where(arrived & (t_arrive < NEVER), np.minimum(t_arrive, horizon) * dt, est_time)
    return RolloutScores(crashed=crashed, deficit=deficit, est_time=est_time)


def rank(scores: RolloutScores) -> IntArr:
    """Candidate indices best-to-worst. np.lexsort sorts by the LAST key first,
    so the order is: crashed (False beats True), then deficit, then time. CEM
    only ever needs this total order -- no unit-stitching scalar required."""
    out: IntArr = np.lexsort((scores.est_time, scores.deficit, scores.crashed))
    return out


# --- the CEM search --------------------------------------------------------------------
@dataclass(frozen=True)
class CEMIteration:
    """One CEM round, recorded for visualization/analysis: the sampled
    population, its best-to-worst ordering, the refit mean, and every
    candidate's imagined ground track. Only materialized when a trace list is
    passed to cem_plan -- planning itself never pays for it."""

    candidates: FloatArr  # (population, K) as actually rolled (post-clip)
    order: IntArr  # rank(scores); order[:n_elites] are the elites
    mean: FloatArr  # (K,) the Gaussian mean refit AFTER this round
    imagined_xy: FloatArr  # (population, H+1, 2) where each candidate believed it would fly


@dataclass(frozen=True)
class Plan:
    """One replan's result: the winning candidate, the refit sampling mean
    (next replan's warm start), and the winner's imagined future (the ghost --
    what the planner BELIEVED would happen, kept for honest post-mortems)."""

    segments: FloatArr  # (K,) winning per-segment bank commands
    mean: FloatArr  # (K,) final refit Gaussian mean
    imagined: FloatArr  # (H+1, 9) the winner's imagined panels


def cem_plan(
    ck: Checkpoint,
    panel0: FloatArr,
    goal: Goal,
    polar: GlidePolar,
    cfg: PlannerConfig,
    rng: np.random.Generator,
    pitch_cmd: float,
    warm_mean: FloatArr | None = None,
    trace: list[CEMIteration] | None = None,
) -> Plan:
    """One full CEM search from the current sensed panel.

    Iterate: sample a population around (mean, std), clip to the training
    support, imagine all candidates as one batch, rank, refit mean/std to the
    elites. The incumbent mean is always injected as candidate 0 so a good
    warm-started plan can never be lost to sampling luck. std restarts wide
    every replan -- exploration must survive the warm start.
    """
    mean = np.zeros(cfg.n_segments) if warm_mean is None else warm_mean.copy()
    std = np.full(cfg.n_segments, cfg.init_std)
    best = Plan(segments=mean, mean=mean, imagined=np.empty(0))  # overwritten below
    for _ in range(cfg.iterations):
        cands = rng.normal(mean, std, size=(cfg.population, cfg.n_segments))
        cands[0] = mean  # keep the incumbent in the running
        cands = np.clip(cands, -cfg.max_bank_cmd, cfg.max_bank_cmd)
        imagined = imagine(ck, panel0, expand_segments(cands, cfg.ticks_per_segment), pitch_cmd)
        order = rank(score_rollouts(imagined, ck.sensor_names, goal, polar, ck.dt))
        elites = cands[order[: cfg.n_elites]]
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), cfg.std_floor)
        best = Plan(segments=cands[order[0]], mean=mean, imagined=imagined[order[0]])
        if trace is not None:
            xc, yc = ck.sensor_names.index("x"), ck.sensor_names.index("y")
            trace.append(
                CEMIteration(
                    candidates=cands,
                    order=order,
                    mean=mean.copy(),
                    imagined_xy=imagined[:, :, [xc, yc]],
                )
            )
    return best


# --- the MPC executor: imagination proposes, reality disposes ---------------------------
@dataclass
class Flight:
    """One attempted flight to a goal. sim.history holds the full trajectory;
    plans holds what the planner believed at each replan (ghost material)."""

    outcome: Literal["arrived", "crashed", "timeout"]
    seconds: float
    goal: Goal
    plans: list[Plan] = field(default_factory=list)


def fly_to_goal(
    sim: Simulation,
    ck: Checkpoint,
    goal: Goal,
    cfg: PlannerConfig,
    rng: np.random.Generator,
    max_seconds: float = 120.0,
) -> Flight:
    """The receding-horizon loop: sense -> CEM -> fly the first replan_ticks of
    the winner -> repeat. The planner sees ONLY sense() panels and its own
    imagination; the Simulation holds the omniscient truth and grades arrival.

    Warm start: the refit mean is reused unshifted -- 1 s into a 3 s segment it
    is slightly stale, but the next CEM re-optimizes from it anyway (logged
    approximation; revisit only if the planner visibly dithers at boundaries).
    """
    polar = best_glide(sim.glider)
    pitch_cmd = polar.v_best_glide  # pitch pinned to trim: bank is the only decision
    flight = Flight(outcome="timeout", seconds=0.0, goal=goal)
    warm: FloatArr | None = None
    ticks = 0
    max_ticks = int(max_seconds / sim.dt)
    while ticks < max_ticks:
        panel_now = sim.sense()
        panel0 = np.array([panel_now[name] for name in ck.sensor_names])
        plan = cem_plan(ck, panel0, goal, polar, cfg, rng, pitch_cmd, warm_mean=warm)
        flight.plans.append(plan)
        warm = plan.mean
        bank_ticks = expand_segments(plan.segments[None, :], cfg.ticks_per_segment)[0]
        for tick in range(cfg.replan_ticks):
            state = sim.step(float(bank_ticks[tick]), pitch_cmd)
            ticks += 1
            if np.hypot(state.x - goal.x, state.y - goal.y) <= goal.radius:
                flight.outcome = "arrived"
                flight.seconds = ticks * sim.dt
                return flight
            if sim.crashed:
                flight.outcome = "crashed"
                flight.seconds = ticks * sim.dt
                return flight
    flight.seconds = max_ticks * sim.dt
    return flight


# --- the demo: mechanics on the one-thermal world ---------------------------------------
def main() -> None:
    """Fly the CURRENT fixed world (one thermal at the origin) to a goal on the
    far side of it. With a healthy starting altitude this is navigation, not
    survival -- it demonstrates the sense->imagine->rank->act LOOP works; the
    decision-forcing task (climb or fail) arrives with the two-thermal world."""
    here = Path(__file__).resolve().parent
    ck = load_checkpoint(here / "data" / "model_full.pt")
    glider, air = make_world()
    polar = best_glide(glider)
    print(f"own polar: best glide {polar.v_best_glide:.1f} m/s at L/D {polar.glide_ratio:.1f}")

    start = GliderState(x=-150.0, y=-150.0, z=350.0, heading=0.0, airspeed=24.0, bank=0.0)
    goal = Goal(x=250.0, y=250.0)  # past the thermal, ~570 m out
    sim = Simulation(glider, air, start)
    cfg = PlannerConfig()
    print(f"flying to ({goal.x:g}, {goal.y:g}) r={goal.radius:g} m from ({start.x:g}, {start.y:g})")

    flight = fly_to_goal(sim, ck, goal, cfg, rng=np.random.default_rng(0))

    s = sim.state
    straight = float(np.hypot(goal.x - start.x, goal.y - start.y))
    floor_time = straight / polar.v_best_glide
    print(f"\noutcome: {flight.outcome.upper()} in {flight.seconds:.1f} s")
    print(f"  straight-line {straight:.0f} m; best-glide straight-run ~{floor_time:.1f} s")
    print(
        f"  final state: z={s.z:.0f} m, airspeed={s.airspeed:.1f} m/s, {len(flight.plans)} replans"
    )


if __name__ == "__main__":
    main()
