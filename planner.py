"""
planner.py -- choose actions by IMAGINING futures through the learned model (t3).

This is LeCun's "Mode-2" loop made concrete, the smallest honest version of
V-JEPA-2-AC's planner: nothing here trains. The world model is frozen; planning
is pure SEARCH through its imagination, done fresh at every replan:

  sense the panel -> sample candidate action sequences -> roll each through
  train.predict_delta (the exact call the keystone certified) -> rank the
  imagined futures -> fly the best one's first second -> reality corrects ->
  repeat.

The search is the Cross-Entropy Method (CEM). Each candidate is a short block
sequence of PIECEWISE-CONSTANT (bank, speed) commands -- both stick axes.
Per-tick white noise would be low-pass filtered by the airframe's command lag
into "fly straight" (every candidate alike, nothing to rank), and speed must
be plannable because climbing requires it: at fixed best-glide speed the turn
radius is wider than a thermal core -- slowing down inside lift is how circles
fit inside thermals. Elites are never re-rolled: their statistics refit the
sampling Gaussian, and a fresh population is drawn from the sharpened
distribution -- the dice learn where the good actions live.

Two numbers are anchored, not tuned:
  * candidates hold bank commands inside +/-MAX_BANK_CMD, the exact support of
    the training data -- a planner that samples outside it would be optimizing
    into regions where the model has never been graded (hallucination-chasing);
  * the horizon is 30 s -- the measured edge of trustworthy imagination (the
    keystone probe: sigma 0.87 at 30 s, ~1.0 by 45 s = persistence-useless).
    15 s was fully certified but cannot CONTAIN THE COMMITMENT: the sink-band
    crossing takes ~50 s, and a horizon that can't see a commitment's cost
    cannot weigh it -- 30 s sees most of the band; a margined tail covers the
    rest. Planning deeper than ~30 s would be planning through mush.

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
from train import Checkpoint, FloatArr, IntArr, clamp_panel, load_checkpoint, predict_delta

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

    n_segments: int = 6  # decision blocks per candidate (each block = bank + speed)
    ticks_per_segment: int = 50  # 5 s each: one block sustains half a thermalling circle
    population: int = 512  # candidates per CEM iteration
    n_elites: int = 64  # top-scoring candidates that refit the Gaussian
    iterations: int = 4  # sample -> rank -> refit rounds per replan
    init_std: float = np.radians(30.0)  # first-round bank spread: cover the range
    std_floor: float = np.radians(5.0)  # never let bank exploration collapse
    pitch_init_std: float = 4.0  # first-round speed-command spread (m/s)
    pitch_std_floor: float = 0.5  # never let speed exploration collapse (m/s)
    replan_ticks: int = 10  # execute 1 s of the plan, then re-sense and replan
    max_bank_cmd: float = MAX_BANK_CMD  # the training-support clip (+/-50 deg)
    pitch_lo: float = 17.0  # speed-command clip: inside training support (15-35),
    pitch_hi: float = 32.0  # with a margin off the sampled extremes
    reserve_height: float = 50.0  # the pilot's arrival reserve (m): the tail only
    #   counts energy height ABOVE this -- plan to arrive with altitude in hand,
    #   not at ground level. Own-body arithmetic, zero field knowledge.
    glide_margin: float = 0.6  # the pilot's pessimism dial: the terminal value uses
    #   glide_ratio * margin, so "goal is makeable" needs real reserve. Without it
    #   the planner flies zero-margin final glides that sit exactly at the deficit-0
    #   boundary, where +/-10 m of imagination noise flips the verdict (it did: it
    #   raced at the band at 32 m/s on dreams that ended precisely at 'just enough').
    #   Own-body arithmetic (a MacCready-style setting), zero field knowledge.

    @property
    def horizon(self) -> int:
        return self.n_segments * self.ticks_per_segment


# --- imagination: batched free-running rollouts through the frozen model -------------
def expand_segments(segments: FloatArr, ticks_per_segment: int) -> FloatArr:
    """(N, K, ...) per-segment commands -> (N, K*ticks, ...): each block held."""
    return np.repeat(segments, ticks_per_segment, axis=1)


def imagine(
    ck: Checkpoint, panel0: FloatArr, bank_ticks: FloatArr, pitch_ticks: FloatArr
) -> FloatArr:
    """Roll N candidate command schedules through the model as ONE batch.

    (9,) sensed start panel + (N, H) per-tick bank and speed commands ->
    (N, H+1, 9) imagined panels: row 0 is the shared true start, every later
    row is built from the model's own previous output -- keystone.free_run's
    exact feedback loop, with the planner's candidate actions instead of
    logged ones.
    """
    n, horizon = bank_ticks.shape
    bank_col = ck.action_names.index("bank_cmd")
    pitch_col = ck.action_names.index("pitch_cmd")
    out = np.empty((n, horizon + 1, len(ck.sensor_names)))
    panel = np.tile(panel0, (n, 1))
    out[:, 0] = panel
    action = np.empty((n, len(ck.action_names)))
    for h in range(1, horizon + 1):
        action[:, bank_col] = bank_ticks[:, h - 1]
        action[:, pitch_col] = pitch_ticks[:, h - 1]
        # the imagination step, pinned to the training range (clamp_panel).
        # Load-bearing here: CEM would otherwise actively SEEK divergent
        # dreams -- an imagined z running away to +1e9 scores deficit 0.
        panel = clamp_panel(panel + predict_delta(ck, panel, action), ck.stats)
        out[:, h] = panel
    return out


# --- the cost: task arithmetic only, ranked lexicographically -------------------------
@dataclass(frozen=True)
class RolloutScores:
    """Per-candidate verdicts. Alive futures compare (deficit, then time).
    Crashed futures rank below ALL alive ones and compare by SURVIVAL TIME
    (die later beats die closer): when every imagined future ends in the
    ground, racing to minimize distance-at-death is a kamikaze dive -- the
    right move is to keep flying the airplane, because the next replan is
    anchored on reality, which may be kinder than a pessimistic dream."""

    crashed: BoolArr  # imagined ground contact before reaching the goal
    deficit: FloatArr  # m of glide range still missing (0 = goal in glide/reached)
    est_time: FloatArr  # s elapsed + still-air time-to-go estimate
    t_crash: FloatArr  # s until imagined ground contact (inf if it stays flying)


def score_rollouts(
    imagined: FloatArr,
    sensor_names: tuple[str, ...],
    goal: Goal,
    polar: GlidePolar,
    dt: float,
    ground_z: float = 0.0,
    reserve_height: float = 0.0,
) -> RolloutScores:
    """Grade (N, H+1, 9) imagined futures against the task -- nothing else.

    `ground_z` is the death line, and it must be the IMAGINATION'S floor, not
    0: clamp_panel pins imagined z at the training minimum (~0.3 m -- crashed
    episodes end just above impact, so z=0 exactly never occurs in data), and
    with a 0 threshold no dream could ever die. The planner found and
    exploited exactly that: immortal dreams skimming the pinned floor at full
    speed, "arriving" on kinetic energy alone. A dream pinned to the data's
    altitude floor IS a dream that hit the ground.

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
    ground_mask = imagined[:, :, zc] <= ground_z + 1e-9
    t_arrive = np.where(arrive_mask.any(axis=1), arrive_mask.argmax(axis=1), NEVER)
    t_ground = np.where(ground_mask.any(axis=1), ground_mask.argmax(axis=1), NEVER)
    arrived = t_arrive <= t_ground  # touching ground AFTER arriving doesn't count
    crashed = (~arrived) & (t_ground < NEVER)

    # solvency arithmetic at CHECKPOINTS along the dream, not only its end:
    # kinetic energy above best-glide speed is convertible altitude (the sim's own
    # speed<->height exchange makes it fungible), so reach = energy height * L/D.
    # Scored at the halfway point AND the end, taking the WORST -- imagination
    # degrades with depth (keystone: sigma 0.87 at 30 s vs 0.55 at 15 s), and a
    # plan whose solvency lives entirely in its late, weakly-certified dream is
    # a procrastinator's plan: the executed first second drifts while the payoff
    # is forever 20 s away (t3 flying: every winning plan promised a glorious
    # late climb; the flown prefixes quietly left the thermal and died).
    deficit = np.zeros(len(imagined))
    est_time = np.zeros(len(imagined))
    for h in (horizon // 2, horizon):
        rem_h = dist[:, h]
        z_h = imagined[:, h, zc]
        v_h = imagined[:, h, vc]
        energy_height = np.maximum(z_h + (v_h**2 - polar.v_best_glide**2) / (2.0 * G), 0.0)
        usable_height = np.maximum(energy_height - reserve_height, 0.0)
        deficit = np.maximum(deficit, rem_h - usable_height * polar.glide_ratio)
        est_time = np.maximum(est_time, h * dt + rem_h / polar.v_best_glide)
    deficit = np.maximum(deficit, 0.0)

    # arrived futures: nothing missing, their time is the actual arrival time
    deficit = np.where(arrived & (t_arrive < NEVER), 0.0, deficit)
    est_time = np.where(arrived & (t_arrive < NEVER), np.minimum(t_arrive, horizon) * dt, est_time)
    t_crash = np.where(crashed, t_ground * dt, np.inf)
    return RolloutScores(crashed=crashed, deficit=deficit, est_time=est_time, t_crash=t_crash)


def rank(scores: RolloutScores) -> IntArr:
    """Candidate indices best-to-worst. Alive: (deficit, then time). Crashed
    rank below all alive and compare by (-survival, then deficit) -- die LATER
    first; distance-at-death only breaks exact survival ties. The np.where
    keys mix units across the alive/crashed boundary, which is legal because
    the primary `crashed` key means they are never compared across it.
    np.lexsort sorts by the LAST key first; CEM only ever needs this total
    order -- no unit-stitching scalar required."""
    key2 = np.where(scores.crashed, -scores.t_crash, scores.deficit)
    key3 = np.where(scores.crashed, scores.deficit, scores.est_time)
    out: IntArr = np.lexsort((key3, key2, scores.crashed))
    return out


def combine_scores(per_member: list[RolloutScores]) -> RolloutScores:
    """Merge per-ensemble-member scores by MEDIAN / MAJORITY: it takes 2 of 3
    members to call a candidate crashed, and the deficit/time verdicts are the
    middle member's. One net cannot say "I don't know" -- where imagination
    leaves the training manifold, independently-trained members disagree --
    and the median is the robust vote both ways: a lone hallucinating OPTIMIST
    cannot carry a candidate (t3 finding: a single net dreamed +3.4 m/s of
    lift in dead air and, under mean scoring, the planner flew 1 km into the
    fog chasing it), and a lone PESSIMIST cannot veto the right maneuver
    (also measured: worst-member ranking let one bagged member that dreamed
    the winning circle into the ground ground the whole fleet -- the planner
    could never climb again). Dial-free: no threshold to calibrate."""
    n = len(per_member)
    crashed = np.sum([m.crashed for m in per_member], axis=0) * 2 >= n
    deficit = np.median([m.deficit for m in per_member], axis=0)
    est_time = np.median([m.est_time for m in per_member], axis=0)
    t_crash = np.median([m.t_crash for m in per_member], axis=0)
    return RolloutScores(crashed=crashed, deficit=deficit, est_time=est_time, t_crash=t_crash)


# --- the CEM search --------------------------------------------------------------------
@dataclass(frozen=True)
class CEMIteration:
    """One CEM round, recorded for visualization/analysis: the sampled
    population, its best-to-worst ordering, the refit mean, and every
    candidate's imagined ground track. Only materialized when a trace list is
    passed to cem_plan -- planning itself never pays for it."""

    candidates: FloatArr  # (population, K, 2) as actually rolled (post-clip)
    order: IntArr  # rank(scores); order[:n_elites] are the elites
    mean: FloatArr  # (K, 2) the Gaussian mean refit AFTER this round
    imagined_xy: FloatArr  # (population, H+1, 2) where each candidate believed it would fly


@dataclass(frozen=True)
class Plan:
    """One replan's result: the winning candidate, the refit sampling mean
    (next replan's warm start), and the winner's imagined future (the ghost --
    what the planner BELIEVED would happen, kept for honest post-mortems)."""

    segments: FloatArr  # (K, 2) winning per-segment (bank, speed) commands
    mean: FloatArr  # (K, 2) final refit Gaussian mean
    imagined: FloatArr  # (H+1, 9) the winner's imagined panels


def cem_plan(
    ensemble: list[Checkpoint],
    panel0: FloatArr,
    goal: Goal,
    polar: GlidePolar,
    cfg: PlannerConfig,
    rng: np.random.Generator,
    warm_mean: FloatArr | None = None,
    trace: list[CEMIteration] | None = None,
) -> Plan:
    """One full CEM search from the current sensed panel.

    Candidates are (K, 2) blocks of (bank_cmd, speed_cmd): BOTH stick axes are
    planned. Speed is not a luxury -- at fixed best-glide speed the turn radius
    (~70 m) is wider than a thermal core (60 m), so climbing is physically
    impossible; slowing down inside lift is how real pilots (and now the
    planner) make circles that fit. Iterate: sample a population around
    (mean, std), clip to the training support, imagine all candidates as one
    batch, rank, refit mean/std to the elites. The incumbent mean is always
    injected as candidate 0 so a good warm-started plan can never be lost to
    sampling luck. std restarts wide every replan -- exploration must survive
    the warm start.
    """
    polar_speed = polar.v_best_glide
    # the terminal value flies the tail on a DEGRADED polar (the safety margin);
    # the imagined horizon itself stays honest -- margin applies only to the
    # beyond-horizon arithmetic, not to the model's dynamics
    scored_polar = GlidePolar(
        v_best_glide=polar.v_best_glide, glide_ratio=polar.glide_ratio * cfg.glide_margin
    )
    if warm_mean is None:
        mean = np.zeros((cfg.n_segments, 2))
        mean[:, 1] = polar_speed  # neutral prior: wings level at trim
    else:
        mean = warm_mean.copy()
    std = np.tile(np.array([cfg.init_std, cfg.pitch_init_std]), (cfg.n_segments, 1))
    floor = np.array([cfg.std_floor, cfg.pitch_std_floor])
    best = Plan(segments=mean, mean=mean, imagined=np.empty(0))  # overwritten below
    # the pilot's primitive repertoire, injected as standing candidates every
    # round: sustained circle left / circle right (min-sink-ish speed, the
    # radius that fits a core) and run-straight-at-trim. Own-body maneuvers,
    # zero field knowledge. Without them, sustained thermalling is a needle in
    # 12-dim search space: a warm-started mean that encodes ANY drift keeps
    # sampling drift, and "hold this bank for 30 s" never gets drawn (t3
    # flying: the planner brushed real lift again and again but never once
    # sampled the clean circle that would have kept it there).
    archetypes = np.array(
        [
            [[+0.70, 19.0]] * cfg.n_segments,  # circle right, slow
            [[-0.70, 19.0]] * cfg.n_segments,  # circle left, slow
            [[0.0, polar_speed]] * cfg.n_segments,  # run straight at trim
        ]
    )
    for _ in range(cfg.iterations):
        cands = rng.normal(mean, std, size=(cfg.population, cfg.n_segments, 2))
        cands[0] = mean  # keep the incumbent in the running
        cands[1 : 1 + len(archetypes)] = archetypes
        cands[..., 0] = np.clip(cands[..., 0], -cfg.max_bank_cmd, cfg.max_bank_cmd)
        cands[..., 1] = np.clip(cands[..., 1], cfg.pitch_lo, cfg.pitch_hi)
        ticks = expand_segments(cands, cfg.ticks_per_segment)
        ck = ensemble[0]  # member 0 is THE model (keystone-graded); extras vote
        ground_z = float(ck.stats.panel_lo[ck.sensor_names.index("z")])
        dreams = [imagine(m, panel0, ticks[..., 0], ticks[..., 1]) for m in ensemble]
        imagined = dreams[0]
        order = rank(
            combine_scores(
                [
                    score_rollouts(
                        d, ck.sensor_names, goal, scored_polar, ck.dt, ground_z, cfg.reserve_height
                    )
                    for d in dreams
                ]
            )
        )
        elites = cands[order[: cfg.n_elites]]
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), floor)
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
    ensemble: list[Checkpoint],
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
    flight = Flight(outcome="timeout", seconds=0.0, goal=goal)
    warm: FloatArr | None = None
    ticks = 0
    max_ticks = int(max_seconds / sim.dt)
    while ticks < max_ticks:
        panel_now = sim.sense()
        panel0 = np.array([panel_now[name] for name in ensemble[0].sensor_names])
        plan = cem_plan(ensemble, panel0, goal, polar, cfg, rng, warm_mean=warm)
        flight.plans.append(plan)
        warm = plan.mean
        cmd_ticks = expand_segments(plan.segments[None], cfg.ticks_per_segment)[0]
        for tick in range(cfg.replan_ticks):
            state = sim.step(float(cmd_ticks[tick, 0]), float(cmd_ticks[tick, 1]))
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
    """THE t3 task: from 60 m -- winch-launch height -- reach a goal 1.6 km on
    the far side of the sink band. Verified unreachable without climbing:
    NINE hand-crafted no-climb routes (straight, five around-the-band
    variants, through-B, and around-then-refuel-through-B) all crash in the
    real sim from this start. The only winning shape is a pilot's: climb at
    A, commit across the band, top up at B if needed, final-glide in.
    Nothing in the cost says any of that -- if it happens, it emerged."""
    here = Path(__file__).resolve().parent
    ensemble = [
        load_checkpoint(p)
        for p in sorted((here / "data").glob("model_full*.pt"))  # full, full_s1, full_s2
    ]
    glider, air = make_world()
    polar = best_glide(glider)
    print(f"ensemble: {len(ensemble)} member(s)")
    print(f"own polar: best glide {polar.v_best_glide:.1f} m/s at L/D {polar.glide_ratio:.1f}")

    start = GliderState(x=-80.0, y=0.0, z=60.0, heading=0.0, airspeed=24.0, bank=0.0)
    goal = Goal(x=1500.0, y=0.0)
    reach = (start.z + (start.airspeed**2 - polar.v_best_glide**2) / 19.62) * polar.glide_ratio
    need = float(np.hypot(goal.x - start.x, goal.y - start.y))
    print(f"to goal: {need:.0f} m; still-air reach from start: {reach:.0f} m -> MUST climb")

    sim = Simulation(glider, air, start)
    flight = fly_to_goal(
        sim, ensemble, goal, PlannerConfig(), np.random.default_rng(0), max_seconds=360.0
    )

    s = sim.state
    print(f"\noutcome: {flight.outcome.upper()} in {flight.seconds:.1f} s")
    print(
        f"  final state: z={s.z:.0f} m, airspeed={s.airspeed:.1f} m/s, {len(flight.plans)} replans"
    )


if __name__ == "__main__":
    main()
