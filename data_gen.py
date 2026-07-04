"""
data_gen.py -- turn the sim into a DATA FACTORY (roadmap step 1 / task t1).

Runs many rollouts in ONE fixed world (the steps-1-4 regime: a frozen field --
since t3, the DECISION world: thermal A, a sink band, thermal B), flying
randomized stick commands, and logs every tick as a row:

    (true_state_t, action_t, sensors_t, true_state_t+1)

saved as parallel NumPy arrays in a compressed .npz. This file is the training
and test set for step 2's MLP.

WHO EATS WHAT (the naming convention carries the firewall):
    sensors + actions (+ Glider params)  ->  model food
    true_states / true_next_states       ->  evaluation ONLY (the answer key)
  Anything `true_*` in a model's input diet is a bug by definition. In the
  fixed-field phase sensors[:, :6] literally mirror the true kinematics (self
  is fully observable through the panel) -- the discipline costs nothing now,
  and at step 5 it's the whole experiment.

Design decisions (each deliberate; logged in progress.txt):
  - RAW values, one tick per row. No normalization, no deltas, no K-step
    jumps -- those are MODELING choices and belong to step 2. Multi-step
    consequences live in the data as contiguous episodes (chain the rows);
    K-step pairs are derivable downstream, the reverse is impossible.
  - PIECEWISE-CONSTANT commands: sample a (bank_cmd, pitch_cmd), HOLD ~1 s,
    resample. With command lag this is load-bearing: actions need time for
    their consequences to materialize (roll-in arcs, zooms, stalls). Per-tick
    resampling would jitter around neutral and never complete a maneuver.
  - Command ranges deliberately reach into stall territory (slow pitch_cmd),
    so the dataset CONTAINS stalls -- the model must learn the wing quitting.
  - EPISODE ids per row; episodes end EARLY on ground impact (variable
    length). Step 2 must split train/test BY EPISODE -- adjacent rows are
    nearly identical, so a row-level split would leak and fake a good result.
  - SELF-DESCRIBING file: channel name arrays (state/action/sensor/param) are
    stored inside the .npz, so future sessions read structure from the file,
    never from memory of this code.

Run it:
    .venv/bin/python data_gen.py     ->  data/dataset.npz  (+ a sanity summary)
"""

from dataclasses import fields
from pathlib import Path

import numpy as np
import numpy.typing as npt

from glider_sim import (
    ACTION_NAMES,
    SENSOR_NAMES,
    STATE_NAMES,
    Glider,
    GliderState,
    Simulation,
    Thermal,
    ThermalMap,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
StrArray = npt.NDArray[np.str_]
# one dataset = named parallel arrays (rows) + name arrays + 0-d scalars
Dataset = dict[str, FloatArray | IntArray | StrArray]

DEFAULT_OUT = Path(__file__).parent / "data" / "dataset.npz"

# command-sampling ranges: banks cover both turn directions; speeds run from
# below stall (so stalls happen and get logged) up to fast cruise.
MAX_BANK_CMD = np.radians(50.0)
SPEED_CMD_RANGE = (15.0, 35.0)  # m/s; stall is ~16, so the slow end mushes


def make_world() -> tuple[Glider, ThermalMap]:
    """THE fixed world of steps 1-4 -- since t3, the DECISION world.

    One thermal proved the loop (t1/t2). t3's task -- REACH A GOAL -- only
    forces a decision if the straight line cannot work, so the world becomes:
    home thermal A at the origin, a broad SINK band walling off the corridor
    around x~500 (real thermals are ringed by compensating sink; here it is
    arranged so "just glide at the goal" is a losing plan), and a weaker
    thermal B beyond it as the far-side top-up. From a low spawn, a goal past
    B is out of glide range until the glider CLIMBS at A first.

    Still ONE frozen field for every episode, so the MLP absorbs it into its
    weights (the point of the fixed-field phase). Randomizing worlds/gliders
    is step 5's job, and it happens HERE.
    """
    glider = Glider()  # ASK-21-class trainer (see glider_sim.Glider)
    air = ThermalMap(
        thermals=[
            Thermal(x0=0.0, y0=0.0, w_peak=4.0, radius=110.0),  # A: the home climb
            #   (radius 110: real thermal scale, and a circle at sane speeds
            #    FITS inside -- at r=60 only a perfect min-speed circle could)
            # the sink WALL: five overlapping bowls spanning y ~ -700..700,
            # strong enough that no non-climbing route survives it (verified
            # empirically at the demo spawn: straight, around-the-end and
            # refuel-through-B variants all crash) and long enough that
            # around-the-end is STRICTLY dominated -- the world presents one
            # clear commitment, not a maze of near-viable cheats
            Thermal(x0=500.0, y0=-500.0, w_peak=-3.0, radius=200.0),
            Thermal(x0=500.0, y0=-250.0, w_peak=-3.0, radius=200.0),
            Thermal(x0=500.0, y0=0.0, w_peak=-3.0, radius=200.0),
            Thermal(x0=500.0, y0=250.0, w_peak=-3.0, radius=200.0),
            Thermal(x0=500.0, y0=500.0, w_peak=-3.0, radius=200.0),
            Thermal(x0=1000.0, y0=0.0, w_peak=3.5, radius=90.0),  # B: the far top-up
        ]
    )
    return glider, air


def random_start(rng: np.random.Generator) -> GliderState:
    """A random spawn anywhere in the task corridor: A's neighborhood, the
    sink band, past B -- the model must have FLOWN all the air it will later
    be asked to imagine (a planner routed through un-flown sky is scoring
    ungraded hallucinations). Altitude now reaches near-ground: the decision
    task spawns low, so low-altitude dynamics must be in the data -- which
    also means some rollouts CRASH, and their shortened episodes are real,
    wanted rows (the ground is part of the world).
    """
    return GliderState(
        x=float(rng.uniform(-250.0, 1250.0)),
        y=float(rng.uniform(-350.0, 350.0)),
        z=float(rng.uniform(30.0, 600.0)),
        heading=float(rng.uniform(0.0, 2.0 * np.pi)),
        airspeed=float(rng.uniform(18.0, 30.0)),
        bank=0.0,
    )


def sample_commands(
    rng: np.random.Generator, n_steps: int, hold_steps: int
) -> tuple[FloatArray, FloatArray]:
    """Piecewise-constant stick schedules: one (bank_cmd, pitch_cmd) draw per
    block of `hold_steps` ticks, held so consequences complete (see module
    docstring). Bank covers both signs; pitch_cmd is a target AIRSPEED (m/s).
    """
    n_blocks = -(-n_steps // hold_steps)  # ceil division

    def held(levels: FloatArray) -> FloatArray:
        return np.repeat(levels, hold_steps)[:n_steps].astype(np.float64)

    banks = rng.uniform(-MAX_BANK_CMD, MAX_BANK_CMD, size=n_blocks)
    speeds = rng.uniform(SPEED_CMD_RANGE[0], SPEED_CMD_RANGE[1], size=n_blocks)
    return held(banks), held(speeds)


def fly_rollout(sim: Simulation, banks: FloatArray, speeds: FloatArray) -> None:
    """Fly one rollout: step through the command schedules, stopping early if
    the glider hits the ground (the flight is over; shorter episode logged)."""
    for bank_cmd, pitch_cmd in zip(banks, speeds, strict=True):
        if sim.crashed:
            break
        sim.step(float(bank_cmd), float(pitch_cmd))


def rows_from_rollout(
    history: list[tuple[GliderState, tuple[float, float], dict[str, float]]],
    final_state: GliderState,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Flatten one finished rollout into aligned row arrays:
    (true_states, actions, sensors, true_next_states).

    FIREWALL: this is the only function that builds dataset rows, and its
    inputs are ONLY the flown trajectory -- `Simulation.history` rows
    (state_t, commands_t, panel_t) plus the final state. No Simulation, no
    ThermalMap in the signature, so thermal truth is structurally unreachable.

    history[i+1]'s state IS the state after step i, so next-state rows come
    from shifting history by one and closing with `final_state`. Channel
    orders come from glider_sim's canonical *_NAMES tuples.
    """
    true_states = np.array(
        [[getattr(s, n) for n in STATE_NAMES] for s, _, _ in history], dtype=np.float64
    )
    actions = np.array([list(a) for _, a, _ in history], dtype=np.float64)
    sensors = np.array(
        [[panel[n] for n in SENSOR_NAMES] for _, _, panel in history], dtype=np.float64
    )
    nexts = [s for s, _, _ in history[1:]] + [final_state]
    true_next_states = np.array(
        [[getattr(s, n) for n in STATE_NAMES] for s in nexts], dtype=np.float64
    )
    return true_states, actions, sensors, true_next_states


def generate_dataset(
    n_rollouts: int = 600,  # t3 corridor is ~10x the t1 box's area; 3x rows is the
    #   coverage bet -- card.py + keystone.py must ratify it (or we add data)
    steps_per_rollout: int = 600,  # 60 s of flight at dt=0.1
    hold_steps: int = 10,  # resample the stick every 1 s
    dt: float = 0.1,
    seed: int = 0,
    out_path: str | Path | None = DEFAULT_OUT,
) -> Dataset:
    """Fly `n_rollouts` random rollouts in the fixed world and return (and
    optionally save) the dataset. Fully deterministic for a given seed.

    Saved/returned keys:
      true_states, true_next_states  (N, 6) -- answer key: EVALUATION ONLY
      sensors                        (N, 9) -- the panel: model food
      actions                        (N, 2) -- the commands: model food
      episode                        (N,)   -- rollout id (split by this!)
      state_names, action_names, sensor_names -- channel orders, in-file
      glider_params, glider_param_names -- the airframe (proposal-1 channel:
        constant, hence inert, until gliders are randomized ~step 5)
      dt, seed -- sim constants
    """
    rng = np.random.default_rng(seed)
    glider, air = make_world()

    per_ep: list[tuple[FloatArray, FloatArray, FloatArray, FloatArray]] = []
    episode_ids: list[IntArray] = []
    n_crashed = 0
    for ep in range(n_rollouts):
        sim = Simulation(glider, air, random_start(rng), dt=dt)
        banks, speeds = sample_commands(rng, steps_per_rollout, hold_steps)
        fly_rollout(sim, banks, speeds)
        n_crashed += int(sim.crashed)
        per_ep.append(rows_from_rollout(sim.history, sim.state))
        episode_ids.append(np.full(len(sim.history), ep, dtype=np.int64))

    # airframe params, straight off the dataclass so names can't drift
    param_names = tuple(f.name for f in fields(glider))
    param_values = np.array([getattr(glider, n) for n in param_names], dtype=np.float64)

    data: Dataset = {
        "true_states": np.concatenate([e[0] for e in per_ep]),
        "actions": np.concatenate([e[1] for e in per_ep]),
        "sensors": np.concatenate([e[2] for e in per_ep]),
        "true_next_states": np.concatenate([e[3] for e in per_ep]),
        "episode": np.concatenate(episode_ids),
        "state_names": np.array(STATE_NAMES),
        "action_names": np.array(ACTION_NAMES),
        "sensor_names": np.array(SENSOR_NAMES),
        "glider_params": param_values,
        "glider_param_names": np.array(param_names),
        "dt": np.array(dt, dtype=np.float64),
        "seed": np.array(seed, dtype=np.int64),
    }

    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # keys enumerated one by one (not **data) so the file's exact contents
        # are visible right here at the save site -- the firewall's allowed
        # channels, and nothing else. (Also: numpy's stubs can't type a **dict.)
        np.savez_compressed(
            path,
            true_states=data["true_states"],
            actions=data["actions"],
            sensors=data["sensors"],
            true_next_states=data["true_next_states"],
            episode=data["episode"],
            state_names=data["state_names"],
            action_names=data["action_names"],
            sensor_names=data["sensor_names"],
            glider_params=data["glider_params"],
            glider_param_names=data["glider_param_names"],
            dt=data["dt"],
            seed=data["seed"],
        )
    return data


if __name__ == "__main__":
    dataset = generate_dataset()
    # Mode-B sanity summary: eyeball these before trusting the file. Ranges
    # that look wrong (all-zero varios, altitude exploding, zero crashes with
    # stall-range commands) mean a broken factory.
    true_states = np.asarray(dataset["true_states"], dtype=np.float64)
    panel = np.asarray(dataset["sensors"], dtype=np.float64)
    z, v, te = true_states[:, 2], true_states[:, 4], panel[:, 7]
    n_eps = int(np.asarray(dataset["episode"], dtype=np.int64).max()) + 1
    print(f"saved -> {DEFAULT_OUT}  ({DEFAULT_OUT.stat().st_size / 1e6:.1f} MB)")
    print(f"rows: {len(dataset['actions'])}   episodes: {n_eps}")
    print(f"altitude: {z.min():7.1f} .. {z.max():7.1f} m")
    print(f"airspeed: {v.min():7.1f} .. {v.max():7.1f} m/s")
    print(f"TE vario: {te.min():+7.2f} .. {te.max():+7.2f} m/s")
