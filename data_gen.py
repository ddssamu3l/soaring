"""
data_gen.py -- turn the sim into a DATA FACTORY (roadmap step 1 / task t1).

Runs many rollouts in ONE fixed world (the steps-1-4 regime: a single frozen
thermal), flying randomized bank angles, and logs every tick as a training row:

    (state_t, action_t, next_state_t+1, vario_t)

saved as parallel NumPy arrays in a compressed .npz. This file is the training
and test set for step 2's MLP: `(state, action) -> next_state`.

Design decisions (each one is deliberate; the ledger tracks them):
  - RAW values only. No normalization, no delta-encoding, no feature scaling --
    those are MODELING choices and belong to step 2. Logging stays pure so the
    dataset bakes in zero assumptions.
  - PIECEWISE-CONSTANT random banks: sample a bank, HOLD it ~1 s, resample.
    Per-tick random banks would average to zero net turn (straight-ish wiggly
    lines); real controllers fly sustained arcs. Holding gives the dataset the
    curved, circling trajectories the MLP must learn to predict.
  - EPISODE ids logged per row. Step 2 must split train/test BY ROLLOUT --
    adjacent rows within one rollout are nearly identical, so a random row-level
    split would leak test data into training and fake a good result.
  - VARIO logged but NOT needed by step 2's inputs (fixed field: position alone
    determines lift, so (state, action) suffices). It's here so the SAME dataset
    format survives step 5, when the field varies and sensed lift becomes the
    only honest world-channel.

THE FIREWALL (the invariant that keeps every future result real):
  The Simulation is omniscient -- it owns the true ThermalMap. A learner is NOT.
  Rows are packed by `rows_from_rollout()`, whose inputs are ONLY the glider's
  own trajectory (state, bank, vario per tick) -- it cannot see the Simulation
  or ThermalMap, so the thermal's true (x0, y0, w_peak, radius) has no path
  into the dataset. The saved file contains: own kinematics, action, sensed
  vario, episode id, and airframe/sim constants. Never thermal truth.

Run it:
    .venv/bin/python data_gen.py     ->  data/dataset.npz  (+ a sanity summary)
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt

from glider_sim import Glider, GliderState, Simulation, Thermal, ThermalMap

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
# one dataset = named parallel arrays (rows) + 0-d scalars (metadata)
Dataset = dict[str, FloatArray | IntArray]

DEFAULT_OUT = Path(__file__).parent / "data" / "dataset.npz"


def make_world() -> tuple[Glider, ThermalMap]:
    """THE fixed world of steps 1-4: one thermal at the origin, no wind, and the
    one default airframe. Every rollout flies this exact world, so the MLP can
    absorb the field into its weights (that's the point of the fixed-field
    phase). Randomizing worlds/gliders is step 5's job, and it happens HERE.
    """
    glider = Glider()  # airspeed 15 m/s, base sink 0.7 m/s
    air = ThermalMap(thermals=[Thermal(x0=0.0, y0=0.0, w_peak=4.0, radius=60.0)])
    return glider, air


def random_start(rng: np.random.Generator) -> GliderState:
    """A random spawn: anywhere in a +/-150 m box around the thermal (radius
    60 m), any heading. That covers the three regimes the MLP must learn --
    strong core, gradient edge, dead air. Altitude varies too: z is dynamically
    IRRELEVANT in v1 (nothing depends on it), and spawning at different
    altitudes keeps z decorrelated from time-in-episode so the MLP can discover
    that irrelevance instead of inheriting a spurious pattern.
    """
    return GliderState(
        x=float(rng.uniform(-150.0, 150.0)),
        y=float(rng.uniform(-150.0, 150.0)),
        z=float(rng.uniform(300.0, 700.0)),
        heading=float(rng.uniform(0.0, 2.0 * np.pi)),
    )


def sample_banks(
    rng: np.random.Generator, n_steps: int, hold_steps: int, max_bank: float
) -> FloatArray:
    """A piecewise-constant bank schedule: one uniform draw in [-max, +max] per
    block of `hold_steps` ticks. Negative = left turns, positive = right, so
    coverage is symmetric.
    """
    n_blocks = -(-n_steps // hold_steps)  # ceil division
    levels = rng.uniform(-max_bank, max_bank, size=n_blocks)
    return np.repeat(levels, hold_steps)[:n_steps].astype(np.float64)


def rows_from_rollout(
    history: list[tuple[GliderState, float, float]], final_state: GliderState
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Flatten one finished rollout into aligned row arrays.

    FIREWALL: this is the only function that builds dataset rows, and its inputs
    are ONLY the flown trajectory -- `Simulation.history` rows (state_t, bank_t,
    vario_t) plus the final state. No Simulation, no ThermalMap in the
    signature, so thermal truth is structurally unreachable from here.

    history[i+1]'s state IS the state after step i, so next_state rows come from
    shifting history by one and closing with `final_state`.
    """
    states = np.array([[s.x, s.y, s.z, s.heading] for s, _, _ in history], dtype=np.float64)
    actions = np.array([bank for _, bank, _ in history], dtype=np.float64)
    varios = np.array([vario for _, _, vario in history], dtype=np.float64)
    nexts = [s for s, _, _ in history[1:]] + [final_state]
    next_states = np.array([[s.x, s.y, s.z, s.heading] for s in nexts], dtype=np.float64)
    return states, actions, next_states, varios


def generate_dataset(
    n_rollouts: int = 200,
    steps_per_rollout: int = 600,  # 60 s of flight at dt=0.1
    hold_steps: int = 10,  # resample the bank every 1 s
    max_bank_deg: float = 50.0,  # beyond ~50 deg the sink penalty explodes
    dt: float = 0.1,
    seed: int = 0,
    out_path: str | Path | None = DEFAULT_OUT,
) -> Dataset:
    """Fly `n_rollouts` random rollouts in the fixed world and return (and
    optionally save) the dataset. Fully deterministic for a given seed.

    Saved/returned keys:
      states       (N, 4) float64 -- own kinematics (x, y, z, heading) at t
      actions      (N,)   float64 -- bank angle commanded at t (radians)
      next_states  (N, 4) float64 -- own kinematics at t+1 (the target)
      varios       (N,)   float64 -- sensed local lift at t (m/s) [see docstring]
      episode      (N,)   int64   -- rollout id, for leak-free train/test splits
      dt, airspeed, base_sink, seed -- 0-d scalars: sim + airframe constants
        (airframe params are the proposal-1 input channel -- constant, hence
        inert, until gliders are randomized ~step 5)
    """
    rng = np.random.default_rng(seed)
    glider, air = make_world()
    max_bank = float(np.radians(max_bank_deg))

    per_ep: list[tuple[FloatArray, FloatArray, FloatArray, FloatArray]] = []
    episode_ids: list[IntArray] = []
    for ep in range(n_rollouts):
        sim = Simulation(glider, air, random_start(rng), dt=dt)
        for bank in sample_banks(rng, steps_per_rollout, hold_steps, max_bank):
            sim.step(float(bank))
        per_ep.append(rows_from_rollout(sim.history, sim.state))
        episode_ids.append(np.full(steps_per_rollout, ep, dtype=np.int64))

    data: Dataset = {
        "states": np.concatenate([e[0] for e in per_ep]),
        "actions": np.concatenate([e[1] for e in per_ep]),
        "next_states": np.concatenate([e[2] for e in per_ep]),
        "varios": np.concatenate([e[3] for e in per_ep]),
        "episode": np.concatenate(episode_ids),
        "dt": np.array(dt, dtype=np.float64),
        "airspeed": np.array(glider.airspeed, dtype=np.float64),
        "base_sink": np.array(glider.base_sink, dtype=np.float64),
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
            states=data["states"],
            actions=data["actions"],
            next_states=data["next_states"],
            varios=data["varios"],
            episode=data["episode"],
            dt=data["dt"],
            airspeed=data["airspeed"],
            base_sink=data["base_sink"],
            seed=data["seed"],
        )
    return data


if __name__ == "__main__":
    dataset = generate_dataset()
    # Mode-B sanity summary: eyeball these before trusting the file. Ranges that
    # look wrong here (all-zero vario, altitude exploding) mean a broken factory.
    z = dataset["states"][:, 2]
    v = dataset["varios"]
    print(f"saved -> {DEFAULT_OUT}  ({DEFAULT_OUT.stat().st_size / 1e6:.1f} MB)")
    print(f"rows: {len(dataset['actions'])}   episodes: {int(dataset['episode'].max()) + 1}")
    print(f"altitude range: {z.min():7.1f} .. {z.max():7.1f} m")
    print(f"vario range:    {v.min():7.2f} .. {v.max():7.2f} m/s")
    print(
        f"bank range:     {np.degrees(dataset['actions'].min()):+6.1f} .. "
        f"{np.degrees(dataset['actions'].max()):+6.1f} deg"
    )
