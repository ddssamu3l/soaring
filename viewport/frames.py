"""
viewport/frames.py -- flight-data sources for the viewport. Pure NumPy, no ursina.

The one design rule that keeps the viewport from breaking as the glider grows
sensors: NOTHING downstream indexes a channel by position or assumes a channel
count. Every per-tick value travels as a name-keyed Frame; the names come from
the dataset's own in-file name arrays (replay) or glider_sim's canonical
*_NAMES tuples (live). Add a sensor -> a gauge appears. Remove one -> it
disappears. No viewport edit either way.

Two sources, one shape:
  ReplayFlight -- one logged episode out of a dataset .npz. Scrubbing is pure
                  array indexing; NO physics runs during replay.
  LiveFlight   -- wraps a real Simulation and flies it tick by tick (manual
                  flight). Records itself for free (Simulation.history) and
                  save() writes the SAME self-describing .npz schema the data
                  factory emits -- a manual flight is just another episode,
                  loadable right back into replay.
The renderer and instrument panel only ever see Frames and (n, 3) paths, so
replay and manual flight are indistinguishable to them.

Firewall note: replay reads true_states and the world's thermal layout for
DRAWING. That is legal -- the sensor firewall forbids thermal truth as MODEL
input, not as pixels for human eyes (see CLAUDE.md / glider_sim docstring).
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import numpy.typing as npt

from data_gen import rows_from_rollout
from glider_sim import (
    ACTION_NAMES,
    SENSOR_NAMES,
    STATE_NAMES,
    Glider,
    GliderState,
    Simulation,
    ThermalMap,
)

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Frame:
    """Everything true at one tick, keyed by channel NAME (never position):
    the instrument panel plus the commands being applied. This is the only
    currency the panel/renderer accept."""

    sensors: dict[str, float]
    actions: dict[str, float]


def _climbs(z: FloatArray, dt: float) -> FloatArray:
    """Per-tick climb rate (m/s) from an altitude series: forward difference,
    last value repeated so the array length matches the path (color per point)."""
    if len(z) < 2:
        return np.zeros(len(z), dtype=np.float64)
    dz = np.diff(z) / dt
    return np.concatenate([dz, dz[-1:]])


# ---------------------------------------------------------------------------
# Replay: a logged dataset, sliced into episodes.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReplayFlight:
    """One episode's aligned rows. frame(i) / path are pure indexing."""

    sensor_names: tuple[str, ...]
    action_names: tuple[str, ...]
    state_names: tuple[str, ...]
    sensors: FloatArray  # (n, S)
    actions: FloatArray  # (n, A)
    true_states: FloatArray  # (n, len(state_names))
    dt: float

    @property
    def n(self) -> int:
        return len(self.sensors)

    def frame(self, i: int) -> Frame:
        """The Frame at tick i (clamped into range, so scrub math can't crash)."""
        i = min(self.n - 1, max(0, i))
        return Frame(
            sensors=dict(zip(self.sensor_names, self.sensors[i], strict=True)),
            actions=dict(zip(self.action_names, self.actions[i], strict=True)),
        )

    def _state_col(self, name: str) -> FloatArray:
        return self.true_states[:, self.state_names.index(name)]

    @property
    def path(self) -> FloatArray:
        """(n, 3) flown positions (x east, y north, z up) -- ribbon vertices."""
        return np.stack([self._state_col("x"), self._state_col("y"), self._state_col("z")], axis=1)

    @property
    def climbs(self) -> FloatArray:
        """(n,) climb rate per point -- the ribbon's color channel."""
        return _climbs(self._state_col("z"), self.dt)


@dataclass(frozen=True)
class FlightLog:
    """A whole dataset file: parallel row arrays + the episode index that
    slices them. Self-describing -- channel names are read from the file."""

    sensor_names: tuple[str, ...]
    action_names: tuple[str, ...]
    state_names: tuple[str, ...]
    sensors: FloatArray
    actions: FloatArray
    true_states: FloatArray
    episode: npt.NDArray[np.int64]
    dt: float

    @classmethod
    def load(cls, path: str | Path) -> FlightLog:
        with np.load(path) as d:
            return cls(
                sensor_names=tuple(str(n) for n in d["sensor_names"]),
                action_names=tuple(str(n) for n in d["action_names"]),
                state_names=tuple(str(n) for n in d["state_names"]),
                sensors=d["sensors"].astype(np.float64),
                actions=d["actions"].astype(np.float64),
                true_states=d["true_states"].astype(np.float64),
                episode=d["episode"].astype(np.int64),
                dt=float(d["dt"]),
            )

    @property
    def episode_ids(self) -> list[int]:
        """Distinct episode ids, in first-appearance order."""
        seen: dict[int, None] = {}
        for e in self.episode:
            seen.setdefault(int(e), None)
        return list(seen)

    def flight(self, ep: int) -> ReplayFlight:
        """Slice out one episode as a ReplayFlight (rows are contiguous)."""
        rows = self.episode == ep
        return ReplayFlight(
            sensor_names=self.sensor_names,
            action_names=self.action_names,
            state_names=self.state_names,
            sensors=self.sensors[rows],
            actions=self.actions[rows],
            true_states=self.true_states[rows],
            dt=self.dt,
        )


# ---------------------------------------------------------------------------
# Live: a real Simulation, flown by hand, recording itself.
# ---------------------------------------------------------------------------
def default_start() -> GliderState:
    """The manual-flight spawn: west of the thermal at 500 m, pointing east
    (heading 0) straight at it, at a comfortable cruise. Deterministic on
    purpose -- every manual flight starts from the same problem."""
    return GliderState(x=-150.0, y=0.0, z=500.0, heading=0.0, airspeed=25.0, bank=0.0)


@dataclass
class LiveFlight:
    """A Simulation being flown interactively. Same Frame/path/climbs surface
    as ReplayFlight, plus step() to fly and save() to keep the log."""

    glider: Glider
    air: ThermalMap
    dt: float = 0.1
    start: GliderState = field(default_factory=default_start)

    def __post_init__(self) -> None:
        self.sim = Simulation(self.glider, self.air, self.start, dt=self.dt)

    @property
    def n(self) -> int:
        return len(self.sim.history)

    @property
    def crashed(self) -> bool:
        return self.sim.crashed

    def step(self, bank_cmd: float, pitch_cmd: float) -> Frame:
        """Advance one tick under the given stick commands and return the
        Frame AFTER the move (panel shows the freshly elapsed tick)."""
        self.sim.step(bank_cmd, pitch_cmd)
        return Frame(
            sensors=self.sim.sense(),
            actions=dict(zip(ACTION_NAMES, (bank_cmd, pitch_cmd), strict=True)),
        )

    @property
    def path(self) -> FloatArray:
        """(k+1, 3) positions flown so far: every history row + where we are now."""
        states = [s for s, _, _ in self.sim.history] + [self.sim.state]
        return np.array([[s.x, s.y, s.z] for s in states], dtype=np.float64)

    @property
    def climbs(self) -> FloatArray:
        return _climbs(self.path[:, 2], self.dt)

    def save(self, out_dir: str | Path) -> Path:
        """Write the flight as a dataset .npz with the EXACT schema data_gen
        emits (one episode, id 0; seed -1 marks 'human, not rng'). The saved
        file loads straight back into FlightLog for replay, and is model food
        like any other episode. Refuses to save a flight with no steps."""
        if self.n == 0:
            raise ValueError("nothing flown yet -- no rows to save")
        true_states, actions, sensors, true_next_states = rows_from_rollout(
            self.sim.history, self.sim.state
        )
        param_names = tuple(f.name for f in fields(self.glider))
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / _time.strftime("manual-%Y%m%d-%H%M%S.npz")
        np.savez_compressed(
            path,
            true_states=true_states,
            actions=actions,
            sensors=sensors,
            true_next_states=true_next_states,
            episode=np.zeros(len(actions), dtype=np.int64),
            state_names=np.array(STATE_NAMES),
            action_names=np.array(ACTION_NAMES),
            sensor_names=np.array(SENSOR_NAMES),
            glider_params=np.array(
                [getattr(self.glider, n) for n in param_names], dtype=np.float64
            ),
            glider_param_names=np.array(param_names),
            dt=np.array(self.dt, dtype=np.float64),
            seed=np.array(-1, dtype=np.int64),
        )
        return path
