"""
viewport/app.py -- the flight viewport application: replay + manual flight.

Run it (from the repo root):
    .venv/bin/python -m viewport.app                 # replay data/dataset.npz
    .venv/bin/python -m viewport.app --episode 12    # jump to an episode
    .venv/bin/python -m viewport.app --fly           # straight into the cockpit
    .venv/bin/python -m viewport.app --data data/flights/manual-....npz

Two modes over one scene (F toggles):
  REPLAY -- scrub logged episodes. No physics runs: a frame index moves
            through precomputed arrays (ReplayFlight), that's all.
            SPACE play/pause · left/right scrub 1 s · , . single-step ·
            up/down playback speed · [ ] switch episode · R restart ·
            G cycle ghost predictor · click the timeline to seek.
            GHOST-COMPARE: pass a rollouts.npz (keystone.py writes it) and
            the model's IMAGINED flight rides along in violet -- ghost path,
            ghost glider, and a second instrument column -- scrubbing in
            lockstep with the true flight it was rolled from (rollout step h
            IS dataset row starts[i]+h; one clock, two beliefs about it).
            Violet ticks on the timeline mark where each rollout begins.
  FLY    -- hand-fly a REAL Simulation with the arrow keys. Left/right push
            the bank command; up/down push the speed command (pitch: pull
            up = slower). The same step() every other pilot uses -- so
            stalls, energy bleed and crashes are all live. S saves the
            flight as a dataset-schema .npz under data/flights/ (loadable
            straight back into replay, and legal model food); R respawns.
  TAB cycles cameras (chase / top-down analysis / tower) in both modes.

Architecture notes:
  - All state advancement lives in _tick(dt), all key handling in
    _input(key), all drawing in _render(surface). run() is just the pygame
    pump wiring real events into those three -- so tests drive the app
    deterministically with no window, no clock, no display.
  - Held keys arrive through self.held (a plain dict run() refreshes from
    pygame each frame); tests set it directly.
  - The HUD is rebuilt from the CURRENT source's channel names via
    panel.build_panel(), so a new sensor gets a gauge with no edits here.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame

from data_gen import make_world
from glider_sim import ACTION_NAMES, SENSOR_NAMES
from viewport import hud
from viewport.camera import Camera
from viewport.colors import GHOST_VIOLET
from viewport.frames import FlightLog, Frame, LiveFlight, ReplayFlight, RolloutSet, npz_kind
from viewport.panel import GaugeSpec, build_panel, read
from viewport.scene import WorldScene, build_world, draw_glider, draw_path, draw_world

MODE_REPLAY = "replay"
MODE_FLY = "fly"

WINDOW_SIZE = (1280, 780)

# stick feel: how fast holding an arrow key pushes the commands around.
BANK_CMD_RATE = 1.0  # rad/s of commanded bank per second held
SPEED_CMD_RATE = 5.0  # m/s of commanded airspeed per second held
BANK_CMD_LIMIT = 1.0472  # +/- 60 deg -- the airframe clamp, mirrored
SPEED_CMD_RANGE = (12.0, 50.0)  # into stall territory on purpose (it's real)

HELP_REPLAY = (
    "SPACE play  |  left/right scrub  |  , . step  |  up/down speed  |  [ ] episode"
    "  |  G ghost  |  R restart  |  F fly  |  TAB camera"
)
HELP_FLY = (
    "left/right bank  |  up/down speed (pull up = slower)  |  S save log"
    "  |  R restart  |  F replay  |  TAB camera"
)

# pygame key -> the input-name vocabulary _input()/held use
KEY_NAMES = {
    pygame.K_TAB: "tab",
    pygame.K_f: "f",
    pygame.K_SPACE: "space",
    pygame.K_COMMA: ",",
    pygame.K_PERIOD: ".",
    pygame.K_LEFTBRACKET: "[",
    pygame.K_RIGHTBRACKET: "]",
    pygame.K_r: "r",
    pygame.K_s: "s",
    pygame.K_g: "g",
    pygame.K_LEFT: "left arrow",
    pygame.K_RIGHT: "right arrow",
    pygame.K_UP: "up arrow",
    pygame.K_DOWN: "down arrow",
}
ARROWS = ("left arrow", "right arrow", "up arrow", "down arrow")


@dataclass(frozen=True)
class GhostChannel:
    """One selectable imagination stream: a predictor out of a RolloutSet.
    The G key cycles OFF -> each channel -> OFF."""

    label: str  # what the HUD calls it ("full", "twin", ...)
    rollouts: RolloutSet
    predictor: str  # key into rollouts.predictors


@dataclass(frozen=True)
class ActiveGhost:
    """The rollout under the playback cursor right now: rollout `i` of
    `channel`, whose h=0 sits at episode-local frame `local_start`."""

    channel: GhostChannel
    i: int
    local_start: int


class ViewportApp:
    """The whole application. Owns the scene, both flight sources, and the
    mode/input state machine. Rendering happens onto whatever surface is
    passed to _render -- the window (run) or an offscreen buffer (tests)."""

    def __init__(
        self,
        data_path: str | Path | None = "data/dataset.npz",
        start_mode: str | None = None,
        flights_dir: str | Path = "data/flights",
        size: tuple[int, int] = WINDOW_SIZE,
        rollout_paths: Sequence[str | Path] = (),
    ):
        # THE world -- the same fixed world every pilot flies (data_gen owns it).
        self.glider_def, self.air = make_world()
        self.world: WorldScene = build_world(self.air)
        self.cam = Camera(width=size[0], height=size[1])
        self.flights_dir = Path(flights_dir)
        self.size = size

        # sources: a logged dataset (if present) and/or a live sim.
        self.log: FlightLog | None = None
        if data_path is not None and Path(data_path).exists():
            self.log = FlightLog.load(data_path)
        self.flight: ReplayFlight | None = None
        self.live: LiveFlight | None = None

        # ghost-compare: saved imaginations, flattened to selectable channels.
        # ghost_idx 0 = off, i>0 = ghost_channels[i-1]; starts ON when any
        # rollouts loaded (whoever passes rollouts wants to SEE them).
        self.ghost_channels: list[GhostChannel] = []
        for rp in rollout_paths:
            if Path(rp).exists():
                self._bind_rollouts(RolloutSet.load(rp))
        self.ghost_idx = 1 if self.ghost_channels else 0

        # replay state
        self.ep_pos = 0  # index into log.episode_ids
        self.ep_row0 = 0  # absolute dataset row of the episode's first frame
        self.frame_pos = 0.0  # fractional frame cursor (scrub math)
        self.playing = True
        self.speed = 1.0
        # fly state
        self.bank_cmd = 0.0
        self.pitch_cmd = 25.0
        self._acc = 0.0  # real-time accumulator -> fixed sim ticks
        self.status = ""  # one-line toast (saves, warnings)
        self.banner = ""  # big center banner (CRASHED)
        self.held: dict[str, int] = dict.fromkeys(ARROWS, 0)

        self.specs: list[GaugeSpec] = []
        self.mode = MODE_REPLAY if (start_mode or MODE_REPLAY) == MODE_REPLAY else MODE_FLY
        if self.log is None:
            self.mode = MODE_FLY  # nothing to replay
        if self.mode == MODE_REPLAY:
            self._load_episode(0)
        else:
            self._reset_flight()

    # ------------------------------------------------------------- sources
    def _bind_rollouts(self, rset: RolloutSet) -> None:
        """Accept a RolloutSet as ghost channels -- IF it belongs to the loaded
        dataset. Rollout starts are absolute row indices into a specific file,
        so a mismatched pairing (e.g. rollouts from the big dataset over a
        manual-flight log) would scrub garbage; verify one rollout's answer-key
        slice against the log's actual rows and refuse on any mismatch."""
        log = self.log
        if log is None or rset.n == 0:
            return
        s0 = int(rset.starts[0])
        aligned = (
            rset.sensor_names == log.sensor_names
            and rset.dt == log.dt
            and int(rset.starts.max()) + rset.horizon < len(log.sensors)
            and np.allclose(rset.true[0], log.sensors[s0 : s0 + rset.horizon + 1])
        )
        if not aligned:
            self.status = "rollouts ignored: not from this dataset"
            return
        for name in rset.predictors:
            self.ghost_channels.append(GhostChannel(label=name, rollouts=rset, predictor=name))

    def _ghost(self) -> ActiveGhost | None:
        """The rollout the playback cursor is inside right now: of the selected
        channel's rollouts in THIS episode, the latest one already begun (its
        step h = current frame - local start, so truth and imagination share
        the clock). None while the ghost is off or between rollouts."""
        if self.mode != MODE_REPLAY or self.log is None or self.ghost_idx == 0:
            return None
        ch = self.ghost_channels[self.ghost_idx - 1]
        rset = ch.rollouts
        ep_id = self.log.episode_ids[self.ep_pos]
        frame = int(self.frame_pos)
        best: ActiveGhost | None = None
        for i in np.nonzero(rset.episode == ep_id)[0]:
            local_start = int(rset.starts[i]) - self.ep_row0
            if local_start <= frame <= local_start + rset.horizon:
                if best is None or local_start > best.local_start:
                    best = ActiveGhost(channel=ch, i=int(i), local_start=local_start)
        return best

    def _load_episode(self, ep_pos: int) -> None:
        """Point replay at episode #ep_pos (wraps around), rebuild the panel."""
        if self.log is None:
            return
        ids = self.log.episode_ids
        self.ep_pos = ep_pos % len(ids)
        self.flight = self.log.flight(ids[self.ep_pos])
        self.ep_row0 = int(np.nonzero(self.log.episode == ids[self.ep_pos])[0][0])
        self.frame_pos = 0.0
        self.playing = True
        self.banner = ""
        self.specs = build_panel(self.flight.sensor_names, self.flight.action_names)

    def _reset_flight(self) -> None:
        """A fresh manual flight at the standard spawn."""
        self.live = LiveFlight(self.glider_def, self.air)
        self.bank_cmd, self.pitch_cmd = 0.0, 25.0
        self._acc = 0.0
        self.banner = ""
        self.specs = build_panel(SENSOR_NAMES, ACTION_NAMES)

    def _current_frame(self) -> Frame:
        if self.mode == MODE_REPLAY and self.flight is not None:
            return self.flight.frame(int(self.frame_pos))
        assert self.live is not None
        return Frame(
            sensors=self.live.sim.sense(),
            actions={"bank_cmd": self.bank_cmd, "pitch_cmd": self.pitch_cmd},
        )

    # ------------------------------------------------------------ main loop
    def _tick(self, dt: float) -> None:
        """Advance one frame's worth of app time. Deterministic given dt and
        self.held -- the ONLY entry point that moves state."""
        if self.mode == MODE_REPLAY and self.flight is not None:
            if self.playing:
                self.frame_pos += dt * self.speed / self.flight.dt
                if self.frame_pos >= self.flight.n - 1:
                    self.frame_pos = float(self.flight.n - 1)
                    self.playing = False  # hold on the last frame
        elif self.mode == MODE_FLY and self.live is not None:
            self._fly(dt)

    def _fly(self, dt: float) -> None:
        """Manual flight: held arrows slew the commands; real time converts
        to fixed sim ticks through an accumulator (frame-rate independent)."""
        live = self.live
        assert live is not None
        if live.crashed:
            self.banner = "CRASHED -- R restart, S save log"
            return
        # bank_cmd is stored in the SIM's convention -- positive bank = LEFT
        # turn (heading rate g*tan(bank)/V is CCW-positive) -- so the RIGHT
        # arrow pushes it NEGATIVE. The HUD's bank gauges render sim values
        # with invert=True, so needle and stick agree with the pilot.
        stick_x = self.held["left arrow"] - self.held["right arrow"]
        self.bank_cmd += stick_x * BANK_CMD_RATE * dt
        self.bank_cmd = min(BANK_CMD_LIMIT, max(-BANK_CMD_LIMIT, self.bank_cmd))
        # UP arrow = pull the nose up = fly SLOWER (pitch_cmd is a speed).
        stick_y = self.held["down arrow"] - self.held["up arrow"]
        self.pitch_cmd += stick_y * SPEED_CMD_RATE * dt
        self.pitch_cmd = min(SPEED_CMD_RANGE[1], max(SPEED_CMD_RANGE[0], self.pitch_cmd))
        self._acc += dt
        while self._acc >= live.dt and not live.crashed:
            self._acc -= live.dt
            live.step(self.bank_cmd, self.pitch_cmd)

    # --------------------------------------------------------------- input
    def _input(self, key: str) -> None:
        if key == "tab":
            self.cam.next_mode()
        elif key == "f":
            self._toggle_mode()
        elif self.mode == MODE_REPLAY:
            self._input_replay(key)
        elif self.mode == MODE_FLY:
            self._input_fly(key)

    def _toggle_mode(self) -> None:
        self.status = ""
        if self.mode == MODE_REPLAY:
            self.mode = MODE_FLY
            self._reset_flight()
        elif self.log is not None:
            self.mode = MODE_REPLAY
            self._load_episode(self.ep_pos)
        else:
            self.status = "no dataset loaded -- replay unavailable"

    def _input_replay(self, key: str) -> None:
        if self.flight is None:
            return
        one_second = 1.0 / self.flight.dt
        if key == "space":
            self.playing = not self.playing
            if self.playing and self.frame_pos >= self.flight.n - 1:
                self.frame_pos = 0.0  # replay from the top
        elif key == "right arrow":
            self.frame_pos = min(float(self.flight.n - 1), self.frame_pos + one_second)
        elif key == "left arrow":
            self.frame_pos = max(0.0, self.frame_pos - one_second)
        elif key == ".":
            self.playing = False
            self.frame_pos = min(float(self.flight.n - 1), self.frame_pos + 1.0)
        elif key == ",":
            self.playing = False
            self.frame_pos = max(0.0, self.frame_pos - 1.0)
        elif key == "up arrow":
            self.speed = min(16.0, self.speed * 2.0)
        elif key == "down arrow":
            self.speed = max(0.25, self.speed / 2.0)
        elif key == "]":
            self._load_episode(self.ep_pos + 1)
        elif key == "[":
            self._load_episode(self.ep_pos - 1)
        elif key == "g":
            if not self.ghost_channels:
                self.status = "no rollouts loaded -- run keystone.py, then pass data/rollouts.npz"
            else:
                self.ghost_idx = (self.ghost_idx + 1) % (len(self.ghost_channels) + 1)
        elif key == "r":
            self.frame_pos = 0.0
            self.playing = True

    def _click(self, pos: tuple[int, int], surface: pygame.Surface) -> None:
        """Timeline click-to-seek (replay only)."""
        if self.mode != MODE_REPLAY or self.flight is None:
            return
        track = hud.timeline_rect(surface).inflate(0, 12)  # generous hit target
        if track.collidepoint(pos):
            frac = (pos[0] - track.left) / track.width
            self.frame_pos = min(1.0, max(0.0, frac)) * (self.flight.n - 1)

    def _input_fly(self, key: str) -> None:
        if self.live is None:
            return
        if key == "r":
            self._reset_flight()
            self.status = ""
        elif key == "s":
            if self.live.n == 0:
                self.status = "nothing flown yet"
            else:
                path = self.live.save(self.flights_dir)
                self.status = f"saved {path.as_posix()}  ({self.live.n} ticks)"

    # ------------------------------------------------------------ rendering
    def _status_line(self) -> str:
        if self.mode == MODE_REPLAY and self.flight is not None and self.log is not None:
            i, n = int(self.frame_pos), self.flight.n
            state = "PLAY" if self.playing else "PAUSE"
            ghost = ""
            if self.ghost_idx > 0:
                ghost = f"  ghost {self.ghost_channels[self.ghost_idx - 1].label}"
            return (
                f"REPLAY  ep {self.log.episode_ids[self.ep_pos]}"
                f" ({self.ep_pos + 1}/{len(self.log.episode_ids)})"
                f"  {state} x{self.speed:g}  {i * self.flight.dt:6.1f}s"
                f"  frame {i}/{n - 1}  cam {self.cam.mode}{ghost}"
            )
        if self.live is not None:
            return f"FLY  {self.live.n * self.live.dt:6.1f}s  cam {self.cam.mode}"
        return ""

    def _render(self, surface: pygame.Surface) -> None:
        """Draw the whole frame. Pure function of current state."""
        frame = self._current_frame()
        s = frame.sensors
        x, y, z = s.get("x", 0.0), s.get("y", 0.0), s.get("z", 0.0)
        heading, bank = s.get("heading", 0.0), s.get("bank", 0.0)

        self.cam.width, self.cam.height = surface.get_size()
        self.cam.update(x, y, z, heading)
        draw_world(surface, self.cam, self.world)

        if self.mode == MODE_REPLAY and self.flight is not None:
            draw_path(surface, self.cam, self.flight.path, climbs=self.flight.climbs)
        elif self.live is not None:
            draw_path(surface, self.cam, self.live.path, climbs=self.live.climbs)

        # the ghost: what the model BELIEVED this flight would do from the
        # latest rollout start behind the cursor -- path, airframe, and its
        # own instrument column, all in violet, all loudly IMAGINED.
        ghost = self._ghost()
        gp: dict[str, float] = {}
        if ghost is not None:
            rset, pred = ghost.channel.rollouts, ghost.channel.predictor
            draw_path(surface, self.cam, rset.path(pred, ghost.i), flat_color=GHOST_VIOLET)
            gp = rset.panel(pred, ghost.i, int(self.frame_pos) - ghost.local_start)
            draw_glider(
                surface,
                self.cam,
                gp.get("x", 0.0),
                gp.get("y", 0.0),
                gp.get("z", 0.0),
                gp.get("heading", 0.0),
                gp.get("bank", 0.0),
                self.glider_def.wingspan,
                color=GHOST_VIOLET,
            )

        draw_glider(surface, self.cam, x, y, z, heading, bank, self.glider_def.wingspan)

        values = {**frame.sensors, **frame.actions}
        readings = [read(spec, values[spec.name]) for spec in self.specs if spec.name in values]
        hud.draw_panel(surface, readings, title="TRUE" if ghost is not None else "")
        if ghost is not None:
            ghost_readings = [read(s, gp[s.name]) for s in self.specs if s.name in gp]
            hud.draw_panel(
                surface,
                ghost_readings,
                x=surface.get_width() - hud.PANEL_W - hud.PANEL_X,
                title=f"IMAGINED ({ghost.channel.label})",
                title_color=GHOST_VIOLET,
            )
        if self.mode == MODE_REPLAY and self.flight is not None:
            marks: list[float] = []
            if self.ghost_idx > 0 and self.log is not None:
                rs = self.ghost_channels[self.ghost_idx - 1].rollouts
                ep_id = self.log.episode_ids[self.ep_pos]
                span = max(1, self.flight.n - 1)
                for j in np.nonzero(rs.episode == ep_id)[0]:
                    marks.append((int(rs.starts[j]) - self.ep_row0) / span)
            hud.draw_timeline(
                surface,
                self.frame_pos / max(1, self.flight.n - 1),
                marks=marks,
                mark_color=GHOST_VIOLET,
            )
        hud.draw_chrome(
            surface,
            self._status_line(),
            HELP_REPLAY if self.mode == MODE_REPLAY else HELP_FLY,
            toast=self.status,
            banner=self.banner,
        )

    # ------------------------------------------------------------ the pump
    def run(self) -> None:
        """Open the window and run until quit. Everything real happens in
        _input/_tick/_render; this just pumps pygame into them."""
        pygame.init()
        screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("soaring")
        clock = pygame.time.Clock()
        running = True
        while running:
            dt = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in KEY_NAMES:
                        name = KEY_NAMES[event.key]
                        # arrows are HELD controls in fly mode, tap events in replay
                        if self.mode == MODE_FLY and name in ARROWS:
                            pass
                        else:
                            self._input(name)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._click(event.pos, screen)
            pressed = pygame.key.get_pressed()
            self.held = {
                "left arrow": int(pressed[pygame.K_LEFT]),
                "right arrow": int(pressed[pygame.K_RIGHT]),
                "up arrow": int(pressed[pygame.K_UP]),
                "down arrow": int(pressed[pygame.K_DOWN]),
            }
            self._tick(dt)
            self._render(screen)
            pygame.display.flip()
        pygame.quit()


def _sort_files(files: list[str]) -> tuple[str | None, list[str]]:
    """Split CLI file args into (dataset, rollout files) by their CONTENT --
    every trajectory .npz self-describes, so there is no flag to remember.
    No files at all = the standard pair: data/dataset.npz, plus the ghosts in
    data/rollouts.npz whenever keystone.py has written them (either may be
    absent -- no dataset just means the app boots into fly mode). A file the
    user NAMED must exist; np.load's error says so plainly."""
    if not files:
        files = [f for f in ("data/dataset.npz", "data/rollouts.npz") if Path(f).exists()]
    dataset: str | None = None
    rollouts: list[str] = []
    for f in files:
        if npz_kind(f) == "rollouts":
            rollouts.append(f)
        elif dataset is None:
            dataset = f
        else:
            raise SystemExit(f"two datasets given ({dataset}, {f}) -- replay takes one")
    return dataset, rollouts


def main() -> None:
    p = argparse.ArgumentParser(description="soaring 3D flight viewport (replay + manual flight)")
    p.add_argument(
        "files",
        nargs="*",
        help="trajectory .npz files -- a dataset and/or rollouts, told apart by "
        "content (default: data/dataset.npz + data/rollouts.npz if present)",
    )
    p.add_argument("--episode", type=int, default=None, help="episode POSITION to open at")
    p.add_argument("--fly", action="store_true", help="start in manual-flight mode")
    args = p.parse_args()
    dataset, rollouts = _sort_files(list(args.files))
    app = ViewportApp(
        data_path=dataset,
        start_mode=MODE_FLY if args.fly else MODE_REPLAY,
        rollout_paths=rollouts,
    )
    if args.episode is not None and app.log is not None:
        app._load_episode(args.episode)
    app.run()


if __name__ == "__main__":
    main()
