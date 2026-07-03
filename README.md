# soaring — minimal glider-in-a-thermal sim

Pure Python + NumPy, **no ML**. This is the little "world" whose future a JEPA
will later try to predict. Build intuition for it now; predict it later.

![two gliders holding a constant bank: green circles a thermal core and climbs, blue circles in dead air and sinks](soaring_first_flight.png)

## what's here
- `glider_sim.py` — the physics. The only function that matters is `step()`:
  it takes the glider's state + an action (bank angle) and returns the state
  one tick later. Read it top to bottom; it's ~one screen of real code.
- `fly.py` — flies a dumb constant-bank policy (two gliders: one on the
  thermal core, one out in dead air) and saves a plot.

## setup (one time, uses `uv`)
```
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python numpy matplotlib pygame
```

## run
```
.venv/bin/python fly.py
```
Writes `soaring_first_flight.png` (top-down paths + altitude-over-time).

## the viewport — watch flights, fly one yourself, or replay the model's imagination
A 3D flight viewer (vector-projected, pygame — no GPU stack) with two modes:

```
.venv/bin/python data_gen.py                # once: build data/dataset.npz
.venv/bin/python -m viewport.app            # REPLAY logged episodes
.venv/bin/python -m viewport.app --fly      # FLY with the arrow keys
```

- **REPLAY**: SPACE play/pause · ←/→ scrub · `,` `.` single-step · ↑/↓ speed ·
  `[` `]` switch episode · `G` ghost predictor · click the timeline to seek.
- **FLY**: ←/→ bank, ↑/↓ speed (pull up = slower — it's a real energy
  exchange, you can stall). `S` saves the flight as a dataset-schema `.npz`
  under `data/flights/` — loadable straight back into replay (pass the file
  as an argument).
- **TAB** cycles cameras: chase / top-down analysis (the updraft heatmap IS
  the ground) / tower. `F` toggles modes. The flight ribbon is colored by
  climb rate: blue gaining, red losing.
- The instrument panel builds itself from the dataset's own channel names —
  new sensors appear automatically.

**Ghost-compare** — after `keystone.py` writes `data/rollouts.npz`, replay
picks it up automatically (or name the files: `python -m viewport.app
data/dataset.npz data/rollouts.npz` — they're told apart by content). In
episodes that have rollouts, the model's IMAGINED flight rides along in
violet — ghost path, ghost airframe, its own instrument column — scrubbed in
lockstep with the true flight it was rolled from. `G` cycles predictors
(full / twin / teacher-forced / off); violet timeline ticks mark where each
15 s imagination begins. Watch the blindfolded twin's VARIO drift from
reality while the full model's needle stays honest — the keystone plot,
animated.

## quick menu
```
python3 scripts/menu.py
```
prints every CLI, the automatic hooks, and the
escape hatches at a glance — the fastest way to see what you can run.

## tinker — this is the actual point
Open `fly.py` and change the lines tagged `# <-- TRY`, then re-run:
- **thermal `w_peak` / `radius`** — stronger / wider rising air.
- **`bank`** — steeper turns make a tighter circle (stays in the core) but
  sink faster. Find where it stops climbing.
- **start positions** — who sits in the core vs. the edge.

Open `glider_sim.py` and poke the physics in `step()` and `sink_rate()`.
Anything unclear: ask Claude about the exact line.

## next (stage 4, later)
Log a pile of `(state, action, next_state)` from this sim → train a tiny
predictor → plot free-running vs teacher-forced error vs horizon (the
keystone). That predictor is the first JEPA. Not yet — get comfy here first.
