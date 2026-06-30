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
uv pip install --python .venv/bin/python numpy matplotlib
```

## run
```
.venv/bin/python fly.py
```
Writes `soaring_first_flight.png` (top-down paths + altitude-over-time).

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
