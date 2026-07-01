"""
fly.py -- fly a dumb policy in the glider sim and SEE what happens.

Run it:
    .venv/bin/python fly.py

It flies two gliders that each just hold a constant bank angle (so they circle
forever). The only difference is WHERE they circle:
    A) far from the thermal -> circles in dead air -> sinks
    B) on the thermal core  -> circles in rising air -> climbs

It then saves a picture, `soaring_first_flight.png`, with:
    left  -- the paths seen from above, over a heatmap of the rising air
    right -- altitude over time (the payoff plot: one line up, one line down)

TINKER (this is the whole point): change the numbers tagged  # <-- TRY  and
re-run. Watch the plot change. That is how you build intuition for the sim
your JEPA will later have to predict.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

# Make `from glider_sim import ...` work no matter what folder you run from,
# and put the output picture next to this file.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from glider_sim import Glider, GliderState, Simulation, Thermal, ThermalMap  # noqa: E402


def fly_a_circle(
    glider: Glider,
    air: ThermalMap,
    start: GliderState,
    bank_angle: float,
    seconds: float = 120,
    dt: float = 0.1,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Hold a constant bank for `seconds` and record the whole trajectory.

    Spins up one Simulation and calls step() in a loop -- the simplest possible
    "policy" (do the same thing every tick). Returns four NumPy arrays: x, y, z
    over time, plus the time stamps.
    """
    sim = Simulation(glider, air, start, dt=dt)
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    ts: list[float] = []
    for i in range(int(seconds / dt)):
        st = sim.step(bank_angle)
        xs.append(st.x)
        ys.append(st.y)
        zs.append(st.z)
        ts.append(i * dt)
    return np.array(xs), np.array(ys), np.array(zs), np.array(ts)


def main() -> None:
    # ---- the aircraft + the air ----
    glider = Glider(airspeed=15.0, base_sink=0.7)  # <-- TRY (different airframe)
    thermal = Thermal(x0=0.0, y0=0.0, w_peak=4.0, radius=60.0)  # <-- TRY (strength / width)
    air = ThermalMap(thermals=[thermal])  # wind defaults to (0, 0)

    bank = np.radians(40.0)  # 40-deg bank -> steady circle  # <-- TRY (steeper = tighter)

    # Two identical gliders, different starting spots:
    far = GliderState(x=-100.0, y=0.0, z=500.0, heading=0.0)  # <-- TRY (move it around)
    near = GliderState(x=0.0, y=0.0, z=500.0, heading=0.0)  # circles on/around the core

    xa, ya, za, ta = fly_a_circle(glider, air, far, bank)
    xb, yb, zb, tb = fly_a_circle(glider, air, near, bank)

    # ---- draw it ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # background heatmap: the thermal's updraft field
    gx, gy = np.meshgrid(np.linspace(-140, 140, 220), np.linspace(-140, 140, 220))
    ax1.contourf(gx, gy, thermal.updraft(gx, gy), levels=20, cmap="YlOrRd")
    ax1.plot(xa, ya, lw=1.6, color="tab:blue", label="far from core")
    ax1.plot(xb, yb, lw=1.6, color="tab:green", label="on the core")
    ax1.scatter([thermal.x0], [thermal.y0], marker="+", s=200, c="black", label="thermal center")
    ax1.set_title("top-down view  (red = rising air)")
    ax1.set_xlabel("east (m)")
    ax1.set_ylabel("north (m)")
    ax1.set_aspect("equal")
    ax1.legend(loc="upper right")

    ax2.axhline(500, color="gray", ls="--", lw=1, alpha=0.6)  # starting altitude
    ax2.plot(ta, za, color="tab:blue", label="far from core  (sinks)")
    ax2.plot(tb, zb, color="tab:green", label="on the core    (climbs)")
    ax2.set_title("altitude over time")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("altitude (m)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(HERE, "soaring_first_flight.png")
    fig.savefig(out, dpi=110)
    # plt.show()   # <-- uncomment to pop an interactive window when running locally

    print(f"saved plot -> {out}")
    print(f"far  glider: start 500.0 m, end {za[-1]:6.1f} m   ({za[-1] - 500:+.1f} m)")
    print(f"near glider: start 500.0 m, end {zb[-1]:6.1f} m   ({zb[-1] - 500:+.1f} m)")


if __name__ == "__main__":
    main()
