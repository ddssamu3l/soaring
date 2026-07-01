"""Property tests for specific_energy.

These assert physics that must hold for any correct implementation — energy is
monotone in both altitude and airspeed, and a hand-computed case pins the exact
formula — rather than echoing whatever the code happens to return.
"""

from energy import specific_energy


def test_energy_rises_with_altitude() -> None:
    # more height at the same speed == strictly more stored energy.
    low = specific_energy(z=100.0, airspeed=15.0)
    high = specific_energy(z=200.0, airspeed=15.0)
    assert high > low


def test_energy_rises_with_airspeed() -> None:
    # more speed at the same height == strictly more stored energy.
    slow = specific_energy(z=100.0, airspeed=10.0)
    fast = specific_energy(z=100.0, airspeed=30.0)
    assert fast > slow


def test_known_hand_computed_value() -> None:
    # g*z + 0.5*v**2 = 9.81*100 + 0.5*20**2 = 981 + 200 = 1181 J/kg.
    assert specific_energy(z=100.0, airspeed=20.0, g=9.81) == 1181.0
