"""Specific mechanical energy — the currency a soaring glider actually trades.

Altitude and airspeed are interconvertible: a glider can trade height for speed
(dive) or speed for height (zoom). What a thermal *adds* is total energy, so the
quantity worth tracking is specific total energy (energy per unit mass, in J/kg),
not altitude alone. This module exposes that single scalar.
"""

from __future__ import annotations


def specific_energy(z: float, airspeed: float, g: float = 9.81) -> float:
    """Total specific mechanical energy per unit mass (J/kg).

    The sum of specific potential energy (g*z) and specific kinetic energy
    (0.5*airspeed**2). Mass cancels, so this is the height-plus-speed "budget"
    a glider spends to stay aloft.

    Args:
        z: altitude above the reference datum, in metres.
        airspeed: speed through the air, in metres per second.
        g: gravitational acceleration, in m/s**2 (default 9.81).

    Returns:
        Specific total energy in joules per kilogram (J/kg).
    """
    return g * z + 0.5 * airspeed**2
