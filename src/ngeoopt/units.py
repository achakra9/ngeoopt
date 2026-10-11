"""Unit conversions. Internally ngeoopt works in Bohr and Hartree."""

BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCALMOL = 627.5094740631
HARTREE_TO_KJMOL = 2625.4996394799

# factor that converts a value in the given unit into Hartree
ENERGY_UNITS = {
    "hartree": 1.0,
    "au": 1.0,
    "ev": 1.0 / HARTREE_TO_EV,
    "kcal/mol": 1.0 / HARTREE_TO_KCALMOL,
    "kj/mol": 1.0 / HARTREE_TO_KJMOL,
}

# factor that converts a length in the given unit into Bohr
LENGTH_UNITS = {
    "bohr": 1.0,
    "au": 1.0,
    "angstrom": ANGSTROM_TO_BOHR,
    "ang": ANGSTROM_TO_BOHR,
}


def energy_to_hartree(value: float, unit: str) -> float:
    try:
        return value * ENERGY_UNITS[unit.lower()]
    except KeyError:
        raise ValueError(f"Unknown energy unit {unit!r}; use one of {sorted(ENERGY_UNITS)}")


def length_factor(unit: str) -> float:
    try:
        return LENGTH_UNITS[unit.lower()]
    except KeyError:
        raise ValueError(f"Unknown length unit {unit!r}; use one of {sorted(LENGTH_UNITS)}")
