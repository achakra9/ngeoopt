import numpy as np
import pytest

from ngeoopt import Molecule
from ngeoopt.geometry import project_rigid, read_xyz, rigid_body_modes, write_xyz
from ngeoopt.units import ANGSTROM_TO_BOHR


def test_xyz_roundtrip(tmp_path, water):
    p = tmp_path / "w.xyz"
    write_xyz(p, water.symbols, water.coords_angstrom, comment="water")
    xyz = read_xyz(p)
    assert xyz.symbols == ["O", "H", "H"]
    assert xyz.comment == "water"
    np.testing.assert_allclose(xyz.coords, water.coords_angstrom, atol=1e-9)


def test_molecule_units(tmp_path):
    p = tmp_path / "h2.xyz"
    p.write_text("2\n\nH 0 0 0\nh 0 0 0.74\n")
    m = Molecule.from_xyz(p)
    assert m.symbols == ["H", "H"]
    assert m.coords[1, 2] == pytest.approx(0.74 * ANGSTROM_TO_BOHR)
    mb = Molecule.from_xyz(p, units="bohr")
    assert mb.coords[1, 2] == pytest.approx(0.74)


def test_bad_xyz(tmp_path):
    p = tmp_path / "bad.xyz"
    p.write_text("3\ncomment\nH 0 0 0\n")
    with pytest.raises(ValueError):
        read_xyz(p)


def test_rigid_modes(water, triatomic):
    assert rigid_body_modes(water.coords).shape[0] == 6
    linear = np.array([[0, 0, 0], [0, 0, 1.0], [0, 0, 2.5]])
    assert rigid_body_modes(linear).shape[0] == 5
    v = np.random.default_rng(1).normal(size=triatomic.coords.shape)
    p = project_rigid(v, triatomic.coords)
    assert np.abs(p.sum(0)).max() < 1e-12                       # no net translation
    assert np.abs(np.cross(triatomic.coords, p).sum(0)).max() < 1e-10   # no net rotation
