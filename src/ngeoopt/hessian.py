"""Initial Hessian guesses and quasi-Newton updates (Cartesian, Bohr/Hartree)."""

from __future__ import annotations

import itertools

import numpy as np

# Lindh, Bernhardsson, Karlstrom, Malmqvist, CPL 241, 423 (1995)
_LINDH_ALPHA = np.array([[1.0000, 0.3949, 0.3949],
                         [0.3949, 0.2800, 0.2800],
                         [0.3949, 0.2800, 0.2800]])
_LINDH_RREF = np.array([[1.35, 2.10, 2.53],
                        [2.10, 2.87, 3.40],
                        [2.53, 3.40, 3.40]])
_K_STRETCH, _K_BEND, _K_TORSION = 0.45, 0.15, 0.005

_ROW1 = {"H", "He"}
_ROW2 = {"Li", "Be", "B", "C", "N", "O", "F", "Ne"}


def _row(sym: str) -> int:
    if sym in _ROW1:
        return 0
    if sym in _ROW2:
        return 1
    return 2


def _angle(x, i, j, k):
    u = x[i] - x[j]
    v = x[k] - x[j]
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.arccos(np.clip(c, -1.0, 1.0))


def _dihedral(x, i, j, k, l):
    b0 = x[i] - x[j]
    b1 = x[k] - x[j]
    b2 = x[l] - x[k]
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    return np.arctan2(np.dot(np.cross(b1n, v), w), np.dot(v, w))


def _bvec(func, x, atoms, h=1e-5):
    """Numerical Wilson B-row of an internal coordinate (only `atoms` move)."""
    b = np.zeros_like(x)
    for a in atoms:
        for k in range(3):
            xp = x.copy()
            xm = x.copy()
            xp[a, k] += h
            xm[a, k] -= h
            d = func(xp) - func(xm)
            d = (d + np.pi) % (2 * np.pi) - np.pi  # dihedral wrap-around
            b[a, k] = d / (2 * h)
    return b.ravel()


def lindh_hessian(symbols, coords, rho_cut: float = 1e-3) -> np.ndarray:
    """Lindh model Hessian in Cartesian coordinates (3N x 3N)."""
    x = np.asarray(coords, dtype=float).reshape(-1, 3)
    n = len(symbols)
    rows = [_row(s) for s in symbols]
    dist = np.linalg.norm(x[:, None] - x[None], axis=-1)
    rho = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        a = _LINDH_ALPHA[rows[i], rows[j]]
        r0 = _LINDH_RREF[rows[i], rows[j]]
        rho[i, j] = rho[j, i] = np.exp(a * (r0 ** 2 - dist[i, j] ** 2))

    H = np.zeros((3 * n, 3 * n))
    # stretches (analytic B-row)
    for i, j in itertools.combinations(range(n), 2):
        if rho[i, j] < rho_cut:
            continue
        e = (x[i] - x[j]) / dist[i, j]
        b = np.zeros((n, 3))
        b[i], b[j] = e, -e
        b = b.ravel()
        H += _K_STRETCH * rho[i, j] * np.outer(b, b)

    neigh = [[j for j in range(n) if j != i and rho[i, j] >= rho_cut] for i in range(n)]
    linear = np.deg2rad(175.0)
    # bends i-j-k centred on j
    for j in range(n):
        for i, k in itertools.combinations(neigh[j], 2):
            if _angle(x, i, j, k) > linear:
                continue
            kf = _K_BEND * rho[i, j] * rho[j, k]
            b = _bvec(lambda y: _angle(y, i, j, k), x, (i, j, k))
            H += kf * np.outer(b, b)
    # torsions i-j-k-l around j-k
    for j, k in itertools.combinations(range(n), 2):
        if rho[j, k] < rho_cut:
            continue
        for jj, kk in ((j, k), (k, j)):
            if jj > kk:
                continue
            for i in neigh[jj]:
                if i == kk:
                    continue
                for l in neigh[kk]:
                    if l in (jj, i):
                        continue
                    if _angle(x, i, jj, kk) > linear or _angle(x, jj, kk, l) > linear:
                        continue
                    kt = _K_TORSION * rho[i, jj] * rho[jj, kk] * rho[kk, l]
                    if kt < 1e-6:
                        continue
                    b = _bvec(lambda y: _dihedral(y, i, jj, kk, l), x, (i, jj, kk, l))
                    H += kt * np.outer(b, b)
    return 0.5 * (H + H.T)


def bfgs_update(H: np.ndarray, s: np.ndarray, y: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """BFGS update; skipped when the curvature condition fails."""
    sy = float(s @ y)
    if sy <= eps * np.linalg.norm(s) * np.linalg.norm(y):
        return H
    Hs = H @ s
    return H + np.outer(y, y) / sy - np.outer(Hs, Hs) / float(s @ Hs)


def sr1_update(H: np.ndarray, s: np.ndarray, y: np.ndarray, r: float = 1e-8) -> np.ndarray:
    d = y - H @ s
    den = float(d @ s)
    if abs(den) < r * np.linalg.norm(s) * np.linalg.norm(d):
        return H
    return H + np.outer(d, d) / den


def psb_update(H: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Powell-symmetric-Broyden (does not enforce positive definiteness)."""
    ss = float(s @ s)
    if ss < 1e-16:
        return H
    d = y - H @ s
    return (H + (np.outer(d, s) + np.outer(s, d)) / ss
            - float(d @ s) * np.outer(s, s) / ss ** 2)


def bofill_update(H, s, y):
    """Bofill mix of SR1 and PSB, commonly used for transition states."""
    d = y - H @ s
    num = float(d @ s) ** 2
    den = float(d @ d) * float(s @ s)
    phi = num / den if den > 0 else 0.0
    return phi * sr1_update(H, s, y) + (1 - phi) * psb_update(H, s, y)


UPDATES = {"bfgs": bfgs_update, "sr1": sr1_update, "psb": psb_update,
           "bofill": bofill_update, "none": lambda H, s, y: H}
