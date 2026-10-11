"""Spin helpers shared by backends that report <S^2> instead of a multiplicity."""

from __future__ import annotations

import math
from typing import Optional


def multiplicity_from_s2(s2: float, nelec: Optional[int] = None,
                         tol: float = 0.25) -> Optional[int]:
    """2S+1 from <S^2>, or None when the state is spin-contaminated.

    With the electron count known, only S values of the right parity are
    allowed (integer S for even electrons), so e.g. <S^2> = 0.95 in a
    spin-flip run is reported as contaminated rather than as a doublet.
    """
    s = (math.sqrt(1.0 + 4.0 * max(s2, 0.0)) - 1.0) / 2.0
    if nelec is None:
        cand = round(2 * s) / 2.0
    elif nelec % 2 == 0:
        cand = float(round(s))
    else:
        cand = max(0.5, round(s - 0.5) + 0.5)
    if abs(s2 - cand * (cand + 1)) <= tol:
        return int(round(2 * cand + 1))
    return None
