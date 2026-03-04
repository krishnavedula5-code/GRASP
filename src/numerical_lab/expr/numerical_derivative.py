from __future__ import annotations
import math
from typing import Callable, Optional, Tuple

def choose_h(x: float) -> float:
    # deterministic h (no randomness), scale with x
    # sqrt(machine_epsilon) ~ 1.49e-8 for float64
    eps = 2.220446049250313e-16
    return math.sqrt(eps) * (1.0 + abs(x))

def central_diff(
    f: Callable[[float], Optional[float]],
    x: float,
) -> Tuple[Optional[float], float, Optional[float], Optional[float]]:
    """
    Returns (df, h, f(x+h), f(x-h)).
    df=None if any eval nonfinite.
    """
    h = choose_h(x)
    xp = x + h
    xm = x - h
    fp = f(xp)
    fm = f(xm)
    if fp is None or fm is None:
        return None, h, fp, fm
    return (fp - fm) / (2.0 * h), h, fp, fm