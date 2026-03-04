# src/numerical_lab/benchmarks/functions.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import math


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    f: Callable[[float], float]
    df: Optional[Callable[[float], float]]  # for Newton-like
    # For bracketing methods / hybrid:
    bracket: Optional[Tuple[float, float]]
    # For open methods:
    x0: Optional[float]
    x1: Optional[float]
    # Reference solution (for assertions/metrics)
    root: float


def get_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            name="cosx_minus_x",
            f=lambda x: math.cos(x) - x,
            df=lambda x: -math.sin(x) - 1.0,
            bracket=(0.0, 1.0),
            x0=0.5,
            x1=1.0,
            root=0.7390851332151607,
        ),
        BenchmarkCase(
            name="cubic_x3_minus_x_minus_2",
            f=lambda x: x**3 - x - 2,
            df=lambda x: 3 * x**2 - 1,
            bracket=(1.0, 2.0),
            x0=1.5,
            x1=2.0,
            root=1.5213797068045676,
        ),
        BenchmarkCase(
            name="exp_minus_3x",
            f=lambda x: math.exp(x) - 3 * x,
            df=lambda x: math.exp(x) - 3,
            bracket=(0.0, 1.0),
            x0=0.5,
            x1=1.0,
            root=0.6190612867359452,
        ),
        # A “multiple root” case to test slower convergence (Newton becomes linear)
        BenchmarkCase(
            name="multiple_root_(x-1)^2",
            f=lambda x: (x - 1.0) ** 2,
            df=lambda x: 2.0 * (x - 1.0),
            bracket=(0.0, 2.0),
            x0=1.5,
            x1=0.5,
            root=1.0,
        ),
    ]