from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Any, Dict, List, Optional, Literal

# ✅ IMPORTANT:
# Do NOT import log_event at module import time.
# It creates circular imports:
# base.py -> trace.py -> summary.py -> base.py
# We'll import it lazily inside _event().


Status = Literal[
    "converged",
    "max_iter",
    "bad_bracket",
    "derivative_zero",
    "nan_or_inf",
    "stagnation",
    "error",
]

StopReason = Literal[
    "TOL_F",
    "TOL_X",
    "TOL_BRACKET",
    "MAX_ITER",
    "BAD_BRACKET",
    "DERIVATIVE_ZERO",
    "NAN_INF",
    "DOMAIN_ERROR"
    "STAGNATION",
    "EXACT_ROOT",
    "ERROR",
]


@dataclass
class IterationRecord:
    k: int
    x: float
    fx: float
    step_error: Optional[float]
    residual: float

    step_type: Optional[str] = None
    accepted: bool = True
    reject_reason: Optional[str] = None

    a: Optional[float] = None
    b: Optional[float] = None
    interval_width: Optional[float] = None
    m: Optional[float] = None
    fm: Optional[float] = None

    dfm: Optional[float] = None
    x_newton: Optional[float] = None
    fx_newton: Optional[float] = None

    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverResult:
    method: str
    root: Optional[float]
    status: Status
    message: str
    iterations: int
    records: List[IterationRecord] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)

    stop_reason: StopReason = "ERROR"
    tol: float = 0.0
    best_x: Optional[float] = None
    best_fx: Optional[float] = None

    n_f: int = 0
    n_df: int = 0

    @property
    def x_history(self) -> List[float]:
        return [r.x for r in self.records]

    @property
    def residual_history(self) -> List[float]:
        return [r.residual for r in self.records]

    @property
    def step_error_history(self) -> List[Optional[float]]:
        return [r.step_error for r in self.records]


class RootSolver(ABC):
    def __init__(
        self,
        f: Callable[[float], float],
        tol: float = 1e-8,
        max_iter: int = 100,
    ):
        self.f = f
        self.tol = float(tol)
        self.max_iter = int(max_iter)

        self._records: List[IterationRecord] = []
        self.events: List[Dict[str, Any]] = []

        self.n_f: int = 0
        self.n_df: int = 0

    @property
    def records(self) -> List[IterationRecord]:
        return self._records

    def _safe_eval(self, x: float) -> Optional[float]:
        try:
            y = self.f(x)
        except ValueError as e:
            # common for log(x<=0), sqrt(x<0), etc.
            self._event("domain_error", code="DOMAIN_ERROR", level="error", x=x, err=str(e))
            return None
        except OverflowError as e:
            self._event("overflow", code="OVERFLOW", level="error", x=x, err=str(e))
            return None
        except Exception as e:
            self._event("eval_error", code="ERROR", level="error", x=x, err=str(e))
            return None

        if y is None or not isinstance(y, (int, float)):
            self._event("nonfinite", code="NONFINITE", level="error", x=x, where="non_numeric")
            return None

        y = float(y)

        if y != y or y in (float("inf"), float("-inf")):
            self._event("nonfinite", code="NONFINITE", level="error", x=x, where="nan_or_inf")
            return None

        self.n_f += 1
        return y

    def _record(
        self,
        *,
        k: int,
        x: float,
        fx: float,
        x_prev: Optional[float],
        step_type: Optional[str] = None,
        accepted: bool = True,
        reject_reason: Optional[str] = None,
        a: Optional[float] = None,
        b: Optional[float] = None,
        m: Optional[float] = None,
        fm: Optional[float] = None,
        dfm: Optional[float] = None,
        x_newton: Optional[float] = None,
        fx_newton: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        step_error = None if x_prev is None else abs(x - x_prev)
        interval_width = None if (a is None or b is None) else abs(b - a)

        self._records.append(
            IterationRecord(
                k=int(k),
                x=float(x),
                fx=float(fx),
                step_error=(None if step_error is None else float(step_error)),
                residual=float(abs(fx)),
                step_type=step_type,
                accepted=accepted,
                reject_reason=reject_reason,
                a=(None if a is None else float(a)),
                b=(None if b is None else float(b)),
                interval_width=(None if interval_width is None else float(interval_width)),
                m=(None if m is None else float(m)),
                fm=(None if fm is None else float(fm)),
                dfm=(None if dfm is None else float(dfm)),
                x_newton=(None if x_newton is None else float(x_newton)),
                fx_newton=(None if fx_newton is None else float(fx_newton)),
                meta=(meta or {}),
            )
        )

    def _event(
        self,
        kind: str,
        *,
        k: Optional[int] = None,
        code: Optional[str] = None,
        level: str = "info",
        **data: Any,
    ) -> None:
        if code is None:
            code = str(kind).upper()

        # ✅ Lazy import here prevents circular import at module-load time.
        from numerical_lab.engine.trace import log_event

        log_event(
            {"events": self.events},
            kind=kind,
            data=data,
            k=k,
            code=code,
            level=level,
        )

    @abstractmethod
    def solve(self) -> SolverResult:
        ...