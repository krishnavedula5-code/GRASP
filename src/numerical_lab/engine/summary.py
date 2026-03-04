from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, List
if TYPE_CHECKING:
    from numerical_lab.core.base import SolverResult

from numerical_lab.diagnostics.convergence import ConvergenceReport
from numerical_lab.diagnostics.stability import StabilityReport

import math


@dataclass
class MethodSummary:
    method: str
    status: str
    stop_reason: Optional[str]
    root: Optional[float]
    iterations: int
    last_residual: Optional[float]
    best_residual: Optional[float]
    observed_order: Optional[float]
    convergence_class: str
    stability_label: str
    key_notes: List[str]


def estimate_convergence_class_from_residuals(records):
    rs = [abs(getattr(r, "residual", None) or 0.0) for r in records]
    rs = [r for r in rs if r > 0 and math.isfinite(r)]

    if len(rs) < 5:
        return "insufficient_data"

    ps = []
    start = max(2, len(rs) - 10)
    for i in range(start, len(rs)):
        r0, r1, r2 = rs[i - 2], rs[i - 1], rs[i]
        if r0 <= 0 or r1 <= 0 or r2 <= 0:
            continue
        a = math.log(r2 / r1)
        b = math.log(r1 / r0)
        if not math.isfinite(a) or not math.isfinite(b) or abs(b) < 1e-15:
            continue
        ps.append(a / b)

    if len(ps) < 2:
        return "insufficient_data"

    ps.sort()
    p = ps[len(ps) // 2]

    if p >= 1.8:
        return "quadratic_or_better"
    if p >= 1.2:
        return "superlinear"
    if p >= 0.8:
        return "linear"
    return "sublinear_or_irregular"


def build_method_summary(
    method: str,
    result: SolverResult,
    conv: ConvergenceReport,
    stab,   # accept object OR dict safely
) -> MethodSummary:

    last_residual = result.residual_history[-1] if result.residual_history else None

    # Best residual
    best_residual: Optional[float] = None
    if getattr(result, "best_fx", None) is not None:
        best_residual = abs(result.best_fx)
    elif result.residual_history:
        best_residual = min(result.residual_history)

    # ----------------------------
    # Notes handling (robust)
    # ----------------------------
    notes: List[str] = []

    # Convergence notes
    try:
        notes.extend((conv.notes or [])[:2])
    except Exception:
        pass

    # Stability notes (compat: dict or object)
    stab_notes: List[str] = []
    stab_label: str = "unknown"

    try:
        if stab is None:
            pass
        elif isinstance(stab, dict):
            stab_notes = list(stab.get("notes") or [])
            stab_label = stab.get("label", "unknown")
        else:
            stab_notes = list(getattr(stab, "notes", []) or [])
            stab_label = getattr(stab, "label", "unknown")
    except Exception:
        stab_label = "unknown"

    if stab_notes:
        notes.extend(stab_notes[:1])

    conv_class = estimate_convergence_class_from_residuals(result.records)

    return MethodSummary(
        method=method,
        status=result.status,
        stop_reason=getattr(result, "stop_reason", None),
        root=result.root,
        iterations=result.iterations,
        last_residual=last_residual,
        best_residual=best_residual,
        observed_order=conv.observed_order,
        convergence_class=conv_class,
        stability_label=stab_label,
        key_notes=notes,
    )


def build_comparison_summary(comparison_results):
    summaries = {}
    for method, triple in comparison_results.items():
        result, conv, stab = triple
        summaries[method] = build_method_summary(method, result, conv, stab)
    return summaries