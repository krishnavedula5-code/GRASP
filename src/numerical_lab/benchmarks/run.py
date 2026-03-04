# src/numerical_lab/benchmarks/run.py
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional
import csv
import math

from numerical_lab.benchmarks.functions import get_cases

# IMPORTANT: adjust these imports to match your project entrypoints.
# We'll map methods to the existing solver calls.
from numerical_lab.methods import bisection, newton, secant  # <-- change if needed


def _safe_abs(x: Optional[float]) -> Optional[float]:
    return None if x is None else abs(x)


def run_all(out_csv_path: str, tol: float = 1e-10, max_iter: int = 100) -> None:
    cases = get_cases()

    rows: list[Dict[str, Any]] = []

    for case in cases:
        # --- Bisection (needs bracket)
        if case.bracket is not None:
            a, b = case.bracket
            res_bis = bisection.solve(case.f, a, b, tol=tol, max_iter=max_iter)
            rows.append({
                "case": case.name,
                "method": "bisection",
                "status": getattr(res_bis, "status", None),
                "iters": getattr(res_bis, "iterations", None),
                "x": getattr(res_bis, "x", None),
                "abs_err": _safe_abs(getattr(res_bis, "x", None) - case.root if getattr(res_bis, "x", None) is not None else None),
                "abs_f": _safe_abs(getattr(res_bis, "fx", None)),
            })

        # --- Newton (needs df and x0)
        if case.df is not None and case.x0 is not None:
            res_new = newton.solve(case.f, case.df, case.x0, tol=tol, max_iter=max_iter)
            rows.append({
                "case": case.name,
                "method": "newton",
                "status": getattr(res_new, "status", None),
                "iters": getattr(res_new, "iterations", None),
                "x": getattr(res_new, "x", None),
                "abs_err": _safe_abs(getattr(res_new, "x", None) - case.root if getattr(res_new, "x", None) is not None else None),
                "abs_f": _safe_abs(getattr(res_new, "fx", None)),
            })

        # --- Secant (needs x0,x1)
        if case.x0 is not None and case.x1 is not None:
            res_sec = secant.solve(case.f, case.x0, case.x1, tol=tol, max_iter=max_iter)
            rows.append({
                "case": case.name,
                "method": "secant",
                "status": getattr(res_sec, "status", None),
                "iters": getattr(res_sec, "iterations", None),
                "x": getattr(res_sec, "x", None),
                "abs_err": _safe_abs(getattr(res_sec, "x", None) - case.root if getattr(res_sec, "x", None) is not None else None),
                "abs_f": _safe_abs(getattr(res_sec, "fx", None)),
            })

    # Write CSV
    fieldnames = ["case", "method", "status", "iters", "x", "abs_err", "abs_f"]
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)