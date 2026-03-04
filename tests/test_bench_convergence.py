# tests/test_bench_convergence.py
import math
import pytest

from numerical_lab.api import NumericalEngine
from numerical_lab.benchmarks.functions import get_cases


def _get_x(res):
    """
    Extract the final/root x from the solver result object used by numerical_lab.
    Your SolverResult has .root and .best_x.
    """
    if hasattr(res, "root") and res.root is not None:
        return res.root
    if hasattr(res, "best_x") and res.best_x is not None:
        return res.best_x

    if isinstance(res, dict):
        return res.get("root") or res.get("best_x") or res.get("x")

    return None


def _is_finite_number(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False


def _valid_bisection_bracket(f, a, b) -> bool:
    """
    Bisection requires f(a)*f(b) <= 0 (sign change or touch).
    If f(a) or f(b) is not finite, treat as invalid.
    """
    try:
        fa = f(a)
        fb = f(b)
        if not (_is_finite_number(fa) and _is_finite_number(fb)):
            return False
        return fa * fb <= 0
    except Exception:
        return False


def test_benchmark_cases_converge_reasonably():
    cases = get_cases()
    tol = 1e-8
    max_iter = 200

    for case in cases:
        # ---- BISECTION (only if bracket provided AND valid) ----
        if getattr(case, "bracket", None) is not None:
            a, b = case.bracket

            if _valid_bisection_bracket(case.f, a, b):
                res, conv, stab = NumericalEngine.solve_bisection(
                    case.f, a, b, tol=tol, max_iter=max_iter
                )
                # If it ran with a valid bracket, we expect convergence in benchmarks
                assert getattr(res, "status", None) == "converged", f"{case.name}: {getattr(res,'message',res)}"
                x = _get_x(res)
                assert x is not None, f"{case.name}: missing root in result"
                assert _is_finite_number(x), f"{case.name}: non-finite root {x}"
            else:
                # bracket exists but doesn't satisfy bisection assumptions; skip bisection
                pass

        # ---- NEWTON (only if df and x0 exist) ----
        if getattr(case, "df", None) is not None and getattr(case, "x0", None) is not None:
            res, conv, stab = NumericalEngine.solve_newton(
                case.f, case.df, case.x0, tol=tol, max_iter=max_iter
            )
            assert getattr(res, "status", None) == "converged", f"{case.name} (newton): {getattr(res,'message',res)}"
            x = _get_x(res)
            assert x is not None, f"{case.name} (newton): missing root in result"
            assert _is_finite_number(x), f"{case.name} (newton): non-finite root {x}"

        # ---- SECANT (only if x0 and x1 exist) ----
         # ---- SECANT (only if x0 and x1 exist) ----
        if getattr(case, "x0", None) is not None and getattr(case, "x1", None) is not None:
            f0 = case.f(case.x0)
            f1 = case.f(case.x1)

            # Secant step is ill-posed if f(x1) ≈ f(x0)
            if not (_is_finite_number(f0) and _is_finite_number(f1)):
                continue
            if abs(f1 - f0) < 1e-14:
                continue

            res, conv, stab = NumericalEngine.solve_secant(
                case.f, case.x0, case.x1, tol=tol, max_iter=max_iter
            )
            assert getattr(res, "status", None) == "converged", f"{case.name} (secant): {getattr(res,'message',res)}"
            x = _get_x(res)
            assert x is not None, f"{case.name} (secant): missing root in result"
            assert _is_finite_number(x), f"{case.name} (secant): non-finite root {x}"
            
        # ---- HYBRID (only if df + bracket exist AND bracket valid) ----
        if getattr(case, "df", None) is not None and getattr(case, "bracket", None) is not None:
            a, b = case.bracket
            if _valid_bisection_bracket(case.f, a, b):
                res, conv, stab = NumericalEngine.solve_hybrid(
                    case.f, case.df, a, b, tol=tol, max_iter=max_iter
                )
                assert getattr(res, "status", None) == "converged", f"{case.name} (hybrid): {getattr(res,'message',res)}"
                x = _get_x(res)
                assert x is not None, f"{case.name} (hybrid): missing root in result"
                assert _is_finite_number(x), f"{case.name} (hybrid): non-finite root {x}"