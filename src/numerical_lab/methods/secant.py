from __future__ import annotations

from typing import Callable, Optional

from numerical_lab.core.base import RootSolver, SolverResult


class SecantSolver(RootSolver):
    """
    Secant method for f(x)=0 (derivative-free).
    Requires two initial guesses x0, x1.

    Commercial requirements:
    - Full iteration history
    - Detects division-by-near-zero in denominator (f1 - f0)
    - Detects NaN/Inf
    - Detects stagnation
    - Teaching trace: records step type + denom info + stop_reason
    """

    def __init__(
        self,
        f: Callable[[float], float],
        x0: float,
        x1: float,
        tol: float = 1e-8,
        max_iter: int = 50,
        denom_tol: float = 1e-14,
        stagnation_tol: float = 1e-14,
        tol_x: Optional[float] = None,  # ✅ optional step tolerance (defaults to tol)
    ):
        super().__init__(f=f, tol=tol, max_iter=max_iter)
        self.x0 = float(x0)
        self.x1 = float(x1)
        self.denom_tol = float(denom_tol)
        self.stagnation_tol = float(stagnation_tol)

        # ✅ Separate tolerances (UI still provides only `tol`)
        self.tol_f = float(tol)
        self.tol_x = float(tol_x) if tol_x is not None else float(tol)

    def solve(self) -> SolverResult:
        x_prev = self.x0
        x = self.x1

        f_prev = self._safe_eval(x_prev)
        f_curr = self._safe_eval(x)

        if f_prev is None or f_curr is None:
            self._event(
                "nonfinite",
                k=0,
                code="NONFINITE",
                level="error",
                x0=x_prev,
                x1=x,
                where="f(x0) or f(x1)",
            )
            return SolverResult(
                method="secant",
                root=None,
                status="nan_or_inf",
                stop_reason="NAN_INF",
                message="f(x0) or f(x1) could not be evaluated (NaN/Inf/error).",
                iterations=0,
                records=[],
                events=self.events,
                best_x=None,
                best_fx=None,
                tol=self.tol_f,
                n_f=self.n_f,
                n_df=self.n_df,
            )

        # record initial points
        self._record(k=0, x=x_prev, fx=f_prev, x_prev=None, step_type="secant")
        self._event(
            "init",
            k=0,
            code="INIT_X0",
            level="info",
            x=x_prev,
            fx=f_prev,
            which="x0",
            tol_f=self.tol_f,
            tol_x=self.tol_x,
        )

        self._record(k=1, x=x, fx=f_curr, x_prev=x_prev, step_type="secant")
        self._event(
            "init",
            k=1,
            code="INIT_X1",
            level="info",
            x=x,
            fx=f_curr,
            which="x1",
            tol_f=self.tol_f,
            tol_x=self.tol_x,
        )

        best_x: Optional[float] = x_prev
        best_fx: Optional[float] = f_prev
        if abs(f_curr) < abs(best_fx):
            best_x, best_fx = x, f_curr

        if abs(f_curr) <= self.tol_f:
            self._event("termination", k=1, code="TOL_F", level="info", reason="initial_residual", tol_f=self.tol_f)
            return SolverResult(
                method="secant",
                root=x,
                status="converged",
                stop_reason="TOL_F",
                message="Converged at initial guess by residual.",
                iterations=1,
                records=self.records,
                events=self.events,
                best_x=best_x,
                best_fx=best_fx,
                tol=self.tol_f,
                n_f=self.n_f,
                n_df=self.n_df,
            )

        # secant iterations (k >= 2)
        # ✅ FIX: ensure k never exceeds max_iter (prevents iterations=max_iter+1)
        for k in range(2, self.max_iter + 1):
            denom = f_curr - f_prev
            self._event(
                "secant_denom",
                k=k,
                code="SECANT_DENOM",
                level="info",
                denom=denom,
                f_curr=f_curr,
                f_prev=f_prev,
                denom_tol=self.denom_tol,
            )

            if abs(denom) < self.denom_tol:
                self._event(
                    "denom_too_small",
                    k=k,
                    code="DENOM_TOO_SMALL",
                    level="error",
                    denom=denom,
                    denom_tol=self.denom_tol,
                    x_prev=x_prev,
                    x=x,
                    f_prev=f_prev,
                    f_curr=f_curr,
                )
                self._event("termination", k=k, code="ERROR", level="error", reason="denom_too_small")
                return SolverResult(
                    method="secant",
                    root=x,
                    status="error",
                    stop_reason="ERROR",
                    message=f"Denominator too small (|f(x1)-f(x0)|<{self.denom_tol}).",
                    iterations=k - 1,
                    records=self.records,
                    events=self.events,
                    best_x=best_x,
                    best_fx=best_fx,
                    tol=self.tol_f,
                    n_f=self.n_f,
                    n_df=self.n_df,
                )

            x_new = x - f_curr * (x - x_prev) / denom
            dx = abs(x_new - x)
            self._event(
                "secant_step",
                k=k,
                code="SECANT_STEP",
                level="info",
                x_prev=x_prev,
                x=x,
                x_new=x_new,
                denom=denom,
                dx=dx,
            )

            # stagnation: no movement
            if dx < self.stagnation_tol:
                f_new = self._safe_eval(x_new)
                if f_new is None:
                    self._event("nonfinite", k=k, code="NONFINITE", level="error", x=x_new, where="f(x_new)_stagnation")
                    return SolverResult(
                        method="secant",
                        root=None,
                        status="nan_or_inf",
                        stop_reason="NAN_INF",
                        message="f(x_new) could not be evaluated (NaN/Inf/error).",
                        iterations=k - 1,
                        records=self.records,
                        events=self.events,
                        best_x=best_x,
                        best_fx=best_fx,
                        tol=self.tol_f,
                        n_f=self.n_f,
                        n_df=self.n_df,
                    )

                self._record(
                    k=k,
                    x=x_new,
                    fx=f_new,
                    x_prev=x,
                    step_type="secant",
                    reject_reason="stagnation",
                    meta={"denom": denom, "x_prev": x_prev, "f_prev": f_prev, "f_curr": f_curr},
                )
                self._event(
                    "stagnation",
                    k=k,
                    code="STAGNATION",
                    level="warn",
                    dx=dx,
                    stagnation_tol=self.stagnation_tol,
                )

                if best_fx is None or abs(f_new) < abs(best_fx):
                    best_x, best_fx = x_new, f_new

                return SolverResult(
                    method="secant",
                    root=x_new,
                    status="stagnation",
                    stop_reason="STAGNATION",
                    message=f"Stagnation detected (|Δx|<{self.stagnation_tol}).",
                    iterations=k,
                    records=self.records,
                    events=self.events,
                    best_x=best_x,
                    best_fx=best_fx,
                    tol=self.tol_f,
                    n_f=self.n_f,
                    n_df=self.n_df,
                )

            f_new = self._safe_eval(x_new)
            if f_new is None:
                self._event("nonfinite", k=k, code="NONFINITE", level="error", x=x_new, where="f(x_new)")
                return SolverResult(
                    method="secant",
                    root=None,
                    status="nan_or_inf",
                    stop_reason="NAN_INF",
                    message="f(x_new) could not be evaluated (NaN/Inf/error).",
                    iterations=k - 1,
                    records=self.records,
                    events=self.events,
                    best_x=best_x,
                    best_fx=best_fx,
                    tol=self.tol_f,
                    n_f=self.n_f,
                    n_df=self.n_df,
                )

            self._record(
                k=k,
                x=x_new,
                fx=f_new,
                x_prev=x,
                step_type="secant",
                meta={"denom": denom, "x_prev": x_prev, "f_prev": f_prev, "f_curr": f_curr},
            )

            if best_fx is None or abs(f_new) < abs(best_fx):
                best_x, best_fx = x_new, f_new

            # stopping criteria
            if abs(f_new) <= self.tol_f:
                self._event("termination", k=k, code="TOL_F", level="info", reason="residual", tol_f=self.tol_f)
                return SolverResult(
                    method="secant",
                    root=x_new,
                    status="converged",
                    stop_reason="TOL_F",
                    message="Converged by residual tolerance.",
                    iterations=k,
                    records=self.records,
                    events=self.events,
                    best_x=best_x,
                    best_fx=best_fx,
                    tol=self.tol_f,
                    n_f=self.n_f,
                    n_df=self.n_df,
                )

            # step tolerance is not convergence unless near root
            if dx <= self.tol_x:
                abs_fnew = abs(f_new)
                if abs_fnew <= 10.0 * self.tol_f:
                    self._event(
                        "termination",
                        k=k,
                        code="TOL_X_NEAR_ROOT",
                        level="info",
                        reason="step_near_root",
                        tol_x=self.tol_x,
                        dx=dx,
                        abs_f=abs_fnew,
                        tol_f=self.tol_f,
                    )
                    return SolverResult(
                        method="secant",
                        root=x_new,
                        status="converged",
                        stop_reason="TOL_X_NEAR_ROOT",
                        message="Converged by step tolerance (near-root guard satisfied).",
                        iterations=k,
                        records=self.records,
                        events=self.events,
                        best_x=best_x,
                        best_fx=best_fx,
                        tol=self.tol_f,
                        n_f=self.n_f,
                        n_df=self.n_df,
                    )
                else:
                    self._event(
                        "step_small_but_residual_large",
                        k=k,
                        code="STEP_SMALL_RESIDUAL_LARGE",
                        level="warn",
                        tol_x=self.tol_x,
                        dx=dx,
                        abs_f=abs_fnew,
                        tol_f=self.tol_f,
                        note="Step size small but residual is not; continuing (prevents false convergence).",
                    )

            # shift
            x_prev, f_prev = x, f_curr
            x, f_curr = x_new, f_new

        # max iterations reached (no convergence)
        last_x = self.records[-1].x if self.records else None
        last_fx = self.records[-1].fx if self.records else None
        if best_x is None:
            best_x, best_fx = last_x, last_fx

        # ✅ FIX: terminate at k=max_iter (not max_iter+1) and cap iterations to max_iter
        self._event(
            "termination",
            k=self.max_iter,
            code="MAX_ITER",
            level="warn",
            reason="max_iter_reached",
            max_iter=self.max_iter,
        )
        return SolverResult(
            method="secant",
            root=last_x,
            status="max_iter",
            stop_reason="MAX_ITER",
            message="Maximum iterations reached without convergence.",
            iterations=self.max_iter,
            records=self.records,
            events=self.events,
            best_x=best_x,
            best_fx=best_fx,
            tol=self.tol_f,
            n_f=self.n_f,
            n_df=self.n_df,
        )