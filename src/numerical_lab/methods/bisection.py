from __future__ import annotations

from typing import Callable, Optional

from numerical_lab.core.base import RootSolver, SolverResult


class BisectionSolver(RootSolver):
    """
    Bisection method for f(x)=0 on [a,b] with f(a)*f(b)<0.

    Commercial requirements:
    - Records every iterate
    - Returns structured SolverResult
    - Uses rigorous interval error bound for stopping: (b-a)/2 < tol
    - Teaching trace: step_type + bracket width + midpoint context
    """

    def __init__(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        tol: float = 1e-8,
        max_iter: int = 100,
    ):
        super().__init__(f=f, tol=tol, max_iter=max_iter)
        self.a = float(a)
        self.b = float(b)

        # ✅ Split tolerances (UI still supplies one tol)
        self.tol_f = float(tol)
        self.tol_bracket = float(tol)

    def solve(self) -> SolverResult:
        a, b = self.a, self.b

        fa = self._safe_eval(a)
        fb = self._safe_eval(b)

        if fa is None or fb is None:
            # If a DOMAIN_ERROR event already happened, prefer DOMAIN_ERROR stop reason
            ev_list = getattr(self, "_events", None) or getattr(self, "events", None) or []
            has_domain = any(getattr(e, "code", None) == "DOMAIN_ERROR" for e in ev_list)
            self._event(
                "nonfinite",
                k=0,
                code="NONFINITE",
                level="error",
                a=a,
                b=b,
                fa=fa,
                fb=fb,
            )
            return SolverResult(
                method="bisection",
                root=None,
                status="nan_or_inf",
                stop_reason=("DOMAIN_ERROR" if has_domain else "NAN_INF"),
                message="f(a) or f(b) could not be evaluated (NaN/Inf/error).",
                iterations=0,
                records=[],
                events=ev_list,
                best_x=None,
                best_fx=None,
                tol=self.tol_f,
                n_f=self.n_f,
                n_df=self.n_df,
            )

        if fa == 0.0:
            self._record(
                k=0,
                x=a,
                fx=fa,
                x_prev=None,
                step_type="exact",
                a=a,
                b=b,
                meta={"endpoint": "a"},
            )
            self._event("exact_root", k=0, code="EXACT_ROOT", level="info", x=a, fx=fa)
            self._event("termination", k=0, code="EXACT_ROOT", level="info", reason="endpoint_a")
            return SolverResult(
                method="bisection",
                root=a,
                status="converged",
                stop_reason="EXACT_ROOT",
                message="Exact root at a.",
                iterations=0,
                records=self.records,
                events=self.events,
                best_x=a,
                best_fx=fa,
                tol=self.tol_f,
                n_f=self.n_f,
                n_df=self.n_df,
            )

        if fb == 0.0:
            self._record(
                k=0,
                x=b,
                fx=fb,
                x_prev=None,
                step_type="exact",
                a=a,
                b=b,
                meta={"endpoint": "b"},
            )
            self._event("exact_root", k=0, code="EXACT_ROOT", level="info", x=b, fx=fb)
            self._event("termination", k=0, code="EXACT_ROOT", level="info", reason="endpoint_b")
            return SolverResult(
                method="bisection",
                root=b,
                status="converged",
                stop_reason="EXACT_ROOT",
                message="Exact root at b.",
                iterations=0,
                records=self.records,
                events=self.events,
                best_x=b,
                best_fx=fb,
                tol=self.tol_f,
                n_f=self.n_f,
                n_df=self.n_df,
            )

        if fa * fb > 0:
            self._event(
                "invalid_bracket",
                k=0,
                code="BAD_BRACKET",
                level="error",
                a=a,
                b=b,
                fa=fa,
                fb=fb,
            )
            # explicit termination event helps the UI timeline
            self._event("termination", k=0, code="BAD_BRACKET", level="error", reason="same_sign_endpoints")
            return SolverResult(
                method="bisection",
                root=None,
                status="bad_bracket",
                stop_reason="BAD_BRACKET",
                message="Invalid bracket: f(a) and f(b) must have opposite signs.",
                iterations=0,
                records=[],
                events=self.events,
                best_x=None,
                best_fx=None,
                tol=self.tol_f,
                n_f=self.n_f,
                n_df=self.n_df,
            )

        x_prev: Optional[float] = None
        best_x: Optional[float] = None
        best_fx: Optional[float] = None

        self._event(
            "init_bracket",
            k=0,
            code="INIT_BRACKET",
            level="info",
            a=a,
            b=b,
            fa=fa,
            fb=fb,
            tol_f=self.tol_f,
            tol_bracket=self.tol_bracket,
        )

        for k in range(1, self.max_iter + 1):
            c = (a + b) / 2.0
            fc = self._safe_eval(c)

            if fc is None:
                self._event(
                    "nonfinite",
                    k=k,
                    code="NONFINITE",
                    level="error",
                    m=c,
                    where="f(midpoint)",
                )
                return SolverResult(
                    method="bisection",
                    root=None,
                    status="nan_or_inf",
                    stop_reason="NAN_INF",
                    message="f(c) could not be evaluated (NaN/Inf/error).",
                    iterations=k - 1,
                    records=self.records,
                    events=self.events,
                    best_x=best_x,
                    best_fx=best_fx,
                    tol=self.tol_f,
                    n_f=self.n_f,
                    n_df=self.n_df,
                )

            interval_err = (b - a) / 2.0

            self._event(
                "midpoint",
                k=k,
                code="MIDPOINT",
                level="info",
                a=a,
                b=b,
                m=c,
                fm=fc,
                interval_err=interval_err,
            )

            self._record(
                k=k,
                x=c,
                fx=fc,
                x_prev=x_prev,
                step_type="bisection",
                a=a,
                b=b,
                m=c,
                fm=fc,
            )

            x_prev = c

            if best_fx is None or abs(fc) < abs(best_fx):
                best_x, best_fx = c, fc

            abs_fc = abs(fc)

            if abs_fc <= self.tol_f:
                self._event(
                    "termination",
                    k=k,
                    code="TOL_F",
                    level="info",
                    reason="residual",
                    abs_fc=abs_fc,
                    tol_f=self.tol_f,
                )
                return SolverResult(
                    method="bisection",
                    root=c,
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

            if interval_err <= self.tol_bracket:
                self._event(
                    "termination",
                    k=k,
                    code="TOL_BRACKET",
                    level="info",
                    reason="interval",
                    interval_err=interval_err,
                    tol_bracket=self.tol_bracket,
                )
                return SolverResult(
                    method="bisection",
                    root=c,
                    status="converged",
                    stop_reason="TOL_BRACKET",
                    message="Converged by interval tolerance.",
                    iterations=k,
                    records=self.records,
                    events=self.events,
                    best_x=best_x,
                    best_fx=best_fx,
                    tol=self.tol_f,
                    n_f=self.n_f,
                    n_df=self.n_df,
                )

            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc

            self._event(
                "bracket_update",
                k=k,
                code="BRACKET_UPDATE",
                level="info",
                a=a,
                b=b,
                fa=fa,
                fb=fb,
            )

        last_x = self.records[-1].x if self.records else None
        last_fx = self.records[-1].fx if self.records else None
        if best_x is None:
            best_x, best_fx = last_x, last_fx

        self._event(
            "termination",
            k=self.max_iter,
            code="MAX_ITER",
            level="warn",
            reason="max_iter_exceeded",
            max_iter=self.max_iter,
        )

        return SolverResult(
            method="bisection",
            root=last_x,
            status="max_iter",
            stop_reason="MAX_ITER",
            message="Maximum iterations exceeded without convergence.",
            iterations=self.max_iter,
            records=self.records,
            events=self.events,
            best_x=best_x,
            best_fx=best_fx,
            tol=self.tol_f,
            n_f=self.n_f,
            n_df=self.n_df,
        )