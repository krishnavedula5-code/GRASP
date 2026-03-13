from __future__ import annotations

import math
from typing import Callable, Optional

from numerical_lab.core.base import RootSolver, SolverResult


class BrentSolver(RootSolver):
    """
    Brent's method for f(x)=0 on [a,b] with f(a)*f(b)<0.

    Features
    --------
    - Bracketing safety
    - Secant / inverse quadratic interpolation acceleration
    - Bisection fallback when interpolation is unsafe
    - Structured teaching/research trace compatible with existing framework
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

        # Residual tolerance and x/bracket tolerance
        self.tol_f = float(tol)
        self.tol_x = float(tol)

    def solve(self) -> SolverResult:
        a, b = self.a, self.b

        fa = self._safe_eval(a)
        fb = self._safe_eval(b)

        if fa is None or fb is None:
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
                method="brent",
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
                method="brent",
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
                method="brent",
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
            self._event(
                "termination",
                k=0,
                code="BAD_BRACKET",
                level="error",
                reason="same_sign_endpoints",
            )

            return SolverResult(
                method="brent",
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

        # Standard Brent state
        c = a
        fc = fa
        d = b - a
        e = d

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
            tol_x=self.tol_x,
        )

        eps = 2.220446049250313e-16  # machine epsilon for float64

        for k in range(1, self.max_iter + 1):
            # Maintain the invariant that fb and fc have opposite signs
            if fb * fc > 0:
                c = a
                fc = fa
                d = b - a
                e = d

            # Ensure |fb| <= |fc|
            if abs(fc) < abs(fb):
                a, b, c = b, c, b
                fa, fb, fc = fb, fc, fb

            tol_step = 2.0 * eps * abs(b) + 0.5 * self.tol_x
            xm = 0.5 * (c - b)

            if best_fx is None or abs(fb) < abs(best_fx):
                best_x, best_fx = b, fb

            # Record current iterate
            self._record(
                k=k,
                x=b,
                fx=fb,
                x_prev=x_prev,
                step_type="brent",
                a=min(b, c),
                b=max(b, c),
                m=b,
                fm=fb,
                meta={
                    "bracket_halfwidth": abs(xm),
                    "tol_step": tol_step,
                },
            )
            x_prev = b

            # Residual convergence
            if abs(fb) <= self.tol_f:
                self._event(
                    "termination",
                    k=k,
                    code="TOL_F",
                    level="info",
                    reason="residual",
                    abs_fb=abs(fb),
                    tol_f=self.tol_f,
                )
                return SolverResult(
                    method="brent",
                    root=b,
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

            # Bracket-width convergence
            if abs(xm) <= tol_step:
                self._event(
                    "termination",
                    k=k,
                    code="TOL_X",
                    level="info",
                    reason="bracket_width",
                    bracket_halfwidth=abs(xm),
                    tol_x=tol_step,
                )
                return SolverResult(
                    method="brent",
                    root=b,
                    status="converged",
                    stop_reason="TOL_X",
                    message="Converged by bracket width tolerance.",
                    iterations=k,
                    records=self.records,
                    events=self.events,
                    best_x=best_x,
                    best_fx=best_fx,
                    tol=self.tol_f,
                    n_f=self.n_f,
                    n_df=self.n_df,
                )

            step_type = "bisection"

            # Attempt interpolation only if safe
            if abs(e) >= tol_step and abs(fa) > abs(fb):
                s = fb / fa

                if a == c:
                    # Secant step
                    p = 2.0 * xm * s
                    q = 1.0 - s
                    candidate_step_type = "secant"
                else:
                    # Inverse quadratic interpolation
                    q_ = fa / fc
                    r = fb / fc
                    p = s * (2.0 * xm * q_ * (q_ - r) - (b - a) * (r - 1.0))
                    q = (q_ - 1.0) * (r - 1.0) * (s - 1.0)
                    candidate_step_type = "iqi"

                if p > 0:
                    q = -q
                else:
                    p = -p

                if q != 0.0 and (2.0 * p < min(3.0 * xm * q - abs(tol_step * q), abs(e * q))):
                    e = d
                    d = p / q
                    step_type = candidate_step_type
                else:
                    d = xm
                    e = d
                    step_type = "bisection"
            else:
                d = xm
                e = d
                step_type = "bisection"

            # Advance
            a = b
            fa = fb

            if abs(d) > tol_step:
                b = b + d
            else:
                b = b + math.copysign(tol_step, xm)

            fb = self._safe_eval(b)

            if fb is None:
                self._event(
                    "nonfinite",
                    k=k,
                    code="NONFINITE",
                    level="error",
                    where="candidate_eval",
                    x_new=b,
                )
                return SolverResult(
                    method="brent",
                    root=None,
                    status="nan_or_inf",
                    stop_reason="NAN_INF",
                    message="Candidate evaluation produced NaN/Inf/error.",
                    iterations=k - 1,
                    records=self.records,
                    events=self.events,
                    best_x=best_x,
                    best_fx=best_fx,
                    tol=self.tol_f,
                    n_f=self.n_f,
                    n_df=self.n_df,
                )

            self._event(
                "candidate_step",
                k=k,
                code="STEP",
                level="info",
                step_type=step_type,
                x_new=b,
                fx_new=fb,
                a=a,
                c=c,
            )

            self._event(
                "bracket_update",
                k=k,
                code="BRACKET_UPDATE",
                level="info",
                a=a,
                b=b,
                c=c,
                fa=fa,
                fb=fb,
                fc=fc,
                step_type=step_type,
            )

        last_x = self.records[-1].x if self.records else b
        last_fx = self.records[-1].fx if self.records else fb

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
            method="brent",
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