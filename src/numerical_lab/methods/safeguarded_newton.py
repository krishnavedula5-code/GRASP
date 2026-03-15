from __future__ import annotations

import math
from typing import Callable, Optional
from numerical_lab.core.base import RootSolver, SolverResult


class SafeguardedNewtonSolver(RootSolver):
    """
    Safeguarded Newton method.

    Uses Newton step when safe.
    Falls back to bisection if the Newton step exits the bracket
    or derivative becomes unstable.
    """

    def __init__(
        self,
        f: Callable[[float], float],
        df: Optional[Callable[[float], float]],
        a: float,
        b: float,
        x0: Optional[float] = None,
        tol: float = 1e-10,
        max_iter: int = 100,
        numerical_derivative: bool = False,
        df_tol: float = 1e-14,
    ):
        super().__init__(f=f, tol=tol, max_iter=max_iter)

        self.df = df
        self.numerical_derivative = bool(numerical_derivative)
        self.df_tol = float(df_tol)

        self.a = float(a)
        self.b = float(b)

        if x0 is None:
            x0 = 0.5 * (self.a + self.b)

        # Keep initial guess inside the bracket
        self.x0 = min(max(float(x0), self.a), self.b)

    def _safe_eval_df(self, x: float) -> float:
        try:
            if self.df is None:
                return float("nan")
            val = float(self.df(x))
            self.n_df += 1
            if not math.isfinite(val):
                return float("nan")
            return val
        except Exception:
            return float("nan")

    def _numerical_df(self, x: float) -> tuple[float, float, float, float]:
        """
        Central difference derivative with scale-aware step.
        Returns: (dfx, h, f(x+h), f(x-h))
        """
        h = 1e-6 * max(1.0, abs(x))
        fxph = self._safe_eval(x + h)
        fxmh = self._safe_eval(x - h)
        self.n_df += 1
        if fxph is None or fxmh is None:
            return (float("nan"), h, float("nan"), float("nan"))
        return ((fxph - fxmh) / (2.0 * h), h, fxph, fxmh)

    def solve(self) -> SolverResult:
        a = self.a
        b = self.b
        x = self.x0

        x_prev = None

        fa = self._safe_eval(a)
        fb = self._safe_eval(b)

        if fa is None or fb is None:
            return SolverResult(
                method="safeguarded_newton",
                root=None,
                status="nan_or_inf",
                message="Invalid function values at bracket",
                iterations=0,
                records=self.records,
                events=self.events,
                stop_reason="NAN_INF",
                tol=self.tol,
                n_f=self.n_f,
                n_df=self.n_df,
            )

        if fa * fb > 0:
            self._event(
                "bad_bracket",
                k=0,
                code="BAD_BRACKET",
                level="error",
                a=a,
                b=b,
                fa=fa,
                fb=fb,
            )
            return SolverResult(
                method="safeguarded_newton",
                root=None,
                status="bad_bracket",
                message="Bracket does not contain a sign change.",
                iterations=0,
                records=self.records,
                events=self.events,
                stop_reason="BAD_BRACKET",
                tol=self.tol,
                n_f=self.n_f,
                n_df=self.n_df,
            )

        best_x = x
        best_fx = None

        for k in range(1, self.max_iter + 1):
            fx = self._safe_eval(x)
            if fx is None:
                return SolverResult(
                    method="safeguarded_newton",
                    root=None,
                    status="nan_or_inf",
                    message="Function evaluation returned non-finite",
                    iterations=k,
                    records=self.records,
                    events=self.events,
                    stop_reason="NAN_INF",
                    tol=self.tol,
                    best_x=best_x,
                    best_fx=best_fx,
                    n_f=self.n_f,
                    n_df=self.n_df,
                )

            if best_fx is None or abs(fx) < abs(best_fx):
                best_x = x
                best_fx = fx

            # convergence test
            if abs(fx) < self.tol:
                self._record(
                    k=k,
                    x=x,
                    fx=fx,
                    x_prev=x_prev,
                    step_type="accepted",
                    a=a,
                    b=b,
                    meta={
                        "used_bisection": 0,
                        "used_newton": 0,
                    },
                )
                return SolverResult(
                    method="safeguarded_newton",
                    root=x,
                    status="converged",
                    message="Converged (|f(x)| < tol)",
                    iterations=k,
                    records=self.records,
                    events=self.events,
                    stop_reason="TOL_F",
                    tol=self.tol,
                    best_x=x,
                    best_fx=fx,
                    n_f=self.n_f,
                    n_df=self.n_df,
                )

            use_num = self.numerical_derivative or (self.df is None)

            if use_num:
                dfx, h, fp, fm = self._numerical_df(x)
                self._event(
                    "num_deriv",
                    k=k,
                    code="NUM_DERIV",
                    level="info",
                    x=x,
                    h=h,
                    fp=fp,
                    fm=fm,
                    dfx=dfx,
                )
            else:
                dfx = self._safe_eval_df(x)

            step_type = "newton"
            x_new = None

            if dfx is None or not math.isfinite(dfx):
                step_type = "bisection"
                self._event(
                    "nonfinite",
                    k=k,
                    code="NONFINITE",
                    level="warn",
                    x=x,
                    where="df(x)",
                )
            elif abs(dfx) < self.df_tol:
                step_type = "bisection"
                self._event(
                    "derivative_zero",
                    k=k,
                    code="DERIVATIVE_ZERO",
                    level="warn",
                    x=x,
                    dfx=dfx,
                    df_tol=self.df_tol,
                )
            else:
                candidate = x - fx / dfx

                if candidate < a or candidate > b:
                    step_type = "bisection"
                    self._event(
                        "newton_outside_bracket",
                        k=k,
                        code="NEWTON_OUTSIDE_BRACKET",
                        level="info",
                        x=x,
                        fx=fx,
                        dfx=dfx,
                        candidate=candidate,
                        a=a,
                        b=b,
                    )
                else:
                    x_new = candidate

            if x_new is None:
                x_new = 0.5 * (a + b)
                self._event(
                    "safeguard_bisection",
                    k=k,
                    code="SAFEGUARD_BISECTION",
                    level="info",
                    x=x,
                    x_new=x_new,
                    a=a,
                    b=b,
                )

            fx_new = self._safe_eval(x_new)

            if fx_new is None:
                return SolverResult(
                    method="safeguarded_newton",
                    root=None,
                    status="nan_or_inf",
                    message="Non-finite during update",
                    iterations=k,
                    records=self.records,
                    events=self.events,
                    stop_reason="NAN_INF",
                    tol=self.tol,
                    best_x=best_x,
                    best_fx=best_fx,
                    n_f=self.n_f,
                    n_df=self.n_df,
                )

            if abs(fx_new) < abs(best_fx) if best_fx is not None else True:
                best_x = x_new
                best_fx = fx_new

            self._record(
                k=k,
                x=x_new,
                fx=fx_new,
                x_prev=x,
                step_type=step_type,
                a=a,
                b=b,
                meta={
                    "used_bisection": 1 if step_type == "bisection" else 0,
                    "used_newton": 1 if step_type == "newton" else 0,
                },
            )

            if fa * fx_new <= 0:
                b = x_new
                fb = fx_new
            else:
                a = x_new
                fa = fx_new

            x_prev = x
            x = x_new

        return SolverResult(
            method="safeguarded_newton",
            root=x,
            status="max_iter",
            message="Maximum iterations reached",
            iterations=self.max_iter,
            records=self.records,
            events=self.events,
            stop_reason="MAX_ITER",
            tol=self.tol,
            best_x=best_x,
            best_fx=best_fx,
            n_f=self.n_f,
            n_df=self.n_df,
        )