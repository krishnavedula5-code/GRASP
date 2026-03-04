from __future__ import annotations

from typing import Callable, Dict

from numerical_lab.methods.hybrid import HybridBisectionNewtonSolver
from numerical_lab.diagnostics.stability import detect_stability
from numerical_lab.methods.bisection import BisectionSolver
from numerical_lab.methods.newton import NewtonSolver
from numerical_lab.methods.secant import SecantSolver

from numerical_lab.diagnostics.convergence import classify_convergence


class NumericalEngine:
    """
    Commercial orchestration layer.

    Responsibilities:
    - Run solvers
    - Return structured results
    - Attach diagnostics
    - Support comparison mode
    """

    @staticmethod
    def solve_bisection(f: Callable[[float], float], a: float, b: float, **kwargs):
        solver = BisectionSolver(f, a, b, **kwargs)
        result = solver.solve()
        report = classify_convergence(result)
        stab = detect_stability(result)
        return result, report, stab

    @staticmethod
    def solve_newton(f, df, x0, **kwargs):
        # numerical_derivative is a controller/API concern; solvers shouldn't receive it
        kwargs.pop("numerical_derivative", None)
        solver = NewtonSolver(f, df, x0, **kwargs)
        result = solver.solve()
        report = classify_convergence(result)
        stab = detect_stability(result)
        return result, report, stab

    @staticmethod
    def solve_secant(f, x0, x1, **kwargs):
        solver = SecantSolver(f, x0, x1, **kwargs)
        result = solver.solve()
        report = classify_convergence(result)
        stab = detect_stability(result)
        return result, report, stab
    
    @staticmethod
    def solve_hybrid(f, df, a, b, **kwargs):
        # numerical_derivative is a controller/API concern; solvers shouldn't receive it
        kwargs.pop("numerical_derivative", None)
        solver = HybridBisectionNewtonSolver(f, df, a, b, **kwargs)
        result = solver.solve()
        report = classify_convergence(result)
        stab = detect_stability(result)
        return result, report, stab
    


    @staticmethod
    def compare_methods(
        f,
        df,
        bracket,
        secant_guesses,
        **kwargs
    ) -> Dict[str, object]:
        """
        Run all methods for comparison mode.
        Core feature for commercial differentiation.

        Notes:
        - Newton init: prefer explicit newton_x0 passed via kwargs; else use bracket midpoint.
        - Secant uses secant_guesses (x0, x1) exactly.
        """

        a, b = bracket
        x0, x1 = secant_guesses

        results = {}
        

        # --- Newton initialization policy ---
        # Allow API layer to pass newton_x0 explicitly; otherwise use midpoint of bracket.
        # (We use pop so NewtonSolver doesn't receive an unexpected keyword argument.)
        newton_x0 = kwargs.pop("newton_x0", None)
        if newton_x0 is None:
            newton_x0 = (a + b) / 2.0
        else:
            newton_x0 = float(newton_x0)

        bis_res, bis_rep, bis_stab = NumericalEngine.solve_bisection(f, a, b, **kwargs)
        new_res, new_rep, new_stab = NumericalEngine.solve_newton(f, df, newton_x0,**kwargs)
        sec_res, sec_rep, sec_stab = NumericalEngine.solve_secant(f, x0, x1, **kwargs)
        hyb_res, hyb_conv, hyb_stab = NumericalEngine.solve_hybrid(f, df, a, b,  **kwargs)

        results["bisection"] = (bis_res, bis_rep, bis_stab)
        results["newton"] = (new_res, new_rep, new_stab)
        results["secant"] = (sec_res, sec_rep, sec_stab)
        results["hybrid"] = (hyb_res, hyb_conv, hyb_stab)

        return results