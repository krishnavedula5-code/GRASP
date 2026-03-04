from numerical_lab.methods.hybrid import HybridBisectionNewtonSolver
from numerical_lab.methods.newton import NewtonSolver


def test_hybrid_bad_bracket():
    def f(x): return x**2 + 1  # no real root
    def df(x): return 2*x

    res = HybridBisectionNewtonSolver(f, df, a=-1.0, b=1.0, tol=1e-12, max_iter=10).solve()
    assert res.status == "bad_bracket"
    assert res.stop_reason == "BAD_BRACKET"


def test_newton_derivative_zero_detected():
    def f(x): return 1.0
    def df(x): return 0.0

    res = NewtonSolver(f, df, x0=0.0, tol=1e-12, max_iter=5, df_tol=1e-14).solve()
    assert res.status == "derivative_zero"
    assert res.stop_reason == "DERIVATIVE_ZERO"