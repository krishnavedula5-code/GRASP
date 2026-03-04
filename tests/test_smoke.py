from numerical_lab.methods.bisection import BisectionSolver
from numerical_lab.methods.newton import NewtonSolver
from numerical_lab.methods.secant import SecantSolver
from numerical_lab.methods.hybrid import HybridBisectionNewtonSolver


def f(x): return x**3 - x - 2
def df(x): return 3*x**2 - 1


def test_bisection_smoke():
    res = BisectionSolver(f, a=1.0, b=2.0, tol=1e-12, max_iter=200).solve()
    assert res.status == "converged"
    assert res.stop_reason in {"TOL_F", "TOL_BRACKET", "EXACT_ROOT"}
    assert res.n_f > 0
    assert res.n_df == 0
    assert len(res.records) > 0


def test_newton_smoke():
    res = NewtonSolver(f, df, x0=1.5, tol=1e-12, max_iter=50).solve()
    assert res.status == "converged"
    assert res.stop_reason in {"TOL_F", "TOL_X", "EXACT_ROOT"}
    assert res.n_f > 0
    assert res.n_df > 0
    assert any(r.step_type == "newton" for r in res.records)


def test_secant_smoke():
    res = SecantSolver(f, x0=1.0, x1=2.0, tol=1e-12, max_iter=80).solve()
    assert res.status == "converged"
    assert res.stop_reason in {"TOL_F", "TOL_X", "EXACT_ROOT"}
    assert res.n_f > 0
    assert res.n_df == 0
    assert any(r.step_type == "secant" for r in res.records)


def test_hybrid_smoke():
    res = HybridBisectionNewtonSolver(f, df, a=1.0, b=2.0, tol=1e-12, max_iter=100).solve()
    assert res.status == "converged"
    assert res.stop_reason in {"TOL_F", "TOL_BRACKET", "EXACT_ROOT"}
    assert res.n_f > 0
    assert res.n_df > 0
    assert any(r.step_type in {"newton", "bisection"} for r in res.records)