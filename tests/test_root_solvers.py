import math

from numerical_lab.methods.bisection import BisectionSolver
from numerical_lab.methods.newton import NewtonSolver
from numerical_lab.methods.secant import SecantSolver
from numerical_lab.methods.hybrid import HybridBisectionNewtonSolver


def f(x):
    return x**3 - x - 2


def df(x):
    return 3 * x**2 - 1


TRUE_ROOT = 1.5213797068045676  # reference value (high precision)


def assert_close(x, y, tol=1e-10):
    assert x is not None
    assert abs(x - y) < tol


def test_bisection_converges():
    res = BisectionSolver(f, 1.0, 2.0, tol=1e-12, max_iter=200).solve()
    assert res.status == "converged"
    assert_close(res.root, TRUE_ROOT, tol=1e-9)
    assert len(res.records) > 0


def test_newton_converges():
    res = NewtonSolver(f, df, 1.5, tol=1e-12, max_iter=50).solve()
    assert res.status == "converged"
    assert_close(res.root, TRUE_ROOT, tol=1e-9)
    assert len(res.records) > 0


def test_secant_converges():
    res = SecantSolver(f, 1.0, 2.0, tol=1e-12, max_iter=100).solve()
    assert res.status == "converged"
    assert_close(res.root, TRUE_ROOT, tol=1e-9)
    assert len(res.records) > 0


def test_hybrid_converges():
    res = HybridBisectionNewtonSolver(f, df, 1.0, 2.0, tol=1e-12, max_iter=100).solve()
    assert res.status == "converged"
    assert_close(res.root, TRUE_ROOT, tol=1e-9)
    assert len(res.records) > 0


def test_bisection_bad_bracket():
    res = BisectionSolver(f, 2.0, 3.0, tol=1e-10, max_iter=50).solve()
    assert res.status == "bad_bracket"
    assert res.root is None