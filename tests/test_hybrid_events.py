from numerical_lab.methods.hybrid import HybridBisectionNewtonSolver

def f(x): return x**3 - x - 2
def df(x): return 3*x**2 - 1

def test_hybrid_emits_events():
    res = HybridBisectionNewtonSolver(f, df, 1.0, 2.0, tol=1e-10, max_iter=50).solve()
    assert isinstance(res.events, list)
    assert len(res.events) > 0
    kinds = {e.get("kind") for e in res.events}
    assert "midpoint" in kinds
    assert ("newton_attempt" in kinds) or ("newton_reject" in kinds)