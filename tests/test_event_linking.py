from numerical_lab.methods.hybrid import HybridBisectionNewtonSolver


def f(x): return x**3 - x - 2
def df(x): return 3*x**2 - 1


def test_event_k_links_to_records():
    res = HybridBisectionNewtonSolver(f, df, a=1.0, b=2.0, tol=1e-12, max_iter=60).solve()
    ks = {r.k for r in res.records}

    # Events with "k" must refer to an existing record iteration index.
    for e in res.events:
        if "k" in e:
            assert e["k"] in ks
        assert "kind" in e and "data" in e