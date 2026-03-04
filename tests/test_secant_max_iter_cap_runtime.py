def test_secant_runtime_iters_never_exceed_max_iter():
    from numerical_lab.methods.secant import SecantSolver  # adjust import if needed

    def f(x): 
        return x**3 - 2*x + 2

    max_iter = 10
    res = SecantSolver(f=f, x0=-4.0, x1=-3.9, tol=1e-10, max_iter=max_iter).solve()

    assert res.iterations <= max_iter
    if res.status == "max_iter":
        assert res.iterations == max_iter