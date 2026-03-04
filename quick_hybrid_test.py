from numerical_lab.engine.controller import NumericalEngine

def f(x):
    return x**3 - x - 2

def df(x):
    return 3*x**2 - 1

res, rep, stab = NumericalEngine.solve_hybrid(f, df, 1, 2, tol=1e-12, max_iter=100)
print(res.iterations)