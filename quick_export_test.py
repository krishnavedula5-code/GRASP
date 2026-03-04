from numerical_lab.methods.bisection import BisectionSolver
from numerical_lab.engine.export import export_iterations_csv

def f(x):
    return x**3 - x - 2

res = BisectionSolver(f, 1, 2, tol=1e-10).solve()
export_iterations_csv(res, "bisection_run.csv")

print("ok")