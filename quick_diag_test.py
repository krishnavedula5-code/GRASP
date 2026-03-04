from numerical_lab.methods.newton import NewtonSolver
from numerical_lab.diagnostics.convergence import classify_convergence

def f(x):
    return x**3 - x - 2

def df(x):
    return 3*x**2 - 1

res = NewtonSolver(f, df, x0=1.5, tol=1e-12, max_iter=50).solve()
rep = classify_convergence(res)

print("status:", res.status)
print("observed_order:", rep.observed_order)
print("class:", rep.classification)
print("notes:", " | ".join(rep.notes))