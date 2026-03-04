from numerical_lab.methods.newton import NewtonSolver

def f(x):
    return x**3 - x - 2

def df(x):
    return 3*x**2 - 1

solver = NewtonSolver(f, df, x0=1.5, tol=1e-12, max_iter=50)
res = solver.solve()

print("status:", res.status)
print("root:", res.root)
print("iters:", res.iterations)
print("last residual:", res.residual_history[-1] if res.residual_history else None)