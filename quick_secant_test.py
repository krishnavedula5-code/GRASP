from numerical_lab.methods.secant import SecantSolver

def f(x):
    return x**3 - x - 2

solver = SecantSolver(f, x0=1.0, x1=2.0, tol=1e-12, max_iter=50)
res = solver.solve()

print("status:", res.status)
print("root:", res.root)
print("iters:", res.iterations)
print("last residual:", res.residual_history[-1] if res.residual_history else None)