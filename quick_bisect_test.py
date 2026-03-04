from numerical_lab.methods.bisection import BisectionSolver

def f(x):
    return x**3 - x - 2

solver = BisectionSolver(f, a=1, b=2, tol=1e-10, max_iter=200)
res = solver.solve()

print("status:", res.status)
print("root:", res.root)
print("iters:", res.iterations)
print("last residual:", res.residual_history[-1] if res.residual_history else None)