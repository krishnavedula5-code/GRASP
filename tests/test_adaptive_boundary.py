from numerical_lab.engine.controller import NumericalEngine
from numerical_lab.diagnostics.adaptive_boundaries import _run_solver_at_x, refine_interval

def f(x):
    return (x - 1)**2 * (x + 2)

def df(x):
    return 2*(x - 1)*(x + 2) + (x - 1)**2

engine = NumericalEngine()

print("=== single run test ===")
res = _run_solver_at_x(
    engine=engine,
    f=f,
    df=df,
    method="newton",
    x=-1.0,
    domain=(-4, 4),
    tol=1e-10,
    max_iter=100,
)
print(res)

print("\n=== refinement test ===")
rb = refine_interval(
    engine=engine,
    f=f,
    df=df,
    method="newton",
    domain=(-4, 4),
    left_x=-1.05,
    right_x=-0.95,
    left_label="root:-2.0",
    right_label="root:1.0",
    reason="label_change",
    tol_x=1e-4,
    max_depth=12,
    root_digits=8,
    solve_tol=1e-10,
    max_iter=100,
)
print(rb)