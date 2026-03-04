from numerical_lab.engine.controller import NumericalEngine
from numerical_lab.engine.summary import build_comparison_summary

def f(x):
    return x**3 - x - 2

def df(x):
    return 3*x**2 - 1

comp = NumericalEngine.compare_methods(
    f=f,
    df=df,
    bracket=(1.0, 2.0),
    secant_guesses=(1.0, 2.0),
    tol=1e-12,
    max_iter=100
)

summ = build_comparison_summary(comp)

for m in summ:
    print(m, summ[m].iterations)