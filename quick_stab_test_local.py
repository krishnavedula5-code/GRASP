from numerical_lab.engine.controller import NumericalEngine

def f(x):
    return x**3 - x - 2

def df(x):
    return 3*x**2 - 1

res, conv, stab = NumericalEngine.solve_newton(f, df, 1.5)
print(stab.label)