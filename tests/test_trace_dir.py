from pathlib import Path

from numerical_lab.engine.controller import NumericalEngine
from numerical_lab.engine.trace_dir import export_compare_traces_json_dir


def f(x):
    return x**3 - x - 2

def df(x):
    return 3*x**2 - 1


def test_export_compare_traces_json_dir(tmp_path: Path):
    comp = NumericalEngine.compare_methods(
        f=f, df=df, bracket=(1.0, 2.0), secant_guesses=(1.0, 2.0), tol=1e-10, max_iter=100
    )
    out_dir = tmp_path / "traces"
    export_compare_traces_json_dir(comp, str(out_dir))

    for name in ["bisection", "newton", "secant", "hybrid"]:
        assert (out_dir / f"{name}.json").exists()