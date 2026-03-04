import json
from pathlib import Path

from numerical_lab.engine.controller import NumericalEngine
from numerical_lab.engine.trace_dir import export_compare_traces_json_dir


def f(x): return x**3 - x - 2
def df(x): return 3*x**2 - 1


def test_compare_dir_traces_have_explanations(tmp_path: Path):
    comp = NumericalEngine.compare_methods(
        f=f, df=df, bracket=(1.0, 2.0), secant_guesses=(1.0, 2.0), tol=1e-10, max_iter=100
    )
    out_dir = tmp_path / "traces"
    export_compare_traces_json_dir(comp, str(out_dir))

    data = json.loads((out_dir / "hybrid.json").read_text(encoding="utf-8"))
    assert "extra" in data
    assert "explanation" in data["extra"]
    assert len(data["extra"]["explanation"]) > 20