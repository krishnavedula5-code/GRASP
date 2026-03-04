import json
from pathlib import Path

from numerical_lab.engine.controller import NumericalEngine
from numerical_lab.engine.trace_dir import export_compare_traces_json_dir


def f(x): return x**3 - x - 2
def df(x): return 3*x**2 - 1


def test_hybrid_decisions_present_only_for_hybrid(tmp_path: Path):
    comp = NumericalEngine.compare_methods(
        f=f, df=df, bracket=(1.0, 2.0), secant_guesses=(1.0, 2.0), tol=1e-10, max_iter=100
    )

    out_dir = tmp_path / "traces"
    export_compare_traces_json_dir(comp, str(out_dir))

    for name in ["bisection", "newton", "secant", "hybrid"]:
        p = out_dir / f"{name}.json"
        assert p.exists()

        data = json.loads(p.read_text(encoding="utf-8"))
        assert "extra" in data and "diagnostics" in data["extra"]
        diag = data["extra"]["diagnostics"]

        if name == "hybrid":
            assert "hybrid_decisions" in diag
            hd = diag["hybrid_decisions"]
            # basic contract
            for k in [
                "newton_attempts", "newton_accepts", "newton_rejects",
                "accept_rate", "reject_reasons",
                "bisection_steps", "newton_steps", "dominant_mode",
            ]:
                assert k in hd
        else:
            assert "hybrid_decisions" not in diag