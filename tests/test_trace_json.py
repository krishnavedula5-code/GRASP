import json
from pathlib import Path
from dataclasses import asdict

from numerical_lab.methods.newton import NewtonSolver
from numerical_lab.engine.trace import export_trace_json
from numerical_lab.engine.summary import build_method_summary
from numerical_lab.diagnostics.convergence import classify_convergence
from numerical_lab.diagnostics.stability import detect_stability
from numerical_lab.diagnostics.explain import explain_run


def f(x): return x**3 - x - 2
def df(x): return 3*x**2 - 1


def test_export_trace_json(tmp_path: Path):
    res = NewtonSolver(f, df, x0=1.5, tol=1e-12, max_iter=50).solve()
    conv = classify_convergence(res)
    stab = detect_stability(res)
    summ = build_method_summary("newton", res, conv, stab)

    out = tmp_path / "trace.json"
    export_trace_json(
        res,
        str(out),
        method_summary=summ,
        extra={
            "explanation": explain_run(summ, res),
            "diagnostics": {
                "convergence": asdict(conv),
                "stability": asdict(stab),
            },
        },
    )

    data = json.loads(out.read_text(encoding="utf-8"))

    # ---- trace contract invariants (v1.1) ----
    assert "records" in data and len(data["records"]) >= 1
    assert "summary" in data

    s = data["summary"]
    assert s["method"] == "newton"
    assert s["method"] == data["method"]

    # status/stop_reason must match top-level contract
    assert s["status"] == data["status"]
    assert s["stop_reason"] == data["stop_reason"]
    assert s["status"] in {
        "converged",
        "max_iter",
        "stagnation",
        "bad_bracket",
        "nan_or_inf",
        "derivative_zero",
        "error",
    }
    assert isinstance(s["stop_reason"], str)

    for k in ["step_type", "reject_reason", "a", "b", "m", "fm", "dfm", "x_newton", "fx_newton", "meta"]:
        assert k in data["records"][0]

    # iterations invariant: equals last record index k
    assert isinstance(s["iterations"], int)
    assert s["iterations"] == data["iterations"]
    assert data["iterations"] == data["records"][-1]["k"]

    # last_residual should agree with last record residual when present
    last_rec = data["records"][-1]
    if s["last_residual"] is not None:
        assert abs(s["last_residual"] - last_rec["residual"]) <= 1e-18

    # Option A accounting exists
    assert isinstance(data["n_f"], int) and data["n_f"] >= 0
    assert isinstance(data["n_df"], int) and data["n_df"] >= 0

    #  new assertions for diagnostics presence
    assert "diagnostics" in data["extra"]
    assert "convergence" in data["extra"]["diagnostics"]
    assert "stability" in data["extra"]["diagnostics"]
    assert "classification" in data["extra"]["diagnostics"]["convergence"]