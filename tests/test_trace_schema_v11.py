import json
from pathlib import Path

from numerical_lab.methods.newton import NewtonSolver
from numerical_lab.engine.trace import export_trace_json


def f(x): return x**3 - x - 2
def df(x): return 3*x**2 - 1


def test_trace_schema_v11_contract(tmp_path: Path):
    res = NewtonSolver(f, df, x0=1.5, tol=1e-12, max_iter=50).solve()

    out = tmp_path / "trace.json"
    export_trace_json(res, str(out))

    data = json.loads(out.read_text(encoding="utf-8"))

    # top-level contract
    assert data["schema_version"] == "1.1"
    for k in ["generated_utc", "method", "status", "stop_reason", "message", "root", "iterations", "records", "events"]:
        assert k in data

    assert isinstance(data["records"], list) and len(data["records"]) > 0

    # per-record contract (keys should exist even if null)
    r0 = data["records"][0]
    required = [
        "k", "x", "fx", "residual", "step_error",
        "step_type", "accepted", "reject_reason",
        "a", "b", "interval_width", "m", "fm",
        "dfm", "x_newton", "fx_newton", "meta",
    ]
    for k in required:
        assert k in r0

    # accounting must exist (Option A)
    assert isinstance(data["n_f"], int) and data["n_f"] >= 1
    assert isinstance(data["n_df"], int) and data["n_df"] >= 1