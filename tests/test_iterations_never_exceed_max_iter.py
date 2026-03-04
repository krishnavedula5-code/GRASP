import json
from pathlib import Path
import pytest

FILES = [
    "data/v1/sweep_P1_x3_minus_2x_plus_2.json",
    "data/v1/sweep_P2_x3_minus_x_minus_2.json",
    "data/v1/sweep_P3_cosx_minus_x.json",
    "data/v1/sweep_P4_multiroot.json",
]
METHODS = ["newton", "secant", "bisection", "hybrid"]

def load_records(fname: str):
    p = Path(fname)
    assert p.exists(), (
        f"Missing {fname}. Put the sweep JSONs in the project root (same folder you run pytest from), "
        f"or update FILES to the correct path."
    )
    return json.loads(p.read_text(encoding="utf-8"))

@pytest.mark.parametrize("fname", FILES)
@pytest.mark.parametrize("method", METHODS)
def test_iterations_do_not_exceed_request_max_iter(fname, method):
    records = load_records(fname)

    assert isinstance(records, list), f"{fname}: expected list of records, got {type(records)}"

    for i, r in enumerate(records):
        req = r.get("request", {})
        assert "max_iter" in req, f"{fname}[{i}] missing request.max_iter"
        max_iter = int(req["max_iter"])

        assert method in r, f"{fname}[{i}] missing {method} block"
        blk = r[method]
        assert "summary" in blk, f"{fname}[{i}].{method} missing summary"
        summ = blk["summary"]

        it = summ.get("iterations", None)
        assert it is not None, f"{fname}[{i}].{method} missing summary.iterations"
        it = int(it)

        assert it <= max_iter, (
            f"{fname}[{i}].{method}: iterations={it} exceeds request.max_iter={max_iter} "
            f"(status={summ.get('status')}, stop_reason={summ.get('stop_reason')})"
        )