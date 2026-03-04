from __future__ import annotations

import os
from dataclasses import asdict
from typing import Dict, Any

from numerical_lab.engine.trace import export_trace_json
from numerical_lab.engine.summary import build_method_summary
from numerical_lab.diagnostics.explain import explain_run
from numerical_lab.diagnostics.hybrid_decisions import hybrid_decision_report  # ✅ ADD


def export_compare_traces_json_dir(comparison_results: Dict[str, Any], out_dir: str) -> None:
    """
    Export one JSON trace per *canonical method name* into out_dir:
      out_dir/bisection.json
      out_dir/newton.json
      out_dir/secant.json
      out_dir/hybrid.json
    """
    os.makedirs(out_dir, exist_ok=True)

    for _key, triple in comparison_results.items():
        result, conv, stab = triple

        # ✅ canonical filename (stable contract)
        canonical = result.method

        summ = build_method_summary(canonical, result, conv, stab)

        # ✅ diagnostics payload (always includes conv/stab)
        diag = {
            "convergence": asdict(conv),
            "stability": asdict(stab),
        }

        # ✅ add hybrid intelligence only for hybrid solver
        if canonical == "hybrid":
            diag["hybrid_decisions"] = asdict(hybrid_decision_report(result))

        export_trace_json(
            result=result,
            filepath=os.path.join(out_dir, f"{canonical}.json"),
            method_summary=summ,
            extra={
                "explanation": explain_run(summ, result),
                "diagnostics": diag,
            },
        )