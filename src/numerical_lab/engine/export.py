from __future__ import annotations

import csv
from typing import Optional

from numerical_lab.core.base import SolverResult


def export_iterations_csv(result: SolverResult, filepath: str) -> None:
    """
    Export iteration table to CSV.
    This is a commercial requirement for classroom use.
    """
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "x", "f(x)", "|f(x)|", "|Δx|"])

        for r in result.records:
            writer.writerow([
                r.k,
                r.x,
                r.fx,
                r.residual,
                "" if r.step_error is None else r.step_error,
            ])