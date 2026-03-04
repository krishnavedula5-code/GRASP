from __future__ import annotations

from typing import Dict, Any, Optional
from datetime import datetime

from numerical_lab.engine.summary import build_comparison_summary
from numerical_lab.diagnostics.explain import explain_run


def export_markdown_report(
    comparison_results: Dict[str, Any],
    out_path: str,
    title: str = "Numerical Lab Report",
    problem: Optional[str] = None,
) -> None:
    summaries = build_comparison_summary(comparison_results)

    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")

    if problem:
        lines.append("## Problem")
        lines.append("")
        lines.append(f"`f(x) = {problem}`")
        lines.append("")

    lines.append("## Method Summary")
    lines.append("")
    lines.append("| Method | Status | Iters | Root | Last |f(x)| | Order | Stability |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for method in ["bisection", "newton", "secant", "hybrid"]:
        if method not in comparison_results:
            continue
        s = summaries[method]
        root = "" if s.root is None else f"{s.root:.12g}"
        last = "" if s.last_residual is None else f"{s.last_residual:.3e}"
        order = "" if s.observed_order is None else f"{s.observed_order:.3f}"
        lines.append(f"| {method} | {s.status} | {s.iterations} | {root} | {last} | {order} | {s.stability_label} |")
    lines.append("")

    lines.append("## Explanations")
    lines.append("")
    for method in ["bisection", "newton", "secant", "hybrid"]:
        if method not in comparison_results:
            continue
        result, conv, stab = comparison_results[method]
        s = summaries[method]
        lines.append(f"### {method}")
        lines.append("")
        lines.append(explain_run(s, result))
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))