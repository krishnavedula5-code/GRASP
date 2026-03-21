from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .consistency_checks import run_consistency_checks
from .problem_checks import run_problem_checks
from .schema import render_validation_text, summarize_issues
from .solver_checks import run_solver_checks


def _collect_methods(summary: Dict[str, Any], explicit_methods: Optional[List[str]] = None) -> List[str]:
    if explicit_methods:
        return list(explicit_methods)

    methods_block = summary.get("methods", {})
    if isinstance(methods_block, dict):
        return list(methods_block.keys())

    methods = summary.get("methods_list")
    if isinstance(methods, list):
        return [str(m) for m in methods]

    return []


def run_validation(
    summary: Dict[str, Any],
    interpretation: Dict[str, Any],
    expectations: Dict[str, Any],
    output_dir: str | Path,
    methods: Optional[List[str]] = None,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    resolved_methods = _collect_methods(summary, methods)

    problem_checks = run_problem_checks(
        summary=summary,
        interpretation=interpretation,
        expectations=expectations,
        methods=resolved_methods,
    )
    solver_checks = run_solver_checks(
        summary=summary,
        interpretation=interpretation,
        expectations=expectations,
        methods=resolved_methods,
    )
    consistency_checks = run_consistency_checks(
        summary=summary,
        interpretation=interpretation,
        expectations=expectations,
        methods=resolved_methods,
    )

    all_issues = (
        problem_checks.get("issues", [])
        + solver_checks.get("issues", [])
        + consistency_checks.get("issues", [])
    )

    validation: Dict[str, Any] = {
        "validation_version": "v1",
        "methods": resolved_methods,
        "overview": summarize_issues(all_issues),
        "problem_checks": problem_checks,
        "solver_checks": solver_checks,
        "consistency_checks": consistency_checks,
    }

    validation_json_path = output_path / "validation.json"
    validation_txt_path = output_path / "validation.txt"

    with open(validation_json_path, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2, ensure_ascii=False)

    validation_text = render_validation_text(validation)
    with open(validation_txt_path, "w", encoding="utf-8") as f:
        f.write(validation_text)

    validation["artifacts"] = {
        "validation_json": str(validation_json_path),
        "validation_txt": str(validation_txt_path),
    }

    return validation