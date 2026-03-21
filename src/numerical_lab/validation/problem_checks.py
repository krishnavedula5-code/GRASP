from __future__ import annotations

from typing import Any, Dict, List, Optional

from .schema import (
    SEVERITY_PASS,
    SEVERITY_WARNING,
    SEVERITY_SUSPICIOUS,
    _safe_int,
    make_issue,
    summarize_issues,
)


def _get_expected_root_count(expectations: Dict[str, Any]) -> Optional[int]:
    for key in ("expected_root_count", "root_count"):
        value = expectations.get(key)
        iv = _safe_int(value)
        if iv is not None:
            return iv

    roots = expectations.get("roots")
    if isinstance(roots, list):
        return len(roots)

    return None


def _get_observed_root_count(summary: Dict[str, Any]) -> Optional[int]:
    for key in ("root_count", "observed_root_count", "discovered_root_count"):
        value = summary.get(key)
        iv = _safe_int(value)
        if iv is not None:
            return iv

    coverage = summary.get("coverage_summary", {})
    for key in ("root_count", "observed_root_count", "discovered_root_count"):
        value = coverage.get(key)
        iv = _safe_int(value)
        if iv is not None:
            return iv

    observed_roots = summary.get("observed_roots")
    if isinstance(observed_roots, list):
        return len(observed_roots)

    return None


def _get_coverage_count_for_method(method_summary: Dict[str, Any]) -> Optional[int]:
    for key in ("root_coverage_count", "coverage_count"):
        value = method_summary.get(key)
        iv = _safe_int(value)
        if iv is not None:
            return iv
    return None


def run_problem_checks(
    summary: Dict[str, Any],
    interpretation: Dict[str, Any],
    expectations: Dict[str, Any],
    methods: List[str],
) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []

    expected_root_count = _get_expected_root_count(expectations)
    observed_root_count = _get_observed_root_count(summary)

    if expected_root_count is None or observed_root_count is None:
        issues.append(
            make_issue(
                code="root_count_missing",
                severity=SEVERITY_WARNING,
                message="Could not fully validate expected vs observed root count because one side is missing.",
                expected=expected_root_count,
                observed=observed_root_count,
            )
        )
    else:
        if expected_root_count == observed_root_count:
            issues.append(
                make_issue(
                    code="root_count_match",
                    severity=SEVERITY_PASS,
                    message="Observed root count matches expected root count.",
                    expected=expected_root_count,
                    observed=observed_root_count,
                )
            )
        elif observed_root_count < expected_root_count:
            issues.append(
                make_issue(
                    code="root_count_under",
                    severity=SEVERITY_WARNING,
                    message="Observed root count is smaller than expected root count.",
                    expected=expected_root_count,
                    observed=observed_root_count,
                )
            )
        else:
            issues.append(
                make_issue(
                    code="root_count_over",
                    severity=SEVERITY_SUSPICIOUS,
                    message="Observed root count exceeds expected root count. This may indicate clustering, detection, or reporting issues.",
                    expected=expected_root_count,
                    observed=observed_root_count,
                )
            )

    methods_block = summary.get("methods", {})
    if expected_root_count is not None and isinstance(methods_block, dict):
        for method in methods:
            ms = methods_block.get(method, {})
            coverage_count = _get_coverage_count_for_method(ms)
            if coverage_count is None:
                continue

            if coverage_count <= expected_root_count:
                issues.append(
                    make_issue(
                        code="coverage_count_sane",
                        severity=SEVERITY_PASS,
                        method=method,
                        message="Coverage count does not exceed expected root count.",
                        expected=expected_root_count,
                        observed=coverage_count,
                    )
                )
            else:
                issues.append(
                    make_issue(
                        code="coverage_count_exceeds_expected",
                        severity=SEVERITY_SUSPICIOUS,
                        method=method,
                        message="Method reports coverage count larger than expected root count.",
                        expected=expected_root_count,
                        observed=coverage_count,
                    )
                )

    return {
        "overview": summarize_issues(issues),
        "issues": issues,
    }