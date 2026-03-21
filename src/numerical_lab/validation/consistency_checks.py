from __future__ import annotations

from typing import Any, Dict, List

from .schema import (
    SEVERITY_PASS,
    SEVERITY_WARNING,
    SEVERITY_SUSPICIOUS,
    _safe_float,
    _safe_int,
    make_issue,
    summarize_issues,
)


POSITIVE_WORDS = (
    "robust",
    "stable",
    "high success",
    "reliable",
    "performed well",
    "strong",
)
NEGATIVE_WORDS = (
    "poor",
    "unstable",
    "fragile",
    "failed",
    "sensitive",
    "pathology",
    "low success",
    "inconsistent",
)


def _method_text(interpretation: Dict[str, Any], method: str) -> str:
    methods_block = interpretation.get("methods", {})
    if isinstance(methods_block, dict):
        block = methods_block.get(method, {})
        if isinstance(block, dict):
            parts: List[str] = []
            for key in ("summary", "interpretation", "text", "analysis"):
                value = block.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
            return " ".join(parts).lower()

    top_summary = interpretation.get("summary")
    if isinstance(top_summary, str):
        return top_summary.lower()

    return ""


def run_consistency_checks(
    summary: Dict[str, Any],
    interpretation: Dict[str, Any],
    expectations: Dict[str, Any],
    methods: List[str],
) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    methods_block = summary.get("methods", {})

    for method in methods:
        ms = methods_block.get(method, {})
        if not isinstance(ms, dict):
            continue

        success_probability = None
        for key in ("success_probability", "success_rate", "p_success"):
            success_probability = _safe_float(ms.get(key))
            if success_probability is not None:
                break

        if success_probability is None:
            status_counts = ms.get("status_counts", {})
            if isinstance(status_counts, dict):
                total = 0
                succ = 0
                for k, v in status_counts.items():
                    iv = _safe_int(v)
                    if iv is None:
                        continue
                    total += iv
                    if str(k).lower() == "converged":
                        succ += iv
                if total > 0:
                    success_probability = succ / total

        text = _method_text(interpretation, method)
        if not text:
            issues.append(
                make_issue(
                    code="interpretation_missing",
                    severity=SEVERITY_WARNING,
                    method=method,
                    message="No interpretation text found for contradiction checking.",
                )
            )
            continue

        has_positive = any(word in text for word in POSITIVE_WORDS)
        has_negative = any(word in text for word in NEGATIVE_WORDS)

        if success_probability is None:
            issues.append(
                make_issue(
                    code="contradiction_check_partial",
                    severity=SEVERITY_WARNING,
                    method=method,
                    message="Could not fully assess interpretation-vs-summary contradiction because success probability is unavailable.",
                )
            )
            continue

        if success_probability >= 0.9 and has_negative and not has_positive:
            issues.append(
                make_issue(
                    code="interpretation_contradicts_summary_negative",
                    severity=SEVERITY_WARNING,
                    method=method,
                    message="Interpretation sounds strongly negative despite very high observed success probability.",
                    observed=round(success_probability, 6),
                )
            )
        elif success_probability <= 0.3 and has_positive and not has_negative:
            issues.append(
                make_issue(
                    code="interpretation_contradicts_summary_positive",
                    severity=SEVERITY_SUSPICIOUS,
                    method=method,
                    message="Interpretation sounds strongly positive despite low observed success probability.",
                    observed=round(success_probability, 6),
                )
            )
        else:
            issues.append(
                make_issue(
                    code="interpretation_summary_consistent",
                    severity=SEVERITY_PASS,
                    method=method,
                    message="Interpretation tone is broadly consistent with summary statistics.",
                    observed=round(success_probability, 6),
                )
            )

    return {
        "overview": summarize_issues(issues),
        "issues": issues,
    }