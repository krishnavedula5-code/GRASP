from __future__ import annotations

from typing import Any, Dict, List, Optional

from .schema import (
    SEVERITY_PASS,
    SEVERITY_WARNING,
    SEVERITY_SUSPICIOUS,
    _safe_float,
    _safe_int,
    make_issue,
    summarize_issues,
)


BRACKET_METHODS = {"bisection", "brent", "hybrid", "safeguarded_newton"}
NEWTON_LIKE_METHODS = {"newton", "safeguarded_newton"}


def _sum_int_values(d: Any) -> Optional[int]:
    if not isinstance(d, dict):
        return None
    total = 0
    found = False
    for v in d.values():
        iv = _safe_int(v)
        if iv is not None:
            total += iv
            found = True
    return total if found else None


def _extract_total_trials(method_summary: Dict[str, Any]) -> Optional[int]:
    for key in ("samples", "num_samples", "total_runs", "trial_count"):
        iv = _safe_int(method_summary.get(key))
        if iv is not None:
            return iv

    status_counts = method_summary.get("status_counts", {})
    return _sum_int_values(status_counts)


def _extract_success_count(method_summary: Dict[str, Any]) -> Optional[int]:
    direct = _safe_int(method_summary.get("success_count"))
    if direct is not None:
        return direct

    status_counts = method_summary.get("status_counts", {})
    if isinstance(status_counts, dict):
        total = 0
        found = False
        for k, v in status_counts.items():
            if str(k).lower() == "converged":
                iv = _safe_int(v)
                if iv is not None:
                    total += iv
                    found = True
        if found:
            return total

    return None


def _extract_failure_count(method_summary: Dict[str, Any]) -> Optional[int]:
    direct = _safe_int(method_summary.get("failure_count"))
    if direct is not None:
        return direct

    status_counts = method_summary.get("status_counts", {})
    if isinstance(status_counts, dict):
        total = 0
        found = False
        for k, v in status_counts.items():
            if str(k).lower() != "converged":
                iv = _safe_int(v)
                if iv is not None:
                    total += iv
                    found = True
        if found:
            return total

    return None


def _extract_success_probability(method_summary: Dict[str, Any]) -> Optional[float]:
    for key in ("success_probability", "success_rate", "p_success"):
        fv = _safe_float(method_summary.get(key))
        if fv is not None:
            return fv
    return None


def _extract_basin_shares(method_summary: Dict[str, Any]) -> Optional[Dict[str, float]]:
    basin_shares = method_summary.get("basin_shares")
    if isinstance(basin_shares, dict):
        out: Dict[str, float] = {}
        for k, v in basin_shares.items():
            fv = _safe_float(v)
            if fv is not None:
                out[str(k)] = fv
        return out if out else None

    basin_counts = method_summary.get("basin_counts")
    total = _extract_total_trials(method_summary)
    if isinstance(basin_counts, dict) and total and total > 0:
        out = {}
        for k, v in basin_counts.items():
            iv = _safe_int(v)
            if iv is not None:
                out[str(k)] = iv / total
        return out if out else None

    return None


def _extract_newton_pathology_expectation(expectations: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    newton_expectation = expectations.get("newton_pathology")
    if isinstance(newton_expectation, dict):
        return newton_expectation

    method_expectations = expectations.get("methods", {})
    if isinstance(method_expectations, dict):
        newton_block = method_expectations.get("newton", {})
        pathology = newton_block.get("pathology") or newton_block.get("expected_pathology")
        if isinstance(pathology, dict):
            return pathology

    return None


def _extract_bracket_robustness_expectation(expectations: Dict[str, Any], method: str) -> Optional[str]:
    method_expectations = expectations.get("methods", {})
    if isinstance(method_expectations, dict):
        block = method_expectations.get(method, {})
        robustness = block.get("robustness") or block.get("expected_robustness")
        if robustness is not None:
            return str(robustness).lower()
    return None


def run_solver_checks(
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
            issues.append(
                make_issue(
                    code="method_summary_missing",
                    severity=SEVERITY_WARNING,
                    method=method,
                    message="Method summary block is missing or malformed.",
                )
            )
            continue

        total = _extract_total_trials(ms)
        success = _extract_success_count(ms)
        failure = _extract_failure_count(ms)
        success_prob = _extract_success_probability(ms)

        # success/failure consistency
        if total is not None and success is not None and failure is not None:
            if success + failure == total:
                issues.append(
                    make_issue(
                        code="success_failure_consistent",
                        severity=SEVERITY_PASS,
                        method=method,
                        message="Success and failure counts add up to total trials.",
                        expected=total,
                        observed=success + failure,
                    )
                )
            else:
                issues.append(
                    make_issue(
                        code="success_failure_inconsistent",
                        severity=SEVERITY_SUSPICIOUS,
                        method=method,
                        message="Success and failure counts do not add up to total trials.",
                        expected=total,
                        observed=success + failure,
                        details={
                            "success_count": success,
                            "failure_count": failure,
                        },
                    )
                )
        else:
            issues.append(
                make_issue(
                    code="success_failure_missing",
                    severity=SEVERITY_WARNING,
                    method=method,
                    message="Could not fully validate success/failure counts because one or more values are missing.",
                    details={
                        "total": total,
                        "success_count": success,
                        "failure_count": failure,
                    },
                )
            )

        # success probability sanity
        if total is not None and success is not None and success_prob is not None and total > 0:
            implied = success / total
            if abs(implied - success_prob) <= 0.02:
                issues.append(
                    make_issue(
                        code="success_probability_consistent",
                        severity=SEVERITY_PASS,
                        method=method,
                        message="Reported success probability is consistent with counts.",
                        expected=round(implied, 6),
                        observed=round(success_prob, 6),
                    )
                )
            else:
                issues.append(
                    make_issue(
                        code="success_probability_inconsistent",
                        severity=SEVERITY_SUSPICIOUS,
                        method=method,
                        message="Reported success probability is inconsistent with success count / total trials.",
                        expected=round(implied, 6),
                        observed=round(success_prob, 6),
                    )
                )

        # basin share sanity
        basin_shares = _extract_basin_shares(ms)
        if basin_shares:
            total_share = sum(v for v in basin_shares.values() if v is not None)
            if 0.0 <= total_share <= 1.05:
                sev = SEVERITY_PASS if total_share <= 1.01 else SEVERITY_WARNING
                msg = (
                    "Basin shares are within sane range."
                    if sev == SEVERITY_PASS
                    else "Basin shares are slightly above 1.0; check rounding or normalization."
                )
                issues.append(
                    make_issue(
                        code="basin_share_sane",
                        severity=sev,
                        method=method,
                        message=msg,
                        observed=round(total_share, 6),
                    )
                )
            else:
                issues.append(
                    make_issue(
                        code="basin_share_invalid",
                        severity=SEVERITY_SUSPICIOUS,
                        method=method,
                        message="Basin shares sum to an invalid value. This suggests normalization or counting issues.",
                        observed=round(total_share, 6),
                    )
                )

        # expected Newton pathology vs observed
        if method == "newton":
            pathology = _extract_newton_pathology_expectation(expectations)
            if pathology:
                pathology_expected = bool(pathology.get("expected", True))
                sensitivity_expected = bool(
                    pathology.get("sensitive_near_critical_points", pathology_expected)
                )

                observed_failure_rate = None
                if total is not None and failure is not None and total > 0:
                    observed_failure_rate = failure / total

                if sensitivity_expected and observed_failure_rate is not None:
                    if observed_failure_rate > 0.0:
                        issues.append(
                            make_issue(
                                code="newton_pathology_observed",
                                severity=SEVERITY_PASS,
                                method=method,
                                message="Expected Newton sensitivity/pathology is reflected in observed nonzero failure rate.",
                                expected="some sensitivity/pathology",
                                observed=round(observed_failure_rate, 6),
                            )
                        )
                    else:
                        issues.append(
                            make_issue(
                                code="newton_pathology_not_observed",
                                severity=SEVERITY_WARNING,
                                method=method,
                                message="Analytic expectations suggest Newton sensitivity/pathology, but no failures were observed. This may be valid, but should be checked against sampling, clustering, and test difficulty.",
                                expected="some sensitivity/pathology",
                                observed=0.0,
                            )
                        )

        # bracket-method robustness consistency
        if method in BRACKET_METHODS:
            robustness = _extract_bracket_robustness_expectation(expectations, method)
            if robustness in {"high", "robust", "expected_robust"}:
                observed_failure_rate = None
                if total is not None and failure is not None and total > 0:
                    observed_failure_rate = failure / total

                if observed_failure_rate is not None:
                    if observed_failure_rate <= 0.05:
                        issues.append(
                            make_issue(
                                code="bracket_robustness_consistent",
                                severity=SEVERITY_PASS,
                                method=method,
                                message="Observed failure rate is consistent with expected bracket-method robustness.",
                                observed=round(observed_failure_rate, 6),
                            )
                        )
                    elif observed_failure_rate <= 0.20:
                        issues.append(
                            make_issue(
                                code="bracket_robustness_weaker_than_expected",
                                severity=SEVERITY_WARNING,
                                method=method,
                                message="Bracket method is less robust than expected, but not implausibly so.",
                                observed=round(observed_failure_rate, 6),
                            )
                        )
                    else:
                        issues.append(
                            make_issue(
                                code="bracket_robustness_contradiction",
                                severity=SEVERITY_SUSPICIOUS,
                                method=method,
                                message="Bracket method shows a high failure rate despite an expectation of robustness.",
                                observed=round(observed_failure_rate, 6),
                            )
                        )

    return {
        "overview": summarize_issues(issues),
        "issues": issues,
    }