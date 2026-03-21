from __future__ import annotations

from typing import Any, Dict, List, Sequence


BRACKET_METHODS = {"bisection", "brent", "hybrid", "safeguarded_newton"}
OPEN_METHODS = {"newton", "secant"}


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def _safe_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _get_newton_risk(expectations: Dict[str, Any]) -> Dict[str, Any]:
    data = _safe_get(expectations, "analytic_checks", "newton_pathology", default={})
    return data if isinstance(data, dict) else {}


def _normalize_method_name(method: str) -> str:
    return str(method or "").strip().lower()


def _band_to_score(band: str) -> float:
    band = _normalize_method_name(band)
    mapping = {
        "low": 0.85,
        "moderate": 0.55,
        "high": 0.20,
    }
    return mapping.get(band, 0.50)


def _recommendation_confidence_band(score: float) -> str:
    score = _clamp(score)
    if score < 0.40:
        return "low"
    if score < 0.70:
        return "moderate"
    return "high"


def _build_base_problem_features(
    expectations: Dict[str, Any],
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metadata = metadata or {}

    sign_change_count = int(
        _safe_get(expectations, "analytic_checks", "sign_change_interval_count", default=0)
    )
    root_candidate_count = int(
        _safe_get(expectations, "analytic_checks", "root_candidate_count", default=0)
    )
    critical_point_count = int(
        _safe_get(expectations, "analytic_checks", "critical_point_count", default=0)
    )

    symmetry = _safe_get(expectations, "analytic_checks", "symmetry", default={}) or {}
    symmetry_type = str(symmetry.get("symmetry_type", "none")).strip().lower()

    problem_summary = expectations.get("problem_summary") or {}
    dexpr_provided = bool(problem_summary.get("dexpr_provided", False))

    numerical_derivative = bool(metadata.get("numerical_derivative", False))
    derivative_available = dexpr_provided or numerical_derivative

    newton_risk = _get_newton_risk(expectations)
    newton_risk_band = str(newton_risk.get("expected_newton_risk_band", "unknown")).strip().lower()
    newton_risk_score = float(newton_risk.get("expected_newton_risk_score", 0.50))

    instability_fraction = float(
        _safe_get(newton_risk, "instability_regions", "instability_fraction", default=0.0)
    )
    degenerate_fraction = float(
        _safe_get(newton_risk, "derivative_degeneracy", "degenerate_fraction", default=0.0)
    )
    high_step_fraction = float(
        _safe_get(newton_risk, "step_risk", "high_step_fraction", default=0.0)
    )

    return {
        "sign_change_count": sign_change_count,
        "root_candidate_count": root_candidate_count,
        "critical_point_count": critical_point_count,
        "symmetry_type": symmetry_type,
        "derivative_available": derivative_available,
        "newton_risk_band": newton_risk_band,
        "newton_risk_score": newton_risk_score,
        "instability_fraction": instability_fraction,
        "degenerate_fraction": degenerate_fraction,
        "high_step_fraction": high_step_fraction,
    }


def _score_newton(features: Dict[str, Any]) -> Dict[str, Any]:
    derivative_available = bool(features["derivative_available"])
    risk_band = str(features["newton_risk_band"])
    risk_score_raw = float(features["newton_risk_score"])
    critical_point_count = int(features["critical_point_count"])

    reasons_for: List[str] = []
    reasons_against: List[str] = []

    if not derivative_available:
        return {
            "method": "newton",
            "analytic_score": 0.0,
            "recommended": False,
            "reasons_for": reasons_for,
            "reasons_against": ["Derivative information is unavailable, so plain Newton is not usable."],
        }

    score = 0.50
    band_score = _band_to_score(risk_band)
    score = 0.45 * score + 0.55 * band_score

    if risk_band == "low":
        reasons_for.append("Analytic Newton pathology scan indicates low structural risk.")
        score += 0.20
    elif risk_band == "moderate":
        reasons_against.append("Analytic Newton pathology scan indicates moderate structural risk.")
        score -= 0.05
    elif risk_band == "high":
        reasons_against.append("Analytic Newton pathology scan indicates high structural risk.")
        score -= 0.30

    if critical_point_count == 0:
        reasons_for.append("No derivative-critical points were detected on the sampled domain.")
        score += 0.10
    else:
        reasons_against.append(
            f"Detected approximately {critical_point_count} derivative-critical location(s), which can reduce Newton robustness."
        )
        score -= min(0.05 * critical_point_count, 0.15)

    if risk_score_raw < 0.20:
        reasons_for.append("Quantitative Newton risk score is low.")
    elif risk_score_raw >= 0.50:
        reasons_against.append("Quantitative Newton risk score is high.")

    score = _clamp(score)

    return {
        "method": "newton",
        "analytic_score": score,
        "recommended": score >= 0.60,
        "reasons_for": reasons_for,
        "reasons_against": reasons_against,
    }


def _score_secant(features: Dict[str, Any]) -> Dict[str, Any]:
    derivative_available = bool(features["derivative_available"])
    risk_band = str(features["newton_risk_band"])
    critical_point_count = int(features["critical_point_count"])

    reasons_for: List[str] = []
    reasons_against: List[str] = []

    score = 0.55

    if not derivative_available:
        reasons_for.append("Derivative information is unavailable, so secant is more practical than Newton.")
        score += 0.15
    else:
        reasons_for.append("Secant can be used as a derivative-free alternative even when derivatives exist.")

    if risk_band == "high":
        reasons_against.append("High Newton-type structural risk may also indicate flat or unstable local behavior affecting secant slopes.")
        score -= 0.10
    elif risk_band == "moderate":
        reasons_against.append("Moderate Newton-type structural risk suggests some local sensitivity may also affect secant.")
        score -= 0.05

    if critical_point_count > 0:
        reasons_against.append(
            f"Detected approximately {critical_point_count} derivative-critical location(s), which may destabilize secant slope estimates."
        )
        score -= min(0.04 * critical_point_count, 0.12)

    score = _clamp(score)

    return {
        "method": "secant",
        "analytic_score": score,
        "recommended": score >= 0.60,
        "reasons_for": reasons_for,
        "reasons_against": reasons_against,
    }


def _score_bisection(features: Dict[str, Any]) -> Dict[str, Any]:
    sign_change_count = int(features["sign_change_count"])
    root_candidate_count = int(features["root_candidate_count"])

    reasons_for: List[str] = []
    reasons_against: List[str] = []

    score = 0.25

    if sign_change_count > 0:
        reasons_for.append(
            f"Detected {sign_change_count} sign-change interval(s), so bisection has structurally valid bracket targets."
        )
        score += 0.45
    else:
        reasons_against.append("No sign-change intervals were detected, so pure bisection may have no accessible target.")
        score -= 0.30

    if root_candidate_count > sign_change_count:
        reasons_against.append(
            "There appear to be more root candidates than sign-change intervals, so bracket-only coverage may be incomplete."
        )
        score -= 0.05

    reasons_for.append("Bisection prioritizes robustness over speed.")

    score = _clamp(score)

    return {
        "method": "bisection",
        "analytic_score": score,
        "recommended": score >= 0.60,
        "reasons_for": reasons_for,
        "reasons_against": reasons_against,
    }


def _score_brent(features: Dict[str, Any]) -> Dict[str, Any]:
    sign_change_count = int(features["sign_change_count"])
    root_candidate_count = int(features["root_candidate_count"])

    reasons_for: List[str] = []
    reasons_against: List[str] = []

    score = 0.58

    if sign_change_count > 0:
        reasons_for.append(
            f"Detected {sign_change_count} sign-change interval(s), so Brent has structurally valid bracket targets."
        )
        score += 0.30
    else:
        reasons_against.append("No sign-change intervals were detected, so Brent may lack a reliable bracket target.")
        score -= 0.25

    if root_candidate_count > sign_change_count:
        reasons_against.append(
            "There appear to be more root candidates than sign-change intervals, so sign-change-based coverage may be incomplete."
        )
        score -= 0.05

    reasons_for.append("Brent combines bracket robustness with faster practical convergence than pure bisection.")
    score = _clamp(score)

    return {
        "method": "brent",
        "analytic_score": score,
        "recommended": score >= 0.60,
        "reasons_for": reasons_for,
        "reasons_against": reasons_against,
    }


def _score_hybrid(features: Dict[str, Any]) -> Dict[str, Any]:
    derivative_available = bool(features["derivative_available"])
    sign_change_count = int(features["sign_change_count"])
    risk_band = str(features["newton_risk_band"])

    reasons_for: List[str] = []
    reasons_against: List[str] = []

    score = 0.55

    if sign_change_count > 0:
        reasons_for.append("Bracket structure is available, which supports safeguarded/hybrid behavior.")
        score += 0.20
    else:
        reasons_against.append("No clear sign-change bracket structure was detected.")
        score -= 0.10

    if derivative_available:
        reasons_for.append("Derivative information is available, enabling Newton-style acceleration inside a safeguarded strategy.")
        score += 0.10
    else:
        reasons_against.append("Derivative information is unavailable, so hybrid acceleration may be limited.")
        score -= 0.10

    if risk_band == "high":
        reasons_for.append("A safeguarded hybrid is preferable to plain Newton under high Newton risk.")
        score += 0.10
    elif risk_band == "low":
        reasons_for.append("Hybrid retains robustness while still benefiting from smooth local behavior.")
        score += 0.05

    score = _clamp(score)

    return {
        "method": "hybrid",
        "analytic_score": score,
        "recommended": score >= 0.60,
        "reasons_for": reasons_for,
        "reasons_against": reasons_against,
    }


def _score_safeguarded_newton(features: Dict[str, Any]) -> Dict[str, Any]:
    derivative_available = bool(features["derivative_available"])
    sign_change_count = int(features["sign_change_count"])
    risk_band = str(features["newton_risk_band"])

    reasons_for: List[str] = []
    reasons_against: List[str] = []

    if not derivative_available:
        return {
            "method": "safeguarded_newton",
            "analytic_score": 0.0,
            "recommended": False,
            "reasons_for": reasons_for,
            "reasons_against": ["Derivative information is unavailable, so safeguarded Newton is not usable."],
        }

    score = 0.60
    reasons_for.append("Derivative information is available for Newton-style local acceleration.")

    if sign_change_count > 0:
        reasons_for.append("Bracket structure is available, enabling safeguards against unstable Newton steps.")
        score += 0.20
    else:
        reasons_against.append("No sign-change interval was detected, so safeguard opportunities may be limited.")
        score -= 0.10

    if risk_band == "high":
        reasons_for.append("Safeguarding is especially valuable because plain Newton risk is high.")
        score += 0.10
    elif risk_band == "moderate":
        reasons_for.append("Safeguarding improves robustness under moderate Newton risk.")
        score += 0.05
    elif risk_band == "low":
        reasons_for.append("Safeguarded Newton remains viable even though plain Newton risk is already low.")

    score = _clamp(score)

    return {
        "method": "safeguarded_newton",
        "analytic_score": score,
        "recommended": score >= 0.60,
        "reasons_for": reasons_for,
        "reasons_against": reasons_against,
    }


def _score_method(method: str, features: Dict[str, Any]) -> Dict[str, Any]:
    m = _normalize_method_name(method)

    if m == "newton":
        return _score_newton(features)
    if m == "secant":
        return _score_secant(features)
    if m == "bisection":
        return _score_bisection(features)
    if m == "brent":
        return _score_brent(features)
    if m == "hybrid":
        return _score_hybrid(features)
    if m == "safeguarded_newton":
        return _score_safeguarded_newton(features)

    return {
        "method": m,
        "analytic_score": 0.30,
        "recommended": False,
        "reasons_for": [],
        "reasons_against": ["No rule-based suitability model is defined for this method yet."],
    }


def _build_observed_metrics(analytics: Dict[str, Any], audit: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    comparison_rows = (analytics.get("comparison_summary_data") or {}).get("methods", []) or []
    comparison_map = {
        _normalize_method_name(row.get("method")): row
        for row in comparison_rows
        if row.get("method") is not None
    }

    root_cov = (analytics.get("root_coverage_data") or {}).get("solvers", {}) or {}

    audit_issues = audit.get("issues", []) or []

    audit_penalties: Dict[str, float] = {}
    for issue in audit_issues:
        code = str(issue.get("code", "")).strip().lower()
        message = str(issue.get("message", "")).strip().lower()
        sev = str(issue.get("severity", "warning")).strip().lower()

        penalty = 0.0
        if sev == "warning":
            penalty = 0.08
        elif sev == "suspicious":
            penalty = 0.20

        for method in list(root_cov.keys()) + list(comparison_map.keys()):
            m = _normalize_method_name(method)
            if m and m in message:
                audit_penalties[m] = audit_penalties.get(m, 0.0) + penalty

    out: Dict[str, Dict[str, Any]] = {}

    for method in sorted(set(list(comparison_map.keys()) + list(root_cov.keys()))):
        comp = comparison_map.get(method, {}) or {}
        rc = root_cov.get(method, {}) or {}
        true_behavior = rc.get("true_behavior", {}) or {}
        bench = rc.get("benchmark_evaluation", {}) or {}

        success_rate = _safe_float(comp.get("success_rate"), default=0.0)
        mean_iter = _safe_float(comp.get("mean_iter"), default=0.0)
        failure_count = int(_safe_float(comp.get("failure_count"), default=0.0))

        benchmark_coverage = bench.get("benchmark_coverage", None)
        if benchmark_coverage is None:
            benchmark_coverage = 0.0
        benchmark_coverage = _safe_float(benchmark_coverage, default=0.0)

        domain_faithfulness = _safe_float(true_behavior.get("domain_faithfulness"), default=0.0)
        excursion_detected = bool(true_behavior.get("excursion_detected", False))
        out_of_domain_count = int(_safe_float(true_behavior.get("out_of_domain_detected_root_count"), default=0.0))

        iter_score = 0.0
        if mean_iter > 0:
            iter_score = 1.0 / (1.0 + mean_iter / 10.0)

        excursion_penalty = 0.0
        if excursion_detected:
            excursion_penalty = min(0.35, 0.05 * out_of_domain_count)

        failure_penalty = 0.0
        if success_rate < 1.0:
            failure_penalty = 1.0 - success_rate

        audit_penalty = audit_penalties.get(method, 0.0)

        observed_score = (
            0.35 * benchmark_coverage
            + 0.25 * domain_faithfulness
            + 0.20 * success_rate
            + 0.20 * iter_score
            - excursion_penalty
            - 0.25 * failure_penalty
            - audit_penalty
        )
        observed_score = _clamp(observed_score)

        out[method] = {
            "success_rate": success_rate,
            "mean_iter": mean_iter,
            "failure_count": failure_count,
            "benchmark_coverage": benchmark_coverage,
            "domain_faithfulness": domain_faithfulness,
            "excursion_detected": excursion_detected,
            "out_of_domain_count": out_of_domain_count,
            "iter_score": iter_score,
            "excursion_penalty": excursion_penalty,
            "failure_penalty": failure_penalty,
            "audit_penalty": audit_penalty,
            "observed_score": observed_score,
        }

    return out


def build_solver_selection_recommendation(
    *,
    expectations: Dict[str, Any],
    methods: Sequence[str],
    metadata: Dict[str, Any] | None = None,
    analytics: Dict[str, Any] | None = None,
    audit: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metadata = metadata or {}
    analytics = analytics or {}
    audit = audit or {}

    features = _build_base_problem_features(expectations, metadata=metadata)
    observed_metrics = _build_observed_metrics(analytics, audit)

    scored_methods = []
    for method in methods:
        analytic = _score_method(method, features)
        m = _normalize_method_name(method)
        observed = observed_metrics.get(m, {})

        analytic_score = _safe_float(analytic.get("analytic_score"), default=0.0)
        observed_score = _safe_float(observed.get("observed_score"), default=0.0)

        has_observed = bool(observed)
        overall_score = (
            0.40 * analytic_score + 0.60 * observed_score
            if has_observed
            else analytic_score
        )
        overall_score = _clamp(overall_score)

        strengths = []
        weaknesses = []

        if _safe_float(observed.get("benchmark_coverage"), default=0.0) >= 0.99:
            strengths.append("full benchmark coverage")
        if _safe_float(observed.get("domain_faithfulness"), default=0.0) >= 0.99:
            strengths.append("domain-faithful")
        if _safe_float(observed.get("success_rate"), default=0.0) >= 0.99:
            strengths.append("near-perfect observed success")
        if _safe_float(observed.get("mean_iter"), default=999.0) <= 5.0:
            strengths.append("low iteration cost")

        if bool(observed.get("excursion_detected", False)):
            weaknesses.append("out-of-domain excursions observed")
        if _safe_float(observed.get("success_rate"), default=1.0) < 0.95:
            weaknesses.append("nontrivial observed failure rate")
        if _safe_float(observed.get("audit_penalty"), default=0.0) > 0:
            weaknesses.append("audit warnings affect trustworthiness")

        scored_methods.append(
            {
                "method": m,
                "analytic_score": round(analytic_score, 6),
                "observed_score": round(observed_score, 6) if has_observed else None,
                "overall_score": round(overall_score, 6),
                "recommended": overall_score >= 0.60,
                "reasons_for": analytic.get("reasons_for", []),
                "reasons_against": analytic.get("reasons_against", []),
                "strengths": strengths,
                "weaknesses": weaknesses,
                "observed_metrics": observed,
            }
        )

    scored_methods.sort(key=lambda item: item.get("overall_score", 0.0), reverse=True)

    for i, item in enumerate(scored_methods, start=1):
        item["rank"] = i

    primary = scored_methods[0] if scored_methods else None
    secondary = scored_methods[1] if len(scored_methods) > 1 else None

    avoid = [
        item["method"]
        for item in scored_methods
        if float(item.get("overall_score", 0.0)) < 0.40
    ]

    recommendation_confidence_score = 0.0
    if primary is not None:
        top_score = float(primary.get("overall_score", 0.0))
        second_score = float(secondary.get("overall_score", 0.0)) if secondary is not None else 0.0
        separation = max(0.0, top_score - second_score)
        recommendation_confidence_score = _clamp(0.70 * top_score + 0.30 * separation)

    rationale: List[str] = []
    if primary is not None:
        rationale.append(
            f"Primary recommendation is {primary['method']} with overall score {primary['overall_score']:.4f}."
        )
        for reason in primary.get("strengths", [])[:3]:
            rationale.append(f"Strength: {reason}.")
        for reason in primary.get("reasons_for", [])[:2]:
            rationale.append(reason)

    if secondary is not None:
        rationale.append(
            f"Secondary recommendation is {secondary['method']} with overall score {secondary['overall_score']:.4f}."
        )

    if avoid:
        rationale.append(f"Methods to avoid or deprioritize under current evidence: {', '.join(avoid)}.")

    return {
        "problem_features": features,
        "ranked_methods": scored_methods,
        "primary_recommendation": primary["method"] if primary else None,
        "secondary_recommendation": secondary["method"] if secondary else None,
        "avoid": avoid,
        "recommendation_confidence_score": round(recommendation_confidence_score, 6),
        "recommendation_confidence_band": _recommendation_confidence_band(recommendation_confidence_score),
        "rationale": rationale,
    }