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
            "score": 0.0,
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
        "score": score,
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
        "score": score,
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
        "score": score,
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
        "score": score,
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
        "score": score,
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
            "score": 0.0,
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
        "score": score,
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
        "score": 0.30,
        "recommended": False,
        "reasons_for": [],
        "reasons_against": ["No rule-based suitability model is defined for this method yet."],
    }


def build_solver_selection_recommendation(
    *,
    expectations: Dict[str, Any],
    methods: Sequence[str],
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    features = _build_base_problem_features(expectations, metadata=metadata)

    scored_methods = [
        _score_method(method, features)
        for method in methods
    ]
    scored_methods.sort(key=lambda item: item.get("score", 0.0), reverse=True)

    primary = scored_methods[0] if scored_methods else None
    secondary = scored_methods[1] if len(scored_methods) > 1 else None

    avoid = [
        item["method"]
        for item in scored_methods
        if float(item.get("score", 0.0)) < 0.40
    ]

    recommendation_confidence_score = 0.0
    if primary is not None:
        top_score = float(primary.get("score", 0.0))
        second_score = float(secondary.get("score", 0.0)) if secondary is not None else 0.0
        separation = max(0.0, top_score - second_score)
        recommendation_confidence_score = _clamp(0.70 * top_score + 0.30 * separation)

    rationale: List[str] = []
    if primary is not None:
        rationale.append(
            f"Primary recommendation is {primary['method']} with suitability score {primary['score']:.4f}."
        )
        for reason in primary.get("reasons_for", [])[:3]:
            rationale.append(reason)

    if secondary is not None:
        rationale.append(
            f"Secondary recommendation is {secondary['method']} with suitability score {secondary['score']:.4f}."
        )

    if avoid:
        rationale.append(f"Methods to avoid or deprioritize under current analytic checks: {', '.join(avoid)}.")

    return {
        "problem_features": features,
        "ranked_methods": scored_methods,
        "primary_recommendation": primary["method"] if primary else None,
        "secondary_recommendation": secondary["method"] if secondary else None,
        "avoid": avoid,
        "recommendation_confidence_score": recommendation_confidence_score,
        "recommendation_confidence_band": _recommendation_confidence_band(recommendation_confidence_score),
        "rationale": rationale,
    }