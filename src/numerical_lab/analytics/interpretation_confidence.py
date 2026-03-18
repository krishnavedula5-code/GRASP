from __future__ import annotations

from typing import Any, Dict


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def _band_to_numeric_risk(band: str) -> float:
    band = str(band or "").strip().lower()
    mapping = {
        "low": 0.15,
        "moderate": 0.45,
        "high": 0.75,
    }
    return mapping.get(band, 0.50)


def _compute_mismatch_severity(
    predicted_risk_band: str,
    observed_failure_fraction: float,
) -> float:
    predicted = _band_to_numeric_risk(predicted_risk_band)
    observed = _clamp(observed_failure_fraction)
    return _clamp(abs(predicted - observed))


def _compute_sample_strength(
    sample_count: int,
    unknown_fraction: float,
) -> float:
    # Saturating sample-size score:
    # 0 at 0 samples, approaches 1 as count grows.
    sample_score = sample_count / (sample_count + 100.0) if sample_count > 0 else 0.0

    # Unknown statuses reduce confidence.
    knownness_score = 1.0 - _clamp(unknown_fraction)

    return _clamp(0.70 * sample_score + 0.30 * knownness_score)


def _compute_pattern_strength(
    observed_failure_fraction: float,
    observed_success_fraction: float,
) -> float:
    failure = _clamp(observed_failure_fraction)
    success = _clamp(observed_success_fraction)

    # Stronger when one signal is clearly dominant.
    return _clamp(abs(success - failure))


def _compute_interpretation_confidence(
    *,
    mismatch_severity: float,
    sample_strength: float,
    pattern_strength: float,
) -> float:
    # Confidence should go up with:
    # - more data
    # - clearer outcome pattern
    # - lower mismatch between prediction and observation
    agreement = 1.0 - _clamp(mismatch_severity)

    score = (
        0.45 * sample_strength
        + 0.30 * agreement
        + 0.25 * pattern_strength
    )
    return _clamp(score)


def _confidence_band(score: float) -> str:
    score = _clamp(score)
    if score < 0.40:
        return "low"
    if score < 0.70:
        return "moderate"
    return "high"


def _agreement_label(mismatch_severity: float) -> str:
    mismatch_severity = _clamp(mismatch_severity)
    if mismatch_severity < 0.15:
        return "strong"
    if mismatch_severity < 0.35:
        return "partial"
    return "weak"


def build_method_interpretation_confidence(
    *,
    predicted_risk_band: str,
    observed_failure_fraction: float,
    observed_success_fraction: float,
    sample_count: int,
    unknown_fraction: float = 0.0,
) -> Dict[str, Any]:
    mismatch_severity = _compute_mismatch_severity(
        predicted_risk_band=predicted_risk_band,
        observed_failure_fraction=observed_failure_fraction,
    )

    sample_strength = _compute_sample_strength(
        sample_count=sample_count,
        unknown_fraction=unknown_fraction,
    )

    pattern_strength = _compute_pattern_strength(
        observed_failure_fraction=observed_failure_fraction,
        observed_success_fraction=observed_success_fraction,
    )

    confidence_score = _compute_interpretation_confidence(
        mismatch_severity=mismatch_severity,
        sample_strength=sample_strength,
        pattern_strength=pattern_strength,
    )

    return {
        "predicted_risk_band": predicted_risk_band,
        "observed_failure_fraction": _clamp(observed_failure_fraction),
        "observed_success_fraction": _clamp(observed_success_fraction),
        "sample_count": int(sample_count),
        "unknown_fraction": _clamp(unknown_fraction),
        "mismatch_severity": mismatch_severity,
        "sample_strength": sample_strength,
        "pattern_strength": pattern_strength,
        "confidence_score": confidence_score,
        "confidence_band": _confidence_band(confidence_score),
        "agreement_label": _agreement_label(mismatch_severity),
    }