from __future__ import annotations

from typing import List, Optional, Dict, Any
import math

from numerical_lab.core.base import SolverResult
from numerical_lab.engine.summary import MethodSummary


def _median(vals: List[float]) -> Optional[float]:
    xs = sorted(vals)
    if not xs:
        return None
    n = len(xs)
    mid = n // 2
    return xs[mid] if n % 2 == 1 else 0.5 * (xs[mid - 1] + xs[mid])


def _event_counts(events: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for e in events:
        k = e.get("kind", "unknown")
        counts[k] = counts.get(k, 0) + 1
    return counts


def _newton_reject_reasons(events: List[Dict[str, Any]]) -> Dict[str, int]:
    reasons: Dict[str, int] = {}
    for e in events:
        if e.get("kind") == "newton_reject":
            r = (e.get("data") or {}).get("reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1
    return reasons


def explain_run(summary: MethodSummary, result: SolverResult) -> str:
    """
    Deterministic, student-facing explanation using:
      - summary (convergence/stability)
      - iteration history
      - event log (hybrid tutoring)
    """
    parts: List[str] = []

    parts.append(f"{summary.method.upper()} — status: {summary.status}.")
    if summary.root is not None:
        parts.append(f"Root ≈ {summary.root:.12g}.")
    parts.append(f"Iterations: {summary.iterations}.")

    # Residual progress
    res = result.residual_history or []
    if res:
        r0, rN = res[0], res[-1]
        parts.append(f"Final |f(x)| = {rN:.3e}.")
        if r0 > 0 and rN > 0:
            overall_ratio = r0 / rN if rN != 0 else float("inf")
            if math.isfinite(overall_ratio) and overall_ratio >= 10:
                parts.append(f"Residual improved by ~{overall_ratio:.2g}x overall.")

    # Residual monotonic trend
    if len(res) >= 6:
        dec = sum(1 for i in range(1, len(res)) if res[i] <= res[i - 1])
        frac_dec = dec / (len(res) - 1)
        if frac_dec >= 0.8:
            parts.append("Residual decreased consistently (stable progress).")
        elif frac_dec <= 0.4:
            parts.append("Residual often increased (instability or poor initial guess likely).")

    # Step size shrink
    errs = [e for e in result.step_error_history if e is not None and e > 0]
    if len(errs) >= 6:
        early = errs[: max(2, len(errs) // 3)]
        late = errs[-max(2, len(errs) // 3):]
        e_med_early = _median(early)
        e_med_late = _median(late)
        if e_med_early and e_med_late and e_med_late < 0.1 * e_med_early:
            parts.append("Step sizes shrank significantly (approaching the root).")

    # Convergence classification
    if summary.convergence_class != "unknown":
        parts.append(f"Global convergence behavior: {summary.convergence_class.replace('_',' ')}."
        )
    # Estimated order
    if summary.observed_order is not None:
        parts.append(f"Asymptotic order estimate: p ≈ {summary.observed_order:.3f}.")

        # --- Transient spike / basin-entry detection ---
    try:
        res = result.residual_history or []
        spike_info = None

        for i in range(1, len(res)):
            prev_r = res[i - 1]
            curr_r = res[i]

            if prev_r > 0 and curr_r >= 5 * prev_r:
                spike_info = (i, prev_r, curr_r, curr_r / prev_r)
                break

        if spike_info:
            k, prev_r, curr_r, ratio = spike_info

            order = summary.observed_order

            if order is not None and order >= 1.8:
                regime_text = "the local quadratic convergence basin"
            else:
                regime_text = "the local convergence regime"

            parts.append(
                f"A transient residual spike was detected at iteration {k} "
                f"(increase by factor {ratio:.1f}). "
                f"This reflects pre-asymptotic global dynamics before the iteration enters {regime_text}."
            )

    except Exception:
        pass
    # Method-specific teaching notes
    m = summary.method.lower()
    if m == "bisection":
        parts.append("Bisection is guaranteed with a valid bracket and converges linearly by interval halving.")
    elif m == "newton":
        parts.append("Newton is fast near the root but requires a good initial guess and a nonzero derivative.")
    elif m == "secant":
        parts.append("Secant is derivative-free and often superlinear, but may be unstable for poor starting points.")
    elif m == "hybrid":
        parts.append("Hybrid maintains a bracket for safety and uses Newton steps when reliable.")

    # Hybrid event-aware explanation
    events = result.events or []
    if m == "hybrid" and events:
        counts = _event_counts(events)
        attempts = counts.get("newton_attempt", 0)
        accepts = counts.get("newton_accept", 0)
        bis_steps = counts.get("bisection_step", 0)

        parts.append(
            f"Hybrid behavior: Newton accepted {accepts}/{attempts} attempts; "
            f"bisection fallback used {bis_steps} times."
        )

        reasons = _newton_reject_reasons(events)
        if reasons:
            top = sorted(reasons.items(), key=lambda kv: -kv[1])
            top_str = ", ".join(f"{k}:{v}" for k, v in top[:3])
            parts.append(f"Top Newton rejection reasons: {top_str}.")

            dominant = top[0][0]
            if dominant == "df_small_or_invalid":
                parts.append("Action: derivative near zero or invalid — switch to bisection or adjust bracket.")
            elif dominant == "step_outside_bracket":
                parts.append("Action: Newton step left bracket — hybrid or closer initial guess recommended.")
            elif dominant == "no_residual_improvement":
                parts.append("Action: Newton did not reduce residual — try different initial guess.")

    # Stability flag
    if summary.stability_label != "stable":
        parts.append(f"Stability flag: {summary.stability_label.replace('_', ' ')}.")

    # Additional diagnostic notes
    for note in summary.key_notes:
        if note:
            parts.append(note)

    # Final recommendation if not converged
    if summary.status != "converged":
        if summary.stability_label in ("possible_oscillation", "possible_divergence"):
            parts.append("Action: switch to bisection or hybrid, or move initial guess closer to the root.")
        else:
            parts.append("Action: try different initial guesses or increase max_iter after checking stability.")

    return " ".join(parts)