from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal
import math

from numerical_lab.core.base import SolverResult


StabilityLabel = Literal[
    "stable",
    "possible_oscillation",
    "stagnation",
    "possible_divergence",
    "insufficient_data",
]


@dataclass
class StabilityReport:
    label: StabilityLabel
    notes: List[str]


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _median(vals: List[float]) -> float:
    s = sorted(vals)
    return s[len(s) // 2]


def detect_stability(result):
    """
    Returns a dict-like stability report used for diagnostics/badges.
    Must never throw.
    """
    notes = []
    flags = {
        "unstable": False,
        "notes": notes,
    }

    try:
        summary = getattr(result, "summary", None)
        status = getattr(summary, "status", None) if summary else None

        # Fallback: many of your Result objects store status directly
        if status is None:
            status = getattr(result, "status", None)

        max_iter_hit = False
        # You likely have these fields; if not, this stays False
        if hasattr(result, "hit_max_iter"):
            max_iter_hit = bool(getattr(result, "hit_max_iter"))
        elif hasattr(result, "iterations") and hasattr(result, "max_iter"):
            try:
                max_iter_hit = int(result.iterations) >= int(result.max_iter)
            except Exception:
                max_iter_hit = False

        if status in ("max_iter", "iteration_limit", "no_convergence") or max_iter_hit:
            notes.append("Solver terminated by iteration limit; convergence unclear.")

        # If events exist, mark instability based on severity patterns (optional)
        events = getattr(result, "events", None) or []
        if any(getattr(e, "severity", "") in ("warn", "error") for e in events):
            # don't automatically call it unstable; just add note
            notes.append("Warnings were emitted during the run (see event timeline).")

    except Exception as e:
        flags["unstable"] = True
        notes.append(f"detect_stability failed safely: {type(e).__name__}: {e}")

    return flags
    """
    Stability analysis (pedagogical + low false-positive):
    - oscillation heuristic: direction flips + residual not consistently decreasing
    - stagnation heuristic: flat step tail + no meaningful residual improvement
    - divergence heuristic: residuals grow materially from early to late
    """
    notes = []
    if getattr(result, "stop_reason", None) == "STAGNATION":
        return StabilityReport("stagnation",
                                ["Solver stopped due to stagnation (step size too small)."])

    if getattr(result, "stop_reason", None) == "MAX_ITER":
        notes.append("Solver terminated by iteration limit; convergence unclear.")
    
    xs = result.x_history
    res = result.residual_history
    errs = [e for e in result.step_error_history if e is not None and e > 0]

    if len(xs) < 4 or len(res) < 4:
        return StabilityReport("insufficient_data", ["Not enough iterations to analyze stability."])

    notes: List[str] = []

    # Helper: residual improvement check (robust)
    # "improving" means last residual is smaller than early median by a decent factor
    early = res[: max(2, len(res) // 3)]
    late = res[-max(2, len(res) // 3):]
    early_med = _median(early)
    late_med = _median(late)
    improving = (late_med < 0.5 * early_med) if early_med > 0 else False

    # 1) Oscillation: step direction flips repeatedly
    directions = [_sign(xs[i] - xs[i - 1]) for i in range(1, len(xs))]

    flip_count = 0
    nonzero_dirs = 0
    for i in range(1, len(directions)):
        if directions[i] != 0:
            nonzero_dirs += 1
        if directions[i] != 0 and directions[i - 1] != 0 and directions[i] != directions[i - 1]:
            flip_count += 1

    osc_flag = False
    if nonzero_dirs >= 3:
        flip_rate = flip_count / max(1, (len(directions) - 1))
        # Flag oscillation only if flips are frequent AND not clearly improving
        if flip_rate >= 0.45 and not improving:
            osc_flag = True
            notes.append("Frequent direction changes detected; possible oscillation.")
            notes.append("Try a bracketing method (bisection) or provide a better initial guess.")

    # 2) Stagnation: tail step errors nearly flat AND residual not improving
    stag_flag = False
    if len(errs) >= 6:
        tail = errs[-6:]
        tail_min = min(tail)
        tail_max = max(tail)

        # "flat" tail AND not already tiny
        flat_tail = (tail_min > 0 and (tail_max / tail_min) < 1.3)
        not_tiny = tail_max > max(1e-16, result.tol * 0.1)  # scale-aware

        if flat_tail and not_tiny and not improving:
            stag_flag = True
            notes.append("Step sizes are not decreasing much; possible stagnation.")
            notes.append("Try different starting points or switch to hybrid method.")

    # 3) Divergence: residual grows materially
    div_flag = False
    if len(res) >= 6 and early_med > 0:
        if late_med > 5.0 * early_med and result.status not in ("converged",):
            div_flag = True
            notes.append("Residuals increased significantly; possible divergence.")
            notes.append("Use bisection/hybrid, or choose an initial guess nearer the root.")

    # Priority
    if div_flag:
        return StabilityReport("possible_divergence", notes)
    if stag_flag:
        return StabilityReport("stagnation", notes)
    if osc_flag:
        return StabilityReport("possible_oscillation", notes)

    return StabilityReport("stable", ["No strong instability patterns detected."])