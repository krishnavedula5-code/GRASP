from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional,Literal

from numerical_lab.core.base import SolverResult


@dataclass
class HybridDecisionReport:
    newton_attempts: int
    newton_accepts: int
    newton_rejects: int
    accept_rate: float
    reject_reasons: Dict[str, int]
    bisection_steps: int
    newton_steps: int
    DominantMode = Literal["newton-dominant", "bisection-dominant", "mixed"]
    dominant_mode: DominantMode
    note: Optional[str] = None


def hybrid_decision_report(result: SolverResult) -> HybridDecisionReport:
    # events are the cleanest source for attempt/accept/reject
    attempts = sum(1 for e in result.events if e.get("kind") == "newton_attempt")
    accepts = sum(1 for e in result.events if e.get("kind") == "newton_accept")
    rejects = sum(1 for e in result.events if e.get("kind") == "newton_reject")

    reasons: Dict[str, int] = {}
    for e in result.events:
        if e.get("kind") == "newton_reject":
            r = (e.get("data") or {}).get("reason", "unknown")
            reasons[str(r)] = reasons.get(str(r), 0) + 1

    b_steps = sum(1 for r in result.records if getattr(r, "step_type", None) == "bisection")
    n_steps = sum(1 for r in result.records if getattr(r, "step_type", None) == "newton")

    total = max(1, b_steps + n_steps)
    b_frac = b_steps / total
    n_frac = n_steps / total

    if n_frac >= 0.7:
        dom = "newton-dominant"
    elif b_frac >= 0.7:
        dom = "bisection-dominant"
    else:
        dom = "mixed"

    ar = (accepts / attempts) if attempts > 0 else 0.0

    note = None
    if attempts > 0 and ar < 0.2:
        note = "Newton was frequently rejected; bracketed safety dominated."
    if dom == "bisection-dominant":
        note = (note or "") + (" " if note else "") + "Hybrid behaved mostly like bisection."

    return HybridDecisionReport(
        newton_attempts=attempts,
        newton_accepts=accepts,
        newton_rejects=rejects,
        accept_rate=float(ar),
        reject_reasons=reasons,
        bisection_steps=b_steps,
        newton_steps=n_steps,
        dominant_mode=dom,
        note=note,
    )