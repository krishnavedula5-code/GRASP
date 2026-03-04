# numerical_lab/engine/trace.py
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from numerical_lab.engine.summary import MethodSummary
    from numerical_lab.core.base import SolverResult

SCHEMA_VERSION = "1.1"


def log_event(
    trace: Dict[str, Any],
    *,
    kind: str,
    data: Optional[Dict[str, Any]] = None,
    k: Optional[int] = None,
    code: Optional[str] = None,
    level: str = "info",
) -> None:
    """
    Deterministic event logger.
    Backward compatible with legacy events that had only {kind, data, k}.
    New fields:
      - code: stable machine code (e.g., NONFINITE, STEP_REJECTED)
      - level: info|warn|error
    """
    events = trace.setdefault("events", [])
    ev: Dict[str, Any] = {"kind": kind, "data": data or {}}

    if k is not None:
        ev["k"] = k
    if code is not None:
        ev["code"] = code
    if level is not None:
        ev["level"] = level

    events.append(ev)


def build_trace_payload(
    result: "SolverResult",
    method_summary: Optional["MethodSummary"] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a stable JSON payload for UI/classroom trace playback.
    """
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": result.method,
        "status": result.status,
        "stop_reason": getattr(result, "stop_reason", None),
        "message": result.message,
        "root": result.root,
        "iterations": result.iterations,
        "best_x": getattr(result, "best_x", None),
        "best_fx": getattr(result, "best_fx", None),
        "n_f": getattr(result, "n_f", None),
        "n_df": getattr(result, "n_df", None),
        "records": [asdict(r) for r in result.records],
        "events": result.events,
    }

    if method_summary is not None:
        payload["summary"] = asdict(method_summary)

    if extra:
        payload["extra"] = extra

    return payload


def export_trace_json(
    result: "SolverResult",
    filepath: str,
    method_summary: Optional["MethodSummary"] = None,
    extra: Optional[Dict[str, Any]] = None,
    indent: int = 2,
) -> None:
    """
    Export the trace payload to a JSON file.
    """
    payload = build_trace_payload(result, method_summary=method_summary, extra=extra)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)