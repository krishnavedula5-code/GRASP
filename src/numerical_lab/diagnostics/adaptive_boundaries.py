from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class CandidateInterval:
    left_x: float
    right_x: float
    left_label: str
    right_label: str
    reason: str
    coarse_width: float


@dataclass
class RefinedBoundary:
    left_x: float
    right_x: float
    estimated_x: float
    final_width: float
    depth_reached: int
    left_label: str
    right_label: str
    reason: str
    resolved: bool


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _normalize_root_label(root_value: Any, ndigits: int = 8) -> str:
    v = _safe_float(root_value)
    if v is None:
        return "no_root"
    return f"root:{round(v, ndigits)}"


def classify_record(row: Dict[str, Any], root_digits: int = 8) -> str:
    status = str(row.get("status", "unknown")).strip().lower()
    if status != "converged":
        return f"status:{status}"
    return _normalize_root_label(row.get("root"), ndigits=root_digits)


def detect_candidate_intervals(
    records: List[Dict[str, Any]],
    root_digits: int = 8,
    iteration_jump_threshold: int = 8,
) -> List[CandidateInterval]:
    rows = []

    for r in records:
        x0 = _safe_float(r.get("initial_x", r.get("x0")))
        if x0 is None:
            continue

        rr = dict(r)
        rr["_x0"] = x0
        rr["_label"] = classify_record(rr, root_digits=root_digits)
        rr["_iters"] = int(_safe_float(rr.get("iterations", 0), 0) or 0)
        rows.append(rr)

    rows.sort(key=lambda z: z["_x0"])

    out: List[CandidateInterval] = []

    for a, b in zip(rows[:-1], rows[1:]):
        left_x = a["_x0"]
        right_x = b["_x0"]
        left_label = a["_label"]
        right_label = b["_label"]

        reasons: List[str] = []

        if left_label != right_label:
            reasons.append("label_change")

        if str(a.get("status", "")).lower() != str(b.get("status", "")).lower():
            reasons.append("status_change")

        if abs(a["_iters"] - b["_iters"]) >= iteration_jump_threshold:
            reasons.append("iteration_jump")

        if not reasons:
            continue

        out.append(
            CandidateInterval(
                left_x=left_x,
                right_x=right_x,
                left_label=left_label,
                right_label=right_label,
                reason="|".join(sorted(set(reasons))),
                coarse_width=right_x - left_x,
            )
        )

    return out


def _normalize_result_payload(
    x: float,
    result: Any,
) -> Dict[str, Any]:
    """
    Normalize solver result object or dict to:
    {
      initial_x, status, root, iterations
    }
    """
    if isinstance(result, dict):
        status = result.get("status", "unknown")
        root = result.get("root")
        iterations = result.get("iterations", 0)
    else:
        status = getattr(result, "status", "unknown")
        root = getattr(result, "root", None)
        iterations = getattr(result, "iterations", 0)

    return {
        "initial_x": x,
        "status": status,
        "root": root,
        "iterations": iterations,
    }


def _make_local_bracket(
    f: Callable[[float], float],
    x: float,
    domain: Tuple[float, float],
    growth: float = 1.8,
    max_steps: int = 25,
    initial_half_width: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    """
    Try to build a local sign-changing bracket around x inside domain.
    """
    x_min, x_max = domain

    if initial_half_width is None:
        initial_half_width = 0.01 * max(1.0, x_max - x_min)

    half = initial_half_width

    for _ in range(max_steps):
        a = max(x_min, x - half)
        b = min(x_max, x + half)

        try:
            fa = f(a)
            fb = f(b)
            if math.isfinite(fa) and math.isfinite(fb) and fa * fb <= 0:
                return (a, b)
        except Exception:
            pass

        half *= growth

        if a <= x_min and b >= x_max:
            break

    return None


def _run_solver_at_x(
    engine: Any,
    f: Callable[[float], float],
    df: Optional[Callable[[float], float]],
    method: str,
    x: float,
    domain: Tuple[float, float],
    tol: float = 1e-10,
    max_iter: int = 100,
    secant_delta: float = 1e-3,
) -> Dict[str, Any]:
    method = str(method).strip().lower()

    if method == "newton":
        if df is None:
            return {
                "initial_x": x,
                "status": "missing_derivative",
                "root": None,
                "iterations": 0,
            }

        result, _, _ = engine.solve_newton(
            f,
            df,
            x0=x,
            tol=tol,
            max_iter=max_iter,
        )
        return _normalize_result_payload(x, result)

    if method == "secant":
        x_min, x_max = domain
        x1 = min(x_max, x + secant_delta)
        if x1 == x:
            x1 = max(x_min, x - secant_delta)

        result, _, _ = engine.solve_secant(
            f,
            x0=x,
            x1=x1,
            tol=tol,
            max_iter=max_iter,
        )
        return _normalize_result_payload(x, result)

    if method == "hybrid":
        if df is None:
            return {
                "initial_x": x,
                "status": "missing_derivative",
                "root": None,
                "iterations": 0,
            }

        bracket = _make_local_bracket(f, x, domain)
        if bracket is None:
            return {
                "initial_x": x,
                "status": "no_local_bracket",
                "root": None,
                "iterations": 0,
            }

        a, b = bracket
        result, _, _ = engine.solve_hybrid(
            f,
            df,
            a,
            b,
            tol=tol,
            max_iter=max_iter,
        )
        return _normalize_result_payload(x, result)

    if method == "safeguarded_newton":
        if df is None:
            return {
                "initial_x": x,
                "status": "missing_derivative",
                "root": None,
                "iterations": 0,
            }

        bracket = _make_local_bracket(f, x, domain)
        if bracket is None:
            return {
                "initial_x": x,
                "status": "no_local_bracket",
                "root": None,
                "iterations": 0,
            }

        a, b = bracket
        result, _, _ = engine.solve_safeguarded_newton(
            f,
            df,
            a,
            b,
            x0=x,
            tol=tol,
            max_iter=max_iter,
        )
        return _normalize_result_payload(x, result)

    if method == "bisection":
        return {
            "initial_x": x,
            "status": "unsupported_for_adaptive_refinement",
            "root": None,
            "iterations": 0,
        }

    return {
        "initial_x": x,
        "status": f"unknown_method:{method}",
        "root": None,
        "iterations": 0,
    }


def _classify_run_result(result: Dict[str, Any], root_digits: int = 8) -> str:
    return classify_record(result, root_digits=root_digits)


def refine_interval(
    engine: Any,
    f: Callable[[float], float],
    df: Optional[Callable[[float], float]],
    method: str,
    domain: Tuple[float, float],
    left_x: float,
    right_x: float,
    left_label: str,
    right_label: str,
    reason: str,
    tol_x: float = 1e-4,
    max_depth: int = 12,
    root_digits: int = 8,
    solve_tol: float = 1e-10,
    max_iter: int = 100,
) -> RefinedBoundary:
    a = float(left_x)
    b = float(right_x)
    la = left_label
    lb = right_label
    depth = 0

    while depth < max_depth and (b - a) > tol_x and la != lb:
        m = 0.5 * (a + b)

        mid_result = _run_solver_at_x(
            engine=engine,
            f=f,
            df=df,
            method=method,
            x=m,
            domain=domain,
            tol=solve_tol,
            max_iter=max_iter,
        )
        lm = _classify_run_result(mid_result, root_digits=root_digits)

        if lm == la:
            a = m
            la = lm
        elif lm == lb:
            b = m
            lb = lm
        else:
            # third behavior appeared in the midpoint
            # keep one informative side
            if la != lm:
                b = m
                lb = lm
            else:
                a = m
                la = lm

        depth += 1

    return RefinedBoundary(
        left_x=a,
        right_x=b,
        estimated_x=0.5 * (a + b),
        final_width=b - a,
        depth_reached=depth,
        left_label=la,
        right_label=lb,
        reason=reason,
        resolved=(la != lb),
    )


def summarize_refined_boundaries(boundaries: List[RefinedBoundary]) -> Dict[str, Any]:
    if not boundaries:
        return {
            "num_refined_boundaries": 0,
            "resolved_share": 0.0,
            "mean_final_width": None,
            "max_final_width": None,
            "mean_depth": None,
        }

    widths = [b.final_width for b in boundaries]
    depths = [b.depth_reached for b in boundaries]
    resolved = [1.0 if b.resolved else 0.0 for b in boundaries]

    return {
        "num_refined_boundaries": len(boundaries),
        "resolved_share": mean(resolved),
        "mean_final_width": mean(widths),
        "max_final_width": max(widths),
        "mean_depth": mean(depths),
    }


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def plot_adaptive_boundary_overlay(
    records: List[Dict[str, Any]],
    boundaries: List[RefinedBoundary],
    outpath: Path,
) -> None:
    xs = []
    ys = []

    for r in records:
        x0 = _safe_float(r.get("initial_x", r.get("x0")))
        it = _safe_float(r.get("iterations"))
        if x0 is None or it is None:
            continue
        xs.append(x0)
        ys.append(it)

    plt.figure(figsize=(10, 5))
    if xs and ys:
        plt.scatter(xs, ys, s=14, alpha=0.7)

    for b in boundaries:
        plt.axvspan(b.left_x, b.right_x, alpha=0.15)
        plt.axvline(b.estimated_x, linestyle="--", linewidth=1)

    plt.xlabel("Initial guess")
    plt.ylabel("Iterations")
    plt.title("Adaptive boundary refinement overlay")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def run_adaptive_boundary_refinement(
    *,
    records: List[Dict[str, Any]],
    engine: Any,
    f: Callable[[float], float],
    df: Optional[Callable[[float], float]],
    method: str,
    domain: Tuple[float, float],
    output_dir: str | Path,
    root_digits: int = 8,
    iteration_jump_threshold: int = 8,
    tol_x: float = 1e-4,
    max_depth: int = 12,
    solve_tol: float = 1e-10,
    max_iter: int = 100,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = detect_candidate_intervals(
        records=records,
        root_digits=root_digits,
        iteration_jump_threshold=iteration_jump_threshold,
    )

    refined: List[RefinedBoundary] = []
    for c in candidates:
        rb = refine_interval(
            engine=engine,
            f=f,
            df=df,
            method=method,
            domain=domain,
            left_x=c.left_x,
            right_x=c.right_x,
            left_label=c.left_label,
            right_label=c.right_label,
            reason=c.reason,
            tol_x=tol_x,
            max_depth=max_depth,
            root_digits=root_digits,
            solve_tol=solve_tol,
            max_iter=max_iter,
        )
        refined.append(rb)

    summary = summarize_refined_boundaries(refined)

    candidates_json = [asdict(c) for c in candidates]
    refined_json = [asdict(b) for b in refined]

    save_json(output_dir / "adaptive_boundary_candidates.json", candidates_json)
    save_json(output_dir / "adaptive_boundaries.json", refined_json)
    save_json(output_dir / "adaptive_boundary_summary.json", summary)

    plot_adaptive_boundary_overlay(
        records=records,
        boundaries=refined,
        outpath=output_dir / "adaptive_boundary_overlay.png",
    )

    return {
        "candidates_path": str(output_dir / "adaptive_boundary_candidates.json"),
        "boundaries_path": str(output_dir / "adaptive_boundaries.json"),
        "summary_path": str(output_dir / "adaptive_boundary_summary.json"),
        "overlay_path": str(output_dir / "adaptive_boundary_overlay.png"),
        "summary": summary,
    }