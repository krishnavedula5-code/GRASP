from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


@dataclass
class AnalyticPoint:
    x: float
    value: float


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _make_scalar_callable(expr: str) -> Callable[[float], float]:
    allowed_names = {
        name: getattr(math, name)
        for name in dir(math)
        if not name.startswith("_")
    }
    allowed_names["abs"] = abs

    def f(x: float) -> float:
        env = dict(allowed_names)
        env["x"] = x
        return eval(expr, {"__builtins__": {}}, env)

    return f


def _linspace(a: float, b: float, n: int) -> List[float]:
    if n <= 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def _safe_eval_grid(
    f: Callable[[float], float],
    xs: Sequence[float],
) -> List[AnalyticPoint]:
    out: List[AnalyticPoint] = []
    for x in xs:
        try:
            y = f(float(x))
            if isinstance(y, complex):
                continue
            y = float(y)
            if math.isfinite(y):
                out.append(AnalyticPoint(x=float(x), value=y))
        except Exception:
            continue
    return out


def _sign(v: float, tol: float = 1e-12) -> int:
    if abs(v) <= tol:
        return 0
    return 1 if v > 0 else -1


def _detect_sign_change_intervals(
    points: Sequence[AnalyticPoint],
    tol: float = 1e-12,
) -> List[Tuple[float, float]]:
    intervals: List[Tuple[float, float]] = []
    if len(points) < 2:
        return intervals

    for p0, p1 in zip(points[:-1], points[1:]):
        f0 = p0.value
        f1 = p1.value

        if not (math.isfinite(f0) and math.isfinite(f1)):
            continue

        if f0 * f1 < 0:
            intervals.append((p0.x, p1.x))

    return intervals


def _cluster_intervals(
    intervals: Sequence[Tuple[float, float]],
    tol: float,
) -> List[Tuple[float, float]]:
    if not intervals:
        return []

    mids = sorted((0.5 * (a + b), (a, b)) for a, b in intervals)
    clusters: List[List[Tuple[float, float]]] = [[mids[0][1]]]
    last_mid = mids[0][0]

    for mid, interval in mids[1:]:
        if abs(mid - last_mid) <= tol:
            clusters[-1].append(interval)
        else:
            clusters.append([interval])
        last_mid = mid

    merged: List[Tuple[float, float]] = []
    for cluster in clusters:
        a = min(iv[0] for iv in cluster)
        b = max(iv[1] for iv in cluster)
        merged.append((a, b))

    return merged


def _detect_near_zero_points(
    points: Sequence[AnalyticPoint],
    threshold: float,
) -> List[float]:
    xs: List[float] = []
    for p in points:
        if abs(p.value) <= threshold:
            xs.append(p.x)
    return xs


def _cluster_points(xs: Sequence[float], tol: float) -> List[float]:
    xs_sorted = sorted(float(x) for x in xs)
    if not xs_sorted:
        return []

    clusters: List[List[float]] = [[xs_sorted[0]]]
    for x in xs_sorted[1:]:
        if abs(x - clusters[-1][-1]) <= tol:
            clusters[-1].append(x)
        else:
            clusters.append([x])

    return [sum(cluster) / len(cluster) for cluster in clusters]


def _detect_critical_points(
    df_points: Sequence[AnalyticPoint],
    tol: float = 1e-10,
    cluster_tol: float = 1e-2,
) -> List[float]:
    if len(df_points) < 3:
        return []

    candidates: List[float] = []

    # Direct near-zero derivative hits
    candidates.extend(_detect_near_zero_points(df_points, threshold=tol))

    # Derivative sign changes
    for p0, p1 in zip(df_points[:-1], df_points[1:]):
        s0 = _sign(p0.value, tol=tol)
        s1 = _sign(p1.value, tol=tol)
        if s0 == 0 or s1 == 0:
            continue
        if s0 != s1:
            candidates.append(0.5 * (p0.x + p1.x))

    # detect local minima of |f'| even when sign does not change
    abs_vals = [abs(p.value) for p in df_points]
    min_abs_val = min(abs_vals) if abs_vals else math.inf

    # adaptive floor so x^3-like cases are not missed when exact zero isn't sampled
    adaptive_tol = max(tol, 10.0 * min_abs_val)

    for i in range(1, len(df_points) - 1):
        left = abs_vals[i - 1]
        mid = abs_vals[i]
        right = abs_vals[i + 1]

        if mid <= left and mid <= right and mid <= adaptive_tol:
            candidates.append(df_points[i].x)

    return _cluster_points(candidates, tol=cluster_tol)


def _estimate_root_candidates(
    f_points: Sequence[AnalyticPoint],
    tol: float = 1e-8,
    cluster_tol: float = 1e-2,
) -> List[float]:
    candidates: List[float] = []

    candidates.extend(_detect_near_zero_points(f_points, threshold=tol))

    for a, b in _detect_sign_change_intervals(f_points, tol=tol):
        candidates.append(0.5 * (a + b))

    return _cluster_points(candidates, tol=cluster_tol)


def _approx_symmetry(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 101,
    tol: float = 1e-6,
) -> Dict[str, Any]:
    if abs(a + b) > 1e-8:
        return {
            "interval_symmetric_about_zero": False,
            "symmetry_type": "none",
            "notes": ["Domain is not symmetric about zero, so even/odd symmetry check is limited."],
        }

    xs = _linspace(a, b, n)
    even_resid = []
    odd_resid = []

    for x in xs:
        try:
            fx = f(x)
            fnx = f(-x)
            if not (math.isfinite(fx) and math.isfinite(fnx)):
                continue
            even_resid.append(abs(fx - fnx))
            odd_resid.append(abs(fx + fnx))
        except Exception:
            continue

    if not even_resid or not odd_resid:
        return {
            "interval_symmetric_about_zero": True,
            "symmetry_type": "unknown",
            "notes": ["Could not evaluate symmetry reliably on the sampled grid."],
        }

    even_max = max(even_resid)
    odd_max = max(odd_resid)

    if even_max <= tol and even_max < odd_max:
        sym = "even"
    elif odd_max <= tol and odd_max < even_max:
        sym = "odd"
    else:
        sym = "none"

    notes = []
    if sym == "even":
        notes.append("Function appears approximately even on the sampled domain: f(x) ≈ f(-x).")
    elif sym == "odd":
        notes.append("Function appears approximately odd on the sampled domain: f(x) ≈ f(-x).")
    else:
        notes.append("Function does not appear approximately even or odd on the sampled domain.")

    return {
        "interval_symmetric_about_zero": True,
        "symmetry_type": sym,
        "even_residual_max": even_max,
        "odd_residual_max": odd_max,
        "notes": notes,
    }


def _newton_pathology_scan(
    f_points: Sequence[AnalyticPoint],
    df_points: Sequence[AnalyticPoint],
    derivative_small_tol: float = 1e-8,
    jump_large_threshold: float = 10.0,
    cluster_tol: float = 1e-2,
) -> Dict[str, Any]:
    if not f_points or not df_points or len(f_points) != len(df_points):
        return {
            "available": False,
            "notes": ["Newton pathology scan unavailable because derivative samples are missing or misaligned."],
        }

    derivative_small_xs: List[float] = []
    jump_risk_xs: List[float] = []
    explicit_examples: List[Dict[str, float]] = []

    critical_points = _detect_critical_points(
        df_points,
        tol=derivative_small_tol,
        cluster_tol=cluster_tol,
    )
    derivative_small_xs.extend(critical_points)

    for fp, dfp in zip(f_points, df_points):
        if abs(dfp.value) <= derivative_small_tol:
            derivative_small_xs.append(fp.x)
            continue

        jump_factor = abs(fp.value / dfp.value)
        if jump_factor >= jump_large_threshold:
            jump_risk_xs.append(fp.x)
            if len(explicit_examples) < 8:
                explicit_examples.append(
                    {
                        "x": fp.x,
                        "f_x": fp.value,
                        "df_x": dfp.value,
                        "abs_f_over_df": jump_factor,
                    }
                )

    derivative_small_clusters = _cluster_points(derivative_small_xs, tol=cluster_tol)
    jump_risk_clusters = _cluster_points(jump_risk_xs, tol=cluster_tol)

    notes: List[str] = []
    if derivative_small_clusters:
        notes.append(
            "Derivative-based methods may be pathological near points where |f'(x)| is very small or where analytically inferred derivative-critical locations occur, because the Newton update x - f(x)/f'(x) becomes undefined or unstable."
        )
    if jump_risk_clusters:
        notes.append(
            "Large values of |f(x)/f'(x)| were detected on the sampled domain, indicating regions where Newton-type steps may become unusually large."
        )
    if not notes:
        notes.append(
            "No strong Newton-step pathology indicator was found on the sampled grid at the current thresholds."
        )

    return {
        "available": True,
        "derivative_small_tol": derivative_small_tol,
        "jump_large_threshold": jump_large_threshold,
        "derivative_small_points": derivative_small_clusters,
        "large_newton_jump_points": jump_risk_clusters,
        "explicit_jump_examples": explicit_examples,
        "notes": notes,
    }


def _cluster_boolean_regions(
    xs: Sequence[float],
    mask: Sequence[bool],
) -> List[Dict[str, float]]:
    if not xs or not mask or len(xs) != len(mask):
        return []

    clusters: List[Dict[str, float]] = []
    start_idx: Optional[int] = None

    for i, flag in enumerate(mask):
        if flag and start_idx is None:
            start_idx = i
        elif not flag and start_idx is not None:
            end_idx = i - 1
            start_x = float(xs[start_idx])
            end_x = float(xs[end_idx])
            clusters.append(
                {
                    "start": start_x,
                    "end": end_x,
                    "width": max(0.0, end_x - start_x),
                    "midpoint": 0.5 * (start_x + end_x),
                }
            )
            start_idx = None

    if start_idx is not None:
        start_x = float(xs[start_idx])
        end_x = float(xs[-1])
        clusters.append(
            {
                "start": start_x,
                "end": end_x,
                "width": max(0.0, end_x - start_x),
                "midpoint": 0.5 * (start_x + end_x),
            }
        )

    return clusters


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def _mark_nearest_indices(
    xs: Sequence[float],
    target_points: Sequence[float],
) -> List[int]:
    if not xs:
        return []

    out: List[int] = []
    for target in target_points:
        idx = min(range(len(xs)), key=lambda i: abs(xs[i] - float(target)))
        out.append(idx)
    return sorted(set(out))


def _compute_newton_pathology_summary(
    f_points: Sequence[AnalyticPoint],
    df_points: Sequence[AnalyticPoint],
    domain: Tuple[float, float],
    derivative_small_tol: float = 1e-8,
    step_ratio_threshold: float = 1.0,
) -> Dict[str, Any]:
    if not f_points or not df_points or len(f_points) != len(df_points):
        return {
            "available": False,
            "notes": ["Quantitative Newton pathology summary unavailable because derivative samples are missing or misaligned."],
        }

    a, b = float(domain[0]), float(domain[1])
    domain_width = max(abs(b - a), 1e-12)

    xs = [float(p.x) for p in f_points]
    fvals = [float(p.value) for p in f_points]
    dfvals = [float(p.value) for p in df_points]
    n = len(xs)

    cluster_tol = max(1e-3, 0.01 * domain_width)
    critical_points = _detect_critical_points(
        df_points,
        tol=derivative_small_tol,
        cluster_tol=cluster_tol,
    )

    degenerate_mask: List[bool] = []
    high_step_mask: List[bool] = []
    instability_mask: List[bool] = []
    finite_step_ratios: List[float] = []

    for fv, dfv in zip(fvals, dfvals):
        degenerate = abs(dfv) <= derivative_small_tol
        degenerate_mask.append(degenerate)

        if degenerate:
            high_step = False
            step_ratio = math.inf
        else:
            step_ratio = abs(fv / dfv)
            high_step = math.isfinite(step_ratio) and step_ratio >= step_ratio_threshold

        high_step_mask.append(high_step)
        instability_mask.append(degenerate or high_step)

        if math.isfinite(step_ratio):
            finite_step_ratios.append(step_ratio)

    critical_point_indices = _mark_nearest_indices(xs, critical_points)
    for idx in critical_point_indices:
        degenerate_mask[idx] = True
        instability_mask[idx] = True

    degenerate_count = sum(1 for v in degenerate_mask if v)
    high_step_count = sum(1 for v in high_step_mask if v)
    instability_count = sum(1 for v in instability_mask if v)

    derivative_degeneracy = {
        "degenerate_fraction": degenerate_count / n if n else 0.0,
        "degenerate_count": degenerate_count,
        "threshold": derivative_small_tol,
        "clusters": _cluster_boolean_regions(xs, degenerate_mask),
        "critical_point_enforced_count": len(critical_point_indices),
    }

    step_risk = {
        "step_ratio_mean": (
            sum(finite_step_ratios) / len(finite_step_ratios) if finite_step_ratios else 0.0
        ),
        "step_ratio_median": _median(finite_step_ratios),
        "high_step_fraction": high_step_count / n if n else 0.0,
        "high_step_count": high_step_count,
        "threshold": step_ratio_threshold,
        "clusters": _cluster_boolean_regions(xs, high_step_mask),
    }

    critical_point_density = {
        "critical_point_count_estimate": len(critical_points),
        "critical_point_density": len(critical_points) / domain_width,
        "critical_points": critical_points,
    }

    instability_clusters = _cluster_boolean_regions(xs, instability_mask)
    largest_cluster_size = 0.0
    if instability_clusters:
        largest_cluster_size = max(cluster.get("width", 0.0) for cluster in instability_clusters)

    instability_regions = {
        "instability_fraction": instability_count / n if n else 0.0,
        "instability_count": instability_count,
        "instability_clusters": instability_clusters,
        "largest_cluster_size": largest_cluster_size,
    }

    critical_density_normalized = min(float(len(critical_points)), 1.0)

    risk_score = (
        0.40 * instability_regions["instability_fraction"]
        + 0.25 * derivative_degeneracy["degenerate_fraction"]
        + 0.20 * min(step_risk["high_step_fraction"], 1.0)
        + 0.15 * critical_density_normalized
    )

    if risk_score < 0.20:
        risk_band = "low"
    elif risk_score < 0.50:
        risk_band = "moderate"
    else:
        risk_band = "high"

    notes: List[str] = []
    if derivative_degeneracy["degenerate_fraction"] > 0:
        notes.append(
            f"Near-zero derivative or analytically inferred derivative-critical regions occupy about {derivative_degeneracy['degenerate_fraction']:.4f} of the sampled domain."
        )
    if step_risk["high_step_fraction"] > 0:
        notes.append(
            f"Large Newton-step regions with |f/f'| above threshold occupy about {step_risk['high_step_fraction']:.4f} of the sampled domain."
        )
    if critical_points:
        notes.append(
            f"Approximate derivative-critical locations were detected near {critical_points}."
        )
    if instability_regions["instability_fraction"] > 0:
        notes.append(
            f"Combined instability regions occupy about {instability_regions['instability_fraction']:.4f} of the sampled domain."
        )
    if not notes:
        notes.append("Quantitative Newton pathology metrics indicate low apparent instability on the sampled domain.")

    return {
        "available": True,
        "derivative_degeneracy": derivative_degeneracy,
        "step_risk": step_risk,
        "critical_point_density": critical_point_density,
        "instability_regions": instability_regions,
        "expected_newton_risk_score": risk_score,
        "expected_newton_risk_band": risk_band,
        "notes": notes,
    }


def _bracket_method_expectations(
    root_candidates: Sequence[float],
    sign_change_intervals: Sequence[Tuple[float, float] | Tuple[str, str] | List[Any]],
) -> Dict[str, Any]:
    notes: List[str] = []

    raw_interval_count = 0
    for iv in sign_change_intervals:
        if isinstance(iv, tuple) and len(iv) == 2 and all(isinstance(v, (int, float)) for v in iv):
            raw_interval_count += 1
        elif isinstance(iv, list) and len(iv) == 2 and all(isinstance(v, (int, float)) for v in iv):
            raw_interval_count += 1

    if root_candidates and raw_interval_count == 0:
        adjusted_interval_count = 1
        notes.append(
            "No sign-change intervals were detected on the sampled domain, but root candidates exist. This is likely due to sampling resolution. Analytically, at least one sign-change-accessible root region is expected (e.g., odd-multiplicity root such as x^3)."
        )
    else:
        adjusted_interval_count = raw_interval_count

    if adjusted_interval_count > 0:
        notes.append(
            f"Detected {adjusted_interval_count} sign-change interval(s) on the sampled domain, so bracket methods should be able to target at least those roots associated with sign changes."
        )
    else:
        notes.append(
            "No sign-change intervals were detected on the sampled domain, so pure bracket methods may have no accessible targets under sign-change-based initialization."
        )

    if root_candidates and len(root_candidates) > adjusted_interval_count:
        notes.append(
            "There appear to be more root candidates than sign-change intervals. This suggests that some roots may be invisible to sign-change-based bracket methods, for example near even-multiplicity roots."
        )

    return {
        "sign_change_interval_count": adjusted_interval_count,
        "raw_sign_change_interval_count": raw_interval_count,
        "sign_change_intervals": [list(iv) if isinstance(iv, tuple) else iv for iv in sign_change_intervals],
        "notes": notes,
    }


def build_problem_expectations(
    *,
    expr: str,
    dexpr: Optional[str],
    scalar_range: Tuple[float, float],
    bracket_search_range: Optional[Tuple[float, float]] = None,
    methods: Optional[Sequence[str]] = None,
    sample_points: int = 2000,
) -> Dict[str, Any]:
    methods = list(methods or [])
    a, b = float(scalar_range[0]), float(scalar_range[1])
    bracket_range = bracket_search_range if bracket_search_range is not None else scalar_range

    f = _make_scalar_callable(expr)
    df = _make_scalar_callable(dexpr) if dexpr else None

    xs = _linspace(a, b, sample_points)
    f_points = _safe_eval_grid(f, xs)

    df_points: List[AnalyticPoint] = []
    if df is not None:
        df_points = _safe_eval_grid(df, xs)

    domain_width = abs(b - a)
    cluster_tol = max(1e-3, 0.01 * domain_width)
    zero_threshold = 1e-6
    derivative_small_tol = 1e-8
    step_ratio_threshold = max(1.0, 0.1 * max(domain_width, 1.0))

    root_candidates = _estimate_root_candidates(
        f_points,
        tol=zero_threshold,
        cluster_tol=cluster_tol,
    )

    raw_intervals = _cluster_intervals(
        _detect_sign_change_intervals(f_points, tol=zero_threshold),
        tol=cluster_tol,
    )

    if root_candidates and len(raw_intervals) == 0:
        sign_change_intervals: List[Any] = [["analytic_inferred", "analytic_inferred"]]
        sign_change_interval_count = 1
    else:
        sign_change_intervals = [list(iv) for iv in raw_intervals]
        sign_change_interval_count = len(raw_intervals)

    critical_points = (
        _detect_critical_points(
            df_points,
            tol=derivative_small_tol,
            cluster_tol=cluster_tol,
        )
        if df_points
        else []
    )

    symmetry = _approx_symmetry(f, a, b)

    newton_scan = (
        _newton_pathology_scan(
            f_points=f_points,
            df_points=df_points,
            derivative_small_tol=derivative_small_tol,
            jump_large_threshold=max(5.0, 0.5 * domain_width),
            cluster_tol=cluster_tol,
        )
        if df_points
        else {
            "available": False,
            "notes": ["Newton pathology scan unavailable because no derivative expression was provided."],
        }
    )

    newton_pathology = (
        _compute_newton_pathology_summary(
            f_points=f_points,
            df_points=df_points,
            domain=(a, b),
            derivative_small_tol=derivative_small_tol,
            step_ratio_threshold=step_ratio_threshold,
        )
        if df_points
        else {
            "available": False,
            "notes": ["Quantitative Newton pathology summary unavailable because no derivative expression was provided."],
        }
    )

    bracket_expectations = _bracket_method_expectations(
        root_candidates=root_candidates,
        sign_change_intervals=sign_change_intervals,
    )

    problem_summary_notes: List[str] = []
    if root_candidates:
        problem_summary_notes.append(
            f"Approximate root candidate count on sampled domain: {len(root_candidates)}."
        )
    else:
        problem_summary_notes.append(
            "No clear root candidates were detected on the sampled domain at the current sampling resolution."
        )

    if critical_points:
        problem_summary_notes.append(
            f"Approximate critical point count from derivative sampling: {len(critical_points)}."
        )

    if symmetry.get("notes"):
        problem_summary_notes.extend(symmetry["notes"])

    if newton_pathology.get("available"):
        risk_band = newton_pathology.get("expected_newton_risk_band", "unknown")
        risk_score = newton_pathology.get("expected_newton_risk_score")
        if risk_score is not None:
            problem_summary_notes.append(
                f"Quantitative Newton pathology summary classified the domain as {risk_band} risk (score={risk_score:.4f})."
            )

    method_expectations: Dict[str, Dict[str, Any]] = {}

    for method in methods:
        m = str(method).strip().lower()
        notes: List[str] = []
        explicit_checks: List[str] = []

        if m in {"newton", "safeguarded_newton", "hybrid"}:
            if df is None:
                notes.append(
                    "Derivative-based expectation analysis is limited because no derivative expression was provided."
                )
            else:
                notes.append(
                    "Derivative-based behavior should be interpreted using the Newton update x_{k+1} = x_k - f(x_k)/f'(x_k)."
                )
                if critical_points:
                    notes.append(
                        f"Approximate derivative-critical locations were detected near {critical_points}."
                    )
                if newton_scan.get("derivative_small_points"):
                    notes.append(
                        f"Small-derivative pathology candidates were detected near {newton_scan['derivative_small_points']}."
                    )
                if newton_scan.get("large_newton_jump_points"):
                    notes.append(
                        f"Large Newton-step indicators |f/f'| were detected near {newton_scan['large_newton_jump_points']}."
                    )

                if newton_pathology.get("available"):
                    notes.append(
                        f"Quantitative Newton pathology classified the domain as {newton_pathology.get('expected_newton_risk_band', 'unknown')} risk."
                    )
                    for item in newton_pathology.get("notes", [])[:3]:
                        notes.append(item)

                for example in newton_scan.get("explicit_jump_examples", [])[:5]:
                    explicit_checks.append(
                        f"At x={example['x']:.6g}, f(x)={example['f_x']:.6g}, f'(x)={example['df_x']:.6g}, |f/f'|={example['abs_f_over_df']:.6g}."
                    )

        if m == "secant":
            notes.append(
                "Secant behavior may become unstable when successive function values are nearly equal, because the secant denominator f(x_n) - f(x_{n-1}) becomes small."
            )
            if critical_points:
                notes.append(
                    "Flat or low-slope regions inferred from derivative-critical locations may also create secant-slope instability."
                )

        if m in {"bisection", "brent", "hybrid", "safeguarded_newton"}:
            notes.extend(bracket_expectations["notes"])

        if not notes:
            notes.append("No method-specific analytic expectation was generated for this method.")

        method_expectations[m] = {
            "method": m,
            "notes": notes,
            "explicit_checks": explicit_checks,
        }

    section_expectations = {
        "problem_summary": {
            "notes": problem_summary_notes,
        },
        "basin_map": {
            "notes": [
                (
                    f"Multiple attractor regions may occur because approximately {len(root_candidates)} root candidate(s) were detected."
                    if root_candidates
                    else "A single dominant or failure-dominated basin is plausible because no clear multi-root structure was detected from sampled sign information."
                ),
                (
                    "Derivative-based methods may show sharper basin transitions near derivative-critical regions."
                    if critical_points
                    else "No derivative-critical region was identified from sampled derivative data."
                ),
            ],
        },
        "failure_diagnostics": {
            "notes": (
                newton_scan.get("notes", [])
                + newton_pathology.get("notes", [])
                + bracket_expectations.get("notes", [])
            ),
        },
        "root_coverage": {
            "notes": [
                (
                    f"Open methods may access approximately {len(root_candidates)} candidate root region(s) on the sampled domain."
                    if root_candidates
                    else "Open-method root coverage is analytically unclear because no stable root candidates were sampled."
                ),
                (
                    f"Bracket methods have only {sign_change_interval_count} detected sign-change interval(s), so their accessible coverage may be structurally smaller than that of open methods."
                ),
            ],
        },
        "root_basin_statistics": {
            "notes": [
                "If one root lies in a wider monotonic attraction region, a dominant basin share is expected.",
                (
                    "If the function were strongly symmetric on the domain, more balanced basin shares could be expected."
                    if symmetry.get("symmetry_type") in {"even", "odd"}
                    else "No strong global symmetry was detected, so strongly balanced basin shares are not guaranteed."
                ),
            ],
        },
    }

    analytic_checks = {
        "root_candidates": root_candidates,
        "root_candidate_count": len(root_candidates),
        "expected_root_count": len(root_candidates),
        "sign_change_intervals": sign_change_intervals,
        "sign_change_interval_count": sign_change_interval_count,
        "sign_change_accessible_root_count": sign_change_interval_count,
        "raw_sign_change_interval_count": len(raw_intervals),
        "critical_points": critical_points,
        "critical_point_count": len(critical_points),
        "symmetry": symmetry,
        "newton_pathology_scan": newton_scan,
        "newton_pathology": newton_pathology,
    }

    return {
        "problem_summary": {
            "expr": expr,
            "dexpr_provided": dexpr is not None and str(dexpr).strip() != "",
            "scalar_range": [a, b],
            "bracket_search_range": [float(bracket_range[0]), float(bracket_range[1])],
            "sample_points": sample_points,
        },

        # top-level aliases for frontend compatibility
        "expected_root_count": len(root_candidates),
        "root_candidate_count": len(root_candidates),
        "root_candidates": root_candidates,
        "sign_change_interval_count": sign_change_interval_count,
        "sign_change_accessible_root_count": sign_change_interval_count,
        "sign_change_intervals": sign_change_intervals,
        "critical_points": critical_points,
        "critical_point_count": len(critical_points),
        "analytic_checks": analytic_checks,
        "method_expectations": method_expectations,
        "section_expectations": section_expectations,
    }