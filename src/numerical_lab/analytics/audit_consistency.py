from __future__ import annotations

from typing import Any, Dict, List, Optional


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _add_issue(issues: List[Dict[str, Any]], code: str, severity: str, message: str) -> None:
    issues.append(
        {
            "code": code,
            "severity": severity,
            "message": message,
        }
    )


def _final_status(issues: List[Dict[str, Any]]) -> str:
    if any(i.get("severity") == "suspicious" for i in issues):
        return "suspicious"
    if any(i.get("severity") == "warning" for i in issues):
        return "warning"
    return "ok"


def _audit_root_coverage_domain_aware(
    *,
    root_coverage_data: Dict[str, Any] | None,
    known_roots: List[float] | None,
) -> Dict[str, Any]:
    root_coverage_data = root_coverage_data or {}
    known_roots = known_roots or []

    issues: List[Dict[str, Any]] = []

    global_behavior = root_coverage_data.get("global_behavior", {}) or {}
    benchmark_eval = root_coverage_data.get("benchmark_evaluation", {}) or {}
    solvers = root_coverage_data.get("solvers", {}) or {}

    global_all_count = int(global_behavior.get("all_detected_root_count", 0))
    global_in_domain_count = int(global_behavior.get("in_domain_detected_root_count", 0))
    global_out_of_domain_count = int(global_behavior.get("out_of_domain_detected_root_count", 0))

    known_root_count = int(benchmark_eval.get("known_root_count", len(known_roots)))

    # -----------------------------
    # Global benchmark consistency
    # -----------------------------
    if known_root_count > 0:
        if global_in_domain_count > known_root_count:
            _add_issue(
                issues,
                code="in_domain_root_overcount",
                severity="suspicious",
                message=(
                    f"In-domain detected roots ({global_in_domain_count}) exceed known benchmark roots "
                    f"({known_root_count})."
                ),
            )
        elif global_in_domain_count < known_root_count:
            _add_issue(
                issues,
                code="in_domain_root_undercount",
                severity="warning",
                message=(
                    f"In-domain detected roots ({global_in_domain_count}) are fewer than known benchmark roots "
                    f"({known_root_count})."
                ),
            )

    if global_out_of_domain_count > 0:
        _add_issue(
            issues,
            code="global_out_of_domain_excursions",
            severity="warning",
            message=(
                f"{global_out_of_domain_count} out-of-domain root(s) were detected globally. "
                "This reflects true solver behavior and should not be treated as benchmark overcount."
            ),
        )

    # -----------------------------
    # Per-method benchmark consistency
    # -----------------------------
    for method, payload in sorted(solvers.items()):
        true_behavior = payload.get("true_behavior", {}) or {}
        bench = payload.get("benchmark_evaluation", {}) or {}

        all_count = int(true_behavior.get("all_detected_root_count", 0))
        in_count = int(true_behavior.get("in_domain_detected_root_count", 0))
        out_count = int(true_behavior.get("out_of_domain_detected_root_count", 0))
        excursion_detected = bool(true_behavior.get("excursion_detected", False))
        faithfulness = _safe_float(true_behavior.get("domain_faithfulness"), default=0.0)

        benchmark_matched_count = bench.get("benchmark_matched_count", None)
        benchmark_known_root_count = bench.get("benchmark_known_root_count", None)
        benchmark_coverage = bench.get("benchmark_coverage", None)
        unmatched_detected = bench.get("unmatched_detected_in_domain_roots", []) or []
        unmatched_known = bench.get("unmatched_known_roots", []) or []

        if excursion_detected and out_count > 0:
            _add_issue(
                issues,
                code="solver_excursion_detected",
                severity="warning",
                message=(
                    f"Method '{method}' converged to {out_count} out-of-domain root(s) "
                    f"(domain faithfulness={faithfulness:.3f})."
                ),
            )

        if benchmark_known_root_count is not None and benchmark_matched_count is not None:
            if unmatched_detected:
                _add_issue(
                    issues,
                    code="unmatched_in_domain_detected_roots",
                    severity="suspicious",
                    message=(
                        f"Method '{method}' has {len(unmatched_detected)} in-domain detected root(s) "
                        "that do not match known benchmark roots."
                    ),
                )

            if unmatched_known:
                _add_issue(
                    issues,
                    code="missed_known_roots",
                    severity="warning",
                    message=(
                        f"Method '{method}' missed {len(unmatched_known)} known benchmark root(s)."
                    ),
                )

            if benchmark_coverage is not None and float(benchmark_coverage) == 1.0 and out_count == 0:
                # fully benchmark-correct and domain-faithful
                pass
            elif benchmark_coverage is not None and float(benchmark_coverage) == 1.0 and out_count > 0:
                # benchmark-correct but excursion-prone
                pass

    status = _final_status(issues)

    if status == "ok":
        summary = "Benchmark-consistent in-domain root behavior with no audit issues detected."
    elif status == "warning":
        summary = "Results are benchmark-consistent in-domain, but excursion or interpretation cautions remain."
    else:
        summary = "Benchmark consistency is suspicious due to in-domain mismatches against known roots."

    return {
        "status": status,
        "issues": issues,
        "summary": summary,
        "known_root_count": known_root_count,
        "global_root_count_detected": global_all_count,
        "global_in_domain_root_count_detected": global_in_domain_count,
        "global_out_of_domain_root_count_detected": global_out_of_domain_count,
    }


def audit_consistency(
    *,
    benchmark_id: Optional[str],
    benchmark_name: Optional[str],
    benchmark_category: Optional[str],
    known_roots: Optional[List[float]],
    comparison_summary_data: Optional[Dict[str, Any]],
    root_coverage_data: Optional[Dict[str, Any]],
    root_basin_statistics_data: Optional[Dict[str, Any]],
    basin_entropy_data: Optional[Dict[str, Any]],
    failure_statistics_data: Optional[Dict[str, Any]],
    interpretation_summary_data: Optional[Dict[str, Any]],
    problem_expectations_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []

    known_roots = known_roots or []

    # ---------------------------------------------------------
    # A. comparison summary sanity
    # ---------------------------------------------------------
    methods = (comparison_summary_data or {}).get("methods", []) or []
    success_rates: List[float] = []

    for row in methods:
        method = row.get("method", "unknown")

        sr = _safe_float(row.get("success_rate"))
        if sr is not None:
            success_rates.append(sr)
            if sr < 0 or sr > 1:
                _add_issue(
                    issues,
                    "success_rate_out_of_range",
                    "suspicious",
                    f"{method} has success_rate={sr}, outside [0,1].",
                )

        failure_count = _safe_float(row.get("failure_count"))
        if failure_count is not None and failure_count < 0:
            _add_issue(
                issues,
                "negative_failure_count",
                "suspicious",
                f"{method} has negative failure_count={failure_count}.",
            )

        mean_iter = _safe_float(row.get("mean_iter"))
        if mean_iter is not None and mean_iter < 0:
            _add_issue(
                issues,
                "negative_mean_iterations",
                "suspicious",
                f"{method} has negative mean_iter={mean_iter}.",
            )

    if success_rates and len(success_rates) > 1:
        if max(success_rates) - min(success_rates) < 1e-12:
            _add_issue(
                issues,
                "ranking_degenerate",
                "warning",
                "All methods have essentially identical success rates; ranking by success alone is not informative.",
            )

    # ---------------------------------------------------------
    # B. root coverage sanity (NEW DOMAIN-AWARE VERSION)
    # ---------------------------------------------------------
    root_coverage_data = root_coverage_data or {}
    global_behavior = root_coverage_data.get("global_behavior", {}) or {}
    benchmark_eval = root_coverage_data.get("benchmark_evaluation", {}) or {}
    solver_cov = root_coverage_data.get("solvers", {}) or {}

    global_all_roots = global_behavior.get("all_detected_roots", []) or []
    global_in_domain_roots = global_behavior.get("in_domain_detected_roots", []) or []
    global_out_of_domain_roots = global_behavior.get("out_of_domain_detected_roots", []) or []

    global_root_count_detected = int(global_behavior.get("all_detected_root_count", len(global_all_roots)))
    global_in_domain_root_count_detected = int(
        global_behavior.get("in_domain_detected_root_count", len(global_in_domain_roots))
    )
    global_out_of_domain_root_count_detected = int(
        global_behavior.get("out_of_domain_detected_root_count", len(global_out_of_domain_roots))
    )

    benchmark_known_root_count = int(
        benchmark_eval.get("known_root_count", len(known_roots))
    )

    # Compare only IN-DOMAIN roots against benchmark roots
    if benchmark_known_root_count > 0:
        if global_in_domain_root_count_detected > benchmark_known_root_count:
            _add_issue(
                issues,
                "in_domain_root_overcount",
                "suspicious",
                f"Detected {global_in_domain_root_count_detected} in-domain roots, but benchmark specifies {benchmark_known_root_count} known roots.",
            )
        elif global_in_domain_root_count_detected < benchmark_known_root_count:
            _add_issue(
                issues,
                "in_domain_root_undercount",
                "warning",
                f"Detected {global_in_domain_root_count_detected} in-domain roots, but benchmark specifies {benchmark_known_root_count} known roots.",
            )

    # Out-of-domain roots are warnings, not benchmark errors
    if global_out_of_domain_root_count_detected > 0:
        _add_issue(
            issues,
            "global_out_of_domain_excursions",
            "warning",
            f"{global_out_of_domain_root_count_detected} out-of-domain roots were detected globally; this reflects solver excursions, not benchmark overcount.",
        )

    for solver, info in solver_cov.items():
        true_behavior = info.get("true_behavior", {}) or {}
        bench_info = info.get("benchmark_evaluation", {}) or {}

        all_count = int(true_behavior.get("all_detected_root_count", 0))
        in_count = int(true_behavior.get("in_domain_detected_root_count", 0))
        out_count = int(true_behavior.get("out_of_domain_detected_root_count", 0))
        excursion_detected = bool(true_behavior.get("excursion_detected", False))
        domain_faithfulness = _safe_float(true_behavior.get("domain_faithfulness"))

        benchmark_coverage = _safe_float(bench_info.get("benchmark_coverage"))
        benchmark_matched_count = bench_info.get("benchmark_matched_count")
        benchmark_known_count = bench_info.get("benchmark_known_root_count")
        unmatched_detected = bench_info.get("unmatched_detected_in_domain_roots", []) or []
        unmatched_known = bench_info.get("unmatched_known_roots", []) or []

        if all_count < in_count:
            _add_issue(
                issues,
                "all_count_less_than_in_domain_count",
                "suspicious",
                f"{solver} reports total detected roots={all_count} < in-domain roots={in_count}.",
            )

        if excursion_detected and out_count == 0:
            _add_issue(
                issues,
                "excursion_flag_inconsistent",
                "warning",
                f"{solver} is marked as excursion_detected=True but has zero out-of-domain roots.",
            )

        if (not excursion_detected) and out_count > 0:
            _add_issue(
                issues,
                "excursion_flag_missing",
                "warning",
                f"{solver} has {out_count} out-of-domain roots but excursion_detected=False.",
            )

        if domain_faithfulness is not None and (domain_faithfulness < 0 or domain_faithfulness > 1):
            _add_issue(
                issues,
                "domain_faithfulness_out_of_range",
                "suspicious",
                f"{solver} has domain_faithfulness={domain_faithfulness}, outside [0,1].",
            )

        if benchmark_coverage is not None and (benchmark_coverage < 0 or benchmark_coverage > 1):
            _add_issue(
                issues,
                "benchmark_coverage_out_of_range",
                "suspicious",
                f"{solver} has benchmark_coverage={benchmark_coverage}, outside [0,1].",
            )

        if isinstance(benchmark_matched_count, int) and isinstance(benchmark_known_count, int):
            if benchmark_matched_count > benchmark_known_count:
                _add_issue(
                    issues,
                    "benchmark_matched_exceeds_known",
                    "suspicious",
                    f"{solver} reports benchmark_matched_count={benchmark_matched_count} > benchmark_known_root_count={benchmark_known_count}.",
                )

        if unmatched_detected:
            _add_issue(
                issues,
                "unmatched_in_domain_detected_roots",
                "suspicious",
                f"{solver} has {len(unmatched_detected)} in-domain detected roots that do not match benchmark roots.",
            )

        if unmatched_known:
            _add_issue(
                issues,
                "missed_known_roots",
                "warning",
                f"{solver} missed {len(unmatched_known)} known benchmark roots.",
            )

        if excursion_detected and out_count > 0:
            faith_text = f"{domain_faithfulness:.3f}" if domain_faithfulness is not None else "unknown"
            _add_issue(
                issues,
                "solver_excursion_detected",
                "warning",
                f"{solver} converged to {out_count} out-of-domain roots (domain faithfulness={faith_text}).",
            )
            
    # single-root benchmark fragmentation checks
    entropy_methods = (basin_entropy_data or {}).get("methods", []) or []
    if len(known_roots) == 1:
        for row in entropy_methods:
            method = row.get("method", "unknown")
            num_basins = row.get("num_basins")
            entropy = _safe_float(row.get("entropy"))

            if isinstance(num_basins, int) and num_basins > 1:
                _add_issue(
                    issues,
                    "single_root_multiple_basins",
                    "warning",
                    f"{method} shows num_basins={num_basins} for a benchmark with one known root.",
                )

            if entropy is not None and entropy > 1e-6:
                _add_issue(
                    issues,
                    "single_root_positive_entropy",
                    "warning",
                    f"{method} shows entropy={entropy:.6g} for a benchmark with one known root.",
                )

    # ---------------------------------------------------------
    # C. structural applicability checks
    # ---------------------------------------------------------
    interpretation = interpretation_summary_data or {}
    top_summary = interpretation.get("top_summary", []) or []
    top_text = " ".join(str(x) for x in top_summary).lower()

    if benchmark_id and str(benchmark_id).startswith("multi_"):
        if (
            "linear" not in top_text
            and "repeated root" not in top_text
            and "multiplicity" not in top_text
        ):
            _add_issue(
                issues,
                "missing_repeated_root_interpretation",
                "warning",
                "Repeated-root benchmark detected, but interpretation does not explicitly mention repeated-root effects or degraded Newton convergence.",
            )

    # ---------------------------------------------------------
    # D. analytic-vs-observed mismatch
    # ---------------------------------------------------------
    newton_path = (
        (interpretation_summary_data or {}).get("newton_pathology_interpretation")
        or {}
    )
    newton_message = str(newton_path.get("message", "")).lower()

    if "high" in newton_message:
        newton_row = next((r for r in methods if r.get("method") == "newton"), None)
        if newton_row is not None:
            sr = _safe_float(newton_row.get("success_rate"))
            failures = _safe_float(newton_row.get("failure_count"))
            if sr == 1.0 and failures == 0:
                _add_issue(
                    issues,
                    "analytic_observed_newton_mismatch",
                    "warning",
                    "Analytic Newton risk is high, but observed Newton runs show perfect success; interpretation should frame this as a mismatch, not direct instability evidence.",
                )

    # ---------------------------------------------------------
    # E. interpretation overclaiming
    # ---------------------------------------------------------
    failure_interp = (
        (interpretation_summary_data or {}).get("failure_interpretation") or {}
    )
    global_notes = failure_interp.get("global_notes", []) or []
    global_failure_text = " ".join(str(x) for x in global_notes).lower()

    no_failures_all = True
    for row in methods:
        fc = _safe_float(row.get("failure_count"))
        if fc is None or fc > 0:
            no_failures_all = False
            break

    if no_failures_all and any(
        token in global_failure_text
        for token in ["instability", "unstable", "pathological", "stagnation"]
    ):
        _add_issue(
            issues,
            "interpretation_not_conditioned",
            "warning",
            "Interpretation uses instability/pathology language despite zero observed failures across methods.",
        )

    status = _final_status(issues)

    if status == "ok":
        summary = "Results are numerically and structurally consistent with the benchmark definition."
    elif status == "warning":
        summary = "Results are numerically plausible, but some interpretation, applicability, or excursion issues require caution."
    else:
        summary = "One or more suspicious internal inconsistencies were detected."

    return {
        "status": status,
        "issues": issues,
        "summary": summary,
        "benchmark_id": benchmark_id,
        "benchmark_name": benchmark_name,
        "benchmark_category": benchmark_category,
        "known_root_count": benchmark_known_root_count,
        "global_root_count_detected": global_root_count_detected,
        "global_in_domain_root_count_detected": global_in_domain_root_count_detected,
        "global_out_of_domain_root_count_detected": global_out_of_domain_root_count_detected,
    }