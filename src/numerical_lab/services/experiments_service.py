from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List
from dataclasses import asdict
from types import SimpleNamespace

from numerical_lab.analytics.sweep_analytics import generate_sweep_analytics
from numerical_lab.services.experiment_jobs import update_job
from numerical_lab.experiments import sweep as sweep_module
from numerical_lab.experiments import detect_basin_boundaries as boundary_module


def _find_problem(problem_id: str):
    for p in sweep_module.DEFAULT_PROBLEMS:
        if str(p.problem_id).strip().lower() == str(problem_id).strip().lower():
            return p
    raise ValueError(f"Unknown problem_id: {problem_id}")


def _build_custom_problem(payload: Dict[str, Any]):
    expr = str(payload.get("expr", "")).strip()
    if not expr:
        raise ValueError("Custom problem requires expr")

    dexpr = str(payload.get("dexpr", "")).strip() or None

    scalar_range = payload.get("scalar_range", [-4, 4])
    bracket_search_range = payload.get("bracket_search_range", scalar_range)

    if not isinstance(scalar_range, (list, tuple)) or len(scalar_range) != 2:
        raise ValueError("scalar_range must be a 2-element list")
    if not isinstance(bracket_search_range, (list, tuple)) or len(bracket_search_range) != 2:
        raise ValueError("bracket_search_range must be a 2-element list")

    scalar_range = (float(scalar_range[0]), float(scalar_range[1]))
    bracket_search_range = (float(bracket_search_range[0]), float(bracket_search_range[1]))

    return SimpleNamespace(
        problem_id="custom",
        expr=expr,
        dexpr=dexpr,
        scalar_range=scalar_range,
        bracket_search_range=bracket_search_range,
    )


def _create_job_output_folder(base: str = "outputs/sweeps") -> Path:
    base_path = Path(base)
    base_path.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    folder = base_path / f"sweep_{ts}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _records_to_plain_dicts(records) -> List[dict]:
    out = []
    for r in records:
        if hasattr(r, "__dict__"):
            out.append(dict(r.__dict__))
        else:
            out.append(dict(r))
    return out


def _compute_cluster_tol(problem, n_points: int, tol: float) -> float:
    try:
        a, b = problem.scalar_range
        a = float(a)
        b = float(b)
        if n_points >= 2:
            grid_spacing = abs(b - a) / (n_points - 1)
        else:
            grid_spacing = abs(b - a)
    except Exception:
        grid_spacing = 0.0

    return max(10.0 * float(tol), 0.25 * float(grid_spacing))


def run_sweep_job(job_id: str, payload: Dict[str, Any]) -> None:
    try:
        update_job(
            job_id,
            status="running",
            started_at=time.time(),
            progress=0.05,
            message="Starting sweep",
        )

        problem_mode = str(payload.get("problem_mode", "benchmark")).strip().lower()
        boundary_method = payload.get("boundary_method", "newton")

        n_points = int(payload.get("n_points", 100))
        max_iter = int(payload.get("max_iter", 100))
        tol = float(payload.get("tol", 1e-10))

        if problem_mode == "custom":
            problem = _build_custom_problem(payload)
        else:
            problem_id = payload.get("problem_id") or "p4"
            problem = _find_problem(problem_id)

        update_job(
            job_id,
            progress=0.15,
            message=f"Running problem sweep for {problem.problem_id}",
        )

        records = sweep_module.run_problem_sweeps(
            problem=problem,
            scalar_points=n_points,
            secant_points=n_points,
            bracket_points=n_points,
            tol=tol,
            max_iter=max_iter,
        )

        update_job(
            job_id,
            progress=0.55,
            message="Sweep finished, saving outputs",
        )

        sweep_path = _create_job_output_folder()

        sweep_module.records_to_csv(records, sweep_path / "records.csv")
        sweep_module.records_to_json(records, sweep_path / "records.json")

        summary = sweep_module.summarize_records(records)
        sweep_module.summary_to_json(summary, sweep_path / "summary.json")

        methods_requested = payload.get("methods", ["newton"])
        methods_present = sorted({r.method for r in records if getattr(r, "method", None)})
        methods_to_use = [m for m in methods_requested if m in methods_present]

        if not methods_to_use:
            methods_to_use = methods_present

        cluster_tol = _compute_cluster_tol(problem, n_points=n_points, tol=tol)

        analytics_dir = sweep_path / problem.problem_id
        analytics = generate_sweep_analytics(
            rows=[asdict(r) for r in records],
            methods=methods_to_use,
            outdir=analytics_dir,
            cluster_tol=cluster_tol,
        )

        metadata = {
            "problem_mode": problem_mode,
            "problem_id": problem.problem_id,
            "expr": problem.expr,
            "dexpr": problem.dexpr,
            "scalar_range": list(problem.scalar_range),
            "bracket_search_range": list(problem.bracket_search_range)
            if getattr(problem, "bracket_search_range", None)
            else None,
            "n_points": n_points,
            "tol": tol,
            "max_iter": max_iter,
            "methods_requested": payload.get("methods", ["newton"]),
            "cluster_tol": cluster_tol,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(sweep_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        update_job(
            job_id,
            progress=0.75,
            message="Detecting basin boundaries",
        )

        boundaries = []
        if hasattr(boundary_module, "detect_boundaries"):
            with open(sweep_path / "records.csv", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            subset = [
                r
                for r in rows
                if (r.get("problem_id") or "").strip().lower() == str(problem.problem_id).lower()
                and (r.get("method") or "").strip().lower() == str(boundary_method).lower()
            ]
            boundaries = boundary_module.detect_boundaries(subset)

            update_job(
                job_id,
                progress=0.9,
                message="Generating basin map",
            )

            try:
                from numerical_lab.experiments.plot_basin_map import (
                    plot_basin_map,
                    load_rows,
                    extract_problem_method_rows,
                )

                rows = load_rows(sweep_path / "records.csv")
                matched = extract_problem_method_rows(rows, problem.problem_id, boundary_method)

                if matched:
                    plot_basin_map(
                        rows=matched,
                        problem_id=problem.problem_id,
                        method=boundary_method,
                        output_dir=sweep_path,
                    )
            except Exception as e:
                print(f"[warn] basin map generation failed: {e}")

        basin_map_path = sweep_path / f"basin_map_{problem.problem_id}_{boundary_method}.png"
        if not basin_map_path.exists():
            basin_map_path = sweep_path / "basin_map.png"

        result = {
            "latest_sweep_dir": str(sweep_path),
            "records_csv": f"/outputs/sweeps/{sweep_path.name}/records.csv",
            "records_json": f"/outputs/sweeps/{sweep_path.name}/records.json",
            "summary_json": f"/outputs/sweeps/{sweep_path.name}/summary.json",
            "metadata_json": f"/outputs/sweeps/{sweep_path.name}/metadata.json",
            "problem_mode": problem_mode,
            "problem_id": problem.problem_id,
            "boundary_method": boundary_method,
            "boundaries": boundaries,
            "artifacts": {
                "basin_map": (
                    f"/outputs/sweeps/{sweep_path.name}/{basin_map_path.name}"
                    if basin_map_path.exists()
                    else None
                ),
                "analytics": {
                    problem.problem_id: {
                        "histogram": {
                            method: f"/outputs/sweeps/{sweep_path.name}/{problem.problem_id}/iterations_histogram_{method}.png"
                            for method in analytics.get("histogram", {}).keys()
                        },
                        "ccdf": {
                            method: f"/outputs/sweeps/{sweep_path.name}/{problem.problem_id}/iterations_ccdf_{method}.png"
                            for method in analytics.get("ccdf", {}).keys()
                        },
                        "failure_region": analytics["failure_region"],
                        "pareto": analytics["pareto"],
                        "basin_entropy": analytics["basin_entropy"],
                        "basin_entropy_data": analytics["basin_entropy_data"],
                        "basin_distribution": analytics["basin_distribution"],
                        "comparison_summary": f"/outputs/sweeps/{sweep_path.name}/{problem.problem_id}/comparison_summary.json",
                        "comparison_summary_data": analytics.get("comparison_summary_data"),
                    }
                },
            },
        }

        update_job(
            job_id,
            status="completed",
            finished_at=time.time(),
            progress=1.0,
            message="Experiment completed",
            result=result,
        )

    except Exception as e:
        update_job(
            job_id,
            status="failed",
            finished_at=time.time(),
            error=str(e),
            message="Experiment failed",
        )


def start_sweep_job(job_id: str, payload: Dict[str, Any]) -> None:
    t = threading.Thread(target=run_sweep_job, args=(job_id, payload), daemon=True)
    t.start()