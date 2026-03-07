from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Any, List


def find_latest_sweep_folder(base_dir: str | Path = "outputs/sweeps") -> Path:
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Sweep base directory not found: {base}")

    candidates = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("sweep_")]
    if not candidates:
        raise FileNotFoundError(f"No sweep folders found in: {base}")

    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def load_summary_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_summary(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for key, block in summary.items():
        row = {
            "key": key,
            "problem_id": block.get("problem_id"),
            "method": block.get("method"),
            "n_total": block.get("n_total"),
            "n_success": block.get("n_success"),
            "n_failure": block.get("n_failure"),
            "success_rate": block.get("success_rate"),
            "failure_rate": block.get("failure_rate"),
            "status_counts": json.dumps(block.get("status_counts", {})),
            "iter_mean_all": block.get("iterations_all", {}).get("mean"),
            "iter_median_all": block.get("iterations_all", {}).get("median"),
            "iter_p90_all": block.get("iterations_all", {}).get("p90"),
            "iter_p95_all": block.get("iterations_all", {}).get("p95"),
            "iter_p99_all": block.get("iterations_all", {}).get("p99"),
            "iter_max_all": block.get("iterations_all", {}).get("max"),
            "iter_mean_success": block.get("iterations_success_only", {}).get("mean"),
            "iter_median_success": block.get("iterations_success_only", {}).get("median"),
            "iter_p90_success": block.get("iterations_success_only", {}).get("p90"),
            "iter_p95_success": block.get("iterations_success_only", {}).get("p95"),
            "iter_p99_success": block.get("iterations_success_only", {}).get("p99"),
            "iter_max_success": block.get("iterations_success_only", {}).get("max"),
            "cap_hit_rate_all": block.get("cap_hit_rates", {}).get("all_runs"),
            "cap_hit_rate_success": block.get("cap_hit_rates", {}).get("success_only"),
            "residual_mean_all": block.get("residuals_all", {}).get("mean"),
            "residual_median_all": block.get("residuals_all", {}).get("median"),
            "residual_p95_all": block.get("residuals_all", {}).get("p95"),
            "residual_max_all": block.get("residuals_all", {}).get("max"),
            "residual_mean_success": block.get("residuals_success_only", {}).get("mean"),
            "residual_median_success": block.get("residuals_success_only", {}).get("median"),
            "residual_p95_success": block.get("residuals_success_only", {}).get("p95"),
            "residual_max_success": block.get("residuals_success_only", {}).get("max"),
            "derivative_zero_rate": block.get("event_flags", {}).get("derivative_zero_rate"),
            "stagnation_rate": block.get("event_flags", {}).get("stagnation_rate"),
            "nonfinite_rate": block.get("event_flags", {}).get("nonfinite_rate"),
            "bad_bracket_rate": block.get("event_flags", {}).get("bad_bracket_rate"),
        }
        rows.append(row)

    rows.sort(key=lambda r: (str(r["problem_id"]), str(r["method"])))
    return rows


def write_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows to write.")

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    sweep_dir = find_latest_sweep_folder("outputs/sweeps")
    print(f"Using sweep folder: {sweep_dir}")

    summary_path = sweep_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found in: {sweep_dir}")

    summary = load_summary_json(summary_path)
    rows = flatten_summary(summary)

    out_csv = sweep_dir / "summary_table.csv"
    write_csv(rows, out_csv)

    print(f"Summary table written to: {out_csv}")
    print(f"Rows written: {len(rows)}")


if __name__ == "__main__":
    main()