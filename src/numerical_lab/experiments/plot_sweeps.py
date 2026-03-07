from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

METHOD_ORDER = [
    "newton",
    "secant",
    "bisection",
    "hybrid",
    "safeguarded_newton",
]

METHOD_LABELS = {
    "newton": "Newton",
    "secant": "Secant",
    "bisection": "Bisection",
    "hybrid": "Hybrid",
    "safeguarded_newton": "Safeguarded Newton",
}

STATUS_MARKERS = {
    "converged": "o",
    "max_iter": "x",
    "derivative_zero": "^",
    "stagnation": "s",
    "nan_or_inf": "D",
    "bad_bracket": "v",
    "error": "P",
}


# ---------------------------------------------------------
# CSV loading
# ---------------------------------------------------------

def parse_optional_float(value: str) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "null":
        return None
    try:
        return float(text)
    except Exception:
        return None


def parse_optional_int(value: str) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "null":
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def load_records_csv(csv_path: str | Path) -> List[dict]:
    csv_path = Path(csv_path)
    rows: List[dict] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["run_index"] = parse_optional_int(row.get("run_index"))
            row["x0"] = parse_optional_float(row.get("x0"))
            row["x1"] = parse_optional_float(row.get("x1"))
            row["a"] = parse_optional_float(row.get("a"))
            row["b"] = parse_optional_float(row.get("b"))
            row["iterations"] = parse_optional_int(row.get("iterations"))
            row["root"] = parse_optional_float(row.get("root"))
            row["abs_f_final"] = parse_optional_float(row.get("abs_f_final"))
            row["event_count"] = parse_optional_int(row.get("event_count")) or 0

            for flag in (
                "has_derivative_zero",
                "has_stagnation",
                "has_nonfinite",
                "has_bad_bracket",
            ):
                row[flag] = str(row.get(flag, "")).strip().lower() == "true"

            rows.append(row)

    return rows


# ---------------------------------------------------------
# Grouping / filtering
# ---------------------------------------------------------

def group_by_problem(records: List[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for r in records:
        pid = r["problem_id"]
        out.setdefault(pid, []).append(r)
    return out


def filter_problem_method(records: List[dict], problem_id: str, method: str) -> List[dict]:
    return [
        r for r in records
        if r.get("problem_id") == problem_id and r.get("method") == method
    ]


def extract_iterations(records: List[dict]) -> List[int]:
    return [r["iterations"] for r in records if r.get("iterations") is not None]


def extract_x_for_method(records: List[dict], method: str) -> List[Optional[float]]:
    xs: List[Optional[float]] = []
    for r in records:
        if method in ("newton", "safeguarded_newton", "hybrid"):
            xs.append(r.get("x0"))
        elif method == "secant":
            xs.append(r.get("x0"))
        elif method == "bisection":
            a = r.get("a")
            b = r.get("b")
            xs.append(None if a is None or b is None else 0.5 * (a + b))
        else:
            xs.append(None)
    return xs


# ---------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ccdf(values: List[int]) -> Tuple[List[int], List[float]]:
    if not values:
        return [], []

    vals = sorted(values)
    n = len(vals)
    xs = sorted(set(vals))
    ys = []

    for x in xs:
        count = sum(v > x for v in vals)
        ys.append(count / n)

    return xs, ys


def safe_problem_title(problem_id: str) -> str:
    return problem_id.upper()


# ---------------------------------------------------------
# Plot 1: Iteration histogram overlay
# ---------------------------------------------------------

def plot_iteration_histogram(
    records: List[dict],
    problem_id: str,
    output_dir: str | Path,
) -> Optional[Path]:
    plt.figure(figsize=(10, 6))

    plotted_any = False
    global_max = 0

    for method in METHOD_ORDER:
        subset = filter_problem_method(records, problem_id, method)
        vals = extract_iterations(subset)
        if not vals:
            continue

        plotted_any = True
        global_max = max(global_max, max(vals))

        bins = list(range(min(vals), max(vals) + 2))
        plt.hist(
            vals,
            bins=bins,
            alpha=0.45,
            label=METHOD_LABELS.get(method, method),
            density=False,
        )

    if not plotted_any:
        plt.close()
        return None

    plt.xlabel("Iterations")
    plt.ylabel("Count")
    plt.title(f"{safe_problem_title(problem_id)} — Iteration Histogram")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = ensure_dir(output_dir) / f"{problem_id}_hist_iterations.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


# ---------------------------------------------------------
# Plot 2: CCDF tail plot
# ---------------------------------------------------------

def plot_iteration_ccdf(
    records: List[dict],
    problem_id: str,
    output_dir: str | Path,
) -> Optional[Path]:
    plt.figure(figsize=(10, 6))

    plotted_any = False

    for method in METHOD_ORDER:
        subset = filter_problem_method(records, problem_id, method)
        vals = extract_iterations(subset)
        if not vals:
            continue

        xs, ys = ccdf(vals)
        if not xs:
            continue

        ys_plot = [max(y, 1e-6) for y in ys]
        plt.plot(xs, ys_plot, marker="o", label=METHOD_LABELS.get(method, method))
        plotted_any = True

    if not plotted_any:
        plt.close()
        return None

    plt.yscale("log")
    plt.xlabel("k")
    plt.ylabel("P(K > k)")
    plt.title(f"{safe_problem_title(problem_id)} — Iteration Tail CCDF")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = ensure_dir(output_dir) / f"{problem_id}_ccdf_iterations.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


# ---------------------------------------------------------
# Plot 3: Initialization / basin-style scatter
# ---------------------------------------------------------

def plot_init_vs_iterations(
    records: List[dict],
    problem_id: str,
    method: str,
    output_dir: str | Path,
) -> Optional[Path]:
    subset = filter_problem_method(records, problem_id, method)
    if not subset:
        return None

    xs = extract_x_for_method(subset, method)
    ys = [r.get("iterations") for r in subset]
    statuses = [r.get("status", "unknown") for r in subset]

    points = [
        (x, y, s)
        for x, y, s in zip(xs, ys, statuses)
        if x is not None and y is not None
    ]

    if not points:
        return None

    plt.figure(figsize=(10, 6))

    unique_statuses = []
    for _, _, s in points:
        if s not in unique_statuses:
            unique_statuses.append(s)

    for status in unique_statuses:
        sx = [x for x, y, s in points if s == status]
        sy = [y for x, y, s in points if s == status]
        marker = STATUS_MARKERS.get(status, "o")

        plt.scatter(
            sx,
            sy,
            marker=marker,
            alpha=0.75,
            label=status,
        )

    plt.xlabel("Initialization")
    plt.ylabel("Iterations")
    plt.title(f"{safe_problem_title(problem_id)} — {METHOD_LABELS.get(method, method)} Basin/Iteration Plot")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = ensure_dir(output_dir) / f"{problem_id}_{method}_init_vs_iterations.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


# ---------------------------------------------------------
# Plot 4: Status distribution bar chart
# ---------------------------------------------------------

def plot_status_distribution(
    records: List[dict],
    problem_id: str,
    output_dir: str | Path,
) -> Optional[Path]:
    methods_present = []
    all_statuses = []

    for method in METHOD_ORDER:
        subset = filter_problem_method(records, problem_id, method)
        if not subset:
            continue
        methods_present.append(method)
        for r in subset:
            s = r.get("status", "unknown")
            if s not in all_statuses:
                all_statuses.append(s)

    if not methods_present:
        return None

    all_statuses = sorted(all_statuses)
    x = list(range(len(methods_present)))
    width = 0.8 / max(len(all_statuses), 1)

    plt.figure(figsize=(11, 6))

    for j, status in enumerate(all_statuses):
        counts = []
        for method in methods_present:
            subset = filter_problem_method(records, problem_id, method)
            c = sum(1 for r in subset if r.get("status", "unknown") == status)
            counts.append(c)

        x_shifted = [xi + (j - (len(all_statuses) - 1) / 2) * width for xi in x]
        plt.bar(x_shifted, counts, width=width, label=status)

    plt.xticks(x, [METHOD_LABELS.get(m, m) for m in methods_present], rotation=15)
    plt.xlabel("Method")
    plt.ylabel("Count")
    plt.title(f"{safe_problem_title(problem_id)} — Status Distribution")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)

    out_path = ensure_dir(output_dir) / f"{problem_id}_status_distribution.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


# ---------------------------------------------------------
# Sweep folder discovery
# ---------------------------------------------------------

def find_latest_sweep_folder(base_dir: str | Path = "outputs/sweeps") -> Path:
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Sweep base directory not found: {base}")

    candidates = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("sweep_")]
    if not candidates:
        raise FileNotFoundError(f"No sweep folders found in: {base}")

    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


# ---------------------------------------------------------
# Main plot pipeline
# ---------------------------------------------------------

def generate_plots_for_sweep(
    sweep_dir: str | Path,
    *,
    selected_problem_ids: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    sweep_dir = Path(sweep_dir)
    records_path = sweep_dir / "records.csv"

    if not records_path.exists():
        raise FileNotFoundError(f"records.csv not found in: {sweep_dir}")

    records = load_records_csv(records_path)
    by_problem = group_by_problem(records)

    problem_ids = sorted(by_problem.keys())
    if selected_problem_ids:
        wanted = set(selected_problem_ids)
        problem_ids = [p for p in problem_ids if p in wanted]

    figures_dir = ensure_dir(sweep_dir / "figures")
    generated: Dict[str, List[str]] = {}

    for problem_id in problem_ids:
        generated[problem_id] = []

        out = plot_iteration_histogram(records, problem_id, figures_dir)
        if out:
            generated[problem_id].append(str(out))

        out = plot_iteration_ccdf(records, problem_id, figures_dir)
        if out:
            generated[problem_id].append(str(out))

        out = plot_status_distribution(records, problem_id, figures_dir)
        if out:
            generated[problem_id].append(str(out))

        for method in ("newton", "secant", "bisection", "hybrid", "safeguarded_newton"):
            out = plot_init_vs_iterations(records, problem_id, method, figures_dir)
            if out:
                generated[problem_id].append(str(out))

    manifest_path = sweep_dir / "figures" / "manifest.json"
    manifest_path.write_text(json.dumps(generated, indent=2), encoding="utf-8")

    return generated


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    latest = find_latest_sweep_folder("outputs/sweeps")
    print(f"Using sweep folder: {latest}")

    generated = generate_plots_for_sweep(latest)

    print("\nGenerated figures:")
    for problem_id, paths in generated.items():
        print(f"\n{problem_id}:")
        for p in paths:
            print(f"  {p}")