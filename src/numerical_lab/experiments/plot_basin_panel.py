# src/numerical_lab/experiments/plot_basin_panel.py

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def find_latest_sweep(base: str = "outputs/sweeps") -> Path:
    base_path = Path(base)
    folders = [p for p in base_path.iterdir() if p.is_dir() and p.name.startswith("sweep_")]
    if not folders:
        raise FileNotFoundError(f"No sweep folders found in {base_path}")
    folders.sort()
    return folders[-1]


def load_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def normalize_root_id(row: Dict[str, str]) -> str:
    status = (row.get("status") or "").strip().lower()
    root_id = (row.get("root_id") or "").strip()
    if status == "converged" and root_id:
        return root_id
    return "FAIL"


def extract_problem_method_rows(
    rows: List[Dict[str, str]],
    problem_id: str,
    method: str,
) -> List[Dict[str, str]]:
    target_problem = problem_id.strip().lower()
    target_method = method.strip().lower()

    out = []
    for r in rows:
        rp = (r.get("problem_id") or "").strip().lower()
        rm = (r.get("method") or "").strip().lower()
        if rp == target_problem and rm == target_method:
            out.append(r)
    return out


def collect_global_labels(method_rows_map: Dict[str, List[Dict[str, str]]]) -> List[str]:
    labels = []
    for rows in method_rows_map.values():
        for r in rows:
            labels.append(normalize_root_id(r))
    unique = sorted(set(labels), key=lambda s: (s == "FAIL", s))
    return unique


def make_cmap_and_norm(ordered_labels: List[str]) -> Tuple[ListedColormap, BoundaryNorm]:
    base_colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
    ]
    colors = [base_colors[i % len(base_colors)] for i in range(len(ordered_labels))]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries=list(range(len(ordered_labels) + 1)), ncolors=len(ordered_labels))
    return cmap, norm


def infer_root_label_map(rows: List[Dict[str, str]]) -> Dict[str, str]:
    buckets: Dict[str, List[float]] = {}

    for r in rows:
        status = (r.get("status") or "").strip().lower()
        rid = (r.get("root_id") or "").strip()
        root_str = (r.get("root") or "").strip()

        if status != "converged" or not rid or not root_str:
            continue

        try:
            root_val = float(root_str)
        except Exception:
            continue

        buckets.setdefault(rid, []).append(root_val)

    label_map: Dict[str, str] = {}
    for rid, vals in buckets.items():
        vals_sorted = sorted(vals)
        representative = vals_sorted[len(vals_sorted) // 2]
        label_map[rid] = f"root ≈ {representative:.6f}"

    label_map["FAIL"] = "fail"
    return label_map


def method_display_name(method: str) -> str:
    mapping = {
        "newton": "Newton",
        "secant": "Secant",
        "hybrid": "Hybrid",
        "safeguarded_newton": "Safeguarded Newton",
        "bisection": "Bisection",
    }
    return mapping.get(method, method)


def plot_basin_panel(
    rows: List[Dict[str, str]],
    problem_id: str,
    methods: List[str],
    output_dir: Path,
) -> Path:
    method_rows_map: Dict[str, List[Dict[str, str]]] = {}
    for method in methods:
        subset = extract_problem_method_rows(rows, problem_id, method)
        if subset:
            method_rows_map[method] = subset

    if not method_rows_map:
        raise ValueError(f"No rows found for problem={problem_id}")

    ordered_labels = collect_global_labels(method_rows_map)
    label_to_int = {label: i for i, label in enumerate(ordered_labels)}
    cmap, norm = make_cmap_and_norm(ordered_labels)
    global_xmin = None
    global_xmax = None

    for subset in method_rows_map.values():
        xs_subset = [parse_float(r.get("x0", "0")) for r in subset]
        if not xs_subset:
            continue
        xmin = min(xs_subset)
        xmax = max(xs_subset)

        global_xmin = xmin if global_xmin is None else min(global_xmin, xmin)
        global_xmax = xmax if global_xmax is None else max(global_xmax, xmax)
    all_rows_flat = []
    for subset in method_rows_map.values():
        all_rows_flat.extend(subset)
    root_label_map = infer_root_label_map(all_rows_flat)

    fig, axes = plt.subplots(
        nrows=len(method_rows_map),
        ncols=1,
        figsize=(12, 1.6 * len(method_rows_map) + 0.8),
        sharex=False,
    )

    if len(method_rows_map) == 1:
        axes = [axes]

    image_for_colorbar = None

    for ax, method in zip(axes, method_rows_map.keys()):
        subset = method_rows_map[method]

        processed = []
        for r in subset:
            x0 = parse_float(r.get("x0", "0"))
            basin_label = normalize_root_id(r)
            processed.append((x0, basin_label))

        processed.sort(key=lambda t: t[0])

        xs = [t[0] for t in processed]
        labels = [t[1] for t in processed]
        values = [label_to_int[label] for label in labels]
        data = [values]

        im = ax.imshow(
            data,
            aspect="auto",
            cmap=cmap,
            norm=norm,
            extent=[xs[0], xs[-1], 0, 1],
            interpolation="nearest",
        )
        image_for_colorbar = im
        ax.set_xlim(global_xmin, global_xmax)
        ax.set_yticks([])
        ax.set_ylabel(method_display_name(method), rotation=0, labelpad=50, va="center")
        for spine in ax.spines.values():
            spine.set_visible(True)

    axes[-1].set_xlabel(r"Initial guess $x_0$")
    fig.suptitle(f"Basin comparison — {problem_id.upper()}", y=0.98)

    display_labels = [root_label_map.get(label, label) for label in ordered_labels]
    cbar = fig.colorbar(
        image_for_colorbar,
        ax=axes,
        ticks=list(range(len(ordered_labels))),
        fraction=0.03,
        pad=0.06,
    )
    cbar.ax.set_yticklabels(display_labels)
    cbar.set_label("Basin label")

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"basin_panel_{problem_id}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    latest = find_latest_sweep()
    rows = load_rows(latest / "records.csv")

    problem_id = "p4"
    methods = ["newton", "secant"]

    out_path = plot_basin_panel(
        rows=rows,
        problem_id=problem_id,
        methods=methods,
        output_dir=latest,
    )
    print(f"Saved basin panel to: {out_path}")


if __name__ == "__main__":
    main()