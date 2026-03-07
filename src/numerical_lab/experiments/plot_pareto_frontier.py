from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt


def find_latest_sweep(base: str | Path = "outputs/sweeps") -> Path:
    base = Path(base)
    folders = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("sweep_")]
    folders.sort()
    if not folders:
        raise FileNotFoundError(f"No sweep folders found in {base}")
    return folders[-1]


def load_rows(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_dominated(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """
    We want:
      - lower expected_iterations is better
      - higher success_rate is better

    a is dominated by b if:
      b is at least as good in both objectives,
      and strictly better in at least one.
    """
    a_x = float(a["expected_iterations"])
    a_y = float(a["success_rate"])

    b_x = float(b["expected_iterations"])
    b_y = float(b["success_rate"])

    no_worse = (b_x <= a_x) and (b_y >= a_y)
    strictly_better = (b_x < a_x) or (b_y > a_y)

    return no_worse and strictly_better


def compute_pareto_front(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    front = []

    for i, a in enumerate(rows):
        dominated = False
        for j, b in enumerate(rows):
            if i == j:
                continue
            if is_dominated(a, b):
                dominated = True
                break
        if not dominated:
            front.append(a)

    # Sort frontier left-to-right by expected iterations
    front.sort(key=lambda r: float(r["expected_iterations"]))
    return front


def main() -> None:
    sweep = find_latest_sweep()
    table_path = sweep / "paper_table.csv"

    rows = load_rows(table_path)

    method_colors = {
        "newton": "red",
        "secant": "orange",
        "bisection": "green",
        "hybrid": "blue",
        "safeguarded_newton": "purple",
    }

    problem_markers = {
        "p1": "o",
        "p2": "s",
        "p3": "^",
        "p4": "D",
    }

    pareto = compute_pareto_front(rows)
    pareto_ids = {id(r) for r in pareto}

    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    # Plot all points
    for r in rows:
        x = float(r["expected_iterations"])
        y = float(r["success_rate"])
        method = r["method"]
        problem = r["problem"]

        ax.scatter(
            x,
            y,
            color=method_colors.get(method, "black"),
            marker=problem_markers.get(problem, "o"),
            s=140,
            alpha=0.9,
            edgecolors="black",
            linewidth=1.2,
        )

    # Highlight Pareto-optimal points
    px = []
    py = []
    for r in pareto:
        x = float(r["expected_iterations"])
        y = float(r["success_rate"])
        px.append(x)
        py.append(y)

        ax.scatter(
            x,
            y,
            s=260,
            facecolors="none",
            edgecolors="black",
            linewidth=2.5,
        )

    # Draw frontier line
    if len(px) >= 2:
        ax.plot(px, py, linestyle="--", linewidth=2)

    ax.set_xlabel("Expected Iterations  E[K]", fontsize=13)
    ax.set_ylabel("Success Rate", fontsize=13)
    ax.set_title("Pareto Frontier: Reliability vs Computational Cost", fontsize=15)

    ax.set_ylim(0.70, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Method legend
    method_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=c,
            markeredgecolor="black",
            markersize=10,
            label=m,
        )
        for m, c in method_colors.items()
    ]

    # Problem legend
    problem_handles = [
        plt.Line2D(
            [0], [0],
            marker=m,
            color="black",
            linestyle="None",
            markersize=10,
            label=p,
        )
        for p, m in problem_markers.items()
    ]

    # Pareto legend
    pareto_handle = plt.Line2D(
        [0], [0],
        marker="o",
        color="black",
        markerfacecolor="none",
        linestyle="--",
        markersize=10,
        linewidth=2,
        label="Pareto-optimal",
    )

    legend1 = ax.legend(handles=method_handles, title="Solver Method", loc="lower right")
    ax.add_artist(legend1)

    legend2 = ax.legend(handles=problem_handles, title="Benchmark Problem", loc="lower left")
    ax.add_artist(legend2)

    ax.legend(handles=[pareto_handle], loc="upper right")

    out = sweep / "pareto_frontier.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)

    print("Saved:", out)
    print("\nPareto-optimal points:")
    for r in pareto:
        print(
            f"  {r['problem']}:{r['method']} | "
            f"E[K]={float(r['expected_iterations']):.3f}, "
            f"success={float(r['success_rate']):.3f}"
        )

    plt.show()


if __name__ == "__main__":
    main()