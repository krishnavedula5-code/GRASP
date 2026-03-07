from __future__ import annotations

import csv
import random
from pathlib import Path
import matplotlib.pyplot as plt


def find_latest_sweep(base="outputs/sweeps"):
    base = Path(base)
    folders = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("sweep_")]
    folders.sort()
    return folders[-1]


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():

    sweep = find_latest_sweep()
    table_path = sweep / "paper_table.csv"

    rows = load_rows(table_path)

    # Method colors
    method_colors = {
        "newton": "red",
        "secant": "orange",
        "bisection": "green",
        "hybrid": "blue",
        "safeguarded_newton": "purple",
    }

    # Problem markers
    problem_markers = {
        "p1": "o",
        "p2": "s",
        "p3": "^",
        "p4": "D",
    }

    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    for r in rows:

        x = float(r["expected_iterations"])
        y = float(r["success_rate"])

        method = r["method"]
        problem = r["problem"]

        color = method_colors.get(method, "black")
        marker = problem_markers.get(problem, "o")

        # Small horizontal jitter to avoid overlapping points
        jitter = random.uniform(-0.35, 0.35)

        size = 120
        edge_width = 1

        # Highlight Safeguarded Newton (best method)
        if method == "safeguarded_newton":
            size = 220
            edge_width = 2.5

        ax.scatter(
            x + jitter,
            y,
            color=color,
            marker=marker,
            s=size,
            alpha=0.90,
            edgecolors="black",
            linewidth=edge_width,
        )

    # Labels
    ax.set_xlabel("Expected Iterations  E[K]", fontsize=13)
    ax.set_ylabel("Success Rate", fontsize=13)

    ax.set_title("Method Reliability vs Computational Cost", fontsize=15)

    ax.grid(True, linestyle="--", alpha=0.5)

    # Zoom Y-axis for clarity
    ax.set_ylim(0.70, 1.02)
    
    # ---------------------------
    # Legend for methods
    # ---------------------------

    method_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=c,
            markeredgecolor="black",
            markersize=10,
            label=m,
        )
        for m, c in method_colors.items()
    ]

    # ---------------------------
    # Legend for problems
    # ---------------------------

    problem_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=m,
            color="black",
            linestyle="None",
            markersize=10,
            label=p,
        )
        for p, m in problem_markers.items()
    ]

    legend1 = ax.legend(handles=method_handles, title="Solver Method", loc="lower right")
    ax.add_artist(legend1)

    ax.legend(handles=problem_handles, title="Benchmark Problem", loc="lower left")

    # Save figure
    out = sweep / "method_comparison_clean.png"

    plt.tight_layout()
    plt.savefig(out, dpi=300)

    print("Saved:", out)

    plt.show()


if __name__ == "__main__":
    main()