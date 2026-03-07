from __future__ import annotations

import csv
from pathlib import Path


def find_latest_sweep(base="outputs/sweeps"):
    base = Path(base)
    folders = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("sweep_")]
    folders.sort()
    return folders[-1]


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_paper_rows(rows):

    paper_rows = []

    for r in rows:

        paper_rows.append({
            "problem": r["problem_id"],
            "method": r["method"],

            "success_rate": float(r["success_rate"]),

            "expected_iterations": float(r["iter_mean_all"]),

            "median_iterations": float(r["iter_median_success"])
                if r["iter_median_success"] else None,

            "p95_iterations": float(r["iter_p95_success"])
                if r["iter_p95_success"] else None,

            "mean_residual": float(r["residual_mean_success"])
                if r["residual_mean_success"] else None
        })

    return paper_rows


def write_csv(rows, path):

    with open(path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
    f,
    fieldnames=[
        "problem",
        "method",
        "success_rate",
        "expected_iterations",
        "median_iterations",
        "p95_iterations",
        "mean_residual"
    ]
)

        writer.writeheader()
        writer.writerows(rows)


def main():

    sweep = find_latest_sweep()

    summary_table = sweep / "summary_table.csv"

    rows = load_csv(summary_table)

    paper_rows = build_paper_rows(rows)

    out = sweep / "paper_table.csv"

    write_csv(paper_rows, out)

    print("Paper table written:", out)


if __name__ == "__main__":
    main()