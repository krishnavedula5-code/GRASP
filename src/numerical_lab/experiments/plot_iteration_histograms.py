from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt

from numerical_lab.methods.newton import NewtonSolver
from numerical_lab.methods.safeguarded_newton import SafeguardedNewtonSolver


# ============================================================
# Benchmark definitions
# ============================================================

BENCHMARKS: Dict[str, Dict[str, object]] = {
    "P1": {
        "title": r"$x^3 - 2x + 2$",
        "f": lambda x: x**3 - 2 * x + 2,
        "df": lambda x: 3 * x**2 - 2,
        "xmin": -4.0,
        "xmax": 4.0,
        "a": -3.0,
        "b": 0.0,
    },
    "P2": {
        "title": r"$x^3 - x - 2$",
        "f": lambda x: x**3 - x - 2,
        "df": lambda x: 3 * x**2 - 1,
        "xmin": -4.0,
        "xmax": 4.0,
        "a": 1.0,
        "b": 2.0,
    },
    "P3": {
        "title": r"$\cos(x) - x$",
        "f": lambda x: math.cos(x) - x,
        "df": lambda x: -math.sin(x) - 1.0,
        "xmin": -4.0,
        "xmax": 4.0,
        "a": 0.0,
        "b": 1.0,
    },
    "P4": {
        "title": r"$(x-1)^2(x+2)$",
        "f": lambda x: (x - 1.0) ** 2 * (x + 2.0),
        "df": lambda x: 2.0 * (x - 1.0) * (x + 2.0) + (x - 1.0) ** 2,
        "xmin": -4.0,
        "xmax": 4.0,
        "a": -3.0,
        "b": -1.0,
    },
}


# ============================================================
# Helpers
# ============================================================

def linspace(a: float, b: float, n: int) -> List[float]:
    if n <= 1:
        return [float(a)]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def summarize(values: List[int]) -> Dict[str, float]:
    if not values:
        return {}

    vals = sorted(values)
    n = len(vals)

    def q(p: float) -> float:
        idx = int(round((n - 1) * p))
        idx = max(0, min(idx, n - 1))
        return float(vals[idx])

    return {
        "n": float(n),
        "mean": sum(vals) / n,
        "min": float(vals[0]),
        "q50": q(0.50),
        "q90": q(0.90),
        "q95": q(0.95),
        "q99": q(0.99),
        "max": float(vals[-1]),
    }


# ============================================================
# Collect iteration counts
# ============================================================

def collect_newton_iterations(
    f: Callable[[float], float],
    df: Callable[[float], float],
    xs: List[float],
    tol: float,
    max_iter: int,
    include_failures_as_cap: bool = True,
) -> List[int]:
    values: List[int] = []

    for x0 in xs:
        solver = NewtonSolver(
            f=f,
            df=df,
            x0=x0,
            tol=tol,
            max_iter=max_iter,
        )
        result = solver.solve()

        if result.status == "converged":
            values.append(int(result.iterations))
        elif include_failures_as_cap:
            values.append(int(max_iter))

    return values


def collect_safeguarded_iterations(
    f: Callable[[float], float],
    df: Callable[[float], float],
    a: float,
    b: float,
    xs: List[float],
    tol: float,
    max_iter: int,
    include_failures_as_cap: bool = True,
) -> List[int]:
    values: List[int] = []

    for x0 in xs:
        solver = SafeguardedNewtonSolver(
            f=f,
            df=df,
            a=a,
            b=b,
            x0=x0,
            tol=tol,
            max_iter=max_iter,
        )
        result = solver.solve()

        if result.status == "converged":
            values.append(int(result.iterations))
        elif include_failures_as_cap:
            values.append(int(max_iter))

    return values


# ============================================================
# Plotting
# ============================================================

def plot_histogram_overlay(
    path: Path,
    newton_values: List[int],
    safeguarded_values: List[int],
    title: str,
    bins: int = 30,
) -> None:
    plt.figure(figsize=(8.6, 5.2))
    plt.hist(newton_values, bins=bins, alpha=0.6, label="Newton")
    plt.hist(safeguarded_values, bins=bins, alpha=0.6, label="Safeguarded Newton")
    plt.xlabel("iterations")
    plt.ylabel("frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_histogram_panel(
    path: Path,
    newton_values: List[int],
    safeguarded_values: List[int],
    title: str,
    bins: int = 30,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.8), sharex=True)

    axes[0].hist(newton_values, bins=bins)
    axes[0].set_title("Newton")
    axes[0].set_ylabel("frequency")

    axes[1].hist(safeguarded_values, bins=bins)
    axes[1].set_title("Safeguarded Newton")
    axes[1].set_ylabel("frequency")
    axes[1].set_xlabel("iterations")

    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Experiment runner
# ============================================================

def run_problem(
    problem_key: str,
    n: int,
    tol: float,
    max_iter: int,
    output_root: Path,
    include_failures_as_cap: bool,
) -> None:
    cfg = BENCHMARKS[problem_key]

    f = cfg["f"]  # type: ignore[assignment]
    df = cfg["df"]  # type: ignore[assignment]
    a = float(cfg["a"])
    b = float(cfg["b"])
    xmin = float(cfg["xmin"])
    xmax = float(cfg["xmax"])

    xs = linspace(xmin, xmax, n)

    newton_values = collect_newton_iterations(
        f=f,
        df=df,
        xs=xs,
        tol=tol,
        max_iter=max_iter,
        include_failures_as_cap=include_failures_as_cap,
    )

    safeguarded_values = collect_safeguarded_iterations(
        f=f,
        df=df,
        a=a,
        b=b,
        xs=xs,
        tol=tol,
        max_iter=max_iter,
        include_failures_as_cap=include_failures_as_cap,
    )

    out_dir = output_root / problem_key
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_histogram_overlay(
        out_dir / f"{problem_key}_histogram.png",
        newton_values,
        safeguarded_values,
        f"{problem_key}: {cfg['title']} — iteration histogram",
    )

    plot_histogram_panel(
        out_dir / f"{problem_key}_histogram_panel.png",
        newton_values,
        safeguarded_values,
        f"{problem_key}: {cfg['title']} — iteration histogram comparison",
    )

    print(f"\n[{problem_key}] Newton histogram summary")
    for k, v in summarize(newton_values).items():
        print(f"  {k}: {v:.6f}")

    print(f"\n[{problem_key}] Safeguarded Newton histogram summary")
    for k, v in summarize(safeguarded_values).items():
        print(f"  {k}: {v:.6f}")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot iteration histograms for Newton and Safeguarded Newton."
    )
    parser.add_argument("--problem", type=str, default="all", help="P1, P2, P3, P4, or all")
    parser.add_argument("--n", type=int, default=1000, help="Number of initial guesses")
    parser.add_argument("--tol", type=float, default=1e-10, help="Residual tolerance")
    parser.add_argument("--max-iter", type=int, default=100, help="Maximum iterations")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/histograms",
        help="Output root directory",
    )
    parser.add_argument(
        "--include-failures-as-cap",
        action="store_true",
        help="Include non-converged runs as max_iter in the histogram.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)

    if args.problem.lower() == "all":
        problem_keys = list(BENCHMARKS.keys())
    else:
        if args.problem not in BENCHMARKS:
            raise ValueError(
                f"Unknown problem: {args.problem}. Choose from {list(BENCHMARKS)} or 'all'."
            )
        problem_keys = [args.problem]

    for problem_key in problem_keys:
        run_problem(
            problem_key=problem_key,
            n=args.n,
            tol=args.tol,
            max_iter=args.max_iter,
            output_root=output_root,
            include_failures_as_cap=args.include_failures_as_cap,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()