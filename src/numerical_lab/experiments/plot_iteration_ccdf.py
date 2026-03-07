from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from numerical_lab.methods.newton import NewtonSolver
from numerical_lab.methods.safeguarded_newton import SafeguardedNewtonSolver


# ============================================================
# Benchmark problems
# ============================================================

BENCHMARKS: Dict[str, Dict[str, object]] = {
    "P1": {
        "name": "P1",
        "title": r"$x^3 - 2x + 2$",
        "f": lambda x: x**3 - 2 * x + 2,
        "df": lambda x: 3 * x**2 - 2,
        "xmin": -4.0,
        "xmax": 4.0,
        "a": -3.0,
        "b": 0.0,
    },
    "P2": {
        "name": "P2",
        "title": r"$x^3 - x - 2$",
        "f": lambda x: x**3 - x - 2,
        "df": lambda x: 3 * x**2 - 1,
        "xmin": -4.0,
        "xmax": 4.0,
        "a": 1.0,
        "b": 2.0,
    },
    "P3": {
        "name": "P3",
        "title": r"$\cos(x) - x$",
        "f": lambda x: math.cos(x) - x,
        "df": lambda x: -math.sin(x) - 1.0,
        "xmin": -4.0,
        "xmax": 4.0,
        "a": 0.0,
        "b": 1.0,
    },
    "P4": {
        "name": "P4",
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


def empirical_ccdf(values: List[int]) -> Tuple[List[int], List[float]]:
    """
    Return x=k and y=P(K >= k) for integer iteration counts.
    """
    if not values:
        return [], []

    max_k = max(values)
    ks = list(range(0, max_k + 1))
    n = len(values)
    ys = [sum(1 for v in values if v >= k) / n for k in ks]
    return ks, ys


def summarize_iterations(values: List[int]) -> Dict[str, float]:
    if not values:
        return {}

    vals = sorted(values)
    n = len(vals)

    def q(p: float) -> float:
        if n == 1:
            return float(vals[0])
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
# Run samplers
# ============================================================

def collect_newton_iterations(
    *,
    f: Callable[[float], float],
    df: Callable[[float], float],
    xs: List[float],
    tol: float,
    max_iter: int,
    df_tol: float,
    stagnation_tol: float,
    tol_x: Optional[float],
    numerical_derivative: bool,
    include_failures_as_cap: bool,
) -> List[int]:
    values: List[int] = []

    for x0 in xs:
        solver = NewtonSolver(
            f=f,
            df=df,
            x0=x0,
            tol=tol,
            max_iter=max_iter,
            numerical_derivative=numerical_derivative,
            df_tol=df_tol,
            stagnation_tol=stagnation_tol,
            tol_x=tol_x,
        )
        result = solver.solve()

        if result.status == "converged":
            values.append(int(result.iterations))
        elif include_failures_as_cap:
            values.append(int(max_iter))

    return values


def collect_safeguarded_newton_iterations(
    *,
    f: Callable[[float], float],
    df: Callable[[float], float],
    a: float,
    b: float,
    xs: List[float],
    tol: float,
    max_iter: int,
    include_failures_as_cap: bool,
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

def plot_ccdf(
    path: Path,
    curves: List[Tuple[str, List[int]]],
    title: str,
    logy: bool = True,
) -> None:
    plt.figure(figsize=(8.8, 5.2))

    for label, values in curves:
        ks, ys = empirical_ccdf(values)
        if ks:
            plt.step(ks, ys, where="post", label=label)

    plt.xlabel("iterations k")
    plt.ylabel(r"$P(K \geq k)$")
    plt.title(title)
    if logy:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_ccdf_compare_panel(
    path: Path,
    title: str,
    newton_values: List[int],
    safeguarded_values: List[int],
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.0), sharex=True)

    panels = [
        (axes[0], "Newton", newton_values),
        (axes[1], "Safeguarded Newton", safeguarded_values),
    ]

    for ax, label, values in panels:
        ks, ys = empirical_ccdf(values)
        if ks:
            ax.step(ks, ys, where="post")
        ax.set_yscale("log")
        ax.set_ylabel(r"$P(K \geq k)$")
        ax.set_title(label)

    axes[-1].set_xlabel("iterations k")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main experiment runner
# ============================================================

def run_problem(
    *,
    problem_key: str,
    output_root: Path,
    n: int,
    tol: float,
    max_iter: int,
    df_tol: float,
    stagnation_tol: float,
    tol_x: Optional[float],
    numerical_derivative: bool,
    xmin: Optional[float],
    xmax: Optional[float],
    include_failures_as_cap: bool,
) -> None:
    cfg = BENCHMARKS[problem_key]

    f = cfg["f"]  # type: ignore[assignment]
    df = cfg["df"]  # type: ignore[assignment]
    a = float(cfg["a"])
    b = float(cfg["b"])

    x_left = float(cfg["xmin"] if xmin is None else xmin)
    x_right = float(cfg["xmax"] if xmax is None else xmax)
    xs = linspace(x_left, x_right, n)

    newton_values = collect_newton_iterations(
        f=f,
        df=df,
        xs=xs,
        tol=tol,
        max_iter=max_iter,
        df_tol=df_tol,
        stagnation_tol=stagnation_tol,
        tol_x=tol_x,
        numerical_derivative=numerical_derivative,
        include_failures_as_cap=include_failures_as_cap,
    )

    safeguarded_values = collect_safeguarded_newton_iterations(
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

    plot_ccdf(
        out_dir / f"{problem_key}_ccdf.png",
        [
            ("Newton", newton_values),
            ("Safeguarded Newton", safeguarded_values),
        ],
        title=f"{problem_key}: {cfg['title']} — Iteration CCDF",
        logy=True,
    )

    plot_ccdf_compare_panel(
        out_dir / f"{problem_key}_ccdf_panel.png",
        title=f"{problem_key}: {cfg['title']} — Iteration CCDF comparison",
        newton_values=newton_values,
        safeguarded_values=safeguarded_values,
    )

    print(f"\n[{problem_key}] Newton iteration summary")
    for k, v in summarize_iterations(newton_values).items():
        print(f"  {k}: {v:.6f}")

    print(f"\n[{problem_key}] Safeguarded Newton iteration summary")
    for k, v in summarize_iterations(safeguarded_values).items():
        print(f"  {k}: {v:.6f}")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot empirical CCDFs of iteration counts for Newton and Safeguarded Newton."
    )
    parser.add_argument("--problem", type=str, default="all", help="P1, P2, P3, P4, or all")
    parser.add_argument("--n", type=int, default=1000, help="Number of initial guesses")
    parser.add_argument("--tol", type=float, default=1e-10, help="Residual tolerance")
    parser.add_argument("--max-iter", type=int, default=100, help="Maximum iterations")
    parser.add_argument("--df-tol", type=float, default=1e-14, help="Derivative near-zero tolerance for Newton")
    parser.add_argument("--stagnation-tol", type=float, default=1e-14, help="Stagnation tolerance for Newton")
    parser.add_argument("--tol-x", type=float, default=None, help="Optional step tolerance for Newton")
    parser.add_argument(
        "--numerical-derivative",
        action="store_true",
        help="Use numerical derivative in NewtonSolver",
    )
    parser.add_argument("--xmin", type=float, default=None, help="Override xmin")
    parser.add_argument("--xmax", type=float, default=None, help="Override xmax")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/ccdf",
        help="Output root directory",
    )
    parser.add_argument(
        "--include-failures-as-cap",
        action="store_true",
        help="Include non-converged runs as K=max_iter in the CCDF.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)

    if args.problem.lower() == "all":
        problem_keys = list(BENCHMARKS.keys())
    else:
        if args.problem not in BENCHMARKS:
            raise ValueError(f"Unknown problem: {args.problem}. Choose from {list(BENCHMARKS)} or 'all'.")
        problem_keys = [args.problem]

    for problem_key in problem_keys:
        run_problem(
            problem_key=problem_key,
            output_root=output_root,
            n=args.n,
            tol=args.tol,
            max_iter=args.max_iter,
            df_tol=args.df_tol,
            stagnation_tol=args.stagnation_tol,
            tol_x=args.tol_x,
            numerical_derivative=args.numerical_derivative,
            xmin=args.xmin,
            xmax=args.xmax,
            include_failures_as_cap=args.include_failures_as_cap,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()