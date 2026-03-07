from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

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
        "roots": [-1.7692923542386314],
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
        "roots": [1.5213797068045676],
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
        "roots": [0.7390851332151607],
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
        "roots": [-2.0, 1.0],
    },
}

STATUS_ORDER = [
    "converged",
    "max_iter",
    "derivative_zero",
    "stagnation",
    "nan_or_inf",
    "bad_bracket",
    "error",
]
STATUS_TO_INT = {s: i for i, s in enumerate(STATUS_ORDER)}


# ============================================================
# Data model
# ============================================================

@dataclass
class ScanRow:
    problem: str
    method: str
    x0: float
    status: str
    stop_reason: str
    iterations: int
    root: Optional[float]
    best_x: Optional[float]
    best_fx: Optional[float]
    residual: Optional[float]
    root_label: int
    n_f: int
    n_df: int
    event_derivative_zero: int
    event_stagnation: int
    event_nonfinite: int
    event_max_iter: int
    converged: int
    used_bisection: int
    num_bisection_steps: int
    used_newton_step: int
    num_newton_steps: int


# ============================================================
# Helpers
# ============================================================

def linspace(a: float, b: float, n: int) -> List[float]:
    if n <= 1:
        return [float(a)]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def has_event(result, code: str, kind: Optional[str] = None) -> int:
    code_u = code.upper()
    kind_l = None if kind is None else kind.lower()

    for ev in result.events:
        ev_code = str(ev.get("code", "")).upper()
        ev_kind = str(ev.get("kind", "")).lower()
        if ev_code == code_u:
            return 1
        if kind_l is not None and ev_kind == kind_l:
            return 1
    return 0


def safe_status(status: object) -> str:
    s = str(status)
    if s in STATUS_TO_INT:
        return s
    return "error"


def extract_residual(best_fx: Optional[float]) -> Optional[float]:
    if best_fx is None:
        return None
    return abs(float(best_fx))


def extract_step_counts(result) -> tuple[int, int, int, int]:
    num_bisection = 0
    num_newton = 0

    for rec in result.records:
        if rec.step_type == "bisection":
            num_bisection += 1
        elif rec.step_type == "newton":
            num_newton += 1

    used_bisection = 1 if num_bisection > 0 else 0
    used_newton = 1 if num_newton > 0 else 0
    return used_bisection, num_bisection, used_newton, num_newton


def classify_root(root: Optional[float], known_roots: List[float], tol: float = 1e-6) -> int:
    if root is None:
        return -1
    for i, r in enumerate(known_roots):
        if abs(float(root) - float(r)) <= tol:
            return i
    return -1


# ============================================================
# Solver wrappers
# ============================================================

def run_newton_scan(
    *,
    problem_key: str,
    f: Callable[[float], float],
    df: Callable[[float], float],
    known_roots: List[float],
    xs: List[float],
    tol: float,
    max_iter: int,
    df_tol: float,
    stagnation_tol: float,
    tol_x: Optional[float],
    numerical_derivative: bool,
) -> List[ScanRow]:
    rows: List[ScanRow] = []

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

        used_bisection, num_bisection_steps, used_newton_step, num_newton_steps = extract_step_counts(result)
        root_label = classify_root(result.root, known_roots)

        row = ScanRow(
            problem=problem_key,
            method="newton",
            x0=float(x0),
            status=safe_status(result.status),
            stop_reason=str(result.stop_reason),
            iterations=int(result.iterations),
            root=result.root,
            best_x=result.best_x,
            best_fx=result.best_fx,
            residual=extract_residual(result.best_fx),
            root_label=root_label,
            n_f=int(result.n_f),
            n_df=int(result.n_df),
            event_derivative_zero=has_event(result, "DERIVATIVE_ZERO", kind="derivative_too_small"),
            event_stagnation=has_event(result, "STAGNATION", kind="stagnation"),
            event_nonfinite=has_event(result, "NONFINITE", kind="nonfinite"),
            event_max_iter=1 if str(result.status) == "max_iter" else 0,
            converged=1 if str(result.status) == "converged" else 0,
            used_bisection=used_bisection,
            num_bisection_steps=num_bisection_steps,
            used_newton_step=used_newton_step,
            num_newton_steps=num_newton_steps,
        )
        rows.append(row)

    return rows


def run_safeguarded_newton_scan(
    *,
    problem_key: str,
    f: Callable[[float], float],
    df: Callable[[float], float],
    known_roots: List[float],
    a: float,
    b: float,
    xs: List[float],
    tol: float,
    max_iter: int,
) -> List[ScanRow]:
    rows: List[ScanRow] = []

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

        used_bisection, num_bisection_steps, used_newton_step, num_newton_steps = extract_step_counts(result)
        root_label = classify_root(result.root, known_roots)

        row = ScanRow(
            problem=problem_key,
            method="safeguarded_newton",
            x0=float(x0),
            status=safe_status(result.status),
            stop_reason=str(result.stop_reason),
            iterations=int(result.iterations),
            root=result.root,
            best_x=result.best_x,
            best_fx=result.best_fx,
            residual=extract_residual(result.best_fx),
            root_label=root_label,
            n_f=int(result.n_f),
            n_df=int(result.n_df),
            event_derivative_zero=has_event(result, "DERIVATIVE_ZERO", kind="derivative_zero"),
            event_stagnation=has_event(result, "STAGNATION", kind="stagnation"),
            event_nonfinite=has_event(result, "NONFINITE", kind="nonfinite"),
            event_max_iter=1 if str(result.status) == "max_iter" else 0,
            converged=1 if str(result.status) == "converged" else 0,
            used_bisection=used_bisection,
            num_bisection_steps=num_bisection_steps,
            used_newton_step=used_newton_step,
            num_newton_steps=num_newton_steps,
        )
        rows.append(row)

    return rows


# ============================================================
# CSV
# ============================================================

def write_csv(path: Path, rows: List[ScanRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


# ============================================================
# Plotting
# ============================================================

def plot_iterations(path: Path, rows: List[ScanRow], title: str, max_iter: int) -> None:
    xs_ok = [r.x0 for r in rows if r.status == "converged"]
    ys_ok = [r.iterations for r in rows if r.status == "converged"]

    xs_fail = [r.x0 for r in rows if r.status != "converged"]
    ys_fail = [r.iterations for r in rows if r.status != "converged"]

    plt.figure(figsize=(10, 4.8))
    if xs_ok:
        plt.scatter(xs_ok, ys_ok, s=10, label="converged")
    if xs_fail:
        plt.scatter(xs_fail, ys_fail, s=10, marker="x", label="failed")
    plt.axhline(max_iter, linestyle="--", linewidth=1, label="max_iter")
    plt.xlabel(r"initial guess $x_0$")
    plt.ylabel("iterations")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_outcomes(path: Path, rows: List[ScanRow], title: str) -> None:
    xs = [r.x0 for r in rows]
    ys = [STATUS_TO_INT.get(r.status, STATUS_TO_INT["error"]) for r in rows]

    plt.figure(figsize=(10, 4.0))
    plt.scatter(xs, ys, s=10)
    plt.yticks(range(len(STATUS_ORDER)), STATUS_ORDER)
    plt.xlabel(r"initial guess $x_0$")
    plt.ylabel("outcome")
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_bisection_fallbacks(path: Path, rows: List[ScanRow], title: str) -> None:
    xs = [r.x0 for r in rows]
    ys = [r.num_bisection_steps for r in rows]

    plt.figure(figsize=(10, 4.2))
    plt.scatter(xs, ys, s=10)
    plt.xlabel(r"initial guess $x_0$")
    plt.ylabel("number of bisection fallback steps")
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_root_basins(path: Path, rows: List[ScanRow], known_roots: List[float], title: str) -> None:
    xs = [r.x0 for r in rows]
    ys = [r.root_label for r in rows]

    plt.figure(figsize=(10, 4.0))
    plt.scatter(xs, ys, s=10)

    tick_positions = [-1] + list(range(len(known_roots)))
    tick_labels = ["failure/unclassified"] + [f"root {i}: {r:.6g}" for i, r in enumerate(known_roots)]

    plt.yticks(tick_positions, tick_labels)
    plt.xlabel(r"initial guess $x_0$")
    plt.ylabel("root label")
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()

def plot_root_basins_colored(path: Path, rows: List[ScanRow], known_roots: List[float], title: str) -> None:
    plt.figure(figsize=(10, 4.2))

    # failures / unclassified
    xs_fail = [r.x0 for r in rows if r.root_label == -1]
    ys_fail = [-1 for _ in xs_fail]
    if xs_fail:
        plt.scatter(xs_fail, ys_fail, s=12, label="failure/unclassified")

    # each classified root gets its own series
    for i, root_val in enumerate(known_roots):
        xs_i = [r.x0 for r in rows if r.root_label == i]
        ys_i = [i for _ in xs_i]
        if xs_i:
            plt.scatter(xs_i, ys_i, s=12, label=f"root {i}: {root_val:.6g}")

    tick_positions = [-1] + list(range(len(known_roots)))
    tick_labels = ["failure/unclassified"] + [f"root {i}: {r:.6g}" for i, r in enumerate(known_roots)]

    plt.yticks(tick_positions, tick_labels)
    plt.xlabel(r"initial guess $x_0$")
    plt.ylabel("attractor")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()

def plot_iteration_heatmap(path: Path, rows: List[ScanRow], title: str) -> None:
    xs = [r.x0 for r in rows]
    ys = [0.0 for _ in rows]  # single strip; color carries the iteration information
    cs = [r.iterations for r in rows]

    plt.figure(figsize=(10, 2.8))
    sc = plt.scatter(xs, ys, c=cs, s=14)
    plt.yticks([])
    plt.xlabel(r"initial guess $x_0$")
    plt.title(title)
    cbar = plt.colorbar(sc)
    cbar.set_label("iterations")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()

def plot_compare_iterations(
    path: Path,
    rows_newton: List[ScanRow],
    rows_safe: List[ScanRow],
    problem_title: str,
    max_iter: int,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    panels = [
        (axes[0], rows_newton, "Newton"),
        (axes[1], rows_safe, "Safeguarded Newton"),
    ]

    for ax, rows, label in panels:
        xs_ok = [r.x0 for r in rows if r.status == "converged"]
        ys_ok = [r.iterations for r in rows if r.status == "converged"]

        xs_fail = [r.x0 for r in rows if r.status != "converged"]
        ys_fail = [r.iterations for r in rows if r.status != "converged"]

        if xs_ok:
            ax.scatter(xs_ok, ys_ok, s=10, label="converged")
        if xs_fail:
            ax.scatter(xs_fail, ys_fail, s=10, marker="x", label="failed")
        ax.axhline(max_iter, linestyle="--", linewidth=1)
        ax.set_ylabel("iterations")
        ax.set_title(label)
        ax.legend()

    axes[-1].set_xlabel(r"initial guess $x_0$")
    fig.suptitle(f"{problem_title}: iterations vs initial guess")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_compare_outcomes(
    path: Path,
    rows_newton: List[ScanRow],
    rows_safe: List[ScanRow],
    problem_title: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)

    panels = [
        (axes[0], rows_newton, "Newton"),
        (axes[1], rows_safe, "Safeguarded Newton"),
    ]

    for ax, rows, label in panels:
        xs = [r.x0 for r in rows]
        ys = [STATUS_TO_INT.get(r.status, STATUS_TO_INT["error"]) for r in rows]
        ax.scatter(xs, ys, s=10)
        ax.set_yticks(range(len(STATUS_ORDER)))
        ax.set_yticklabels(STATUS_ORDER)
        ax.set_ylabel("outcome")
        ax.set_title(label)

    axes[-1].set_xlabel(r"initial guess $x_0$")
    fig.suptitle(f"{problem_title}: outcome map")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_compare_root_basins(
    path: Path,
    rows_newton: List[ScanRow],
    rows_safe: List[ScanRow],
    known_roots: List[float],
    problem_title: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.8), sharex=True)

    panels = [
        (axes[0], rows_newton, "Newton"),
        (axes[1], rows_safe, "Safeguarded Newton"),
    ]

    tick_positions = [-1] + list(range(len(known_roots)))
    tick_labels = ["failure/unclassified"] + [f"root {i}: {r:.6g}" for i, r in enumerate(known_roots)]

    for ax, rows, label in panels:
        xs = [r.x0 for r in rows]
        ys = [r.root_label for r in rows]
        ax.scatter(xs, ys, s=10)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
        ax.set_ylabel("root label")
        ax.set_title(label)

    axes[-1].set_xlabel(r"initial guess $x_0$")
    fig.suptitle(f"{problem_title}: root-attractor map")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def plot_compare_iteration_heatmaps(
    path: Path,
    rows_newton: List[ScanRow],
    rows_safe: List[ScanRow],
    problem_title: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True)

    panels = [
        (axes[0], rows_newton, "Newton"),
        (axes[1], rows_safe, "Safeguarded Newton"),
    ]

    scatters = []
    for ax, rows, label in panels:
        xs = [r.x0 for r in rows]
        ys = [0.0 for _ in rows]
        cs = [r.iterations for r in rows]
        sc = ax.scatter(xs, ys, c=cs, s=14)
        ax.set_yticks([])
        ax.set_title(label)
        scatters.append(sc)

    axes[-1].set_xlabel(r"initial guess $x_0$")
    fig.suptitle(f"{problem_title}: iteration heatmaps")
    cbar = fig.colorbar(scatters[-1], ax=axes, shrink=0.95)
    cbar.set_label("iterations")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
# ============================================================
# Summaries
# ============================================================

def summarize(rows: List[ScanRow]) -> Dict[str, float]:
    n = len(rows)
    if n == 0:
        return {}

    success = sum(r.converged for r in rows)

    return {
        "n": float(n),
        "success_rate": success / n,
        "failure_rate": 1.0 - success / n,
        "expected_iterations": sum(r.iterations for r in rows) / n,
        "avg_n_f": sum(r.n_f for r in rows) / n,
        "avg_n_df": sum(r.n_df for r in rows) / n,
        "derivative_zero_event_rate": sum(r.event_derivative_zero for r in rows) / n,
        "stagnation_event_rate": sum(r.event_stagnation for r in rows) / n,
        "nonfinite_event_rate": sum(r.event_nonfinite for r in rows) / n,
        "max_iter_rate": sum(r.event_max_iter for r in rows) / n,
        "used_bisection_rate": sum(r.used_bisection for r in rows) / n,
        "avg_num_bisection_steps": sum(r.num_bisection_steps for r in rows) / n,
    }


def print_summary(problem_key: str, method: str, rows: List[ScanRow]) -> None:
    s = summarize(rows)
    print(f"\n[{problem_key}] {method}")
    for k, v in s.items():
        print(f"  {k}: {v:.6f}")


# ============================================================
# Runner
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
    compare: bool,
) -> None:
    cfg = BENCHMARKS[problem_key]

    f = cfg["f"]  # type: ignore[assignment]
    df = cfg["df"]  # type: ignore[assignment]
    a = float(cfg["a"])
    b = float(cfg["b"])
    known_roots = list(cfg["roots"])  # type: ignore[arg-type]

    x_left = float(cfg["xmin"] if xmin is None else xmin)
    x_right = float(cfg["xmax"] if xmax is None else xmax)
    xs = linspace(x_left, x_right, n)

    out_dir = output_root / problem_key
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_newton = run_newton_scan(
        problem_key=problem_key,
        f=f,
        df=df,
        known_roots=known_roots,
        xs=xs,
        tol=tol,
        max_iter=max_iter,
        df_tol=df_tol,
        stagnation_tol=stagnation_tol,
        tol_x=tol_x,
        numerical_derivative=numerical_derivative,
    )

    write_csv(out_dir / "newton_failure_scan.csv", rows_newton)
    plot_iterations(
        out_dir / "newton_iterations.png",
        rows_newton,
        f"{problem_key}: {cfg['title']} — Newton iterations vs initial guess",
        max_iter=max_iter,
    )
    plot_outcomes(
        out_dir / "newton_outcomes.png",
        rows_newton,
        f"{problem_key}: {cfg['title']} — Newton outcome map",
    )
    plot_root_basins(
        out_dir / "newton_root_basins.png",
        rows_newton,
        known_roots,
        f"{problem_key}: {cfg['title']} — Newton root-attractor map",
    )
    plot_root_basins_colored(
        out_dir / "newton_root_basins_colored.png",
        rows_newton,
        known_roots,
        f"{problem_key}: {cfg['title']} — Newton colored root-attractor map",
    )
    plot_iteration_heatmap(
        out_dir / "newton_iteration_heatmap.png",
        rows_newton,
        f"{problem_key}: {cfg['title']} — Newton iteration heatmap",
    )
    print_summary(problem_key, "newton", rows_newton)

    if compare:
        rows_safe = run_safeguarded_newton_scan(
            problem_key=problem_key,
            f=f,
            df=df,
            known_roots=known_roots,
            a=a,
            b=b,
            xs=xs,
            tol=tol,
            max_iter=max_iter,
        )

        write_csv(out_dir / "safeguarded_newton_failure_scan.csv", rows_safe)
        plot_iterations(
            out_dir / "safeguarded_newton_iterations.png",
            rows_safe,
            f"{problem_key}: {cfg['title']} — Safeguarded Newton iterations vs initial guess",
            max_iter=max_iter,
        )
        plot_outcomes(
            out_dir / "safeguarded_newton_outcomes.png",
            rows_safe,
            f"{problem_key}: {cfg['title']} — Safeguarded Newton outcome map",
        )
        plot_bisection_fallbacks(
            out_dir / "safeguarded_newton_bisection_fallbacks.png",
            rows_safe,
            f"{problem_key}: {cfg['title']} — Safeguarded Newton bisection fallback count",
        )
        plot_root_basins(
            out_dir / "safeguarded_newton_root_basins.png",
            rows_safe,
            known_roots,
            f"{problem_key}: {cfg['title']} — Safeguarded Newton root-attractor map",
        )
        plot_root_basins_colored(
            out_dir / "safeguarded_newton_root_basins_colored.png",
            rows_safe,
            known_roots,
            f"{problem_key}: {cfg['title']} — Safeguarded Newton colored root-attractor map",
        )
        plot_iteration_heatmap(
            out_dir / "safeguarded_newton_iteration_heatmap.png",
            rows_safe,
            f"{problem_key}: {cfg['title']} — Safeguarded Newton iteration heatmap",
        )
        plot_compare_iterations(
            out_dir / "compare_iterations_newton_vs_safeguarded.png",
            rows_newton,
            rows_safe,
            problem_title=f"{problem_key}: {cfg['title']}",
            max_iter=max_iter,
        )
        plot_compare_outcomes(
            out_dir / "compare_outcomes_newton_vs_safeguarded.png",
            rows_newton,
            rows_safe,
            problem_title=f"{problem_key}: {cfg['title']}",
        )
        plot_compare_root_basins(
            out_dir / "compare_root_basins_newton_vs_safeguarded.png",
            rows_newton,
            rows_safe,
            known_roots,
            problem_title=f"{problem_key}: {cfg['title']}",
        )

        plot_compare_iteration_heatmaps(
            out_dir / "compare_iteration_heatmaps_newton_vs_safeguarded.png",
            rows_newton,
            rows_safe,
            problem_title=f"{problem_key}: {cfg['title']}",
        )

        

        print_summary(problem_key, "safeguarded_newton", rows_safe)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Newton failure regions and compare with platform-native safeguarded Newton."
    )
    parser.add_argument("--problem", type=str, default="P1", help="P1, P2, P3, P4, or all")
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
        default="outputs/newton_regions",
        help="Output root directory",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also run safeguarded Newton and produce side-by-side comparison figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)

    if args.problem.lower() == "all":
        problems = list(BENCHMARKS.keys())
    else:
        if args.problem not in BENCHMARKS:
            raise ValueError(
                f"Unknown problem: {args.problem}. Choose from {list(BENCHMARKS)} or 'all'."
            )
        problems = [args.problem]

    for problem_key in problems:
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
            compare=args.compare,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()