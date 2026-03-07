from __future__ import annotations

"""
Automatic root discovery via sweep of initial guesses.

Purpose
-------
Given an equation f(x) and solver, sweep many initial guesses,
run the solver, collect converged terminal roots, and cluster them.

This enables:
- automatic basin detection
- per-root statistics
- generalized experiments
"""

from dataclasses import dataclass, field
from typing import List
from numerical_lab.expr.safe_eval import compile_expr
from numerical_lab.methods.newton import NewtonSolver


# ------------------------------
# Data structures
# ------------------------------

@dataclass
class RootCluster:
    root_id: int
    center: float
    members: List[float] = field(default_factory=list)
    support: int = 0


# ------------------------------
# Utility functions
# ------------------------------

def linspace(a: float, b: float, n: int) -> List[float]:
    if n < 2:
        return [float(a)]

    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]

def run_newton_once(
    expr: str,
    dexpr: str | None,
    x0: float,
    tol: float = 1e-10,
    max_iter: int = 100,
):
    f = compile_expr(expr)
    df = compile_expr(dexpr) if dexpr else None

    solver = NewtonSolver(
        f=f,
        df=df,
        x0=x0,
        tol=tol,
        max_iter=max_iter,
        numerical_derivative=(dexpr is None),
    )

    return solver.solve()


# ------------------------------
# Root clustering
# ------------------------------

def cluster_roots(roots: List[float], cluster_tol: float = 1e-6) -> List[RootCluster]:

    clusters: List[RootCluster] = []

    for r in roots:

        matched = False

        for c in clusters:

            if abs(r - c.center) <= cluster_tol:
                c.members.append(r)
                c.support += 1
                c.center = sum(c.members) / len(c.members)
                matched = True
                break

        if not matched:
            new_cluster = RootCluster(
                root_id=len(clusters),
                center=r,
                members=[r],
                support=1,
            )
            clusters.append(new_cluster)

    return clusters


# ------------------------------
# Main discovery pipeline
# ------------------------------

def discover_roots(
    expr: str,
    dexpr: str | None,
    xmin: float = -4.0,
    xmax: float = 4.0,
    n: int = 1000,
    tol: float = 1e-10,
    max_iter: int = 100,
    cluster_tol: float = 1e-6,
    residual_tol: float = 1e-8,
) -> List[RootCluster]:

    xs = linspace(xmin, xmax, n)
    f = compile_expr(expr)

    accepted_roots: List[float] = []

    for x0 in xs:
        result = run_newton_once(
            expr=expr,
            dexpr=dexpr,
            x0=x0,
            tol=tol,
            max_iter=max_iter,
        )

        candidate = result.root if result.root is not None else result.best_x

        if result.status != "converged":
            continue

        if candidate is None:
            continue

        try:
            fx = f(candidate)
        except Exception:
            continue

        if abs(fx) > residual_tol:
            continue

        accepted_roots.append(float(candidate))

    clusters = cluster_roots(accepted_roots, cluster_tol)
    return clusters

if __name__ == "__main__":
    clusters = discover_roots(
        expr="(x-1)**2*(x+2)",
        dexpr="2*(x-1)*(x+2) + (x-1)**2",
        xmin=-4.0,
        xmax=4.0,
        n=200,
        tol=1e-10,
        max_iter=100,
        cluster_tol=1e-4,
        residual_tol=1e-8,
    )

    print("\nDiscovered clusters:\n")
    for c in clusters:
        print(f"root_{c.root_id}: {c.center:.10f}  support={c.support}")