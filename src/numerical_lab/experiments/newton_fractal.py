from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


@dataclass
class RootCluster:
    root: complex
    count: int = 1


def cluster_root(roots: List[RootCluster], z: complex, tol: float = 1e-6) -> int:
    """
    Add z to an existing cluster if close enough, else create a new cluster.
    Returns the cluster index.
    """
    for i, cluster in enumerate(roots):
        if abs(z - cluster.root) <= tol:
            # simple running average update
            new_count = cluster.count + 1
            cluster.root = (cluster.root * cluster.count + z) / new_count
            cluster.count = new_count
            return i

    roots.append(RootCluster(root=z, count=1))
    return len(roots) - 1


def newton_iterate(
    f: Callable[[complex], complex],
    df: Callable[[complex], complex],
    z0: complex,
    tol: float = 1e-10,
    max_iter: int = 50,
    deriv_tol: float = 1e-14,
) -> tuple[str, Optional[complex], int]:
    """
    Run Newton iteration from complex initial guess z0.

    Returns:
        status: 'converged', 'derivative_zero', 'max_iter', 'nonfinite'
        root: converged point if converged else None
        iterations: number of iterations used
    """
    z = z0

    for k in range(1, max_iter + 1):
        fz = f(z)
        dfz = df(z)

        if not (np.isfinite(z.real) and np.isfinite(z.imag) and
                np.isfinite(fz.real) and np.isfinite(fz.imag) and
                np.isfinite(dfz.real) and np.isfinite(dfz.imag)):
            return "nonfinite", None, k

        if abs(dfz) < deriv_tol:
            return "derivative_zero", None, k

        z_next = z - fz / dfz

        if not (np.isfinite(z_next.real) and np.isfinite(z_next.imag)):
            return "nonfinite", None, k

        if abs(f(z_next)) < tol:
            return "converged", z_next, k

        z = z_next

    return "max_iter", None, max_iter


def compute_newton_fractal(
    f: Callable[[complex], complex],
    df: Callable[[complex], complex],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    nx: int = 600,
    ny: int = 600,
    tol: float = 1e-10,
    max_iter: int = 50,
    cluster_tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, List[RootCluster]]:
    """
    Compute root-id and iteration arrays over a complex grid.

    Returns:
        basin_ids: shape (ny, nx), integer root id or -1 for failure
        iters: shape (ny, nx), iteration count
        roots: discovered root clusters
    """
    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)

    basin_ids = np.full((ny, nx), -1, dtype=int)
    iters = np.zeros((ny, nx), dtype=int)
    roots: List[RootCluster] = []

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            z0 = complex(x, y)
            status, root, k = newton_iterate(
                f=f,
                df=df,
                z0=z0,
                tol=tol,
                max_iter=max_iter,
            )
            iters[iy, ix] = k

            if status == "converged" and root is not None:
                rid = cluster_root(roots, root, tol=cluster_tol)
                basin_ids[iy, ix] = rid
            else:
                basin_ids[iy, ix] = -1

    return basin_ids, iters, roots


def make_discrete_cmap(n_roots: int) -> ListedColormap:
    """
    Root colors + black for failure.
    Index -1 is remapped later to last color.
    """
    base_colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#17becf",  # cyan
        "#bcbd22",  # olive
        "#7f7f7f",  # gray
    ]

    colors = [base_colors[i % len(base_colors)] for i in range(n_roots)]
    colors.append("#000000")  # failures
    return ListedColormap(colors)


def save_root_summary(path: Path, roots: List[RootCluster]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("Discovered roots:\n")
        for i, cluster in enumerate(roots):
            f.write(
                f"root_{i}: "
                f"{cluster.root.real:.12f} "
                f"{cluster.root.imag:+.12f}i "
                f"(count={cluster.count})\n"
            )


def plot_newton_fractal(
    basin_ids: np.ndarray,
    roots: List[RootCluster],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    title: str,
    out_path: Path,
) -> None:
    """
    Plot basin ids as a discrete fractal image.
    """
    display_ids = basin_ids.copy()
    failure_index = len(roots)
    display_ids[display_ids < 0] = failure_index

    cmap = make_discrete_cmap(len(roots))

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        display_ids,
        origin="lower",
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_xlabel("Re(z₀)")
    ax.set_ylabel("Im(z₀)")
    ax.set_title(title)

    tick_positions = list(range(len(roots) + 1))
    tick_labels = [f"root_{i}" for i in range(len(roots))] + ["fail"]

    cbar = plt.colorbar(im, ax=ax, ticks=tick_positions)
    cbar.ax.set_yticklabels(tick_labels)
    cbar.set_label("Basin label")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    # Example 1: your P4 polynomial in the complex plane
    def f(z):
        return z**3 - 1

    def df(z):
        return 3*z**2

    xlim = (-2,2)
    ylim = (-2,2)

    basin_ids, iters, roots = compute_newton_fractal(
        f=f,
        df=df,
        xlim=xlim,
        ylim=ylim,
        nx=700,
        ny=700,
        tol=1e-10,
        max_iter=60,
        cluster_tol=1e-5,
    )

    out_dir = Path("outputs/fractals")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_path = out_dir / "newton_fractal_p4_complex.png"
    summary_path = out_dir / "newton_fractal_p4_complex_roots.txt"

    plot_newton_fractal(
        basin_ids=basin_ids,
        roots=roots,
        xlim=xlim,
        ylim=ylim,
        title="Newton fractal — (z-1)^2(z+2)",
        out_path=plot_path,
    )
    save_root_summary(summary_path, roots)

    print(f"Saved fractal image to: {plot_path}")
    print(f"Saved root summary to: {summary_path}")
    print("Discovered roots:")
    for i, cluster in enumerate(roots):
        print(
            f"  root_{i}: "
            f"{cluster.root.real:.12f} "
            f"{cluster.root.imag:+.12f}i"
        )


if __name__ == "__main__":
    main()