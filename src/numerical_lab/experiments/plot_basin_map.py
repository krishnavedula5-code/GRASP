from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import csv
from pathlib import Path
from typing import List, Dict, Tuple

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
    """
    Return a basin label for plotting.

    Rules:
    - converged + valid root_id -> that root_id
    - anything else -> FAIL
    """
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
    out = []
    for r in rows:
        if (r.get("problem_id") or "").strip() == problem_id and (r.get("method") or "").strip() == method:
            out.append(r)
    return out


def build_label_mapping(labels: List[str]) -> Tuple[Dict[str, int], List[str]]:
    """
    Put normal root labels first, FAIL last.
    Example:
        ['root_0', 'root_1', 'FAIL'] -> {'root_0':0, 'root_1':1, 'FAIL':2}
    """
    unique = sorted(set(labels), key=lambda s: (s == "FAIL", s))
    mapping = {label: i for i, label in enumerate(unique)}
    return mapping, unique


def make_cmap(n: int) -> Tuple[ListedColormap, BoundaryNorm]:
    """
    Simple discrete colormap.
    Last color reserved for FAIL if present.
    """
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

    colors = []
    for i in range(n):
        colors.append(base_colors[i % len(base_colors)])

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries=list(range(n + 1)), ncolors=n)
    return cmap, norm


def plot_basin_map(
    rows: List[Dict[str, str]],
    problem_id: str,
    method: str,
    output_dir: Path,
) -> Path:
    if not rows:
        raise ValueError(f"No rows found for problem={problem_id}, method={method}")

    processed = []
    for r in rows:
        x0 = parse_float(r.get("x0", "0"))
        basin_label = normalize_root_id(r)
        processed.append((x0, basin_label))

    processed.sort(key=lambda t: t[0])

    xs = [t[0] for t in processed]
    labels = [t[1] for t in processed]

    label_to_int, ordered_labels = build_label_mapping(labels)
    values = [label_to_int[label] for label in labels]

    cmap, norm = make_cmap(len(ordered_labels))

    # 1D strip as an image with height 1
    data = [values]

    fig, ax = plt.subplots(figsize=(12, 2.2))
    im = ax.imshow(
        data,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        extent=[xs[0], xs[-1], 0, 1],
        interpolation="nearest",
    )

    ax.set_yticks([])
    ax.set_xlabel(r"Initial guess $x_0$")
    ax.set_title(f"Basin map — {problem_id} — {method}")
    
    # colorbar with readable labels
    cbar = plt.colorbar(im, ax=ax, ticks=list(range(len(ordered_labels))))
    label_map = {
        "0": "root ≈ -2",
        "1": "root ≈ 1",
        "FAIL": "fail"
    }

    display_labels = [label_map.get(l, l) for l in ordered_labels]
    cbar.ax.set_yticklabels(display_labels)
    cbar.set_label("Basin label")

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"basin_map_{problem_id}_{method}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    latest = find_latest_sweep()
    rows = load_rows(latest / "records.csv")

    # Change these as needed
    problem_id = "p4"
    method = "newton"

    out_path = plot_basin_map(
        rows=extract_problem_method_rows(rows, problem_id, method),
        problem_id=problem_id,
        method=method,
        output_dir=latest,
    )

    print(f"Saved basin map to: {out_path}")


if __name__ == "__main__":
    main()