from pathlib import Path
import csv


def find_latest_sweep():
    base = Path("outputs/sweeps")
    folders = sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("sweep_"))
    return folders[-1]


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_root_id(row):
    status = (row.get("status") or "").strip().lower()
    rid = (row.get("root_id") or "").strip()
    if status == "converged" and rid:
        return rid
    return "FAIL"


def detect_boundaries(rows):

    data = []
    for r in rows:
        try:
            x0 = float(r["x0"])
        except:
            continue
        label = normalize_root_id(r)
        data.append((x0, label))

    data.sort()

    boundaries = []

    for i in range(1, len(data)):
        x_prev, l_prev = data[i - 1]
        x_cur, l_cur = data[i]

        if l_prev != l_cur:
            boundary = 0.5 * (x_prev + x_cur)
            boundaries.append(boundary)

    return boundaries


def main():

    latest = find_latest_sweep()
    rows = load_rows(latest / "records.csv")

    problem = "p4"
    method = "newton"

    subset = [
        r for r in rows
        if (r["problem_id"] == problem and r["method"] == method)
    ]

    boundaries = detect_boundaries(subset)

    print("Detected basin boundaries:")
    for b in boundaries:
        print(f"x ≈ {b:.6f}")


if __name__ == "__main__":
    main()