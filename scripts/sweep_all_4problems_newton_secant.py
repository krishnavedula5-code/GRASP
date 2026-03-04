import json
import math
import os
import time
import urllib.request
from datetime import datetime

# -----------------------------
# Config
# -----------------------------
BASE_URL = "http://127.0.0.1:8000"
COMPARE_URL = BASE_URL + "/compare"

OUT_DIR = "sweeps_out"
os.makedirs(OUT_DIR, exist_ok=True)

N = 1000
TOL = 1e-10
MAX_ITER = 100

# Secant second guess: x1 = x0 + DELTA
DELTA = 1e-3

# Sweep domain for x0
XMIN, XMAX = -4.0, 4.0

# Problems: (tag, expr, dexpr)
PROBLEMS = [
    ("P1_cubic_x3_minus_2x_plus_2", "x**3 - 2*x + 2", "3*x**2 - 2"),
    ("P2_cubic_x3_minus_x_minus_2", "x**3 - x - 2", "3*x**2 - 1"),
    ("P3_cosx_minus_x",            "math.cos(x) - x", "-math.sin(x) - 1"),
    ("P4_multiple_root",           "(x - 1)**2 * (x + 2)", "2*(x - 1)*(x + 2) + (x - 1)**2"),
]

# -----------------------------
# Helpers
# -----------------------------
def linspace(a, b, n):
    if n <= 1:
        return [float(a)]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]

def post_json(url, payload, timeout=60):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def now_utc():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

# -----------------------------
# Main sweep
# -----------------------------
def run_sweep(problem_tag, expr, dexpr):
    xs = linspace(XMIN, XMAX, N)

    out_path = os.path.join(
        OUT_DIR,
        f"sweep_{problem_tag}_N{N}_tol{TOL}_max{MAX_ITER}_{now_utc().replace(':','').replace('-','')}.json"
    )

    results = []
    failures = 0

    print(f"\n=== Running {problem_tag} ===")
    print(f"expr  = {expr}")
    print(f"dexpr = {dexpr}")
    print(f"x0 in [{XMIN}, {XMAX}], N={N}, secant delta={DELTA}, tol={TOL}, max_iter={MAX_ITER}")
    print(f"Output -> {out_path}\n")

    t0 = time.time()

    for i, x0 in enumerate(xs, start=1):
        payload = {
            "expr": expr,
            "dexpr": dexpr,
            "x0": float(x0),
            "x1": float(x0 + DELTA),   # used by secant
            "tol": TOL,
            "max_iter": MAX_ITER,
        }

        try:
            resp = post_json(COMPARE_URL, payload)

            # Store minimal + useful fields; keep whole response too if you want
            rec = {
                "request": payload,
                "_meta": {
                    "index": i,
                    "problem": problem_tag,
                },
                "newton": {
                    "summary": safe_get(resp, "newton", "summary", default=safe_get(resp, "newton", default={})),
                },
                "secant": {
                    "summary": safe_get(resp, "secant", "summary", default=safe_get(resp, "secant", default={})),
                },
                # keep run_id if your API returns it anywhere
                "run_id": safe_get(resp, "_meta", "run_id", default=safe_get(resp, "run_id")),
            }
            results.append(rec)

        except Exception as e:
            failures += 1
            results.append({
                "request": payload,
                "_meta": {"index": i, "problem": problem_tag, "error": str(e)},
                "newton": {"summary": {"status": "error", "stop_reason": "request_error"}},
                "secant": {"summary": {"status": "error", "stop_reason": "request_error"}},
            })

        if i % 50 == 0:
            elapsed = time.time() - t0
            print(f"[{problem_tag}] {i}/{N} done, failures={failures}, elapsed={elapsed:.1f}s")

    # Write output (single JSON file)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nDONE {problem_tag}. Wrote {len(results)} records. Failures={failures}")
    return out_path

def main():
    paths = []
    for tag, expr, dexpr in PROBLEMS:
        paths.append(run_sweep(tag, expr, dexpr))

    print("\nAll sweeps complete.")
    print("Files written:")
    for p in paths:
        print(" -", p)

if __name__ == "__main__":
    main()