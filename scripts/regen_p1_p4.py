import json
import os
import urllib.request
from pathlib import Path

def linspace(a, b, n):
    if n < 2:
        return [float(a)]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]

def post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

BASE = "http://localhost:8000"   # or your ngrok base
URL = BASE + "/compare"

N = 1000
XS = linspace(-4.0, 4.0, N)

TOL = 1e-10
MAX_ITER = 100
X1_DELTA = 0.1

# ✅ ONLY regenerate the two affected sweeps (P1, P4)
PROBLEMS = [
    # tag matches your filenames used in tests
    ("P1_x3_minus_2x_plus_2", "x**3 - 2*x + 2", "3*x**2 - 2", -2.0, 0.0),
    ("P4_multiroot",          "(x-1)**2*(x+2)", "2*(x-1)*(x+2) + (x-1)**2", -3.0, -1.0),
]

# Save in project root so pytest FILES list works
OUT_DIR = Path(".")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for tag, expr, dexpr, a, b in PROBLEMS:
    runs = []
    print(f"\n=== Running {tag} ===")
    print("expr:", expr)
    print("dexpr:", dexpr)
    print("bracket:", (a, b))

    for i, x0 in enumerate(XS, start=1):
        payload = {
            "expr": expr,
            "dexpr": dexpr,
            "a": float(a),
            "b": float(b),
            "x0": float(x0),
            "x1": float(x0) + X1_DELTA,
            "tol": TOL,
            "max_iter": MAX_ITER,
            "numerical_derivative": False
        }

        runs.append(post_json(URL, payload))

        if i % 100 == 0:
            print(f"{tag}: {i}/{N} done")

    # ✅ Atomic write: write to tmp then replace
    out_path = OUT_DIR / f"sweep_{tag}.json"
    tmp_path = OUT_DIR / f"sweep_{tag}.json.tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)

    os.replace(tmp_path, out_path)
    print("Saved", out_path)

print("\nDONE: regenerated P1 + P4 sweeps.")