import json
import os
import urllib.request

def linspace(a, b, n):
    if n < 2: return [float(a)]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]

def post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

BASE = "http://localhost:8000"   # or your ngrok base
URL = BASE + "/compare"

OUT_DIR = os.path.join("data", "v1")
os.makedirs(OUT_DIR, exist_ok=True)

N = 1000
XS = linspace(-4.0, 4.0, N)

TOL = 1e-10
MAX_ITER = 100
X1_DELTA = 0.1

# IMPORTANT: use only functions your safe-eval supports.
# If your engine supports "cos(x)" and "sin(x)" directly, use those.
# If it only supports "math.cos(x)", then use "math.cos(x)".
PROBLEMS = [
    # tag, expr, dexpr, (a,b)
    ("P1_x3_minus_2x_plus_2", "x**3 - 2*x + 2", "3*x**2 - 2", -2.0, 0.0),
    ("P2_x3_minus_x_minus_2", "x**3 - x - 2", "3*x**2 - 1", 1.0, 2.0),
    ("P3_cosx_minus_x",       "cos(x) - x",     "-sin(x) - 1", 0.0, 1.0),
    # Multiple root has no sign change at x=1. Use the simple root at x=-2.
    ("P4_multiroot",          "(x-1)**2*(x+2)", "2*(x-1)*(x+2) + (x-1)**2", -3.0, -1.0),
]

for tag, expr, dexpr, a, b in PROBLEMS:
    runs = []
    print(f"\n=== Running {tag} ===")
    print("expr:", expr)
    print("dexpr:", dexpr)
    print("bracket:", (a,b))

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

    tmp_path = os.path.join(OUT_DIR, f"sweep_{tag}.json.tmp")
    out_path = os.path.join(OUT_DIR, f"sweep_{tag}.json")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)

    os.replace(tmp_path, out_path)  # atomic on Windows
    print("Saved", out_path)

print("\nALL DONE.")