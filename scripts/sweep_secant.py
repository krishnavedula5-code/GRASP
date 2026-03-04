import numpy as np
import requests
import json
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================

API_URL = "http://localhost:8000/compare"  # adjust if needed
OUTPUT_FILE = "sweep_secant_1000.json"

N_RUNS = 1000
X_MIN = -4.0
X_MAX = 1.0

DELTA = 0.1  # x1 = x0 + DELTA

# Fixed solver parameters
TOL = 1e-10
MAX_ITER = 100

EXPR = "x**3 - 2*x + 2"
DEXPR = "3*x**2 - 2"

# ==========================================
# SWEEP
# ==========================================

x0_values = np.linspace(X_MIN, X_MAX, N_RUNS)

results = []

for x0 in tqdm(x0_values):

    payload = {
        "expr": EXPR,
        "dexpr": DEXPR,
        "a": -2.0,
        "b": 0.0,
        "x0": float(x0),
        "x1": float(x0 + DELTA),
        "tol": TOL,
        "max_iter": MAX_ITER,
        "numerical_derivative": False
    }

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            results.append(response.json())
        else:
            print("Warning: status", response.status_code)

    except Exception as e:
        print("Error:", e)

# ==========================================
# SAVE
# ==========================================

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} runs to {OUTPUT_FILE}")