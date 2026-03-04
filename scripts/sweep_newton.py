import json
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

BASE = "http://localhost:8000"  # or your ngrok base
URL = BASE + "/compare"         # adjust if different

expr="x**3 - 2*x + 2"
dexpr="3*x**2 - 2"
a,b=-2.0,0.0

xs = linspace(-4.0, 4.0, 1000)
runs = []

for x0 in xs:
    payload = {
        "expr": expr,
        "dexpr": dexpr,
        "a": a, "b": b,
        "x0": float(x0),
        "x1": float(x0) + 0.1,
        "tol": 1e-10,
        "max_iter": 100,
        "numerical_derivative": False
    }
    runs.append(post_json(URL, payload))

open("outputs/sweep_api.json", "w").write(json.dumps(runs, indent=2))
print("Saved outputs/sweep_api.json")