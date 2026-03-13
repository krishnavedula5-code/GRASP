from numerical_lab.methods.brent import BrentSolver


def run_case(name, f, a, b, tol=1e-8, max_iter=100):
    print(f"\n=== {name} ===")
    solver = BrentSolver(f, a, b, tol=tol, max_iter=max_iter)
    result = solver.solve()

    print("method      :", result.method)
    print("status      :", result.status)
    print("stop_reason :", result.stop_reason)
    print("root        :", result.root)
    print("iterations  :", result.iterations)
    print("best_x      :", result.best_x)
    print("best_fx     :", result.best_fx)
    print("n_f         :", result.n_f)
    print("n_df        :", result.n_df)

    if result.events:
        print("last_event  :", result.events[-1])

    return result


def main():
    # Test 1: valid bracket
    run_case(
        name="Valid bracket",
        f=lambda x: x**2 - 2,
        a=1.0,
        b=2.0,
    )

    # Test 2: exact endpoint root
    run_case(
        name="Exact endpoint root",
        f=lambda x: (x - 1.0) * (x + 2.0),
        a=1.0,
        b=3.0,
    )

    # Test 3: bad bracket
    run_case(
        name="Bad bracket",
        f=lambda x: x**2 + 1,
        a=-1.0,
        b=1.0,
    )


if __name__ == "__main__":
    main()