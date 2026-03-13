from numerical_lab.engine.controller import NumericalEngine


def main():
    f = lambda x: x**2 - 2
    df = lambda x: 2 * x

    results = NumericalEngine.compare_methods(
        f=f,
        df=df,
        bracket=(1.0, 2.0),
        secant_guesses=(1.0, 2.0),
        tol=1e-8,
        max_iter=100,
    )

    print("Returned methods:")
    for method, payload in results.items():
        result, report, stab = payload
        print(
            f"{method:20s} "
            f"status={result.status:12s} "
            f"root={result.root} "
            f"iters={result.iterations}"
        )


if __name__ == "__main__":
    main()