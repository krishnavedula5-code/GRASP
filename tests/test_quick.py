from numerical_lab.engine.controller import NumericalEngine


def main():
    f = lambda x: x**2 - 2

    result, report, stab = NumericalEngine.solve_brent(
        f=f,
        a=1.0,
        b=2.0,
        tol=1e-8,
        max_iter=100,
    )

    print("RESULT")
    print("method      :", result.method)
    print("status      :", result.status)
    print("stop_reason :", result.stop_reason)
    print("root        :", result.root)
    print("iterations  :", result.iterations)

    print("\nREPORT")
    print(report)

    print("\nSTABILITY")
    print(stab)


if __name__ == "__main__":
    main()