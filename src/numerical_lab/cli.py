from __future__ import annotations

import argparse
from dataclasses import asdict
from numerical_lab.diagnostics.hybrid_decisions import hybrid_decision_report
from typing import Tuple

from numerical_lab.engine.controller import NumericalEngine
from numerical_lab.engine.export import export_iterations_csv
from numerical_lab.engine.summary import build_comparison_summary
from numerical_lab.expr.safe_eval import compile_expr
from numerical_lab.expr.numerical_derivative import numerical_derivative
from numerical_lab.engine.trace import export_trace_json
from numerical_lab.engine.trace_dir import export_compare_traces_json_dir
from numerical_lab.diagnostics.explain import explain_run
from numerical_lab.engine.report import export_markdown_report


# ---------- Built-in demo functions (MVP) ----------
# Keep it small and high-quality. Expand later.

def f_cubic(x: float) -> float:
    # root near 1.521...
    return x**3 - x - 2

def df_cubic(x: float) -> float:
    return 3 * x**2 - 1

def f_cos_minus_x(x: float) -> float:
    # root near 0.739...
    import math
    return math.cos(x) - x

def df_cos_minus_x(x: float) -> float:
    import math
    return -math.sin(x) - 1

def f_exp_minus_3x(x: float) -> float:
    # root near ~0.619...
    import math
    return math.exp(x) - 3*x

def df_exp_minus_3x(x: float) -> float:
    import math
    return math.exp(x) - 3


EXAMPLES = {
    "cubic": (f_cubic, df_cubic, (1.0, 2.0), (1.0, 2.0), 1.5),
    "cosx_minus_x": (f_cos_minus_x, df_cos_minus_x, (0.0, 1.0), (0.0, 1.0), 0.7),
    "exp_minus_3x": (f_exp_minus_3x, df_exp_minus_3x, (0.0, 1.0), (0.0, 1.0), 0.5),
}
# tuple layout:
# (f, df, bracket(a,b), secant_guesses(x0,x1), newton_x0)


def _print_one_line_summary(method: str, summary) -> None:
    # summary is MethodSummary
    root_str = "None" if summary.root is None else f"{summary.root:.12g}"
    last_res = "None" if summary.last_residual is None else f"{summary.last_residual:.3e}"
    order = "None" if summary.observed_order is None else f"{summary.observed_order:.3f}"

    print(
        f"{method:10s}  status={summary.status:14s}  iters={summary.iterations:4d}  "
        f"root={root_str:>14s}  last|f|={last_res:>10s}  order≈{order:>6s}  "
        f"stab={summary.stability_label}"
    )


def _get_example(name: str):
    if name not in EXAMPLES:
        raise SystemExit(
            f"Unknown example '{name}'. Choose from: {', '.join(sorted(EXAMPLES.keys()))}"
        )
    return EXAMPLES[name]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="numerical_lab",
        description="Numerical Lab CLI (commercial MVP): run root-finding methods with diagnostics.",
    )

    source = parser.add_mutually_exclusive_group(required=True)

    source.add_argument(
        "--example",
        choices=sorted(EXAMPLES.keys()),
        help="Built-in function example to solve.",
    )

    source.add_argument(
        "--expr",
        help='Safe expression for f(x). Example: "x**3 - x - 2" or "cos(x) - x"',
    )

    parser.add_argument(
        "--dexpr",
        default=None,
        help='Optional safe expression for derivative f\'(x). Example: "3*x**2 - 1"',
    )

    parser.add_argument(
        "--numerical-derivative",
        action="store_true",
        help="Use safe numerical derivative if dexpr is not provided (Newton/Hybrid only).",
    )

    parser.add_argument(
        "--mode",
        default="compare",
        choices=["compare", "bisection", "newton", "secant", "hybrid"],
        help="Run a single method or compare all methods.",
    )

    parser.add_argument("--tol", type=float, default=1e-10, help="Tolerance.")
    parser.add_argument("--max-iter", type=int, default=100, help="Maximum iterations.")

    # Optional overrides (advanced)
    parser.add_argument("--a", type=float, default=None, help="Bracket left endpoint.")
    parser.add_argument("--b", type=float, default=None, help="Bracket right endpoint.")
    parser.add_argument("--x0", type=float, default=None, help="Initial guess x0 (Newton/Secant).")
    parser.add_argument("--x1", type=float, default=None, help="Second guess x1 (Secant).")

    # Export options
    parser.add_argument(
        "--export-csv",
        default=None,
        help="Export iteration table of the chosen method to CSV file path (single-method mode).",
    )
    
    parser.add_argument(
    "--export-json",
    default=None,
    help="Export JSON trace for the chosen method to file path (single-method mode).",
    )

    parser.add_argument(
    "--export-json-dir",
    default=None,
    help="(compare mode) Export JSON traces for all methods into this directory.",
    )
    
    parser.add_argument(
    "--export-report",
    default=None,
    help="(compare mode) Export a Markdown report to this file path.",
    )

    args = parser.parse_args()

    if args.example:
        f, df, bracket, secant_guesses, newton_x0 = _get_example(args.example)
    else:
        # expression mode
        f = compile_expr(args.expr)

        df = None
        if args.dexpr is not None:
            df = compile_expr(args.dexpr)
        elif args.numerical_derivative:
            df = numerical_derivative(f)

        # In expr-mode, user must provide sensible numerical inputs
        bracket = (args.a, args.b)
        secant_guesses = (args.x0, args.x1)
        newton_x0 = args.x0

    a, b = bracket
    sx0, sx1 = secant_guesses

    # Apply overrides if provided
    if args.a is not None:
        a = args.a
    if args.b is not None:
        b = args.b
    if args.x0 is not None:
        newton_x0 = args.x0
        sx0 = args.x0
    if args.x1 is not None:
        sx1 = args.x1
    
    if args.example is None:
    # expression mode: enforce required numeric inputs depending on mode

        if args.mode in ("bisection", "hybrid", "compare"):
            if a is None or b is None:
                raise SystemExit("In --expr mode, --a and --b are required for bisection/hybrid/compare.")
        if args.mode in ("secant", "compare"):
            if sx0 is None or sx1 is None:
                raise SystemExit("In --expr mode, --x0 and --x1 are required for secant/compare.")
        if args.mode in ("newton", "hybrid", "compare"):
            if newton_x0 is None:
                raise SystemExit("In --expr mode, --x0 is required for newton/hybrid/compare.")
            if df is None:
                raise SystemExit("Newton/Hybrid require derivative: provide --dexpr or use --numerical-derivative.")

    common_kwargs = {"tol": args.tol, "max_iter": args.max_iter}

    if args.mode == "compare":
        comp = NumericalEngine.compare_methods(
            f=f,
            df=df,
            bracket=(a, b),
            secant_guesses=(sx0, sx1),
            **common_kwargs,
        )
        summaries = build_comparison_summary(comp)
        for method in ["bisection", "newton", "secant", "hybrid"]:
            _print_one_line_summary(method, summaries[method])
            if args.export_report:
                export_markdown_report(comp, args.export_report, problem=(args.expr if args.expr else args.example))
                print(f"Report exported to: {args.export_report}")
            if args.export_json_dir:
                export_compare_traces_json_dir(comp, args.export_json_dir)
                print(f"JSON traces exported to directory: {args.export_json_dir}")
        return

    # Single-method modes
    if args.mode == "bisection":
        res, conv, stab = NumericalEngine.solve_bisection(f, a, b, **common_kwargs)
        summaries = build_comparison_summary({"bisection": (res, conv, stab)})
        _print_one_line_summary("bisection", summaries["bisection"])
        if args.export_csv:
            export_iterations_csv(res, args.export_csv)
            print(f"CSV exported to: {args.export_csv}")
        if args.export_json:
            export_trace_json(res, args.export_json, method_summary=summaries["bisection"],
                              extra={"explanation": explain_run(summaries["bisection"],res),
                                     "diagnostics": {"convergence": asdict(conv),
                                                     "stability": asdict(stab)},})
            print(f"JSON trace exported to: {args.export_json}")
        return

    if args.mode == "newton":
        res, conv, stab = NumericalEngine.solve_newton(f, df, newton_x0, **common_kwargs)
        summaries = build_comparison_summary({"newton": (res, conv, stab)})
        _print_one_line_summary("newton", summaries["newton"])
        if args.export_csv:
            export_iterations_csv(res, args.export_csv)
            print(f"CSV exported to: {args.export_csv}")
        if args.export_json:
            export_trace_json(res, args.export_json, method_summary=summaries["newton"],
                              extra={"explanation": explain_run(summaries["newton"], res),
                                     "diagnostics": {"convergence": asdict(conv),
                                                       "stability": asdict(stab)},})
            print(f"JSON trace exported to: {args.export_json}")
        return

    if args.mode == "secant":
        res, conv, stab = NumericalEngine.solve_secant(f, sx0, sx1, **common_kwargs)
        summaries = build_comparison_summary({"secant": (res, conv, stab)})
        _print_one_line_summary("secant", summaries["secant"])
        if args.export_csv:
            export_iterations_csv(res, args.export_csv)
            print(f"CSV exported to: {args.export_csv}")
        if args.export_json:
            export_trace_json(res, args.export_json, method_summary=summaries["secant"],
                              extra={"explanation": explain_run(summaries["secant"], res),
                                     "diagnostics": {"convergence": asdict(conv),
                                                     "stability": asdict(stab)},})
            print(f"JSON trace exported to: {args.export_json}")
        return
    
    if args.mode == "hybrid":
        res, conv, stab = NumericalEngine.solve_hybrid(f, df, a, b, **common_kwargs)
        summaries = build_comparison_summary({"hybrid": (res, conv, stab)})

        _print_one_line_summary("hybrid", summaries["hybrid"])

        if args.export_csv:
            export_iterations_csv(res, args.export_csv)
            print(f"CSV exported to: {args.export_csv}")

        if args.export_json:
            hyb = hybrid_decision_report(res)
            print("DEBUG: writing JSON to", args.export_json)
            export_trace_json(
                res,
                args.export_json,
                method_summary=summaries["hybrid"],
                extra={
                    "explanation": explain_run(summaries["hybrid"], res),
                    "diagnostics": {
                        "convergence": asdict(conv),
                        "stability": asdict(stab),
                        "hybrid_decisions": asdict(hyb),
                    },
                },
            )
            print(f"JSON trace exported to: {args.export_json}")
        return

    

if __name__ == "__main__":
    main()