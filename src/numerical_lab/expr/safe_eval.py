from __future__ import annotations

import ast
import math
from typing import Callable, Dict, Any


_ALLOWED_FUNCS: Dict[str, Any] = {
    # common
    "abs": abs,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "floor": math.floor,
    "ceil": math.ceil,
}

_ALLOWED_CONSTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


class UnsafeExpressionError(ValueError):
    pass


def _eval_node(node: ast.AST, env: Dict[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, env)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise UnsafeExpressionError("Only numeric constants are allowed.")

    if isinstance(node, ast.Name):
        if node.id in env:
            return float(env[node.id])
        if node.id in _ALLOWED_CONSTS:
            return float(_ALLOWED_CONSTS[node.id])
        raise UnsafeExpressionError(f"Unknown name: {node.id}")

    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise UnsafeExpressionError("Operator not allowed.")
        left = _eval_node(node.left, env)
        right = _eval_node(node.right, env)

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right

        raise UnsafeExpressionError("Unsupported binary operator.")

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            raise UnsafeExpressionError("Unary operator not allowed.")
        val = _eval_node(node.operand, env)
        if isinstance(node.op, ast.UAdd):
            return +val
        if isinstance(node.op, ast.USub):
            return -val
        raise UnsafeExpressionError("Unsupported unary operator.")

    if isinstance(node, ast.Call):
        # Only allow direct function calls like sin(x), exp(x)
        if not isinstance(node.func, ast.Name):
            raise UnsafeExpressionError("Only direct function calls are allowed (no attributes).")
        fname = node.func.id
        if fname not in _ALLOWED_FUNCS:
            raise UnsafeExpressionError(f"Function not allowed: {fname}")

        if len(node.keywords) != 0:
            raise UnsafeExpressionError("Keyword arguments are not allowed.")

        args = [_eval_node(a, env) for a in node.args]
        return float(_ALLOWED_FUNCS[fname](*args))

    # Disallow everything else: Attribute, Subscript, Lambda, Comprehensions, etc.
    raise UnsafeExpressionError(f"Disallowed expression element: {type(node).__name__}")


def compile_expr(expr: str) -> Callable[[float], float]:
    """
    Compile a safe math expression in variable `x` into a callable f(x).

    Allowed:
      - numbers, x, pi, e
      - + - * / **, unary +/-
      - calls to a safe whitelist: sin, cos, exp, log, sqrt, abs, ...

    Disallowed:
      - attribute access (math.sin)
      - imports, names other than x/pi/e
      - indexing, comprehensions, lambdas, assignments, etc.
    """
    if not isinstance(expr, str) or not expr.strip():
        raise UnsafeExpressionError("Expression must be a non-empty string.")

    tree = ast.parse(expr, mode="eval")
    _validate_ast(tree)
    def f(x: float) -> float:
        return _eval_node(tree, {"x": float(x)})

    return f

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Constant,
)

def _validate_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        # disallow all statement-level nodes by virtue of parse(mode="eval"),
        # but still block dangerous expression-level nodes explicitly.
        if isinstance(node, (ast.Attribute, ast.Subscript, ast.Lambda, ast.DictComp, ast.ListComp,
                             ast.SetComp, ast.GeneratorExp, ast.Await, ast.Yield, ast.YieldFrom)):
            raise UnsafeExpressionError(f"Disallowed AST node: {type(node).__name__}")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise UnsafeExpressionError("Only direct function calls are allowed (no attributes).")
            if node.func.id not in _ALLOWED_FUNCS:
                raise UnsafeExpressionError(f"Function not allowed: {node.func.id}")
            if node.keywords:
                raise UnsafeExpressionError("Keyword arguments are not allowed.")

        if isinstance(node, ast.Name):
            if node.id not in ("x", *(_ALLOWED_CONSTS.keys()), *(_ALLOWED_FUNCS.keys())):
                # Allow x/pi/e and function names (function names appear in AST as Name within Call)
                raise UnsafeExpressionError(f"Unknown name: {node.id}")

        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BINOPS):
                raise UnsafeExpressionError("Operator not allowed.")

        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_UNARYOPS):
                raise UnsafeExpressionError("Unary operator not allowed.")