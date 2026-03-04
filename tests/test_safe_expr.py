import pytest
import math

from numerical_lab.expr.safe_eval import compile_expr, UnsafeExpressionError


def test_basic_expression():
    f = compile_expr("x**3 - x - 2")
    assert abs(f(1.0) - (-2.0)) < 1e-12


def test_math_functions():
    f = compile_expr("cos(x) - x")
    assert abs(f(0.0) - 1.0) < 1e-12


def test_constants():
    f = compile_expr("pi - 3.141592653589793")
    assert abs(f(0.0)) < 1e-12


def test_reject_attribute_access():
    with pytest.raises(UnsafeExpressionError):
        compile_expr("math.sin(x)")


def test_reject_import():
    with pytest.raises(Exception):
        compile_expr("__import__('os').system('echo hacked')")


def test_reject_subscript():
    with pytest.raises(UnsafeExpressionError):
        compile_expr("(1).__class__")


def test_reject_lambda():
    with pytest.raises(UnsafeExpressionError):
        compile_expr("(lambda x: x)(2)")