from numerical_lab.benchmarks.benchmark_types import BenchmarkProblem
from numerical_lab.benchmarks.loader import load_benchmarks
from numerical_lab.benchmarks.registry import get, list_all, list_ids, register

__all__ = [
    "BenchmarkProblem",
    "load_benchmarks",
    "register",
    "get",
    "list_all",
    "list_ids",
]