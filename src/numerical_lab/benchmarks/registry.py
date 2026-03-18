from __future__ import annotations

from typing import Dict, List

from numerical_lab.benchmarks.benchmark_types import BenchmarkProblem


_BENCHMARKS: Dict[str, BenchmarkProblem] = {}


def register(problem: BenchmarkProblem) -> None:
    if problem.problem_id in _BENCHMARKS:
        raise ValueError(f"Benchmark '{problem.problem_id}' is already registered.")
    _BENCHMARKS[problem.problem_id] = problem


def get(problem_id: str) -> BenchmarkProblem:
    if problem_id not in _BENCHMARKS:
        raise KeyError(f"Unknown benchmark_id: {problem_id}")
    return _BENCHMARKS[problem_id]


def list_all() -> List[BenchmarkProblem]:
    return list(_BENCHMARKS.values())


def list_ids() -> List[str]:
    return list(_BENCHMARKS.keys())


def clear() -> None:
    _BENCHMARKS.clear()