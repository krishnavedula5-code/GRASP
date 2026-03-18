from __future__ import annotations


_LOADED = False


def load_benchmarks() -> None:
    global _LOADED
    if _LOADED:
        return

    import numerical_lab.benchmarks.library.polynomial  # noqa: F401
    import numerical_lab.benchmarks.library.multiple_roots  # noqa: F401
    import numerical_lab.benchmarks.library.transcendental  # noqa: F401
    import numerical_lab.benchmarks.library.oscillatory  # noqa: F401
    import numerical_lab.benchmarks.library.pathological  # noqa: F401

    _LOADED = True