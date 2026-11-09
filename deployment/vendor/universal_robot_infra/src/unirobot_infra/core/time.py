from time import monotonic_ns


def now_ns() -> int:
    return monotonic_ns()
