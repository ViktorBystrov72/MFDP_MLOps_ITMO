from __future__ import annotations


def safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def area_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return abs(a - b)


def floor_close(a: str | None, b: str | None, tol: int = 1) -> bool:
    try:
        fa = int(float(str(a).replace(",", ".")))
        fb = int(float(str(b).replace(",", ".")))
    except (TypeError, ValueError):
        return False
    return abs(fa - fb) <= tol


def rooms_equal(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    return str(a).strip().lower() == str(b).strip().lower()
