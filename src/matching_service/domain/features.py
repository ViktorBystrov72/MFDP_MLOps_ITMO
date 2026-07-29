from __future__ import annotations

from typing import Optional


def safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def area_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return abs(a - b)


def floor_close(a: Optional[str], b: Optional[str], tol: int = 1) -> bool:
    try:
        fa = int(float(str(a).replace(",", ".")))
        fb = int(float(str(b).replace(",", ".")))
    except (TypeError, ValueError):
        return False
    return abs(fa - fb) <= tol


def rooms_equal(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return False
    return str(a).strip().lower() == str(b).strip().lower()
