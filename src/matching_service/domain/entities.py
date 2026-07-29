from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from uuid import UUID


class CoincidenceDegree(str, Enum):
    FULL = "Полное совпадение"
    PART = "Частичное совпадение"
    ZERO = "Нет совпадений"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Deal:
    id: UUID
    location_id: UUID | None
    complex_id: UUID | None
    building_id: UUID | None
    contract_date: date | None
    floor: str | None
    area: float | None
    room_count: str | None
    entrance_number: str | None
    number_on_floor: int | None
    flat_number: str | None
    price: int | None
    is_primary: bool


@dataclass(frozen=True)
class Listing:
    id: UUID
    location_id: UUID | None
    complex_id: UUID | None
    building_id: UUID | None
    floor: str | None
    area: float | None
    room_count: str | None
    entrance_number: str | None
    number_on_floor: int | None
    flat_number: str | None
    price: int | None
    is_active: bool
    created_at: date | None
    actualized_at: date | None


@dataclass(frozen=True)
class MatchCandidate:
    deal_id: UUID
    listing_id: UUID
    score: float
    stage: str
    coincidence: CoincidenceDegree


@dataclass(frozen=True)
class MatchResult:
    deal_id: UUID
    listing_id: UUID | None
    score: float
    stage: str
    coincidence: CoincidenceDegree
    needs_review: bool
