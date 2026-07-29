from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional
from uuid import UUID


class CoincidenceDegree(str, Enum):
    FULL = "Полное совпадение"
    PART = "Частичное совпадение"
    ZERO = "Нет совпадений"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Deal:
    id: UUID
    location_id: Optional[UUID]
    complex_id: Optional[UUID]
    building_id: Optional[UUID]
    contract_date: Optional[date]
    floor: Optional[str]
    area: Optional[float]
    room_count: Optional[str]
    entrance_number: Optional[str]
    number_on_floor: Optional[int]
    flat_number: Optional[str]
    price: Optional[int]
    is_primary: bool


@dataclass(frozen=True)
class Listing:
    id: UUID
    location_id: Optional[UUID]
    complex_id: Optional[UUID]
    building_id: Optional[UUID]
    floor: Optional[str]
    area: Optional[float]
    room_count: Optional[str]
    entrance_number: Optional[str]
    number_on_floor: Optional[int]
    flat_number: Optional[str]
    price: Optional[int]
    is_active: bool
    created_at: Optional[date]
    actualized_at: Optional[date]


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
    listing_id: Optional[UUID]
    score: float
    stage: str
    coincidence: CoincidenceDegree
    needs_review: bool
