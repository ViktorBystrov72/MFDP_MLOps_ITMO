"""Read-only PostgreSQL repository for deal↔exposition matching."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Connection, Engine

from matching_service.infrastructure.db import get_engine


@dataclass(frozen=True)
class MatchingFrames:
    deals: pd.DataFrame
    flats: pd.DataFrame
    weak_labels: pd.DataFrame


class MatchingRepository:
    """Loads normalized matching data without mutating the source database."""

    def __init__(self, engine: Engine | None = None) -> None:
        self.engine = engine or get_engine()

    @staticmethod
    def _read(connection: Connection, query: str, params: dict) -> pd.DataFrame:
        return pd.read_sql(text(query), connection, params=params)

    def load_city_frames(
        self,
        city: str = "Тюмень",
        contract_from: date = date(2024, 1, 1),
    ) -> MatchingFrames:
        with self.engine.connect() as connection:
            transaction = connection.begin()
            connection.execute(text("SET TRANSACTION READ ONLY"))
            try:
                params = {"city": city, "contract_from": contract_from}
                deals = self._read(connection, self._deals_query(), params)
                flats = self._read(connection, self._flats_query(), {"city": city})
                labels = self._read(connection, self._labels_query(), params)
            finally:
                transaction.rollback()
        return MatchingFrames(deals=deals, flats=flats, weak_labels=labels)

    def load_candidates_for_deals(
        self,
        deal_ids: list[str],
        max_candidates_per_deal: int = 30,
    ) -> pd.DataFrame:
        if not deal_ids:
            return pd.DataFrame()
        statement = text(self._deal_candidates_query()).bindparams(bindparam("deal_ids", expanding=True))
        with self.engine.connect() as connection:
            transaction = connection.begin()
            connection.execute(text("SET TRANSACTION READ ONLY"))
            try:
                frame = pd.read_sql(
                    statement,
                    connection,
                    params={
                        "deal_ids": list(dict.fromkeys(deal_ids)),
                        "max_candidates": max_candidates_per_deal,
                    },
                )
            finally:
                transaction.rollback()
        return frame

    @staticmethod
    def _deals_query() -> str:
        return """
        SELECT
            d.id::text AS deal_id,
            d.location_id::text AS location_id_deal,
            l.building_id::text AS building_id_deal,
            l.complex_id::text AS complex_id_deal,
            ab.ndrf_object_id AS ndrf_object_id,
            d.contract_date,
            d.registration_date,
            d.realisation_contract,
            d.floor::text AS floor_deal,
            d.area::double precision AS area_deal,
            d.room_count::text AS room_count_deal,
            d.entrance_number::text AS entrance_deal,
            d.number_on_floor AS number_on_floor_deal,
            d.object_number_egrn::text AS object_number_egrn,
            d.object_number_pd::text AS object_number_pd,
            pu.real_estate_number::text AS planned_premise_number,
            coalesce(
                nullif(d.object_number_pd::text, ''),
                nullif(pu.real_estate_number::text, ''),
                nullif(d.object_number_egrn::text, '')
            ) AS flat_number_deal,
            d.price::double precision AS price_deal,
            d.object_description AS object_description_deal,
            d.location_description AS location_description_deal,
            d.planned_premise_id::text AS planned_premise_id,
            d.planned_premise_strategy,
            pu.floor AS pd_floor,
            pu.entrance_number::text AS pd_entrance,
            pu.area::double precision AS pd_area,
            pu.living_area::double precision AS pd_living_area,
            pu.room_count::text AS pd_room_count,
            pu.ceiling_height::double precision AS pd_ceiling_height,
            pu.purpose AS pd_purpose,
            pu.posted_at AS pd_posted_at,
            pu.is_actual AS pd_is_actual,
            d.is_primary,
            d.is_residential
        FROM public.deals d
        JOIN public.locations l ON l.id = d.location_id
        JOIN public.cities c ON c.id = l.city_id
        LEFT JOIN public.apartment_buildings ab ON ab.id = l.building_id
        LEFT JOIN public.planned_premises_unique pu ON pu.id = d.planned_premise_id
        WHERE c.name = :city
          AND d.is_primary
          AND d.is_residential
          AND NOT d.to_delete
          AND d.re_registered_deal_id IS NULL
          AND d.location_id IS NOT NULL
          AND d.contract_date >= :contract_from
        """

    @staticmethod
    def _flats_query() -> str:
        return """
        SELECT
            f.id::text AS flat_id,
            f.location_id::text AS location_id_exp,
            l.building_id::text AS building_id_exp,
            l.complex_id::text AS complex_id_exp,
            f.source_id::text AS source_id,
            s.name AS source_name,
            f.advert_id::text AS advert_id,
            f.advert_url,
            f.floor::text AS floor_exp,
            f.area::double precision AS area_exp,
            f.living_area::double precision AS living_area_exp,
            f.room_count::text AS room_count_exp,
            f.entrance_number::text AS entrance_exp,
            f.number_on_floor AS number_on_floor_exp,
            f.flat_number::text AS flat_number_exp,
            f.price::double precision AS price_exp,
            f.description AS description_exp,
            f.is_active,
            f.created_at,
            f.actualized_at
        FROM exposition.flats f
        JOIN public.locations l ON l.id = f.location_id
        JOIN public.cities c ON c.id = l.city_id
        LEFT JOIN public.sources s ON s.id = f.source_id
        WHERE c.name = :city
        """

    @staticmethod
    def _labels_query() -> str:
        return """
        SELECT
            con.deal_id::text AS deal_id,
            con.flat_id::text AS flat_id,
            con.coincidence_degree::text AS coincidence_degree,
            con.e_updated_at AS label_updated_at,
            CASE
                WHEN con.coincidence_degree::text = 'Полное совпадение' THEN 1
                WHEN con.coincidence_degree::text = 'Нет совпадений' THEN 0
                ELSE NULL
            END AS label
        FROM exposition.combined_flats_concatenation con
        JOIN public.deals d ON d.id = con.deal_id
        JOIN public.locations l ON l.id = d.location_id
        JOIN public.cities c ON c.id = l.city_id
        WHERE c.name = :city
          AND d.contract_date >= :contract_from
        """

    @staticmethod
    def _deal_candidates_query() -> str:
        return """
        WITH selected_deals AS (
            SELECT
                d.id::text AS deal_id,
                d.location_id,
                d.location_id::text AS location_id_deal,
                l.building_id::text AS building_id_deal,
                l.complex_id::text AS complex_id_deal,
                ab.ndrf_object_id,
                d.contract_date,
                d.registration_date,
                d.realisation_contract,
                d.floor::text AS floor_deal,
                d.area::double precision AS area_deal,
                d.room_count::text AS room_count_deal,
                d.entrance_number::text AS entrance_deal,
                d.number_on_floor AS number_on_floor_deal,
                d.object_number_egrn::text AS object_number_egrn,
                d.object_number_pd::text AS object_number_pd,
                pu.real_estate_number::text AS planned_premise_number,
                coalesce(
                    nullif(d.object_number_pd::text, ''),
                    nullif(pu.real_estate_number::text, ''),
                    nullif(d.object_number_egrn::text, '')
                ) AS flat_number_deal,
                d.price::double precision AS price_deal,
                d.object_description AS object_description_deal,
                d.location_description AS location_description_deal,
                d.planned_premise_id::text AS planned_premise_id,
                d.planned_premise_strategy,
                pu.floor AS pd_floor,
                pu.entrance_number::text AS pd_entrance,
                pu.area::double precision AS pd_area,
                pu.living_area::double precision AS pd_living_area,
                pu.room_count::text AS pd_room_count,
                pu.ceiling_height::double precision AS pd_ceiling_height,
                pu.purpose AS pd_purpose,
                pu.posted_at AS pd_posted_at,
                pu.is_actual AS pd_is_actual
            FROM public.deals d
            JOIN public.locations l ON l.id = d.location_id
            LEFT JOIN public.apartment_buildings ab ON ab.id = l.building_id
            LEFT JOIN public.planned_premises_unique pu ON pu.id = d.planned_premise_id
            WHERE d.id::text IN :deal_ids
              AND NOT d.to_delete
              AND d.location_id IS NOT NULL
        )
        SELECT
            d.*,
            f.flat_id,
            f.location_id_exp,
            f.building_id_exp,
            f.complex_id_exp,
            f.source_id,
            f.source_name,
            f.advert_id,
            f.advert_url,
            f.floor_exp,
            f.area_exp,
            f.living_area_exp,
            f.room_count_exp,
            f.entrance_exp,
            f.number_on_floor_exp,
            f.flat_number_exp,
            f.price_exp,
            f.description_exp,
            f.is_active,
            f.created_at,
            f.actualized_at,
            f.candidate_generation_rank
        FROM selected_deals d
        JOIN LATERAL (
            SELECT
                f.id::text AS flat_id,
                f.location_id::text AS location_id_exp,
                l.building_id::text AS building_id_exp,
                l.complex_id::text AS complex_id_exp,
                f.source_id::text AS source_id,
                s.name AS source_name,
                f.advert_id::text AS advert_id,
                f.advert_url,
                f.floor::text AS floor_exp,
                f.area::double precision AS area_exp,
                f.living_area::double precision AS living_area_exp,
                f.room_count::text AS room_count_exp,
                f.entrance_number::text AS entrance_exp,
                f.number_on_floor AS number_on_floor_exp,
                f.flat_number::text AS flat_number_exp,
                f.price::double precision AS price_exp,
                f.description AS description_exp,
                f.is_active,
                f.created_at,
                f.actualized_at,
                row_number() OVER (
                    ORDER BY
                        CASE
                            WHEN f.flat_number::text = d.flat_number_deal THEN 0
                            ELSE 1
                        END,
                        abs(f.area - d.area_deal),
                        f.actualized_at DESC,
                        f.id
                ) AS candidate_generation_rank
            FROM exposition.flats f
            JOIN public.locations l ON l.id = f.location_id
            LEFT JOIN public.sources s ON s.id = f.source_id
            WHERE f.location_id = d.location_id
              AND f.created_at::date < d.contract_date
              AND f.actualized_at::date >= d.contract_date - INTERVAL '3 months'
              AND (
                  d.area_deal IS NULL
                  OR f.area IS NULL
                  OR abs(f.area - d.area_deal) <= GREATEST(7.0, d.area_deal * 0.06)
              )
            ORDER BY
                CASE
                    WHEN f.flat_number::text = d.flat_number_deal THEN 0
                    ELSE 1
                END,
                abs(f.area - d.area_deal),
                f.actualized_at DESC,
                f.id
            LIMIT :max_candidates
        ) f ON TRUE
        """
