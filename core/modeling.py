from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.ingestion import LoadedTable
from core.profiling import DatasetProfile


@dataclass
class AnalysisModel:
    domain: str
    domain_candidates: list[dict[str, float]]
    primary_table: str
    integrated_df: pd.DataFrame
    tables: dict[str, pd.DataFrame]
    relationships: list[dict[str, Any]]
    join_notes: list[str] = field(default_factory=list)
    measures: list[str] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=list)
    date_columns: list[str] = field(default_factory=list)


def _safe_merge(left: pd.DataFrame, right: pd.DataFrame, join_column: str, suffix: str) -> pd.DataFrame:
    overlap = [column for column in right.columns if column in left.columns and column != join_column]
    renamed = right.rename(columns={column: f"{suffix}_{column}" for column in overlap})
    return left.merge(renamed, on=join_column, how="left")


def build_analysis_model(tables: list[LoadedTable], profile: DatasetProfile) -> AnalysisModel:
    table_map = {table.name: table.df.copy() for table in tables}
    primary_df = table_map[profile.recommended_primary_table].copy()
    join_notes: list[str] = []
    used_tables = {profile.recommended_primary_table}

    for relationship in profile.relationships:
        if relationship["left_table"] in used_tables and relationship["right_table"] not in used_tables:
            right_name = relationship["right_table"]
            primary_df = _safe_merge(primary_df, table_map[right_name], relationship["column"], right_name)
            used_tables.add(right_name)
            join_notes.append(
                f"Merged '{right_name}' into '{profile.recommended_primary_table}' on '{relationship['column']}' ({relationship['cardinality']})."
            )
        elif relationship["right_table"] in used_tables and relationship["left_table"] not in used_tables:
            left_name = relationship["left_table"]
            primary_df = _safe_merge(primary_df, table_map[left_name], relationship["column"], left_name)
            used_tables.add(left_name)
            join_notes.append(
                f"Merged '{left_name}' into '{profile.recommended_primary_table}' on '{relationship['column']}' ({relationship['cardinality']})."
            )

    return AnalysisModel(
        domain=profile.domain,
        domain_candidates=profile.domain_candidates,
        primary_table=profile.recommended_primary_table,
        integrated_df=primary_df,
        tables=table_map,
        relationships=profile.relationships,
        join_notes=join_notes,
        measures=profile.measures,
        dimensions=profile.dimensions,
        date_columns=profile.date_columns,
    )
