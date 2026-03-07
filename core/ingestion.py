from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
import re
from typing import Any

import pandas as pd


@dataclass
class LoadedTable:
    name: str
    source_name: str
    df: pd.DataFrame
    sheet_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_column_name(name: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "unnamed_column"


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    renamed = {}
    seen: dict[str, int] = {}

    for col in df.columns:
        base = normalize_column_name(col)
        count = seen.get(base, 0)
        seen[base] = count + 1
        renamed[col] = f"{base}_{count + 1}" if count else base

    df.columns = [renamed[col] for col in df.columns]
    return df


def _read_excel_tables(uploaded_file: Any) -> list[LoadedTable]:
    raw = uploaded_file.getvalue()
    workbook = pd.ExcelFile(BytesIO(raw))
    tables: list[LoadedTable] = []

    for sheet_name in workbook.sheet_names:
        df = workbook.parse(sheet_name=sheet_name)
        if df.empty:
            continue
        tables.append(
            LoadedTable(
                name=normalize_column_name(f"{uploaded_file.name}_{sheet_name}"),
                source_name=uploaded_file.name,
                sheet_name=sheet_name,
                df=normalize_dataframe(df),
                metadata={"file_type": "excel"},
            )
        )

    return tables


def _read_csv_table(uploaded_file: Any) -> LoadedTable:
    df = pd.read_csv(uploaded_file)
    return LoadedTable(
        name=normalize_column_name(uploaded_file.name.rsplit(".", 1)[0]),
        source_name=uploaded_file.name,
        df=normalize_dataframe(df),
        metadata={"file_type": "csv"},
    )


def load_uploaded_tables(uploaded_files: list[Any]) -> list[LoadedTable]:
    tables: list[LoadedTable] = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name.lower()
        if filename.endswith(".csv"):
            tables.append(_read_csv_table(uploaded_file))
        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            tables.extend(_read_excel_tables(uploaded_file))

    return [table for table in tables if not table.df.empty]
