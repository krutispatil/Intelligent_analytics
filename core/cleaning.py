from __future__ import annotations

from dataclasses import dataclass, field
import re

import numpy as np
import pandas as pd

from core.ingestion import LoadedTable
from core.profiling import TableProfile


@dataclass
class CleaningReport:
    table_name: str
    actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    logical_rules: list[dict[str, str]] = field(default_factory=list)
    original_shape: tuple[int, int] = (0, 0)
    final_shape: tuple[int, int] = (0, 0)


def _record_rule(report: CleaningReport, rule_name: str, columns: str, impact: str) -> None:
    report.logical_rules.append({"rule": rule_name, "columns": columns, "impact": impact})


def _coerce_numeric_text(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(r"[\$,€£,%]", "", regex=True)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _canonical_category_label(values: list[str]) -> str:
    cleaned_values = [value.strip() for value in values if value and value.strip()]
    if not cleaned_values:
        return ""

    representative = max(cleaned_values, key=len)
    words = representative.split()
    if words and all(word.isalpha() for word in words):
        return " ".join(word.capitalize() for word in words)
    return representative


def _standardize_categorical_text(series: pd.Series) -> tuple[pd.Series, int]:
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return series, 0

    normalized = non_null.str.strip().str.replace(r"\s+", " ", regex=True)
    lowered = normalized.str.casefold()

    if lowered.nunique() == normalized.nunique():
        return series, 0

    mapping: dict[str, str] = {}
    changes = 0
    for key in lowered.unique():
        original_variants = normalized[lowered == key].tolist()
        canonical = _canonical_category_label(original_variants)
        mapping[key] = canonical
        changes += sum(1 for value in original_variants if value != canonical)

    result = series.copy()
    mask = result.notna()
    result.loc[mask] = (
        result.loc[mask]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .map(lambda value: mapping.get(value.casefold(), value))
    )
    return result, changes


def _find_matching_columns(df: pd.DataFrame, patterns: list[str]) -> list[str]:
    matches: list[str] = []
    for column in df.columns:
        lowered = column.lower()
        if any(pattern in lowered for pattern in patterns) and pd.api.types.is_numeric_dtype(df[column]):
            matches.append(column)
    return matches


def _apply_non_negative_rules(df: pd.DataFrame, report: CleaningReport, protected_columns: set[str]) -> None:
    non_negative_patterns = [
        "count",
        "quantity",
        "units",
        "inventory",
        "stock",
        "impression",
        "click",
        "view",
        "download",
        "listen",
        "follower",
        "subscriber",
        "salary",
        "price",
        "cost",
        "spend",
        "revenue",
        "rent",
        "sqft",
        "area",
        "amount",
        "loan",
        "value",
    ]
    for column in _find_matching_columns(df, non_negative_patterns):
        invalid_mask = df[column] < 0
        invalid_count = int(invalid_mask.sum())
        if invalid_count:
            df.loc[invalid_mask, column] = np.nan
            protected_columns.add(column)
            report.actions.append(f"Set {invalid_count} negative values in '{column}' to null due to non-negative consistency rules.")
            _record_rule(report, "Non-negative enforcement", column, f"{invalid_count} values set to null")


def _apply_strict_positive_rules(df: pd.DataFrame, report: CleaningReport, protected_columns: set[str]) -> None:
    strict_positive_patterns = [
        "property_value",
        "valuation",
        "market_value",
        "loan_amount",
        "price",
        "salary",
        "rent",
        "sqft",
        "area",
    ]
    for column in _find_matching_columns(df, strict_positive_patterns):
        invalid_mask = df[column] <= 0
        invalid_count = int(invalid_mask.sum())
        if invalid_count:
            df.loc[invalid_mask, column] = np.nan
            protected_columns.add(column)
            report.actions.append(f"Set {invalid_count} non-positive values in '{column}' to null because the metric should be strictly positive.")
            _record_rule(report, "Strict-positive enforcement", column, f"{invalid_count} values set to null")


def _apply_percentage_normalization(df: pd.DataFrame, report: CleaningReport) -> None:
    for column in _find_matching_columns(df, ["pct", "percent", "percentage"]):
        valid_mask = df[column].between(1, 100, inclusive="both")
        if valid_mask.mean() > 0.8 and df[column].max() > 1:
            df.loc[valid_mask, column] = df.loc[valid_mask, column] / 100.0
            report.actions.append(f"Normalized '{column}' from percentage scale to 0-1 scale.")
            _record_rule(report, "Percentage normalization", column, "values converted from 0-100 to 0-1")


def _apply_pairwise_cap_rule(
    df: pd.DataFrame,
    report: CleaningReport,
    protected_columns: set[str],
    smaller_patterns: list[str],
    larger_patterns: list[str],
    rule_name: str,
) -> None:
    smaller_cols = _find_matching_columns(df, smaller_patterns)
    larger_cols = _find_matching_columns(df, larger_patterns)
    for smaller_col in smaller_cols:
        for larger_col in larger_cols:
            if smaller_col == larger_col:
                continue
            invalid_mask = df[smaller_col] > df[larger_col]
            invalid_count = int(invalid_mask.sum())
            if invalid_count:
                df.loc[invalid_mask, smaller_col] = df.loc[invalid_mask, larger_col]
                protected_columns.add(smaller_col)
                report.actions.append(
                    f"Capped {invalid_count} values in '{smaller_col}' to '{larger_col}' using the {rule_name} consistency rule."
                )
                _record_rule(report, rule_name, f"{smaller_col} <= {larger_col}", f"{invalid_count} values capped")
                break


def _apply_date_order_rules(df: pd.DataFrame, report: CleaningReport, protected_columns: set[str]) -> None:
    start_patterns = ["start", "created", "opened", "hire", "checkin", "pickup"]
    end_patterns = ["end", "closed", "resolved", "exit", "checkout", "dropoff"]
    start_cols = [col for col in df.columns if any(pattern in col.lower() for pattern in start_patterns) and pd.api.types.is_datetime64_any_dtype(df[col])]
    end_cols = [col for col in df.columns if any(pattern in col.lower() for pattern in end_patterns) and pd.api.types.is_datetime64_any_dtype(df[col])]

    for start_col in start_cols:
        for end_col in end_cols:
            invalid_mask = df[start_col].notna() & df[end_col].notna() & (df[start_col] > df[end_col])
            invalid_count = int(invalid_mask.sum())
            if invalid_count:
                df.loc[invalid_mask, end_col] = pd.NaT
                protected_columns.add(end_col)
                report.actions.append(
                    f"Cleared {invalid_count} invalid dates in '{end_col}' where it occurred before '{start_col}'."
                )
                _record_rule(report, "Date order validation", f"{start_col} <= {end_col}", f"{invalid_count} invalid dates cleared")
                break


def _apply_logical_consistency_rules(df: pd.DataFrame, report: CleaningReport) -> set[str]:
    protected_columns: set[str] = set()
    _apply_non_negative_rules(df, report, protected_columns)
    _apply_strict_positive_rules(df, report, protected_columns)
    _apply_percentage_normalization(df, report)

    pair_rules = [
        (["occupied"], ["total", "capacity"], "occupied-vs-total"),
        (["sold"], ["inventory", "stock", "available", "total"], "sold-vs-available"),
        (["click"], ["impression", "view"], "click-vs-impression"),
        (["open"], ["sent", "delivered"], "open-vs-sent"),
        (["conversion"], ["click", "visit", "lead"], "conversion-vs-traffic"),
        (["loan"], ["property_value", "market_value", "valuation"], "loan-vs-value"),
        (["bedroom", "bathroom"], ["room"], "room-count"),
        (["used", "consumed"], ["total", "capacity"], "used-vs-total"),
    ]
    for smaller_patterns, larger_patterns, rule_name in pair_rules:
        _apply_pairwise_cap_rule(df, report, protected_columns, smaller_patterns, larger_patterns, rule_name)

    _apply_date_order_rules(df, report, protected_columns)
    return protected_columns


def clean_table(table: LoadedTable, profile: TableProfile) -> tuple[LoadedTable, CleaningReport]:
    df = table.df.copy()
    report = CleaningReport(table_name=table.name, original_shape=df.shape)

    for column in df.columns:
        semantic = next(col.semantic_type for col in profile.columns if col.name == column)
        if semantic == "datetime_text":
            converted = pd.to_datetime(df[column], errors="coerce")
            if converted.notna().sum() > 0:
                df[column] = converted
                report.actions.append(f"Converted '{column}' to datetime.")
        elif semantic == "numeric_text":
            converted = _coerce_numeric_text(df[column])
            if converted.notna().sum() > 0:
                if df[column].astype(str).str.contains("%", regex=False).mean() > 0.5:
                    converted = converted / 100.0
                df[column] = converted
                report.actions.append(f"Converted '{column}' to numeric values.")
        elif semantic == "categorical":
            unique_count = df[column].nunique(dropna=True)
            if 1 < unique_count <= 50:
                standardized, changes = _standardize_categorical_text(df[column])
                if changes:
                    df[column] = standardized
                    report.actions.append(
                        f"Standardized {changes} categorical values in '{column}' for case and spacing consistency."
                    )

    protected_columns = _apply_logical_consistency_rules(df, report)

    duplicates = int(df.duplicated().sum())
    if duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
        report.actions.append(f"Removed {duplicates} duplicate rows.")

    for column in df.columns:
        missing = int(df[column].isna().sum())
        if not missing:
            continue
        if column in protected_columns:
            report.warnings.append(f"'{column}' has {missing} null values retained because logical consistency rules flagged them for review.")
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            fill_value = df[column].median() if abs(df[column].skew(skipna=True)) > 1 else df[column].mean()
            if pd.isna(fill_value):
                continue
            df[column] = df[column].fillna(fill_value)
            report.actions.append(f"Filled {missing} missing values in '{column}'.")
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            report.warnings.append(f"'{column}' has {missing} missing dates that were left as null.")
        else:
            mode = df[column].mode(dropna=True)
            if not mode.empty:
                df[column] = df[column].fillna(mode.iloc[0])
                report.actions.append(f"Imputed {missing} missing values in '{column}' with mode.")

    outlier_total = 0
    for column in df.select_dtypes(include=[np.number]).columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        mask = (df[column] < (q1 - 1.5 * iqr)) | (df[column] > (q3 + 1.5 * iqr))
        outlier_total += int(mask.sum())
    if outlier_total:
        report.warnings.append(f"Detected {outlier_total} outlier values across numeric columns.")

    report.final_shape = df.shape
    return (
        LoadedTable(
            name=table.name,
            source_name=table.source_name,
            sheet_name=table.sheet_name,
            df=df,
            metadata=table.metadata,
        ),
        report,
    )


def clean_tables(tables: list[LoadedTable], table_profiles: list[TableProfile]) -> tuple[list[LoadedTable], list[CleaningReport]]:
    profile_lookup = {profile.table_name: profile for profile in table_profiles}
    cleaned_tables: list[LoadedTable] = []
    reports: list[CleaningReport] = []

    for table in tables:
        cleaned, report = clean_table(table, profile_lookup[table.name])
        cleaned_tables.append(cleaned)
        reports.append(report)

    return cleaned_tables, reports
