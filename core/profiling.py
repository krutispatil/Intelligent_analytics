from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any
import warnings

import numpy as np
import pandas as pd

from core.ingestion import LoadedTable


DOMAIN_PATTERNS = {
    "sales": ["sales", "revenue", "order", "invoice", "customer", "product", "discount", "gmv"],
    "marketing": ["campaign", "ctr", "conversion", "click", "impression", "roas", "cpc", "lead"],
    "finance": ["account", "balance", "credit", "debit", "expense", "profit", "cash", "transaction"],
    "hr": ["employee", "salary", "department", "tenure", "attrition", "performance", "headcount"],
    "real_estate": ["property", "listing", "rent", "sqft", "bedroom", "broker", "lease", "occupancy"],
    "podcast": ["episode", "download", "listen", "guest", "show", "subscriber", "audience", "stream"],
    "instagram_engagement": [
        "instagram",
        "reel",
        "story",
        "post",
        "likes",
        "comments",
        "shares",
        "saves",
        "followers",
        "reach",
        "impressions",
        "engagement_rate",
        "profile_visit",
        "creator",
        "caption",
        "hashtag",
    ],
    "social_media": [
        "engagement",
        "followers",
        "likes",
        "comments",
        "shares",
        "views",
        "impressions",
        "reach",
        "post",
        "creator",
        "handle",
        "subscribers",
    ],
    "streaming_service": [
        "watch_time",
        "watchtime",
        "viewer",
        "viewership",
        "content",
        "title",
        "genre",
        "subscription",
        "plan",
        "device",
        "stream",
        "streaming",
        "episode",
        "season",
        "completion_rate",
        "minutes_watched",
        "hours_watched",
        "engagement",
        "platform",
        "viewer_id",
    ],
    "e_commerce": [
        "cart",
        "checkout",
        "sku",
        "inventory",
        "product_id",
        "basket",
        "aov",
        "refund",
        "fulfillment",
    ],
    "retail": [
        "store",
        "store_id",
        "pos",
        "cashier",
        "receipt",
        "footfall",
        "aisle",
        "same_store_sales",
    ],
    "customer_support": [
        "ticket",
        "case",
        "resolution",
        "sla",
        "agent",
        "response_time",
        "first_response",
        "csat",
        "support_queue",
    ],
    "education": [
        "student",
        "course",
        "grade",
        "class",
        "teacher",
        "attendance",
        "exam",
        "assignment",
        "semester",
        "school",
    ],
    "healthcare": [
        "patient",
        "diagnosis",
        "treatment",
        "hospital",
        "doctor",
        "medical",
        "admission",
        "discharge",
        "prescription",
        "claim",
    ],
    "insurance": [
        "policy",
        "premium",
        "claim",
        "underwriting",
        "insured",
        "coverage",
        "renewal",
        "deductible",
    ],
    "banking": [
        "account",
        "branch",
        "ifsc",
        "loan",
        "deposit",
        "withdrawal",
        "balance",
        "interest",
        "emi",
    ],
    "investment": [
        "portfolio",
        "asset",
        "ticker",
        "stock",
        "bond",
        "nav",
        "return",
        "holdings",
        "benchmark",
    ],
    "manufacturing": [
        "plant",
        "machine",
        "production",
        "downtime",
        "defect",
        "assembly",
        "throughput",
        "oee",
        "batch",
        "quality_check",
    ],
    "supply_chain": [
        "supplier",
        "procurement",
        "purchase_order",
        "warehouse",
        "stock",
        "inventory",
        "lead_time",
        "shipment",
        "reorder",
    ],
    "web_analytics": ["session", "page", "bounce", "traffic", "visit", "duration", "source", "utm"],
    "shipping": ["shipment", "delivery", "carrier", "warehouse", "tracking", "freight", "fulfillment"],
    "saas": ["mrr", "arr", "churn", "retention", "activation", "subscription", "nps", "ltv"],
    "product_analytics": [
        "feature",
        "event",
        "activation",
        "retention",
        "funnel",
        "cohort",
        "user_id",
        "session_id",
        "clickstream",
        "screen",
    ],
    "mobile_app": [
        "install",
        "uninstall",
        "session",
        "device_id",
        "app_version",
        "screen_view",
        "in_app_purchase",
        "push",
        "os_version",
    ],
    "advertising": [
        "ad",
        "adset",
        "creative",
        "campaign",
        "impression",
        "click",
        "ctr",
        "cpm",
        "cpc",
        "roas",
    ],
    "subscription_business": [
        "subscription",
        "renewal",
        "billing",
        "plan",
        "upgrade",
        "downgrade",
        "cancellation",
        "churn",
        "trial",
    ],
    "telecom": [
        "subscriber",
        "sim",
        "arpu",
        "network",
        "call_drop",
        "data_usage",
        "tower",
        "recharge",
        "prepaid",
        "postpaid",
    ],
    "energy_utilities": [
        "meter",
        "consumption",
        "usage",
        "kwh",
        "outage",
        "tariff",
        "grid",
        "utility",
        "billing_cycle",
    ],
    "transportation": [
        "route",
        "trip",
        "passenger",
        "vehicle",
        "fleet",
        "arrival",
        "departure",
        "delay",
        "ride",
    ],
    "travel_hospitality": [
        "booking",
        "reservation",
        "hotel",
        "room",
        "guest",
        "checkin",
        "checkout",
        "occupancy",
        "adr",
        "revpar",
    ],
    "food_delivery": [
        "restaurant",
        "menu",
        "delivery_time",
        "driver",
        "order_value",
        "prep_time",
        "cuisine",
        "rating",
        "tip",
    ],
    "ride_hailing": [
        "driver",
        "rider",
        "pickup",
        "dropoff",
        "fare",
        "trip_distance",
        "surge",
        "ride_time",
    ],
    "iot_sensor": [
        "sensor",
        "temperature",
        "humidity",
        "pressure",
        "reading",
        "device",
        "telemetry",
        "signal",
        "timestamp",
    ],
    "cybersecurity": [
        "threat",
        "incident",
        "severity",
        "vulnerability",
        "attack",
        "ip_address",
        "malware",
        "alert",
        "firewall",
        "auth",
    ],
    "real_time_operations": [
        "event_time",
        "latency",
        "throughput",
        "uptime",
        "downtime",
        "error_rate",
        "availability",
        "queue",
    ],
    "sports": [
        "team",
        "player",
        "match",
        "score",
        "season",
        "league",
        "coach",
        "minute",
        "goal",
        "assist",
    ],
    "gaming": [
        "player",
        "match",
        "level",
        "xp",
        "quest",
        "session_length",
        "guild",
        "in_game_purchase",
        "achievement",
    ],
    "nonprofit": [
        "donation",
        "donor",
        "campaign",
        "fundraising",
        "pledge",
        "grant",
        "volunteer",
        "beneficiary",
    ],
    "government_public_sector": [
        "citizen",
        "municipality",
        "permit",
        "license",
        "ward",
        "district",
        "public_service",
        "taxpayer",
    ],
    "legal": [
        "case",
        "matter",
        "court",
        "hearing",
        "lawyer",
        "judge",
        "filing",
        "contract",
        "compliance",
    ],
    "procurement": [
        "vendor",
        "purchase_order",
        "rfq",
        "invoice",
        "contract",
        "supplier",
        "sourcing",
        "tender",
    ],
    "construction": [
        "project",
        "site",
        "contractor",
        "material",
        "schedule",
        "milestone",
        "budget",
        "labor_hours",
    ],
    "agriculture": [
        "farm",
        "crop",
        "yield",
        "soil",
        "harvest",
        "acre",
        "irrigation",
        "fertilizer",
    ],
    "pharmaceutical": [
        "drug",
        "trial",
        "dosage",
        "compound",
        "adverse_event",
        "patient",
        "site",
        "protocol",
    ],
    "research": [
        "experiment",
        "sample",
        "hypothesis",
        "observation",
        "lab",
        "study",
        "variable",
        "control_group",
    ],
    "crm": [
        "lead",
        "opportunity",
        "account",
        "contact",
        "pipeline",
        "deal_stage",
        "owner",
        "close_date",
    ],
    "survey_feedback": [
        "response",
        "respondent",
        "question",
        "answer",
        "rating",
        "feedback",
        "sentiment",
        "nps",
        "csat",
    ],
    "content_media": [
        "article",
        "author",
        "publication",
        "headline",
        "content",
        "topic",
        "read_time",
        "views",
        "newsletter",
    ],
    "youtube_analytics": [
        "video",
        "channel",
        "watch_hours",
        "thumbnail",
        "subscriber",
        "views",
        "likes",
        "comments",
        "impressions",
        "ctr",
    ],
    "general": [],
}


@dataclass
class ColumnProfile:
    name: str
    semantic_type: str
    dtype: str
    missing_pct: float
    unique_pct: float
    sample_values: list[str] = field(default_factory=list)


@dataclass
class TableProfile:
    table_name: str
    row_count: int
    column_count: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    id_like_columns: list[str]
    columns: list[ColumnProfile]


@dataclass
class DatasetProfile:
    domain: str
    domain_scores: dict[str, int]
    domain_candidates: list[dict[str, float]]
    tables: list[TableProfile]
    relationships: list[dict[str, Any]]
    recommended_primary_table: str
    measures: list[str]
    dimensions: list[str]
    date_columns: list[str]
    warnings: list[str]


def _semantic_type(series: pd.Series, name: str) -> str:
    lowered = name.lower()
    non_null = series.dropna()
    if non_null.empty:
        return "unknown"

    if "id" in lowered or lowered.endswith("_key"):
        return "id"

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if pd.api.types.is_numeric_dtype(series):
        if "pct" in lowered or "percent" in lowered or "rate" in lowered:
            return "percentage"
        if any(token in lowered for token in ["amount", "price", "cost", "revenue", "salary", "spend", "income"]):
            return "currency"
        return "numeric"

    stringified = non_null.astype(str).str.strip()
    if stringified.str.match(r"^[\$€£]?\s?-?\d[\d,]*(\.\d+)?%?$").mean() > 0.8:
        return "numeric_text"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed_dates = pd.to_datetime(stringified, errors="coerce")
    if parsed_dates.notna().mean() > 0.8:
        return "datetime_text"

    unique_ratio = non_null.nunique() / max(len(non_null), 1)
    avg_len = stringified.str.len().mean()
    looks_like_identifier = (
        stringified.str.contains(r"\d", regex=True).mean() > 0.6
        or stringified.str.contains(r"[_\-]", regex=True).mean() > 0.6
        or stringified.str.match(r"^[A-Z]{2,}\d+$").mean() > 0.3
    )

    if unique_ratio > 0.9 and avg_len < 24 and looks_like_identifier:
        return "id"
    if avg_len > 40:
        return "text"
    return "categorical"


def profile_table(table: LoadedTable) -> TableProfile:
    df = table.df
    profiles: list[ColumnProfile] = []
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    datetime_columns: list[str] = []
    id_like_columns: list[str] = []

    for column in df.columns:
        series = df[column]
        semantic_type = _semantic_type(series, column)
        profile = ColumnProfile(
            name=column,
            semantic_type=semantic_type,
            dtype=str(series.dtype),
            missing_pct=float(series.isna().mean() * 100),
            unique_pct=float(series.nunique(dropna=True) / max(len(series), 1) * 100),
            sample_values=series.dropna().astype(str).head(3).tolist(),
        )
        profiles.append(profile)

        if semantic_type in {"numeric", "currency", "percentage"} and pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
        elif semantic_type in {"datetime", "datetime_text"}:
            datetime_columns.append(column)
        elif semantic_type == "id":
            id_like_columns.append(column)
        else:
            categorical_columns.append(column)

    return TableProfile(
        table_name=table.name,
        row_count=len(df),
        column_count=len(df.columns),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        id_like_columns=id_like_columns,
        columns=profiles,
    )


def _build_dataset_text(tables: list[LoadedTable]) -> str:
    parts: list[str] = []
    for table in tables:
        parts.extend(table.df.columns.tolist())
        sample = table.df.head(50).astype(str)
        parts.extend(sample.values.flatten().tolist())
    return " ".join(parts).lower()


def detect_domain(tables: list[LoadedTable]) -> tuple[str, dict[str, int]]:
    text = _build_dataset_text(tables)
    scores = {domain: 0 for domain in DOMAIN_PATTERNS}
    for domain, terms in DOMAIN_PATTERNS.items():
        for term in terms:
            scores[domain] += text.count(term)

    all_columns = {column.lower() for table in tables for column in table.df.columns}
    for table in tables:
        for column in table.df.columns:
            normalized = column.lower()
            for domain, terms in DOMAIN_PATTERNS.items():
                if any(term in normalized for term in terms):
                    scores[domain] += 3

    signature_terms = {
        "streaming_service": {"watch_time", "minutes_watched", "hours_watched", "completion_rate", "season", "episode", "stream_quality", "viewer_id"},
        "instagram_engagement": {"reel", "story", "saves", "followers", "profile_visit", "hashtag", "caption"},
        "youtube_analytics": {"watch_hours", "thumbnail", "channel", "video", "subscriber"},
        "podcast": {"episode", "downloads", "listens", "guest", "show_name"},
        "real_estate": {"listing_price", "sqft", "bedroom", "bathroom", "property_type", "broker"},
        "sales": {"order_id", "invoice_id", "customer_id", "revenue", "units_sold"},
        "e_commerce": {"cart", "sku", "checkout", "refund", "inventory"},
        "marketing": {"campaign", "ctr", "roas", "impressions", "conversions"},
        "finance": {"balance", "credit", "debit", "transaction_id", "ledger"},
        "hr": {"employee_id", "department", "salary", "attrition", "tenure"},
        "shipping": {"tracking", "carrier", "shipment_id", "delivery_date", "freight"},
        "web_analytics": {"pageviews", "utm_source", "bounce_rate", "session_duration", "landing_page"},
        "saas": {"mrr", "arr", "churn", "trial", "activation_rate"},
        "customer_support": {"ticket_id", "sla", "resolution_time", "csat", "agent"},
        "education": {"student_id", "course", "grade", "attendance", "teacher"},
        "healthcare": {"patient_id", "diagnosis", "treatment", "admission_date", "doctor"},
        "manufacturing": {"machine_id", "downtime", "defect_rate", "throughput", "batch_id"},
        "sports": {"team", "player", "match", "score", "league"},
        "gaming": {"level", "xp", "guild", "session_length", "achievement"},
        "travel_hospitality": {"booking_id", "checkin", "checkout", "adr", "revpar"},
        "food_delivery": {"restaurant", "delivery_time", "driver_id", "cuisine", "tip"},
    }
    for domain, terms in signature_terms.items():
        hits = sum(1 for column in all_columns if column in terms or any(term in column for term in terms))
        if hits >= 2:
            scores[domain] += hits * 5

    best_domain = max(scores, key=scores.get)
    if scores[best_domain] < 3:
        best_domain = "general"
    return best_domain, scores


def rank_domain_candidates(scores: dict[str, int]) -> list[dict[str, float]]:
    total_score = sum(max(score, 0) for score in scores.values())
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    candidates: list[dict[str, float]] = []
    for domain, score in ranked[:3]:
        confidence = (score / total_score * 100) if total_score else 0.0
        candidates.append({"domain": domain, "score": float(score), "confidence": round(confidence, 1)})
    return candidates


def choose_business_measures(table_profile: TableProfile) -> list[str]:
    measures: list[str] = []
    profile_lookup = {column.name: column for column in table_profile.columns}
    exclude_patterns = ["_id", " id", "zipcode", "zip", "phone", "lat", "lon", "longitude", "latitude", "year", "month", "day"]
    preferred_patterns = [
        "revenue",
        "sales",
        "price",
        "cost",
        "spend",
        "profit",
        "amount",
        "value",
        "salary",
        "rent",
        "rate",
        "score",
        "count",
        "units",
        "views",
        "clicks",
        "impressions",
        "downloads",
        "hours",
        "minutes",
        "occupancy",
        "conversion",
    ]

    for column in table_profile.numeric_columns:
        profile = profile_lookup[column]
        lowered = column.lower()
        if any(pattern in lowered for pattern in exclude_patterns):
            continue
        if profile.unique_pct > 98 and not any(pattern in lowered for pattern in preferred_patterns):
            continue
        measures.append(column)

    if not measures:
        measures = [
            column
            for column in table_profile.numeric_columns
            if not any(pattern in column.lower() for pattern in exclude_patterns)
        ]

    return measures


def choose_dimensions(table_profile: TableProfile) -> list[str]:
    dimensions = []
    for column in table_profile.categorical_columns + table_profile.id_like_columns:
        lowered = column.lower()
        if lowered.endswith("_id") and table_profile.row_count > 0:
            continue
        dimensions.append(column)
    return dimensions


def infer_relationships(table_profiles: list[TableProfile], tables: list[LoadedTable]) -> list[dict[str, Any]]:
    table_lookup = {table.name: table.df for table in tables}
    relationships: list[dict[str, Any]] = []

    for i, left in enumerate(table_profiles):
        for right in table_profiles[i + 1:]:
            shared = sorted(set(left.id_like_columns + left.categorical_columns) & set(right.id_like_columns + right.categorical_columns))
            for column in shared:
                left_df = table_lookup[left.table_name]
                right_df = table_lookup[right.table_name]
                left_unique = left_df[column].nunique(dropna=True) / max(len(left_df), 1)
                right_unique = right_df[column].nunique(dropna=True) / max(len(right_df), 1)
                overlap = len(set(left_df[column].dropna().astype(str)) & set(right_df[column].dropna().astype(str)))
                if overlap == 0:
                    continue
                strength = overlap / max(
                    min(left_df[column].nunique(dropna=True), right_df[column].nunique(dropna=True)),
                    1,
                )
                if strength < 0.3:
                    continue
                relationships.append(
                    {
                        "left_table": left.table_name,
                        "right_table": right.table_name,
                        "column": column,
                        "strength": round(strength, 2),
                        "cardinality": _cardinality_label(left_unique, right_unique),
                    }
                )

    relationships.sort(key=lambda rel: rel["strength"], reverse=True)
    return relationships


def _cardinality_label(left_unique: float, right_unique: float) -> str:
    if left_unique > 0.95 and right_unique > 0.95:
        return "one_to_one"
    if left_unique > right_unique:
        return "one_to_many"
    if right_unique > left_unique:
        return "many_to_one"
    return "many_to_many"


def build_dataset_profile(tables: list[LoadedTable]) -> DatasetProfile:
    table_profiles = [profile_table(table) for table in tables]
    domain, scores = detect_domain(tables)
    domain_candidates = rank_domain_candidates(scores)
    relationships = infer_relationships(table_profiles, tables)

    ranked = sorted(
        table_profiles,
        key=lambda profile: (
            len(profile.numeric_columns) * 3
            + len(profile.datetime_columns) * 2
            + math.log(profile.row_count + 1)
        ),
        reverse=True,
    )
    primary = ranked[0].table_name

    primary_profile = next(profile for profile in table_profiles if profile.table_name == primary)
    warnings: list[str] = []
    if len(tables) > 1 and not relationships:
        warnings.append("Multiple files were loaded, but no reliable join keys were detected.")
    if not primary_profile.numeric_columns:
        warnings.append("The primary table has no numeric measures; insights will be mostly categorical.")

    return DatasetProfile(
        domain=domain,
        domain_scores=scores,
        domain_candidates=domain_candidates,
        tables=table_profiles,
        relationships=relationships,
        recommended_primary_table=primary,
        measures=choose_business_measures(primary_profile),
        dimensions=choose_dimensions(primary_profile),
        date_columns=primary_profile.datetime_columns,
        warnings=warnings,
    )
