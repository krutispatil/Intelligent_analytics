from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st

from core.insights import InsightEngine

PLOT_BACKGROUND = "rgba(0,0,0,0)"
PAPER_BACKGROUND = "rgba(0,0,0,0)"
FONT_COLOR = "#d7e1ec"
CYAN = "#73c6de"
CYAN_SOFT = "rgba(115, 198, 222, 0.65)"
STEEL = "#8ea0b8"
GOLD = "#d6b25e"
GOLD_SOFT = "#e4c87d"
RED = "#c96a6a"
GRID = "rgba(142, 160, 184, 0.18)"


def create_kpi_cards(df: pd.DataFrame, numeric_cols: list[str]) -> None:
    metric_scores: dict[str, float] = {}
    for column in numeric_cols:
        filled = df[column].fillna(df[column].median())
        if len(filled) < 2:
            continue
        trend = abs(stats.linregress(range(len(filled)), filled)[2])
        volatility = filled.std() / filled.mean() if filled.mean() else 0
        metric_scores[column] = float(trend + volatility)

    top_metrics = sorted(metric_scores.items(), key=lambda item: item[1], reverse=True)[:4]
    if not top_metrics:
        return

    cols = st.columns(len(top_metrics))
    for idx, (column, _) in enumerate(top_metrics):
        with cols[idx]:
            value = df[column].mean()
            delta = 0.0
            if len(df) > 1 and df[column].iloc[0] not in (0, np.nan):
                delta = float((df[column].iloc[-1] - df[column].iloc[0]) / df[column].iloc[0] * 100)
            st.metric(column.replace("_", " ").title(), f"{value:,.2f}", f"{delta:.1f}%" if delta else None)


def plot_distribution_analysis(df: pd.DataFrame, column: str, engine: InsightEngine) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, subplot_titles=(f"Distribution of {column}", "Box Plot"))
    fig.add_trace(
        go.Histogram(x=df[column], nbinsx=30, marker_color=CYAN_SOFT, name="Distribution"),
        row=1,
        col=1,
    )
    fig.add_trace(go.Box(x=df[column], marker_color=GOLD, name=column), row=2, col=1)

    anomalies = engine.detect_anomalies(df[column])
    if len(anomalies) > 0:
        fig.add_trace(
            go.Scatter(
                x=anomalies,
                y=[0] * len(anomalies),
                mode="markers",
                marker=dict(color=RED, size=9, symbol="x"),
                name=f"Anomalies ({len(anomalies)})",
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        height=600,
        template="plotly_dark",
        paper_bgcolor=PAPER_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        font_color=FONT_COLOR,
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def plot_time_series(df: pd.DataFrame, date_col: str, metric_col: str, engine: InsightEngine) -> tuple[go.Figure, dict[str, float | str]]:
    trend = engine.trend_analysis(date_col, metric_col)
    chart_df = df[[date_col, metric_col]].dropna().sort_values(date_col)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df[date_col],
            y=chart_df[metric_col],
            mode="lines+markers",
            name="Actual",
            line=dict(color=CYAN, width=3),
            marker=dict(color=GOLD, size=6),
        )
    )

    x = np.arange(len(chart_df))
    if len(chart_df) >= 2:
        baseline = chart_df[metric_col].mean() - trend["slope"] * len(chart_df) / 2
        fig.add_trace(
            go.Scatter(
                x=chart_df[date_col],
                y=trend["slope"] * x + baseline,
                mode="lines",
                name=f"Trend (R²={trend['r_squared']:.2f})",
                line=dict(dash="dash", color=GOLD_SOFT, width=2),
            )
        )

    fig.update_layout(
        title=f"{metric_col} over time",
        template="plotly_dark",
        paper_bgcolor=PAPER_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        font_color=FONT_COLOR,
        height=500,
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig, trend


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str]) -> go.Figure:
    corr = df[numeric_cols].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale=["#b45454", "#0e2235", "#73c6de"], title="Correlation Matrix")
    fig.update_layout(template="plotly_dark", paper_bgcolor=PAPER_BACKGROUND, height=600, font_color=FONT_COLOR)
    return fig


def plot_categorical_breakdown(df: pd.DataFrame, category_col: str, metric_col: str) -> go.Figure:
    fig = px.box(
        df,
        x=category_col,
        y=metric_col,
        color=category_col,
        title=f"{metric_col} by {category_col}",
        template="plotly_dark",
        color_discrete_sequence=[CYAN, GOLD, STEEL, "#5f89b0", "#c1a25a", "#6cc1aa"],
    )
    fig.update_layout(paper_bgcolor=PAPER_BACKGROUND, plot_bgcolor=PLOT_BACKGROUND, font_color=FONT_COLOR)
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def plot_segmentation_3d(df: pd.DataFrame, numeric_cols: list[str]) -> go.Figure | None:
    if len(numeric_cols) < 3 or "segment" not in df.columns:
        return None
    fig = px.scatter_3d(df, x=numeric_cols[0], y=numeric_cols[1], z=numeric_cols[2], color="segment", opacity=0.7, template="plotly_dark")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    return fig
