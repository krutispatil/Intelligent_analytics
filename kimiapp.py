import warnings

import plotly.graph_objects as go
from scipy import stats
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core.cleaning import clean_tables
from core.ingestion import load_uploaded_tables
from core.insights import InsightEngine
from core.modeling import build_analysis_model
from core.profiling import build_dataset_profile
from core.visuals import (
    create_kpi_cards,
    plot_categorical_breakdown,
    plot_correlation_heatmap,
    plot_distribution_analysis,
    plot_time_series,
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AutoInsight Studio",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=IBM+Plex+Sans:wght@400;600;700&display=swap');

    :root {
        --bg-top: #06101c;
        --bg-bottom: #0d1b2d;
        --panel: rgba(12, 27, 44, 0.88);
        --panel-soft: rgba(20, 38, 60, 0.86);
        --steel: rgba(148, 163, 184, 0.22);
        --text-main: #f5f7fb;
        --text-muted: #b8c4d6;
        --soft-cyan: #8fd3e8;
        --soft-cyan-strong: #4fb8d3;
        --gold: #d6b25e;
        --gold-soft: rgba(214, 178, 94, 0.16);
        --success: #4fbf9f;
        --danger: #d46a6a;
    }

    * { font-family: 'IBM Plex Sans', 'Manrope', sans-serif; }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(143, 211, 232, 0.10), transparent 26%),
            radial-gradient(circle at top right, rgba(214, 178, 94, 0.12), transparent 22%),
            linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
    }
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: var(--text-main);
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: var(--text-muted);
        margin-bottom: 2rem;
        max-width: 760px;
        font-size: 1.08rem;
    }
    .panel {
        background: linear-gradient(180deg, var(--panel) 0%, var(--panel-soft) 100%);
        border: 1px solid var(--steel);
        border-radius: 18px;
        padding: 1.2rem;
        box-shadow: 0 22px 48px rgba(0, 0, 0, 0.28);
        backdrop-filter: blur(10px);
    }
    .insight-card {
        background: linear-gradient(135deg, rgba(18, 32, 49, 0.97), rgba(11, 25, 40, 0.97));
        border: 1px solid rgba(143, 211, 232, 0.18);
        border-radius: 16px;
        padding: 1rem 1.05rem;
        margin-bottom: 0.85rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
    }
    .insight-title {
        color: var(--text-main);
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .insight-detail {
        color: var(--text-muted);
        margin: 0;
        line-height: 1.55;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(8, 18, 31, 0.98), rgba(12, 24, 39, 0.98));
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }
    [data-testid="stSidebar"] * {
        color: var(--text-main);
    }
    [data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(11, 25, 40, 0.92), rgba(17, 33, 52, 0.92));
        border: 1px solid rgba(143, 211, 232, 0.16);
        border-radius: 16px;
        padding: 0.85rem 1rem;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--text-main) !important;
    }
    [data-testid="stMetricDelta"] {
        color: var(--gold) !important;
    }
    .stButton button {
        background: linear-gradient(135deg, var(--soft-cyan-strong), #2c8faa);
        color: #04111b;
        border: 0;
        font-weight: 700;
        border-radius: 12px;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #9bd8ea, #49aac8);
        color: #04111b;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(20, 38, 60, 0.72);
        border-radius: 10px 10px 0 0;
        color: var(--text-muted);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(79, 184, 211, 0.24), rgba(214, 178, 94, 0.16)) !important;
        color: var(--text-main) !important;
    }
    .stAlert {
        border-radius: 14px;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: var(--text-main);
        letter-spacing: -0.02em;
    }
    .stMarkdown p, .stMarkdown li, .stCaption {
        color: var(--text-muted);
    }
</style>
""",
    unsafe_allow_html=True,
)


def generate_sample_data(dataset_type: str) -> list[pd.DataFrame]:
    np.random.seed(42)
    n = 200

    if dataset_type == "Sales + Customers":
        orders = pd.DataFrame(
            {
                "order_id": range(1001, 1001 + n),
                "order_date": pd.date_range("2025-01-01", periods=n, freq="D"),
                "customer_id": np.random.randint(1, 80, n),
                "product_category": np.random.choice(["Electronics", "Home", "Sports", "Beauty"], n),
                "revenue": np.random.lognormal(4.5, 0.5, n),
                "units_sold": np.random.poisson(4, n),
                "discount_pct": np.random.choice([0, 0.05, 0.1, 0.15], n),
            }
        )
        customers = pd.DataFrame(
            {
                "customer_id": range(1, 80),
                "region": np.random.choice(["North", "South", "East", "West"], 79),
                "segment": np.random.choice(["SMB", "Mid-Market", "Enterprise"], 79),
                "tenure_months": np.random.randint(1, 60, 79),
            }
        )
        return [orders, customers]

    if dataset_type == "HR":
        employees = pd.DataFrame(
            {
                "employee_id": range(2001, 2001 + n),
                "hire_date": pd.date_range("2021-01-01", periods=n, freq="W"),
                "department": np.random.choice(["Engineering", "Sales", "Marketing", "Support"], n),
                "salary": np.random.lognormal(11, 0.25, n),
                "performance_score": np.clip(np.random.normal(3.5, 0.8, n), 1, 5),
                "attrition_risk": np.random.choice(["Low", "Medium", "High"], n, p=[0.55, 0.3, 0.15]),
                "overtime_hours": np.random.exponential(5, n),
            }
        )
        return [employees]

    marketing = pd.DataFrame(
        {
            "campaign_id": range(301, 301 + n),
            "date": pd.date_range("2025-06-01", periods=n, freq="D"),
            "channel": np.random.choice(["Google", "Meta", "LinkedIn", "TikTok"], n),
            "spend": np.random.lognormal(5, 0.45, n),
            "impressions": np.random.lognormal(11, 0.6, n),
            "clicks": np.random.lognormal(7, 0.5, n),
            "conversions": np.random.poisson(20, n),
        }
    )
    return [marketing]


def sample_tables_to_uploaded_frames(sample_frames: list[pd.DataFrame], dataset_type: str):
    from core.ingestion import LoadedTable, normalize_dataframe, normalize_column_name

    tables = []
    for idx, frame in enumerate(sample_frames, start=1):
        tables.append(
            LoadedTable(
                name=normalize_column_name(f"{dataset_type}_{idx}"),
                source_name=f"{dataset_type}_{idx}.csv",
                df=normalize_dataframe(frame),
                metadata={"file_type": "sample"},
            )
        )
    return tables


def render_insight_cards(records):
    for record in records:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">{record.title}</div>
                <p class="insight-detail">{record.detail}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_actionable_business_insights(insights) -> None:
    for insight in insights:
        direction_color = {"good": "#10b981", "bad": "#ef4444", "neutral": "#f59e0b"}.get(insight.direction, "#3b82f6")
        st.markdown(
            f"""
            <div class="insight-card" style="border-left: 5px solid {direction_color};">
                <div class="insight-title">{insight.title}</div>
                <p class="insight-detail"><b>What happened:</b> {insight.what_happened}</p>
                <p class="insight-detail"><b>Why it happened:</b> {insight.why_it_happened}</p>
                <p class="insight-detail"><b>Actionable insight:</b> {insight.action}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def prettify_label(value: str) -> str:
    return value.replace("_", " ").title()


def summarize_cleaning_reports(cleaning_reports) -> dict[str, int]:
    return {
        "tables": len(cleaning_reports),
        "actions": sum(len(report.actions) for report in cleaning_reports),
        "warnings": sum(len(report.warnings) for report in cleaning_reports),
        "rules": sum(len(report.logical_rules) for report in cleaning_reports),
    }


def build_distribution_commentary(df: pd.DataFrame, metric: str, engine: InsightEngine) -> str:
    series = df[metric].dropna()
    if series.empty:
        return f"{metric.replace('_', ' ').title()} has no valid values to interpret."

    mean_value = series.mean()
    median_value = series.median()
    skew = series.skew()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    anomalies = engine.detect_anomalies(series)

    if abs(skew) < 0.3:
        shape_text = "fairly balanced around the center"
    elif skew > 0:
        shape_text = "right-skewed, with most values clustered lower and a tail extending toward higher values"
    else:
        shape_text = "left-skewed, with most values clustered higher and a tail extending toward lower values"

    if iqr == 0:
        spread_text = "very little spread between the middle 50% of observations"
    else:
        spread_text = f"the middle 50% of values fall roughly between {q1:,.2f} and {q3:,.2f}"

    if len(anomalies) == 0:
        outlier_text = "The box plot does not show notable extreme outliers."
    elif len(anomalies) == 1:
        outlier_text = "The box plot shows one clear extreme outlier beyond the main range."
    else:
        outlier_text = f"The box plot shows {len(anomalies)} clear extreme outliers beyond the main range."

    center_text = (
        f"The histogram suggests {metric.replace('_', ' ')} is {shape_text}. "
        f"Its average is {mean_value:,.2f} and median is {median_value:,.2f}, "
        f"which indicates {'a fairly symmetric center' if abs(mean_value - median_value) / max(abs(mean_value), 1) < 0.1 else 'the center is being pulled by uneven tails or large values'}."
    )

    return f"{center_text} In the box plot, {spread_text}. {outlier_text}"


def build_metric_takeaway(df: pd.DataFrame, metric: str, engine: InsightEngine) -> tuple[str, str]:
    series = df[metric].dropna()
    if series.empty:
        return (
            f"{metric.replace('_', ' ').title()} does not have enough usable values for interpretation.",
            "Upload more complete records for this metric to understand its pattern.",
        )

    metric_label = metric.replace("_", " ").title()
    mean_value = series.mean()
    median_value = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    anomalies = engine.detect_anomalies(series)
    skew = series.skew()

    if abs(skew) < 0.3:
        shape_summary = "mostly centered in a stable middle range"
    elif skew > 0:
        shape_summary = "clustered around lower values with a smaller number of high-value records"
    else:
        shape_summary = "clustered around higher values with a smaller number of low-value records"

    if len(anomalies) == 0:
        outlier_summary = "There are no major extreme records distorting the picture."
    elif len(anomalies) <= 3:
        outlier_summary = f"There are a few unusual records ({len(anomalies)}) that stand apart from the rest."
    else:
        outlier_summary = f"There are several unusual records ({len(anomalies)}) that are stretching the distribution."

    takeaway = (
        f"{metric_label} is {shape_summary}. Most records sit roughly between {q1:,.2f} and {q3:,.2f}, "
        f"while the typical value is closer to {median_value:,.2f} than the average of {mean_value:,.2f}."
    )

    if abs(mean_value - median_value) / max(abs(mean_value), 1) < 0.1:
        business_impact = "This metric is fairly consistent, so the average is a reasonable summary of typical performance."
    elif mean_value > median_value:
        business_impact = "A few larger values are pulling the average upward, so the average looks better than what a typical record is experiencing."
    else:
        business_impact = "A few lower values are pulling the average downward, so the average may understate what typical records are achieving."

    return takeaway, f"{business_impact} {outlier_summary}"


def choose_primary_category(df: pd.DataFrame, categories: list[str]) -> str | None:
    candidates = []
    for column in categories:
        unique_count = df[column].nunique(dropna=True)
        if 2 <= unique_count <= 12:
            candidates.append((column, unique_count))
    if candidates:
        candidates.sort(key=lambda item: item[1])
        return candidates[0][0]
    return categories[0] if categories else None


def render_visual_story(df: pd.DataFrame, engine: InsightEngine) -> None:
    rendered_any = False
    primary_metric = engine.numeric_cols[0] if engine.numeric_cols else None
    secondary_metric = engine.numeric_cols[1] if len(engine.numeric_cols) > 1 else primary_metric
    primary_date = engine.date_cols[0] if engine.date_cols else None
    primary_category = choose_primary_category(df, engine.cat_cols)

    if primary_date and primary_metric:
        st.subheader("Trend View")
        fig, trend = plot_time_series(df, primary_date, primary_metric, engine)
        st.plotly_chart(fig, use_container_width=True)
        time_df = df[[primary_date, primary_metric]].dropna().sort_values(primary_date)
        if not time_df.empty:
            peak_row = time_df.loc[time_df[primary_metric].idxmax()]
            low_row = time_df.loc[time_df[primary_metric].idxmin()]
            st.caption(
                f"{primary_metric.replace('_', ' ').title()} is {trend['trend'].lower()} over time, with the highest point around {peak_row[primary_date]} and the lowest around {low_row[primary_date]}. "
                f"The spread over time looks {trend['volatility_desc'].lower()}, so movement in this metric is {'noticeable' if trend['volatility_desc'] != 'Low' else 'fairly steady'} rather than flat."
            )
            rendered_any = True

    if primary_date and secondary_metric:
        st.subheader("Area Trend")
        area_df = df[[primary_date, secondary_metric]].dropna().sort_values(primary_date)
        if not area_df.empty:
            area_fig = px.area(
                area_df,
                x=primary_date,
                y=secondary_metric,
                title=f"{secondary_metric.replace('_', ' ').title()} Over Time",
                template="plotly_dark",
            )
            area_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(area_fig, use_container_width=True)
            change_pct = 0.0
            if len(area_df) > 1 and area_df[secondary_metric].iloc[0] != 0:
                change_pct = ((area_df[secondary_metric].iloc[-1] - area_df[secondary_metric].iloc[0]) / area_df[secondary_metric].iloc[0]) * 100
            st.caption(
                f"The filled area makes it easier to see how the total volume of {secondary_metric.replace('_', ' ')} builds and fades across time. "
                f"From start to end, this metric changed by {change_pct:.1f}%, which helps show whether the overall movement is expanding or contracting."
            )
            rendered_any = True

    if primary_category and primary_metric:
        grouped = (
            df[[primary_category, primary_metric]]
            .dropna()
            .groupby(primary_category, as_index=False)[primary_metric]
            .sum()
            .sort_values(primary_metric, ascending=False)
        )
        if not grouped.empty:
            st.subheader("Category Contribution")
            bar_fig = px.bar(
                grouped,
                x=primary_category,
                y=primary_metric,
                color=primary_metric,
                title=f"{primary_metric.replace('_', ' ').title()} by {primary_category.replace('_', ' ').title()}",
                template="plotly_dark",
                color_continuous_scale="Sunset",
            )
            bar_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(bar_fig, use_container_width=True)
            top_group = grouped.iloc[0]
            bottom_group = grouped.iloc[-1]
            st.caption(
                f"{top_group[primary_category]} is contributing the most to {primary_metric.replace('_', ' ')}, while {bottom_group[primary_category]} is contributing the least. "
                f"The bar spread shows whether performance is concentrated in a few categories or distributed more evenly across the dataset."
            )
            rendered_any = True

            if grouped[primary_category].nunique() <= 6:
                st.subheader("Share of Total")
                pie_fig = px.pie(
                    grouped,
                    names=primary_category,
                    values=primary_metric,
                    title=f"Share of {primary_metric.replace('_', ' ').title()}",
                    template="plotly_dark",
                )
                pie_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(pie_fig, use_container_width=True)
                share = (top_group[primary_metric] / grouped[primary_metric].sum()) * 100 if grouped[primary_metric].sum() else 0
                st.caption(
                    f"The pie chart shows how much of the total {primary_metric.replace('_', ' ')} is owned by each {primary_category.replace('_', ' ')} group. "
                    f"{top_group[primary_category]} alone accounts for about {share:.1f}% of the total, which indicates whether one segment dominates the mix."
                )
                rendered_any = True

            if len(grouped) >= 3:
                st.subheader("Waterfall Contribution")
                waterfall_fig = go.Figure(
                    go.Waterfall(
                        name=primary_metric,
                        orientation="v",
                        measure=["relative"] * len(grouped),
                        x=grouped[primary_category].astype(str),
                        y=grouped[primary_metric],
                        connector={"line": {"color": "rgba(148,163,184,0.6)"}},
                    )
                )
                waterfall_fig.update_layout(
                    title=f"Stepwise Contribution to {primary_metric.replace('_', ' ').title()}",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=500,
                )
                st.plotly_chart(waterfall_fig, use_container_width=True)
                st.caption(
                    f"The waterfall chart shows how each {primary_category.replace('_', ' ')} group adds to the total {primary_metric.replace('_', ' ')} step by step. "
                    "It highlights whether the total is built steadily across groups or driven by a few large jumps."
                )
                rendered_any = True

            if grouped[primary_category].nunique() <= 12:
                st.subheader("Distribution by Category")
                st.plotly_chart(plot_categorical_breakdown(df, primary_category, primary_metric), use_container_width=True)
                st.caption(
                    f"The box plot compares the spread of {primary_metric.replace('_', ' ')} inside each {primary_category.replace('_', ' ')} group, not just their averages. "
                    "This helps reveal whether a leading category is consistently strong or lifted by only a few unusually large records."
                )
                rendered_any = True

    if len(engine.numeric_cols) >= 2:
        first_metric, second_metric = engine.numeric_cols[:2]
        st.subheader("Metric Relationship")
        scatter_df = df[[first_metric, second_metric]].dropna()
        if not scatter_df.empty:
            scatter_fig = px.scatter(
                scatter_df,
                x=first_metric,
                y=second_metric,
                title=f"{first_metric.replace('_', ' ').title()} vs {second_metric.replace('_', ' ').title()}",
                template="plotly_dark",
            )
            if len(scatter_df) >= 2:
                slope, intercept = np.polyfit(scatter_df[first_metric], scatter_df[second_metric], 1)
                x_sorted = np.sort(scatter_df[first_metric].to_numpy())
                y_line = slope * x_sorted + intercept
                scatter_fig.add_trace(
                    go.Scatter(
                        x=x_sorted,
                        y=y_line,
                        mode="lines",
                        name="Trend",
                        line=dict(color="#f97316", width=2),
                    )
                )
            scatter_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(scatter_fig, use_container_width=True)
            corr = scatter_df[first_metric].corr(scatter_df[second_metric])
            corr_label = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
            direction = "positive" if corr > 0 else "negative"
            st.caption(
                f"The scatter plot suggests a {corr_label} {direction} relationship between {first_metric.replace('_', ' ')} and {second_metric.replace('_', ' ')}. "
                "Points tightly following the fitted line indicate consistent linkage, while wide scatter suggests other factors are driving results."
            )
            rendered_any = True

        st.subheader("Correlation Map")
        st.plotly_chart(plot_correlation_heatmap(df, engine.numeric_cols), use_container_width=True)
        corr_pairs = engine.correlation_insights()
        if corr_pairs:
            strongest = corr_pairs[0]
            st.caption(
                f"The heatmap shows how the numeric measures move together across the dataset. "
                f"The clearest relationship here is between {strongest['var1']} and {strongest['var2']} at {strongest['correlation']:.2f}, which stands out as the most meaningful shared pattern."
            )
        else:
            st.caption(
                "The heatmap does not show many strong pairwise relationships, which suggests the numeric measures are moving relatively independently."
            )
        rendered_any = True

    if primary_metric:
        st.subheader("Overall Distribution")
        st.plotly_chart(plot_distribution_analysis(df, primary_metric, engine), use_container_width=True)
        st.caption(build_distribution_commentary(df, primary_metric, engine))
        rendered_any = True

    if not rendered_any:
        st.info("This dataset does not yet have enough numeric, categorical, or time-based structure to generate a richer visual story.")


def run_intelligent_analysis(tables):
    dataset_profile = build_dataset_profile(tables)
    cleaned_tables, cleaning_reports = clean_tables(tables, dataset_profile.tables)
    cleaned_profile = build_dataset_profile(cleaned_tables)
    model = build_analysis_model(cleaned_tables, cleaned_profile)
    engine = InsightEngine(model)
    analysis_df = model.integrated_df.copy()

    with st.sidebar:
        st.markdown("### Navigation")
        selected_page = st.radio(
            "Go To",
            ["Executive Summary", "Data Cleaning", "Metric Analysis", "Visualizations", "Actionable Insights"],
            label_visibility="collapsed",
        )
        st.markdown("### Dataset Profile")
        st.write(f"**Domain:** {cleaned_profile.domain.replace('_', ' ').title()}")
        if cleaned_profile.domain_candidates:
            top_candidate = cleaned_profile.domain_candidates[0]
            st.write(f"**Confidence:** {top_candidate['confidence']:.1f}%")
        st.write(f"**Tables:** {len(cleaned_tables)}")
        st.write(f"**Primary Table:** {model.primary_table}")
        st.write(f"**Integrated Shape:** {analysis_df.shape[0]:,} rows x {analysis_df.shape[1]} cols")

    if selected_page == "Executive Summary":
        st.subheader("Key Metrics")
        key_metrics = engine.key_metrics_summary()
        if key_metrics:
            cols = st.columns(len(key_metrics))
            for idx, metric in enumerate(key_metrics):
                cols[idx].metric(metric["label"], metric["value"], metric["delta"] or None)
        elif engine.numeric_cols:
            create_kpi_cards(analysis_df, engine.numeric_cols)

        st.subheader("Data Storytelling")
        render_insight_cards(engine.data_storytelling())

        st.subheader("What the system understood")
        render_insight_cards(engine.overview_insights())
        summary_cols = st.columns(4)
        summary_cols[0].metric("Tables Loaded", len(cleaned_tables))
        summary_cols[1].metric("Relationships", len(cleaned_profile.relationships))
        summary_cols[2].metric("Measures", len(engine.numeric_cols))
        summary_cols[3].metric("Dimensions", len(engine.cat_cols))
        if cleaned_profile.warnings:
            for warning in cleaned_profile.warnings:
                st.warning(warning)
        st.subheader("Observed Patterns and Trends")
        render_insight_cards(engine.pattern_overview())

    elif selected_page == "Data Cleaning":
        st.subheader("Data Cleaning Overview")
        cleaning_summary = summarize_cleaning_reports(cleaning_reports)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tables Reviewed", cleaning_summary["tables"])
        c2.metric("Cleaning Steps Applied", cleaning_summary["actions"])
        c3.metric("Logical Rules Triggered", cleaning_summary["rules"])
        c4.metric("Items to Review", cleaning_summary["warnings"])
        st.caption(
            "This page shows what was automatically standardized, corrected, or flagged before analysis. "
            "Use it to confirm the data now reflects the business meaning you expect."
        )

        for report in cleaning_reports:
            preview_table = next(table.df for table in cleaned_tables if table.name == report.table_name)
            rows_before, cols_before = report.original_shape
            rows_after, cols_after = report.final_shape

            with st.container(border=True):
                st.markdown(f"### {prettify_label(report.table_name)}")
                top_cols = st.columns(4)
                top_cols[0].metric("Rows Before", f"{rows_before:,}")
                top_cols[1].metric("Rows After", f"{rows_after:,}")
                top_cols[2].metric("Columns", f"{cols_after:,}")
                top_cols[3].metric("Rule Checks Fired", len(report.logical_rules))

                if report.actions:
                    st.success(f"{len(report.actions)} cleaning updates were applied to this table.")
                else:
                    st.info("No direct cleaning changes were needed for this table.")

                if report.warnings:
                    st.warning(f"{len(report.warnings)} item(s) still need review in this table.")

                section_left, section_right = st.columns([1.4, 1])

                with section_left:
                    st.markdown("#### What Changed")
                    if report.actions:
                        for action in report.actions:
                            st.write(f"- {action}")
                    else:
                        st.write("- No cleaning steps were needed.")

                    st.markdown("#### Data Preview")
                    st.dataframe(preview_table.head(15), use_container_width=True)

                with section_right:
                    st.markdown("#### Business Rule Checks")
                    if report.logical_rules:
                        rules_df = pd.DataFrame(report.logical_rules).rename(
                            columns={"rule": "Rule", "columns": "Columns Checked", "impact": "What Changed"}
                        )
                        st.dataframe(rules_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No logical consistency rules were triggered for this table.")

                    st.markdown("#### Review Notes")
                    if report.warnings:
                        for warning in report.warnings:
                            st.write(f"- {warning}")
                    else:
                        st.write("- No remaining review warnings for this table.")

    elif selected_page == "Metric Analysis":
        if not engine.numeric_cols:
            st.info("No numeric measures available for detailed statistical analysis.")
        else:
            metric = st.selectbox("Select metric", engine.numeric_cols)
            takeaway, business_impact = build_metric_takeaway(analysis_df, metric, engine)
            st.subheader("Main Takeaway")
            st.write(takeaway)
            st.info(f"Why this matters: {business_impact}")

            c1, c2 = st.columns([2, 1])
            with c1:
                st.plotly_chart(plot_distribution_analysis(analysis_df, metric, engine), use_container_width=True)
                st.caption(build_distribution_commentary(analysis_df, metric, engine))
            with c2:
                stats_data = analysis_df[metric].describe()
                st.write(f"**Average value:** {stats_data['mean']:,.2f}")
                st.write(f"**Typical value:** {analysis_df[metric].median():,.2f}")
                st.write(f"**Middle range:** {analysis_df[metric].quantile(0.25):,.2f} to {analysis_df[metric].quantile(0.75):,.2f}")
                st.write(f"**Records analysed:** {len(analysis_df[metric].dropna()):,}")
                if len(analysis_df[metric].dropna()) >= 8:
                    _, p_value = stats.normaltest(analysis_df[metric].dropna())
                    if p_value < 0.05:
                        st.warning("Values are unevenly spread rather than following a smooth bell-shaped pattern.")
                    else:
                        st.success("Values are fairly balanced around the middle of the distribution.")
                anomalies = engine.detect_anomalies(analysis_df[metric])
                if len(anomalies):
                    st.error(f"{len(anomalies)} unusual records detected")
                    st.dataframe(anomalies.to_frame(name=metric), use_container_width=True)

    elif selected_page == "Visualizations":
        st.subheader("Visual Story of the Current Dataset")
        render_visual_story(analysis_df, engine)

    elif selected_page == "Actionable Insights":
        st.subheader("Business Actions From the Current Data")
        render_actionable_business_insights(engine.business_actionable_insights())


st.markdown('<div class="main-header">AutoInsight Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Turn raw data into clear analysis, polished visuals, and decision-ready insights in minutes.</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Data Source")
data_source = st.sidebar.radio("Choose Data Source", ["Upload Files", "Use Sample Data"])
tables = []

if data_source == "Upload Files":
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV or Excel files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        tables = load_uploaded_tables(uploaded_files)
        if tables:
            st.sidebar.success(f"Loaded {len(tables)} table(s) from {len(uploaded_files)} file(s).")
        else:
            st.sidebar.error("No readable tables were found in the uploaded files.")
else:
    sample = st.sidebar.selectbox("Select Sample", ["Sales + Customers", "HR", "Marketing"])
    if st.sidebar.button("Generate Sample Data"):
        tables = sample_tables_to_uploaded_frames(generate_sample_data(sample), sample)
        st.sidebar.success(f"Generated {len(tables)} sample table(s).")

if tables:
    run_intelligent_analysis(tables)
else:
    st.markdown(
        """
        <div class="panel">
            <h3 style="color:#f8fafc;">Built for real analytics work</h3>
            <p style="color:#cbd5e1;">Upload business data, explore patterns fast, and present insights in a way that works for clients and stakeholders.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
