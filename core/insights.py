from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from core.modeling import AnalysisModel


@dataclass
class InsightRecord:
    title: str
    detail: str
    severity: str = "info"


@dataclass
class ActionableInsight:
    title: str
    direction: str
    what_happened: str
    why_it_happened: str
    action: str


class InsightEngine:
    def __init__(self, model: AnalysisModel):
        self.model = model
        self.df = model.integrated_df.copy()
        fallback_numeric = self.df.select_dtypes(include=[np.number]).columns.tolist()
        fallback_dates = self.df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
        fallback_categories = self.df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        self.numeric_cols = [column for column in model.measures if column in self.df.columns] or fallback_numeric
        self.date_cols = [column for column in model.date_columns if column in self.df.columns] or fallback_dates
        self.cat_cols = [column for column in model.dimensions if column in self.df.columns] or fallback_categories
        self._segmented_df: pd.DataFrame | None = None

    def overview_insights(self) -> list[InsightRecord]:
        insights: list[InsightRecord] = []
        missing_pct = float(self.df.isna().sum().sum() / max(self.df.shape[0] * self.df.shape[1], 1) * 100)
        insights.append(
            InsightRecord(
                title="Data Quality",
                detail=f"{missing_pct:.1f}% of values are missing after cleaning across the integrated model.",
                severity="warning" if missing_pct > 5 else "success",
            )
        )
        insights.append(
            InsightRecord(
                title="Detected Domain",
                detail=f"The dataset is classified as '{self.model.domain.replace('_', ' ')}' using schema and content signals.",
            )
        )
        if getattr(self.model, "domain_candidates", None):
            top_candidates = ", ".join(
                f"{candidate['domain'].replace('_', ' ')} ({candidate['confidence']:.1f}%)"
                for candidate in self.model.domain_candidates[:3]
            )
            insights.append(
                InsightRecord(
                    title="Domain Confidence",
                    detail=f"Top domain candidates: {top_candidates}.",
                )
            )
        if self.model.join_notes:
            insights.append(
                InsightRecord(
                    title="Model Integration",
                    detail=f"{len(self.model.join_notes)} related table merges were applied to the primary table.",
                )
            )
        if len(self.numeric_cols) >= 2:
            insights.append(
                InsightRecord(
                    title="Analytics Readiness",
                    detail=f"{len(self.numeric_cols)} numeric measures and {len(self.cat_cols)} descriptive dimensions are available.",
                )
            )
        return insights

    def key_metrics_summary(self) -> list[dict[str, str]]:
        metrics: list[dict[str, str]] = []
        for column in self.numeric_cols[:4]:
            series = self.df[column].dropna()
            if series.empty:
                continue
            average_value = series.mean()
            delta = 0.0
            if len(series) > 1 and series.iloc[0] != 0:
                delta = float((series.iloc[-1] - series.iloc[0]) / series.iloc[0] * 100)
            metrics.append(
                {
                    "label": column.replace("_", " ").title(),
                    "value": f"{average_value:,.2f}",
                    "delta": f"{delta:.1f}%" if delta else "",
                }
            )
        return metrics

    def data_storytelling(self) -> list[InsightRecord]:
        stories: list[InsightRecord] = []

        if self.date_cols and self.numeric_cols:
            primary_metric = self.numeric_cols[0]
            trend = self.trend_analysis(self.date_cols[0], primary_metric)
            stories.append(
                InsightRecord(
                    title="Trend Story",
                    detail=f"{primary_metric.replace('_', ' ').title()} is {trend['trend'].lower()} over time with {trend['volatility_desc'].lower()} volatility. {trend['recommendation']}",
                )
            )

        strong_pairs = self.correlation_insights()
        if strong_pairs:
            strongest = strong_pairs[0]
            direction = "moves with" if strongest["correlation"] > 0 else "moves opposite to"
            stories.append(
                InsightRecord(
                    title="Relationship Story",
                    detail=f"{strongest['var1']} {direction} {strongest['var2']}. That makes this relationship useful for monitoring early movement in performance.",
                )
            )

        if self.cat_cols and self.numeric_cols:
            category_col = self.cat_cols[0]
            metric_col = self.numeric_cols[0]
            grouped = (
                self.df[[category_col, metric_col]]
                .dropna()
                .groupby(category_col, as_index=False)[metric_col]
                .mean()
                .sort_values(metric_col, ascending=False)
            )
            if not grouped.empty:
                best_group = grouped.iloc[0]
                stories.append(
                    InsightRecord(
                        title="Segment Story",
                        detail=f"{best_group[category_col]} is the leading segment for average {metric_col.replace('_', ' ')}. This is the clearest starting point for understanding what strong performance looks like in this dataset.",
                    )
                )

        if not stories:
            stories.append(
                InsightRecord(
                    title="Storytelling",
                    detail="The dataset can be profiled and cleaned, but it does not yet expose enough structured measures and dimensions for a richer narrative.",
                )
            )

        return stories

    def pattern_overview(self) -> list[InsightRecord]:
        patterns: list[InsightRecord] = []

        if self.numeric_cols:
            variability = {
                column: float(self.df[column].std() / self.df[column].mean())
                for column in self.numeric_cols
                if self.df[column].mean() not in (0, np.nan)
            }
            if variability:
                metric = max(variability, key=variability.get)
                patterns.append(
                    InsightRecord(
                        title="Variation Pattern",
                        detail=f"{metric.replace('_', ' ').title()} shows the highest relative variation in the dataset, which suggests that performance is uneven across records or periods.",
                    )
                )

        if self.date_cols and self.numeric_cols:
            metric = self.numeric_cols[0]
            trend = self.trend_analysis(self.date_cols[0], metric)
            patterns.append(
                InsightRecord(
                    title="Trend Pattern",
                    detail=f"{metric.replace('_', ' ').title()} is currently {trend['trend'].lower()} over time with {trend['volatility_desc'].lower()} volatility.",
                )
            )

        correlations = self.correlation_insights()
        if correlations:
            strongest = correlations[0]
            direction = "positive" if strongest["correlation"] > 0 else "negative"
            patterns.append(
                InsightRecord(
                    title="Relationship Pattern",
                    detail=f"The strongest observed relationship is between {strongest['var1']} and {strongest['var2']}, with a {direction} correlation of {strongest['correlation']:.2f}.",
                )
            )

        if not patterns:
            patterns.append(
                InsightRecord(
                    title="Pattern Overview",
                    detail="The dataset has been profiled successfully, but it does not yet expose enough numeric or time-based structure for stronger trend and pattern summaries.",
                )
            )

        return patterns

    def detect_anomalies(self, series: pd.Series, threshold: float = 3.0) -> pd.Series:
        non_null = series.dropna()
        if non_null.empty or non_null.std() == 0:
            return non_null.iloc[0:0]
        scores = np.abs(stats.zscore(non_null))
        return non_null[scores > threshold]

    def trend_analysis(self, date_col: str, metric_col: str) -> dict[str, float | str]:
        df = self.df[[date_col, metric_col]].dropna().sort_values(date_col)
        if len(df) < 3:
            return {
                "trend": "Insufficient data",
                "slope": 0.0,
                "r_squared": 0.0,
                "p_value": 1.0,
                "volatility": 0.0,
                "volatility_desc": "Unknown",
                "recommendation": "Add more time periods to support trend analysis.",
                "next_value": float(df[metric_col].iloc[-1]) if len(df) else 0.0,
            }

        x = np.arange(len(df))
        y = df[metric_col].values
        slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
        volatility = float(np.std(y) / np.mean(y)) if np.mean(y) else 0.0
        if abs(r_value) < 0.3:
            trend = "Stable"
            recommendation = "Monitor the metric; current movement is not directional enough for intervention."
        elif slope > 0:
            trend = "Growing"
            recommendation = "Scale the drivers behind this metric while monitoring capacity and cost."
        else:
            trend = "Declining"
            recommendation = "Investigate recent changes and isolate the segment or period driving the decline."

        return {
            "trend": trend,
            "slope": float(slope),
            "r_squared": float(r_value**2),
            "p_value": float(p_value),
            "volatility": volatility,
            "volatility_desc": "High" if volatility > 0.3 else "Moderate" if volatility > 0.15 else "Low",
            "recommendation": recommendation,
            "next_value": float(intercept + slope * len(df)),
        }

    def correlation_insights(self) -> list[dict[str, float | str]]:
        if len(self.numeric_cols) < 2:
            return []
        corr = self.df[self.numeric_cols].corr(numeric_only=True)
        pairs: list[dict[str, float | str]] = []
        for i, left in enumerate(corr.columns):
            for right in corr.columns[i + 1:]:
                value = corr.loc[left, right]
                if abs(value) >= 0.5:
                    pairs.append(
                        {
                            "var1": left,
                            "var2": right,
                            "correlation": float(value),
                            "strength": "Strong" if abs(value) >= 0.8 else "Moderate",
                        }
                    )
        return sorted(pairs, key=lambda item: abs(item["correlation"]), reverse=True)[:5]

    def segmentation_analysis(self) -> list[dict[str, object]]:
        if len(self.numeric_cols) < 2 or len(self.df) < 20:
            self._segmented_df = None
            return []
        X = self.df[self.numeric_cols].fillna(self.df[self.numeric_cols].median(numeric_only=True))
        scaled = StandardScaler().fit_transform(X)
        n_clusters = min(4, max(2, len(self.df) // 50))
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(scaled)
        segmented = self.df.copy()
        segmented["segment"] = labels
        self._segmented_df = segmented
        results: list[dict[str, object]] = []
        for label in sorted(segmented["segment"].unique()):
            group = segmented[segmented["segment"] == label]
            dominant_features: list[str] = []
            characteristics: dict[str, dict[str, float]] = {}
            for column in self.numeric_cols:
                overall_std = segmented[column].std()
                z_score = 0.0 if overall_std == 0 else (group[column].mean() - segmented[column].mean()) / overall_std
                characteristics[column] = {"mean": float(group[column].mean()), "vs_overall": float(z_score)}
                if abs(z_score) > 0.5:
                    dominant_features.append(f"{'High' if z_score > 0 else 'Low'} {column}")
            results.append(
                {
                    "id": int(label),
                    "name": " + ".join(dominant_features[:2]) if dominant_features else f"Segment {label + 1}",
                    "size": int(len(group)),
                    "percentage": float(len(group) / len(segmented) * 100),
                    "characteristics": characteristics,
                }
            )
        return results

    def segmented_dataframe(self) -> pd.DataFrame:
        if self._segmented_df is None:
            self.segmentation_analysis()
        return self._segmented_df.copy() if self._segmented_df is not None else self.df.copy()

    def _root_cause_hint(self, metric: str) -> str:
        causes: list[str] = []

        if self.date_cols:
            trend = self.trend_analysis(self.date_cols[0], metric)
            if trend["trend"] != "Stable":
                causes.append(f"time trend is {trend['trend'].lower()}")

        correlations = [pair for pair in self.correlation_insights() if pair["var1"] == metric or pair["var2"] == metric]
        if correlations:
            strongest = correlations[0]
            other_metric = strongest["var2"] if strongest["var1"] == metric else strongest["var1"]
            relation = "moves with" if strongest["correlation"] > 0 else "moves opposite to"
            causes.append(f"{metric.replace('_', ' ')} {relation} {other_metric.replace('_', ' ')}")

        if self.cat_cols:
            category = self.cat_cols[0]
            grouped = (
                self.df[[category, metric]]
                .dropna()
                .groupby(category, as_index=False)[metric]
                .mean()
                .sort_values(metric, ascending=False)
            )
            if len(grouped) >= 2:
                top_group = grouped.iloc[0][category]
                bottom_group = grouped.iloc[-1][category]
                causes.append(f"performance differs sharply between {top_group} and {bottom_group}")

        anomalies = self.detect_anomalies(self.df[metric])
        if len(anomalies):
            causes.append(f"{len(anomalies)} unusual records are stretching the distribution")

        return "; ".join(causes[:3]) if causes else "no single dominant driver stands out, so the pattern is likely being shaped by multiple smaller factors"

    def business_actionable_insights(self) -> list[ActionableInsight]:
        insights: list[ActionableInsight] = []

        if self.numeric_cols:
            variability = {
                column: float(self.df[column].std() / self.df[column].mean())
                for column in self.numeric_cols
                if self.df[column].mean() not in (0, np.nan)
            }
            if variability:
                volatile_metric = max(variability, key=variability.get)
                volatile_value = variability[volatile_metric] * 100
                insights.append(
                    ActionableInsight(
                        title="Largest Risk Area",
                        direction="bad",
                        what_happened=f"{volatile_metric.replace('_', ' ').title()} is the least stable measure in the dataset, with relative volatility around {volatile_value:.1f}%.",
                        why_it_happened=self._root_cause_hint(volatile_metric).capitalize() + ".",
                        action=f"Prioritize a root-cause review for {volatile_metric.replace('_', ' ')} and break it down by time, segment, and unusual records before making operating decisions from the average alone.",
                    )
                )

        if self.date_cols and self.numeric_cols:
            trend_metric = self.numeric_cols[0]
            trend = self.trend_analysis(self.date_cols[0], trend_metric)
            if trend["trend"] == "Growing":
                metric_df = self.df[[self.date_cols[0], trend_metric]].dropna().sort_values(self.date_cols[0])
                start_value = metric_df[trend_metric].iloc[0]
                end_value = metric_df[trend_metric].iloc[-1]
                insights.append(
                    ActionableInsight(
                        title="Growth Signal",
                        direction="good",
                        what_happened=f"{trend_metric.replace('_', ' ').title()} is trending upward over time, moving from {start_value:,.2f} to {end_value:,.2f}.",
                        why_it_happened=self._root_cause_hint(trend_metric).capitalize() + ".",
                        action=f"Protect and scale the conditions behind this movement, and monitor whether growth is broad-based or concentrated in a small set of periods or segments.",
                    )
                )
            elif trend["trend"] == "Declining":
                metric_df = self.df[[self.date_cols[0], trend_metric]].dropna().sort_values(self.date_cols[0])
                start_value = metric_df[trend_metric].iloc[0]
                end_value = metric_df[trend_metric].iloc[-1]
                insights.append(
                    ActionableInsight(
                        title="Decline Signal",
                        direction="bad",
                        what_happened=f"{trend_metric.replace('_', ' ').title()} is trending downward over time, moving from {start_value:,.2f} to {end_value:,.2f}.",
                        why_it_happened=self._root_cause_hint(trend_metric).capitalize() + ".",
                        action=f"Investigate when the decline started, identify the segments contributing most to the drop, and intervene before the weaker periods become the new baseline.",
                    )
                )

        if self.cat_cols and self.numeric_cols:
            category = self.cat_cols[0]
            metric = self.numeric_cols[0]
            grouped = (
                self.df[[category, metric]]
                .dropna()
                .groupby(category, as_index=False)[metric]
                .mean()
                .sort_values(metric, ascending=False)
            )
            if len(grouped) >= 2:
                top_group = grouped.iloc[0]
                bottom_group = grouped.iloc[-1]
                gap_pct = ((top_group[metric] - bottom_group[metric]) / max(abs(bottom_group[metric]), 1)) * 100
                insights.append(
                    ActionableInsight(
                        title="Best Performing Segment",
                        direction="good",
                        what_happened=f"{top_group[category]} is leading on average {metric.replace('_', ' ')}, outperforming {bottom_group[category]} by about {gap_pct:.1f}%.",
                        why_it_happened=f"This usually indicates meaningful differences in mix, quality, pricing, audience, operations, or execution across {category.replace('_', ' ')} groups.",
                        action=f"Study what is working in {top_group[category]} and apply those practices to weaker groups, while checking whether the gap is structural or operational.",
                    )
                )

        corr_pairs = self.correlation_insights()
        if corr_pairs:
            strongest = corr_pairs[0]
            other_metric = strongest["var2"]
            base_metric = strongest["var1"]
            direction = "positive" if strongest["correlation"] > 0 else "negative"
            insights.append(
                ActionableInsight(
                    title="Key Driver Relationship",
                    direction="good" if strongest["correlation"] > 0 else "bad",
                    what_happened=f"{base_metric.replace('_', ' ').title()} and {other_metric.replace('_', ' ')} show a strong {direction} relationship.",
                    why_it_happened=f"The two metrics are moving closely enough together that one likely reflects operational or commercial conditions affecting the other.",
                    action=f"Use {other_metric.replace('_', ' ')} as a monitoring signal when managing {base_metric.replace('_', ' ')}, and validate whether changes in one are leading changes in the other.",
                )
            )

        if not insights:
            insights.append(
                ActionableInsight(
                    title="Limited Signals",
                    direction="neutral",
                    what_happened="The dataset does not yet expose enough strong numeric, time, or segment patterns for confident business actions.",
                    why_it_happened="The current structure is likely too sparse, too small, or too categorical to isolate strong drivers.",
                    action="Add more detailed measures, consistent time fields, or segment-level attributes to make root-cause analysis more reliable.",
                )
            )

        return insights[:5]

    def actionable_insights(self) -> list[InsightRecord]:
        findings: list[InsightRecord] = []

        if self.numeric_cols:
            volatility = {
                column: float(self.df[column].std() / self.df[column].mean())
                for column in self.numeric_cols
                if self.df[column].mean() not in (0, np.nan)
            }
            if volatility:
                metric = max(volatility, key=volatility.get)
                findings.append(
                    InsightRecord(
                        title="Highest Volatility Metric",
                        detail=f"'{metric}' has the highest relative volatility, so it is the best candidate for root-cause analysis and alerting.",
                    )
                )

        for pair in self.correlation_insights()[:2]:
            direction = "positively" if pair["correlation"] > 0 else "negatively"
            findings.append(
                InsightRecord(
                    title="Strong Relationship",
                    detail=f"{pair['var1']} and {pair['var2']} move {direction} together ({pair['correlation']:.2f}); use one as a leading indicator for the other.",
                )
            )

        domain_recommendations = {
            "hr": "Review attrition, pay equity, and performance differences across departments and tenure bands.",
            "sales": "Concentrate on the customer, product, or region segments contributing the largest share of revenue variance.",
            "marketing": "Shift budget toward channels with stronger conversion efficiency and lower spend volatility.",
            "finance": "Track unusual movements in expense and cash-related metrics, then validate whether they align with seasonality.",
            "real_estate": "Compare occupancy, rent, and property-type performance to identify underperforming assets.",
            "podcast": "Identify which episode formats and publishing cadences correlate with higher audience retention.",
            "web_analytics": "Focus on acquisition sources or pages with strong traffic but weak downstream engagement.",
            "shipping": "Prioritize routes or carriers driving delivery variability and missed SLA patterns.",
            "saas": "Use retention and activation metrics as the first layer for diagnosing revenue movement.",
        }
        findings.append(
            InsightRecord(
                title="Domain Playbook",
                detail=domain_recommendations.get(
                    self.model.domain,
                    "Start with the most volatile metrics, strongest relationships, and highest-value segments to prioritize actions.",
                ),
            )
        )

        return findings
