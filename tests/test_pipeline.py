import unittest

import pandas as pd

from core.cleaning import clean_table
from core.ingestion import LoadedTable, normalize_dataframe
from core.profiling import build_dataset_profile, profile_table


class PipelineTests(unittest.TestCase):
    def test_domain_detection_returns_ranked_candidates(self):
        df = pd.DataFrame(
            {
                "viewer_id": [1, 2, 3],
                "minutes_watched": [45, 30, 55],
                "subscription_plan": ["Premium", "Basic", "Premium"],
                "genre": ["Drama", "Comedy", "Drama"],
                "device_type": ["TV", "Mobile", "Tablet"],
            }
        )
        table = LoadedTable(name="streaming", source_name="streaming.csv", df=normalize_dataframe(df))

        profile = build_dataset_profile([table])

        self.assertEqual(profile.domain, "streaming_service")
        self.assertTrue(profile.domain_candidates)
        self.assertEqual(profile.domain_candidates[0]["domain"], "streaming_service")

    def test_metric_selection_excludes_id_like_numeric_columns(self):
        df = pd.DataFrame(
            {
                "property_id": [1001, 1002, 1003],
                "revenue": [1200, 1800, 1600],
                "occupied_units": [8, 9, 10],
                "property_type": ["Residential", "Commercial", "Residential"],
            }
        )
        table = LoadedTable(name="real_estate", source_name="real_estate.csv", df=normalize_dataframe(df))

        profile = build_dataset_profile([table])

        self.assertIn("revenue", profile.measures)
        self.assertNotIn("property_id", profile.measures)

    def test_cleaning_standardizes_categories_and_applies_logical_rules(self):
        df = pd.DataFrame(
            {
                "property_type": ["Residential", "residential", "Commercial"],
                "occupied_units": [12, 8, 7],
                "total_units": [10, 8, 7],
                "property_value": [500000, -1000, 700000],
                "loan_amount": [600000, 200000, 300000],
            }
        )
        table = LoadedTable(name="properties", source_name="properties.csv", df=normalize_dataframe(df))
        table_profile = profile_table(table)

        cleaned_table, report = clean_table(table, table_profile)
        cleaned_df = cleaned_table.df

        self.assertEqual(sorted(cleaned_df["property_type"].dropna().unique().tolist()), ["Commercial", "Residential"])
        self.assertEqual(cleaned_df.loc[0, "occupied_units"], 10)
        self.assertTrue(pd.isna(cleaned_df.loc[1, "property_value"]))
        self.assertEqual(cleaned_df.loc[0, "loan_amount"], 500000)
        self.assertTrue(report.logical_rules)


if __name__ == "__main__":
    unittest.main()
