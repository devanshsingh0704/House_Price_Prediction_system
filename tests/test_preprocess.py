import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocess import (
    clean_dataframe,
    drop_pps_outliers,
    extract_bhk,
    group_rare_locations,
    normalize_location,
    parse_sqft,
)


class ExtractBhkTests(unittest.TestCase):
    def test_parses_bhk_and_bedroom_labels(self):
        self.assertEqual(extract_bhk("2 BHK"), 2)
        self.assertEqual(extract_bhk("3 Bedroom"), 3)
        self.assertEqual(extract_bhk("1 BHK"), 1)

    def test_rejects_rk_and_empty(self):
        self.assertIsNone(extract_bhk("1 RK"))
        self.assertIsNone(extract_bhk(""))
        self.assertIsNone(extract_bhk(None))


class ParseSqftTests(unittest.TestCase):
    def test_plain_number(self):
        self.assertEqual(parse_sqft("1056"), 1056.0)

    def test_range_uses_average(self):
        self.assertEqual(parse_sqft("2100 - 2850"), 2475.0)

    def test_converts_sq_meter_and_yards(self):
        self.assertAlmostEqual(parse_sqft("1000Sq. Meter"), 1000 * 10.7639, places=2)
        self.assertEqual(parse_sqft("1100Sq. Yards"), 9900.0)

    def test_drops_land_units(self):
        self.assertIsNone(parse_sqft("5.31Acres"))
        self.assertIsNone(parse_sqft("3Cents"))
        self.assertIsNone(parse_sqft("24Guntha"))
        self.assertIsNone(parse_sqft("4125Perch"))


class NormalizeLocationTests(unittest.TestCase):
    def test_groups_hsr_and_electronic_city_variants(self):
        self.assertEqual(normalize_location("Sector 7 HSR Layout"), "HSR Layout")
        self.assertEqual(normalize_location("Electronics City Phase 1"), "Electronic City Phase I")
        self.assertEqual(normalize_location("Electronic City Phase II"), "Electronic City Phase II")
        self.assertEqual(normalize_location("Electronic city phase 1,"), "Electronic City Phase I")

    def test_does_not_merge_doddabommasandra_into_bommasandra(self):
        self.assertEqual(normalize_location("Bommasandra"), "Bommasandra")
        self.assertEqual(
            normalize_location("Bommasandra Industrial Area"),
            "Bommasandra Industrial Area",
        )
        self.assertEqual(normalize_location("Doddabommasandra"), "Doddabommasandra")

    def test_normalizes_common_search_areas(self):
        self.assertEqual(normalize_location("Indira Nagar"), "Indira Nagar")
        self.assertEqual(normalize_location("1st Stage Indira Nagar"), "Indira Nagar")
        self.assertEqual(normalize_location("8th block Koramangala"), "Koramangala")
        self.assertEqual(normalize_location("BTM 2nd Stage"), "BTM Layout")
        self.assertEqual(normalize_location("Whitefield,"), "Whitefield")
        self.assertEqual(normalize_location("Hebbal Kempapura"), "Hebbal")
        self.assertEqual(normalize_location("Yelahanka New Town"), "Yelahanka")

    def test_unknown_location_is_left_alone_for_the_rare_bucket_to_handle(self):
        # normalize_location only fixes spelling; deciding what becomes "Other" is
        # group_rare_locations' job, and predict.py honours the same vocabulary.
        self.assertEqual(normalize_location("Kanakpura Road"), "Kanakpura Road")


class GroupRareLocationsTests(unittest.TestCase):
    def test_keeps_frequent_names_and_buckets_rare(self):
        import pandas as pd

        series = pd.Series(["HSR Layout"] * 10 + ["Tiny Place", "Tiny Place"])
        grouped = group_rare_locations(series, min_count=10)
        self.assertEqual(grouped.value_counts().to_dict(), {"HSR Layout": 10, "Other": 2})


class CleanDataframeTests(unittest.TestCase):
    def test_keeps_1_to_3_bhk_and_drops_rk(self):
        import pandas as pd

        rows = []
        for _ in range(10):
            rows.append(["Super built-up Area", "HSR Layout", "2 BHK", "1200", 2, 1, 80])
            rows.append(["Super built-up Area", "Bommasandra", "3 Bedroom", "1500", 3, 2, 55])
        rows.append(["Super built-up Area", "Electronic City", "1 RK", "400", 1, 0, 20])
        rows.append(["Super built-up Area", "Whitefield", "4 BHK", "2200", 4, 2, 140])
        raw = pd.DataFrame(
            rows,
            columns=["area_type", "location", "size", "total_sqft", "bath", "balcony", "price"],
        )
        raw["availability"] = "Ready To Move"
        raw["society"] = ""
        cleaned = clean_dataframe(raw)
        self.assertEqual(set(cleaned["bhk"]), {2, 3})
        self.assertTrue(set(cleaned["location"]).issubset({"HSR Layout", "Bommasandra"}))

    def test_leaves_missing_bath_and_outliers_for_the_training_split(self):
        import pandas as pd

        rows = [["Super built-up Area", "HSR Layout", "2 BHK", "1200", None, None, 80]]
        rows += [["Super built-up Area", "HSR Layout", "2 BHK", "1200", 2, 1, 80]] * 9
        raw = pd.DataFrame(
            rows,
            columns=["area_type", "location", "size", "total_sqft", "bath", "balcony", "price"],
        )
        cleaned = clean_dataframe(raw)
        # Imputation belongs to the pipeline in train.py, not here -- so the NaN survives.
        self.assertEqual(cleaned["bath"].isna().sum(), 1)
        self.assertEqual(len(cleaned), 10)


class DropPpsOutliersTests(unittest.TestCase):
    def _frame(self):
        import pandas as pd

        # Nine listings at ~5000/sqft in one locality, plus one absurd one.
        rows = [["HSR Layout", 1000.0, 50.0]] * 9 + [["HSR Layout", 1000.0, 350.0]]
        return pd.DataFrame(rows, columns=["location", "total_sqft", "price"])

    def test_wide_trim_keeps_more_rows_than_narrow_trim(self):
        df = self._frame()
        self.assertGreaterEqual(len(drop_pps_outliers(df, n_std=3)), len(drop_pps_outliers(df, n_std=1)))

    def test_drops_the_absurd_listing(self):
        kept = drop_pps_outliers(self._frame(), n_std=1)
        self.assertNotIn(350.0, set(kept["price"]))


if __name__ == "__main__":
    unittest.main()
