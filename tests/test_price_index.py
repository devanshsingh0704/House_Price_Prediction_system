import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.price_index import CITY_KEY, inflation_factor, project, project_band

MARKET = {CITY_KEY: 12100.0, "Electronic City Phase II": 7750.0}
DATASET = {"city": 5091.1, "by_location": {"Electronic City Phase II": 2968.0, "Hebbal": 6000.0}}


class InflationFactorTests(unittest.TestCase):
    def test_uses_the_locality_rate_when_both_ends_are_known(self):
        factor, basis = inflation_factor("Electronic City Phase II", MARKET, DATASET)
        self.assertAlmostEqual(factor, 7750.0 / 2968.0, places=6)
        self.assertEqual(basis, "Electronic City Phase II")

    def test_falls_back_to_the_city_average_and_says_so(self):
        factor, basis = inflation_factor("Hebbal", MARKET, DATASET)
        self.assertAlmostEqual(factor, 12100.0 / 5091.1, places=6)
        self.assertIn("city average", basis)

    def test_unknown_locality_also_falls_back(self):
        _, basis = inflation_factor("Other", MARKET, DATASET)
        self.assertIn("city average", basis)


class ProjectionTests(unittest.TestCase):
    def test_compounds(self):
        self.assertAlmostEqual(project(100.0, 4, 0.10), 146.41, places=2)

    def test_zero_years_is_a_no_op(self):
        self.assertAlmostEqual(project(100.0, 0, 0.10), 100.0)

    def test_band_is_ordered_low_to_high(self):
        low, high = project_band(70.0, 4, cagr=(0.10, 0.08))
        self.assertLess(low, high)
        self.assertAlmostEqual(low, 70 * 1.08 ** 4, places=6)
        self.assertAlmostEqual(high, 70 * 1.10 ** 4, places=6)


class ShippedRatesTests(unittest.TestCase):
    def test_the_csv_parses_and_has_the_city_fallback_row(self):
        from src.price_index import load_market_rates

        rates = load_market_rates()
        self.assertIn(CITY_KEY, rates)
        self.assertTrue(all(v > 0 for v in rates.values()))


class ShippedIntervalTests(unittest.TestCase):
    """The interval in metadata.json is what predict.py turns into a range."""

    def setUp(self):
        import json

        from src.config import METADATA_PATH

        if not METADATA_PATH.exists():
            self.skipTest("no metadata.json -- run `python -m src.train`")
        self.interval = json.loads(METADATA_PATH.read_text())["interval"]

    def test_brackets_the_point_estimate(self):
        self.assertLess(self.interval["lo"], 1.0)
        self.assertGreater(self.interval["hi"], 1.0)

    def test_calibration_lands_near_its_target(self):
        # Calibrated out-of-fold on training rows, then checked against the test
        # set. If these drift apart, the interval is lying about its coverage.
        self.assertAlmostEqual(
            self.interval["observed_coverage_on_test"],
            self.interval["coverage"],
            delta=0.05,
        )


if __name__ == "__main__":
    unittest.main()
