import json
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classify import label_above_median, locality_medians, price_per_sqft
from src.config import CLF_METADATA_PATH


def flats(*rows):
    """rows of (location, total_sqft, price_in_lakhs)"""
    return pd.DataFrame(list(rows), columns=["location", "total_sqft", "price"])


class PricePerSqftTests(unittest.TestCase):
    def test_converts_lakhs_to_rupees_per_sqft(self):
        # 50 lakh over 1000 sqft = Rs 5,000/sqft
        self.assertAlmostEqual(price_per_sqft(flats(("HSR Layout", 1000, 50)))[0], 5000.0)


class LocalityMediansTests(unittest.TestCase):
    def test_median_is_per_locality(self):
        train = flats(
            ("HSR Layout", 1000, 40), ("HSR Layout", 1000, 50), ("HSR Layout", 1000, 60),
            ("Chandapura", 1000, 10), ("Chandapura", 1000, 20), ("Chandapura", 1000, 30),
        )
        medians, fallback = locality_medians(train)
        self.assertAlmostEqual(medians["HSR Layout"], 5000.0)
        self.assertAlmostEqual(medians["Chandapura"], 2000.0)
        self.assertAlmostEqual(fallback, 3500.0)  # median of all six


class LabelTests(unittest.TestCase):
    def setUp(self):
        self.medians = pd.Series({"HSR Layout": 5000.0, "Chandapura": 2000.0})
        self.fallback = 3500.0

    def test_above_and_below(self):
        df = flats(("HSR Layout", 1000, 60), ("HSR Layout", 1000, 40))
        self.assertEqual(list(label_above_median(df, self.medians, self.fallback)), [1, 0])

    def test_exactly_at_the_median_counts_as_not_above(self):
        df = flats(("HSR Layout", 1000, 50))
        self.assertEqual(list(label_above_median(df, self.medians, self.fallback)), [0])

    def test_unknown_locality_uses_the_fallback(self):
        # 40 lakh / 1000 sqft = 4000/sqft, above the 3500 city fallback.
        df = flats(("Nowhere Nagar", 1000, 40))
        self.assertEqual(list(label_above_median(df, self.medians, self.fallback)), [1])

    def test_test_rows_are_labelled_with_TRAINING_thresholds(self):
        """The leakage guard. A test flat must be judged against the training
        median for its locality, never against the test set's own median."""
        test_rows = flats(
            ("Chandapura", 1000, 25), ("Chandapura", 1000, 26), ("Chandapura", 1000, 27)
        )
        # Judged against the training median (Rs 2,000/sqft) all three are above.
        with_train = list(label_above_median(test_rows, self.medians, self.fallback))
        self.assertEqual(with_train, [1, 1, 1])

        # Judged against their own median (Rs 2,600/sqft) only one would be --
        # which is the wrong, leaky answer this function must not produce.
        own_medians, own_fallback = locality_medians(test_rows)
        with_own = list(label_above_median(test_rows, own_medians, own_fallback))
        self.assertEqual(with_own, [0, 0, 1])
        self.assertNotEqual(with_train, with_own)


class ShippedClassifierTests(unittest.TestCase):
    def setUp(self):
        if not CLF_METADATA_PATH.exists():
            self.skipTest("no classifier metadata -- run `python -m src.classify`")
        self.meta = json.loads(CLF_METADATA_PATH.read_text())

    def test_beats_the_baseline(self):
        metrics = self.meta["test_metrics"]
        best = metrics[self.meta["selected_model"]]
        baseline = metrics["Baseline(most frequent)"]
        self.assertGreater(best["roc_auc"], baseline["roc_auc"])
        self.assertGreater(best["accuracy"], baseline["accuracy"])

    def test_auc_is_better_than_a_coin_flip(self):
        best = self.meta["test_metrics"][self.meta["selected_model"]]
        self.assertGreater(best["roc_auc"], 0.6)

    def test_classes_are_close_to_balanced(self):
        # If this drifts far from 50% the accuracy figures stop being readable.
        self.assertAlmostEqual(self.meta["class_balance"]["train_positive"], 0.5, delta=0.1)

    def test_price_is_not_a_feature(self):
        # The label is derived from price, so price as an input would be circular.
        self.assertNotIn("price", self.meta["features"])


if __name__ == "__main__":
    unittest.main()
