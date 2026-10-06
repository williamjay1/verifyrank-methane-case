"""Synthetic checks for the declared exploratory masking stress test."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluate_masking", ROOT / "scripts/evaluate_masking.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class MaskingTests(unittest.TestCase):
    def test_exact_mask_size_and_repeatability(self):
        for mechanism in MOD.MECHANISMS:
            first = MOD.make_mask(9, .25, mechanism, np.arange(9), np.random.default_rng(1))
            second = MOD.make_mask(9, .25, mechanism, np.arange(9), np.random.default_rng(1))
            self.assertEqual(int(first.sum()), 3)
            np.testing.assert_array_equal(first, second)

    def test_mask_rejects_bad_history(self):
        with self.assertRaises(ValueError):
            MOD.make_mask(3, .5, "history_weighted", [1, np.nan, 3], np.random.default_rng(1))

    def test_frozen_weights_and_conditional_reranking_differ(self):
        y = np.array([1., 0., 0., 1.])
        a, b = np.array([4., 3., 2., 1.]), np.array([1., 2., 4., 3.])
        wa, wb = MOD.EV.tie_selection_weights(a, 1), MOD.EV.tie_selection_weights(b, 1)
        row, c = MOD.evaluate_pair(y, a, b, wa, wb, np.array([False, True, True, True]))
        self.assertEqual(row["true_difference"], 1.)
        self.assertEqual(row["conditional_difference"], 0.)
        self.assertEqual(row["zero_filled_difference"], 0.)
        self.assertEqual(row["paired_lower"], 0.)
        self.assertEqual(row["paired_upper"], 1.)
        np.testing.assert_array_equal(wa, [1, 0, 0, 0])
        np.testing.assert_array_equal(c, [1, 0, -1, 0])

    def test_shared_unknowns_cancel(self):
        y = np.array([1., 0., 1., 0.])
        a = np.ones(4)
        wa = MOD.EV.tie_selection_weights(a, 1)
        row, _ = MOD.evaluate_pair(y, a, a, wa, wa, np.array([True, False, False, False]))
        self.assertEqual(row["paired_width"], 0.)
        self.assertEqual(row["independent_width"], 1.5)
        self.assertFalse(row["paired_zero_excluded"])

    def test_all_binary_truths_contained_with_fractional_ties(self):
        a, b = np.array([3., 2., 2., 1.]), np.zeros(4)
        wa, wb = MOD.EV.tie_selection_weights(a, 2), MOD.EV.tie_selection_weights(b, 2)
        for code in range(16):
            y = np.array([(code >> j) & 1 for j in range(4)], dtype=float)
            for mask_code in range(1, 16):
                retained = np.array([(mask_code >> j) & 1 for j in range(4)], dtype=bool)
                row, _ = MOD.evaluate_pair(y, a, b, wa, wb, retained)
                self.assertTrue(row["paired_contains_truth"])
                self.assertTrue(row["independent_contains_truth"])
                self.assertLessEqual(row["paired_width"], row["independent_width"] + 1e-12)

    def test_reveal_priorities_and_width_reduction(self):
        y = np.array([1., 0., 1., 0.])
        wa, wb = np.array([1., 0., 0., 0.]), np.array([0., 0., 1., 0.])
        kept = np.array([False, True, False, False])
        curves, needed = MOD.verification_diagnostics(y, wa, wb, kept, np.array(["a", "b", "c", "d"]))
        widths = [row["targeted_remaining_width"] for row in curves]
        self.assertTrue(all(x >= z - 1e-12 for x, z in zip(widths, widths[1:])))
        self.assertEqual(widths[0], 2.)
        self.assertEqual(widths[-1], 0.)
        self.assertEqual(needed["needed_to_resolve_interval"], 2)
        self.assertIsNone(needed["needed_for_sign_certification"])
        for row in curves:
            self.assertAlmostEqual(row["random_expected_remaining_width"], 2 * (1 - row["revealed_count"] / 3))
            self.assertGreaterEqual(row["width_reduction_advantage"], -1e-12)
            self.assertTrue(row["contains_truth"])

    def test_full_retention_no_unknown_edge(self):
        y = np.array([1., 0.])
        curves, needed = MOD.verification_diagnostics(y, [1, 0], [0, 1], np.ones(2, bool), ["a", "b"])
        self.assertTrue(all(row["targeted_remaining_width"] == 0 for row in curves))
        self.assertEqual(needed["needed_for_sign_certification"], 0)

    def test_primary_filter_and_duplicate_rejection(self):
        rows = pd.DataFrame({"scenario": [MOD.PRIMARY] * 2, "campaign": [2024] * 2,
            "evaluation_mode": [MOD.MODE] * 2, "decision_at": ["2024-05-01"] * 2,
            "source_id": ["a", "a"], "outcome_label": [1., np.nan], "history_observation_count": [1, 1],
            **{f"score_{model}": [0., 0.] for model in MOD.MODELS}})
        with self.assertRaises(ValueError):
            MOD.select_primary(rows)
        rows.loc[1, "source_id"] = "b"
        selected = MOD.select_primary(rows)
        self.assertEqual(len(selected), 2)
        self.assertTrue(pd.isna(selected.iloc[1].outcome_label))


if __name__ == "__main__":
    unittest.main()
