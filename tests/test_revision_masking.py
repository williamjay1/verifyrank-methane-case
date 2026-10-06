"""Synthetic correctness tests for revision diagnostics; no field results."""
import importlib.util
from itertools import combinations
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("revision_masking", ROOT / "scripts/revision_masking.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class RevisionMaskingTests(unittest.TestCase):
    def values(self, y, wa, wb, retained, mechanism="uniform", sa=None, sb=None):
        sa = np.asarray(wa) if sa is None else sa
        sb = np.asarray(wb) if sb is None else sb
        rows, bound = MOD.evaluate_pair(y, sa, sb, wa, wb, retained, mechanism)
        return {r["method"]: r for r in rows}, bound

    def test_ht_enumeration_unbiased_with_fractional_ties(self):
        y = np.array([1., 0., 1., 0., 1.])
        wa, wb = np.array([1., .5, .5, 0., 0.]), np.array([0., 0., .5, .5, 1.])
        for k in (1, 2, 3, 4, 5):
            values = []
            for chosen in combinations(range(5), k):
                kept = np.zeros(5, bool)
                kept[list(chosen)] = True
                values.append(MOD.ht_difference(y, wa, wb, kept, k / 5))
            self.assertAlmostEqual(float(np.mean(values)), float(np.dot((wa-wb)/2, y)), places=13)

    def test_ht_actual_ceil_probability_not_nominal_fraction(self):
        rows, _ = self.values([1, 0, 0], [1, 0, 0], [0, 1, 0], [True, False, True])
        self.assertAlmostEqual(rows["uniform_ht"]["uniform_inclusion_probability"], 2/3)
        self.assertAlmostEqual(rows["uniform_ht"]["estimate"], 1.5)

    def test_ht_not_applied_to_history_or_group_weights(self):
        for mechanism in ("history_weighted", *MOD.GROUP_MECHANISMS):
            rows, _ = self.values([1, 0], [1, 0], [0, 1], [True, False], mechanism)
            self.assertTrue(np.isnan(rows["uniform_ht"]["estimate"]))
            self.assertEqual(rows["uniform_ht"]["undefined_reason"], "not_uniform_fixed_size_row_design")

    def test_frozen_conditional_uses_policy_specific_verified_mass(self):
        rows, _ = self.values([1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 1, 0], [True, True, False, False])
        row = rows["frozen_verified_conditional"]
        self.assertEqual(row["verified_selected_mass_a"], 2)
        self.assertEqual(row["verified_selected_mass_b"], 1)
        self.assertAlmostEqual(row["estimate"], .5)

    def test_frozen_zero_mass_is_na_not_zero(self):
        rows, bound = self.values([1, 0, 1], [1, 0, 0], [0, 1, 0], [True, False, True])
        self.assertFalse(rows["frozen_verified_conditional"]["defined"])
        self.assertTrue(np.isnan(rows["frozen_verified_conditional"]["estimate"]))
        self.assertTrue(bound["contains_truth"])

    def test_original_fraction_uses_retained_population(self):
        rows, _ = self.values([1, 0, 1, 0, 1, 0, 1, 0], [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0], [True]*4+[False]*4)
        self.assertEqual(rows["rerank_original_count"]["retained_count_budget"], 2)
        self.assertEqual(rows["rerank_original_fraction"]["retained_fraction_budget"], 1)

    def test_quarter_retention_original_count_structural_tie(self):
        rows, _ = self.values([1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0], [True, False, True, False, False, False, False, False])
        self.assertEqual(rows["rerank_original_count"]["estimate"], 0)
        self.assertTrue(rows["rerank_original_count"]["estimated_tie"])

    def test_group_zero_retention_day_preserves_bounds_and_missing_methods(self):
        rows, bound = self.values([1, 0], [1, 0], [0, 1], [False, False], "observation_day_group")
        self.assertTrue(np.isnan(rows["rerank_original_count"]["estimate"]))
        self.assertTrue(np.isnan(rows["rerank_original_fraction"]["estimate"]))
        self.assertTrue(np.isnan(rows["frozen_verified_conditional"]["estimate"]))
        self.assertEqual(rows["zero_filled"]["estimate"], 0)
        self.assertEqual(bound["paired_lower"], -1)
        self.assertEqual(bound["paired_upper"], 1)

    def test_shared_reference_mask_consistent_across_decisions(self):
        keys = ["24|a|s", "24|a|s", "24|b|s", "24|c|t", "24|c|t"]
        first, groups = MOD.group_mask(keys, .5, np.random.default_rng(4))
        second, groups2 = MOD.group_mask(keys, .5, np.random.default_rng(4))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(groups, groups2)
        self.assertEqual(first[0], first[1])
        self.assertEqual(first[3], first[4])
        self.assertEqual(len(groups), 2)

    def test_reference_key_includes_source_not_only_scene(self):
        data = pd.DataFrame({"campaign": ["2024"]*3, "source_id": ["a", "b", "a"],
            "outcome_scene": ["s"]*3, "outcome_label": [1, 0, 1], "outcome_observed_at": ["2024-05-01T12:00:00Z"]*3})
        keyed = MOD.add_reference_keys(data)
        self.assertEqual(keyed.reference_key.nunique(), 2)
        self.assertEqual(keyed.scene_key.nunique(), 1)
        self.assertEqual(keyed.observation_day_key.nunique(), 1)

    def test_shared_reference_inconsistent_outcome_rejected(self):
        data = pd.DataFrame({"campaign": ["2024"]*2, "source_id": ["a"]*2, "outcome_scene": ["s"]*2,
            "outcome_label": [1, 0], "outcome_observed_at": ["2024-05-01T12:00:00Z"]*2})
        with self.assertRaises(ValueError):
            MOD.add_reference_keys(data)

    def test_magnitude_bins_are_left_closed(self):
        for x, expected in ((0, "[0,0.02)"), (.02, "[0.02,0.05)"), (-.05, "[0.05,0.10)"), (.1, "[0.10,0.20)"), (.2, "[0.20,inf)")):
            self.assertEqual(MOD.magnitude_bin(x), expected)

    def test_all_binary_truths_and_empty_masks_contained(self):
        wa, wb = np.array([1, .5, .5, 0]), np.array([.5, .5, .5, .5])
        for ycode in range(16):
            y = [(ycode >> i) & 1 for i in range(4)]
            for maskcode in range(16):
                rows, bound = self.values(y, wa, wb, [(maskcode >> i) & 1 for i in range(4)], "scene_group")
                self.assertTrue(bound["contains_truth"])
                self.assertLessEqual(bound["paired_width"], bound["independent_width"] + 1e-12)

    def test_frozen_conditional_shared_weights_cancel_when_defined(self):
        rows, bound = self.values([1, 0, 1, 0], [.5]*4, [.5]*4, [False, True, False, False])
        self.assertEqual(rows["frozen_verified_conditional"]["estimate"], 0)
        self.assertEqual(bound["paired_width"], 0)

    def test_summary_retains_undefined_and_common_denominators(self):
        records = []
        for repetition, kept in enumerate(([True, False, True], [True, True, False])):
            rows, _ = MOD.evaluate_pair([1, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], kept, "uniform")
            for row in rows:
                records.append({**row, "pair": "a_vs_b", "mechanism": "uniform", "nominal_retention": .5,
                    "decision_at": "2024-05-01", "repetition": repetition, "actual_row_retention": 2/3})
        summary = MOD.summarize_methods(pd.DataFrame(records))
        count = summary[summary.method.eq("rerank_original_count")]
        self.assertEqual(int(count[count.summary_population.eq("method_defined")].n_defined.iloc[0]), 2)
        self.assertEqual(int(count[count.summary_population.eq("common_point_defined")].n_defined.iloc[0]), 1)
        frozen = summary[summary.method.eq("frozen_verified_conditional") & summary.summary_population.eq("method_defined")].iloc[0]
        self.assertEqual(int(frozen.n_total_date_masks), 2)
        self.assertEqual(int(frozen.n_defined), 1)
        self.assertEqual(int(frozen.n_method_undefined), 1)
        self.assertEqual(int(frozen.unique_decision_dates_defined), 1)


if __name__ == "__main__":
    unittest.main()
