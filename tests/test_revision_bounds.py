import importlib.util
import itertools
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("revision_bounds", ROOT / "scripts/revision_bounds.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class SharedBoundsTests(unittest.TestCase):
    def test_exhaustive_small_shared_completions(self):
        c = np.array([.3, -.2, -.4, .1, .2])
        groups = np.array(["g", "g", "h", "j", "j"])
        for status in itertools.product([-1, 0, 1], repeat=3):
            value = dict(zip(["g", "h", "j"], status))
            y = np.array([np.nan if value[g] == -1 else value[g] for g in groups])
            result = MOD.grouped_bounds(c, y, groups)
            missing = [g for g in value if value[g] == -1]
            truth = []
            for bits in itertools.product([0, 1], repeat=len(missing)):
                complete = {**value, **dict(zip(missing, bits))}
                truth.append(float(np.dot(c, [complete[g] for g in groups])))
            self.assertAlmostEqual(result["lower"], min(truth))
            self.assertAlmostEqual(result["upper"], max(truth))
            self.assertLessEqual(result["width"], result["outer_width"] + 1e-12)

    def test_identical_policies(self):
        result = MOD.grouped_bounds([0, 0], [np.nan, np.nan], ["a", "b"])
        self.assertEqual(result["width"], 0)

    def test_all_known(self):
        result = MOD.grouped_bounds([.5, -.2, -.3], [1, 1, 0], ["a", "a", "b"])
        self.assertAlmostEqual(result["lower"], .3)
        self.assertEqual(result["width"], 0)

    def test_all_unknown_same_reference_cancellation(self):
        result = MOD.grouped_bounds([.5, -.5], [np.nan, np.nan], ["same", "same"])
        self.assertEqual(result["width"], 0)
        self.assertEqual(result["outer_width"], 1)

    def test_no_scene_different_decisions_not_shared(self):
        f = pd.DataFrame({"campaign": ["2024"]*2, "source_id": ["s"]*2,
            "decision_at": ["d1", "d2"], "outcome_scene": [np.nan, np.nan], "outcome_label": [np.nan, np.nan]})
        keys, reasons = MOD.reference_groups(f)
        self.assertNotEqual(keys[0], keys[1])
        result = MOD.grouped_bounds([.5, -.5], f.outcome_label, keys)
        self.assertEqual(result["width"], 1)
        self.assertTrue((reasons == "no_future_scene").all())

    def test_scene_alone_never_defines_reference(self):
        f = pd.DataFrame({"campaign": ["2024"]*2, "source_id": ["s1", "s2"],
            "decision_at": ["d1", "d1"], "outcome_scene": ["scene"]*2, "outcome_label": [0, 1]})
        keys, _ = MOD.reference_groups(f)
        self.assertNotEqual(keys[0], keys[1])

    def test_conflicting_shared_truth_rejected(self):
        with self.assertRaises(ValueError):
            MOD.grouped_bounds([.1, .2], [0, 1], ["same", "same"])

    def test_known_reference_resolves_its_unknown_copy(self):
        result = MOD.grouped_bounds([.3, -.1], [1, np.nan], ["same", "same"])
        self.assertAlmostEqual(result["lower"], .2)
        self.assertEqual(result["width"], 0)

    def test_revelation_order_is_label_independent(self):
        a = MOD.reveal_path([.3, -.2, .1, -.2], [1, 0, 1, 0], [False]*4, ["a", "b", "c", "d"])
        b = MOD.reveal_path([.3, -.2, .1, -.2], [0, 1, 0, 1], [False]*4, ["a", "b", "c", "d"])
        np.testing.assert_array_equal(a["reveal_order"], b["reveal_order"])
        np.testing.assert_allclose(a["width"], b["width"])
        self.assertEqual(a["needed_for_width"]["0.2"], 3)
        self.assertEqual(a["needed_for_direction"], 2)
        self.assertTrue(np.all(a["width"] <= a["uniform_expected_width"] + 1e-12))

    def test_all_known_revelation_and_negligible_truth(self):
        result = MOD.reveal_path([.1, -.1], [1, 1], [True, True], ["a", "b"])
        self.assertEqual(len(result["budget"]), 1)
        self.assertEqual(result["needed_for_width"]["0.05"], 0)
        self.assertIsNone(result["needed_for_direction"])
        self.assertTrue(result["within_threshold_band"][0])


if __name__ == "__main__":
    unittest.main()
