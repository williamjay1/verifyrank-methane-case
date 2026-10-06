"""Synthetic invariants for campaign evaluation; no scientific results here."""

import importlib.util
import itertools
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluate_campaigns", ROOT / "scripts/evaluate_campaigns.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def snapshot(campaign="2023", day=10, source="a", label=1, block="cell_1"):
    t = pd.Timestamp(f"{campaign}-08-{day:02d}T00:00:00Z")
    return {"campaign": campaign, "scenario": "qaccepted_r250_h3_l24", "decision_at": t,
        "source_id": source, "window_end": t + pd.Timedelta(days=3),
        "outcome_available_at": t + pd.Timedelta(days=2), "outcome_observed_at": t + pd.Timedelta(days=1),
        "outcome_scene": "scene_" + str(day), "outcome_label": label, "window_complete": 1,
        "first_future_scene": "scene_" + str(day), "first_future_status": "known" if label is not None else "unknown",
        "first_future_at": t + pd.Timedelta(days=1), "candidate_known_at": t - pd.Timedelta(days=2),
        "feature_max_observed_at": t - pd.Timedelta(days=2), "feature_max_available_at": t - pd.Timedelta(days=1),
        "history_detect_count": 1, "history_observation_count": 2, "latest_label": 0,
        "days_since_last_detection": 2, "max_emission": 20, "has_emission": 1,
        "region": "synthetic", "cell_id": block}


class CampaignTests(unittest.TestCase):
    def test_tie_weights_sum_to_budget_without_using_labels(self):
        w = MOD.tie_selection_weights(np.array([3, 2, 2, 2, 1]), 2)
        np.testing.assert_allclose(w, [1, 1 / 3, 1 / 3, 1 / 3, 0])
        self.assertAlmostEqual(w.sum(), 2)
        self.assertTrue(((w >= 0) & (w <= 1)).all())

    def test_uniform_ties_and_fraction_budget(self):
        np.testing.assert_allclose(MOD.tie_selection_weights(np.ones(5), 2), np.repeat(0.4, 5))
        self.assertEqual(MOD.effective_budget("fraction", 0.25, 9), 3)
        self.assertEqual(MOD.effective_budget("k", 50, 9), 9)

    def test_identical_policies_cancel_shared_unknown_completely(self):
        y = np.array([1, np.nan, np.nan])
        w = np.array([0.5, 0.5, 0])
        result, c = MOD.completion_bounds(y, w, w)
        self.assertEqual(result["lower"], 0)
        self.assertEqual(result["upper"], 0)
        self.assertEqual(result["interval_width"], 0)
        np.testing.assert_array_equal(c, np.zeros(3))

    def test_completion_bounds_match_exhaustive_binary_completions(self):
        y = np.array([1, np.nan, 0, np.nan, np.nan])
        wa = MOD.tie_selection_weights(np.array([5, 4, 3, 2, 1]), 2)
        wb = MOD.tie_selection_weights(np.array([1, 2, 3, 3, 3]), 2)
        result, c = MOD.completion_bounds(y, wa, wb)
        possible = []
        for completion in itertools.product([0, 1], repeat=3):
            filled = y.copy()
            filled[~np.isfinite(filled)] = completion
            possible.append(float(np.dot(c, filled)))
        self.assertAlmostEqual(result["lower"], min(possible))
        self.assertAlmostEqual(result["upper"], max(possible))
        self.assertAlmostEqual(result["interval_width"], np.abs(c[~np.isfinite(y)]).sum())

    def test_completion_bounds_reject_unequal_budget(self):
        with self.assertRaisesRegex(ValueError, "identical"):
            MOD.completion_bounds(np.array([1, np.nan]), np.array([1, 0]), np.array([1, 1]))

    def test_resolving_one_unknown_reduces_width_by_abs_coefficient(self):
        y = np.array([1, np.nan, np.nan])
        a, b = np.array([0, 1, 0]), np.array([0.5, 0, 0.5])
        before, c = MOD.completion_bounds(y, a, b)
        y[1] = 0
        after, _ = MOD.completion_bounds(y, a, b)
        self.assertAlmostEqual(before["interval_width"] - after["interval_width"], abs(c[1]))

    def test_first_unknown_cannot_be_replaced_by_later_label(self):
        row = snapshot(label=None)
        row["outcome_scene"] = "later_scene"
        with self.assertRaisesRegex(ValueError, "first eligible scene"):
            MOD.validate_snapshots(pd.DataFrame([row]))

    def test_feature_and_candidate_availability_must_precede_decision(self):
        for column in ["candidate_known_at", "feature_max_observed_at", "feature_max_available_at"]:
            row = snapshot()
            row[column] = row["decision_at"]
            with self.assertRaisesRegex(ValueError, "strictly before"):
                MOD.validate_snapshots(pd.DataFrame([row]))

    def test_source_spatial_block_cannot_change_between_decisions(self):
        with self.assertRaisesRegex(ValueError, "spatial block"):
            MOD.validate_snapshots(pd.DataFrame([snapshot(day=10), snapshot(day=11, block="different")]))

    def test_training_requires_complete_window_and_available_outcome(self):
        rows = [snapshot(day=10, source="a"), snapshot(day=11, source="b"), snapshot(day=10, source="c")]
        rows[2]["outcome_available_at"] = pd.Timestamp("2023-08-14T00:00:00Z")
        frame, _ = MOD.validate_snapshots(pd.DataFrame(rows))
        train = MOD.training_rows(frame, pd.Timestamp("2023-08-13T00:00:00Z"), "2023")
        self.assertEqual(train.source_id.tolist(), ["a"])

    def test_administratively_incomplete_windows_excluded(self):
        row = snapshot()
        row["window_complete"] = 0
        frame, audit = MOD.validate_snapshots(pd.DataFrame([row]))
        self.assertTrue(frame.empty)
        self.assertEqual(audit["administratively_incomplete_rows_excluded"], 1)

    def test_fixed_hgb_does_not_early_stop_on_holdout(self):
        model = MOD.build_estimator("hist_gradient_boosting")
        self.assertFalse(model.early_stopping)
        self.assertEqual(model.max_iter, 100)
        self.assertEqual(model.max_leaf_nodes, 7)
        self.assertEqual(model.min_samples_leaf, 20)

    def test_future_labels_do_not_change_external_predictions(self):
        rows = []
        for i in range(20):
            rows += [snapshot(source=f"s{i}", label=i % 2, block=f"c{i}"), snapshot(campaign="2024", source=f"s{i}", label=i % 2, block=f"c{i}")]
        frame, _ = MOD.validate_snapshots(pd.DataFrame(rows))
        modified = frame.copy()
        test = modified.campaign.eq("2024")
        modified.loc[test, "outcome_label"] = 1 - modified.loc[test, "outcome_label"]
        a, _ = MOD.score_campaigns(frame)
        b, _ = MOD.score_campaigns(modified)
        a = a.loc[a.evaluation_mode.eq("external_2023_to_2024")]
        b = b.loc[b.evaluation_mode.eq("external_2023_to_2024")]
        pd.testing.assert_frame_equal(a[["score_logistic", "score_hist_gradient_boosting"]], b[["score_logistic", "score_hist_gradient_boosting"]])

    def test_missing_2023_strict_data_skips_external_learning(self):
        frame, _ = MOD.validate_snapshots(pd.DataFrame([snapshot(campaign="2024")]))
        scored, audit = MOD.score_campaigns(frame)
        self.assertTrue(scored.score_logistic.isna().all())
        self.assertTrue(audit.status.eq("skipped").all())

    def test_external_logistic_coefficients_and_training_scaler_are_auditable(self):
        rows = [snapshot(source=f"s{i}", label=i % 2, block=f"c{i}") for i in range(20)]
        frame, _ = MOD.validate_snapshots(pd.DataFrame(rows))
        fitted, audit = MOD.fit_fixed_models(frame)
        row = next(value for value in audit if value["model"] == "logistic")
        self.assertEqual(row["status"], "fitted")
        for feature in MOD.FEATURES:
            self.assertTrue(np.isfinite(row[f"standardized_coefficient_{feature}"]))
            self.assertAlmostEqual(row[f"training_scaler_mean_{feature}"], MOD.feature_matrix(frame)[feature].mean())

    def test_fixed_calibration_bins_keep_unknown_counts_and_include_one(self):
        rows = [snapshot(campaign="2024", source=f"s{i}", label=label, block=f"c{i}") for i, label in enumerate([1, 0, None, 1])]
        frame, _ = MOD.validate_snapshots(pd.DataFrame(rows))
        scored, _ = MOD.score_campaigns(frame)
        scored = scored.loc[scored.evaluation_mode.eq("external_2023_to_2024")].copy()
        scored["score_logistic"] = [0.0, 0.2, 0.8, 1.0]
        calibration = MOD.external_calibration(scored)
        bins = calibration.loc[calibration.model.eq("logistic")].reset_index(drop=True)
        self.assertEqual(bins.snapshot_count.tolist(), [1, 1, 0, 0, 2])
        self.assertEqual(bins.iloc[4].known_outcome_count, 1)
        self.assertEqual(bins.iloc[4].unknown_outcome_count, 1)
        self.assertEqual(bins.iloc[4].observed_positive_fraction_known, 1)

    def test_walkforward_future_window_labels_cannot_change_current_predictions(self):
        rows = [snapshot(campaign="2024", day=day, source=f"s{i}", label=i % 2, block=f"c{i}") for day in [10, 13, 16] for i in range(20)]
        frame, _ = MOD.validate_snapshots(pd.DataFrame(rows))
        modified = frame.copy()
        current = pd.Timestamp("2024-08-13T00:00:00Z")
        future = modified.decision_at.ge(current)
        modified.loc[future, "outcome_label"] = 1 - modified.loc[future, "outcome_label"]
        a, _ = MOD.score_campaigns(frame)
        b, _ = MOD.score_campaigns(modified)
        mask_a = a.evaluation_mode.eq("within_campaign_walkforward") & a.decision_at.eq(current)
        mask_b = b.evaluation_mode.eq("within_campaign_walkforward") & b.decision_at.eq(current)
        self.assertTrue(a.loc[mask_a, "score_logistic"].notna().all())
        pd.testing.assert_frame_equal(a.loc[mask_a, ["score_logistic", "score_hist_gradient_boosting"]], b.loc[mask_b, ["score_logistic", "score_hist_gradient_boosting"]])

    def test_spatial_draw_preserves_all_repeated_rows_in_each_block(self):
        cells = np.array(["a", "b", "a", "c", "b", "a"])
        multiplicities, blocks = MOD.block_multiplicities(cells, np.random.default_rng(5))
        self.assertEqual(len(blocks), 3)
        for block in blocks:
            self.assertEqual(len(np.unique(multiplicities[cells == block])), 1)

    def test_weighted_ranking_matches_explicit_block_duplicates(self):
        scores = np.array([3, 2, 2, 1])
        mass = np.array([2, 1, 3, 0])
        aggregated = MOD.tie_selection_weights(scores, 4, mass)
        expanded_scores = np.repeat(scores, mass)
        expanded_ids = np.repeat(np.arange(len(scores)), mass)
        expanded_weights = MOD.tie_selection_weights(expanded_scores, 4)
        expected = np.bincount(expanded_ids, weights=expanded_weights, minlength=len(scores))
        np.testing.assert_allclose(aggregated, expected)

    def test_campaign_macro_is_not_row_weighted(self):
        values = np.array([[0.0], [1.0], [1.0]])
        campaigns = np.array(["2023", "2024", "2024"])
        self.assertEqual(MOD._macro_array(values, campaigns)[0], 0.5)

    def test_bootstrap_skips_insufficient_blocks_and_never_bootstraps_bounds(self):
        frame, _ = MOD.validate_snapshots(pd.DataFrame([snapshot(source="a", label=1), snapshot(source="b", label=0)]))
        scored, _ = MOD.score_campaigns(frame)
        intervals = MOD.spatial_bootstrap(scored, reps=5)
        self.assertTrue(intervals.status.eq("insufficient_spatial_blocks").all())
        self.assertTrue(intervals.ci_lower.isna().all())
        self.assertTrue(intervals.metric.str.startswith("conditional_").all())

    def test_bootstrap_point_matches_same_budget_daily_macro(self):
        rows = []
        for day in [10, 11]:
            for i in range(20):
                row = snapshot(campaign="2024", day=day, source=f"s{i}", label=(i + day) % 3 == 0, block=f"block_{i}")
                row["history_observation_count"] = 2 + i % 4
                row["latest_label"] = i % 2
                rows.append(row)
        frame, _ = MOD.validate_snapshots(pd.DataFrame(rows))
        scored, _ = MOD.score_campaigns(frame)
        external = scored.loc[scored.evaluation_mode.eq("external_2023_to_2024")]
        tables = MOD.daily_evaluation(external)
        summary = MOD.macro_summary(tables)
        intervals = MOD.spatial_bootstrap(external, reps=4, budget_specs=[("fraction", 0.25)])
        point = summary.loc[summary.model.eq("last_label") & summary.reference.eq("beta_rate") & summary.budget_kind.eq("fraction") & summary.budget_value.eq(0.25)].iloc[0]
        interval = intervals.loc[intervals.model.eq("last_label") & intervals.reference.eq("beta_rate") & intervals.metric.eq("conditional_precision_difference")].iloc[0]
        self.assertAlmostEqual(point.conditional_precision_difference, interval.point_estimate)
        self.assertEqual(interval.valid_bootstrap_replicates, 4)
        self.assertEqual(interval.status, "estimated")


if __name__ == "__main__":
    unittest.main()
