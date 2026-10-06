"""Daily fixed-window methane evaluation across campaigns and walk-forward fits.

Input snapshots are constructed upstream from fixed historical anchors. This
module never constructs sources, judges scene quality, or fills missing labels.
Unknown-reference-outcome completion bounds are evaluation sensitivity
diagnostics, not bounds on actual population emissions or deployment benefits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261005
MODELS = ["beta_rate", "last_label", "max_emission", "uniform_random", "logistic", "hist_gradient_boosting"]
PROBABILITY_MODELS = {"beta_rate", "logistic", "hist_gradient_boosting"}
FEATURES = ["beta_rate", "log1p_history_count", "latest_label", "days_since_last_detection", "log1p_max_emission", "has_emission"]
BUDGETS = [("k", 10), ("k", 25), ("k", 50), ("fraction", 0.1), ("fraction", 0.25)]
PRIMARY_SCENARIO = "qaccepted_r250_h3_l24"
REQUIRED = {
    "campaign", "scenario", "decision_at", "source_id", "window_end", "outcome_available_at",
    "outcome_observed_at", "outcome_scene", "outcome_label", "window_complete", "first_future_scene",
    "first_future_status", "first_future_at", "candidate_known_at", "feature_max_observed_at",
    "feature_max_available_at", "history_detect_count", "history_observation_count", "latest_label",
    "days_since_last_detection", "max_emission", "has_emission", "region", "cell_id",
}
PAIRS = [(model, reference) for model in MODELS if model != "uniform_random" for reference in ["beta_rate", "uniform_random"] if model != reference]


def validate_snapshots(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    missing = REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"Missing snapshot columns: {sorted(missing)}")
    frame = raw.copy()
    for column in ["campaign", "scenario", "source_id", "cell_id"]:
        if frame[column].isna().any():
            raise ValueError(f"{column} cannot be missing")
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"{column} cannot be empty")
    required_times = ["decision_at", "window_end", "candidate_known_at", "feature_max_observed_at", "feature_max_available_at"]
    optional_times = ["outcome_available_at", "outcome_observed_at", "first_future_at"]
    for column in required_times + optional_times:
        frame[column] = pd.to_datetime(frame[column], format="mixed", utc=True, errors="raise")
        if column in required_times and frame[column].isna().any():
            raise ValueError(f"{column} cannot be missing")
    if (frame["decision_at"] != frame["decision_at"].dt.floor("D")).any():
        raise ValueError("decision_at must be UTC midnight")
    for column in ["candidate_known_at", "feature_max_observed_at", "feature_max_available_at"]:
        if (frame[column] >= frame["decision_at"]).any():
            raise ValueError(f"{column} must be strictly before decision_at")
    frame["horizon_days"] = (frame["window_end"] - frame["decision_at"]).dt.total_seconds() / 86400
    if not frame["horizon_days"].isin([3, 7]).all():
        raise ValueError("Only the fixed primary 3-day and sensitivity 7-day windows are permitted")
    if (frame.groupby("scenario")["horizon_days"].nunique() > 1).any():
        raise ValueError("A scenario cannot mix forecast windows")
    frame["window_complete"] = pd.to_numeric(frame["window_complete"], errors="raise")
    if not frame["window_complete"].isin([0, 1]).all():
        raise ValueError("window_complete must be 0/1")
    frame["outcome_label"] = pd.to_numeric(frame["outcome_label"], errors="raise")
    if not frame["outcome_label"].dropna().isin([0, 1]).all():
        raise ValueError("outcome_label must be 0, 1, or missing")
    known = frame["outcome_label"].notna()
    if frame.loc[known, ["outcome_available_at", "outcome_observed_at"]].isna().any().any():
        raise ValueError("A known label requires observation and availability timestamps")
    observed = frame["outcome_observed_at"].notna()
    if ((frame.loc[observed, "outcome_observed_at"] <= frame.loc[observed, "decision_at"]) | (frame.loc[observed, "outcome_observed_at"] > frame.loc[observed, "window_end"])).any():
        raise ValueError("Outcome must be in (decision_at, window_end]")
    available = frame["outcome_available_at"].notna() & observed
    if (frame.loc[available, "outcome_available_at"] < frame.loc[available, "outcome_observed_at"]).any():
        raise ValueError("Outcome availability precedes observation")
    scenes = frame["outcome_scene"].fillna("").astype(str)
    first_scenes = frame["first_future_scene"].fillna("").astype(str)
    if not scenes.eq(first_scenes).all():
        raise ValueError("Outcome must remain the first eligible scene, including an unknown label")
    first_present = frame["first_future_at"].notna() | observed
    if not frame.loc[first_present, "first_future_at"].eq(frame.loc[first_present, "outcome_observed_at"]).all():
        raise ValueError("First eligible observation time cannot be replaced by a later resolved outcome")
    numeric = ["history_detect_count", "history_observation_count", "latest_label", "days_since_last_detection", "max_emission", "has_emission"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in numeric:
        if column != "max_emission" and not np.isfinite(frame[column].to_numpy(float)).all():
            raise ValueError(f"{column} must be finite")
    for column in ["history_detect_count", "history_observation_count"]:
        if (frame[column] < 1).any() or not np.equal(frame[column], np.floor(frame[column])).all():
            raise ValueError(f"{column} must be a positive count for historically detected candidates")
    if (frame["history_detect_count"] > frame["history_observation_count"]).any():
        raise ValueError("Detection count exceeds historical observation count")
    if not frame["latest_label"].isin([0, 1]).all() or not frame["has_emission"].isin([0, 1]).all():
        raise ValueError("latest_label and has_emission must be binary")
    if (frame["days_since_last_detection"] < 0).any():
        raise ValueError("Negative historical recency")
    rates = frame.loc[frame["has_emission"].eq(1), "max_emission"]
    if not np.isfinite(rates.to_numpy(float)).all() or (rates < 0).any():
        raise ValueError("Available emission values must be finite and nonnegative")
    keys = ["campaign", "scenario", "decision_at", "source_id"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate source within a campaign/scenario/decision")
    if (frame.groupby(["campaign", "scenario", "source_id"])["cell_id"].nunique() > 1).any():
        raise ValueError("A fixed source anchor must retain its spatial block across decisions")
    audit = {"input_rows": len(frame), "administratively_incomplete_rows_excluded": int(frame["window_complete"].eq(0).sum())}
    frame = frame.loc[frame["window_complete"].eq(1)].sort_values(keys).reset_index(drop=True)
    audit["complete_rows"] = len(frame)
    return frame, audit


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    has = frame["has_emission"].astype(float)
    return pd.DataFrame({
        "beta_rate": (frame["history_detect_count"] + 1) / (frame["history_observation_count"] + 2),
        "log1p_history_count": np.log1p(frame["history_observation_count"]),
        "latest_label": frame["latest_label"].astype(float),
        "days_since_last_detection": frame["days_since_last_detection"].astype(float),
        "log1p_max_emission": np.log1p(frame["max_emission"].where(has.eq(1), 0).fillna(0)),
        "has_emission": has,
    }, index=frame.index)[FEATURES]


def build_estimator(name: str):
    if name == "logistic":
        return Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(C=1, max_iter=2000, random_state=SEED))])
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, min_samples_leaf=20, learning_rate=0.1, l2_regularization=1, early_stopping=False, random_state=SEED)
    raise ValueError(name)


def training_rows(frame: pd.DataFrame, fit_at: pd.Timestamp, campaign: str) -> pd.DataFrame:
    return frame.loc[
        frame["campaign"].eq(campaign) & frame["outcome_label"].notna()
        & (frame["window_end"] <= fit_at) & (frame["outcome_available_at"] < fit_at)
    ]


def fit_fixed_models(train: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts = {"training_rows": len(train), "training_sources": int(train["source_id"].nunique()), "training_positive": int(train["outcome_label"].eq(1).sum()), "training_negative": int(train["outcome_label"].eq(0).sum())}
    reasons = []
    if counts["training_sources"] < 20:
        reasons.append("fewer_than_20_training_sources")
    if min(counts["training_positive"], counts["training_negative"]) < 5:
        reasons.append("fewer_than_5_outcomes_in_either_class")
    models, audit = {}, []
    for name in ["logistic", "hist_gradient_boosting"]:
        row = {"model": name, **counts, "status": "skipped" if reasons else "fitted", "reason": ";".join(reasons)}
        if not reasons:
            estimator = build_estimator(name)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                try:
                    estimator.fit(feature_matrix(train), train["outcome_label"].astype(int))
                    models[name] = estimator
                    if name == "logistic":
                        scaler = estimator.named_steps["scaler"]
                        logistic = estimator.named_steps["model"]
                        row["standardized_intercept"] = float(logistic.intercept_[0])
                        for i, feature in enumerate(FEATURES):
                            row[f"standardized_coefficient_{feature}"] = float(logistic.coef_[0, i])
                            row[f"training_scaler_mean_{feature}"] = float(scaler.mean_[i])
                            row[f"training_scaler_scale_{feature}"] = float(scaler.scale_[i])
                        row["coefficient_interpretation"] = "Conditional predictive log-odds associations after training-only standardization; not causal effects or independent tests of a mechanism"
                except ConvergenceWarning:
                    row.update(status="skipped", reason="fixed_estimator_did_not_converge")
        audit.append(row)
    return models, audit


def score_campaigns(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs, fit_audit = [], []
    for scenario, scenario_data in frame.groupby("scenario", sort=True):
        for mode in ["external_2023_to_2024", "within_campaign_walkforward"]:
            test = scenario_data.loc[scenario_data["campaign"].eq("2024")] if mode.startswith("external") else scenario_data
            external_models = None
            if test.empty:
                continue
            if mode.startswith("external"):
                fit_at = test["decision_at"].min()
                external_models, audits = fit_fixed_models(training_rows(scenario_data, fit_at, "2023"))
                for audit in audits:
                    fit_audit.append({"scenario": scenario, "mode": mode, "campaign": "2024", "fit_at": fit_at, **audit})
            for (campaign, decision), day in test.groupby(["campaign", "decision_at"], sort=True):
                if external_models is None:
                    fitted, audits = fit_fixed_models(training_rows(scenario_data, decision, campaign))
                    for audit in audits:
                        fit_audit.append({"scenario": scenario, "mode": mode, "campaign": campaign, "fit_at": decision, **audit})
                else:
                    fitted = external_models
                features = feature_matrix(day)
                # All predictions are generated from the six allowlisted columns;
                # no outcome, cell, region, or first-future metadata enters a model.
                scores = {
                    "beta_rate": features["beta_rate"].to_numpy(),
                    "last_label": day["latest_label"].to_numpy(float),
                    "max_emission": day["max_emission"].where(day["has_emission"].eq(1), -1).fillna(-1).to_numpy(float),
                    "uniform_random": np.zeros(len(day)),
                }
                for model in ["logistic", "hist_gradient_boosting"]:
                    scores[model] = fitted[model].predict_proba(features)[:, 1] if model in fitted else np.full(len(day), np.nan)
                scored = day.copy()
                scored["evaluation_mode"] = mode
                for name, values in scores.items():
                    scored[f"score_{name}"] = values
                outputs.append(scored)
    columns = list(frame.columns) + ["evaluation_mode"] + [f"score_{name}" for name in MODELS]
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame(columns=columns), pd.DataFrame(fit_audit)


def score_groups(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Ranking scores must be finite")
    return np.unique(-scores, return_inverse=True)[1]


def selection_from_groups(groups: np.ndarray, k: int, multiplicity: np.ndarray | None = None) -> np.ndarray:
    mass = np.ones(len(groups), dtype=float) if multiplicity is None else np.asarray(multiplicity, dtype=float)
    if len(groups) != len(mass) or not np.isfinite(mass).all() or (mass < 0).any():
        raise ValueError("Invalid row multiplicities")
    if k < 0:
        raise ValueError("Negative budget")
    selected = min(float(k), float(mass.sum()))
    if not len(groups) or selected == 0:
        return np.zeros(len(groups))
    group_mass = np.bincount(groups, weights=mass)
    cumulative = np.cumsum(group_mass)
    boundary = int(np.searchsorted(cumulative, selected, side="left"))
    earlier = cumulative[boundary - 1] if boundary else 0.0
    boundary_probability = (selected - earlier) / group_mass[boundary]
    probability = np.where(groups < boundary, 1.0, np.where(groups == boundary, boundary_probability, 0.0))
    return mass * probability


def tie_selection_weights(scores: np.ndarray, k: int, multiplicity: np.ndarray | None = None) -> np.ndarray:
    """Original rows receive probabilities in [0,1], summing to min(K,N).

    For bootstrap multiplicities, return total selected mass of duplicate rows.
    Labels and source IDs never participate in defining a score tie.
    """
    return selection_from_groups(score_groups(scores), k, multiplicity)


def effective_budget(kind: str, value: float, population: float) -> int:
    requested = int(value) if kind == "k" else int(math.ceil(value * population))
    return min(requested, int(round(population)))


def completion_bounds(y: np.ndarray, weights_a: np.ndarray, weights_b: np.ndarray) -> tuple[dict[str, float | int | None], np.ndarray]:
    """Sharp per-decision completion bounds, conditional on unrestricted binary missing labels.

    The known contribution is sum(c_i*y_i), c_i=(wA_i-wB_i)/m. Each missing
    binary label independently contributes min(0,c_i) or max(0,c_i) at the two
    endpoints. Summing them attains both extrema. This is a finite separable
    calculation, not a confidence interval or a causal identification result.
    """
    y, weights_a, weights_b = np.asarray(y, float), np.asarray(weights_a, float), np.asarray(weights_b, float)
    if not (len(y) == len(weights_a) == len(weights_b)) or not np.isfinite(weights_a).all() or not np.isfinite(weights_b).all():
        raise ValueError("Labels and finite selection weights must align")
    if (weights_a < 0).any() or (weights_b < 0).any() or not np.isclose(weights_a.sum(), weights_b.sum()):
        raise ValueError("Compared policies must have identical nonnegative total selection mass")
    known = np.isfinite(y)
    if not np.isin(y[known], [0, 1]).all():
        raise ValueError("Known outcomes must be binary")
    m = float(weights_a.sum())
    if m == 0:
        return {"effective_budget": 0, "known_contribution": None, "lower": None, "upper": None, "interval_width": None}, np.zeros(len(y))
    c = (weights_a - weights_b) / m
    known_contribution = float(np.dot(c[known], y[known]))
    lower = known_contribution + float(np.minimum(c[~known], 0).sum())
    upper = known_contribution + float(np.maximum(c[~known], 0).sum())
    return {"effective_budget": m, "known_contribution": known_contribution, "lower": lower, "upper": upper, "interval_width": float(np.abs(c[~known]).sum())}, c


def daily_evaluation(scored: pd.DataFrame) -> dict[str, pd.DataFrame]:
    metric_rows, pair_rows, weight_rows, priority_rows, brier_rows = [], [], [], [], []
    group_columns = ["scenario", "evaluation_mode", "campaign", "decision_at"]
    for keys, day in scored.groupby(group_columns, sort=True):
        metadata = dict(zip(group_columns, keys))
        day = day.sort_values("source_id")
        source_ids = day["source_id"].to_numpy(str)
        cells = day["cell_id"].to_numpy(str)
        y = day["outcome_label"].to_numpy(float)
        known = np.isfinite(y)
        n, nv = len(day), int(known.sum())
        available_models = [name for name in MODELS if day[f"score_{name}"].notna().all()]
        for model in available_models:
            if model in PROBABILITY_MODELS:
                probability = day[f"score_{model}"].to_numpy(float)
                brier_rows.append({**metadata, "model": model, "known_count": nv, "brier": float(np.mean((probability[known] - y[known]) ** 2)) if nv else np.nan})
        for kind, value in BUDGETS:
            m = effective_budget(kind, value, n)
            budget_metadata = {**metadata, "budget_kind": kind, "budget_value": value, "effective_budget": m, "candidate_count": n, "known_count": nv}
            full_weights, conditional_p = {}, {}
            for model in available_models:
                score = day[f"score_{model}"].to_numpy(float)
                weights = tie_selection_weights(score, m)
                full_weights[model] = weights
                positive = float(np.dot(weights[known], y[known]))
                unknown = float(weights[~known].sum())
                known_weights = tie_selection_weights(score[known], min(m, nv))
                conditional_p[model] = float(np.dot(known_weights, y[known]) / known_weights.sum()) if nv else np.nan
                metric_rows.append({
                    **budget_metadata, "model": model, "expected_known_positive": positive,
                    "expected_unknown": unknown, "completion_precision_lower": positive / m if m else np.nan,
                    "completion_precision_upper": (positive + unknown) / m if m else np.nan,
                    "conditional_effective_budget": min(m, nv), "conditional_expected_precision": conditional_p[model],
                })
                for sid, block, label, weight in zip(day["source_id"], day["cell_id"], y, weights):
                    weight_rows.append({**budget_metadata, "model": model, "source_id": sid, "cell_id": block, "outcome_label": label, "selection_probability": float(weight)})
            for model, reference in PAIRS:
                if model not in full_weights or reference not in full_weights:
                    continue
                bounds, c = completion_bounds(y, full_weights[model], full_weights[reference])
                pair_rows.append({
                    **budget_metadata, "model": model, "reference": reference,
                    "conditional_precision_difference": conditional_p[model] - conditional_p[reference],
                    "completion_difference_lower": bounds["lower"], "completion_difference_upper": bounds["upper"],
                    "completion_interval_width": bounds["interval_width"], "known_contribution": bounds["known_contribution"],
                    "unknown_with_nonzero_difference_weight": int(((np.abs(c) > 1e-15) & ~known).sum()),
                    "unknown_with_zero_difference_weight": int(((np.abs(c) <= 1e-15) & ~known).sum()),
                })
                # Shared unknowns with c=0 already cancel. They cannot shrink the
                # comparison interval and are counted above, not priority-listed.
                unknown_indices = np.flatnonzero(~known & (np.abs(c) > 1e-15))
                ordered = sorted(unknown_indices, key=lambda i: (-abs(c[i]), source_ids[i]))
                for rank, i in enumerate(ordered, 1):
                    priority_rows.append({
                        **budget_metadata, "model": model, "reference": reference,
                        "source_id": source_ids[i], "cell_id": cells[i],
                        "difference_coefficient": float(c[i]), "ideal_width_reduction_if_resolved": float(abs(c[i])),
                        "priority_rank": rank,
                    })
    return {"daily_metrics": pd.DataFrame(metric_rows), "paired_daily": pd.DataFrame(pair_rows), "selection_weights": pd.DataFrame(weight_rows), "unknown_verification_priority": pd.DataFrame(priority_rows), "brier_daily": pd.DataFrame(brier_rows)}


def macro_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = tables["paired_daily"]
    if frame.empty:
        return pd.DataFrame()
    keys = ["scenario", "evaluation_mode", "model", "reference", "budget_kind", "budget_value"]
    measures = ["conditional_precision_difference", "completion_difference_lower", "completion_difference_upper", "completion_interval_width", "known_contribution"]
    rows = []
    for values, subset in frame.groupby(keys, sort=True):
        campaign_means = subset.groupby("campaign")[measures].mean()
        rows.append({**dict(zip(keys, values)), "paired_decision_count": len(subset), "campaign_count": len(campaign_means), **campaign_means.mean().to_dict()})
    return pd.DataFrame(rows)


def policy_macro_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = tables["daily_metrics"]
    if frame.empty:
        return pd.DataFrame()
    keys = ["scenario", "evaluation_mode", "model", "budget_kind", "budget_value"]
    measures = ["expected_known_positive", "expected_unknown", "completion_precision_lower", "completion_precision_upper", "conditional_expected_precision"]
    rows = []
    for values, subset in frame.groupby(keys, sort=True):
        campaign_means = subset.groupby("campaign")[measures].mean()
        rows.append({**dict(zip(keys, values)), "evaluable_decision_count": len(subset), "campaign_count": len(campaign_means), **campaign_means.mean().to_dict()})
    return pd.DataFrame(rows)


def brier_summaries(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = tables["brier_daily"]
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows, paired_rows = [], []
    for (scenario, mode, model), subset in frame.groupby(["scenario", "evaluation_mode", "model"], sort=True):
        means = subset.groupby("campaign")["brier"].mean().dropna()
        rows.append({"scenario": scenario, "evaluation_mode": mode, "model": model, "evaluable_decision_count": int(subset.brier.notna().sum()), "campaign_count": len(means), "brier": float(means.mean()) if len(means) else np.nan})
    keys = ["scenario", "evaluation_mode", "campaign", "decision_at"]
    reference = frame.loc[frame.model.eq("beta_rate"), keys + ["brier"]].rename(columns={"brier": "reference_brier"})
    for model in ["logistic", "hist_gradient_boosting"]:
        paired = frame.loc[frame.model.eq(model)].merge(reference, on=keys, how="inner", validate="one_to_one")
        paired["brier_difference"] = paired["brier"] - paired["reference_brier"]
        for (scenario, mode), subset in paired.groupby(["scenario", "evaluation_mode"], sort=True):
            means = subset.groupby("campaign")["brier_difference"].mean().dropna()
            paired_rows.append({"scenario": scenario, "evaluation_mode": mode, "model": model, "reference": "beta_rate", "paired_evaluable_decision_count": int(subset.brier_difference.notna().sum()), "campaign_count": len(means), "brier_difference": float(means.mean()) if len(means) else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(paired_rows)


def external_calibration(scored: pd.DataFrame) -> pd.DataFrame:
    """Fixed bins on external 2024 probabilities; known-outcome conditional only.

    Counts are repeated daily snapshot rows, not independent source counts.
    Empty bins are retained. Missing outcomes are reported, never imputed.
    """
    external = scored.loc[scored["evaluation_mode"].eq("external_2023_to_2024") & scored["campaign"].eq("2024")]
    edges = np.linspace(0, 1, 6)
    rows = []
    for scenario, subset in external.groupby("scenario", sort=True):
        for model in sorted(PROBABILITY_MODELS):
            available = subset.loc[subset[f"score_{model}"].notna()]
            if available.empty:
                continue
            probabilities = available[f"score_{model}"].to_numpy(float)
            if ((probabilities < 0) | (probabilities > 1)).any():
                raise ValueError("Calibration probabilities must be in [0,1]")
            bins = np.minimum(np.searchsorted(edges, probabilities, side="right") - 1, 4)
            for bi in range(5):
                selected = available.iloc[np.flatnonzero(bins == bi)]
                known = selected.loc[selected["outcome_label"].notna()]
                rows.append({
                    "scenario": scenario, "campaign": "2024", "evaluation_mode": "external_2023_to_2024", "model": model,
                    "bin_lower": float(edges[bi]), "bin_upper": float(edges[bi + 1]), "right_endpoint_included": bi == 4,
                    "snapshot_count": len(selected), "known_outcome_count": len(known), "unknown_outcome_count": len(selected) - len(known),
                    "unique_source_count": int(selected["source_id"].nunique()), "spatial_block_count": int(selected["cell_id"].nunique()),
                    "mean_predicted_probability_all": float(selected[f"score_{model}"].mean()) if len(selected) else np.nan,
                    "mean_predicted_probability_known": float(known[f"score_{model}"].mean()) if len(known) else np.nan,
                    "observed_positive_count": int(known["outcome_label"].sum()),
                    "observed_positive_fraction_known": float(known["outcome_label"].mean()) if len(known) else np.nan,
                })
    return pd.DataFrame(rows)


def _macro_array(values: np.ndarray, campaigns: np.ndarray) -> np.ndarray:
    campaign_values = []
    for campaign in np.unique(campaigns):
        selected = values[campaigns == campaign]
        count = np.isfinite(selected).sum(axis=0)
        campaign_values.append(np.divide(np.nansum(selected, axis=0), count, out=np.full(count.shape, np.nan, dtype=float), where=count > 0))
    if not campaign_values:
        return np.full(values.shape[1:], np.nan)
    stacked = np.stack(campaign_values)
    count = np.isfinite(stacked).sum(axis=0)
    return np.divide(np.nansum(stacked, axis=0), count, out=np.full(count.shape, np.nan, dtype=float), where=count > 0)


def block_multiplicities(cell_ids: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One draw per spatial cell; all its sources/dates receive identical counts."""
    blocks, inverse = np.unique(np.asarray(cell_ids, str), return_inverse=True)
    counts = np.bincount(rng.integers(0, len(blocks), len(blocks)), minlength=len(blocks)) if len(blocks) else np.array([], int)
    return counts[inverse], blocks


def spatial_bootstrap(scored: pd.DataFrame, reps: int = 1000, seed: int = SEED, minimum_blocks: int = 10, budget_specs: list[tuple[str, float]] | None = None) -> pd.DataFrame:
    """Paired, spatial-cluster percentile intervals conditional on fitted scores.

    Recompute ranking budgets/tie expectations in the resampled population.
    Each drawn grid block carries every source and decision date. Models are
    not refit. Shared flight-day errors across blocks remain outside this CI.
    Completion bounds are deliberately not assigned bootstrap confidence limits.
    """
    if reps < 1:
        return pd.DataFrame()
    budgets = BUDGETS if budget_specs is None else budget_specs
    rows = []
    for (scenario, mode), subset in scored.groupby(["scenario", "evaluation_mode"], sort=True):
        block_ids = np.unique(subset["cell_id"].astype(str))
        block_map = {block: i for i, block in enumerate(block_ids)}
        days, campaigns = [], []
        for (campaign, decision), day in subset.groupby(["campaign", "decision_at"], sort=True):
            y = day["outcome_label"].to_numpy(float)
            known = np.isfinite(y)
            models = {}
            for mi, model in enumerate(MODELS):
                score = day[f"score_{model}"].to_numpy(float)
                if np.isfinite(score).all():
                    models[mi] = {"groups": score_groups(score), "score": score}
            days.append({"y": y, "known": known, "models": models, "block": np.array([block_map[str(cell)] for cell in day["cell_id"]]), "n": len(day)})
            campaigns.append(campaign)
        campaigns = np.asarray(campaigns)
        rng = np.random.default_rng(seed)
        p_draws = np.full((reps + 1, len(PAIRS), len(budgets)), np.nan)
        b_draws = np.full((reps + 1, len(PAIRS)), np.nan)
        run_reps = reps if len(block_ids) >= minimum_blocks else 0
        for repetition in range(run_reps + 1):
            counts = np.ones(len(block_ids)) if repetition == 0 else np.bincount(rng.integers(0, len(block_ids), len(block_ids)), minlength=len(block_ids))
            precision = np.full((len(days), len(MODELS), len(budgets)), np.nan)
            brier = np.full((len(days), len(MODELS)), np.nan)
            for di, day in enumerate(days):
                mass = counts[day["block"]]
                population = float(mass.sum())
                known = day["known"]
                known_mass = mass[known]
                known_population = float(known_mass.sum())
                if not population or not known_population:
                    continue
                y = day["y"][known]
                for mi, data in day["models"].items():
                    if MODELS[mi] in PROBABILITY_MODELS:
                        brier[di, mi] = np.dot(known_mass, (data["score"][known] - y) ** 2) / known_population
                    for bi, (kind, value) in enumerate(budgets):
                        m = effective_budget(kind, value, population)
                        selected = selection_from_groups(data["groups"][known], m, known_mass)
                        precision[di, mi, bi] = np.dot(selected, y) / selected.sum() if selected.sum() else np.nan
            for pi, (model, reference) in enumerate(PAIRS):
                a, b = MODELS.index(model), MODELS.index(reference)
                p_draws[repetition, pi] = _macro_array(precision[:, a] - precision[:, b], campaigns)
                b_draws[repetition, pi] = _macro_array((brier[:, a] - brier[:, b])[:, None], campaigns)[0]
        for pi, (model, reference) in enumerate(PAIRS):
            for bi, (kind, value) in enumerate(budgets):
                point = p_draws[0, pi, bi]
                draws = p_draws[1:, pi, bi]
                valid = draws[np.isfinite(draws)]
                rows.append({"scenario": scenario, "evaluation_mode": mode, "model": model, "reference": reference, "metric": "conditional_precision_difference", "budget_kind": kind, "budget_value": value, "point_estimate": point, "ci_lower": float(np.quantile(valid, 0.025)) if len(valid) else np.nan, "ci_upper": float(np.quantile(valid, 0.975)) if len(valid) else np.nan, "spatial_blocks": len(block_ids), "valid_bootstrap_replicates": len(valid), "requested_bootstrap_replicates": reps, "status": "insufficient_spatial_blocks" if len(block_ids) < minimum_blocks else "no_jointly_evaluable_days" if not np.isfinite(point) else "estimated"})
            if model in PROBABILITY_MODELS and reference in PROBABILITY_MODELS:
                point = b_draws[0, pi]
                valid = b_draws[1:, pi][np.isfinite(b_draws[1:, pi])]
                rows.append({"scenario": scenario, "evaluation_mode": mode, "model": model, "reference": reference, "metric": "conditional_brier_difference", "budget_kind": "not_applicable", "budget_value": np.nan, "point_estimate": point, "ci_lower": float(np.quantile(valid, 0.025)) if len(valid) else np.nan, "ci_upper": float(np.quantile(valid, 0.975)) if len(valid) else np.nan, "spatial_blocks": len(block_ids), "valid_bootstrap_replicates": len(valid), "requested_bootstrap_replicates": reps, "status": "insufficient_spatial_blocks" if len(block_ids) < minimum_blocks else "no_jointly_evaluable_days" if not np.isfinite(point) else "estimated"})
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.drive.upper() != "D:":
        raise ValueError("All outputs must be on D:")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new or empty output directory; existing studies are never overwritten")
    if args.bootstrap_reps < 0:
        raise ValueError("bootstrap-reps cannot be negative")
    started = time.perf_counter()
    input_sha256 = _sha256(args.snapshots)
    frame, validation = validate_snapshots(pd.read_csv(args.snapshots, dtype={"campaign": str, "source_id": str, "cell_id": str}))
    print(json.dumps({"stage": "snapshots_validated", **validation}), flush=True)
    scored, audit = score_campaigns(frame)
    print(json.dumps({"stage": "fixed_model_scoring_completed", "prediction_rows": len(scored)}), flush=True)
    tables = daily_evaluation(scored)
    tables["macro_paired_summary"] = macro_summary(tables)
    tables["macro_policy_summary"] = policy_macro_summary(tables)
    tables["macro_brier_summary"], tables["macro_paired_brier_summary"] = brier_summaries(tables)
    tables["training_audit"] = audit
    tables["external_logistic_explanations"] = audit.loc[audit["mode"].eq("external_2023_to_2024") & audit["model"].eq("logistic") & audit["status"].eq("fitted")].copy() if not audit.empty else pd.DataFrame()
    tables["predictions"] = scored
    tables["external_calibration"] = external_calibration(scored)
    primary_external = scored.loc[scored["scenario"].eq(PRIMARY_SCENARIO) & scored["evaluation_mode"].eq("external_2023_to_2024")]
    profile_started = time.perf_counter()
    profile_reps = min(10, args.bootstrap_reps)
    if profile_reps:
        spatial_bootstrap(primary_external, profile_reps, budget_specs=[("fraction", 0.25)])
    profile_seconds = time.perf_counter() - profile_started
    print(json.dumps({"stage": "bootstrap_profile_completed", "profile_replicates": profile_reps, "seconds": profile_seconds}), flush=True)
    tables["bootstrap_intervals"] = spatial_bootstrap(primary_external, args.bootstrap_reps, budget_specs=[("fraction", 0.25)])
    if _sha256(args.snapshots) != input_sha256:
        raise RuntimeError("Input changed during execution; no results are saved")
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    manifest = {
        "status": "completed", "analysis_type": "exploratory_daily_campaign_validation_and_reference_completion_sensitivity",
        "input_path": str(args.snapshots.resolve()), "input_sha256": input_sha256, "code_sha256": _sha256(Path(__file__)),
        "validation": validation, "scenarios": sorted(frame["scenario"].unique().tolist()),
        "campaigns": sorted(frame["campaign"].unique().tolist()), "seed": SEED, "budgets": BUDGETS,
        "features": FEATURES, "primary_horizon_days": 3, "sensitivity_horizon_days": 7,
        "model_parameters": {"logistic": {"C": 1, "max_iter": 2000, "class_weight": None, "scaler": "StandardScaler fit only on eligible training rows"}, "hist_gradient_boosting": {"max_iter": 100, "max_leaf_nodes": 7, "min_samples_leaf": 20, "learning_rate": 0.1, "l2_regularization": 1, "early_stopping": False}},
        "training_sufficiency_rule": "At least 20 unique training sources and 5 outcomes in each class; otherwise learned models skipped. This is an operational fitting guard, not evidence of sufficient statistical precision or publication readiness.",
        "training_rule": "Labels known; window_end <= fit_at AND outcome_available_at < fit_at; external uses 2023 only, walk-forward uses same campaign only",
        "endpoint": "First scene satisfying scene QC and fixed-anchor geometry in (decision_at,window_end]; excluded plume quality leaves this scene outcome unknown, never replaced by a later scene",
        "conditional_budget": "For fraction budgets compute ceil(fraction * full candidate N), then cap at known N for retrospective conditional ranking",
        "aggregation": "Equal-weight decisions within each campaign, then equal-weight campaigns; learned/baseline paired comparisons use jointly evaluable decisions only",
        "completion_bound_proof": "For c=(wA-wB)/m, known sum(c*y) plus sum(min(0,c_unknown)) or sum(max(0,c_unknown)); each unrestricted binary missing label independently attains an endpoint. Width=sum(abs(c_unknown)); shared unknown selection cancels.",
        "completion_bound_interpretation": "Per-decision sharp only for unrestricted binary reference-outcome completions. Macro averages remain valid but need not be jointly sharp if repeated unknowns have shared-label constraints. Bounds are not confidence intervals.",
        "verification_priority_interpretation": "abs(c) is the ideal width reduction if that particular missing reference outcome becomes perfectly known at equal cost; zero-coefficient shared unknowns are counted but omitted from the priority file. For deterministic untied Top-K, every unknown disagreement has abs(c)=1/K, so this identifies an equally ranked disagreement set, not a novel optimal algorithm. Unequal priorities can arise from tie inclusion probabilities. This is a sensitivity/audit priority, not demonstrated value of scheduling a future flight.",
        "bootstrap": {"replicates": args.bootstrap_reps, "scope": "primary qaccepted_r250_h3_l24, external 2023-to-2024, fraction=0.25 paired conditional precision and Brier only", "cluster": "provided cell_id (planned 5000 m grid)", "minimum_blocks": 10, "interval": "percentile 95%", "refits_models": False, "profile_replicates": profile_reps, "profile_seconds": profile_seconds},
        "model_audit": "External logistic standardized coefficients, intercept, training means/scales are saved. Fixed external probability bins [0,.2),[.2,.4),[.4,.6),[.6,.8),[.8,1] report all/known/unknown snapshot counts and observed fractions; no coefficient or calibration bin is interpreted causally.",
        "limitations": [
            "No claim of true zero emissions, mitigation effects, causal policy benefit, or deployment yield.",
            "This is an explicitly assumed information-availability scenario, not reconstructed historical online publication timing.",
            "2023-to-2024 validation is between campaigns in the same basin, not cross-basin generalization or necessarily disjoint physical facilities.",
            "2023 published-plume quality metadata and 2024 accepted-plume criteria differ; strict 2023 quality labels must not be invented.",
            "The observed first qualifying scene occurs at variable lag within a fixed horizon. Training and test share the same daily-window task, but observation schedules still differ.",
            "Conditional outcome metrics select the retrospectively verified subset; completion bounds only diagnose sensitivity to missing reference outcomes.",
            "Snapshot rows and outcome windows overlap. Spatial-block resampling keeps all source histories together but cannot fully capture common flight-day errors across distinct blocks.",
            "Bootstrap intervals condition on fitted models, the selected campaigns, and block construction; they do not include model-refitting, archival label uncertainty, or all sampling-design uncertainty.",
            "Daily repeated recommendations do not model a cumulative multi-day visit budget, routing, or avoided repeat visits.",
            "All scenarios and both fixed learning algorithms are reported; model superiority is not required for the completion-sensitivity contribution.",
            "Calibration bins are descriptive for the verified external subset, contain correlated repeated snapshots, and do not establish calibration in missing reference outcomes or beyond the observed campaign.",
        ],
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "output_rows": {name: len(table) for name, table in tables.items()}, "wall_seconds": time.perf_counter() - started,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "validation": validation, "outputs": manifest["output_rows"], "wall_seconds": manifest["wall_seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
