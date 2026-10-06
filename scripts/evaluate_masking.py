"""Exploratory label-masking stress test, restricted to already resolved outcomes.

Frozen policies operate on each day's originally known subset. Synthetic masks
therefore test evaluation diagnostics, not the validity of original missingness
assumptions, retrospective reference labels, or a future flight's ability to
recover an earlier state. No estimators are fitted or tuned in this module.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("masking_campaign_evaluator", Path(__file__).with_name("evaluate_campaigns.py"))
EV = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EV)
PRIMARY = "qaccepted_r250_h3_l24"
MODE = "external_2023_to_2024"
SEED = 20261005
REPETITIONS = 200
RETENTION = (.25, .5, .75)
MECHANISMS = ("uniform", "history_weighted")
REVEAL_FRACTIONS = (0., .1, .25, .5, 1.)
PAIRS = (("logistic", "beta_rate"), ("hist_gradient_boosting", "beta_rate"), ("beta_rate", "uniform_random"))
MODELS = tuple(dict.fromkeys(model for pair in PAIRS for model in pair))
EPS = 1e-12


def make_mask(n, retention, mechanism, history, rng):
    """Label-free fixed-size sampling; ceil handles noninteger requested sizes."""
    h = np.asarray(history, dtype=float)
    if n < 1 or len(h) != n or not np.isfinite(h).all() or (h < 0).any():
        raise ValueError("Masking requires finite nonnegative aligned historical counts")
    if not 0 < retention <= 1:
        raise ValueError("Retention must be in (0, 1]")
    if mechanism not in MECHANISMS:
        raise ValueError("Unknown masking mechanism")
    retained_count = min(n, int(math.ceil(retention * n)))
    p = None if mechanism == "uniform" else (1 + h) / (1 + h).sum()
    retained = np.zeros(n, dtype=bool)
    retained[rng.choice(n, size=retained_count, replace=False, p=p)] = True
    return retained


def excludes_zero(lower, upper):
    return bool(lower > EPS or upper < -EPS)


def evaluate_pair(y, score_a, score_b, weight_a, weight_b, retained):
    """Compare two frozen policies with three evaluations of hidden labels."""
    y = np.asarray(y, dtype=float)
    retained = np.asarray(retained, dtype=bool)
    wa, wb = np.asarray(weight_a, float), np.asarray(weight_b, float)
    if not len(y) or not np.isin(y, [0., 1.]).all() or len(retained) != len(y) or not retained.any():
        raise ValueError("Experiment population must have binary complete outcomes and retained labels")
    masked = np.where(retained, y, np.nan)
    bounds, c = EV.completion_bounds(masked, wa, wb)
    budget = float(bounds["effective_budget"])
    if budget <= 0:
        raise ValueError("A positive frozen budget is required")
    truth = float(np.dot(c, y))
    conditional_budget = min(int(round(budget)), int(retained.sum()))
    cond_a = EV.tie_selection_weights(np.asarray(score_a)[retained], conditional_budget)
    cond_b = EV.tie_selection_weights(np.asarray(score_b)[retained], conditional_budget)
    conditional = float(np.dot(cond_a - cond_b, y[retained]) / conditional_budget)
    known_a = float(np.dot(wa[retained], y[retained]) / budget)
    known_b = float(np.dot(wb[retained], y[retained]) / budget)
    upper_a = known_a + float(wa[~retained].sum() / budget)
    upper_b = known_b + float(wb[~retained].sum() / budget)
    independent_lower, independent_upper = known_a - upper_b, upper_a - known_b
    lower, upper = bounds["lower"], bounds["upper"]
    paired_contains = lower - 1e-10 <= truth <= upper + 1e-10
    independent_contains = independent_lower - 1e-10 <= truth <= independent_upper + 1e-10
    if not paired_contains or not independent_contains:
        raise AssertionError("A frozen-policy truth escaped its completion interval")
    if bounds["interval_width"] > independent_upper - independent_lower + 1e-10:
        raise AssertionError("Paired completion interval wider than independent-policy interval")
    zero_filled = bounds["known_contribution"]
    sign_eligible = bool(abs(truth) >= .02 - EPS)
    return {
        "effective_budget": budget, "conditional_budget": conditional_budget,
        "true_difference": truth, "conditional_difference": conditional,
        "zero_filled_difference": zero_filled,
        "conditional_absolute_error": abs(conditional - truth),
        "zero_filled_absolute_error": abs(zero_filled - truth),
        "conditional_error": conditional - truth, "zero_filled_error": zero_filled - truth,
        "sign_eligible": sign_eligible,
        "conditional_sign_reversal": bool(sign_eligible and conditional * truth < -EPS),
        "zero_filled_sign_reversal": bool(sign_eligible and zero_filled * truth < -EPS),
        "conditional_zero_estimate": bool(sign_eligible and abs(conditional) <= EPS),
        "zero_filled_zero_estimate": bool(sign_eligible and abs(zero_filled) <= EPS),
        "paired_lower": lower, "paired_upper": upper, "paired_width": bounds["interval_width"],
        "independent_lower": independent_lower, "independent_upper": independent_upper,
        "independent_width": independent_upper - independent_lower,
        "paired_zero_excluded": excludes_zero(lower, upper),
        "independent_zero_excluded": excludes_zero(independent_lower, independent_upper),
        "paired_contains_truth": paired_contains, "independent_contains_truth": independent_contains,
    }, c


def verification_diagnostics(y, weight_a, weight_b, retained, source_ids):
    """Offline information acquisition; priority uses only coefficients and IDs."""
    y, retained = np.asarray(y, float), np.asarray(retained, bool)
    source_ids = np.asarray(source_ids, str)
    masked = np.where(retained, y, np.nan)
    initial, c = EV.completion_bounds(masked, weight_a, weight_b)
    truth = float(np.dot(c, y))
    unknown = np.flatnonzero(~retained)
    # Equal magnitudes are resolved by source ID, never by hidden labels.
    order = unknown[np.lexsort((source_ids[unknown], -np.abs(c[unknown])))]
    u = len(order)
    increments_lower = c[order] * y[order] - np.minimum(c[order], 0)
    increments_upper = c[order] * y[order] - np.maximum(c[order], 0)
    lower_path = initial["lower"] + np.r_[0., np.cumsum(increments_lower)]
    upper_path = initial["upper"] + np.r_[0., np.cumsum(increments_upper)]
    certified = (lower_path > EPS) | (upper_path < -EPS)
    needed = int(np.flatnonzero(certified)[0]) if certified.any() else None
    rows = []
    for fraction in REVEAL_FRACTIONS:
        b = min(u, int(math.ceil(fraction * u)))
        exposed = masked.copy()
        exposed[order[:b]] = y[order[:b]]
        result, _ = EV.completion_bounds(exposed, weight_a, weight_b)
        width = result["interval_width"]
        expected_random = initial["interval_width"] * (1 - b / u) if u else 0.
        contains = result["lower"] - 1e-10 <= truth <= result["upper"] + 1e-10
        if not contains or width > expected_random + 1e-10:
            raise AssertionError("Invalid information-acquisition diagnostic")
        rows.append({"reveal_fraction": fraction, "revealed_count": b, "unknown_count": u,
            "initial_width": initial["interval_width"], "targeted_remaining_width": width,
            "random_expected_remaining_width": expected_random,
            "targeted_width_reduction": initial["interval_width"] - width,
            "random_expected_width_reduction": initial["interval_width"] - expected_random,
            "width_reduction_advantage": expected_random - width,
            "paired_lower": result["lower"], "paired_upper": result["upper"],
            "zero_excluded": excludes_zero(result["lower"], result["upper"]), "contains_truth": contains})
    return rows, {"unknown_count": u, "initial_width": initial["interval_width"],
        "true_difference": truth, "needed_for_sign_certification": needed,
        "sign_certifiable": needed is not None,
        "needed_to_resolve_interval": int(np.count_nonzero(np.abs(c[order]) > EPS)),
        "initially_sign_certified": bool(certified[0])}


def select_primary(frame):
    required = {"scenario", "campaign", "evaluation_mode", "decision_at", "source_id", "outcome_label",
        "history_observation_count", *[f"score_{model}" for model in MODELS]}
    if required - set(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    selected = frame.loc[frame["scenario"].eq(PRIMARY) & frame["evaluation_mode"].eq(MODE)
        & frame["campaign"].astype(str).eq("2024")].copy()
    if selected.empty:
        raise ValueError("Primary external-2024 prediction subset is empty")
    selected["decision_at"] = pd.to_datetime(selected["decision_at"], utc=True, errors="raise")
    if selected[["decision_at", "source_id"]].isna().any().any():
        raise ValueError("Decision and source IDs must be present")
    selected["source_id"] = selected["source_id"].astype(str)
    if selected.duplicated(["decision_at", "source_id"]).any():
        raise ValueError("Duplicate source within primary decision")
    selected["outcome_label"] = pd.to_numeric(selected["outcome_label"], errors="raise")
    if not selected["outcome_label"].dropna().isin([0, 1]).all():
        raise ValueError("Reference outcomes must be 0/1/missing")
    for name in ["history_observation_count", *[f"score_{model}" for model in MODELS]]:
        selected[name] = pd.to_numeric(selected[name], errors="raise")
    history = selected["history_observation_count"].to_numpy(float)
    if not np.isfinite(history).all() or (history < 0).any():
        raise ValueError("Invalid historical observation counts")
    return selected.sort_values(["decision_at", "source_id"]).reset_index(drop=True)


def summarize_metrics(frame):
    rows = []
    for keys, group in frame.groupby(["pair", "mechanism", "retention"], sort=True):
        eligible = group.loc[group["sign_eligible"]]
        row = dict(zip(["pair", "mechanism", "retention"], keys))
        row.update(n_evaluations=len(group), n_decisions=group["decision_at"].nunique(),
            n_sign_eligible=len(eligible), n_sign_eligible_decisions=eligible["decision_at"].nunique(),
            mean_true_difference=group["true_difference"].mean(),
            conditional_mae=group["conditional_absolute_error"].mean(),
            zero_filled_mae=group["zero_filled_absolute_error"].mean(),
            conditional_bias=group["conditional_error"].mean(),
            zero_filled_bias=group["zero_filled_error"].mean(),
            paired_mean_width=group["paired_width"].mean(),
            independent_mean_width=group["independent_width"].mean(),
            paired_zero_excluded_count=int(group["paired_zero_excluded"].sum()),
            paired_zero_excluded_rate=group["paired_zero_excluded"].mean(),
            independent_zero_excluded_count=int(group["independent_zero_excluded"].sum()),
            independent_zero_excluded_rate=group["independent_zero_excluded"].mean(),
            paired_coverage_count=int(group["paired_contains_truth"].sum()),
            independent_coverage_count=int(group["independent_contains_truth"].sum()))
        for estimator in ["conditional", "zero_filled"]:
            row[f"{estimator}_sign_reversal_count"] = int(eligible[f"{estimator}_sign_reversal"].sum())
            row[f"{estimator}_sign_reversal_rate"] = eligible[f"{estimator}_sign_reversal"].mean()
            row[f"{estimator}_zero_estimate_count"] = int(eligible[f"{estimator}_zero_estimate"].sum())
            row[f"{estimator}_zero_estimate_rate"] = eligible[f"{estimator}_zero_estimate"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_verification(curves, needed):
    curve_rows, needed_rows = [], []
    for keys, group in curves.groupby(["pair", "mechanism", "reveal_fraction"], sort=True):
        row = dict(zip(["pair", "mechanism", "reveal_fraction"], keys))
        row.update(n_evaluations=len(group), n_decisions=group["decision_at"].nunique())
        for column in ["revealed_count", "unknown_count", "initial_width", "targeted_remaining_width",
                       "random_expected_remaining_width", "targeted_width_reduction", "random_expected_width_reduction",
                       "width_reduction_advantage", "zero_excluded", "contains_truth"]:
            row[f"mean_{column}"] = group[column].mean()
        curve_rows.append(row)
    for keys, group in needed.groupby(["pair", "mechanism"], sort=True):
        row = dict(zip(["pair", "mechanism"], keys))
        resolved = group.loc[group["sign_certifiable"]]
        row.update(n_evaluations=len(group), n_decisions=group["decision_at"].nunique(),
            n_sign_certifiable=len(resolved), n_initially_sign_certified=int(group["initially_sign_certified"].sum()),
            mean_needed_for_sign_certification=resolved["needed_for_sign_certification"].mean(),
            median_needed_for_sign_certification=resolved["needed_for_sign_certification"].median(),
            mean_needed_to_resolve_interval=group["needed_to_resolve_interval"].mean(),
            median_needed_to_resolve_interval=group["needed_to_resolve_interval"].median(),
            mean_unknown_count=group["unknown_count"].mean())
        needed_rows.append(row)
    return pd.DataFrame(curve_rows), pd.DataFrame(needed_rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(prediction_path, output):
    started, cpu_started = time.perf_counter(), time.process_time()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "status": "exploratory stress test; fixed configuration saved before evaluation",
        "prediction_path": str(Path(prediction_path).resolve()), "scenario": PRIMARY, "evaluation_mode": MODE,
        "campaign": "2024", "seed": SEED, "repetitions": REPETITIONS,
        "retention_fractions": RETENTION, "retained_count_rule": "ceil(retention * N_originally_known)",
        "masking_mechanisms": MECHANISMS, "history_weight": "1 + history_observation_count",
        "budget_rule": "ceil(0.25 * N_originally_known), frozen before masking",
        "policy_pairs": PAIRS, "tie_convention": "fractional label-independent expected selection weights from shared evaluator",
        "conditional_budget_rule": "min(frozen budget, retained count); rerank retained subset with frozen scores",
        "verification_retention": .5, "verification_reveal_fractions": REVEAL_FRACTIONS,
        "verification_priority": "descending abs((wA-wB)/budget); source_id breaks equal magnitudes",
        "random_width_comparator": "exact expected width under equal-probability b-without-replacement revelations",
        "sign_reversal_eligibility": "abs(full restricted-population difference) >= 0.02; strict opposite signs",
        "aggregate_weight": "equal weight per decision and repetition within each policy pair/mechanism/retention",
        "inference": "No confidence intervals or site-population inference from repeated synthetic masks",
        "limits": "Already-resolved outcome population only; synthetic missingness does not validate real selective verification; no future-flight recovery claim",
    }
    (output / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    primary = select_primary(pd.read_csv(prediction_path, dtype={"campaign": str, "source_id": str}))
    metrics, curves, needed, audits, weights = [], [], [], [], []
    for day_index, (decision, all_day) in enumerate(primary.groupby("decision_at", sort=True)):
        day = all_day.loc[all_day["outcome_label"].notna()].copy()
        n = len(day)
        audit_base = {"decision_at": decision.isoformat(), "original_candidate_count": len(all_day),
            "original_known_count": n, "original_unknown_excluded_count": len(all_day) - n}
        if not n:
            audits.append({**audit_base, "pair": "all", "status": "skipped", "reason": "no_originally_known_outcomes"})
            continue
        k = int(math.ceil(.25 * n))
        y = day["outcome_label"].to_numpy(float)
        h = day["history_observation_count"].to_numpy(float)
        ids = day["source_id"].to_numpy(str)
        scores = {model: day[f"score_{model}"].to_numpy(float) for model in MODELS}
        available = {model: np.isfinite(values).all() for model, values in scores.items()}
        frozen = {model: EV.tie_selection_weights(scores[model], k) for model in MODELS if available[model]}
        active = []
        for a, b in PAIRS:
            name = f"{a}_vs_{b}"
            valid = available[a] and available[b]
            audits.append({**audit_base, "pair": name, "frozen_budget": k,
                "status": "included" if valid else "skipped", "reason": "" if valid else "nonfinite_policy_scores"})
            if valid:
                active.append((a, b, name))
        for model in frozen:
            for i in range(n):
                weights.append({"decision_at": decision.isoformat(), "source_id": ids[i], "policy": model,
                    "outcome_label": y[i], "score": scores[model][i], "frozen_weight": frozen[model][i], "budget": k})
        for mechanism_index, mechanism in enumerate(MECHANISMS):
            for retention_index, retention in enumerate(RETENTION):
                for repetition in range(REPETITIONS):
                    rng = np.random.default_rng(np.random.SeedSequence([SEED, day_index, mechanism_index, retention_index, repetition]))
                    retained = make_mask(n, retention, mechanism, h, rng)
                    context = {"decision_at": decision.isoformat(), "mechanism": mechanism, "retention": retention,
                        "repetition": repetition, "n_complete_population": n, "n_retained": int(retained.sum()),
                        "n_masked": int((~retained).sum())}
                    for a, b, name in active:
                        row, _ = evaluate_pair(y, scores[a], scores[b], frozen[a], frozen[b], retained)
                        metrics.append({**context, "pair": name, **row})
                        if retention == .5:
                            curve, requirement = verification_diagnostics(y, frozen[a], frozen[b], retained, ids)
                            curves.extend({**context, "pair": name, **point} for point in curve)
                            needed.append({**context, "pair": name, **requirement})
    if not metrics:
        raise ValueError("No evaluable policy-day pairs")
    metrics, curves, needed = pd.DataFrame(metrics), pd.DataFrame(curves), pd.DataFrame(needed)
    audit, frozen_weights = pd.DataFrame(audits), pd.DataFrame(weights)
    summary = summarize_metrics(metrics)
    curve_summary, needed_summary = summarize_verification(curves, needed)
    for name, frame in [("replicate_metrics", metrics), ("summary", summary), ("verification_curves", curves),
                        ("verification_summary", curve_summary), ("verification_needed", needed),
                        ("verification_needed_summary", needed_summary), ("population_audit", audit),
                        ("frozen_policy_weights", frozen_weights)]:
        frame.to_csv(output / f"{name}.csv", index=False)
    machine_seconds = time.perf_counter() - started
    cpu_seconds = time.process_time() - cpu_started
    manifest = {"configuration": config, "input_sha256": sha256(prediction_path),
        "script_sha256": sha256(__file__), "shared_evaluator_sha256": sha256(Path(__file__).with_name("evaluate_campaigns.py")),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "machine_wall_seconds": machine_seconds, "process_cpu_seconds": cpu_seconds,
        "primary_rows": len(primary), "primary_decisions": primary["decision_at"].nunique(),
        "originally_known_rows": int(primary["outcome_label"].notna().sum()),
        "originally_unknown_rows_excluded": int(primary["outcome_label"].isna().sum()),
        "replicate_metric_rows": len(metrics), "verification_curve_rows": len(curves),
        "paired_containment_failures": int((~metrics["paired_contains_truth"]).sum()),
        "independent_containment_failures": int((~metrics["independent_contains_truth"]).sum()),
        "verification_containment_failures": int((~curves["contains_truth"]).sum()),
        "skipped_policy_decisions": int(audit["status"].eq("skipped").sum())}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_interpretation(output, summary, curve_summary, manifest)
    return manifest


def write_interpretation(output, summary, curve_summary, manifest):
    lines = ["# Exploratory synthetic masking validation", "",
        "This stress test freezes scores, selection probabilities, and a 25% budget on each decision's originally resolved reference outcomes. It hides labels only after defining that complete experimental population. It is an evaluation stress test, not a new training or model-selection exercise.", "",
        f"The input contains {manifest['primary_decisions']} primary external-2024 decisions, {manifest['originally_known_rows']} originally resolved source-decision rows, and {manifest['originally_unknown_rows_excluded']} originally unknown rows excluded from this particular experiment. Each of two label-free masking mechanisms uses three retention levels and 200 fixed-seed repetitions. Repeated source-decision rows and masks are not independent field observations.", "",
        "All reported errors are differences in expected detection precision, expressed in absolute units below. Conditional evaluation reranks the retained subset with the original scores and caps the budget by its size; zero filling retains the original policy weights. A sign reversal requires a true restricted-population difference of at least 0.02 in magnitude; an estimated zero is reported separately. Means give each included decision and repetition equal weight. No significance tests are attached to mask repetitions.", "",
        "## Main 50% retention diagnostics", "",
        "| Pair | Mask | Evaluations | Conditional MAE | Zero-filled MAE | Paired width | Independent width | Paired zero exclusion | Sign denominator | Conditional reversals |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary.loc[summary.retention.eq(.5)].itertuples():
        lines.append(f"| {row.pair} | {row.mechanism} | {row.n_evaluations} | {row.conditional_mae:.4f} | {row.zero_filled_mae:.4f} | {row.paired_mean_width:.4f} | {row.independent_mean_width:.4f} | {row.paired_zero_excluded_rate:.3f} | {row.n_sign_eligible} | {row.conditional_sign_reversal_rate:.3f} |")
    lines += ["", "All three retention levels, zero-filled sign reversals, zero estimates, exact certification counts, and their denominators are available in summary.csv. The paired bounds contain every tested complete truth by construction; successful containment validates implementation consistency only. It does not validate scene qualification, source matching, synthetic missingness realism, or unseen field labels.", "",
        "## Offline verification-demand diagnostic", "",
        "For the 50% retained-label experiment, the following rows reveal ceil(25% of hidden labels). The priority is descending absolute policy-weight difference with source-ID ties; outcomes do not set the priority. Uniform revelation has an exact expected remaining width equal to initial width times (1 - b/U). This is an equal-cost information-acquisition comparison on synthetic hidden past labels. A later flight cannot recover the past emission state, and this diagnostic makes no such operational claim.", "",
        "| Pair | Mask | Mean revealed count | Targeted remaining width | Uniform expected remaining width | Width reduction advantage |",
        "|---|---|---:|---:|---:|---:|"]
    for row in curve_summary.loc[curve_summary.reveal_fraction.eq(.25)].itertuples():
        lines.append(f"| {row.pair} | {row.mechanism} | {row.mean_revealed_count:.2f} | {row.mean_targeted_remaining_width:.4f} | {row.mean_random_expected_remaining_width:.4f} | {row.mean_width_reduction_advantage:.4f} |")
    lines += ["", "The full curves include 0%, 10%, 25%, 50%, and 100% revelation budgets. verification_needed_summary.csv reports the numbers of revealed labels required to resolve the interval and to certify a sign; the latter mean is conditional on sign certification being possible, with its denominator explicit. Sign certification depends on the revealed values, whereas width reduction depends only on policy disagreement. Both are retrospective information requirements, not estimates of future survey yield.", "",
        "The originally resolved population was selected by the real observation and data-release process. These two synthetic masks do not span all realistic missingness mechanisms and cannot establish population transportability. Uniform-random policy results use exact fractional expected selection weights rather than a sampled random ordering. Bounds are finite-population completion intervals, not confidence intervals or bounds on physical methane emissions.", "",
        f"Machine execution: {manifest['machine_wall_seconds']:.2f} wall-clock seconds and {manifest['process_cpu_seconds']:.2f} process CPU seconds. There were {manifest['paired_containment_failures']} paired, {manifest['independent_containment_failures']} independent, and {manifest['verification_containment_failures']} revelation-path containment failures. No setting was selected based on which mask produced the largest effect.", ""]
    (output / "interpretation.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.output or Path(__file__).resolve().parents[1] / "results" / ("masking_validation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    result = run(args.predictions, directory)
    print(json.dumps({"output": str(directory), "wall_seconds": result["machine_wall_seconds"],
        "metric_rows": result["replicate_metric_rows"], "paired_containment_failures": result["paired_containment_failures"]}, indent=2))
