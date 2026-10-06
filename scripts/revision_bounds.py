"""Revision-stage shared-reference completion bounds and fixed-score diagnostics.

No original analysis files are modified. Only genuine source-scene outcomes can
be shared across decisions; no-future-scene records are independent score slots.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from pyproj import Transformer

SPEC = importlib.util.spec_from_file_location("revision_shared_evaluator", Path(__file__).with_name("evaluate_campaigns.py"))
EV = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EV)
PAIRS = [("logistic", "beta_rate"), ("beta_rate", "uniform_random"), ("hist_gradient_boosting", "beta_rate")]
MODELS = tuple(dict.fromkeys(x for pair in PAIRS for x in pair))
PRIMARY = "qaccepted_r250_h3_l24"
EPS = 1e-12
PROJ = Transformer.from_crs("EPSG:4326", "+proj=aeqd +lat_0=32 +lon_0=-103 +datum=WGS84 +units=m +no_defs", always_xy=True)
CONFIG = {"stage": "2026-10-06 exploratory revision; not part of original analysis plan",
    "scenario": PRIMARY, "evaluation_mode": "external_2023_to_2024", "campaign": "2024",
    "pairs": PAIRS, "budgets": EV.BUDGETS, "primary_budget": ["fraction", .25],
    "reference_group": "campaign/source_id/outcome_scene; no-scene slots include decision_at and are never shared across dates",
    "date_weights": "equal over available dates; all 14 primary dates retained",
    "mask": {"population": "originally known source-decision rows only; daily known-subset budget ceil(.25*Nknown)",
        "unit": "unique campaign/source/reference-scene outcome", "retention": .5, "retained_count": "ceil(.5*unique_outcomes)",
        "mechanism": "uniform exact-size without replacement; same group mask for all three policy pairs",
        "seed": 20261006, "repetitions": 200},
    "verification": {"unit": "one unique historical reference outcome perfectly resolved at equal cost",
        "ordering": "descending abs(aggregate macro coefficient Cg), then stable group ID; no labels used",
        "budgets": "every integer from zero to hidden unique outcome count",
        "width_targets": [.20, .10, .05], "direction_threshold": .02,
        "direction": "lower > .02 or upper < -.02; within-band completion reported separately",
        "uniform_comparator": "analytic expected remaining width only; no inferred direction probability"},
    "bootstrap": {"scales_m": [2500, 5000, 10000], "origin_fraction_both_axes": [0., .5],
        "replicates": 1000, "seed": 20261005, "budget": ["fraction", .25], "model_refitting": False,
        "interpretation": "conditional on fixed models and observed campaign; not all flight-day or training uncertainty"},
    "leave_one_out": "omit each of 14 decision days; recompute macro weights and grouped completion coefficients"}


def reference_groups(frame):
    keys, reasons = [], []
    for row in frame.itertuples():
        has_scene = pd.notna(row.outcome_scene) and str(row.outcome_scene).strip() != ""
        if has_scene:
            keys.append("reference:" + json.dumps([str(row.campaign), str(row.source_id), str(row.outcome_scene)], separators=(",", ":")))
            reasons.append("known" if pd.notna(row.outcome_label) else "ambiguous")
        else:
            if pd.notna(row.outcome_label):
                raise ValueError("Known reference label requires a real scene")
            keys.append("no_scene_slot:" + json.dumps([str(row.campaign), str(row.source_id), str(row.decision_at)], separators=(",", ":")))
            reasons.append("no_future_scene")
    return np.asarray(keys), np.asarray(reasons)


def grouped_bounds(coefficients, labels, groups):
    """Exact linear binary-completion bounds after grouping true shared outcomes."""
    c, y, groups = np.asarray(coefficients, float), np.asarray(labels, float), np.asarray(groups, str)
    if not (len(c) == len(y) == len(groups)) or not np.isfinite(c).all():
        raise ValueError("Finite aligned coefficients, labels, and group IDs required")
    if not np.isin(y[np.isfinite(y)], [0, 1]).all():
        raise ValueError("Known labels must be binary")
    ids, inverse = np.unique(groups, return_inverse=True)
    aggregate = np.bincount(inverse, weights=c, minlength=len(ids))
    group_y = np.full(len(ids), np.nan)
    for j in range(len(ids)):
        observed = y[(inverse == j) & np.isfinite(y)]
        if len(observed):
            if observed.min() != observed.max():
                raise ValueError("Conflicting known values for one shared reference outcome")
            group_y[j] = observed[0]
    known = np.isfinite(group_y)
    fixed = float(np.dot(aggregate[known], group_y[known]))
    lower = fixed + float(np.minimum(aggregate[~known], 0).sum())
    upper = fixed + float(np.maximum(aggregate[~known], 0).sum())
    row_known = np.isfinite(y)
    outer_known = float(np.dot(c[row_known], y[row_known]))
    outer_lower = outer_known + float(np.minimum(c[~row_known], 0).sum())
    outer_upper = outer_known + float(np.maximum(c[~row_known], 0).sum())
    if lower < outer_lower - 1e-10 or upper > outer_upper + 1e-10:
        raise AssertionError("Grouped bounds must be contained in row-wise outer bounds")
    return {"lower": lower, "upper": upper, "width": upper - lower,
        "known_contribution": fixed, "outer_lower": outer_lower, "outer_upper": outer_upper,
        "outer_width": outer_upper - outer_lower, "groups": ids, "coefficients": aggregate,
        "labels": group_y, "inverse": inverse}


def reveal_path(group_coefficients, group_labels, retained, ids, targets=(.20, .10, .05), tau=.02):
    c, y, retained = np.asarray(group_coefficients, float), np.asarray(group_labels, float), np.asarray(retained, bool)
    ids = np.asarray(ids, str)
    if not np.isin(y, [0, 1]).all() or not (len(c) == len(y) == len(retained) == len(ids)):
        raise ValueError("Revelation requires aligned completely known experimental group truths")
    unknown = np.flatnonzero(~retained)
    order = unknown[np.lexsort((ids[unknown], -np.abs(c[unknown])))]
    known_part = float(np.dot(c[retained], y[retained]))
    lower0 = known_part + np.minimum(c[unknown], 0).sum()
    upper0 = known_part + np.maximum(c[unknown], 0).sum()
    lower = lower0 + np.r_[0., np.cumsum(c[order] * y[order] - np.minimum(c[order], 0))]
    upper = upper0 + np.r_[0., np.cumsum(c[order] * y[order] - np.maximum(c[order], 0))]
    width = np.maximum(0., upper - lower)
    truth = float(np.dot(c, y))
    if (lower > truth + 1e-10).any() or (upper < truth - 1e-10).any():
        raise AssertionError("Truth escaped grouped revelation path")
    u = len(unknown)
    budgets = np.arange(u + 1)
    uniform_width = width[0] * (1 - budgets / u) if u else np.zeros(1)
    if (width > uniform_width + 1e-10).any():
        raise AssertionError("Descending coefficient revelation cannot lose on the width objective")
    direction = (lower > tau + EPS) | (upper < -tau - EPS)
    within_band = (lower >= -tau - EPS) & (upper <= tau + EPS)
    needed = {str(target): int(np.flatnonzero(width <= target + EPS)[0]) for target in targets}
    return {"budget": budgets, "lower": lower, "upper": upper, "width": width,
        "uniform_expected_width": uniform_width, "direction_decided": direction,
        "within_threshold_band": within_band, "truth": truth, "unknown_groups": u,
        "truth_exceeds_threshold": bool(abs(truth) > tau + EPS),
        "needed_for_direction": int(np.flatnonzero(direction)[0]) if direction.any() else None,
        "needed_for_width": needed, "reveal_order": order}


def build_comparison_rows(frame, budget_specs):
    daily, coefficients = [], []
    dates = sorted(frame.decision_at.unique())
    for kind, value in budget_specs:
        for decision, day in frame.groupby("decision_at", sort=True):
            y = day.outcome_label.to_numpy(float)
            known = np.isfinite(y)
            m = EV.effective_budget(kind, value, len(day))
            groups, reasons = reference_groups(day)
            w = {model: EV.tie_selection_weights(day[f"score_{model}"].to_numpy(float), m) for model in MODELS}
            for a, b in PAIRS:
                pair = a + "_vs_" + b
                result, c = EV.completion_bounds(y, w[a], w[b])
                conditional = np.nan
                if known.any():
                    mk = min(m, int(known.sum()))
                    ka = EV.tie_selection_weights(day.loc[known, f"score_{a}"].to_numpy(float), mk)
                    kb = EV.tie_selection_weights(day.loc[known, f"score_{b}"].to_numpy(float), mk)
                    conditional = float(np.dot(ka - kb, y[known]) / mk)
                independent_width = float((w[a][~known] + w[b][~known]).sum() / m)
                context = {"budget_kind": kind, "budget_value": value, "decision_at": decision, "pair": pair}
                row = {**context, "candidates": len(day), "known_count": int(known.sum()), "effective_budget": m,
                    "conditional_difference": conditional, "paired_lower": result["lower"], "paired_upper": result["upper"],
                    "paired_width": result["interval_width"], "independent_width": independent_width,
                    "pairing_width_reduction": independent_width - result["interval_width"],
                    "selection_overlap_fraction": float(np.minimum(w[a], w[b]).sum() / m),
                    "unknown_overlap_fraction": float(np.minimum(w[a][~known], w[b][~known]).sum() / m),
                    "unknown_selection_mass_a": float(w[a][~known].sum() / m),
                    "unknown_selection_mass_b": float(w[b][~known].sum() / m)}
                for reason in ["no_future_scene", "ambiguous"]:
                    chosen = reasons == reason
                    row[f"width_{reason}"] = float(np.abs(c[chosen]).sum())
                    row[f"count_{reason}"] = int(chosen.sum())
                if not np.isclose(row["width_no_future_scene"] + row["width_ambiguous"], row["paired_width"]):
                    raise AssertionError("Unknown-reason widths failed to reconcile")
                daily.append(row)
                for i, source in enumerate(day.source_id):
                    coefficients.append({**context, "source_id": source, "reference_group": groups[i], "reason": reasons[i],
                        "label": y[i], "coefficient": c[i] / len(dates), "coefficient_per_day": c[i],
                        "date_weight": 1 / len(dates), "policy_a_weight": w[a][i], "policy_b_weight": w[b][i]})
    return pd.DataFrame(daily), pd.DataFrame(coefficients)


def aggregate_real_bounds(daily, coefficients):
    summaries, group_rows, reasons = [], [], []
    for key, rows in coefficients.groupby(["budget_kind", "budget_value", "pair"], sort=True):
        kind, value, pair = key
        d = daily.loc[daily.budget_kind.eq(kind) & daily.budget_value.eq(value) & daily.pair.eq(pair)]
        result = grouped_bounds(rows.coefficient, rows.label, rows.reference_group)
        if not np.allclose([result["outer_lower"], result["outer_upper"]], [d.paired_lower.mean(), d.paired_upper.mean()]):
            raise AssertionError("Macro daily outer interval differs from rowwise completion")
        context = {"budget_kind": kind, "budget_value": value, "pair": pair}
        summary = {**context, "decision_days": len(d), "source_decision_rows": len(rows),
            "conditional_difference": float(d.conditional_difference.mean()),
            "macro_outer_lower": result["outer_lower"], "macro_outer_upper": result["outer_upper"],
            "macro_outer_width": result["outer_width"], "joint_lower": result["lower"], "joint_upper": result["upper"],
            "joint_width": result["width"], "across_dates_width_reduction": result["outer_width"] - result["width"],
            "independent_policy_width": float(d.independent_width.mean()),
            "within_day_pairing_width_reduction": float(d.pairing_width_reduction.mean()),
            "unknown_selection_mass_a": float(d.unknown_selection_mass_a.mean()),
            "unknown_selection_mass_b": float(d.unknown_selection_mass_b.mean()),
            "selection_overlap_fraction": float(d.selection_overlap_fraction.mean()),
            "unknown_overlap_fraction": float(d.unknown_overlap_fraction.mean()),
            "unique_reference_or_score_slots": len(result["groups"]),
            "unknown_unique_slots": int(np.isnan(result["labels"]).sum())}
        for reason in ["no_future_scene", "ambiguous"]:
            prefix = "no_scene_slot:" if reason == "no_future_scene" else "reference:"
            mask = np.isnan(result["labels"]) & np.char.startswith(result["groups"], prefix)
            width = float(np.abs(result["coefficients"][mask]).sum())
            summary["joint_width_" + reason] = width
            reasons.append({**context, "unknown_reason": reason, "source_decision_rows": int(rows.reason.eq(reason).sum()),
                "unique_slots": int(mask.sum()), "macro_outer_width": float(d["width_" + reason].mean()),
                "joint_width": width, "joint_width_fraction": width / result["width"] if result["width"] > EPS else np.nan})
        if not np.isclose(summary["joint_width_no_future_scene"] + summary["joint_width_ambiguous"], result["width"]):
            raise AssertionError("Grouped reason widths do not add up")
        if not np.isclose(summary["within_day_pairing_width_reduction"], 2 * summary["unknown_overlap_fraction"]):
            raise AssertionError("Paired width reduction must equal twice shared unknown selection mass")
        summaries.append(summary)
        group_size = np.bincount(result["inverse"], minlength=len(result["groups"]))
        for i, group in enumerate(result["groups"]):
            group_rows.append({**context, "reference_group": group, "macro_coefficient": result["coefficients"][i],
                "label": result["labels"][i], "source_decision_rows": int(group_size[i]),
                "shared_across_dates": bool(group_size[i] > 1), "no_scene_slot": group.startswith("no_scene_slot:")})
    return pd.DataFrame(summaries), pd.DataFrame(group_rows), pd.DataFrame(reasons)


def unique_group_mask_experiment(frame):
    known = frame.loc[frame.outcome_label.notna()].copy()
    daily, rows = build_comparison_rows(known, [("fraction", .25)])
    prepared = {}
    common_ids = None
    common_truth = None
    audit = []
    for pair, g in rows.groupby("pair", sort=True):
        result = grouped_bounds(g.coefficient, g.label, g.reference_group)
        if common_ids is None:
            common_ids, common_truth = result["groups"], result["labels"]
        if not np.array_equal(common_ids, result["groups"]) or not np.array_equal(common_truth, result["labels"]):
            raise AssertionError("All pairs must use the same reference-group population and truth")
        if np.isnan(result["labels"]).any():
            raise AssertionError("Mask experiment includes a genuinely unknown label")
        prepared[pair] = (g, result)
        count = np.bincount(result["inverse"])
        audit.append({"pair": pair, "known_source_decision_rows": len(g), "decision_days": g.decision_at.nunique(),
            "unique_reference_outcomes": len(common_ids), "reference_outcomes_reused_across_days": int((count > 1).sum()),
            "maximum_reuse": int(count.max()), "rows_saved_by_grouping": int(len(g) - len(common_ids)),
            "fixed_macro_truth": result["known_contribution"],
            "truth_exceeds_direction_threshold": abs(result["known_contribution"]) > CONFIG["verification"]["direction_threshold"] + EPS})
    total = len(common_ids)
    retained_count = math.ceil(.5 * total)
    unknown_count = total - retained_count
    accum = {p: {name: np.zeros(unknown_count + 1) for name in ["width", "uniform_expected_width", "direction_decided", "within_threshold_band", "lower", "upper"]} for p in prepared}
    rng = np.random.default_rng(CONFIG["mask"]["seed"])
    replicates = []
    for rep in range(CONFIG["mask"]["repetitions"]):
        retained = np.zeros(total, bool)
        retained[rng.choice(total, retained_count, replace=False)] = True
        for pair, (g, result) in prepared.items():
            path = reveal_path(result["coefficients"], result["labels"], retained, result["groups"],
                CONFIG["verification"]["width_targets"], CONFIG["verification"]["direction_threshold"])
            for metric in accum[pair]:
                accum[pair][metric] += path[metric]
            row_unknown = ~retained[result["inverse"]]
            outer_width = float(np.abs(g.coefficient.to_numpy()[row_unknown]).sum())
            record = {"pair": pair, "repetition": rep, "unique_outcomes": total, "retained_unique_outcomes": retained_count,
                "hidden_unique_outcomes": unknown_count, "true_macro_difference": path["truth"],
                "initial_joint_lower": path["lower"][0], "initial_joint_upper": path["upper"][0],
                "initial_joint_width": path["width"][0], "initial_macro_outer_width": outer_width,
                "across_dates_width_reduction": outer_width - path["width"][0],
                "truth_exceeds_direction_threshold": path["truth_exceeds_threshold"],
                "needed_unique_outcomes_for_direction": path["needed_for_direction"],
                "final_within_threshold_band": path["within_threshold_band"][-1],
                "all_path_truth_contained": True}
            for threshold, needed in path["needed_for_width"].items():
                record["needed_for_width_" + threshold] = needed
                w0 = path["width"][0]
                # This is where the expected-width curve crosses the target, not a random stopping-time expectation.
                record["uniform_expected_width_crossing_" + threshold] = 0 if w0 <= float(threshold) + EPS else int(math.ceil(unknown_count * (1 - float(threshold) / w0) - EPS))
            replicates.append(record)
    curves = []
    repetitions = CONFIG["mask"]["repetitions"]
    for pair, data in accum.items():
        truth = float(prepared[pair][1]["known_contribution"])
        eligible = abs(truth) > CONFIG["verification"]["direction_threshold"] + EPS
        for b in range(unknown_count + 1):
            record = {"pair": pair, "revealed_unique_outcomes": b, "initial_hidden_unique_outcomes": unknown_count,
                "revealed_hidden_fraction": b / unknown_count if unknown_count else 0., "repetitions": repetitions,
                "true_macro_difference": truth, "direction_eligible_repetitions": repetitions if eligible else 0}
            for metric, values in data.items():
                record["mean_" + metric] = float(values[b] / repetitions)
            record["direction_certified_count"] = int(data["direction_decided"][b])
            record["direction_certified_fraction_eligible"] = float(data["direction_decided"][b] / repetitions) if eligible else np.nan
            record["width_advantage_over_uniform_expected"] = record["mean_uniform_expected_width"] - record["mean_width"]
            curves.append(record)
    return daily, pd.DataFrame(audit), pd.DataFrame(replicates), pd.DataFrame(curves)


def mask_threshold_summary(replicates):
    rows = []
    for pair, g in replicates.groupby("pair", sort=True):
        for target in CONFIG["verification"]["width_targets"]:
            name = str(target)
            counts = g["needed_for_width_" + name]
            uniform = g["uniform_expected_width_crossing_" + name]
            rows.append({"pair": pair, "width_target": target, "replicates": len(g),
                "hidden_unique_outcomes": int(g.hidden_unique_outcomes.iloc[0]),
                "targeted_needed_mean": float(counts.mean()), "targeted_needed_median": float(counts.median()),
                "targeted_needed_p025": float(counts.quantile(.025)), "targeted_needed_p975": float(counts.quantile(.975)),
                "uniform_expected_width_crossing_mean": float(uniform.mean()),
                "crossing_count_difference_mean": float((uniform - counts).mean()),
                "direction_eligible_replicates": int(g.truth_exceeds_direction_threshold.sum()),
                "direction_reached_replicates": int(g.needed_unique_outcomes_for_direction.notna().sum()),
                "direction_needed_mean_when_reachable": float(g.needed_unique_outcomes_for_direction.mean())})
    return pd.DataFrame(rows)


def spatial_blocks(frame, scale, origin_fraction):
    x, y = PROJ.transform(frame.anchor_longitude.to_numpy(float), frame.anchor_latitude.to_numpy(float))
    bx = np.floor((x - origin_fraction * scale) / scale).astype(int)
    by = np.floor((y - origin_fraction * scale) / scale).astype(int)
    cells = np.array([f"{a}_{b}" for a, b in zip(bx, by)])
    if scale == 5000 and origin_fraction == 0 and not np.array_equal(cells, frame.cell_id.to_numpy(str)):
        raise AssertionError("Reconstructed primary grid differs from original cells")
    return cells


def run_spatial_sensitivity(frame):
    selected = frame.copy()
    for model in set(EV.MODELS) - set(MODELS):
        selected[f"score_{model}"] = np.nan
    intervals, audit = [], []
    for scale in CONFIG["bootstrap"]["scales_m"]:
        for origin in CONFIG["bootstrap"]["origin_fraction_both_axes"]:
            started = time.perf_counter()
            selected["cell_id"] = spatial_blocks(frame, scale, origin)
            values = EV.spatial_bootstrap(selected, reps=CONFIG["bootstrap"]["replicates"], seed=CONFIG["bootstrap"]["seed"], budget_specs=[("fraction", .25)])
            values = values.loc[[((a, b) in PAIRS) for a, b in zip(values.model, values.reference)]].copy()
            values["grid_scale_m"] = scale
            values["grid_origin_fraction_both_axes"] = origin
            intervals.append(values)
            detail = {"grid_scale_m": scale, "grid_origin_fraction_both_axes": origin,
                "all_candidate_blocks": int(selected.cell_id.nunique()),
                "blocks_with_known_outcomes": int(selected.loc[selected.outcome_label.notna(), "cell_id"].nunique()),
                "bootstrap_replicates": CONFIG["bootstrap"]["replicates"], "wall_seconds": time.perf_counter() - started}
            audit.append(detail)
            print(json.dumps({"stage": "spatial_grid_completed", **detail}), flush=True)
    return pd.concat(intervals, ignore_index=True), pd.DataFrame(audit)


def leave_one_day_out(daily, coefficients):
    d = daily.loc[daily.budget_kind.eq("fraction") & daily.budget_value.eq(.25)]
    c = coefficients.loc[coefficients.budget_kind.eq("fraction") & coefficients.budget_value.eq(.25)]
    rows = []
    for pair, values in d.groupby("pair", sort=True):
        all_c = c.loc[c.pair.eq(pair)]
        full = grouped_bounds(all_c.coefficient, all_c.label, all_c.reference_group)
        n = len(values)
        for omitted in sorted(values.decision_at.unique()):
            dd = values.loc[values.decision_at.ne(omitted)]
            cc = all_c.loc[all_c.decision_at.ne(omitted)]
            result = grouped_bounds(cc.coefficient * n / (n - 1), cc.label, cc.reference_group)
            rows.append({"pair": pair, "omitted_decision_at": omitted, "remaining_decision_days": len(dd),
                "conditional_difference": float(dd.conditional_difference.mean()), "full_conditional_difference": float(values.conditional_difference.mean()),
                "joint_lower": result["lower"], "joint_upper": result["upper"], "joint_width": result["width"],
                "macro_outer_lower": result["outer_lower"], "macro_outer_upper": result["outer_upper"],
                "full_joint_lower": full["lower"], "full_joint_upper": full["upper"]})
    detail = pd.DataFrame(rows)
    summary = detail.groupby("pair", as_index=False).agg(
        omitted_days=("omitted_decision_at", "count"), full_conditional_difference=("full_conditional_difference", "first"),
        conditional_difference_min=("conditional_difference", "min"), conditional_difference_max=("conditional_difference", "max"),
        joint_lower_min=("joint_lower", "min"), joint_lower_max=("joint_lower", "max"),
        joint_upper_min=("joint_upper", "min"), joint_upper_max=("joint_upper", "max"))
    return detail, summary


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_interpretation(tables, manifest):
    lines = ["# Shared-reference bounds and fixed-score revision diagnostics", "",
        "These are exploratory analyses added on 2026-10-06 in response to review, not preplanned or preregistered analyses.", "",
        "The full-candidate completion target is an unrestricted binary reference-score extension. A missing future scene is an independent source-decision score slot, not a recovered emission status. Shared labels are imposed only for the same campaign, source, and actual reference scene. Daily policies and scores remain frozen; no models are refit.", "",
        "## Real-data primary comparison", ""]
    primary = tables["joint_bounds_summary"].query("budget_kind == 'fraction' and budget_value == .25")
    for row in primary.itertuples():
        lines.append(f"- {row.pair}: conditional difference {row.conditional_difference:.6f}; joint [{row.joint_lower:.6f}, {row.joint_upper:.6f}], width {row.joint_width:.6f}. No-future-scene width {row.joint_width_no_future_scene:.6f}; ambiguous-scene width {row.joint_width_ambiguous:.6f}; across-date grouping reduces width by {row.across_dates_width_reduction:.9f}. Independent-policy width {row.independent_policy_width:.6f}; paired construction reduces it by {row.within_day_pairing_width_reduction:.6f}.")
    lines += ["", "Only reference-outcome consistency is imposed. Grouping can narrow the macro outer interval, but it is not guaranteed to do so. No real-data narrowing is invented when the repeated unknown groups do not have opposite-sign coefficients.", "",
        "## Controlled unique-reference masking", "",
        "This experiment starts from genuinely known outcomes and excludes the original unknown population. Policies are refrozen on each day's known subset with ceil(0.25 Nknown); the target is the equal-weight mean of all 14 daily fixed-policy differences. One exact half-group mask is shared across all policy pairs in a replicate, so a reused source-scene label is consistently hidden or retained. The experiment does not validate the actual missingness mechanism.", ""]
    for row in tables["mask_population_summary"].itertuples():
        lines.append(f"- {row.pair}: {row.known_source_decision_rows} rows, {row.unique_reference_outcomes} unique reference outcomes, {row.reference_outcomes_reused_across_days} reused outcomes; fixed macro truth {row.fixed_macro_truth:.6f}; exceeds the ±0.02 direction threshold: {row.truth_exceeds_direction_threshold}.")
    lines += ["", "Every integer revelation budget is retained in mask_reveal_curve_summary.csv. Direction certification means the entire completion interval is above +0.02 or below −0.02. The rate among all 200 masks and its eligible denominator are both reported; a true difference within the threshold band cannot be directionally certified even after complete revelation. A separate within-band indicator records intervals wholly inside [−0.02, +0.02]. Width target counts are in mask_threshold_summary.csv.", "",
        "Ordering uses only |Cg| and stable identifiers, never hidden labels. At equal cost with perfect revelation, this ordering minimizes remaining interval width; for b uniformly selected hidden groups the analytic expected width is W(1−b/U). Its threshold crossing is not the expected random stopping time. The uniform expected-width curve does not define a direction-certification probability, so none is imputed. These are offline historical-information diagnostics; a later flight cannot reveal a past emission state.", "",
        "## Conditional uncertainty and day sensitivity", "",
        "Six frozen grid definitions use 2.5, 5, and 10 km cells, each with zero and simultaneous half-cell x/y shifts. Each uses 1000 paired block bootstrap replicates; whole blocks carry all source histories. These percentile intervals condition on fixed scores, the observed campaign, and the grid. They do not include training uncertainty or all common flight-day errors. Grid sensitivity is reported without selecting the narrowest interval.", ""]
    precision = tables["spatial_bootstrap_summary"].query("metric == 'conditional_precision_difference'")
    for (a, b), rows in precision.groupby(["model", "reference"]):
        lines.append(f"- {a}_vs_{b}: point {rows.point_estimate.iloc[0]:.6f}; grid-dependent lower endpoints {rows.ci_lower.min():.6f} to {rows.ci_lower.max():.6f}, upper endpoints {rows.ci_upper.min():.6f} to {rows.ci_upper.max():.6f}.")
    for row in tables["leave_one_day_out_summary"].itertuples():
        lines.append(f"- {row.pair}: leave-one-day-out conditional difference spans {row.conditional_difference_min:.6f} to {row.conditional_difference_max:.6f} over all {row.omitted_days} exclusions. These exclusions are sensitivity summaries, not independent replications or confidence intervals.")
    lines += ["", "## Execution", "", f"Total wall time: {manifest['wall_seconds']:.2f} s; process CPU time: {manifest['cpu_seconds']:.2f} s. Configuration was written before data analyses. Source input and imported evaluator hashes were checked after execution. Enumeration and edge-case tests are in tests/test_revision_bounds.py. Algebraic containment is a correctness check, not field validity."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    started, cpu_started = time.perf_counter(), time.process_time()
    evaluator_path = Path(__file__).with_name("evaluate_campaigns.py")
    checksums = {"predictions": sha256(args.predictions), "script": sha256(Path(__file__)), "evaluator": sha256(evaluator_path)}
    frozen = {**CONFIG, "input_path": str(args.predictions.resolve()), "sha256": checksums,
        "frozen_before_data_analysis_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (output / "revision_config.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    full = pd.read_csv(args.predictions, dtype={"campaign": str, "source_id": str, "cell_id": str})
    frame = full.loc[full.scenario.eq(PRIMARY) & full.evaluation_mode.eq(CONFIG["evaluation_mode"]) & full.campaign.eq("2024")].copy()
    frame = frame.sort_values(["decision_at", "source_id"]).reset_index(drop=True)
    if frame.empty or frame.duplicated(["campaign", "source_id", "decision_at"]).any():
        raise ValueError("Empty or duplicate primary input")
    if frame.decision_at.nunique() != 14:
        raise ValueError("The frozen analysis expects all 14 primary decision days")
    if not np.isfinite(frame[["score_" + m for m in MODELS]].to_numpy()).all():
        raise ValueError("Every primary policy requires finite scores")
    tables = {}
    daily, coefficients = build_comparison_rows(frame, EV.BUDGETS)
    tables["per_day_bounds_summary"] = daily
    tables["coefficient_audit_summary"] = coefficients
    tables["joint_bounds_summary"], tables["reference_groups_summary"], tables["unknown_reason_width_summary"] = aggregate_real_bounds(daily, coefficients)
    print(json.dumps({"stage": "real_bounds_completed", "source_decision_rows": len(frame), "known_rows": int(frame.outcome_label.notna().sum())}), flush=True)
    (tables["mask_per_day_population_summary"], tables["mask_population_summary"], tables["mask_replicate_summary"], tables["mask_reveal_curve_summary"]) = unique_group_mask_experiment(frame)
    tables["mask_threshold_summary"] = mask_threshold_summary(tables["mask_replicate_summary"])
    print(json.dumps({"stage": "group_mask_completed", "comparisons": len(tables["mask_replicate_summary"])}), flush=True)
    tables["leave_one_day_out_detail_summary"], tables["leave_one_day_out_summary"] = leave_one_day_out(daily, coefficients)
    # Save completed analyses before the longer bootstrap; incomplete manifests never claim completion.
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    tables["spatial_bootstrap_summary"], tables["spatial_grid_audit_summary"] = run_spatial_sensitivity(frame)
    if sha256(args.predictions) != checksums["predictions"] or sha256(evaluator_path) != checksums["evaluator"] or sha256(Path(__file__)) != checksums["script"]:
        raise RuntimeError("An input or analysis source changed during execution")
    for name in ["spatial_bootstrap_summary", "spatial_grid_audit_summary"]:
        tables[name].to_csv(output / f"{name}.csv", index=False)
    manifest = {"status": "completed", "sha256": checksums, "primary_rows": len(frame), "decision_days": 14,
        "known_rows": int(frame.outcome_label.notna().sum()), "unknown_rows": int(frame.outcome_label.isna().sum()),
        "wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_started,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "logical_cpus": os.cpu_count(), "output_rows": {key: len(value) for key, value in tables.items()},
        "bounds_are_confidence_intervals": False, "models_refit": False, "new_analyses_are_exploratory_revision": True}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "interpretation.md").write_text(make_interpretation(tables, manifest), encoding="utf-8")
    print(json.dumps({"stage": "completed", "output_dir": str(output), "wall_seconds": manifest["wall_seconds"], "cpu_seconds": manifest["cpu_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
