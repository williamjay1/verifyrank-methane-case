"""Revision-stage exploratory masking diagnostics, with frozen policy targets.

Only reads prepared D-drive scores. Does not train models, alter old experiments,
or assign labels to the originally unknown reference population.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("revision_legacy_masking", Path(__file__).with_name("evaluate_masking.py"))
OLD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OLD)
EV = OLD.EV
EPS = 1e-12
GROUP_SEED = 20261006
GROUP_MECHANISMS = ("unique_reference", "scene_group", "observation_day_group")
POINT_METHODS = ("rerank_original_count", "rerank_original_fraction", "frozen_verified_conditional", "uniform_ht", "zero_filled")
BINS = (0., .02, .05, .10, .20, float("inf"))
BIN_NAMES = ("[0,0.02)", "[0.02,0.05)", "[0.05,0.10)", "[0.10,0.20)", "[0.20,inf)")


def file_hash(path):
    return OLD.sha256(path)


def configuration(predictions):
    return {
        "analysis_status": "revision-stage exploratory; not preregistered; fixed before revision result generation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_path": str(Path(predictions).resolve()), "prediction_sha256": file_hash(predictions),
        "scenario": OLD.PRIMARY, "evaluation_mode": OLD.MODE, "campaign": "2024",
        "experimental_population": "Each decision's originally known reference outcomes only; originally unknown outcomes excluded",
        "frozen_target": "Daily expected precision difference at ceil(0.25*N_originally_known), tie probabilities fixed before masking",
        "repetitions": 200, "legacy_seed": OLD.SEED, "group_seed": GROUP_SEED,
        "legacy_mechanisms": list(OLD.MECHANISMS), "legacy_retention": [.25, .5, .75],
        "legacy_seed_sequence": "[20261005, sorted_decision_index, mechanism_index, retention_index, repetition]",
        "group_mechanisms": list(GROUP_MECHANISMS), "group_retention": [.5],
        "group_seed_sequence": "[20261006, 900, group_mechanism_index, repetition]",
        "group_design": "Sample ceil(0.5*G) unique groups uniformly without replacement across the full activity; project one shared mask to all decisions",
        "reference_key": "campaign|source_id|outcome_scene; not scene alone",
        "scene_key": "campaign|outcome_scene", "observation_day_key": "campaign|UTC outcome acquisition date",
        "methods": list(POINT_METHODS) + ["paired_completion_bounds"],
        "method_estimands": {
            "rerank_original_count": "Rerank retained rows at min(original frozen count, retained N); changes list, population, and selection fraction",
            "rerank_original_fraction": "Rerank retained rows at ceil(0.25*retained N); changes list and population, preserves nominal fraction",
            "frozen_verified_conditional": "Keep original weights; each policy divides observed weighted positives by its own observed selected mass; pair undefined if either mass is zero",
            "uniform_ht": "sum(R*c*y/pi), pi=ceil(retention*N)/N for original uniform fixed-size row sampling ONLY; not clipped; unavailable for weighted/group masks",
            "zero_filled": "Keep original weights and budget; zero-fill hidden labels, a biased plug-in diagnostic",
            "paired_completion_bounds": "Frozen coefficients; exact daily unrestricted binary completion interval, not a CI or an estimator"},
        "effect_magnitude_bin_edges": [0., .02, .05, .10, .20, "inf"],
        "effect_magnitude_bin_closure": "left inclusive, right exclusive; based on absolute complete frozen-target difference",
        "sign_threshold": .02, "tie_tolerance": EPS,
        "summary": "All defined method estimates and a common-defined subset across the four non-HT point methods; equal date-mask weights, plus explicit date-macro quantities when undefined estimates change denominators",
        "inference": "No confidence intervals from mask repetitions; report unique decision dates and reference groups, not independent event counts",
        "planned_matrix": "14 dates x (2 row mechanisms x 3 retentions + 3 group mechanisms x 1 retention) x 200 masks x 3 policy pairs = 75,600 pair-masks; five point estimates plus bounds",
        "resources": "CPU only, no training, no API; prepared input about 51k rows; expected output under 0.5GB and minutes rather than a model sweep",
        "expectations": "Uniform HT is design-unbiased but may have high variance; 25%-retention original-count reranking must tie; no directional performance hypothesis for other methods",
        "limits": "Grouped synthetic masks are structural sensitivity analyses, not recovered survey missingness. Only 10 observed acquisition days are available in the primary known population."}


def freeze_config(predictions, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "experiment_config.json"
    if target.exists():
        raise FileExistsError("Frozen configuration already exists; do not silently replace it")
    config = configuration(predictions)
    target.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def add_reference_keys(known):
    result = known.copy()
    for col in ("campaign", "source_id", "outcome_scene", "outcome_observed_at"):
        if col not in result or result[col].isna().any():
            raise ValueError(f"Known reference requires {col}")
    result["reference_key"] = result.campaign.astype(str) + "|" + result.source_id.astype(str) + "|" + result.outcome_scene.astype(str)
    result["scene_key"] = result.campaign.astype(str) + "|" + result.outcome_scene.astype(str)
    dates = pd.to_datetime(result.outcome_observed_at, utc=True, errors="raise").dt.strftime("%Y-%m-%d")
    result["observation_day_key"] = result.campaign.astype(str) + "|" + dates
    checks = result.groupby("reference_key").agg(labels=("outcome_label", "nunique"), observed_at=("outcome_observed_at", "nunique"))
    if checks.labels.gt(1).any() or checks.observed_at.gt(1).any():
        raise ValueError("Shared reference key has inconsistent labels or acquisition times")
    return result.reset_index(drop=True)


def group_mask(keys, retention, rng):
    keys = np.asarray(keys, dtype=str)
    groups = np.unique(keys)
    if not len(groups) or not 0 < retention <= 1:
        raise ValueError("Groups and positive retention required")
    k = min(len(groups), int(math.ceil(retention * len(groups))))
    selected = groups[rng.choice(len(groups), size=k, replace=False)]
    return np.isin(keys, selected), selected


def magnitude_bin(truth):
    index = int(np.searchsorted(np.asarray(BINS), abs(truth), side="right") - 1)
    return BIN_NAMES[min(index, len(BIN_NAMES) - 1)]


def ht_difference(y, wa, wb, retained, pi):
    y, wa, wb, retained = np.asarray(y, float), np.asarray(wa, float), np.asarray(wb, float), np.asarray(retained, bool)
    if not 0 < pi <= 1 or not np.isclose(wa.sum(), wb.sum()) or wa.sum() <= 0:
        raise ValueError("Positive design probability and equal positive budgets required")
    c = (wa - wb) / wa.sum()
    return float(np.dot(c[retained], y[retained]) / pi)


def evaluate_pair(y, sa, sb, wa, wb, retained, mechanism):
    y, sa, sb = np.asarray(y, float), np.asarray(sa, float), np.asarray(sb, float)
    wa, wb, retained = np.asarray(wa, float), np.asarray(wb, float), np.asarray(retained, bool)
    if len({len(x) for x in (y, sa, sb, wa, wb, retained)}) != 1 or not len(y):
        raise ValueError("Aligned nonempty arrays required")
    if not np.isin(y, [0., 1.]).all() or not np.isfinite(sa).all() or not np.isfinite(sb).all():
        raise ValueError("Complete binary truth and finite frozen scores required")
    bounds, c = EV.completion_bounds(np.where(retained, y, np.nan), wa, wb)
    budget = float(bounds["effective_budget"])
    truth, n_kept = float(np.dot(c, y)), int(retained.sum())
    masses = (float(wa[retained].sum()), float(wb[retained].sum()))
    num_a, num_b = float(np.dot(wa[retained], y[retained])), float(np.dot(wb[retained], y[retained]))
    count_k = min(int(round(budget)), n_kept)
    fraction_k = int(math.ceil(.25 * n_kept)) if n_kept else 0
    estimates = {}
    for name, k in (("rerank_original_count", count_k), ("rerank_original_fraction", fraction_k)):
        estimates[name] = float(np.dot(EV.tie_selection_weights(sa[retained], k) - EV.tie_selection_weights(sb[retained], k), y[retained]) / k) if k else np.nan
    estimates["frozen_verified_conditional"] = num_a / masses[0] - num_b / masses[1] if min(masses) > EPS else np.nan
    pi = n_kept / len(y) if mechanism == "uniform" else np.nan
    estimates["uniform_ht"] = ht_difference(y, wa, wb, retained, pi) if mechanism == "uniform" and pi > 0 else np.nan
    estimates["zero_filled"] = float(bounds["known_contribution"])
    common_defined = all(np.isfinite(estimates[m]) for m in POINT_METHODS if m != "uniform_ht")
    common = {"true_difference": truth, "effect_bin": magnitude_bin(truth), "sign_eligible": abs(truth) >= .02 - EPS,
              "frozen_budget": budget, "retained_count_budget": count_k, "retained_fraction_budget": fraction_k,
              "verified_selected_mass_a": masses[0], "verified_selected_mass_b": masses[1], "uniform_inclusion_probability": pi,
              "common_point_defined": common_defined}
    rows = []
    for method in POINT_METHODS:
        value = estimates[method]
        defined = bool(np.isfinite(value))
        reason = "" if defined else ("not_uniform_fixed_size_row_design" if method == "uniform_ht" and mechanism != "uniform" else "zero_verified_selected_mass" if method == "frozen_verified_conditional" else "no_retained_outcomes")
        rows.append({**common, "method": method, "estimate": value, "defined": defined, "undefined_reason": reason,
                     "error": value - truth, "absolute_error": abs(value - truth),
                     "sign_reversal": bool(defined and value * truth < -EPS),
                     "estimated_tie": bool(defined and abs(value) <= EPS)})
    paired = {**common, "paired_lower": bounds["lower"], "paired_upper": bounds["upper"], "paired_width": bounds["interval_width"],
              "independent_width": float((wa[~retained].sum() + wb[~retained].sum()) / budget),
              "contains_truth": bool(bounds["lower"] - 1e-10 <= truth <= bounds["upper"] + 1e-10),
              "excludes_zero": bool(bounds["lower"] > EPS or bounds["upper"] < -EPS)}
    if not paired["contains_truth"]:
        raise AssertionError("Completion interval missed the frozen truth")
    return rows, paired


def summarize_methods(frame, extra=()):
    fields = ["pair", "mechanism", "nominal_retention", "method", *extra]
    records = []
    for key, g in frame.groupby(fields, sort=True, observed=True):
        for scope in ("method_defined", "common_point_defined"):
            valid = g.defined & (g.common_point_defined if scope == "common_point_defined" else True)
            v = g[valid]
            sign = v[v.sign_eligible]
            byday = v.groupby("decision_at", observed=True)
            eligible_byday = sign.groupby("decision_at", observed=True)
            row = dict(zip(fields, key))
            row.update(summary_population=scope, n_total_date_masks=len(g), n_defined=int(valid.sum()),
                n_method_undefined=int((~g.defined).sum()), n_removed_by_common_requirement=int((g.defined & ~valid).sum()),
                unique_decision_dates_total=int(g.decision_at.nunique()), unique_decision_dates_defined=int(v.decision_at.nunique()),
                n_sign_eligible_total=int(g.sign_eligible.sum()), n_sign_eligible_defined=len(sign), unique_sign_eligible_dates=int(sign.decision_at.nunique()),
                mean_true_difference=float(g.true_difference.mean()), mean_absolute_truth=float(g.true_difference.abs().mean()),
                mean_estimate=v.estimate.mean(), mae=v.absolute_error.mean(), signed_error=v.error.mean(), rmse=float(np.sqrt((v.error ** 2).mean())) if len(v) else np.nan,
                sign_reversal_count=int(sign.sign_reversal.sum()), sign_reversal_rate=sign.sign_reversal.mean(),
                eligible_tie_count=int(sign.estimated_tie.sum()), eligible_tie_rate=sign.estimated_tie.mean(),
                all_nonzero_truth_reversal_rate=v.loc[v.true_difference.abs().gt(EPS), "sign_reversal"].mean(), all_defined_tie_rate=v.estimated_tie.mean(),
                date_macro_mae=byday.absolute_error.mean().mean(), date_macro_signed_error=byday.error.mean().mean(),
                date_macro_sign_reversal_rate=eligible_byday.sign_reversal.mean().mean(),
                mean_actual_row_retention=g.actual_row_retention.mean(), min_actual_row_retention=g.actual_row_retention.min(), max_actual_row_retention=g.actual_row_retention.max())
            records.append(row)
    return pd.DataFrame(records)


def summarize_bounds(frame):
    keys = ["pair", "mechanism", "nominal_retention"]
    return frame.groupby(keys, observed=True).agg(n_date_masks=("contains_truth", "size"), unique_decision_dates=("decision_at", "nunique"),
        mean_lower=("paired_lower", "mean"), mean_upper=("paired_upper", "mean"), mean_width=("paired_width", "mean"),
        mean_independent_width=("independent_width", "mean"), containment_count=("contains_truth", "sum"),
        zero_exclusion_count=("excludes_zero", "sum"), zero_exclusion_rate=("excludes_zero", "mean")).reset_index()


def peak_memory():
    """Windows OS process lifetime peak working set, including Python imports."""
    if platform.system() == "Windows":
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        *[(name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        api = ctypes.windll.psapi.GetProcessMemoryInfo
        api.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if api(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return {"peak_working_set_bytes": int(counters.PeakWorkingSetSize), "peak_committed_bytes": int(counters.PeakPagefileUsage), "memory_method": "Windows GetProcessMemoryInfo; peak since process start; psutil unavailable"}
    return {"peak_working_set_bytes": None, "memory_method": "unavailable"}


def check_legacy(frame, old_path):
    old = pd.read_csv(old_path)
    new = frame[frame.method.eq("rerank_original_count") & frame.mechanism.isin(OLD.MECHANISMS)].copy()
    new = new.rename(columns={"nominal_retention": "retention"})
    for part in (old, new):
        part["decision_at"] = pd.to_datetime(part.decision_at, utc=True)
    keys = ["decision_at", "mechanism", "retention", "repetition", "pair"]
    joined = old.merge(new, on=keys, validate="one_to_one", suffixes=("_old", "_new"))
    if len(joined) != len(old) or len(new) != len(old):
        raise AssertionError("Legacy reproduction join is incomplete")
    difference = np.abs(joined.conditional_difference - joined.estimate)
    if difference.max() > 1e-12:
        raise AssertionError("Legacy original-count diagnostic changed")
    rows = []
    for mechanism in OLD.MECHANISMS:
        group = new[new.pair.eq("logistic_vs_beta_rate") & new.mechanism.eq(mechanism) & new.retention.eq(.5) & new.sign_eligible]
        rows.append({"mechanism": mechanism, "eligible": len(group), "reversals": int(group.sign_reversal.sum()), "rate": float(group.sign_reversal.mean())})
    if rows[0]["reversals"] != 835 or rows[1]["reversals"] != 751 or any(x["eligible"] != 2000 for x in rows):
        raise AssertionError("Expected legacy reversal counts not reproduced")
    return {"matched_rows": len(joined), "maximum_estimate_difference": float(difference.max()), "logistic_half_retention": rows}


def run(predictions, output):
    start, cpu = time.perf_counter(), time.process_time()
    output = Path(output)
    config = json.loads((output / "experiment_config.json").read_text(encoding="utf-8"))
    if config["prediction_sha256"] != file_hash(predictions):
        raise ValueError("Input changed after configuration freeze")
    if (output / "manifest.json").exists() or (output / "method_replicates.csv").exists():
        raise FileExistsError("Completed or partial results already exist; do not overwrite")
    selected = OLD.select_primary(pd.read_csv(predictions, dtype={"campaign": str, "source_id": str}))
    known = add_reference_keys(selected[selected.outcome_label.notna()].copy())
    inventory = []
    column_for = dict(zip(GROUP_MECHANISMS, ("reference_key", "scene_key", "observation_day_key")))
    for mechanism, column in column_for.items():
        for key, g in known.groupby(column, sort=True):
            inventory.append({"mechanism": mechanism, "group_key": key, "source_decision_rows": len(g), "sources": g.source_id.nunique(),
                "unique_references": g.reference_key.nunique(), "decision_dates": g.decision_at.nunique(), "reference_scenes": g.scene_key.nunique(), "observation_days": g.observation_day_key.nunique()})
    pd.DataFrame(inventory).to_csv(output / "group_inventory.csv", index=False)
    known[["campaign", "decision_at", "source_id", "reference_key", "scene_key", "observation_day_key", "outcome_label"]].to_csv(output / "reference_membership.csv", index=False)
    group_masks, group_audit = {}, []
    for mechanism_index, mechanism in enumerate(GROUP_MECHANISMS):
        keys = known[column_for[mechanism]].to_numpy(str)
        for repetition in range(config["repetitions"]):
            mask, groups = group_mask(keys, .5, np.random.default_rng(np.random.SeedSequence([GROUP_SEED, 900, mechanism_index, repetition])))
            group_masks[(mechanism, repetition)] = mask
            group_audit.append({"mechanism": mechanism, "repetition": repetition, "nominal_group_retention": .5,
                "total_groups": len(np.unique(keys)), "retained_groups": len(groups), "retained_rows": int(mask.sum()), "total_rows": len(mask),
                "actual_row_retention": float(mask.mean()), "selected_groups_sha256": hashlib.sha256("\n".join(sorted(groups)).encode()).hexdigest()})
    pd.DataFrame(group_audit).to_csv(output / "group_mask_audit.csv", index=False)
    rows, intervals, day_audit, retentions = [], [], [], []
    for day_index, (decision, day) in enumerate(known.groupby("decision_at", sort=True)):
        n = len(day)
        y, history = day.outcome_label.to_numpy(float), day.history_observation_count.to_numpy(float)
        frozen_k = int(math.ceil(.25 * n))
        scores = {model: day[f"score_{model}"].to_numpy(float) for model in OLD.MODELS}
        if not all(np.isfinite(value).all() for value in scores.values()):
            raise ValueError("Frozen primary policies have unavailable scores")
        weights = {model: EV.tie_selection_weights(score, frozen_k) for model, score in scores.items()}
        day_audit.append({"decision_at": decision.isoformat(), "original_candidates": int(selected.decision_at.eq(decision).sum()),
            "known_rows": n, "frozen_budget": frozen_k, "sources": day.source_id.nunique(), "reference_outcomes": day.reference_key.nunique(),
            "reference_scenes": day.scene_key.nunique(), "observation_days": day.observation_day_key.nunique()})
        settings = [(m, r, mi, ri) for mi, m in enumerate(OLD.MECHANISMS) for ri, r in enumerate(OLD.RETENTION)]
        settings += [(m, .5, None, None) for m in GROUP_MECHANISMS]
        for mechanism, retention, mi, ri in settings:
            for repetition in range(config["repetitions"]):
                if mechanism in OLD.MECHANISMS:
                    rng = np.random.default_rng(np.random.SeedSequence([OLD.SEED, day_index, mi, ri, repetition]))
                    mask = OLD.make_mask(n, retention, mechanism, history, rng)
                else:
                    mask = group_masks[(mechanism, repetition)][day.index.to_numpy()]
                context = {"decision_at": decision.isoformat(), "mechanism": mechanism, "nominal_retention": retention, "repetition": repetition,
                    "n_complete_population": n, "n_retained": int(mask.sum()), "actual_row_retention": float(mask.mean())}
                retentions.append({**context, "reference_scenes": day.scene_key.nunique(), "observation_days": day.observation_day_key.nunique(),
                    "retained_reference_scenes": day.loc[mask, "scene_key"].nunique(), "retained_observation_days": day.loc[mask, "observation_day_key"].nunique()})
                for a, b in OLD.PAIRS:
                    method_rows, bound = evaluate_pair(y, scores[a], scores[b], weights[a], weights[b], mask, mechanism)
                    pair_context = {**context, "pair": f"{a}_vs_{b}"}
                    rows.extend({**pair_context, **row} for row in method_rows)
                    intervals.append({**pair_context, **bound})
        print(f"Completed decision {day_index+1}/14: {decision.date()}", flush=True)
    methods, bounds = pd.DataFrame(rows), pd.DataFrame(intervals)
    del rows, intervals, group_masks
    reproduction = check_legacy(methods, Path(predictions).parent.parent / "masking_validation_20261005T153255_981415Z/replicate_metrics.csv")
    (output / "legacy_reproduction.json").write_text(json.dumps(reproduction, indent=2), encoding="utf-8")
    summary = summarize_methods(methods)
    bydate = summarize_methods(methods, ("decision_at",))
    bins = summarize_methods(methods, ("effect_bin",))
    bound_summary = summarize_bounds(bounds)
    frames = {"method_replicates": methods, "paired_bounds": bounds, "summary": summary, "bydate": bydate,
        "effectbins": bins, "bounds_summary": bound_summary, "population_bydate": pd.DataFrame(day_audit),
        "retention_bydate_replicate": pd.DataFrame(retentions)}
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    # Direct plotting inputs; no further selection or model fitting is needed.
    summary[summary.summary_population.eq("method_defined")].to_csv(output / "plot_method_comparison.csv", index=False)
    bins[bins.summary_population.eq("method_defined")].to_csv(output / "plot_effect_magnitude.csv", index=False)
    bydate[bydate.summary_population.eq("method_defined")].to_csv(output / "plot_bydate.csv", index=False)
    manifest = {"status": "completed_revision_exploratory", "config_sha256": file_hash(output / "experiment_config.json"),
        "input_sha256": file_hash(predictions), "script_sha256": file_hash(__file__), "legacy_script_sha256": file_hash(Path(__file__).with_name("evaluate_masking.py")),
        "primary_rows": len(selected), "known_rows": len(known), "known_sources": int(known.source_id.nunique()), "unknown_rows_excluded": int(selected.outcome_label.isna().sum()),
        "decision_dates": int(known.decision_at.nunique()), "unique_reference_outcomes": int(known.reference_key.nunique()), "reference_scenes": int(known.scene_key.nunique()), "observation_days": int(known.observation_day_key.nunique()),
        "output_rows": {name: len(frame) for name, frame in frames.items()}, "legacy_reproduction": reproduction,
        "wall_seconds": time.perf_counter() - start, "process_cpu_seconds": time.process_time() - cpu, **peak_memory(),
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "containment_failures": int((~bounds.contains_truth).sum()), "gpu_used": False, "models_refit": False,
        "output_total_bytes_before_manifest_and_interpretation": sum(path.stat().st_size for path in output.glob("*") if path.is_file())}
    if manifest["input_sha256"] != config["prediction_sha256"]:
        raise AssertionError("Input mutated during diagnostics")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_interpretation(output, summary, bound_summary, pd.DataFrame(group_audit), pd.DataFrame(retentions), manifest)
    return manifest


def write_interpretation(output, summary, bound_summary, group_audit, retentions, manifest):
    lines = ["# Revision-stage exploratory masking analysis", "",
        "The revised diagnostics share the same frozen target: each decision's originally observed outcomes, with an original 25% tie-aware selection budget. No predictors were refitted. The already missing 2,453 primary reference rows are excluded; this experiment cannot validate their unobserved outcomes or the real survey missingness mechanism.", "",
        "The experiment contains 1,468 repeated source-decision rows, 254 source IDs, 632 unique campaign-source-scene outcomes, 117 reference scenes, 10 acquisition days, and 14 decision dates. Unique outcomes can enter multiple decisions. Neither the dates nor mask repetitions are asserted to be independent field events.", "",
        "## Estimands and denominators", "",
        "Original-count reranking changes the list and inflates the retained-population selection fraction. Original-fraction reranking instead uses ceil(0.25*N_retained), but still changes the list and population. Frozen verified conditional precision preserves selection weights, while dividing each policy's observed positive mass by its own verified selected mass; zero mass makes the pair undefined. These three quantities are compared with the complete frozen target as diagnostics, not represented as unbiased estimators of that target.", "",
        "Uniform Horvitz–Thompson uses the actual fixed-size inclusion probability ceil(r*N)/N and targets the frozen complete difference. It is design-unbiased in the controlled uniform row-mask experiment; finite repetitions need not have zero empirical bias, and high variance remains possible. It is not clipped to [-1,1]. Normalized history weights are not treated as inclusion probabilities. HT is deliberately not reported for weighted or grouped masks. Zero filling is a frozen-weight plug-in diagnostic. Paired completion bounds are intervals, not point estimates or confidence intervals.", "",
        "All defined estimates and their validity counts appear in summary.csv. A second summary_population keeps only masks on which the four non-HT point methods are jointly defined. Row-weighted errors and equal-date macro errors are both saved where missing estimates create unequal counts. Reversal denominators use abs(complete truth)>=0.02 and valid estimates; ties are separate. Effect bins were frozen as [0,.02), [.02,.05), [.05,.10), [.10,.20), [.20,inf), with unique date counts rather than claims of independent event replication.", "",
        "## Nominal half retention: method contrasts", "",
        "| Pair | Mask | Method | Defined/total | Eligible | MAE | Bias | Reversals | Eligible ties |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    s = summary[summary.nominal_retention.eq(.5) & summary.summary_population.eq("method_defined")]
    for row in s.itertuples():
        if row.method == "uniform_ht" and row.mechanism != "uniform":
            continue
        lines.append(f"| {row.pair} | {row.mechanism} | {row.method} | {row.n_defined}/{row.n_total_date_masks} | {row.n_sign_eligible_defined} | {row.mae:.4f} | {row.signed_error:.4f} | {row.sign_reversal_rate:.4f} | {row.eligible_tie_rate:.4f} |")
    lines += ["", "Legacy original-count reranking is reproduced over all 50,400 old pair-masks, including 835/2,000 uniform and 751/2,000 history-weighted logistic reversals at nominal half retention. These reversals concern conditional reranking with a changed population and selection fraction; they are not a generic missing-label error probability.", "",
        "## Structured masks and realized retention", "",
        "One global mask per replicate is applied consistently to every use of a sampled group across decisions. Unique-reference groups use campaign-source-scene; different sources in one scene are distinct reference outcomes. Scene and acquisition-day masks deliberately co-hide larger sets. Only nominal half retention is tested for these three new mechanisms. Equal counts of groups do not guarantee half the source-decision rows are retained.", "",
        "| Group mask | Groups | Retained groups | Global row retention mean/min/max | Zero-label decision-masks |",
        "|---|---:|---:|---|---:|"]
    for mechanism, group in group_audit.groupby("mechanism"):
        r = retentions[retentions.mechanism.eq(mechanism)]
        lines.append(f"| {mechanism} | {int(group.total_groups.iloc[0])} | {int(group.retained_groups.iloc[0])} | {group.actual_row_retention.mean():.4f}/{group.actual_row_retention.min():.4f}/{group.actual_row_retention.max():.4f} | {int(r.n_retained.eq(0).sum())}/{len(r)} |")
    lines += ["", "There are only 10 unique acquisition days. Several decisions refer to one acquisition day, so day-level masking can hide every label on a decision; undefined conditional estimates are retained, not zero-filled. These grouped masks diagnose sensitivity to shared references and observation structure. They do not claim to reconstruct actual route planning, weather losses, or survey design.", "",
        "Daily paired intervals continue to contain the complete truth by construction. Their containment is a software check, not a test of physical label accuracy. Bounds summaries remain averages of daily intervals; cross-day joint completion analysis is a separate calculation. No mask result is used to select predictors or change the frozen bins.", "",
        f"Execution: {manifest['wall_seconds']:.2f} s wall time, {manifest['process_cpu_seconds']:.2f} s process CPU; peak working set {manifest.get('peak_working_set_bytes', 0)/1024**2:.1f} MiB. Memory is the Windows process-lifetime peak including imports. There were {manifest['containment_failures']} completion-containment failures. All calculations read prepared D-drive files and wrote to this revision directory.", "",
        "Plot-ready CSVs are plot_method_comparison.csv, plot_effect_magnitude.csv and plot_bydate.csv. Full estimates, validity flags, separate denominators, realized retention, group membership and reproducible seed design are retained. These results remain exploratory revision analyses of this already observed activity, not a new external campaign.", ""]
    (output / "interpretation.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    if args.freeze_only:
        freeze_config(args.predictions, args.output)
        print(json.dumps({"status": "config_frozen", "output": str(args.output)}))
    else:
        report = run(args.predictions, args.output)
        print(json.dumps({"status": report["status"], "wall_seconds": report["wall_seconds"], "peak_working_set_bytes": report["peak_working_set_bytes"], "rows": report["output_rows"]}))
