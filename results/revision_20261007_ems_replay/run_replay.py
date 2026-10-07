# -*- coding: utf-8 -*-
"""Revision 2026-10-07: verification replay, baseline comparison, exact
cost-constrained selection, synthetic regime sweep and label-error tipping
points for the VerifyRank evaluation layer.

Sources, all preserved analysis outputs of the 2024 primary panel
(scenario qaccepted_r250_h3_l24, external 2023-to-2024, fraction 0.25):
  bounds/reference_groups_summary.csv      macro group coefficients and labels
  bounds/coefficient_audit_summary.csv     row-level coefficients
  bounds/mask_replicate_summary.csv        200 controlled reference masks
  bounds/mask_reveal_curve_summary.csv     aggregated reveal curves
No raw observation is modified and no new environmental data is introduced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "results" / "revision_20261006" / "bounds"
GROUPS = REV / "reference_groups_summary.csv"
COEF = REV / "coefficient_audit_summary.csv"
REPLICATES = REV / "mask_replicate_summary.csv"
CURVES = REV / "mask_reveal_curve_summary.csv"
OUT = ROOT / "results" / "revision_20261007_ems_replay"
PRIMARY = ("fraction", 0.25)
PAIRS = ["logistic_vs_beta_rate", "hist_gradient_boosting_vs_beta_rate", "beta_rate_vs_uniform_random"]
PAIR_LABELS = {"logistic_vs_beta_rate": "logistic vs history",
               "hist_gradient_boosting_vs_beta_rate": "boosting vs history",
               "beta_rate_vs_uniform_random": "history vs random"}
TAU = 0.02
TARGETS = [0.20, 0.10, 0.05]
SEED = 20261007


def log(message):
    print(message, flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def primary_rows(path, columns, keep_reason=None):
    frames = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=500000):
        keep = chunk[(chunk.budget_kind == PRIMARY[0]) & (chunk.budget_value == PRIMARY[1])]
        if keep_reason is not None:
            keep = keep[keep.reason != "no_future_scene"]
        if len(keep):
            frames.append(keep)
    return pd.concat(frames, ignore_index=True)


def load_panel():
    """Macro groups per pair with both the width contribution and the endpoint
    contribution of every group."""
    groups = primary_rows(GROUPS, ["budget_kind", "budget_value", "pair", "reference_group",
                                   "macro_coefficient", "label", "no_scene_slot",
                                   "shared_across_dates"])
    groups["pair"] = groups.pair.astype(str)
    groups["reference_group"] = groups.reference_group.astype(str)
    groups["C"] = groups.macro_coefficient.astype(float)
    groups["y"] = pd.to_numeric(groups.label, errors="coerce")
    groups["is_reference"] = ~groups.no_scene_slot.astype(bool)

    audit = primary_rows(COEF, ["budget_kind", "budget_value", "pair", "reference_group",
                                "reason", "coefficient"])
    audit["pair"] = audit.pair.astype(str)
    audit["reference_group"] = audit.reference_group.astype(str)
    width = audit.groupby(["pair", "reference_group"], sort=True).agg(
        width_remove=("coefficient", lambda values: float(np.abs(values).sum())),
    ).reset_index()
    rows = groups.merge(width, on=["pair", "reference_group"], how="left")
    rows["width_remove"] = rows.width_remove.fillna(0.0)
    panels = {}
    for pair in PAIRS:
        sub = rows[rows.pair == pair].reset_index(drop=True)
        assert len(sub) == 3084, (pair, len(sub))
        panels[pair] = sub
    return panels


def archive_widths(panels, outdir):
    """Width decomposition of the panel an analyst actually holds."""
    log("== archive width decomposition ==")
    rows = []
    for pair in PAIRS:
        table = panels[pair]
        w = table.width_remove.to_numpy(float)
        C = table.C.to_numpy(float)
        y = table.y.to_numpy(float)
        absent = (~table.is_reference).to_numpy()
        labelled = (table.is_reference & table.y.notna()).to_numpy()
        ambiguous = (table.is_reference & table.y.isna()).to_numpy()
        paired = float(np.abs(C[~labelled & ~ambiguous]).sum() + np.abs(C[ambiguous]).sum())
        rows.append(dict(pair=pair, pair_label=PAIR_LABELS[pair],
                         macro_groups=int(len(table)),
                         absent_observation_groups=int(absent.sum()),
                         labelled_reference_groups=int(labelled.sum()),
                         ambiguous_reference_groups=int(ambiguous.sum()),
                         paired_width=paired,
                         absent_width=float(np.abs(C[absent]).sum()),
                         ambiguous_width=float(np.abs(C[ambiguous]).sum()),
                         resolvable_width=float(np.abs(C[ambiguous]).sum()),
                         irrecoverable_floor=float(np.abs(C[absent]).sum()),
                         floor_share=float(np.abs(C[absent]).sum() / paired),
                         complete_data_contrast=float((C[labelled] * y[labelled]).sum())))
    frame = pd.DataFrame(rows)
    frame.to_csv(outdir / "archive_width_decomposition.csv", index=False)
    log(frame.to_string(index=False))
    return frame


def controlled_replay(outdir):
    """Published 200-mask reveal experiment, re-aggregated per reveal count."""
    log("== controlled reference reveal ==")
    rep = pd.read_csv(REPLICATES)
    rep = rep[rep.pair.isin(PAIRS)].copy()
    curves = pd.read_csv(CURVES)
    curves = curves[curves.pair.isin(PAIRS)].copy()
    curves.to_csv(outdir / "controlled_reveal_curves.csv", index=False)
    summary = rep.groupby("pair").agg(
        unique_outcomes=("unique_outcomes", "first"),
        hidden_unique_outcomes=("hidden_unique_outcomes", "first"),
        initial_width_mean=("initial_joint_width", "mean"),
        initial_width_min=("initial_joint_width", "min"),
        initial_width_max=("initial_joint_width", "max"),
        outer_width_mean=("initial_macro_outer_width", "mean"),
        true_macro_difference=("true_macro_difference", "first"),
        needed_width_020_mean=("needed_for_width_0.2", "mean"),
        needed_width_010_mean=("needed_for_width_0.1", "mean"),
        needed_width_005_mean=("needed_for_width_0.05", "mean"),
        uniform_crossing_010_mean=("uniform_expected_width_crossing_0.1", "mean"),
        uniform_crossing_020_mean=("uniform_expected_width_crossing_0.2", "mean"),
        direction_reached=("needed_unique_outcomes_for_direction", lambda s: int(s.notna().sum())),
        within_band_final=("final_within_threshold_band", "sum"),
        replicas=("repetition", "size"),
    ).reset_index()
    summary["pair_label"] = summary.pair.map(PAIR_LABELS)
    summary.to_csv(outdir / "controlled_reveal_summary.csv", index=False)
    log(summary[["pair_label", "initial_width_mean", "needed_width_020_mean",
                 "needed_width_010_mean", "needed_width_005_mean",
                 "uniform_crossing_010_mean", "direction_reached", "replicas"]].to_string(index=False))
    return summary, curves


def label_error(panels, outdir):
    """Tipping point on the controlled population: the width after every
    resolvable reference is returned, and the interval once a stated number of
    returned labels is wrong."""
    log("== label-error tipping point ==")
    pair = "logistic_vs_beta_rate"
    table = panels[pair]
    labelled = table[(table.is_reference) & (table.y.notna())].reset_index(drop=True)
    unresolved = table[(~table.is_reference) | (table.y.isna())].reset_index(drop=True)
    C = labelled.C.to_numpy(float)
    y = labelled.y.to_numpy(float)
    width = float(np.abs(unresolved.C.to_numpy(float)).sum())
    truth = float((C * y).sum())
    rows = []
    for flips in (0, 1, 2, 3, 5, 10, 20, 50, 100, 300):
        values = y.copy()
        values[:flips] = 1.0 - values[:flips]
        fixed = float((C * values).sum())
        lower, upper = fixed - width, fixed + width
        rows.append(dict(pair=pair, pair_label=PAIR_LABELS[pair], wrong_labels=flips,
                         lower=lower, upper=upper, width=upper - lower,
                         fixed_contribution=fixed,
                         assessment=classify(lower, upper),
                         complete_data_contrast=truth))
    frame = pd.DataFrame(rows)
    frame.to_csv(outdir / "label_error_tipping.csv", index=False)
    within = frame[frame.assessment == "practical_equivalence"]
    tipping = dict(pair=pair, tolerance=TAU,
                   references=int(len(labelled)),
                   width_after_full_recovery=width,
                   complete_data_contrast=truth,
                   equivalence_with_correct_labels=bool(
                       frame.loc[frame.wrong_labels == 0, "assessment"].iloc[0] == "practical_equivalence"),
                   smallest_wrong_label_count_inside_band=int(within.wrong_labels.min()) if len(within) else None,
                   largest_wrong_label_count_inside_band=int(within.wrong_labels.max()) if len(within) else None,
                   wrong_label_counts_inside_band=[int(v) for v in within.wrong_labels],
                   direction_decided_without_wrong_labels=bool(
                       frame.loc[frame.wrong_labels == 0, "assessment"].iloc[0] in ("supports_a", "supports_b")))
    with open(outdir / "tipping_point.json", "w", encoding="utf-8") as handle:
        json.dump(tipping, handle, indent=1)
    log(json.dumps(tipping))
    return frame


def classify(lower, upper, tau=TAU):
    if lower > tau:
        return "supports_a"
    if upper < -tau:
        return "supports_b"
    if lower >= -tau and upper <= tau:
        return "practical_equivalence"
    return "undetermined"


def greedy_scan(benefit, cost, budget):
    ratio = benefit / cost
    order = np.argsort(-ratio, kind="stable")
    spend = 0.0
    gain = 0.0
    count = 0
    for i in order:
        if spend + cost[i] <= budget + 1e-12:
            spend += cost[i]
            gain += benefit[i]
            count += 1
    return gain, count


def exact_knapsack(benefit, cost, budget, units=4000):
    """Exact 0/1 knapsack with the budget discretised into a fixed number of
    units, so the dynamic program stays small for any cost profile."""
    total = float(cost.sum())
    if total <= 0 or budget <= 0:
        return 0.0
    scale = units / total
    capacity = int(round(budget * scale))
    if capacity <= 0:
        return 0.0
    weights = np.maximum(1, np.round(cost * scale).astype(np.int64))
    best = np.zeros(capacity + 1)
    for weight, value in zip(weights, benefit):
        if weight > capacity:
            continue
        best[weight:] = np.maximum(best[weight:], best[:capacity + 1 - weight] + value)
    return float(best[capacity])


def run_exact_vs_greedy(panels, outdir):
    log("== exact cost-constrained selection versus greedy ==")
    rng = np.random.default_rng(SEED)
    rows = []
    for pair in PAIRS:
        table = panels[pair]
        refs = table[table.y.notna()].reset_index(drop=True)
        benefit = np.abs(refs.C.to_numpy(float))
        n = len(refs)
        scenarios = {
            "equal_cost": np.ones(n),
            "cost_scaled_by_coefficient": np.maximum(0.5, np.round(5.0 * benefit / benefit.max(), 4)),
            "lognormal_cost": np.round(rng.lognormal(0.0, 0.6, n), 6),
        }
        for name, cost in scenarios.items():
            cost = np.maximum(cost, 1e-6)
            total_cost = float(cost.sum())
            for fraction in np.round(np.linspace(0.0, 1.0, 41), 4):
                budget = float(fraction * total_cost)
                gain, count = greedy_scan(benefit, cost, budget)
                exact = exact_knapsack(benefit, cost, budget)
                rows.append(dict(pair=pair, pair_label=PAIR_LABELS[pair], scenario=name,
                                 budget_fraction=float(fraction), budget=budget,
                                 greedy_width_removed=float(gain), exact_width_removed=exact,
                                 absolute_gap=float(exact - gain),
                                 relative_gap=float((exact - gain) / exact) if exact > 0 else 0.0,
                                 greedy_count=int(count), references=n))
    frame = pd.DataFrame(rows)
    frame.to_csv(outdir / "exact_vs_greedy.csv", index=False)
    worst = frame.loc[frame.relative_gap.idxmax()]
    log("largest relative gap %.4f at budget fraction %.2f (%s, %s)" %
        (worst.relative_gap, worst.budget_fraction, worst.pair_label, worst.scenario))
    log(frame.groupby(["pair_label", "scenario"]).relative_gap.agg(["mean", "max"]).to_string())

    table = panels[PAIRS[0]]
    small = table[table.y.notna()].nlargest(18, "C").reset_index(drop=True)
    small = table[table.y.notna()].assign(k=lambda d: d.C.abs()).nlargest(18, "k").reset_index(drop=True)
    benefit = small.k.to_numpy(float)
    cost = np.round(np.random.default_rng(SEED + 1).uniform(1, 5, size=len(small)), 6)
    budget = float(cost.sum()) * 0.5
    gain, count = greedy_scan(benefit, cost, budget)
    best = 0.0
    for mask in range(1 << len(small)):
        chosen = [k for k in range(len(small)) if (mask >> k) & 1]
        if cost[chosen].sum() <= budget + 1e-12:
            best = max(best, float(benefit[chosen].sum()))
    audit = dict(references=int(len(small)), budget=float(budget),
                 greedy=float(gain), greedy_count=int(count),
                 exhaustive_optimum=float(best), gap=float(best - gain),
                 greedy_is_optimal=bool(abs(best - gain) < 1e-9))
    with open(outdir / "knapsack_exhaustive_audit.json", "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=1)
    log(json.dumps(audit))
    return frame


def run_regime_sweep(outdir, replicates=40):
    log("== synthetic regime sweep ==")
    rng = np.random.default_rng(SEED)
    rows = []
    for groups in (500, 5000):
        for share in (0.05, 0.25, 0.50, 1.00):
            for spread in ("equal", "uniform_1_5", "lognormal"):
                for recovery in (1.0, 0.8, 0.5):
                    for rep in range(replicates):
                        resolvable = max(1, int(round(groups * share)))
                        width = np.abs(rng.normal(0, 1, groups))
                        total = float(width.sum())
                        removable = float(width[:resolvable].sum())
                        floor = total - removable
                        target = 0.5 * total
                        if spread == "equal":
                            cost = np.ones(resolvable)
                        elif spread == "uniform_1_5":
                            cost = rng.uniform(1, 5, resolvable)
                        else:
                            cost = rng.lognormal(0.0, 0.8, resolvable)
                        returned = rng.random(resolvable) < recovery
                        priority = np.argsort(-(width[:resolvable] * returned) / cost, kind="stable")
                        remaining = total
                        spend = 0.0
                        count = 0
                        hit_count = None
                        hit_cost = None
                        for i in priority:
                            spend += cost[i]
                            if returned[i]:
                                remaining -= width[i]
                                count += 1
                            if hit_count is None and remaining <= target:
                                hit_count = count
                                hit_cost = spend
                        rows.append(dict(groups=groups, resolvable_share=share, cost_spread=spread,
                                         recovery=recovery, replicate=rep,
                                         total_width=total, removable_width=removable,
                                         floor=floor, floor_share=floor / total,
                                         target=target, reachable=bool(floor <= target),
                                         references_to_half=hit_count, cost_to_half=hit_cost))
    frame = pd.DataFrame(rows)
    frame.to_csv(outdir / "regime_sweep_raw.csv", index=False)
    by_share = frame.groupby("resolvable_share").agg(
        floor_share_mean=("floor_share", "mean"),
        reachable_rate=("reachable", "mean"),
        references_to_half_mean=("references_to_half", "mean"),
        replicates=("replicate", "size")).reset_index()
    by_share.to_csv(outdir / "regime_sweep_by_share.csv", index=False)
    by_recovery = frame.groupby("recovery").agg(
        reachable_rate=("reachable", "mean"),
        references_to_half_mean=("references_to_half", "mean"),
        cost_to_half_mean=("cost_to_half", "mean")).reset_index()
    by_recovery.to_csv(outdir / "regime_sweep_by_recovery.csv", index=False)
    by_cost = frame.groupby("cost_spread").agg(
        reachable_rate=("reachable", "mean"),
        references_to_half_mean=("references_to_half", "mean"),
        cost_to_half_mean=("cost_to_half", "mean")).reset_index()
    by_cost.to_csv(outdir / "regime_sweep_by_cost.csv", index=False)
    log(by_share.to_string(index=False))
    log(by_recovery.to_string(index=False))
    log(by_cost.to_string(index=False))
    return frame, by_share, by_recovery, by_cost


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Fresh or empty writable output directory")
    parser.add_argument("--replicates", type=int, default=40)
    args = parser.parse_args(argv)
    outdir = Path(args.out).expanduser().resolve()
    if outdir.exists() and any(outdir.iterdir()):
        parser.error("--out must be a new or empty directory; preserve published outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    panels = load_panel()
    log("macro groups per pair: %d" % len(panels[PAIRS[0]]))
    archive_widths(panels, outdir)
    controlled_replay(outdir)
    label_error(panels, outdir)
    run_exact_vs_greedy(panels, outdir)
    run_regime_sweep(outdir, replicates=args.replicates)
    manifest = {
        "stage": "2026-10-07 revision: archive widths, controlled reveal, label-error tipping, exact selection, regime sweep",
        "group_summary": GROUPS.relative_to(ROOT).as_posix(), "group_summary_sha256": sha256(GROUPS),
        "row_audit": COEF.relative_to(ROOT).as_posix(), "row_audit_sha256": sha256(COEF),
        "mask_replicates": REPLICATES.relative_to(ROOT).as_posix(), "mask_replicates_sha256": sha256(REPLICATES),
        "mask_curves": CURVES.relative_to(ROOT).as_posix(), "mask_curves_sha256": sha256(CURVES),
        "primary_budget": list(PRIMARY), "pairs": PAIRS, "tolerance": TAU,
        "width_targets": TARGETS, "seed": SEED,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "seconds": round(time.time() - started, 2),
        "outputs": sorted(p.name for p in outdir.glob("*") if p.is_file()),
        "note": ("Case widths follow the preserved completion calculation on the archived "
                 "primary panel. Synthetic regimes test structural behaviour only."),
    }
    with open(outdir / "run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)
    log("completed in %.1f s" % (time.time() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
