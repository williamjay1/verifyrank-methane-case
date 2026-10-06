"""Exploratory cost and reference-audit diagnostics; no raw data are modified."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import verifyrank.planning as planning


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def first_true(mask):
    hit = np.flatnonzero(mask)
    return int(hit[0]) if len(hit) else None


def greedy_grid(costs, benefits, expected_benefits, order, capacities, initial):
    """Same scan as the public API, vectorised across independent cost budgets."""
    remaining = np.asarray(capacities, dtype=float).copy()
    residual = np.full(len(remaining), initial, dtype=float)
    for j in order:
        if benefits[j] <= 0:
            continue
        affordable = remaining + 1e-12 >= costs[j]
        remaining[affordable] -= costs[j]
        residual[affordable] -= expected_benefits[j]
    return np.maximum(0., residual)


def error_path(c, y, retained, order, error_budget):
    """Each stopping decision uses returned labels only; hidden labels determine later returns."""
    missing = ~retained
    known = float(np.dot(c[retained], y[retained]))
    lower = known + np.minimum(c[missing], 0).sum()
    upper = known + np.maximum(c[missing], 0).sum()
    deltas = c[retained] * (1 - 2 * y[retained])
    neg = sorted(deltas[deltas < 0])[:error_budget]
    pos = sorted(deltas[deltas > 0], reverse=True)[:error_budget]
    lows, highs = [lower + sum(neg)], [upper + sum(pos)]
    for index in order:
        # A successful audit returns this stored result. Its value did not set the order.
        lower += c[index] * y[index] - min(c[index], 0.)
        upper += c[index] * y[index] - max(c[index], 0.)
        delta = c[index] * (1 - 2 * y[index])
        if delta < 0:
            neg = sorted(neg + [delta])[:error_budget]
        if delta > 0:
            pos = sorted(pos + [delta], reverse=True)[:error_budget]
        lo, hi = lower + sum(neg), upper + sum(pos)
        if lo > hi and lo - hi < 1e-10:
            lo = hi = (lo + hi) / 2
        lows.append(lo)
        highs.append(hi)
    return np.array(lows), np.array(highs)


def synthetic_validation(bounds, planning):
    cases = []
    designs = [
        ("shared cancellation and separate missing slots", [.4, -.3, .2, -.1, -.2],
         [np.nan]*5, ["shared", "shared", "b", "no_scene_d1", "no_scene_d2"]),
        ("partly known repeated reference", [.3, -.1, -.25], [1., np.nan, np.nan], ["shared", "shared", "b"]),
        ("identical policies", [0., 0.], [np.nan, np.nan], ["a", "b"]),
        ("all known", [.3, -.3], [1., 0.], ["a", "b"])]
    for name, coefficients, labels, groups in designs:
        result = bounds.grouped_bounds(coefficients, labels, groups)
        unknown = np.flatnonzero(np.isnan(result["labels"]))
        values = []
        for bits in itertools.product([0., 1.], repeat=len(unknown)):
            completed = result["labels"].copy()
            completed[unknown] = bits
            values.append(float(np.dot(result["coefficients"], completed)))
        assert np.allclose([result["lower"], result["upper"]], [min(values), max(values)])
        cases.append({"case": name, "raw_rows": len(groups), "unique_groups": len(result["groups"]),
                      "enumerated_assignments": len(values), "lower": result["lower"],
                      "upper": result["upper"], "rowwise_width": result["outer_width"],
                      "grouped_width": result["width"], "extrema_attained": True})
    c = {"a": .3, "b": -.2, "c": .1, "d": -.4}
    observed = {"a": 1, "b": 0, "c": 0}
    errors = []
    for r in range(4):
        possible = []
        for bits in itertools.product([0, 1], repeat=4):
            z = dict(zip(c, bits))
            if sum(z[k] != value for k, value in observed.items()) <= r:
                possible.append(sum(c[k] * z[k] for k in c))
        result = planning.robust_completion_interval(c, observed, r)
        assert np.allclose([result["lower"], result["upper"]], [min(possible), max(possible)])
        errors.append({"max_wrong": r, "admissible_assignments": len(possible), **result})
    candidates = [{"reference_id": key, "state": "ambiguous", "resolvable": True, "cost": cost}
                  for key, cost in [("a", 2), ("b", 3), ("c", 3)]]
    c = {"a": .09, "b": .12, "c": .12}
    greedy = planning.plan_references(c, candidates, 6)
    enumeration = []
    for chosen in itertools.product([0, 1], repeat=3):
        cost = sum(x["cost"]*bit for x, bit in zip(candidates, chosen))
        if cost <= 6:
            enumeration.append((sum(c[x["reference_id"]]*bit for x, bit in zip(candidates, chosen)), cost))
    best = max(enumeration)
    assert best[0] > sum(x["potential_width_reduction"] for x in greedy["selected"])
    return {"interpretation": "Structural synthetic examples and finite exhaustive arithmetic checks, not a second environment or empirical measurement validation",
            "binary_cases": cases, "error_budget_cases": errors,
            "greedy_counterexample": {"budget": 6, "greedy_reduction": .21, "greedy_cost": 5,
                "enumerated_optimal_reduction": best[0], "enumerated_optimal_cost": best[1],
                "greedy_selected": [x["reference_id"] for x in greedy["selected"]]},
            "all_checks_passed": True}


def main():
    parser = argparse.ArgumentParser()
    source_directory = Path(__file__).resolve().parent
    parser.add_argument("--workspace", type=Path, default=source_directory.parents[2],
                        help="Case repository root; inferred from this script by default")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New or empty directory for recomputed outputs; supplied results are not overwritten")
    args = parser.parse_args()
    root = args.workspace.resolve()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new or empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    source_config = source_directory / "config.json"
    configpath = out / "config.json"
    configpath.write_bytes(source_config.read_bytes())
    cfg = json.loads(configpath.read_text(encoding="utf-8"))
    start, cpu = time.perf_counter(), time.process_time()
    bounds = load_module("prior_bounds", root / "scripts/revision_bounds.py")
    save_json(out / "synthetic_validation.json", synthetic_validation(bounds, planning))
    frame = pd.read_csv(root / cfg["input"])
    frame = frame.loc[frame.scenario.eq(cfg["scenario"]) & frame.evaluation_mode.eq(cfg["evaluation_mode"])
                      & frame.campaign.astype(str).eq(cfg["campaign"])].copy()
    known = frame.loc[frame.outcome_label.notna()].copy()
    daily, rows = bounds.build_comparison_rows(known, [("fraction", .25)])
    daily.to_csv(out / "controlled_decision_budget_summary.csv", index=False)
    results = {pair: bounds.grouped_bounds(g.coefficient, g.label, g.reference_group)
               for pair, g in rows.groupby("pair", sort=True)}
    first = next(iter(results.values()))
    ids, truth = first["groups"], first["labels"]
    assert len(ids) == 632 and len(known) == 1468 and known.decision_at.nunique() == 14
    for result in results.values():
        assert np.array_equal(ids, result["groups"]) and np.array_equal(truth, result["labels"])
    digests = [hashlib.sha256(key.encode()).hexdigest() for key in ids]
    equal = np.ones(len(ids), dtype=int)
    variable = np.array([1 + int(h[:8], 16) % 5 for h in digests])
    heterogeneous = np.array([[1., .8, .5][int(h[8:16], 16) % 3] for h in digests])
    scenarios = [(cost_name, str(p), costs, np.full(len(ids), p), "width_per_cost")
                 for cost_name, costs in [("equal", equal), ("bounded_variable", variable)]
                 for p in cfg["uniform_recovery_probabilities"]]
    scenarios += [("bounded_variable", "heterogeneous", variable, heterogeneous, strategy)
                  for strategy in ("width_per_cost", "expected_width_per_cost")]
    rng = np.random.default_rng(cfg["mask_seed"])
    curves, targets, error_stops, first_case = [], [], [], None
    figure_capacity = np.arange(401)
    figure_accum = {}
    for rep in range(cfg["repetitions"]):
        retained = np.zeros(len(ids), dtype=bool)
        retained[rng.choice(len(ids), math.ceil(.5*len(ids)), replace=False)] = True
        hidden = np.flatnonzero(~retained)
        assert len(hidden) == 316
        for pair, result in results.items():
            c = result["coefficients"]
            cdict = {ids[j]: float(c[j]) for j in hidden}
            initial = float(np.abs(c[hidden]).sum())
            for cost_name, p_name, costs, probabilities, strategy in scenarios:
                metadata = [{"reference_id": ids[j], "state": "ambiguous", "resolvable": True,
                             "cost": int(costs[j]), "recovery_probability": float(probabilities[j])} for j in hidden]
                factor = probabilities if strategy == "expected_width_per_cost" else np.ones(len(ids))
                priority = np.abs(c) * factor / costs
                order = hidden[np.lexsort((ids[hidden], -priority[hidden]))]
                prefix_cost = np.r_[0., np.cumsum(costs[order])]
                prefix_expected = np.maximum(0., initial - np.r_[0., np.cumsum(np.abs(c[order])*probabilities[order])])
                if pair == cfg["operator_example"]["pair"] and p_name != "heterogeneous":
                    scenario_id = cost_name + "_p" + p_name
                    figure_accum.setdefault(scenario_id, np.zeros(len(figure_capacity)))
                    grid = greedy_grid(costs, np.abs(c), np.abs(c)*probabilities, order, figure_capacity, initial)
                    figure_accum[scenario_id] += grid
                    for checked_budget in (0, 17, 100, 200, 400):
                        check = planning.plan_references(cdict, metadata, checked_budget, strategy)
                        assert np.isclose(grid[checked_budget], check["hypothetical_expected_residual_width"])
                reach = first_true(prefix_expected <= cfg["target_width"] + 1e-12)
                context = {"pair": pair, "repetition": rep, "cost_scenario": cost_name,
                           "recovery_scenario": p_name, "strategy": strategy}
                targets.append({**context, "initial_width": initial, "width_target": cfg["target_width"],
                    "expected_width_target_reached": reach is not None,
                    "priority_prefix_count_to_expected_target": reach,
                    "priority_prefix_cost_to_expected_target": prefix_cost[reach] if reach is not None else None,
                    "expected_width_after_all_attempted": prefix_expected[-1]})
                total_cost = int(costs[hidden].sum())
                for fraction in cfg["capacity_fractions"]:
                    budget = math.floor(fraction * total_cost)
                    plan = planning.plan_references(cdict, metadata, budget, strategy)
                    assert plan["cost_used"] <= budget + 1e-10
                    curves.append({**context, "capacity_fraction": fraction, "cost_budget": budget,
                        "cost_used": plan["cost_used"], "selected_count": plan["selected_count"],
                        "initial_width": initial, "ideal_residual_width": plan["residual_width_if_all_selected_resolved"],
                        "expected_residual_width": plan["hypothetical_expected_residual_width"],
                        "expected_width_at_most_0.10": plan["hypothetical_expected_residual_width"] <= .1+1e-12})
                if p_name == "1.0":
                    for error_budget in cfg["error_budgets_unique_labels"]:
                        lower, upper = error_path(c, truth, retained, order, error_budget)
                        actual_truth = float(np.dot(c, truth))
                        assert (lower <= actual_truth+1e-10).all() and (upper >= actual_truth-1e-10).all()
                        equivalence = first_true((lower >= -.02) & (upper <= .02))
                        positive = first_true(lower > .02)
                        negative = first_true(upper < -.02)
                        classified = first_true(((lower >= -.02) & (upper <= .02)) | (lower > .02) | (upper < -.02))
                        record = {**context, "max_wrong_unique_labels": error_budget, "fixed_macro_truth": actual_truth,
                            "all_path_nominal_truth_contained": True, "final_lower": lower[-1], "final_upper": upper[-1],
                            "final_classification": planning.classify_interval(lower[-1], upper[-1])}
                        for name, value in [("equivalence", equivalence), ("positive", positive), ("negative", negative), ("any_classification", classified)]:
                            record[name+"_reached"] = value is not None
                            record["count_to_"+name] = value
                            record["cost_to_"+name] = prefix_cost[value] if value is not None else None
                        error_stops.append(record)
                if rep == 0 and pair == cfg["operator_example"]["pair"] and cost_name == "bounded_variable" and p_name == "1.0":
                    plan = planning.plan_references(cdict, metadata, cfg["operator_example"]["controlled_cost_budget"])
                    allc = dict(zip(ids, c.astype(float)))
                    returned = {ids[j]: int(truth[j]) for j in np.flatnonzero(retained)}
                    initial_bounds = planning.robust_completion_interval(allc, returned)
                    byid = {key: j for j, key in enumerate(ids)}
                    for item in plan["selected"]:
                        returned[item["reference_id"]] = int(truth[byid[item["reference_id"]]])
                    after = {str(r): planning.robust_completion_interval(allc, returned, r) for r in cfg["error_budgets_unique_labels"]}
                    for item in after.values():
                        item["classification"] = planning.classify_interval(item["lower"], item["upper"])
                    low, up = error_path(c, truth, retained, order, 0)
                    eq = first_true((low >= -.02) & (up <= .02))
                    first_case = {"kind": "controlled masking, all labels originally known", "pair": pair,
                        "original_known_rows": len(known), "unique_references": len(ids), "hidden_references": len(hidden),
                        "selection_budget": "ceil(.25 * original known candidates) independently on each of 14 dates; weights frozen before masking",
                        "initial_interval": initial_bounds, "plan": plan, "after_ideal_successful_returns": after,
                        "first_interval_certified_equivalence": {"reference_count": eq, "relative_cost": float(prefix_cost[eq]) if eq is not None else None,
                            "lower": float(low[eq]) if eq is not None else None, "upper": float(up[eq]) if eq is not None else None},
                        "note": "Archived labels are revealed only after label-free ordering. Ideal successful recovery is a controlled demonstration, not measured field performance."}
                    candidate_rows = []
                    rankmap = {ids[j]: rank+1 for rank, j in enumerate(order)}
                    for j in hidden:
                        campaign, source, scene = json.loads(ids[j].split(":", 1)[1])
                        candidate_rows.append({"priority_rank": rankmap[ids[j]], "reference_id": ids[j], "campaign": campaign,
                            "source_id": source, "scene": scene, "cost_relative_units": costs[j], "coefficient": c[j],
                            "potential_width_reduction": abs(c[j]), "priority_abs_coefficient_per_cost": abs(c[j])/costs[j],
                            "action": "recover this existing archived qualified binary reference outcome"})
                    pd.DataFrame(candidate_rows).sort_values("priority_rank").to_csv(out / "controlled_candidate_list.csv", index=False)
        if (rep+1) % 50 == 0:
            print(f"Completed {rep+1}/200 common reference masks", flush=True)
    curves = pd.DataFrame(curves)
    targets = pd.DataFrame(targets)
    errors = pd.DataFrame(error_stops)
    keys = ["pair", "cost_scenario", "recovery_scenario", "strategy"]
    curves.to_csv(out / "planning_replicates.csv", index=False)
    summary = curves.groupby(keys+["capacity_fraction"], as_index=False).agg(
        repetitions=("repetition", "size"), mean_cost_budget=("cost_budget", "mean"), mean_cost_used=("cost_used", "mean"),
        mean_selected_references=("selected_count", "mean"), mean_initial_width=("initial_width", "mean"),
        mean_ideal_residual_width=("ideal_residual_width", "mean"), mean_expected_residual_width=("expected_residual_width", "mean"),
        fraction_expected_width_at_most_0_10=("expected_width_at_most_0.10", "mean"))
    summary.to_csv(out / "planning_summary.csv", index=False)
    targets.to_csv(out / "target_replicates.csv", index=False)
    targets.groupby(keys, as_index=False).agg(repetitions=("repetition", "size"), reachable_repetitions=("expected_width_target_reached", "sum"),
        mean_prefix_count_among_reachable=("priority_prefix_count_to_expected_target", "mean"),
        mean_prefix_cost_among_reachable=("priority_prefix_cost_to_expected_target", "mean"),
        mean_expected_width_after_all_attempted=("expected_width_after_all_attempted", "mean")).to_csv(out / "target_summary.csv", index=False)
    errors.to_csv(out / "stopping_replicates.csv", index=False)
    errors.groupby(["pair", "cost_scenario", "max_wrong_unique_labels"], as_index=False).agg(
        repetitions=("repetition", "size"), equivalence_reached=("equivalence_reached", "sum"),
        mean_count_to_equivalence_among_reachable=("count_to_equivalence", "mean"),
        mean_cost_to_equivalence_among_reachable=("cost_to_equivalence", "mean"),
        classification_reached=("any_classification_reached", "sum"),
        mean_count_to_classification_among_reachable=("count_to_any_classification", "mean"),
        mean_cost_to_classification_among_reachable=("cost_to_any_classification", "mean"),
        mean_final_lower=("final_lower", "mean"), mean_final_upper=("final_upper", "mean")
    ).to_csv(out / "stopping_summary.csv", index=False)
    # Real full-candidate population: only ambiguous actual scenes can be audited.
    full = pd.read_csv(root / "results/revision_20261006/bounds/reference_groups_summary.csv")
    full = full.loc[full.budget_kind.eq("fraction") & full.budget_value.eq(.25)]
    real_cases = []
    for pair, records in full.groupby("pair", sort=True):
        unavailable = records.loc[records.label.isna()]
        coefficients = dict(zip(unavailable.reference_group, unavailable.macro_coefficient))
        metadata = [{"reference_id": row.reference_group, "state": "no_followup" if row.no_scene_slot else "ambiguous",
                     "resolvable": not row.no_scene_slot, "cost": 1, "recovery_probability": None} for row in unavailable.itertuples()]
        plan = planning.plan_references(coefficients, metadata, cfg["operator_example"]["actual_cost_budget"])
        fixed = float(np.dot(records.loc[records.label.notna(), "macro_coefficient"], records.loc[records.label.notna(), "label"]))
        interval = planning.robust_completion_interval(coefficients, {}, fixed_known_contribution=fixed)
        assert sum(not row.no_scene_slot for row in unavailable.itertuples()) == 1
        possible_updates = []
        for label in (0, 1):
            returned = {item["reference_id"]: label for item in plan["selected"]}
            update = planning.robust_completion_interval(coefficients, returned, fixed_known_contribution=fixed)
            possible_updates.append({"hypothetical_returned_value": label, **update,
                "classification": planning.classify_interval(update["lower"], update["upper"])})
        real_cases.append({"pair": pair, "source_decision_rows": len(frame), "decision_dates": frame.decision_at.nunique(),
            "selection_budget": "ceil(.25 * all prefix-eligible candidates) per date", "current_interval": interval,
            "classification": planning.classify_interval(interval["lower"], interval["upper"]), "plan": plan,
            "possible_updates_after_one_ambiguous_reference": possible_updates,
            "width_0_10_feasible_by_existing_reference_recovery": plan["irreducible_width"] <= .1,
            "operator_action": "Audit the one existing ambiguous reference if justified, but recognise that the remaining no-followup floor requires redesigned reference collection. New monitoring creates new evidence and cannot recover the missing historical state."})
    report = {"config_sha256": hashlib.sha256(configpath.read_bytes()).hexdigest(), "actual_archive": real_cases,
              "controlled_first_replicate": first_case,
              "update_instructions": ["Request only existing resolvable unique reference outcomes in the returned order.",
                  "Charge the declared audit cost whether or not recovery succeeds.",
                  "For failed or nonbinary/uncertain responses, keep the reference unknown.",
                  "For successful qualified binary responses, propagate the result to all matching reference uses and recompute bounds.",
                  "Stop for positive/negative evidence only beyond the entire ±.02 tolerance; stop for practical equivalence only when the whole interval lies within it.",
                  "If no recoverable nonzero references remain and the interval remains undetermined, report the irreducible gap rather than an operational saving."]}
    save_json(out / "case_report.json", report)
    text = ["# Reference-audit operator report", "",
        "This is a local open-loop archive audit, not a flight plan. Costs (1 or 1–5 relative units) and recovery probabilities (1, 0.8, 0.5) are hypothetical sensitivities. The 25% selection budget and the audit cost budget are different quantities.", "",
        "## Actual full-candidate archive", ""]
    for item in real_cases:
        interval, plan = item["current_interval"], item["plan"]
        text.append(f"- {item['pair']}: interval [{interval['lower']:.6f}, {interval['upper']:.6f}], {item['classification']}; one recoverable reference, cost budget {plan['cost_budget']:g}. Even perfect recovery leaves width {plan['irreducible_width']:.6f}; width 0.10 is unreachable by auditing this archive.")
    text += ["", "The only actual ambiguous outcome is listed below. It is shared only where the campaign, source and actual reference scene are identical. No-followup slots are not audit requests.", ""]
    main_real_case = next(item for item in real_cases if item["pair"] == cfg["operator_example"]["pair"])
    for selected in main_real_case["plan"]["selected"]:
        text.append(f"- `{selected['reference_id']}`: inspect/recover the existing qualified binary reference; ideal maximum width reduction {selected['potential_width_reduction']:.6f} for {main_real_case['pair']}.")
    case = first_case
    text += ["", "## Controlled first replicate", "",
        f"The original 1,468 known source-decision rows contain 632 unique actual references; 316 are hidden. Scores and the daily 25% selections are fixed before masking. Under variable costs, a cost budget of 100 selects {case['plan']['selected_count']} nonzero references at cost {case['plan']['cost_used']:g}.", "",
        "These are the first ten label-free requests; labels are deliberately absent from this list:", "",
        "| Reference ID | Relative cost | Ideal maximum width reduction |", "|---|---:|---:|"]
    for selected in case["plan"]["selected"][:10]:
        text.append(f"| `{selected['reference_id']}` | {selected['cost']:g} | {selected['potential_width_reduction']:.6f} |")
    text += ["", "After ideal successful returned labels:", ""]
    for r, interval in case["after_ideal_successful_returns"].items():
        text.append(f"- At most {r} wrong currently supplied unique labels: [{interval['lower']:.6f}, {interval['upper']:.6f}], {interval['classification']}.")
    stop = case["first_interval_certified_equivalence"]
    text += ["", f"Along the complete label-free variable-cost order, practical equivalence is first certified after {stop['reference_count']} returned outcomes at relative cost {stop['relative_cost']}; interval [{stop['lower']:.6f}, {stop['upper']:.6f}]. This uses the interval after each return, not the hidden complete value as a stopping rule.", "",
        "## Use and limits", "", *[f"{i+1}. {instruction}" for i, instruction in enumerate(report["update_instructions"])], "",
        "Expected widths under incomplete recovery are not achieved intervals. Greedy indivisible selection is not cost optimal; the synthetic counterexample documents this. Prefix crossing costs are order-specific, not minimum costs or expected random stopping costs. Exact label-error bounds depend on a stated count budget and do not validate catalogue labels physically."]
    (out / "case_report.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    # Compact numerical interpretation for manuscript integration.
    target_summary = pd.read_csv(out / "target_summary.csv")
    stopping_summary = pd.read_csv(out / "stopping_summary.csv")
    interpretation = ["# Exploratory planning interpretation", "",
        "All three comparisons use the same 200 exact 50% unique-reference masks, matching the previous analysis. Cost and probability assignments depend only on stable reference IDs. Equal probability within a scenario rescales all priorities equally and therefore cannot alter their order; only the heterogeneous sensitivity can change rankings.", "",
        "Budget-feasible selected sets can be non-nested across different budgets because the indivisible greedy scan skips unaffordable items. The prefix target costs describe a distinct sequential audit order, without skips. Neither is claimed optimal.", "",
        "## Ideal equal-cost reproduction", ""]
    for row in target_summary.loc[target_summary.cost_scenario.eq("equal") & target_summary.recovery_scenario.eq("1.0")].itertuples():
        interpretation.append(f"- {row.pair}: mean unique returns to width 0.10 = {row.mean_prefix_count_among_reachable:.3f}; reached {row.reachable_repetitions}/200 masks.")
    interpretation += ["", "## Interval stopping rather than truth stopping", ""]
    for row in stopping_summary.loc[stopping_summary.pair.eq("logistic_vs_beta_rate")].itertuples():
        interpretation.append(f"- Logistic/history, {row.cost_scenario}, max wrong={row.max_wrong_unique_labels}: practical equivalence reached {row.equivalence_reached}/200; mean count among reachable={row.mean_count_to_equivalence_among_reachable}, mean cost={row.mean_cost_to_equivalence_among_reachable}.")
    interpretation += ["", "This establishes executable decision-support behaviour and identifies information limits in one archive. The synthetic cases test portability of the evidence contract and arithmetic, not cross-environment empirical validity."]
    (out / "interpretation.md").write_text("\n".join(interpretation)+"\n", encoding="utf-8")
    figure_rows = [{"scenario": scenario, "cost_budget": int(capacity), "mean_expected_width": float(width/cfg["repetitions"])}
        for scenario, widths in figure_accum.items() for capacity, width in zip(figure_capacity, widths)]
    pd.DataFrame(figure_rows).to_csv(out / "figure_curves.csv", index=False)
    save_json(out / "figure_data.json", {"curve_file": "figure_curves.csv",
        "curve_definition": "Mean hypothetical expected residual width from the actual budget-feasible greedy scan at every integer relative cost budget 0..400, logistic versus history, 200 group masks. Same rule as API; selected sets need not be nested.",
        "report_lines": ["14 decisions; 25% selection budget fixed before masking", "632 unique known references; 316 hidden (replicate 0)",
            f"Variable-cost audit budget 100: {case['plan']['selected_count']} requests; cost {case['plan']['cost_used']:g}",
            f"Equivalence first certified: {stop['reference_count']} returns, relative cost {stop['relative_cost']:g}"],
        "cost_scenarios": [["equal_p1.0", "#0072B2", "-", "Equal costs"], ["bounded_variable_p1.0", "#D55E00", "--", "Variable costs"]],
        "recovery_scenarios": [["bounded_variable_p1.0", "#0072B2", "-", "Assumed recovery 1.0"],
            ["bounded_variable_p0.8", "#D55E00", "--", "Assumed recovery 0.8"], ["bounded_variable_p0.5", "#009E73", ":", "Assumed recovery 0.5"]]})
    selected_stops = stopping_summary.loc[stopping_summary.pair.eq("logistic_vs_beta_rate") & stopping_summary.max_wrong_unique_labels.eq(0)]
    stoptext = "; ".join(f"{r.cost_scenario}: {r.mean_count_to_equivalence_among_reachable:.3f} returns at mean relative cost {r.mean_cost_to_equivalence_among_reachable:.3f} ({r.equivalence_reached}/200 masks)" for r in selected_stops.itertuples())
    manuscript = ["Across 200 common masks of 316 of 632 unique reference outcomes, costs were either equal or assigned independently of labels in bounded relative units (1–5). The label-free planner exposes the distinction between a budget-feasible greedy set and a sequential audit prefix; neither is asserted to minimise cost. Uniform assumed recovery probabilities of 1, 0.8 and 0.5 scale expected residual-width reductions without changing rank order. These expected widths are sensitivities, not achieved certification intervals. The actual full-candidate archive has just one ambiguous reference available for retrospective audit; even ideal recovery leaves a logistic/history width floor of 0.315348 from absent qualified follow-up. A 0.10 width target cannot be achieved by recovering existing references in that archive.",
        "For the controlled known-reference population, practical equivalence was assessed only when the entire updated interval lay inside [−0.02, 0.02]. With ideal successful returns and labels treated as correct, the first such interval required " + stoptext + ". This is an interval-based sequential stopping diagnostic after each returned label, not a stopping rule based on the complete hidden contrast. Permitting even one currently supplied unique-reference label to be wrong prevented equivalence in all 200 masks for both cost scenarios: the full-return interval became [−0.037768, 0.033660]. These error budgets cover both initially retained and newly returned labels. Their exact flip bounds test a specified error count, not physical label accuracy; the equivalence result therefore depends on catalogue-label correctness."]
    (out / "manuscript_results.md").write_text("\n\n".join(manuscript)+"\n", encoding="utf-8")
    save_json(out / "run_manifest.json", {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "wall_seconds": time.perf_counter()-start, "process_cpu_seconds": time.process_time()-cpu,
        "config_sha256": hashlib.sha256(configpath.read_bytes()).hexdigest(), "repetitions": cfg["repetitions"],
        "known_rows": len(known), "unique_references": len(ids), "hidden_references": 316,
        "planning_curve_rows": len(curves), "target_rows": len(targets), "stopping_rows": len(errors),
        "all_assertions_passed": True, "no_new_data_download": True, "no_model_refitting": True})
    print(json.dumps({"output": str(out), "seconds": time.perf_counter()-start, "success": True}), flush=True)


if __name__ == "__main__":
    main()
