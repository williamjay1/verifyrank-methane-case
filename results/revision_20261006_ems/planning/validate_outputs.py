"""Read-only checks of the saved second-revision planning outputs."""
import hashlib
import importlib.util
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import verifyrank.planning as planner


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


source_directory = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results-dir", type=Path, required=True,
                    help="Recomputed planning output directory; writes its validation report here")
args = parser.parse_args()
out = args.results_dir.resolve()
runner = module("planning_runner_validation", source_directory / "run_planning.py")
curves = pd.read_csv(out / "planning_replicates.csv")
assert len(curves) == 28800
assert (curves.cost_used <= curves.cost_budget + 1e-10).all()
assert (curves.expected_residual_width >= curves.ideal_residual_width - 1e-10).all()
assert (curves.expected_residual_width <= curves.initial_width + 1e-10).all()
figure = pd.read_csv(out / "figure_curves.csv")
for cost in ("equal", "bounded_variable"):
    ideal = figure.loc[figure.scenario.eq(cost + "_p1.0"), "mean_expected_width"].to_numpy()
    for p in (.8, .5):
        expected = figure.loc[figure.scenario.eq(cost+"_p"+str(p)), "mean_expected_width"].to_numpy()
        assert np.allclose(expected, (1-p)*ideal[0]+p*ideal)
stops = pd.read_csv(out / "stopping_replicates.csv")
keys = ["pair", "cost_scenario", "repetition"]
nominal = stops.loc[stops.max_wrong_unique_labels.eq(0)].set_index(keys)
for r in (1, 3):
    robust = stops.loc[stops.max_wrong_unique_labels.eq(r)].set_index(keys)
    assert (robust.final_lower <= nominal.final_lower+1e-10).all()
    assert (robust.final_upper >= nominal.final_upper-1e-10).all()
# Independent agreement of the efficient path with the public error utility.
rng = np.random.default_rng(2183)
checks = 0
for simulation in range(20):
    c, y = rng.normal(size=12)/10, rng.integers(0, 2, size=12)
    retained = np.zeros(12, dtype=bool)
    retained[rng.choice(12, 6, replace=False)] = True
    order = np.flatnonzero(~retained)[np.argsort(-abs(c[~retained]))]
    for r in (0, 1, 3):
        low, high = runner.error_path(c, y, retained, order, r)
        returned = {str(j): int(y[j]) for j in np.flatnonzero(retained)}
        for i in range(len(order)+1):
            result = planner.robust_completion_interval(dict(zip(map(str, range(12)), c)), returned, r)
            assert np.allclose([low[i], high[i]], [result["lower"], result["upper"]])
            checks += 1
            if i < len(order):
                returned[str(order[i])] = int(y[order[i]])
case = json.loads((out / "case_report.json").read_text(encoding="utf-8"))
for item in case["actual_archive"]:
    assert item["plan"]["selected_count"] == 1
    assert item["plan"]["irreducible_width"] > .1
    assert not item["width_0_10_feasible_by_existing_reference_recovery"]
known_case = case["controlled_first_replicate"]
assert known_case["first_interval_certified_equivalence"]["reference_count"] == 52
assert known_case["after_ideal_successful_returns"]["0"]["classification"] == "undetermined"
source_files = {"config.json": out/"config.json", "run_planning.py": source_directory/"run_planning.py",
                "installed_verifyrank.planning": Path(planner.__file__)}
report = {"all_checks_passed": True, "curve_rows_checked": len(curves), "figure_rows_checked": len(figure),
          "stopping_rows_checked": len(stops), "efficient_path_against_public_error_api_cases": checks,
          "prior_unit_tests": {"tests": 9, "all_passed": True, "source": "unittest discover test_planning.py execution on 2026-10-06 before final analysis run"},
          "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_files.items()}}
(out/"validation_report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
print(json.dumps({"validation": "passed", "path_checks": checks, "curve_rows": len(curves)}))
