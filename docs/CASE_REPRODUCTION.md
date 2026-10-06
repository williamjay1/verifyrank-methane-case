# Reproducing the frozen methane case

This repository starts from a frozen derived observation panel. It can refit the documented models and recompute the controlled masking, completion bounds and verification diagnostics without downloading original imagery or API responses. It does not by itself reconstruct raw measurements, establish historical public release times or physically validate reference labels. Keep the accompanying data attribution and source-specific conditions; the software license does not relicense derived provider data.

The public-portability edits change imports and output paths only. They preserve model settings, masks, seeds, date weights, reference grouping, cost assignments, priority scores, budget rules and stopping thresholds. The planning scripts import an installed `verifyrank.planning` package rather than loading a duplicate source tree. The bounds script accepts a user-specified writable output directory on any supported filesystem. Planning and bounds require new or empty output directories so supplied results remain a comparison baseline.

## Environment and inputs

Use Python 3.12 and the numerical `requirements-lock.txt`, and install VerifyRank 0.2.0 from the companion software release or a local checkout of that repository. Run the following commands from this case repository's root, with the desired virtual environment activated. Set Python and plotting cache directories to a suitable local scratch directory when required by your environment. On the original execution host all working outputs and temporary files were placed on D; this is a local storage choice, not part of the statistical method.

The 27,521-row `datasets/campaign_snapshots.csv` contains both campaigns and all saved scenarios. Its future outcome/scene fields are evaluation sidecar evidence, not model features. `revision_contract.py` passes only six historical fields to the policy scorer; training labels are separately filtered by their availability before fitting. Do not pass all CSV columns to an estimator. Wheels are installation artifacts, never model input data.

The saved 51,099-row prediction table is sufficient to replay fixed-score diagnostics. The 15,684-row `selection_weights_primary.csv` is a computed subset for the primary 2024 external-transfer, 25% budget and four policies; it is neither raw data nor the complete sensitivity-weight export. Model refitting uses the derived panel, not that subset.

## Fresh fitting and subsequent diagnostics

Create a new run name instead of reusing a directory that already contains outputs:

```sh
python -B scripts/revision_evaluate.py --snapshots datasets/campaign_snapshots.csv --output-dir results/recomputed_run/evaluation --bootstrap-reps 1000
python -B scripts/evaluate_masking.py --predictions results/recomputed_run/evaluation/predictions.csv --output results/recomputed_run/original_masking
python -B scripts/revision_masking.py --predictions results/recomputed_run/evaluation/predictions.csv --output results/recomputed_run/masking --freeze-only
python -B scripts/revision_masking.py --predictions results/recomputed_run/evaluation/predictions.csv --output results/recomputed_run/masking
python -B scripts/revision_bounds.py --predictions results/recomputed_run/evaluation/predictions.csv --output-dir results/recomputed_run/bounds
```

The revised masking script imports the original `evaluate_masking.py`, which imports `evaluate_campaigns.py`; keep these modules together in `scripts/`. Its optional legacy-comparison check looks for `masking_validation_20261005T153255_981415Z/replicate_metrics.csv` under the parent of the predictions directory's parent. If that original file is absent, it records the unavailable legacy comparison; the revised masking calculation still runs. Do not rename a newly generated mask file as historical evidence merely to populate that optional comparison. The original and revised experiments remain different diagnostics with documented masks and estimands.

## Recompute cost and error planning

```sh
python -B results/revision_20261006_ems/planning/run_planning.py --workspace . --output-dir results/recomputed_run/planning
python -B results/revision_20261006_ems/planning/validate_outputs.py --results-dir results/recomputed_run/planning
```

The workspace defaults to the case root inferred from the script. An explicit output directory is required. The declared `config.json` selects the preserved prediction file, and the actual-archive case reads the preserved `results/revision_20261006/bounds/reference_groups_summary.csv`. These commands intentionally reproduce the published frozen-score planning analysis. The output includes its unchanged configuration and a new timing/validation record.

To connect freshly fitted predictions and freshly recomputed bounds to planning, use a separate working copy of the case repository: place those newly generated tables at the documented relative input locations within that copy, retaining the published originals for numerical comparison elsewhere. This keeps the historical output baseline intact and avoids treating changed predictions as the preserved run. The scripts do not tune against the test outcomes or infer actual verification costs.

## Figures and result comparison

```sh
python -B scripts/ems_figures.py --part all
```

Figure generation reads the preserved bounds/masking summary CSVs and the planning figure metadata/curve file at their documented relative locations. This redraws saved results; it does not refit models. To draw recomputed results, place them in those locations only in a scratch working copy. Compare numerical tables with the preserved outputs before interpreting the new plots. Timing records and local manifests may differ even when numerical outputs agree.

The public-portability check executed the full 200-mask planning run and output validator in an independent working copy using installed VerifyRank 0.2.0. All nine generated planning CSVs, containing 40,150 rows in total, were byte-identical to the previously verified outputs. The run took 25.95 seconds, and 420 efficient-path comparisons against the public error-bound API passed. Supplied repository CSVs were not overwritten. This establishes computational parity for these portability edits; it does not establish operational recovery rates, independent physical truth or a second environmental validation setting.
