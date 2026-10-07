# Current manuscript figures — 7 October 2026

Four fixed-canvas, two-panel scientific figures, each 183 × 80 mm. PDF and SVG are vectors; vector files have no intrinsic dpi. PNG files are rendered directly at 1200 dpi (8645 × 3779 pixels) and 900 dpi (6484 × 2834 pixels), without raster upsampling. SVG text remains editable. The source-data CSVs contain the values actually plotted.

The style uses the public [SciencePlots](https://github.com/garrettj403/SciencePlots) `science` and `nature` styles with explicit typography and layout settings. Arial is the first font choice, with DejaVu Sans as fallback; font substitution can change text geometry. Each script checks text clipping, text/text overlap and intersections with plotted lines, bars and markers before export. Inspect the generated previews when drawing on a different platform. These checks evaluate artwork geometry, not environmental validity.

## Redraw preserved results

Install the numerical and plotting dependencies listed at the repository root. In PowerShell, from the repository root, set a scratch directory (replace the example) and run:

```powershell
$env:VERIFYRANK_FIGURE_WORK = 'D:\scratch\verifyrank-figures-new'
1..4 | ForEach-Object { python -B "results/revision_20261007_ems_replay/figures/scripts/fig$_.py"; if ($LASTEXITCODE -ne 0) { throw 'Figure generation failed' } }
```

The default inputs are the seven frozen CSVs in `inputs/`. For figures of a newly recomputed replay, set `VERIFYRANK_ANALYSIS` to that replay output directory before running the scripts. Output artwork, layout reports and previews go to the specified scratch directory. Supplied repository figures remain untouched.

## Interpretation

- Figure 1: archive evidence counts and the decomposition of completion widths.
- Figure 2: targeted and uniform reveal curves; effort is averaged over the documented controlled masks.
- Figure 3: the discretized-cost selection reference and signed differences from the original-cost greedy selection. This is not a universal exact-optimality guarantee for the original continuous-cost problem.
- Figure 4: explicitly synthetic regime diagnostics. Reachability in panel a is a floor-feasibility criterion. Panel b averages effort only among scenarios reaching the target; it is not an unconditional recovery-effect estimate.

All derived tables and figures retain the source-specific conditions and attribution in `DATA_TERMS.md`. The figures do not create new physical observations or establish that a nondetection is zero emissions.
