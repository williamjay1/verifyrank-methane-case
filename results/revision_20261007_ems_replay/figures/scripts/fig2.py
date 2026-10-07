"""Redraw Figure 2 from preserved controlled-revelation summaries only.

No experiments are rerun. The uniform 96--289 annotation is the rounded
range of the existing mean per-mask crossings over three pairs and three
targets (minimum 96.470; maximum 288.855), verified against the preserved
bounds/mask_replicate_summary.csv and recorded in data_caption_audit.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import figstyle_nature as F


def main():
    plt = F.style()
    curves = pd.read_csv(F.DATA / "controlled_reveal_curves.csv")
    summary = pd.read_csv(F.DATA / "controlled_reveal_summary.csv").set_index("pair")
    fig = plt.figure(figsize=(F.mm(183), F.mm(80)))
    ax = fig.add_axes([.075, .25, .405, .50])
    bx = fig.add_axes([.615, .25, .36, .50])
    F.header(fig, "a", "Verification return", .025)
    F.header(fig, "b", "Checking effort", .555)
    F.legend(fig, F.pair_handles(), y=.845, ncol=3)

    source = []
    uniform_dash = (0, (4.0, 2.6))
    for pair, color, marker in zip(F.PAIR_KEYS, F.PAIR_COLORS, F.PAIR_MARKERS):
        d = curves.loc[curves.pair == pair].sort_values("revealed_unique_outcomes")
        x = d.revealed_unique_outcomes.to_numpy()
        if not np.array_equal(x, np.arange(317)):
            raise ValueError(f"Expected all 317 preserved reveal counts for {pair}")
        uniform, = ax.plot(x, d.mean_uniform_expected_width, color=color,
                           linewidth=1.05, linestyle=uniform_dash, zorder=2)
        targeted, = ax.plot(x, d.mean_width, color=color, linewidth=1.4,
                            linestyle="-", marker=marker, markerfacecolor="white",
                            markeredgecolor=color, markeredgewidth=.85,
                            markersize=3.3, markevery=[0, 12, 35, 70, 125, 210, 316],
                            zorder=3)
        F.tagged(uniform, f"uniform {pair}")
        F.tagged(targeted, f"targeted {pair}")
        for column, name in [("mean_width", "targeted"),
                             ("mean_uniform_expected_width", "uniform_expected")]:
            source.extend(dict(panel="a", pair=pair, series=name,
                               returned_references=int(row.revealed_unique_outcomes),
                               value=float(getattr(row, column)), repetitions=200,
                               width_target=np.nan)
                          for row in d.itertuples(index=False))
    line = ax.axhline(.10, color=F.C["muted"], linewidth=.8,
                     linestyle=(0, (1.4, 2.2)), zorder=1)
    F.tagged(line, "width target 0.10")
    ax.set(xlim=(0, 316), ylim=(0, .65),
           xlabel="Unique reference outcomes returned", ylabel="Mean completion width")
    ax.set_xticks([0, 100, 200, 300])
    ax.set_yticks([0, .2, .4, .6])
    F.clean(ax)

    target_specs = [("needed_width_020_mean", .20),
                    ("needed_width_010_mean", .10),
                    ("needed_width_005_mean", .05)]
    centers = np.arange(3)
    bar_width = .245
    for i, (pair, color, hatch) in enumerate(zip(F.PAIR_KEYS, F.PAIR_COLORS,
                                                ["", "//", ".."])):
        vals = np.array([float(summary.loc[pair, key]) for key, _ in target_specs])
        positions = centers + (i - 1) * .27
        bars = bx.bar(positions, vals, width=bar_width, facecolor=color,
                      edgecolor=F.C["ink"], linewidth=.45, hatch=hatch, zorder=2)
        for bar, position, value, (_, target) in zip(bars, positions, vals, target_specs):
            F.tagged(bar, f"targeted {pair} target={target:g}")
            bx.text(position, value + 5.0, f"{value:.0f}", ha="center", va="bottom",
                    fontsize=7.8)
            source.append(dict(panel="b", pair=pair, series="targeted_first_crossing",
                               returned_references=np.nan, value=float(value),
                               repetitions=200, width_target=target))
    bx.set(xlim=(-.55, 2.55), ylim=(0, 235), xlabel="Width target",
           ylabel="Returned references (mean)")
    bx.set_xticks(centers, ["0.20", "0.10", "0.05"])
    bx.set_yticks([0, 50, 100, 150, 200])
    F.clean(bx)
    fig.text(.795, .777, "Uniform checking: 96–289 references", fontsize=7.8,
             ha="center", va="bottom", color=F.C["muted"])

    method_handles = [
        Line2D([], [], color=F.C["ink"], linewidth=1.4, linestyle="-",
               label="Targeted checking"),
        Line2D([], [], color=F.C["ink"], linewidth=1.05, linestyle=uniform_dash,
               label="Uniform checking"),
        Line2D([], [], color=F.C["muted"], linewidth=.8, linestyle=(0, (1.4, 2.2)),
               label="Width target 0.10"),
    ]
    F.legend(fig, method_handles, y=.045, ncol=3)
    F.finish(fig, 2, pd.DataFrame(source))
    plt.close(fig)


if __name__ == "__main__":
    main()
