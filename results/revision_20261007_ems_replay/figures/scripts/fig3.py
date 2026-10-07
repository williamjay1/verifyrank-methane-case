"""Figure 3: fixed-size cost comparison with signed gaps preserved."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import figstyle_nature as F


def main():
    plt = F.style()
    raw = pd.read_csv(F.DATA / "exact_vs_greedy.csv")
    scenarios = [
        ("equal_cost", "Equal unit cost", F.C["main"], "o", ""),
        ("cost_scaled_by_coefficient", "Coefficient-scaled cost", F.C["alt"], "s", "//"),
        ("lognormal_cost", "Lognormal cost", F.C["third"], "^", ".."),
    ]
    # The reference solver discretizes costs. Negative gaps therefore remain
    # informative and must not be clipped to zero or described as impossible.
    aggregate = (
        raw.groupby(["scenario", "budget_fraction"], sort=True)
        .agg(
            greedy_width_removed_mean=("greedy_width_removed", "mean"),
            discretized_optimum_width_removed_mean=("exact_width_removed", "mean"),
            relative_gap_mean=("relative_gap", "mean"),
            policy_pairs=("pair", "nunique"),
        )
        .reset_index()
    )
    assert (aggregate.policy_pairs == 3).all()
    assert len(aggregate) == 123
    aggregate["signed_gap_percent_mean"] = 100.0 * aggregate.relative_gap_mean

    fig = plt.figure(figsize=(F.mm(183), F.mm(80)))
    ax = fig.add_axes([.075, .25, .40, .50])
    gap_ax = fig.add_axes([.615, .25, .355, .50])
    F.header(fig, "a", "Cost-constrained verification", .025)
    F.header(fig, "b", "Signed gap to reference", .555)

    greedy_dash = (0, (4.2, 2.0))
    legend_handles = []
    for scenario, label, colour, marker, _ in scenarios:
        rows = aggregate.loc[aggregate.scenario.eq(scenario)].sort_values("budget_fraction")
        F.tagged(ax.plot(
            rows.budget_fraction,
            rows.discretized_optimum_width_removed_mean,
            color=colour, linestyle="-", lw=1.35,
            marker=marker, markevery=(2, 8), markersize=3.3,
            markerfacecolor="white", markeredgewidth=.8,
        )[0], scenario + ": discretized optimum")
        F.tagged(ax.plot(
            rows.budget_fraction, rows.greedy_width_removed_mean,
            color=colour, linestyle=greedy_dash, lw=1.10,
            marker=marker, markevery=(6, 8), markersize=3.3,
            markerfacecolor="white", markeredgewidth=.8,
        )[0], scenario + ": greedy")
        legend_handles.append(Line2D(
            [], [], color=colour, marker=marker, linestyle="none",
            markerfacecolor="white", markeredgewidth=.85,
            markersize=4.0, label=label,
        ))

    ax.set(xlim=(0, 1), ylim=(0, .4))
    ax.set_xticks(np.arange(0, 1.01, .2))
    ax.set_yticks(np.arange(0, .401, .1))
    ax.set_xlabel("Budget share of total\nverification cost")
    ax.set_ylabel("Width removed (mean)")
    F.clean(ax)

    samples = np.arange(.05, 1., .10)
    xpos = np.arange(len(samples))
    bar_width = .24
    sampled_frames = []
    sampled_values = []
    for j, (scenario, _, colour, _, hatch) in enumerate(scenarios):
        group = aggregate.loc[aggregate.scenario.eq(scenario)]
        rows = []
        for budget in samples:
            chosen = group.loc[np.isclose(group.budget_fraction, budget, atol=1e-10, rtol=0)]
            assert len(chosen) == 1
            rows.append(chosen.iloc[0])
        sampled = pd.DataFrame(rows)
        values = sampled.signed_gap_percent_mean.to_numpy()
        sampled_values.extend(values.tolist())
        bars = gap_ax.bar(
            xpos + (j - 1) * bar_width, values, width=bar_width * .93,
            color=colour, edgecolor=F.C["ink"], linewidth=.40,
            hatch=hatch, zorder=3,
        )
        for bar in bars:
            F.tagged(bar, scenario + ": signed gap")
        sampled_frames.append(sampled)

    sampled = pd.concat(sampled_frames, ignore_index=True)
    assert len(sampled) == 30
    assert min(sampled_values) < 0, "Do not silently remove signed negative gaps."
    peak = max(sampled_values)
    F.tagged(gap_ax.axhline(0, color=F.C["edge"], lw=.65, zorder=2), "zero baseline")
    gap_ax.set(xlim=(-.62, len(samples) - .38), ylim=(-.6, 5.8))
    gap_ax.set_xticks(xpos)
    gap_ax.set_xticklabels([f"{budget:.2f}" for budget in samples], fontsize=7.8)
    gap_ax.set_yticks([0, 1, 2, 3, 4, 5])
    gap_ax.set_xlabel("Budget share of total cost")
    gap_ax.set_ylabel("Relative gap (%)")
    F.clean(gap_ax)
    # Keep the numerical annotation outside the data rectangle.
    fig.text(.97, .777, f"Sampled mean peak {peak:.1f}%",
             ha="right", va="bottom", fontsize=7.8, color=F.C["muted"])

    F.legend(fig, legend_handles, y=.85, ncol=3,
             handlelength=1.1, columnspacing=2.1)
    method_handles = [
        Line2D([], [], color=F.C["ink"], linestyle="-", lw=1.35,
               label="Discretized optimum"),
        Line2D([], [], color=F.C["ink"], linestyle=greedy_dash, lw=1.10,
               label="Greedy priority order"),
    ]
    F.legend(fig, method_handles, y=.045, ncol=2, handlelength=2.4,
             columnspacing=3.)

    source = pd.concat([
        aggregate.assign(panel="a", statistic="mean over three policy pairs"),
        sampled.assign(panel="b", statistic="signed percentage gap; mean over three policy pairs"),
    ], ignore_index=True)
    print("Source records:", len(raw), "curve rows:", len(aggregate), "bar rows:", len(sampled))
    print("Signed sampled gap range (%):", min(sampled_values), peak)
    print("Maximum plotted removed width:", aggregate[[
        "greedy_width_removed_mean", "discretized_optimum_width_removed_mean"
    ]].max().max())
    F.finish(fig, 3, source)
    plt.close(fig)


if __name__ == "__main__":
    main()
