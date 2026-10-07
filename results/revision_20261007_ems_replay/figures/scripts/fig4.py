"""Figure 4: declared recoverability regimes and conditional effort summaries.

Only reads frozen analysis CSVs. Counts displayed on bars are rounded for
presentation; the full original values are preserved in the exported source data.
"""
import pandas as pd

import figstyle_nature as F


def main():
    plt = F.style()
    share = pd.read_csv(F.DATA / "regime_sweep_by_share.csv", float_precision="round_trip")
    recovery = pd.read_csv(F.DATA / "regime_sweep_by_recovery.csv", float_precision="round_trip")
    cost = pd.read_csv(F.DATA / "regime_sweep_by_cost.csv", float_precision="round_trip")
    share = share.sort_values("resolvable_share")
    recovery = recovery.set_index("recovery").loc[[0.5, 0.8, 1.0]].reset_index()
    cost = cost.set_index("cost_spread").loc[["equal", "uniform_1_5", "lognormal"]].reset_index()
    source = pd.concat([
        share.assign(panel="a", input_table="regime_sweep_by_share.csv"),
        recovery.assign(panel="b", group="recovery_probability",
                        input_table="regime_sweep_by_recovery.csv"),
        cost.assign(panel="b", group="relative_cost",
                    input_table="regime_sweep_by_cost.csv"),
    ], ignore_index=True, sort=False)

    fig = plt.figure(figsize=(F.mm(183), F.mm(80)))
    F.header(fig, "a", "Recoverability and reachability", x=.025)
    F.header(fig, "b", "Verification effort", x=.56)

    ax = fig.add_axes([.07, .25, .34, .48])
    right = ax.twinx()
    F.clean(ax)
    F.clean(right, left=False, bottom=False)
    right.spines["right"].set_visible(True)
    right.tick_params(axis="y", left=False, right=True, labelleft=False, labelright=True)
    right.tick_params(axis="x", bottom=False, top=False, labelbottom=False)
    # Two variables are fractions and deliberately use identical scale limits.
    ax.set_ylim(0, 1.05)
    right.set_ylim(0, 1.05)
    ax.set_xlim(0, 1.075)
    for one in (ax, right):
        one.set_yticks([0, .25, .5, .75, 1])
        one.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    ax.set_xticks([.05, .25, .5, 1])
    ax.set_xticklabels(["0.05", "0.25", "0.50", "1.00"])
    ax.set_xlabel("Recoverable share")
    ax.set_ylabel("Irrecoverable width share")
    right.set_ylabel("Target reachability", labelpad=6)

    # Bars remain behind the foreground floor curve. Their zero heights are real.
    bars = right.bar(share["resolvable_share"], share["reachable_rate"],
                     width=.10, color=F.C["light"], edgecolor=F.C["edge"],
                     linewidth=.45, zorder=1)
    for patch in bars:
        F.tagged(patch, "target-reachability")
    right.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_visible(False)
    line, = ax.plot(share["resolvable_share"], share["floor_share_mean"],
                    color=F.C["main"], marker="o", markerfacecolor="white",
                    markeredgewidth=.9, zorder=3)
    F.tagged(line, "irrecoverable-floor")
    target = ax.axhline(.5, color=F.C["ink"], linestyle=(0, (4, 3)),
                        linewidth=.75, zorder=2)
    F.tagged(target, "half-width-target")
    F.legend(fig, [line, bars[0]], ["Irrecoverable width share", "Target reachability"],
             x=.25, y=.847, ncol=1, labelspacing=.35)

    bax = fig.add_axes([.65, .25, .32, .48])
    F.clean(bax)
    bax.set_ylim(0, 1600)
    bax.set_yticks([0, 400, 800, 1200, 1600])
    bax.set_ylabel("References required")
    # Cost names need more horizontal room than the three numeric probabilities.
    x = [0, .65, 1.30, 2.55, 3.70, 5.05]
    vals = list(recovery["references_to_half_mean"]) + list(cost["references_to_half_mean"])
    bbars = bax.bar(x, vals, width=.51,
                    color=[F.C["third"]] * 3 + [F.C["fourth"]] * 3,
                    edgecolor="white", linewidth=.6)
    for i, (patch, value) in enumerate(zip(bbars, vals)):
        F.tagged(patch, "conditional-effort-" + str(i))
        bax.annotate(f"{value:,.0f}", (patch.get_x() + patch.get_width() / 2, value),
                     xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                     fontsize=7.8, color=F.C["ink"])
    bax.set_xlim(-.45, 5.72)
    bax.set_xticks(x)
    bax.set_xticklabels(["0.5", "0.8", "1.0", "Equal", "Uniform\n1–5", "Lognormal"],
                        fontsize=7.8)
    for centre, label in ((.65, "Recovery probability"), (3.77, "Relative costs")):
        fx = .65 + .32 * ((centre + .45) / (5.72 + .45))
        fig.text(fx, .785, label, ha="center", va="center", fontsize=7.8)

    fig.text(.50, .107, "Effort means include only simulations that attained the target",
             ha="center", va="center", fontsize=7.8)
    F.legend(fig, [target], ["Target: half of initial width"], x=.50, y=.045, ncol=1)
    F.finish(fig, 4, source)


if __name__ == "__main__":
    main()
