"""PDF reports (the videos carry no numbers; everything is here).

progress.pdf (written by evolve.py after every batch):
    page 1  the task performance of every evaluated design (BO, then CMA-ES) and the best so far, against evaluations and
            hours
    page 2  the baseline: picture and all its data
    page 3  the best design so far: picture and all its data
<prefix>report.pdf (written by render_evaluation.py, final_report.pdf at the end of a run): with a run's re-evaluation, a
    bar plot of it as in the paper (the evaluated design, the paper's design and the baselines); a page for the evaluated design and
    one for every baseline of the paper.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

import design  # noqa: E402

## The look of the paper's figures (legacy code: src/plot_src.py). SERIF: Times-like fonts, as IEEE's (also the
## labels of the videos, render.py), the first one installed
SERIF = ["Liberation Serif", "Linux Libertine", "Times New Roman", "DejaVu Serif"]
plt.rcParams.update({"font.family": "serif", "font.size": 14, "axes.labelsize": 15, "xtick.labelsize": 13,
                     "ytick.labelsize": 13, "legend.fontsize": 13, "legend.frameon": False, "font.serif": SERIF})
OPTIMIZED_COLOR = {"A": "#56b4e9", "B": "#009e73"}[design.TASK]  # the paper's color of the optimized design of the task
GREY = (0.3, 0.3, 0.3)  # titles of the design pages of the baselines
BO_COLOR, CMA_COLOR = "#a9a9a9", "#00a2a2"
BASELINE_COLORS = {"planar": "#b83289", "franchi": "#eb7852", "shen": "#fbbf24", "rajappa": "#f0f921"}
LEGACY = f"paper's task {design.TASK} design"  # the name of this design in the re-evaluations
LEGACY_LABEL = f"Optimized Task {design.TASK}\nLegacy codebase\n(reported in paper)"
OPTIMIZED_LABEL = f"Optimized Task {design.TASK}\nThis repository"
A4 = (11.69, 8.27)  # landscape [in]


def baseline_of(name):
    """"planar" for the re-evaluation row "baseline (planar)", None for any other row."""
    m = name.removeprefix("baseline (").removesuffix(")")
    return m if m in BASELINE_COLORS else None


def row_order(name):
    """Sort key of the re-evaluation rows: as in the paper's figure, the optimized designs first."""
    return 1 if name == LEGACY else 2 + list(BASELINE_COLORS).index(baseline_of(name)) if baseline_of(name) else 0


def bars(ax, runs, colors, hatches=None):
    """The paper's bars: mean and std of the runs of every design, on an y axis cut just below them. Returns the means."""
    means, stds = [np.mean(v) for v in runs], [np.std(v) for v in runs]
    x = np.arange(len(runs))
    ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.5, width=0.6, hatch=hatches)
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black", capsize=4, linewidth=1)
    lo, hi = min(m - d for m, d in zip(means, stds)), max(m + d for m, d in zip(means, stds))
    span = hi - lo if hi > lo else 1.0
    ax.set_ylim(lo - 0.1 * span, hi + 0.15 * span)
    ax.set_xlim(-0.6, len(runs) - 0.4)
    d = 0.008  # the slashes of the cut axis
    kw = dict(transform=ax.transAxes, color="k", clip_on=False, linewidth=0.6)
    ax.plot([-d, d], [-d, d], **kw)
    ax.plot([1 - d, 1 + d], [-d, d], **kw)
    ax.set_ylabel("task performance")
    return means


def data_rows(r):
    ph = design.physical(r["design"])
    rows = [("Task performance", f"{r['fitness']:+.2f}"),
            ("  gates crossed per episode", f"{r['gates_per_episode']:.2f}"),
            ("  gates missed per episode (-10 each)", f"{r['missed_per_episode']:.3f}"),
            ("  crashes (-10 each)", f"{100 * r['crash_rate']:.1f} % of the episodes"),
            ("  evaluation episodes (5 s each)", f"{r['episodes']}"),
            ("", "")]
    if not ph["feasible"]:
        return rows + [("Not trained", ph["reason"])]
    I = ph["inertia"]
    return rows + [("Mass", f"{1000 * ph['mass']:.0f} g"),
                   ("Inertia xx, yy, zz", f"{1e4 * I[0, 0]:.1f}, {1e4 * I[1, 1]:.1f}, {1e4 * I[2, 2]:.1f} kg cm^2"),
                   ("Max force with zero torque / weight", f"{ph['thrust_to_weight']:.2f}"),
                   ("Hover attitude (tilt from level)", f"{ph['hover_tilt_deg']:.0f} deg"),
                   ("Motors moved by the repair (total)", f"{100 * ph['repair_m']:.1f} cm")]


def motor_table(fig, ph, y):
    fig.text(0.62, y, "Motor   position x, y, z [cm]      thrust axis x, y, z", fontsize=9, family="monospace")
    for k, (p, a) in enumerate(zip(ph["pos"], ph["axes"])):
        y -= 0.027
        fig.text(0.62, y, f"  {k}    {100 * p[0]:6.1f} {100 * p[1]:6.1f} {100 * p[2]:6.1f}      "
                          f"{a[0]:+.2f} {a[1]:+.2f} {a[2]:+.2f}", fontsize=9, family="monospace")


def design_page(pdf, title, color, image, r):
    fig = plt.figure(figsize=A4)
    fig.suptitle(title, fontsize=18, color=color, fontweight="bold", x=0.05, ha="left")
    ax = fig.add_axes([0.02, 0.08, 0.58, 0.80])
    ax.imshow(plt.imread(image)[..., :3])
    ax.axis("off")
    y = 0.88
    for k, v in data_rows(r):
        fig.text(0.62, y, k, fontsize=11)
        fig.text(0.98, y, v, fontsize=11, ha="right", fontweight="bold")
        y -= 0.037
    ph = design.physical(r["design"])
    if ph["feasible"]:
        motor_table(fig, ph, y - 0.03)
    pdf.savefig(fig)
    plt.close(fig)


def progress_page(pdf, csv, baseline):
    d = np.atleast_1d(np.genfromtxt(csv, delimiter=",", names=True, dtype=None, encoding=None))
    fig, axs = plt.subplots(2, 1, figsize=A4)
    fig.suptitle(f"Design optimization progress ({len(d)} evaluations)", fontsize=18, x=0.05, ha="left")
    for ax, x, xlabel in ((axs[0], d["evaluation"], "number of evaluated designs"), (axs[1], d["hours"], "time [h]")):
        for phase, c, ls in (("BO", BO_COLOR, "--"), ("CMA-ES", CMA_COLOR, "-")):
            k = np.flatnonzero(d["phase"] == phase)
            if len(k):
                ax.scatter(x[k], d["fitness"][k], s=7, color=c, alpha=0.35, linewidths=0)
                k = np.arange(max(k[0] - 1, 0), k[-1] + 1)  # the line starts where the phase before it ended
                ax.step(x[k], d["best_fitness"][k], where="post", color=c, ls=ls, lw=2.5, label=phase)
        ax.axhline(baseline["fitness"], color="k", ls="-.", lw=1.2)
        ax.text(0.99, baseline["fitness"] + 0.2, "planar", ha="right", va="bottom", transform=ax.get_yaxis_transform())
        lo = np.percentile(d["fitness"], 5) if len(d) > 5 else d["fitness"].min()
        ax.set_ylim(min(lo, baseline["fitness"]) - 1, max(np.nanmax(d["best_fitness"]), baseline["fitness"]) + 1)
        ax.set_xlim(0, x[-1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("task performance")
    axs[0].legend(loc="upper left")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _write(path, pages):
    path = Path(path)
    tmp = path.with_suffix(".tmp.pdf")
    with PdfPages(tmp) as pdf:
        for page in pages:
            page(pdf)
    tmp.rename(path)


def reevaluation_page(pdf, reevals):
    """The re-evaluations: mean and std of the multistarts of every design."""
    fig, axs = plt.subplots(len(reevals), 1, figsize=A4, squeeze=False)
    fig.suptitle("Re-evaluation of the best designs", fontsize=18, x=0.05, ha="left")
    for ax, (stage, rows) in zip(axs[:, 0], reevals):
        rows = sorted(rows, key=lambda r: row_order(r["name"]))
        colors = [BO_COLOR if stage == "BO" and r["name"].startswith("#") else
                  BASELINE_COLORS.get(baseline_of(r["name"]), OPTIMIZED_COLOR) for r in rows]
        bars(ax, [r["fitness_runs"] for r in rows], colors, ["//" if r["name"] == LEGACY else "" for r in rows])
        for k, r in enumerate(rows):
            if baseline_of(r["name"]):  # only the optimized designs: the baselines are not part of the optimization
                continue
            ax.scatter([k + 0.42], [r["fitness_before"]], marker="x", color="k", s=30, zorder=3,
                       label="during the optimization" if k == 0 else None)
        ax.set_xticks(range(len(rows)), [LEGACY_LABEL if r["name"] == LEGACY else r["name"].split()[0] if
                                         r["name"].startswith("#") else baseline_of(r["name"]).capitalize()
                                         for r in rows], fontsize=11)
        ax.set_title("after BO" if stage == "BO" else "after BO + CMA-ES", fontsize=14, loc="left")
        ax.legend(loc="upper right", fontsize=11)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def progress_pdf(path, csv, baseline, best, img_baseline, img_best, reevals):
    pages = [lambda pdf: progress_page(pdf, csv, baseline),
             lambda pdf: design_page(pdf, "Baseline (planar hexarotor)", GREY, img_baseline, baseline),
             lambda pdf: design_page(pdf, "Best design found so far (best single evaluation)", OPTIMIZED_COLOR, img_best,
                                     best)]
    if reevals:
        pages.append(lambda pdf: reevaluation_page(pdf, reevals))
    _write(path, pages)


def comparison_page(pdf, rows):
    """The paper's bars for the re-evaluation of the evaluated design and the references, with how much the evaluated
    design's mean outperforms the mean of every baseline. rows: (name, highlight (the evaluated design), re-evaluation
    row of evolve.py) per design."""
    rows = sorted(rows, key=lambda row: (not row[1], row_order(row[2]["name"])))
    names = [r["name"] for _, _, r in rows]
    fig, ax = plt.subplots(figsize=A4)
    fig.suptitle(f"Re-evaluation: {len(rows[0][2]['fitness_runs'])} independent trainings per design", fontsize=18,
                 x=0.05, ha="left")
    colors = [BASELINE_COLORS.get(baseline_of(n), OPTIMIZED_COLOR) for n in names]
    means = bars(ax, [r["fitness_runs"] for _, _, r in rows], colors, ["//" if n == LEGACY else "" for n in names])
    best = next((m for m, row in zip(means, rows) if row[1]), None)
    if best is not None:
        ax.axhline(best, color="k", ls="--", lw=0.8, zorder=1)
        for k, (m, n) in enumerate(zip(means, names)):
            if baseline_of(n):
                ax.vlines(k, m, best, colors="k", linestyles="--", linewidth=0.8)
                ax.text(k + 0.05, (m + best) / 2, f"+{100 * (best / m - 1):.1f}%" if best >= m
                        else f"{100 * (best / m - 1):.1f}%", ha="left", va="center", fontsize=14)
    ax.set_xticks(range(len(rows)), [OPTIMIZED_LABEL if h else LEGACY_LABEL if r["name"] == LEGACY else
                                     (baseline_of(r["name"]) or r["name"]).capitalize() for _, h, r in rows])
    ax.text(0.99, 0.97, f"task {design.TASK}", transform=ax.transAxes, ha="right", va="top", family="monospace",
            bbox={"facecolor": "white", "edgecolor": "black", "lw": 0.8})
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def evaluation_pdf(path, pages, comparison=None):
    """pages: (title, highlight (the evaluated design), picture, result) per design; comparison: the rows of
    comparison_page() (no such page if None)."""
    _write(path, ([lambda pdf: comparison_page(pdf, comparison)] if comparison else [])
           + [lambda pdf, p=p: design_page(pdf, p[0], OPTIMIZED_COLOR if p[1] else GREY, p[2], p[3]) for p in pages])
