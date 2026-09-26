"""Generate the paper's figures from experiments/results/*.csv."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
OUT = Path(__file__).resolve().parent / "figs"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.3, "grid.linewidth": 0.5, "lines.markersize": 4.5,
})


def read(name):
    with open(RES / name) as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            try:
                d[k] = float(v)
            except (TypeError, ValueError):
                d[k] = v
        out.append(d)
    return out


def finite(rows, key):
    return [r for r in rows if r.get(key) == r.get(key)]


def main():
    OUT.mkdir(exist_ok=True)
    ref, model, knobs = read("refinement.csv"), read("model.csv"), read("knobs.csv")

    # Sized at the final printed width so the fonts come out at their true size.
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.5, 2.6))

    # (a) error vs n
    n = [r["n_grid"] for r in ref]
    for key, style, lbl in (("e_tot", "o-", r"$e_{\mathrm{tot}}$"),
                            ("e_disc", "s--", r"$e_{\mathrm{disc}}$"),
                            ("e_alg", "^-", r"$e_{\mathrm{alg}}$")):
        rs = finite(ref, key)
        ax0.loglog([r["n_grid"] for r in rs], [r[key] for r in rs], style, label=lbl)
    for p, style in ((2, ":"), (4, "-.")):
        a = ref[0]["e_tot"] * 1.7
        ax0.loglog(n, [a * (n[0] / x) ** p for x in n], style, color="0.55",
                   lw=0.9, label=rf"$n^{{-{p}}}$")
    ax0.set_xlabel(r"grid size $n$")
    ax0.set_ylabel("relative sup error")
    ax0.set_title("(a) accuracy under refinement", fontsize=9, loc="left")
    ax0.set_xticks(n)
    ax0.set_xticklabels([f"{int(x)}" for x in n])
    ax0.set_xticks([], minor=True)
    ax0.legend(ncol=2, frameon=False, loc="lower left")

    # (b) work vs N
    N = [r["n_grid"] ** 2 for r in model]
    ax1.loglog(N, [r["flops_ml"] for r in model], "o-", label="multilevel")
    ax1.loglog(N, [r["flops_direct"] for r in model], "s-", label="direct")
    for p, style, lbl in ((1.5, "-.", r"$N^{3/2}$"), (1.0, ":", r"$N$")):
        a = model[0]["flops_ml"]
        ax1.loglog(N, [a * (x / N[0]) ** p for x in N], style, color="0.55",
                   lw=0.9, label=lbl)
    n_run = max(r["n_grid"] for r in ref) ** 2
    ax1.axvline(n_run, color="0.35", lw=0.8, ls="--")
    ax1.annotate("largest run", xy=(n_run, ax1.get_ylim()[0] * 3), rotation=90,
                 fontsize=7, ha="right", va="bottom", color="0.35")
    ax1.set_xlabel(r"image unknowns $N=n^2$")
    ax1.set_ylabel("flops")
    ax1.set_title("(b) cost", fontsize=9, loc="left")
    ax1.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT / "scaling.pdf")
    print(f"wrote {OUT / 'scaling.pdf'}")

    # Separate figure: accuracy against cost as each parameter is varied.
    fig2, ax2 = plt.subplots(figsize=(3.5, 2.6))
    styles = {"n_theta_min": ("o-", r"$n_\theta^{\min}$"),
              "s_oversample": ("s-", r"$\sigma$"),
              "t_oversample": ("^-", r"$\nu$"),
              "theta_order": ("v--", r"$q$"),
              "s_order": ("D--", r"$p_s$"),
              "t_order": ("*--", r"$p_t$")}
    for knob, (style, lbl) in styles.items():
        rs = [r for r in knobs if r["knob"] == knob and r["e_alg"] > 0]
        if rs:
            ax2.loglog([r["flops"] / 1e9 for r in rs], [r["e_alg"] for r in rs],
                       style, label=lbl, lw=1.2)
    ax2.set_xlabel("Gflop")
    ax2.set_ylabel(r"$e_{\mathrm{alg}}$")
    ax2.set_xticks([1, 2, 4, 8])
    ax2.set_xticklabels(["1", "2", "4", "8"])
    ax2.set_xticks([], minor=True)
    ax2.legend(frameon=False, ncol=2, fontsize=7)
    fig2.tight_layout()
    fig2.savefig(OUT / "knobs.pdf")
    print(f"wrote {OUT / 'knobs.pdf'}")


if __name__ == "__main__":
    main()
