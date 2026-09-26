"""Build paper/tomogrid_built.tex by substituting tables generated from the
measured CSVs into the placeholders in paper/tomogrid.tex."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
HERE = Path(__file__).resolve().parent


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


def sci(x, digits=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "---"
    if x == 0:
        return "$0$"
    e = math.floor(math.log10(abs(x)))
    m = x / 10**e
    return rf"${m:.{digits}f}\!\times\!10^{{{e}}}$"


def order(a, b, ratio=2.0):
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in (a, b)):
        return "---"
    return f"{math.log(a / b) / math.log(ratio):.2f}"


def table_refine(rows):
    body = []
    for i, r in enumerate(rows):
        prev = rows[i - 1] if i else None
        body.append(" & ".join([
            f"${int(r['n_grid'])}$", f"${int(r['n_theta'])}$",
            sci(r["e_alg"]), order(prev["e_alg"], r["e_alg"]) if prev else "---",
            sci(r["e_disc"]), order(prev["e_disc"], r["e_disc"]) if prev else "---",
            sci(r["e_tot"]), order(prev["e_tot"], r["e_tot"]) if prev else "---",
            f"{r['flops_ml'] / 1e9:.2f}", f"{r['flops_direct'] / 1e9:.2f}",
            f"{r['t_ml']:.1f}",
            "---" if math.isnan(r["t_direct"]) else f"{r['t_direct']:.1f}",
        ]) + r" \\")
    return r"""\begin{table}[t]
\centering\small
\caption{Refinement study. Activity \texttt{three\_blobs} through attenuation
\texttt{high\_contrast} (peak chord attenuation $E\approx3.1$), with
$n_\theta=n-1$, $n_s=n$, $n_\theta^{\min}=16$, $\sigma=\nu=2$. Errors are
relative sup errors; $p$ is the observed order between consecutive rows.
Direct evaluation at $n=513$ was not run --- its level-0 table alone would be
$12.9$\,GB against $0.4$\,GB for the multilevel schedule --- and its operation
count there comes from the schedule. Times are single-threaded NumPy on one core and are indicative only.}
\label{tab:refine}
\begin{tabular}{rr rr rr rr rr rr}
\toprule
& & \multicolumn{2}{c}{$e_{\mathrm{alg}}$}
  & \multicolumn{2}{c}{$e_{\mathrm{disc}}$}
  & \multicolumn{2}{c}{$e_{\mathrm{tot}}$}
  & \multicolumn{2}{c}{Gflop}
  & \multicolumn{2}{c}{seconds}\\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}
$n$ & $n_\theta$ & err & $p$ & err & $p$ & err & $p$
    & ML & direct & ML & direct\\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}"""


def table_model(rows):
    body = [
        " & ".join([
            f"${int(r['n_grid'])}$",
            f"{r['flops_ml'] / 1e9:.1f}",
            f"{r['flops_direct'] / 1e9:.1f}",
            f"{r['speedup']:.1f}",
        ]) + r" \\"
        for r in rows
    ]
    return r"""\begin{table}[t]
\centering\small
\caption{Operation count from the schedule of Section~\ref{sec:complexity},
with $n_\theta=n-1$, $n_s=n$, $n_\theta^{\min}=16$. Exact arithmetic on the
level structure, not a measurement.}
\label{tab:model}
\begin{tabular}{rrrr}
\toprule
$n$ & multilevel (Gflop) & direct (Gflop) & ratio\\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}"""


def table_knobs(rows):
    label = {"n_theta_min": r"$n_\theta^{\min}$",
             "s_oversample": r"$\sigma$", "t_oversample": r"$\nu$",
             "theta_order": r"$q$ ($\theta$ stencil)",
             "s_order": r"$p_s$ ($s$ stencil)",
             "t_order": r"$p_t$ ($t$ stencil)"}
    body, seen = [], None
    for r in rows:
        if r["knob"] != seen:
            if seen is not None:
                body.append(r"\addlinespace")
            seen = r["knob"]
        body.append(" & ".join([
            label[r["knob"]], f"${int(r['value'])}$",
            f"{r['dtheta_l_over_h']:.3f}" if r["knob"] == "n_theta_min" else "",
            sci(r["e_alg"]), f"{r['flops'] / 1e9:.2f}",
        ]) + r" \\")
    n = int(rows[0]["n_grid"])
    return rf"""\begin{{table}}[t]
\centering\small
\caption{{Accuracy parameters at $n={n}$, $n_\theta={n - 1}$; one varied at a
time from the defaults $n_\theta^{{\min}}=16$, $\sigma=2$, $\nu=2$ and
fourth-order stencils. The third column is the constant $c$
of~\eqref{{eq:angcrit}}, i.e.\ $\max_l \delta\theta_l \ell_l / h$ over the
refining levels. The upper block is the sampling parameters, the lower block
the interpolation stencil widths.}}
\label{{tab:knobs}}
\begin{{tabular}}{{llrrr}}
\toprule
knob & value & $c$ & $e_{{\mathrm{{alg}}}}$ & Gflop\\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}"""


def table_battery(rows):
    mus, acts = [], []
    for r in rows:
        if r["attenuation"] not in mus:
            mus.append(r["attenuation"])
        if r["activity"] not in acts:
            acts.append(r["activity"])
    cell = {(r["attenuation"], r["activity"]): r for r in rows}
    body = []
    for mu in mus:
        peak = max(cell[(mu, a)]["peak_E"] for a in acts)
        vals = [sci(cell[(mu, a)]["e_I"], 1) for a in acts]
        body.append(" & ".join(
            [r"\texttt{" + mu.replace("_", r"\_") + "}", f"{peak:.2f}"] + vals
        ) + r" \\")
    head = " & ".join(r"\texttt{" + a.replace("_", r"\_") + "}" for a in acts)
    worst = max(r["e_I"] for r in rows)
    return rf"""\begin{{table}}[t]
\centering\small
\caption{{Algorithmic error $e_{{\mathrm{{alg}}}}$ in $I$ over the phantom
battery, on a $33^2$ grid with $n_\theta=32$ and $n_\theta^{{\min}}=8$, i.e.\
under maximum stress. Columns are activity fields, rows attenuation fields;
$\max E$ is the largest chord attenuation. Worst entry {sci(worst, 1)}.}}
\label{{tab:battery}}
\begin{{tabular}}{{lr{"r" * len(acts)}}}
\toprule
attenuation & $\max E$ & {head}\\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}"""


def table_accuracy_cost(order_rows, sampling_rows):
    """Two sweeps side by side: stencil order on the left, sampling on the right."""
    left = [[f"${int(r['order'])}$", sci(r["e_alg"]), f"{r['flops'] / 1e9:.2f}"]
            for r in order_rows]
    right = [[f"$({int(r['n_theta_min'])},{int(r['s_oversample'])})$",
              sci(r["e_alg"]), f"{r['flops'] / 1e9:.2f}"]
             for r in sampling_rows]
    body = []
    for i in range(max(len(left), len(right))):
        a = left[i] if i < len(left) else ["", "", ""]
        b = right[i] if i < len(right) else ["", "", ""]
        body.append(" & ".join(a + b) + r" \\")
    n = int(order_rows[0]["n_grid"])
    head = (r"$p$ & $e_{\mathrm{alg}}$ & Gflop & "
            r"$(n_\theta^{\min},\sigma)$ & $e_{\mathrm{alg}}$ & Gflop\\")
    return (r"""\begin{table}[t]
\centering\small
\caption{The cost of a prescribed accuracy at $n=""" + str(n) + r"""$. Left: all
three stencil widths raised together, sampling held fixed. Right:
$n_\theta^{\min}$ and $\sigma$ raised together, stencils fixed at fourth order.
Raising the order stops paying beyond $p=4$ and then reverses; raising the
sampling keeps paying, at $W\sim\varepsilon^{-0.61}$.}
\label{tab:accuracy_cost}
\begin{tabular}{lrr@{\qquad}lrr}
\toprule
\multicolumn{3}{c}{stencil order} &
\multicolumn{3}{c}{sampling}\\
\cmidrule(lr){1-3}\cmidrule(lr){4-6}
""" + head + r"""
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}"""
            )


FIGURE = r"""\begin{figure}[t]
\centering
\includegraphics{figs/scaling.pdf}
\caption{(a) Relative sup errors against grid size, from Table~\ref{tab:refine},
with reference slopes $n^{-2}$ and $n^{-4}$. The algorithmic error
$e_{\mathrm{alg}}$ stays well below the discretisation error $e_{\mathrm{disc}}$,
which is indistinguishable from the total error $e_{\mathrm{tot}}$ at this
scale, and all converge. (b) Operation count against the number of image
unknowns $N=n^2$ from the schedule of Section~\ref{sec:complexity}, with
reference slopes $N$ and $N^{3/2}$; the dashed line marks the largest case
actually executed.}
\label{fig:scaling}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics{figs/knobs.pdf}
\caption{Algorithmic error against operation count at $n=129$ as each parameter
of Table~\ref{tab:knobs} is varied in turn (solid: sampling parameters; dashed:
stencil widths). The transverse oversampling $\sigma$ is the most efficient
direction to spend work in; $\nu$ beyond $2$ buys nothing; and the angular
stencil $q$ moves the error far less than the transverse one $p_s$.}
\label{fig:knobs}
\end{figure}"""


def main():
    ref, model, knobs, battery, order, sampling = (
        read(f"{n}.csv") for n in
        ("refinement", "model", "knobs", "battery", "order", "sampling"))
    meta = json.loads((RES / "meta.json").read_text())
    src = (HERE / "tomogrid.tex").read_text()
    for key, val in (
        ("INPUT_REFINE", table_refine(ref)),
        ("INPUT_MODEL", table_model(model)),
        ("INPUT_KNOBS", table_knobs(knobs)),
        ("INPUT_BATTERY", table_battery(battery)),
        ("INPUT_ACCURACY_COST", table_accuracy_cost(order, sampling)),
        ("INPUT_FIGURE", FIGURE),
        ("INPUT_REFDRIFT", sci(meta["reference_drift"], 1)),
    ):
        if key not in src:
            raise SystemExit(f"placeholder {key} missing from tomogrid.tex")
        src = src.replace(key, val)
    out = HERE / "tomogrid_built.tex"
    out.write_text(src)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
