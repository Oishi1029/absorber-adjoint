"""
The three paper figures, IEEE single-column geometry.

Fig. 1  Gradient verification: FD step-size sweep with truncation and round-off
        branches, the conditioning-predicted floor, and the 50-digit reference
        agreement far below.
Fig. 2  |Gamma| in dB against frequency at four incidence angles, TE and TM,
        for the synthesised five-sheet absorber.
Fig. 3  Forward solves to reach a matched objective against the number of
        design variables: adjoint L-BFGS-B vs the best-tuned differential
        evolution.  (Skipped gracefully until g4_scaling.json exists.)

Conventions: column width 3.379 in (243.3 pt), Times, 8 pt, thin marks,
recessive grid, direct labels, colour never the sole carrier of identity
(linestyle and lightness order survive grayscale printing).
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tmm_adjoint as ta

# ----------------------------------------------------------------- style ----
COL_W = 3.379                       # in
BLUE, VERM = "#0072B2", "#D55E00"   # validated pair
SEQ = ["#a6bddb", "#67a9cf", "#2b8cbe", "#045a8d"]   # single-hue, light->dark
GRAY = "#666666"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.1,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.4,
    "legend.frameon": False,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
})

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight",
                    pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf + .png")


# ------------------------------------------------------------------ fig 1 ---
def fig1():
    stack = [
        ta.make_line(1.20e-3, 6.0 - 1.10j, 2.40 - 1.70j),
        ta.make_line(0.90e-3, 11.0 - 0.40j, 1.30 - 0.25j),
        ta.make_line(2.10e-3, 3.2 - 0.90j, 3.10 - 2.20j),
        ta.make_line(1.50e-3, 15.0 - 3.30j, 1.15 - 0.10j),
        ta.make_line(0.70e-3, 2.1 - 0.05j, 1.40 - 0.60j),
    ]
    variables = ta.variable_names(stack)
    f = np.array([3.0e9, 8.0e9, 12.5e9, 18.0e9])
    th, pol = np.deg2rad(50.0), "TM"
    Gam, jac_a = ta.reflection_and_jacobian(stack, f, th, pol, variables)

    kappa = {}
    for k in variables:
        p = ta.get_param(stack, k[0], k[1])
        kappa[k] = float(np.max(np.abs(Gam) /
                                np.maximum(np.abs(p * jac_a[k]), 1e-300)))
    well = [k for k in variables if kappa[k] <= 1e2]

    hs = np.logspace(-12, -1, 34)
    w_all, w_well = [], []
    for h in hs:
        wa = ww = 0.0
        for (i, nm) in variables:
            p = ta.get_param(stack, i, nm)
            hh = h * max(abs(p), 1e-30)
            gp = ta.reflection(ta.set_param(stack, i, nm, p + hh), f, th, pol)
            gm = ta.reflection(ta.set_param(stack, i, nm, p - hh), f, th, pol)
            fd = (gp - gm) / (2 * hh)
            a = jac_a[(i, nm)]
            e = float(np.max(np.abs(a - fd) /
                             np.maximum(np.abs(a), np.abs(fd))))
            wa = max(wa, e)
            if (i, nm) in well:
                ww = max(ww, e)
        w_all.append(wa)
        w_well.append(ww)
    w_all, w_well = np.array(w_all), np.array(w_well)

    fig, ax = plt.subplots(figsize=(COL_W, 2.45))

    ax.plot(hs, w_all, c=VERM, marker="o", ms=2.4, mew=0,
            label="all 25 variables")
    ax.plot(hs, w_well, c=BLUE, marker="s", ms=2.2, mew=0, ls="--",
            label=r"$\kappa_p \leq 10^{2}$ (18 variables)")

    # branch labels written along the measured curves themselves
    ax.text(9.0e-4, 3.5e-4, r"truncation $\propto h^{2}$", color=GRAY,
            fontsize=7, rotation=44, rotation_mode="anchor", ha="left")
    ax.text(2.7e-10, 1.15e-5, r"round-off $\propto h^{-1}$", color=GRAY,
            fontsize=7, rotation=-31, rotation_mode="anchor", ha="left")

    ax.axhline(7.83e-14, c="k", ls=":", lw=0.9)
    ax.text(2.5e-7, 1.6e-13,
            "adjoint vs 50-digit reference:  $7.8\\times10^{-14}$",
            fontsize=7, ha="center", va="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(3e-13, 3e-1)
    ax.set_ylim(1e-14, 3e0)
    ax.set_xlabel(r"relative finite-difference step $h$")
    ax.set_ylabel("worst relative gradient error")
    ax.grid(True, which="major")
    ax.legend(loc="lower left", bbox_to_anchor=(0.01, 0.13),
              handlelength=1.8, fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "fig1_plateau")


# ------------------------------------------------------------------ fig 2 ---
def fig2():
    import g4_synthesis as g4
    with open("g4_result.json") as fh:
        r = json.load(fh)
    stack = g4.x_to_stack(np.array(r["x"]))
    fq = np.linspace(4e9, 18e9, 561)
    angles = [0, 30, 45, 60]

    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 2.9), sharex=True,
                             gridspec_kw=dict(hspace=0.12))
    for ax, pol, ls in zip(axes, ("TE", "TM"), ("-", "-")):
        for a, c in zip(angles, SEQ):
            g = ta.reflection(stack, fq, np.deg2rad(a), pol)
            ax.plot(fq / 1e9, 20 * np.log10(np.abs(g)), ls, c=c, lw=1.0)
        ax.axhline(-10, c=GRAY, ls="--", lw=0.8)
        ax.set_ylim(-62, 0)
        ax.set_yticks([0, -10, -20, -30, -40, -50, -60])
        ax.grid(True)
        ax.text(0.985, 0.06, pol, transform=ax.transAxes, ha="right",
                fontsize=8, fontweight="bold")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].text(17.85, -8.6, "specification", color=GRAY, fontsize=6.5,
                 ha="right", va="bottom")
    # direct labels for the angle ramp, once, on the top panel
    handles = [plt.Line2D([], [], c=c, lw=1.2) for c in SEQ]
    axes[0].legend(handles, [f"{a}$^\\circ$" for a in angles], ncol=4,
                   loc="lower center", bbox_to_anchor=(0.5, 0.98),
                   columnspacing=1.0, handlelength=1.3, fontsize=7)
    axes[1].set_xlabel("frequency (GHz)")
    fig.supylabel(r"reflection coefficient $|\Gamma|$ (dB)", fontsize=8, x=0.02)
    save(fig, "fig2_absorber")


# ------------------------------------------------------------------ fig 3 ---
def fig3():
    if not os.path.exists("g4_scaling.json"):
        print("  fig3: g4_scaling.json not present yet -- skipped")
        return False
    with open("g4_scaling.json") as fh:
        rows = json.load(fh)
    nv = [r["nvar"] for r in rows]
    adj = [r["tol_0.25"]["adjoint"] for r in rows]
    de = [r["tol_0.25"]["de"] for r in rows]

    fig, ax = plt.subplots(figsize=(COL_W, 2.35))
    ax.plot(nv, de, c=VERM, marker="s", ms=3.4, mew=0,
            label="differential evolution (best-tuned)")
    ax.plot(nv, adj, c=BLUE, marker="o", ms=3.4, mew=0,
            label="adjoint L-BFGS-B")
    for x, a, d in zip(nv, adj, de):
        if a and d:
            ax.annotate(f"{d/a:.0f}$\\times$", xy=(x, np.sqrt(a * d)),
                        ha="center", va="center", fontsize=7, color="#333333")
    ax.set_yscale("log")
    ax.set_xticks(nv)
    ax.set_xlabel("number of design variables")
    ax.set_ylabel("forward solves to matched objective")
    ax.grid(True, which="major")
    ax.legend(loc="upper left", fontsize=7, handlelength=1.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "fig3_scaling")
    return True


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "1"):
        fig1()
    if which in ("all", "2"):
        fig2()
    if which in ("all", "3"):
        fig3()
