"""
G1 -- the analytic adjoint is correct.

(a) PRIMARY.  Analytic adjoint dGamma/dp vs a central finite difference taken
    in 50-decimal-digit arithmetic (mpmath), for EVERY design variable of a
    >=5 layer stack, at >=3 frequencies spanning the band, >=3 incidence
    angles including 0 deg and >=50 deg, both TE and TM.  At 50 digits with a
    relative step of 1e-20 the reference carries truncation error ~1e-40 and
    round-off ~1e-30, so it is exact for this purpose.  This isolates the
    correctness of the DERIVATION from the conditioning of a double-precision
    finite difference.

(b) Analytic adjoint vs a double-precision central finite difference, with the
    step-size plateau, and the per-variable conditioning number that sets the
    floor a double-precision finite difference can reach at all.

(c) Gradient of the real scalar objective the optimiser actually uses.

(d) Cost of one full Jacobian: adjoint vs central differences.
"""

import time
import numpy as np

import tmm_adjoint as ta
import tmm_mp as tm

DPS = 50
HREL_MP = "1e-20"


# ---------------------------------------------------------------------------
def build_stacks():
    stack5 = [
        ta.make_line(1.20e-3, 6.0 - 1.10j, 2.40 - 1.70j),
        ta.make_line(0.90e-3, 11.0 - 0.40j, 1.30 - 0.25j),
        ta.make_line(2.10e-3, 3.2 - 0.90j, 3.10 - 2.20j),
        ta.make_line(1.50e-3, 15.0 - 3.30j, 1.15 - 0.10j),
        ta.make_line(0.70e-3, 2.1 - 0.05j, 1.40 - 0.60j),
    ]
    stack_j = [
        ta.make_sheet(310.0),
        ta.make_line(2.50e-3, 1.06 - 0.0020j, 1.02 - 0.0010j),
        ta.make_sheet(140.0),
        ta.make_line(3.00e-3, 1.06 - 0.0020j, 1.02 - 0.0010j),
        ta.make_sheet(90.0),
        ta.make_line(2.20e-3, 2.20 - 0.0200j, 1.05 - 0.0050j),
    ]
    return [("5-layer lossy magnetic", stack5),
            ("6-element Jaumann w/ resistive sheets", stack_j)]


def fd_jacobian(stack, f, theta, pol, variables, h_rel):
    out = {}
    for (i, nm) in variables:
        p = ta.get_param(stack, i, nm)
        h = h_rel * max(abs(p), 1e-30)
        gp = ta.reflection(ta.set_param(stack, i, nm, p + h), f, theta, pol)
        gm = ta.reflection(ta.set_param(stack, i, nm, p - h), f, theta, pol)
        out[(i, nm)] = (gp - gm) / (2 * h)
    return out


def relerr(a, b):
    den = max(abs(a), abs(b))
    return abs(a - b) / (den if den else 1.0)


# ---------------------------------------------------------------------------
def part_a(stacks, freqs, angles):
    print("=" * 86)
    print(f"G1(a)  ANALYTIC ADJOINT  vs  CENTRAL FD IN {DPS}-DIGIT ARITHMETIC "
          f"(step {HREL_MP})")
    print("=" * 86)
    print(f"  frequencies : {[x/1e9 for x in freqs]} GHz")
    print(f"  angles      : {angles} deg  (includes 0 and >= 50)")
    print(f"  polarisation: TE and TM;  adjoint evaluated in complex128\n")

    grand = 0.0
    grand_where = None
    n_checks = 0
    for name, stack in stacks:
        variables = ta.variable_names(stack)
        worst = 0.0
        where = None
        per_var_worst = {k: 0.0 for k in variables}
        for pol in ("TE", "TM"):
            for a in angles:
                th = np.deg2rad(a)
                for fq in freqs:
                    _, jac = ta.reflection_and_jacobian(
                        stack, np.array([fq]), th, pol, variables)
                    for k in variables:
                        exact = complex(tm.dgamma_dp_mp(
                            stack, k[0], k[1], fq, th, pol, DPS, HREL_MP))
                        e = relerr(jac[k][0], exact)
                        n_checks += 1
                        per_var_worst[k] = max(per_var_worst[k], e)
                        if e > worst:
                            worst, where = e, (pol, a, fq / 1e9, k)
        print(f"  {name}  ({len(variables)} design variables)")
        print(f"    worst relative error over ALL variables x freqs x angles "
              f"x pol : {worst:.3e}")
        print(f"    at {where}")
        top = sorted(per_var_worst.items(), key=lambda kv: -kv[1])[:3]
        print("    three worst variables: " +
              ", ".join(f"{k}={v:.2e}" for k, v in top))
        if worst > grand:
            grand, grand_where = worst, (name,) + where
    print(f"\n  {n_checks} independent gradient comparisons performed")
    print(f"  >>> G1(a) WORST RELATIVE ERROR, EVERY DESIGN VARIABLE: {grand:.3e}")
    print(f"  >>> at {grand_where}")
    print(f"  >>> target <= 1e-8 : {'PASS' if grand <= 1e-8 else 'FAIL'}")
    return grand


# ---------------------------------------------------------------------------
def part_b(stack, freqs, angles):
    print("\n" + "=" * 86)
    print("G1(b)  DOUBLE-PRECISION FINITE-DIFFERENCE STEP-SIZE PLATEAU")
    print("=" * 86)
    variables = ta.variable_names(stack)
    th = np.deg2rad(50.0)
    pol = "TM"
    f = np.asarray(freqs)
    Gam, jac_a = ta.reflection_and_jacobian(stack, f, th, pol, variables)

    # conditioning: kappa_p = |Gamma| / |p dGamma/dp|.  A central difference in
    # precision eps cannot beat ~3 (eps kappa_p)^(2/3) relative accuracy.
    kappa = {}
    for k in variables:
        p = ta.get_param(stack, k[0], k[1])
        kappa[k] = float(np.max(np.abs(Gam) / np.maximum(
            np.abs(p * jac_a[k]), 1e-300)))
    eps = np.finfo(float).eps
    print("  conditioning of each variable (kappa_p = |Gamma| / |p dGamma/dp|)")
    ranked = sorted(kappa.items(), key=lambda kv: -kv[1])
    for k, v in ranked[:3]:
        print(f"    worst  {str(k):>18} kappa = {v:9.3e}  -> FD floor "
              f"~ {3*(eps*v)**(2/3):.1e}")
    for k, v in ranked[-2:]:
        print(f"    best   {str(k):>18} kappa = {v:9.3e}  -> FD floor "
              f"~ {3*(eps*v)**(2/3):.1e}")

    well = [k for k in variables if kappa[k] <= 1e2]
    print(f"\n  {len(well)} of {len(variables)} variables have kappa_p <= 1e2 "
          f"(well-conditioned for a double-precision FD)\n")

    print(f"  {'h_rel':>9} | {'all vars':>12} | {'kappa<=1e2':>12} | trend (all)")
    print("  " + "-" * 62)
    rows = []
    prev = None
    for e in range(-1, -13, -1):
        h = 10.0 ** e
        jac_fd = fd_jacobian(stack, f, th, pol, variables, h)
        w_all = 0.0
        w_well = 0.0
        for k in variables:
            d = float(np.max(np.abs(jac_a[k] - jac_fd[k]) /
                             np.maximum(np.abs(jac_a[k]), np.abs(jac_fd[k]))))
            w_all = max(w_all, d)
            if k in well:
                w_well = max(w_well, d)
        tr = "" if prev is None else f"x{prev/w_all:8.1f}"
        print(f"  {h:9.0e} | {w_all:12.3e} | {w_well:12.3e} | {tr}")
        rows.append((h, w_all, w_well))
        prev = w_all

    floor_all = min(r[1] for r in rows)
    floor_well = min(r[2] for r in rows)
    flat_all = [r[0] for r in rows if r[1] <= 10 * floor_all]
    flat_well = [r[0] for r in rows if r[2] <= 10 * floor_well]
    dec_all = np.log10(max(flat_all) / min(flat_all)) if len(flat_all) > 1 else 0
    dec_well = np.log10(max(flat_well) / min(flat_well)) if len(flat_well) > 1 else 0
    print(f"\n  all variables      : floor {floor_all:.3e}, plateau "
          f"{dec_all:.1f} decades of h")
    print(f"  kappa_p <= 1e2 only: floor {floor_well:.3e}, plateau "
          f"{dec_well:.1f} decades of h")
    print(f"  >>> G1(b) plateau >= 1 decade : "
          f"{'PASS' if dec_all >= 1.0 else 'FAIL'}")
    print("  >>> the h^2 fall then the h^-1 rise is the textbook truncation /")
    print("      round-off crossover: the residual at the floor is round-off in")
    print("      the FD quotient, not error in the analytic gradient (see G1(a)).")
    return floor_all, floor_well, dec_all


# ---------------------------------------------------------------------------
def objective_and_grad(stack, fgrid, tgrid, variables, want_grad=True):
    F = 0.0
    grad = {k: 0.0 for k in variables}
    n = 0
    for pol in ("TE", "TM"):
        if want_grad:
            G, jac = ta.reflection_and_jacobian(stack, fgrid, tgrid[:, None],
                                                pol, variables)
        else:
            G, jac = ta.reflection(stack, fgrid, tgrid[:, None], pol), None
        F += float(np.sum(np.abs(G) ** 2))
        n += G.size
        if want_grad:
            for k in variables:
                grad[k] += float(np.sum(2.0 * np.real(np.conj(G) * jac[k])))
    F /= n
    if want_grad:
        for k in variables:
            grad[k] /= n
    return F, grad


def part_c(stack):
    print("\n" + "=" * 86)
    print("G1(c)  GRADIENT OF THE REAL OBJECTIVE  F = mean over (f,theta,pol) "
          "of |Gamma|^2")
    print("=" * 86)
    variables = ta.variable_names(stack)
    fgrid = np.linspace(2e9, 18e9, 51)
    tgrid = np.deg2rad(np.linspace(0.0, 60.0, 7))
    F, gradF = objective_and_grad(stack, fgrid, tgrid, variables)
    ga = np.array([gradF[k] for k in variables])
    gfd = np.zeros(len(variables))
    for j, (i, nm) in enumerate(variables):
        p = ta.get_param(stack, i, nm)
        h = 1e-5 * abs(p)
        Fp, _ = objective_and_grad(ta.set_param(stack, i, nm, p + h),
                                   fgrid, tgrid, variables, False)
        Fm, _ = objective_and_grad(ta.set_param(stack, i, nm, p - h),
                                   fgrid, tgrid, variables, False)
        gfd[j] = (Fp - Fm) / (2 * h)
    rel = np.abs(ga - gfd) / np.maximum(np.abs(ga), np.abs(gfd))
    print(f"  F = {F:.12e}   over {fgrid.size} freqs x {tgrid.size} angles "
          f"x 2 pol = {fgrid.size*tgrid.size*2} response points")
    print(f"  worst per-variable relative error vs double-precision FD : "
          f"{rel.max():.3e}  (variable {variables[int(np.argmax(rel))]})")
    print(f"  median per-variable relative error                       : "
          f"{np.median(rel):.3e}")
    print(f"  ||ga - gfd||_inf / ||ga||_inf                            : "
          f"{np.abs(ga-gfd).max()/np.abs(ga).max():.3e}")
    return rel


def part_d(stacks):
    print("\n" + "=" * 86)
    print("G1(d)  COST OF ONE FULL JACOBIAN (all N design variables)")
    print("=" * 86)
    fgrid = np.linspace(2e9, 18e9, 51)
    tgrid = np.deg2rad(np.linspace(0.0, 60.0, 7))[:, None]
    out = []
    for name, stack in stacks:
        variables = ta.variable_names(stack)
        n = len(variables)
        t0 = time.perf_counter()
        for _ in range(50):
            ta.reflection(stack, fgrid, tgrid, "TM")
        t_fwd = (time.perf_counter() - t0) / 50
        t0 = time.perf_counter()
        for _ in range(50):
            ta.reflection_and_jacobian(stack, fgrid, tgrid, "TM", variables)
        t_adj = (time.perf_counter() - t0) / 50
        t0 = time.perf_counter()
        for _ in range(5):
            fd_jacobian(stack, fgrid, tgrid, "TM", variables, 1e-5)
        t_fd = (time.perf_counter() - t0) / 5
        print(f"  {name}: N = {n} design variables")
        print(f"    one forward solve                : {t_fwd*1e3:8.3f} ms")
        print(f"    adjoint: Gamma + ALL N gradients : {t_adj*1e3:8.3f} ms = "
              f"{t_adj/t_fwd:6.2f} forward solves")
        print(f"    central FD: ALL N gradients      : {t_fd*1e3:8.3f} ms = "
              f"{t_fd/t_fwd:6.2f} forward solves   (2N = {2*n})")
        print(f"    Jacobian speed-up                : {t_fd/t_adj:6.1f}x")
        out.append((name, n, t_adj / t_fwd, t_fd / t_fwd, t_fd / t_adj))
    return out


def main():
    freqs = [3.0e9, 8.0e9, 12.5e9, 18.0e9]
    angles = [0.0, 25.0, 50.0, 65.0]
    stacks = build_stacks()
    t0 = time.time()
    g_a = part_a(stacks, freqs, angles)
    fl_all, fl_well, dec = part_b(stacks[0][1], freqs, angles)
    rel_c = part_c(stacks[0][1])
    part_d(stacks)
    print("\n" + "=" * 86)
    ok = (g_a <= 1e-8) and (dec >= 1.0)
    print(f"G1 VERDICT: {'PASS' if ok else 'FAIL'}")
    print(f"  adjoint vs exact ({DPS}-digit) FD, every variable : {g_a:.3e}   "
          f"(criterion <= 1e-8)")
    print(f"  double-precision FD floor, all variables          : {fl_all:.3e}")
    print(f"  double-precision FD floor, well-conditioned vars  : {fl_well:.3e}")
    print(f"  step-size plateau                                 : {dec:.1f} decades")
    print(f"  wall clock {time.time()-t0:.1f} s")
    print("=" * 86)


if __name__ == "__main__":
    main()
