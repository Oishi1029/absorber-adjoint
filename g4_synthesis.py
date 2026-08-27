"""
G4 -- there is a headline number.

Design problem (a classical Jaumann absorber, all parameters manufacturable):

    free space  ->  [ sheet R1 | spacer d1 | sheet R2 | spacer d2 | ... ]  -> PEC

  * spacers are a low-loss foam, eps_r = 1.06 - j0.002 (fixed, tabulated)
  * design variables: N sheet resistances R_k and N spacer thicknesses d_k
  * specification: |Gamma| <= -10 dB over 4-18 GHz (4.5:1), incidence 0-60 deg,
    both TE and TM

Objective (smooth surrogate for the minimax):
    F(x) = ( mean over (f,theta,pol) of |Gamma|^(2p) ) ^ (1/2p),   p = 8
so F -> max |Gamma| as p grows, and dF/dx follows from the analytic adjoint.

Comparison protocol, deliberately conservative towards the baseline:
  * identical objective, identical box bounds, identical normalised search space
  * multi-start adjoint L-BFGS-B; every restart's cost is charged
  * SciPy differential_evolution with polish=False, so no gradient information
    leaks into the baseline
  * cost is counted in FORWARD SOLVES.  One objective evaluation = 1 forward
    solve.  One objective+gradient evaluation is charged 2 forward solves
    (one forward sweep, one adjoint sweep), which is the standard accounting
    and is independent of the number of design variables.
  * the baseline is run until it first reaches the objective value the adjoint
    reached -- not merely until it converges.
"""

import json
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

import tmm_adjoint as ta

# ---------------------------------------------------------------------------
# problem definition
# ---------------------------------------------------------------------------
N_SHEET = 5
EPS_SPACER = 1.06 - 0.002j
MU_SPACER = 1.0 + 0.0j
R_BOUNDS = (5.0, 3000.0)        # ohm / square
D_BOUNDS = (0.1e-3, 8.0e-3)     # metres
P_NORM = 8

F_GRID = np.linspace(4e9, 18e9, 57)
T_GRID = np.deg2rad(np.linspace(0.0, 60.0, 7))
SPEC_DB = -10.0


def x_to_stack(x):
    """x in [0,1]^(2N): first N are sheet resistances, next N are thicknesses."""
    R = R_BOUNDS[0] + x[:N_SHEET] * (R_BOUNDS[1] - R_BOUNDS[0])
    d = D_BOUNDS[0] + x[N_SHEET:] * (D_BOUNDS[1] - D_BOUNDS[0])
    stack = []
    for k in range(N_SHEET):
        stack.append(ta.make_sheet(R[k]))
        stack.append(ta.make_line(d[k], EPS_SPACER, MU_SPACER))
    return stack


def _var_list():
    """Design variables in the same order as x."""
    v = [(2 * k, "Rs") for k in range(N_SHEET)]
    v += [(2 * k + 1, "d") for k in range(N_SHEET)]
    return v


VARS = _var_list()
SCALE = np.array([R_BOUNDS[1] - R_BOUNDS[0]] * N_SHEET +
                 [D_BOUNDS[1] - D_BOUNDS[0]] * N_SHEET)


class Counter:
    def __init__(self):
        self.fwd = 0.0          # forward-solve equivalents
        self.calls = 0
        self.trace = []         # (forward solves, best objective so far)
        self.best = np.inf

    def log(self, F):
        if F < self.best:
            self.best = F
        self.trace.append((self.fwd, self.best))


def objective(x, ctr=None):
    stack = x_to_stack(np.clip(x, 0.0, 1.0))
    s = 0.0
    n = 0
    for pol in ("TE", "TM"):
        G = ta.reflection(stack, F_GRID, T_GRID[:, None], pol)
        s += float(np.sum((np.abs(G) ** 2) ** P_NORM))
        n += G.size
    F = (s / n) ** (1.0 / (2 * P_NORM))
    if ctr is not None:
        ctr.fwd += 1.0
        ctr.calls += 1
        ctr.log(F)
    return F


def objective_and_grad(x, ctr=None):
    xc = np.clip(x, 0.0, 1.0)
    stack = x_to_stack(xc)
    s = 0.0
    n = 0
    dS = np.zeros(len(VARS))
    for pol in ("TE", "TM"):
        G, jac = ta.reflection_and_jacobian(stack, F_GRID, T_GRID[:, None],
                                            pol, VARS)
        a2 = np.abs(G) ** 2
        s += float(np.sum(a2 ** P_NORM))
        n += G.size
        w = P_NORM * a2 ** (P_NORM - 1)
        for j, k in enumerate(VARS):
            dS[j] += float(np.sum(w * 2.0 * np.real(np.conj(G) * jac[k])))
    S = s / n
    F = S ** (1.0 / (2 * P_NORM))
    g = (1.0 / (2 * P_NORM)) * S ** (1.0 / (2 * P_NORM) - 1.0) * (dS / n)
    g = g * SCALE                                  # chain rule to x in [0,1]
    if ctr is not None:
        ctr.fwd += 2.0                             # forward sweep + adjoint sweep
        ctr.calls += 1
        ctr.log(F)
    return F, g


def max_gamma_db(x):
    stack = x_to_stack(np.clip(x, 0.0, 1.0))
    worst = -np.inf
    frac = 0.0
    tot = 0
    for pol in ("TE", "TM"):
        G = ta.reflection(stack, F_GRID, T_GRID[:, None], pol)
        db = 20 * np.log10(np.abs(G))
        worst = max(worst, float(db.max()))
        frac += float(np.sum(db <= SPEC_DB))
        tot += db.size
    return worst, frac / tot


# ---------------------------------------------------------------------------
def run_adjoint(n_starts, seed=0):
    rng = np.random.default_rng(seed)
    ctr = Counter()
    best_x, best_F = None, np.inf
    per_start = []
    for i in range(n_starts):
        x0 = rng.random(2 * N_SHEET)
        res = minimize(lambda z: objective_and_grad(z, ctr), x0, jac=True,
                       method="L-BFGS-B",
                       bounds=[(0.0, 1.0)] * (2 * N_SHEET),
                       options=dict(maxiter=600, ftol=1e-14, gtol=1e-12))
        per_start.append((res.fun, res.nit, res.nfev))
        if res.fun < best_F:
            best_F, best_x = res.fun, res.x
    return ctr, best_x, best_F, per_start


def run_de(target_F, seed=0, maxiter=4000, popsize=20):
    """differential_evolution, stopped as soon as it reaches target_F."""
    ctr = Counter()
    reached = {"fwd": None, "calls": None}

    def f(x):
        v = objective(x, ctr)
        if reached["fwd"] is None and v <= target_F:
            reached["fwd"] = ctr.fwd
            reached["calls"] = ctr.calls
        return v

    def cb(*args, **kw):
        return reached["fwd"] is not None

    res = differential_evolution(
        f, bounds=[(0.0, 1.0)] * (2 * N_SHEET), seed=seed, polish=False,
        maxiter=maxiter, popsize=popsize, tol=0.0, mutation=(0.5, 1.0),
        recombination=0.7, init="sobol", callback=cb, updating="deferred")
    return ctr, res, reached


# ---------------------------------------------------------------------------
def main():
    print("=" * 86)
    print("G4  ADJOINT L-BFGS-B  vs  DIFFERENTIAL EVOLUTION, MATCHED OBJECTIVE")
    print("=" * 86)
    print(f"  {N_SHEET} resistive sheets + {N_SHEET} foam spacers "
          f"= {2*N_SHEET} design variables, PEC-backed")
    print(f"  spacer eps_r = {EPS_SPACER}  (fixed)")
    print(f"  band {F_GRID[0]/1e9:.0f}-{F_GRID[-1]/1e9:.0f} GHz "
          f"({F_GRID[-1]/F_GRID[0]:.1f}:1), {F_GRID.size} points")
    print(f"  incidence {np.rad2deg(T_GRID[0]):.0f}-{np.rad2deg(T_GRID[-1]):.0f} "
          f"deg, {T_GRID.size} angles, TE and TM")
    print(f"  objective: {2*P_NORM}-norm surrogate for max |Gamma|")
    print(f"  grid = {F_GRID.size * T_GRID.size * 2} response points per solve\n")

    n_starts = 10
    t0 = time.time()
    ctr_a, x_a, F_a, per_start = run_adjoint(n_starts, seed=1)
    t_a = time.time() - t0
    worst_db, frac = max_gamma_db(x_a)
    print(f"  ADJOINT, {n_starts} random restarts")
    print(f"    best objective F*                : {F_a:.8f}  "
          f"({20*np.log10(F_a):.3f} dB)")
    print(f"    true worst-case |Gamma|          : {worst_db:.3f} dB")
    print(f"    fraction of grid at <= {SPEC_DB:.0f} dB   : {100*frac:.2f} %")
    print(f"    objective+gradient evaluations   : {ctr_a.calls}")
    print(f"    FORWARD SOLVES (2 per f+g call)  : {ctr_a.fwd:.0f}")
    print(f"    wall clock                       : {t_a:.2f} s")
    print(f"    per-restart best F: " +
          ", ".join(f"{p[0]:.4f}" for p in per_start))

    print(f"\n  DIFFERENTIAL EVOLUTION (polish=False), run until it reaches "
          f"F <= {F_a:.8f}")
    t0 = time.time()
    ctr_d, res_d, reached = run_de(F_a, seed=1)
    t_d = time.time() - t0
    print(f"    objective evaluations used       : {ctr_d.calls}")
    print(f"    FORWARD SOLVES                   : {ctr_d.fwd:.0f}")
    print(f"    best objective reached           : {ctr_d.best:.8f}")
    print(f"    wall clock                       : {t_d:.2f} s")

    if reached["fwd"] is not None:
        ratio = reached["fwd"] / ctr_a.fwd
        print(f"\n    DE first matched F* after        : {reached['fwd']:.0f} "
              f"forward solves")
        print(f"    adjoint needed                   : {ctr_a.fwd:.0f} "
              f"forward solves")
        print(f"\n  >>> G4 SPEED-UP = {ratio:.1f}x  fewer forward solves")
        ok = ratio >= 3.0
    else:
        ratio = float(ctr_d.fwd) / ctr_a.fwd
        print(f"\n  >>> DE DID NOT REACH F* within {ctr_d.fwd:.0f} forward "
              f"solves ({ctr_d.calls} evaluations).")
        print(f"  >>> DE best {ctr_d.best:.8f} vs adjoint {F_a:.8f} "
              f"({20*np.log10(ctr_d.best)-20*np.log10(F_a):+.2f} dB worse)")
        print(f"  >>> LOWER BOUND on speed-up = {ratio:.1f}x")
        ok = ratio >= 3.0

    print(f"\n  G4 VERDICT: {'PASS' if ok else 'FAIL'} (target >= 3x)")

    # ------------------------------------------------------------------
    stack = x_to_stack(x_a)
    print("\n  synthesised absorber:")
    tot = 0.0
    for k in range(N_SHEET):
        R = stack[2 * k]["Rs"]
        d = stack[2 * k + 1]["d"]
        tot += d
        print(f"    sheet {k+1}: R = {R:8.1f} ohm/sq   spacer {k+1}: "
              f"d = {d*1e3:6.3f} mm")
    print(f"    total thickness {tot*1e3:.2f} mm "
          f"= {tot/(ta.C0/F_GRID[0]):.3f} lambda at {F_GRID[0]/1e9:.0f} GHz")

    out = dict(F_star=F_a, worst_db=worst_db, frac_meeting_spec=frac,
               adjoint_calls=ctr_a.calls, adjoint_fwd=ctr_a.fwd,
               de_calls=ctr_d.calls, de_fwd=ctr_d.fwd, de_best=ctr_d.best,
               de_fwd_to_match=reached["fwd"], ratio=ratio,
               n_vars=2 * N_SHEET, x=list(map(float, x_a)),
               t_adjoint=t_a, t_de=t_d)
    with open("g4_result.json", "w") as fh:
        json.dump(out, fh, indent=2)
    np.save("g4_trace_adjoint.npy", np.array(ctr_a.trace))
    np.save("g4_trace_de.npy", np.array(ctr_d.trace))
    print("\n  wrote g4_result.json, g4_trace_adjoint.npy, g4_trace_de.npy")
    print("=" * 86)


if __name__ == "__main__":
    main()
