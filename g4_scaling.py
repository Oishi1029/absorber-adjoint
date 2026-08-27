"""
G4, done properly.

The first attempt compared "adjoint reaches F* exactly" against "differential
evolution reaches F* exactly", and charged the adjoint for ten restarts it did
not need.  Both sides of that comparison were wrong, in opposite directions.

This script fixes it:

  * BOTH methods are instrumented identically.  Every objective evaluation is
    logged with its running forward-solve cost, and for each method we record
    the cost at which it FIRST reaches each target objective value.  The
    targets are F* + t dB for a ladder of tolerances t, so like is compared
    with like.
  * The adjoint's restarts are charged cumulatively: if restart 1 succeeds, the
    adjoint is charged only for restart 1.
  * The derivative-free baseline is given its best hyper-parameters, taken from
    the sweep in g4_fairness.py (small population, immediate updating, which is
    markedly stronger than SciPy's defaults on this problem).
  * The experiment is repeated across problem sizes, because the claim being
    tested is not "the adjoint is faster" but "the adjoint's advantage grows
    with the number of design variables", which is what the O(N)-versus-2N
    argument actually predicts.
"""

import json
import sys
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

import tmm_adjoint as ta

# ---------------------------------------------------------------------------
EPS_SPACER = 1.06 - 0.002j
MU_SPACER = 1.0 + 0.0j
R_BOUNDS = (5.0, 3000.0)
D_BOUNDS = (0.1e-3, 6.0e-3)
P_NORM = 8

F_GRID = np.linspace(3e9, 18e9, 61)
T_GRID = np.deg2rad(np.linspace(0.0, 70.0, 8))
TOL_DB = [2.0, 1.0, 0.5, 0.25]
BUDGET = 400_000.0


class Problem:
    def __init__(self, n_sheet):
        self.n = n_sheet
        self.nvar = 2 * n_sheet
        self.vars = [(2 * k, "Rs") for k in range(n_sheet)] + \
                    [(2 * k + 1, "d") for k in range(n_sheet)]
        self.scale = np.array([R_BOUNDS[1] - R_BOUNDS[0]] * n_sheet +
                              [D_BOUNDS[1] - D_BOUNDS[0]] * n_sheet)

    def stack(self, x):
        x = np.clip(x, 0.0, 1.0)
        R = R_BOUNDS[0] + x[:self.n] * (R_BOUNDS[1] - R_BOUNDS[0])
        d = D_BOUNDS[0] + x[self.n:] * (D_BOUNDS[1] - D_BOUNDS[0])
        s = []
        for k in range(self.n):
            s.append(ta.make_sheet(R[k]))
            s.append(ta.make_line(d[k], EPS_SPACER, MU_SPACER))
        return s

    def F(self, x):
        st = self.stack(x)
        s = 0.0
        n = 0
        for pol in ("TE", "TM"):
            G = ta.reflection(st, F_GRID, T_GRID[:, None], pol)
            s += float(np.sum((np.abs(G) ** 2) ** P_NORM))
            n += G.size
        return (s / n) ** (1.0 / (2 * P_NORM))

    def F_and_grad(self, x):
        st = self.stack(x)
        s = 0.0
        n = 0
        dS = np.zeros(self.nvar)
        for pol in ("TE", "TM"):
            G, jac = ta.reflection_and_jacobian(st, F_GRID, T_GRID[:, None],
                                                pol, self.vars)
            a2 = np.abs(G) ** 2
            s += float(np.sum(a2 ** P_NORM))
            n += G.size
            w = P_NORM * a2 ** (P_NORM - 1)
            for j, k in enumerate(self.vars):
                dS[j] += float(np.sum(w * 2.0 * np.real(np.conj(G) * jac[k])))
        S = s / n
        F = S ** (1.0 / (2 * P_NORM))
        g = (1.0 / (2 * P_NORM)) * S ** (1.0 / (2 * P_NORM) - 1.0) * (dS / n)
        return F, g * self.scale

    def worst_db(self, x):
        st = self.stack(x)
        w = -np.inf
        for pol in ("TE", "TM"):
            G = ta.reflection(st, F_GRID, T_GRID[:, None], pol)
            w = max(w, float((20 * np.log10(np.abs(G))).max()))
        return w


class Trace:
    """Identical instrumentation for both methods."""

    def __init__(self):
        self.fwd = 0.0
        self.pts = []       # (cumulative forward solves, best-so-far F)
        self.best = np.inf

    def note(self, F, cost):
        self.fwd += cost
        if F < self.best:
            self.best = F
        self.pts.append((self.fwd, self.best))

    def cost_to_reach(self, target):
        for c, b in self.pts:
            if b <= target:
                return c
        return None


def run_adjoint(pb, n_starts, seed):
    rng = np.random.default_rng(seed)
    tr = Trace()
    best_x, best_F = None, np.inf
    n_success = 0
    for i in range(n_starts):
        x0 = rng.random(pb.nvar)

        def fg(z):
            F, g = pb.F_and_grad(z)
            tr.note(F, 2.0)          # one forward sweep + one adjoint sweep
            return F, g

        res = minimize(fg, x0, jac=True, method="L-BFGS-B",
                       bounds=[(0.0, 1.0)] * pb.nvar,
                       options=dict(maxiter=800, ftol=1e-14, gtol=1e-12))
        if res.fun < best_F:
            best_F, best_x = res.fun, res.x
    for i in range(n_starts):
        pass
    return tr, best_x, best_F


def adjoint_restart_stats(pb, n_starts, seed, F_star):
    """How often does a single random start reach F*?"""
    rng = np.random.default_rng(seed)
    hits, costs = 0, []
    for _ in range(n_starts):
        tr = Trace()
        x0 = rng.random(pb.nvar)

        def fg(z):
            F, g = pb.F_and_grad(z)
            tr.note(F, 2.0)
            return F, g

        res = minimize(fg, x0, jac=True, method="L-BFGS-B",
                       bounds=[(0.0, 1.0)] * pb.nvar,
                       options=dict(maxiter=800, ftol=1e-14, gtol=1e-12))
        if res.fun <= F_star * 1.001:
            hits += 1
            costs.append(tr.fwd)
    return hits, n_starts, (float(np.median(costs)) if costs else None)


def run_de(pb, seed, popsize, updating, target_exact):
    tr = Trace()

    def f(x):
        v = pb.F(x)
        tr.note(v, 1.0)
        return v

    def cb(*a, **k):
        return (tr.best <= target_exact) or (tr.fwd >= BUDGET)

    differential_evolution(f, bounds=[(0.0, 1.0)] * pb.nvar, seed=seed,
                           polish=False, maxiter=100000, popsize=popsize,
                           tol=0.0, mutation=(0.5, 1.0), recombination=0.7,
                           init="latinhypercube", callback=cb,
                           updating=updating)
    return tr


# ---------------------------------------------------------------------------
def main():
    sizes = [3, 5, 7, 9, 11]
    n_starts = 5
    print("=" * 92)
    print("G4  SCALING OF THE ADJOINT ADVANTAGE WITH PROBLEM SIZE")
    print("=" * 92)
    print(f"  band {F_GRID[0]/1e9:.0f}-{F_GRID[-1]/1e9:.0f} GHz "
          f"({F_GRID[-1]/F_GRID[0]:.0f}:1), {F_GRID.size} points; "
          f"incidence 0-{np.rad2deg(T_GRID[-1]):.0f} deg, {T_GRID.size} angles; TE and TM")
    print(f"  objective: {2*P_NORM}-norm surrogate for worst-case |Gamma|")
    print(f"  BOTH methods instrumented identically; cost = forward solves")
    print(f"  adjoint charged 2 forward solves per objective+gradient call")
    print(f"  adjoint restarts charged cumulatively; DE given its best "
          f"hyper-parameters\n", flush=True)

    out = []
    for n in sizes:
        pb = Problem(n)
        t0 = time.time()
        tr_a, x_a, F_star = run_adjoint(pb, n_starts, seed=7)
        hits, tot, med = adjoint_restart_stats(pb, 10, 21, F_star)
        wdb = pb.worst_db(x_a)
        targets = {t: F_star * 10 ** (t / 20.0) for t in TOL_DB}

        best_de = {t: (np.inf, None) for t in TOL_DB}
        for popsize, upd in ((8, "immediate"), (15, "immediate"),
                             (15, "deferred")):
            tr_d = run_de(pb, 7, popsize, upd, F_star)
            for t in TOL_DB:
                c = tr_d.cost_to_reach(targets[t])
                if c is not None and c < best_de[t][0]:
                    best_de[t] = (c, (popsize, upd))

        print(f"  --- {pb.nvar} design variables ({n} sheets + {n} spacers) "
              f"---   [{time.time()-t0:.0f} s]")
        print(f"      F* = {F_star:.6f}  ({20*np.log10(F_star):+.2f} dB), "
              f"true worst-case |Gamma| = {wdb:+.2f} dB")
        print(f"      single random start reached F* in {hits}/{tot} trials"
              + (f", median cost {med:.0f} forward solves" if med else ""))
        row = dict(nvar=pb.nvar, F_star=F_star, worst_db=wdb,
                   restart_hits=hits, restart_trials=tot, restart_median=med)
        print(f"      {'tol':>6} | {'adjoint':>9} | {'best DE':>9} | "
              f"{'speed-up':>9} | DE config")
        for t in TOL_DB:
            ca = tr_a.cost_to_reach(targets[t])
            cd, cfg = best_de[t]
            if ca and np.isfinite(cd):
                r = cd / ca
                print(f"      {t:5.2f}dB | {ca:9.0f} | {cd:9.0f} | "
                      f"{r:8.1f}x | {cfg}")
                row[f"tol_{t}"] = dict(adjoint=ca, de=cd, ratio=r,
                                       cfg=list(cfg))
            else:
                print(f"      {t:5.2f}dB | {str(ca):>9} | "
                      f"{'not reached' if not np.isfinite(cd) else cd:>9} | "
                      f"{'--':>9} |")
                row[f"tol_{t}"] = dict(adjoint=ca,
                                       de=None if not np.isfinite(cd) else cd,
                                       ratio=None, cfg=None)
        out.append(row)
        sys.stdout.flush()

    print("\n" + "=" * 92)
    print("  SUMMARY: speed-up vs number of design variables")
    print("=" * 92)
    print(f"  {'N vars':>7} |" + "".join(f"{t:>10.2f}dB" for t in TOL_DB))
    print("  " + "-" * 60)
    for row in out:
        line = f"  {row['nvar']:>7} |"
        for t in TOL_DB:
            r = row[f"tol_{t}"]["ratio"]
            line += f"{('   --' if r is None else f'{r:9.1f}x'):>12}"
        print(line)

    with open("g4_scaling.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n  wrote g4_scaling.json")
    print("=" * 92)


if __name__ == "__main__":
    main()
