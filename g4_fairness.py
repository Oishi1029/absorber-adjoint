"""
G4 fairness audit.

The headline speed-up is only worth quoting if the derivative-free baseline was
given every reasonable chance.  This script:

  1. sweeps differential_evolution over its own hyper-parameters (population
     size, initialisation, updating mode, recombination) and keeps the BEST
     configuration, i.e. the one that reaches the adjoint's objective in the
     fewest forward solves;
  2. adds a second, structurally different derivative-free baseline
     (dual_annealing with no_local_search=True, so no gradient leaks in);
  3. reports the ratio at several match tolerances, because demanding that a
     stochastic global search reproduce a local optimum to eight decimal places
     is not a fair test.  Tolerances are stated in dB on the objective.
  4. reports the single-restart adjoint cost as well as the 10-restart cost.
"""

import json
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution, dual_annealing

import g4_synthesis as g4
import tmm_adjoint as ta

TOL_DB = [2.0, 1.0, 0.5, 0.25, 0.1, 0.0]


class Rec:
    """Counts forward solves and records when each dB tolerance is first met."""

    def __init__(self, F_star):
        self.fwd = 0.0
        self.calls = 0
        self.best = np.inf
        self.F_star = F_star
        self.hit = {t: None for t in TOL_DB}
        self.targets = {t: F_star * 10 ** (t / 20.0) for t in TOL_DB}

    def note(self, F, cost=1.0):
        self.fwd += cost
        self.calls += 1
        if F < self.best:
            self.best = F
        for t in TOL_DB:
            if self.hit[t] is None and F <= self.targets[t]:
                self.hit[t] = self.fwd


BUDGET = 250000.0          # hard cap on forward solves per baseline run


def de_run(F_star, seed, popsize, init, updating, recomb, maxiter=6000):
    rec = Rec(F_star)

    def f(x):
        v = g4.objective(x)
        rec.note(v)
        return v

    def cb(*a, **k):
        return (rec.hit[0.0] is not None) or (rec.fwd >= BUDGET)

    differential_evolution(
        f, bounds=[(0.0, 1.0)] * (2 * g4.N_SHEET), seed=seed, polish=False,
        maxiter=maxiter, popsize=popsize, tol=0.0, mutation=(0.5, 1.0),
        recombination=recomb, init=init, callback=cb, updating=updating)
    return rec


def da_run(F_star, seed, maxfun=120000):
    rec = Rec(F_star)

    def f(x):
        v = g4.objective(x)
        rec.note(v)
        return v

    dual_annealing(f, bounds=[(0.0, 1.0)] * (2 * g4.N_SHEET), seed=seed,
                   no_local_search=True, maxfun=maxfun)
    return rec


def adjoint_cost(n_starts, seed=1):
    rng = np.random.default_rng(seed)
    fwd = 0.0
    calls = 0
    best_F, best_x = np.inf, None
    firsts = []
    for _ in range(n_starts):
        c = g4.Counter()
        x0 = rng.random(2 * g4.N_SHEET)
        res = minimize(lambda z: g4.objective_and_grad(z, c), x0, jac=True,
                       method="L-BFGS-B",
                       bounds=[(0.0, 1.0)] * (2 * g4.N_SHEET),
                       options=dict(maxiter=600, ftol=1e-14, gtol=1e-12))
        fwd += c.fwd
        calls += c.calls
        firsts.append((res.fun, c.fwd))
        if res.fun < best_F:
            best_F, best_x = res.fun, res.x
    return fwd, calls, best_F, best_x, firsts


def main():
    print("=" * 88)
    print("G4 FAIRNESS AUDIT")
    print("=" * 88)

    fwd10, calls10, F_star, x_star, firsts = adjoint_cost(10)
    fwd1 = firsts[0][1]
    F1 = firsts[0][0]
    worst_db, frac = g4.max_gamma_db(x_star)
    print(f"  adjoint F* (10 restarts)  : {F_star:.8f}  "
          f"({20*np.log10(F_star):.3f} dB), cost {fwd10:.0f} forward solves")
    print(f"  adjoint, FIRST restart    : {F1:.8f} "
          f"({20*np.log10(F1):.3f} dB), cost {fwd1:.0f} forward solves")
    n_at_opt = sum(1 for f_, _ in firsts if f_ <= F_star * 1.001)
    print(f"  restarts reaching F*      : {n_at_opt} of 10")
    print(f"  synthesised design        : worst-case |Gamma| = {worst_db:.3f} dB, "
          f"{100*frac:.1f} % of grid at <= -10 dB")

    # ---------------- DE hyper-parameter sweep ------------------------
    print("\n  differential_evolution hyper-parameter sweep "
          "(cost in forward solves to reach each tolerance)")
    print("  " + "-" * 84)
    hdr = f"  {'popsize':>7} {'init':>14} {'update':>9} {'recomb':>6} |" + \
          "".join(f"{t:>9.2f}dB" for t in TOL_DB)
    print(hdr)
    print("  " + "-" * 84)

    configs = [
        (8,  "latinhypercube", "immediate", 0.7),
        (15, "latinhypercube", "immediate", 0.7),
        (25, "latinhypercube", "immediate", 0.7),
        (15, "latinhypercube", "deferred",  0.7),
        (25, "latinhypercube", "deferred",  0.7),
        (15, "sobol",          "immediate", 0.7),
        (15, "latinhypercube", "immediate", 0.9),
        (15, "latinhypercube", "immediate", 0.5),
    ]

    best_de = {t: (np.inf, None) for t in TOL_DB}
    results = []
    t0 = time.time()
    for (popsize, init, updating, recomb) in configs:
        rec = de_run(F_star, 1, popsize, init, updating, recomb)
        row = f"  {popsize:>7} {init:>14} {updating:>9} {recomb:>6.1f} |"
        for t in TOL_DB:
            h = rec.hit[t]
            row += f"{('  --' if h is None else f'{h:9.0f}'):>11}"
            if h is not None and h < best_de[t][0]:
                best_de[t] = (h, (popsize, init, updating, recomb))
        print(row, flush=True)
        results.append(dict(popsize=popsize, init=init, updating=updating,
                            recomb=recomb,
                            hit={str(k): v for k, v in rec.hit.items()},
                            best=rec.best, fwd=rec.fwd))
    print(f"  (sweep took {time.time()-t0:.0f} s)")

    # ---------------- dual annealing ---------------------------------
    print("\n  dual_annealing (no local search), 2 seeds", flush=True)
    da_best = {t: np.inf for t in TOL_DB}
    for s in (1, 2):
        rec = da_run(F_star, s)
        row = f"    seed {s} |"
        for t in TOL_DB:
            h = rec.hit[t]
            row += f"{('  --' if h is None else f'{h:9.0f}'):>11}"
            if h is not None:
                da_best[t] = min(da_best[t], h)
        row += f"   best F = {rec.best:.6f}"
        print(row)

    # ---------------- the ratios --------------------------------------
    print("\n" + "=" * 88)
    print("  SPEED-UP, best derivative-free baseline vs adjoint, by match tolerance")
    print("=" * 88)
    print(f"  {'tolerance':>12} | {'best DE':>10} | {'best DA':>10} | "
          f"{'vs 10-restart':>14} | {'vs 1-restart':>13}")
    print("  " + "-" * 76)
    table = {}
    for t in TOL_DB:
        de_c = best_de[t][0]
        da_c = da_best[t]
        base = min(de_c, da_c)
        r10 = base / fwd10 if np.isfinite(base) else np.inf
        r1 = base / fwd1 if np.isfinite(base) else np.inf
        table[t] = dict(de=None if not np.isfinite(de_c) else de_c,
                        da=None if not np.isfinite(da_c) else da_c,
                        ratio_10=r10, ratio_1=r1,
                        de_config=best_de[t][1])
        f_de = "  --" if not np.isfinite(de_c) else f"{de_c:10.0f}"
        f_da = "  --" if not np.isfinite(da_c) else f"{da_c:10.0f}"
        print(f"  {t:>9.2f} dB | {f_de} | {f_da} | {r10:13.1f}x | {r1:12.1f}x")

    hero = table[0.25]["ratio_10"]
    print(f"\n  HEADLINE (matched to within 0.25 dB, 10-restart adjoint charged "
          f"in full): {hero:.0f}x")
    print(f"  G4: {'PASS' if hero >= 3.0 else 'FAIL'}  (criterion >= 3x)")

    with open("g4_fairness.json", "w") as fh:
        json.dump(dict(F_star=F_star, fwd_10=fwd10, fwd_1=fwd1,
                       worst_db=worst_db, frac=frac,
                       n_restarts_at_opt=n_at_opt,
                       table={str(k): v for k, v in table.items()},
                       de_sweep=results,
                       da_best={str(k): (None if not np.isfinite(v) else v)
                                for k, v in da_best.items()}), fh, indent=2)
    print("  wrote g4_fairness.json")
    print("=" * 88)


if __name__ == "__main__":
    main()
