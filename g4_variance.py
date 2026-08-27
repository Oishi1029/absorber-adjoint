"""
Seed-robustness check for the G4 numbers at N = 10 design variables.

Five independent seeds for each method.  The adjoint's seed varies its random
restart point; DE's seed varies its population initialisation and mutation
stream.  DE uses its best configuration (popsize 8, immediate updating).
Reported: per-seed cost to reach the seed's own matched objective at 0.25 dB,
plus the cross-seed spread of the speed-up ratio.
"""

import json
import numpy as np
from scipy.optimize import minimize, differential_evolution

from g4_scaling import Problem, Trace, BUDGET

TOL = 0.25


def adjoint_once(pb, seed):
    rng = np.random.default_rng(seed)
    tr = Trace()
    x0 = rng.random(pb.nvar)

    def fg(z):
        F, g = pb.F_and_grad(z)
        tr.note(F, 2.0)
        return F, g

    res = minimize(fg, x0, jac=True, method="L-BFGS-B",
                   bounds=[(0.0, 1.0)] * pb.nvar,
                   options=dict(maxiter=800, ftol=1e-14, gtol=1e-12))
    return tr, res.fun


def de_once(pb, seed, target):
    tr = Trace()

    def f(x):
        v = pb.F(x)
        tr.note(v, 1.0)
        return v

    def cb(*a, **k):
        return (tr.best <= target) or (tr.fwd >= BUDGET)

    differential_evolution(f, bounds=[(0.0, 1.0)] * pb.nvar, seed=seed,
                           polish=False, maxiter=100000, popsize=8,
                           tol=0.0, mutation=(0.5, 1.0), recombination=0.7,
                           init="latinhypercube", callback=cb,
                           updating="immediate")
    return tr


def main():
    pb = Problem(5)                      # 10 design variables
    print("N = 10 design variables, tolerance 0.25 dB, five seeds each")
    print(f"{'seed':>5} | {'F* (adjoint)':>12} | {'adjoint fwd':>11} | "
          f"{'DE fwd':>8} | {'ratio':>7}")
    ratios = []
    rows = []
    for seed in (1, 2, 3, 4, 5):
        tr_a, F_star = adjoint_once(pb, seed)
        target = F_star * 10 ** (TOL / 20.0)
        ca = tr_a.cost_to_reach(target)
        tr_d = de_once(pb, seed, target)
        cd = tr_d.cost_to_reach(target)
        if ca and cd:
            ratio = cd / ca
            ratios.append(ratio)
            print(f"{seed:>5} | {F_star:12.6f} | {ca:11.0f} | "
                  f"{cd:8.0f} | {ratio:6.1f}x")
        else:
            print(f"{seed:>5} | {F_star:12.6f} | {str(ca):>11} | "
                  f"{'BUDGET' if cd is None else cd:>8} |  >= "
                  f"{BUDGET/(ca or 1):.0f}x")
            ratios.append(BUDGET / (ca or 1))
        rows.append(dict(seed=seed, F_star=F_star, adjoint=ca, de=cd))
    ratios = np.array(ratios)
    print(f"\nratio across seeds: min {ratios.min():.1f}x, "
          f"median {np.median(ratios):.1f}x, max {ratios.max():.1f}x")
    with open("g4_variance.json", "w") as fh:
        json.dump(dict(rows=rows, ratios=list(ratios)), fh, indent=2)
    print("wrote g4_variance.json")


if __name__ == "__main__":
    main()
