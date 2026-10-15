# absorber-adjoint

Closed-form adjoint sensitivity analysis for multilayer microwave absorber
synthesis at oblique incidence.

This repository contains the complete implementation and every verification and
figure script behind the paper:

> B. Y. Bong, "Analytic Adjoint Sensitivity Analysis for Multilayer Microwave
> Absorber Synthesis at Oblique Incidence," accepted for IEEE APACE 2026,
> Melaka, Malaysia, 8–10 December 2026.

## What it does

A PEC-backed stratified medium (homogeneous layers with complex permittivity
and permeability, plus zero-thickness resistive sheets) is modelled as a
cascade of 2×2 transverse-equivalent-network matrices, at arbitrary incidence
angle for both TE and TM polarisation. The exact derivative of the complex
reflection coefficient with respect to **every** design parameter — each layer
thickness, each ε′, ε″, μ′, μ″, and each sheet resistance — is obtained in
closed form from **one forward and one backward sweep** of 2×2 matrix–vector
products: all *N* gradients for roughly the cost of two forward solves,
independent of *N*, against 2*N* forward solves for central finite differences.

| File | Purpose |
|---|---|
| `tmm_adjoint.py` | Forward model + closed-form adjoint Jacobian (the paper's method) |
| `tmm_mp.py` | 50-digit-arithmetic reference implementation (mpmath) |
| `g1_adjoint_check.py` | Gradient verification: 1376 comparisons vs 50-digit central differences (worst 7.8e-14), double-precision step-size plateau with conditioning-predicted floor |
| `g2_forward_check.py` | Forward verification vs scikit-rf (1.7e-13), closed-form Salisbury screen, independent impedance recursion |
| `g4_synthesis.py` | Wideband wide-angle Jaumann synthesis, adjoint L-BFGS-B vs differential evolution |
| `g4_fairness.py` | Baseline fairness audit: DE hyper-parameter sweep, dual annealing, tolerance-matched accounting |
| `g4_scaling.py` | Speed-up vs number of design variables (identically instrumented protocol) |
| `g4_variance.py` | Seed-robustness of the speed-up at N = 10 |
| `figures.py` | The three paper figures |
| `tolerance_angle_check.py` | Angular degradation beyond the design range and a 2000-sample manufacturing-tolerance Monte Carlo, with the adjoint first-order sensitivities at the worst-case point (camera-ready additions) |
| `g5_page_budget.py` | Page-budget measurement against the conference template |

## Run it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python g2_forward_check.py     # forward model verification
python g1_adjoint_check.py    # gradient verification
python g4_synthesis.py        # synthesis + baseline comparison
python figures.py             # regenerate the paper figures
```

Total compute for every result in the paper is minutes on a laptop CPU.

## Provenance

All physics is classical stratified-media theory implemented from scratch in
this repository. External dependencies are NumPy, SciPy, scikit-rf (BSD-3,
used only as an independent cross-check) and mpmath (BSD). The closed-form
adjoint verification methodology follows the author's earlier work on a
differentiable superconducting control-line model,
[predistort](https://github.com/Oishi1029/predistort).

## License

Apache-2.0. See `LICENSE`.
