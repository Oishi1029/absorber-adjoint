"""
G2 -- the forward model is independently correct.

Three checks:
  (a) complex Gamma vs scikit-rf (BSD-3, different author, S-parameter /
      Redheffer-star formalism rather than an ABCD product), full band,
      both polarisations, several incidence angles, 5-layer lossy magnetic
      stack plus a resistive-sheet stack;
  (b) a quarter-wave Salisbury screen must reproduce its textbook null;
  (c) an independent in-house impedance-recursion path (different arithmetic
      from the ABCD cascade) as a third opinion.
"""

import numpy as np
import skrf as rf
from skrf.media import DefinedGammaZ0

import tmm_adjoint as ta
from tmm_adjoint import ETA0, C0

np.set_printoptions(precision=3)


# ---------------------------------------------------------------------------
# scikit-rf reference implementation
# ---------------------------------------------------------------------------
def skrf_gamma(stack, f, theta, pol):
    """Reflection coefficient of the PEC-backed stack computed by scikit-rf."""
    freq = rf.Frequency.from_f(f, unit="hz")
    s2 = np.sin(theta) ** 2
    k0 = 2 * np.pi * f / C0
    Zinc = ETA0 / np.cos(theta) if pol == "TE" else ETA0 * np.cos(theta)
    Zinc = np.full_like(f, Zinc, dtype=complex)

    med_inc = DefinedGammaZ0(freq, z0_port=Zinc, z0=Zinc, gamma=1j * k0 * np.cos(theta))

    ntwk = None
    for el in stack:
        if el["kind"] == "line":
            u = np.sqrt(el["eps"] * el["mu"] - s2)
            Z = ETA0 * el["mu"] / u if pol == "TE" else ETA0 * u / el["eps"]
            med = DefinedGammaZ0(freq, z0_port=Zinc,
                                 z0=np.full_like(f, Z, dtype=complex),
                                 gamma=1j * k0 * u)
            piece = med.line(el["d"], unit="m")
        else:
            piece = med_inc.shunt(med_inc.resistor(el["Rs"]) ** med_inc.short())
        ntwk = piece if ntwk is None else ntwk ** piece

    ntwk = ntwk ** med_inc.short()          # PEC backing
    return ntwk.s[:, 0, 0]


# ---------------------------------------------------------------------------
# independent in-house path: impedance recursion (not an ABCD product)
# ---------------------------------------------------------------------------
def recursion_gamma(stack, f, theta, pol):
    k0 = 2 * np.pi * np.asarray(f) / C0
    s2 = np.sin(theta) ** 2
    Z = np.zeros_like(k0, dtype=complex)        # PEC load
    for el in reversed(stack):
        if el["kind"] == "sheet":
            Y = 1.0 / el["Rs"] + np.where(np.abs(Z) > 0, 1.0 / np.where(Z == 0, 1, Z), np.inf)
            Z = np.where(np.abs(Z) == 0, 0.0, 1.0 / Y)
        else:
            u = np.sqrt(el["eps"] * el["mu"] - s2)
            Zc = ETA0 * el["mu"] / u if pol == "TE" else ETA0 * u / el["eps"]
            t = np.tanh(1j * k0 * u * el["d"])
            Z = Zc * (Z + Zc * t) / (Zc + Z * t)
    Zinc = ETA0 / np.cos(theta) if pol == "TE" else ETA0 * np.cos(theta)
    return (Z - Zinc) / (Z + Zinc)


# ---------------------------------------------------------------------------
def relerr(a, b):
    den = np.maximum(np.abs(a), np.abs(b))
    den = np.where(den == 0, 1.0, den)
    return np.abs(a - b) / den


def main():
    f = np.linspace(2e9, 18e9, 401)
    angles_deg = [0.0, 20.0, 35.0, 50.0, 60.0, 70.0]

    # --- 5-layer lossy magnetic stack ------------------------------------
    stack5 = [
        ta.make_line(1.2e-3, 6.0 - 1.1j, 2.4 - 1.7j),
        ta.make_line(0.9e-3, 11.0 - 0.4j, 1.0 - 0.0j),
        ta.make_line(2.1e-3, 3.2 - 0.9j, 3.1 - 2.2j),
        ta.make_line(1.5e-3, 15.0 - 3.3j, 1.0 - 0.0j),
        ta.make_line(0.7e-3, 2.1 - 0.05j, 1.4 - 0.6j),
    ]

    # --- Jaumann-style stack with resistive sheets ------------------------
    stack_sheet = [
        ta.make_sheet(310.0),
        ta.make_line(2.5e-3, 1.06 - 0.002j, 1.0),
        ta.make_sheet(140.0),
        ta.make_line(3.0e-3, 1.06 - 0.002j, 1.0),
        ta.make_sheet(90.0),
        ta.make_line(2.2e-3, 2.2 - 0.02j, 1.0),
    ]

    print("=" * 78)
    print("G2(a)  ANALYTIC ABCD CASCADE  vs  scikit-rf  (independent implementation)")
    print("=" * 78)
    print(f"scikit-rf version {rf.__version__};  {len(f)} frequencies 2-18 GHz")
    worst_overall = 0.0
    for name, stack in (("5-layer lossy magnetic", stack5),
                        ("resistive-sheet Jaumann", stack_sheet)):
        print(f"\n  stack: {name}  ({len(stack)} elements)")
        for pol in ("TE", "TM"):
            row = []
            for a in angles_deg:
                th = np.deg2rad(a)
                g_mine = ta.reflection(stack, f, th, pol)
                g_skrf = skrf_gamma(stack, f, th, pol)
                e = relerr(g_mine, g_skrf).max()
                worst_overall = max(worst_overall, e)
                row.append(f"{a:>4.0f}deg {e:8.2e}")
            print(f"    {pol}:  " + "   ".join(row))
    print(f"\n  >>> WORST relative disagreement with scikit-rf: {worst_overall:.3e}")
    print(f"  >>> G2(a) target <= 1e-10 : "
          f"{'PASS' if worst_overall <= 1e-10 else 'FAIL'}")

    # --- (b) Salisbury screen -------------------------------------------
    print("\n" + "=" * 78)
    print("G2(b)  QUARTER-WAVE SALISBURY SCREEN -- textbook null")
    print("=" * 78)
    f0 = 10e9
    d_q = C0 / (4 * f0)                      # quarter wave in air at 10 GHz
    salisbury = [ta.make_sheet(ETA0), ta.make_line(d_q, 1.0, 1.0)]
    fg = np.linspace(2e9, 18e9, 1601)
    g = ta.reflection(salisbury, fg, 0.0, "TE")
    g0 = ta.reflection(salisbury, np.array([f0]), 0.0, "TE")[0]
    g0_skrf = skrf_gamma(salisbury, np.array([f0]), 0.0, "TE")[0]
    imin = np.argmin(np.abs(g))
    print(f"  sheet Rs = eta0 = {ETA0:.6f} ohm/sq, air spacer d = c/(4 f0) = "
          f"{d_q*1e3:.4f} mm, f0 = {f0/1e9:.1f} GHz")
    print(f"  |Gamma(f0)|                       = {abs(g0):.6e}   "
          f"(exact analytic value 0)")
    print(f"  |Gamma(f0)| from scikit-rf        = {abs(g0_skrf):.6e}")
    print(f"  numerical null located at         = {fg[imin]/1e9:.6f} GHz "
          f"(design 10.000000 GHz)")

    # closed form over the whole band and at oblique incidence:
    #   Zin = eta0 || (j Zp tan(k0 cos(th) d)),  Gamma = (Zin-Zp)/(Zin+Zp)
    #     =>  Gamma = ( j t (eta0 - Zp) - eta0 ) / ( j t (eta0 + Zp) + eta0 )
    #   at normal incidence Zp = eta0  =>  Gamma = -1 / (1 + 2 j t)
    def salisbury_closed_form(fq, th, pol):
        t = np.tan(2 * np.pi * fq / C0 * np.cos(th) * d_q)
        Zp = ETA0 / np.cos(th) if pol == "TE" else ETA0 * np.cos(th)
        return (1j * t * (ETA0 - Zp) - ETA0) / (1j * t * (ETA0 + Zp) + ETA0)

    worst_cf = 0.0
    for pol in ("TE", "TM"):
        for a in (0.0, 30.0, 45.0, 60.0):
            th = np.deg2rad(a)
            e = relerr(ta.reflection(salisbury, fg, th, pol),
                       salisbury_closed_form(fg, th, pol)).max()
            worst_cf = max(worst_cf, e)
    print(f"  vs closed form 1601 freqs x 4 angles x 2 pol: {worst_cf:.3e}")

    # -10 dB bandwidth: closed form gives |Gamma| = 1/sqrt(1+4 tan^2(pi f/2 f0))
    # so |Gamma| <= -10 dB for f/f0 in [1-2/pi*atan(1.5), 1+2/pi*atan(1.5)]
    m = 20 * np.log10(np.abs(g))
    below = fg[m <= -10]
    bw_exact = 2.0 - 4.0 / np.pi * np.arctan(1.5)
    print(f"  -10 dB band                       = {below.min()/1e9:.3f} - "
          f"{below.max()/1e9:.3f} GHz  (fractional "
          f"{100*(below.max()-below.min())/f0:.2f} %, closed form "
          f"{100*bw_exact:.2f} %)")
    sal_ok = abs(g0) < 1e-14 and worst_cf < 1e-10
    print(f"  >>> G2(b) : {'PASS' if sal_ok else 'FAIL'}")

    # --- (c) independent recursion --------------------------------------
    print("\n" + "=" * 78)
    print("G2(c)  ABCD CASCADE  vs  IN-HOUSE IMPEDANCE RECURSION (different arithmetic)")
    print("=" * 78)
    worst_rec = 0.0
    for pol in ("TE", "TM"):
        for a in angles_deg:
            th = np.deg2rad(a)
            e = relerr(ta.reflection(stack5, f, th, pol),
                       recursion_gamma(stack5, f, th, pol)).max()
            worst_rec = max(worst_rec, e)
    print(f"  worst relative disagreement: {worst_rec:.3e}")

    print("\n" + "=" * 78)
    ok = (worst_overall <= 1e-10) and sal_ok
    print(f"G2 VERDICT: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
