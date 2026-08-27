"""
Analytic adjoint sensitivity analysis for a PEC-backed stratified medium.

Transverse equivalent network (TEN) formulation.  A plane wave is incident from
free space at angle theta on a cascade of homogeneous layers and zero-thickness
resistive sheets, terminated by a perfect electric conductor.  For a fixed
transverse wavenumber kx = k0 sin(theta) each layer reduces to a uniform
transmission-line section, so the whole stack is a cascade of 2x2 ABCD matrices.

Conventions
-----------
exp(+j w t) time dependence.  Passive media are eps = eps' - j eps'',
mu = mu' - j mu'' with eps'' >= 0, mu'' >= 0.

Layer transverse wavenumber:      kz = k0 * u,   u = sqrt(eps*mu - sin^2(theta))
Wave (characteristic) impedance:  Z_TE = eta0 * mu / u
                                  Z_TM = eta0 * u  / eps
Propagation constant:             gamma = j*kz

The ABCD matrix of a line section is even in u, so the model is independent of
the branch chosen for the square root.  Only the incident half-space matters,
and there u = cos(theta) > 0.

Element ABCD matrices (V_in, I_in) = M (V_out, I_out):
    line   M = [[cosh(x),   Z sinh(x)],
                [sinh(x)/Z, cosh(x)  ]],   x = gamma * d
    sheet  M = [[1, 0],
                [Y, 1]],                   Y = 1/Rs   (shunt resistive sheet)

Termination: PEC, Z_L = 0.  Hence with v = M @ [0, 1]^T = [B, D]^T,
    Z_in  = B / D
    Gamma = (Z_in - Z0) / (Z_in + Z0) = (B - Z0 D) / (B + Z0 D)

Adjoint sensitivity
-------------------
With M = M_1 M_2 ... M_K and Gamma = w(v)^T v where v = M e2:

    backward sweep   r_{K+1} = e2,   r_k = M_k r_{k+1}         (K matvecs)
    w from B = r_1[0], D = r_1[1]
    forward  sweep   lam_1^T = w^T,  lam_{k+1}^T = lam_k^T M_k (K vecmats)

    dGamma/dp_k = lam_k^T (dM_k/dp_k) r_{k+1}

All gradients for every design variable are obtained from exactly one backward
and one forward sweep of 2x2 matrix-vector products -- O(K) work -- against
2*Nvar forward solves for central finite differences.

Author: Bong Bin Yong.  Written from scratch; no external EM solver is used in
the forward model or in the adjoint.
"""

import numpy as np
from scipy.constants import c as C0, mu_0 as MU0, epsilon_0 as EPS0

ETA0 = np.sqrt(MU0 / EPS0)

# ----------------------------------------------------------------------------
# Element description
# ----------------------------------------------------------------------------
# A stack is an ordered list of elements, front (incident side) first, and is
# terminated in a PEC.  Each element is a dict:
#   {'kind': 'line',  'd': float, 'eps': complex, 'mu': complex}
#   {'kind': 'sheet', 'Rs': float}
#
# Design variables are named per element index:
#   line  : 'd', "eps'", "eps''", "mu'", "mu''"
#   sheet : 'Rs'
# where eps = eps' - 1j*eps'' and mu = mu' - 1j*mu''.

LINE_VARS = ("d", "eps'", "eps''", "mu'", "mu''")
SHEET_VARS = ("Rs",)


def make_line(d, eps=1.0 + 0j, mu=1.0 + 0j):
    return {"kind": "line", "d": float(d), "eps": complex(eps), "mu": complex(mu)}


def make_sheet(Rs):
    return {"kind": "sheet", "Rs": float(Rs)}


def variable_names(stack):
    """Ordered (element_index, variable_name) list for the whole stack."""
    out = []
    for i, el in enumerate(stack):
        names = LINE_VARS if el["kind"] == "line" else SHEET_VARS
        for nm in names:
            out.append((i, nm))
    return out


def get_param(stack, idx, name):
    el = stack[idx]
    if name == "d":
        return el["d"]
    if name == "Rs":
        return el["Rs"]
    if name == "eps'":
        return el["eps"].real
    if name == "eps''":
        return -el["eps"].imag
    if name == "mu'":
        return el["mu"].real
    if name == "mu''":
        return -el["mu"].imag
    raise KeyError(name)


def set_param(stack, idx, name, value):
    """Return a shallow-copied stack with one real parameter replaced."""
    new = [dict(e) for e in stack]
    el = new[idx]
    if name == "d":
        el["d"] = float(value)
    elif name == "Rs":
        el["Rs"] = float(value)
    elif name == "eps'":
        el["eps"] = complex(value, el["eps"].imag)
    elif name == "eps''":
        el["eps"] = complex(el["eps"].real, -float(value))
    elif name == "mu'":
        el["mu"] = complex(value, el["mu"].imag)
    elif name == "mu''":
        el["mu"] = complex(el["mu"].real, -float(value))
    else:
        raise KeyError(name)
    return new


# ----------------------------------------------------------------------------
# Per-element ABCD and its analytic derivatives
# ----------------------------------------------------------------------------

def _line_quantities(el, k0, s2, pol):
    """u, x, Z and the derivative building blocks for one line element.

    k0, s2 are arrays (broadcast over frequency x angle); s2 = sin^2(theta).
    Returns a dict of arrays.
    """
    eps = el["eps"]
    mu = el["mu"]
    d = el["d"]

    u = np.sqrt(eps * mu - s2)              # principal branch; M is even in u
    x = 1j * k0 * u * d
    if pol == "TE":
        Z = ETA0 * mu / u
    elif pol == "TM":
        Z = ETA0 * u / eps
    else:
        raise ValueError(pol)

    du_deps = mu / (2.0 * u)
    du_dmu = eps / (2.0 * u)

    dx_dd = 1j * k0 * u
    dx_deps = 1j * k0 * d * du_deps
    dx_dmu = 1j * k0 * d * du_dmu

    if pol == "TE":
        dZ_deps = -ETA0 * mu / u**2 * du_deps
        dZ_dmu = ETA0 / u - ETA0 * mu / u**2 * du_dmu
    else:
        dZ_deps = ETA0 * du_deps / eps - ETA0 * u / eps**2
        dZ_dmu = ETA0 * du_dmu / eps

    return dict(u=u, x=x, Z=Z,
                dx_dd=dx_dd, dx_deps=dx_deps, dx_dmu=dx_dmu,
                dZ_deps=dZ_deps, dZ_dmu=dZ_dmu)


def _abcd_line(q):
    C = np.cosh(q["x"])
    S = np.sinh(q["x"])
    Z = q["Z"]
    return C, Z * S, S / Z, C


def _abcd_sheet(el, shape):
    one = np.ones(shape, dtype=complex)
    zero = np.zeros(shape, dtype=complex)
    Y = 1.0 / el["Rs"]
    return one, zero, Y * one, one


def _element_abcd(el, k0, s2, pol, shape):
    if el["kind"] == "line":
        q = _line_quantities(el, k0, s2, pol)
        return _abcd_line(q), q
    else:
        return _abcd_sheet(el, shape), None


def _element_dabcd(el, q, name, shape):
    """Analytic dM/dp for one real design variable p of one element.

    Returns (dA, dB, dC, dD) as arrays.
    """
    if el["kind"] == "sheet":
        if name != "Rs":
            raise KeyError(name)
        zero = np.zeros(shape, dtype=complex)
        dY_dRs = -1.0 / el["Rs"] ** 2
        return zero, zero, dY_dRs * np.ones(shape, dtype=complex), zero

    C = np.cosh(q["x"])
    S = np.sinh(q["x"])
    Z = q["Z"]

    # dM/dx  and  dM/dZ
    dM_dx = (S, Z * C, C / Z, S)
    dM_dZ = (0.0, S, -S / Z**2, 0.0)

    if name == "d":
        dx, dZ = q["dx_dd"], 0.0
    elif name in ("eps'", "eps''"):
        dx, dZ = q["dx_deps"], q["dZ_deps"]
        if name == "eps''":                      # eps = eps' - 1j eps''
            dx, dZ = -1j * dx, -1j * dZ
    elif name in ("mu'", "mu''"):
        dx, dZ = q["dx_dmu"], q["dZ_dmu"]
        if name == "mu''":
            dx, dZ = -1j * dx, -1j * dZ
    else:
        raise KeyError(name)

    out = []
    for a, b in zip(dM_dx, dM_dZ):
        out.append(a * dx + b * dZ)
    return tuple(out)


# ----------------------------------------------------------------------------
# Forward solve
# ----------------------------------------------------------------------------

def _incident_impedance(theta, pol):
    ct = np.cos(theta)
    return ETA0 / ct if pol == "TE" else ETA0 * ct


def reflection(stack, f, theta, pol):
    """Complex reflection coefficient of the PEC-backed stack.

    f     : array of frequencies [Hz]
    theta : array of incidence angles [rad]
    Broadcasting: f and theta are broadcast against each other.
    """
    f = np.asarray(f, dtype=float)
    theta = np.asarray(theta, dtype=float)
    k0 = 2.0 * np.pi * f / C0
    s2 = np.sin(theta) ** 2
    k0, s2 = np.broadcast_arrays(k0, s2)
    shape = k0.shape

    # backward sweep: r = M_k r  starting from e2 = [0, 1]
    r0 = np.zeros(shape, dtype=complex)
    r1 = np.ones(shape, dtype=complex)
    for el in reversed(stack):
        (A, B, Cc, D), _ = _element_abcd(el, k0, s2, pol, shape)
        r0, r1 = A * r0 + B * r1, Cc * r0 + D * r1

    B, D = r0, r1
    Z0 = _incident_impedance(theta, pol)
    return (B - Z0 * D) / (B + Z0 * D)


def input_impedance(stack, f, theta, pol):
    f = np.asarray(f, dtype=float)
    theta = np.asarray(theta, dtype=float)
    k0 = 2.0 * np.pi * f / C0
    s2 = np.sin(theta) ** 2
    k0, s2 = np.broadcast_arrays(k0, s2)
    shape = k0.shape
    r0 = np.zeros(shape, dtype=complex)
    r1 = np.ones(shape, dtype=complex)
    for el in reversed(stack):
        (A, B, Cc, D), _ = _element_abcd(el, k0, s2, pol, shape)
        r0, r1 = A * r0 + B * r1, Cc * r0 + D * r1
    return r0 / r1


# ----------------------------------------------------------------------------
# Analytic adjoint: all dGamma/dp in one backward + one forward sweep
# ----------------------------------------------------------------------------

def reflection_and_jacobian(stack, f, theta, pol, variables=None):
    """Gamma and the analytic Jacobian dGamma/dp for every design variable.

    Returns
    -------
    Gamma : complex array of shape S
    jac   : dict {(element_index, var_name): complex array of shape S}
    """
    f = np.asarray(f, dtype=float)
    theta = np.asarray(theta, dtype=float)
    k0 = 2.0 * np.pi * f / C0
    s2 = np.sin(theta) ** 2
    k0, s2 = np.broadcast_arrays(k0, s2)
    shape = k0.shape
    K = len(stack)

    if variables is None:
        variables = variable_names(stack)

    # --- element matrices and their per-element cached quantities -----------
    mats, quants = [], []
    for el in stack:
        M, q = _element_abcd(el, k0, s2, pol, shape)
        mats.append(M)
        quants.append(q)

    # --- backward sweep: r_k = M_k r_{k+1}, r_{K} == e2 ---------------------
    # store r[k] = state entering element k from the load side, k = 0..K
    r = [None] * (K + 1)
    r[K] = (np.zeros(shape, dtype=complex), np.ones(shape, dtype=complex))
    for k in range(K - 1, -1, -1):
        A, B, Cc, D = mats[k]
        r0, r1 = r[k + 1]
        r[k] = (A * r0 + B * r1, Cc * r0 + D * r1)

    B, D = r[0]
    Z0 = _incident_impedance(theta, pol)
    den = B + Z0 * D
    Gamma = (B - Z0 * D) / den

    # --- adjoint seed: w^T = [dGamma/dB, dGamma/dD] ------------------------
    w0 = 2.0 * Z0 * D / den**2
    w1 = -2.0 * Z0 * B / den**2

    # --- forward sweep: lam_{k+1}^T = lam_k^T M_k, lam_0^T = w^T -----------
    jac = {}
    by_element = {}
    for (i, nm) in variables:
        by_element.setdefault(i, []).append(nm)

    lam0, lam1 = w0, w1
    for k in range(K):
        if k in by_element:
            r0, r1 = r[k + 1]
            for nm in by_element[k]:
                dA, dB, dC, dD = _element_dabcd(stack[k], quants[k], nm, shape)
                # lam^T (dM) r
                t0 = dA * r0 + dB * r1
                t1 = dC * r0 + dD * r1
                jac[(k, nm)] = lam0 * t0 + lam1 * t1
        A, Bm, Cc, Dm = mats[k]
        lam0, lam1 = lam0 * A + lam1 * Cc, lam0 * Bm + lam1 * Dm

    return Gamma, jac


# ----------------------------------------------------------------------------
# Cost counting helper
# ----------------------------------------------------------------------------

class SolveCounter:
    """Counts forward solves.  One 'forward solve' = one evaluation of the
    stack response over the full (frequency, angle, polarisation) grid."""

    def __init__(self):
        self.n = 0

    def add(self, k=1):
        self.n += k
