"""
Arbitrary-precision reference implementation of the PEC-backed stratified
reflection coefficient, in mpmath.

Purpose: give central finite differences a truncation- AND round-off-free
reference.  At 60 decimal digits with a relative step of 1e-25, the central
difference carries truncation error ~1e-50 and round-off ~1e-35, so it is an
essentially exact derivative.  Comparing the double-precision analytic adjoint
against it isolates the correctness of the derivation from the conditioning of
a double-precision finite difference.
"""

import mpmath as mp
from scipy.constants import c as C0_f, mu_0 as MU0_f, epsilon_0 as EPS0_f


def constants(dps):
    mp.mp.dps = dps
    C0 = mp.mpf(repr(C0_f))
    MU0 = mp.mpf(repr(MU0_f))
    EPS0 = mp.mpf(repr(EPS0_f))
    ETA0 = mp.sqrt(MU0 / EPS0)
    return C0, ETA0


def _to_mpc(z):
    return mp.mpc(mp.mpf(repr(float(z.real))), mp.mpf(repr(float(z.imag))))


def reflection_mp(stack, f, theta, pol, dps=60):
    """Complex Gamma at a single (f, theta) in dps-digit arithmetic."""
    C0, ETA0 = constants(dps)
    k0 = 2 * mp.pi * mp.mpf(repr(float(f))) / C0
    s2 = mp.sin(mp.mpf(repr(float(theta)))) ** 2

    r0 = mp.mpc(0)
    r1 = mp.mpc(1)
    for el in reversed(stack):
        if el["kind"] == "line":
            eps = _to_mpc(el["eps"])
            mu = _to_mpc(el["mu"])
            d = mp.mpf(repr(float(el["d"])))
            u = mp.sqrt(eps * mu - s2)
            x = mp.mpc(0, 1) * k0 * u * d
            Z = ETA0 * mu / u if pol == "TE" else ETA0 * u / eps
            A = mp.cosh(x)
            S = mp.sinh(x)
            B = Z * S
            Cc = S / Z
            D = A
        else:
            A = mp.mpc(1)
            B = mp.mpc(0)
            Cc = mp.mpc(1) / mp.mpf(repr(float(el["Rs"])))
            D = mp.mpc(1)
        r0, r1 = A * r0 + B * r1, Cc * r0 + D * r1

    ct = mp.cos(mp.mpf(repr(float(theta))))
    Z0 = ETA0 / ct if pol == "TE" else ETA0 * ct
    return (r0 - Z0 * r1) / (r0 + Z0 * r1)


def _set_param_mp(stack, idx, name, delta):
    """Copy of the stack with one real parameter shifted by an mpf delta.

    The shifted parameter is stored as an mpmath value; reflection_mp's
    repr()-based conversions are bypassed for these entries.
    """
    new = [dict(e) for e in stack]
    el = new[idx]
    if name == "d":
        el["d"] = MpScalar(mp.mpf(repr(float(el["d"]))) + delta)
    elif name == "Rs":
        el["Rs"] = MpScalar(mp.mpf(repr(float(el["Rs"]))) + delta)
    elif name == "eps'":
        el["eps"] = MpComplex(_to_mpc(el["eps"]) + delta)
    elif name == "eps''":
        el["eps"] = MpComplex(_to_mpc(el["eps"]) - mp.mpc(0, 1) * delta)
    elif name == "mu'":
        el["mu"] = MpComplex(_to_mpc(el["mu"]) + delta)
    elif name == "mu''":
        el["mu"] = MpComplex(_to_mpc(el["mu"]) - mp.mpc(0, 1) * delta)
    else:
        raise KeyError(name)
    return new


class MpScalar(float):
    """A real mpf carried through code that calls float()/repr() on it."""
    def __new__(cls, v):
        o = super().__new__(cls, float(v))
        o.mp = v
        return o


class MpComplex(complex):
    def __new__(cls, v):
        o = super().__new__(cls, complex(v))
        o.mp = v
        return o


def _mpf_of(x):
    return x.mp if hasattr(x, "mp") else mp.mpf(repr(float(x)))


def _mpc_of(z):
    return z.mp if hasattr(z, "mp") else _to_mpc(z)


def reflection_mp2(stack, f, theta, pol, dps=60):
    """Same as reflection_mp but honours MpScalar / MpComplex entries."""
    C0, ETA0 = constants(dps)
    k0 = 2 * mp.pi * mp.mpf(repr(float(f))) / C0
    th = mp.mpf(repr(float(theta)))
    s2 = mp.sin(th) ** 2

    r0 = mp.mpc(0)
    r1 = mp.mpc(1)
    for el in reversed(stack):
        if el["kind"] == "line":
            eps = _mpc_of(el["eps"])
            mu = _mpc_of(el["mu"])
            d = _mpf_of(el["d"])
            u = mp.sqrt(eps * mu - s2)
            x = mp.mpc(0, 1) * k0 * u * d
            Z = ETA0 * mu / u if pol == "TE" else ETA0 * u / eps
            A = mp.cosh(x)
            S = mp.sinh(x)
            B, Cc, D = Z * S, S / Z, A
        else:
            A, B, D = mp.mpc(1), mp.mpc(0), mp.mpc(1)
            Cc = mp.mpc(1) / _mpf_of(el["Rs"])
        r0, r1 = A * r0 + B * r1, Cc * r0 + D * r1

    ct = mp.cos(th)
    Z0 = ETA0 / ct if pol == "TE" else ETA0 * ct
    return (r0 - Z0 * r1) / (r0 + Z0 * r1)


def dgamma_dp_mp(stack, idx, name, f, theta, pol, dps=60, h_rel="1e-25"):
    """Essentially exact central-difference dGamma/dp at dps digits."""
    mp.mp.dps = dps
    p0 = abs(float(_param_value(stack, idx, name)))
    h = mp.mpf(h_rel) * (mp.mpf(repr(p0)) if p0 != 0 else mp.mpf(1))
    gp = reflection_mp2(_set_param_mp(stack, idx, name, h), f, theta, pol, dps)
    gm = reflection_mp2(_set_param_mp(stack, idx, name, -h), f, theta, pol, dps)
    return (gp - gm) / (2 * h)


def _param_value(stack, idx, name):
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
