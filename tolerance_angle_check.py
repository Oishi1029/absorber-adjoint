import json, numpy as np, sys
sys.path.insert(0, ".")
import tmm_adjoint as ta, g4_synthesis as g4
r = json.load(open("g4_result.json")); x = np.array(r["x"])
stack = g4.x_to_stack(x)
fq = np.linspace(4e9, 18e9, 281)
print("=== angular degradation of the synthesised 5-sheet design (worst |Gamma| over 4-18 GHz) ===")
for a in (0, 30, 45, 60, 65, 70, 75, 80):
    row = []
    for pol in ("TE", "TM"):
        g = ta.reflection(stack, fq, np.deg2rad(a), pol)
        row.append(20*np.log10(np.abs(g)).max())
    print(f"  {a:2d} deg  TE {row[0]:6.2f} dB   TM {row[1]:6.2f} dB")
# tolerance Monte Carlo: +/-10% sheet resistance (uniform), +/-0.1 mm thickness (uniform)
rng = np.random.default_rng(0)
T = np.deg2rad(np.linspace(0, 60, 7))
def worst_db(st):
    w = -np.inf
    for pol in ("TE","TM"):
        g = ta.reflection(st, np.linspace(4e9,18e9,57), T[:,None], pol)
        w = max(w, float((20*np.log10(np.abs(g))).max()))
    return w
base = worst_db(stack)
N = 2000
ws = []
for _ in range(N):
    st = [dict(e) for e in stack]
    for e in st:
        if e["kind"] == "sheet": e["Rs"] = e["Rs"]*(1+rng.uniform(-0.10,0.10))
        else: e["d"] = e["d"] + rng.uniform(-1e-4, 1e-4)
    ws.append(worst_db(st))
ws = np.array(ws)
print(f"\n=== tolerance Monte Carlo, N={N}: Rs +/-10 %, d +/-0.1 mm (uniform) ===")
print(f"  nominal worst-case      : {base:6.2f} dB")
print(f"  median / 95th pct / max : {np.median(ws):6.2f} / {np.percentile(ws,95):6.2f} / {ws.max():6.2f} dB")
print(f"  fraction still <= -10 dB: {100*np.mean(ws <= -10):5.1f} %")
# adjoint-linearised sensitivity of the worst-case point
vars_ = g4.VARS
G, jac = None, None
worst = (-np.inf, None)
for pol in ("TE","TM"):
    Gm, J = ta.reflection_and_jacobian(stack, np.linspace(4e9,18e9,57), T[:,None], pol, vars_)
    db = 20*np.log10(np.abs(Gm)); i = np.unravel_index(db.argmax(), db.shape)
    if db[i] > worst[0]: worst = (db[i], (pol, i, Gm[i], {k: J[k][i] for k in vars_}))
w_db, (pol, i, g0, dj) = worst
print(f"\n=== first-order sensitivity at the worst-case grid point ({pol}, f={np.linspace(4e9,18e9,57)[i[1]]/1e9:.2f} GHz, theta={np.rad2deg(T[i[0]]):.0f} deg) ===")
for k in vars_:
    el = stack[k[0]]
    p = el["Rs"] if k[1]=="Rs" else el["d"]
    dg = dj[k]                     # dGamma/dp (complex)
    d_db_dp = (20/np.log(10)) * np.real(np.conj(g0)*dg)/abs(g0)**2
    tol = 0.10*p if k[1]=="Rs" else 1e-4
    print(f"  {k[1]:2s} elem {k[0]:2d}: d|G|dB/dp * tol = {abs(d_db_dp*tol):5.2f} dB")
