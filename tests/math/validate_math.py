"""
Validacao numerica das formulas do GenerativeModel e do BezierFitter.

1. Cobertura exata de pixel: recorte Sutherland-Hodgman (referencia)
   vs. acumulacao de area assinada + prefix sum (metodo proposto).
2. Gradiente analitico dA_p/dP_i (derivada de forma, migalhas de pao)
   vs. diferencas finitas centrais da referencia.
3. Gradiente da energia de sintese E = sum_p ||c_p - I_p||^2.
4. Gradiente e Hessiana exata do ajuste Bezier E(p, t) vs. diferencas finitas,
   e convergencia de Newton (Schur + LM) vs. alternancia LS/reparametrizacao.

Python puro (sem numpy).
"""
import math
import random

random.seed(7)

# ---------------------------------------------------------------- geometria
def shoelace(poly):
    a = 0.0
    n = len(poly)
    for k in range(n):
        x0, y0 = poly[k]
        x1, y1 = poly[(k + 1) % n]
        a += x0 * y1 - x1 * y0
    return 0.5 * a


def clip_halfplane(poly, inside, intersect):
    out = []
    n = len(poly)
    for k in range(n):
        cur, nxt = poly[k], poly[(k + 1) % n]
        ci, ni = inside(cur), inside(nxt)
        if ci:
            out.append(cur)
            if not ni:
                out.append(intersect(cur, nxt))
        elif ni:
            out.append(intersect(cur, nxt))
    return out


def clip_to_pixel(poly, i, j):
    def ix(a, b, xc):
        t = (xc - a[0]) / (b[0] - a[0])
        return (xc, a[1] + t * (b[1] - a[1]))

    def iy(a, b, yc):
        t = (yc - a[1]) / (b[1] - a[1])
        return (a[0] + t * (b[0] - a[0]), yc)

    p = poly
    p = clip_halfplane(p, lambda q: q[0] >= i, lambda a, b: ix(a, b, i))
    if not p: return p
    p = clip_halfplane(p, lambda q: q[0] <= i + 1, lambda a, b: ix(a, b, i + 1))
    if not p: return p
    p = clip_halfplane(p, lambda q: q[1] >= j, lambda a, b: iy(a, b, j))
    if not p: return p
    p = clip_halfplane(p, lambda q: q[1] <= j + 1, lambda a, b: iy(a, b, j + 1))
    return p


def coverage_reference(poly, W, H):
    return [[shoelace(clip_to_pixel(poly, i, j)) if len(poly) else 0.0
             for i in range(W)] for j in range(H)]


def segment_pieces(p0, p1):
    """Migalhas de pao: divide o segmento nas linhas inteiras da grade.
    Retorna lista de (i, j, ta, tb)."""
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    ts = [0.0, 1.0]
    if dx != 0.0:
        for xc in range(math.floor(min(x0, x1)) + 1, math.ceil(max(x0, x1))):
            ts.append((xc - x0) / dx)
    if dy != 0.0:
        for yc in range(math.floor(min(y0, y1)) + 1, math.ceil(max(y0, y1))):
            ts.append((yc - y0) / dy)
    ts.sort()
    out = []
    for a, b in zip(ts, ts[1:]):
        if b - a <= 1e-15:
            continue
        tm = 0.5 * (a + b)
        out.append((math.floor(x0 + tm * dx), math.floor(y0 + tm * dy), a, b))
    return out


def coverage_accumulate(poly, W, H):
    """Metodo proposto: acumulacao de area assinada + cobertura, prefix sum por linha."""
    acc = [[0.0] * (W + 2) for _ in range(H)]
    n = len(poly)
    for k in range(n):
        p0, p1 = poly[k], poly[(k + 1) % n]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        for (i, j, ta, tb) in segment_pieces(p0, p1):
            xa, xb = p0[0] + ta * dx, p0[0] + tb * dx
            dyp = (tb - ta) * dy
            xm = 0.5 * (xa + xb)
            cover = -dyp
            area = -dyp * (i + 1 - xm)
            acc[j][i] += area
            acc[j][i + 1] += cover - area
    cov = []
    for j in range(H):
        s, row = 0.0, []
        for i in range(W):
            s += acc[j][i]
            row.append(s)
        cov.append(row)
    return cov


def coverage_gradient(poly, W, H):
    """dA_p/dP_i analitico. Retorna dict (i_node) -> dict (pix) -> [gx, gy]."""
    n = len(poly)
    grad = [dict() for _ in range(n)]
    for k in range(n):
        p0, p1 = poly[k], poly[(k + 1) % n]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        nx, ny = dy, -dx  # normal externa (nao normalizada) * L
        for (i, j, ta, tb) in segment_pieces(p0, p1):
            w_out = (tb - ta) - 0.5 * (tb * tb - ta * ta)  # integral de (1-t)
            w_in = 0.5 * (tb * tb - ta * ta)               # integral de t
            for node, w in ((k, w_out), ((k + 1) % n, w_in)):
                g = grad[node].setdefault((i, j), [0.0, 0.0])
                g[0] += nx * w
                g[1] += ny * w
    return grad


# ---------------------------------------------------------------- teste 1-3
def make_blob(cx, cy, r, m, jitter):
    pts = []
    for k in range(m):
        th = 2 * math.pi * k / m
        rr = r * (1 + jitter * (random.random() - 0.5))
        pts.append((cx + rr * math.cos(th) + 1e-3 * random.random(),
                    cy + rr * math.sin(th) + 1e-3 * random.random()))
    if shoelace(pts) < 0:
        pts.reverse()
    return pts


W, H = 14, 12
poly = make_blob(7.13, 6.07, 4.3, 23, 0.6)
ref = coverage_reference(poly, W, H)
acc = coverage_accumulate(poly, W, H)
err_fwd = max(abs(ref[j][i] - acc[j][i]) for j in range(H) for i in range(W))
tot = sum(sum(r) for r in ref)
print(f"[1] cobertura: max |ref - acumulacao| = {err_fwd:.3e}   "
      f"(area poligono {shoelace(poly):.6f}, soma pixels {tot:.6f})")

grad = coverage_gradient(poly, W, H)
h = 1e-6
worst = 0.0
for node in range(len(poly)):
    for comp in (0, 1):
        pp = list(poly); q = list(pp[node]); q[comp] += h; pp[node] = tuple(q)
        pm = list(poly); q = list(pm[node]); q[comp] -= h; pm[node] = tuple(q)
        cp, cm = coverage_reference(pp, W, H), coverage_reference(pm, W, H)
        for j in range(H):
            for i in range(W):
                fd = (cp[j][i] - cm[j][i]) / (2 * h)
                an = grad[node].get((i, j), [0.0, 0.0])[comp]
                worst = max(worst, abs(fd - an))
print(f"[2] dA_p/dP_i: max |analitico - DF| = {worst:.3e}")

# soma sobre pixels deve reproduzir a derivada do shoelace
n = len(poly)
worst_sl = 0.0
for i in range(n):
    xm, ym = poly[i - 1]
    xp, yp = poly[(i + 1) % n]
    sx = sum(g[0] for g in grad[i].values())
    sy = sum(g[1] for g in grad[i].values())
    worst_sl = max(worst_sl, abs(sx - 0.5 * (yp - ym)), abs(sy - 0.5 * (xm - xp)))
print(f"    soma_p dA_p/dP_i vs. derivada do shoelace: {worst_sl:.3e}")

# energia de sintese, 2 regioes (dentro/fora), RGBA
c_in, c_out = (0.1, 0.3, 0.9, 1.0), (1.0, 1.0, 1.0, 1.0)
target_poly = make_blob(7.0, 6.2, 4.0, 31, 0.3)
tc = coverage_reference(target_poly, W, H)
I = [[tuple(tc[j][i] * a + (1 - tc[j][i]) * b for a, b in zip(c_in, c_out))
      for i in range(W)] for j in range(H)]


def energy(pl):
    cv = coverage_accumulate(pl, W, H)
    e = 0.0
    for j in range(H):
        for i in range(W):
            for ch in range(4):
                c = cv[j][i] * c_in[ch] + (1 - cv[j][i]) * c_out[ch]
                e += (c - I[j][i][ch]) ** 2
    return e


cv = coverage_accumulate(poly, W, H)
dc = [a - b for a, b in zip(c_in, c_out)]
worst_e = 0.0
for node in range(n):
    ga = [0.0, 0.0]
    for (i, j), g in grad[node].items():
        if not (0 <= i < W and 0 <= j < H):
            continue
        s = 0.0
        for ch in range(4):
            c = cv[j][i] * c_in[ch] + (1 - cv[j][i]) * c_out[ch]
            s += 2 * (c - I[j][i][ch]) * dc[ch]
        ga[0] += s * g[0]; ga[1] += s * g[1]
    for comp in (0, 1):
        pp = list(poly); q = list(pp[node]); q[comp] += h; pp[node] = tuple(q)
        pm = list(poly); q = list(pm[node]); q[comp] -= h; pm[node] = tuple(q)
        fd = (energy(pp) - energy(pm)) / (2 * h)
        worst_e = max(worst_e, abs(fd - ga[comp]))
print(f"[3] dE/dP_i (energia de sintese): max |analitico - DF| = {worst_e:.3e}")


# ---------------------------------------------------------------- teste 4
def bern(t):
    u = 1 - t
    return (u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t)


def dbern(t):
    u = 1 - t
    return (-3 * u * u, 3 * u * u - 6 * u * t, 6 * u * t - 3 * t * t, 3 * t * t)


def d2bern(t):
    u = 1 - t
    return (6 * u, -12 * u + 6 * t, 6 * u - 12 * t, 6 * t)


def bez(P, t, basis=bern):
    b = basis(t)
    return (sum(b[k] * P[k][0] for k in range(4)), sum(b[k] * P[k][1] for k in range(4)))


def ctrl(P0, P3, p):
    return [P0, (p[0], p[1]), (p[2], p[3]), P3]


def fit_energy(P0, P3, p, ts, Q):
    P = ctrl(P0, P3, p)
    e = 0.0
    for t, q in zip(ts, Q):
        b = bez(P, t)
        e += 0.5 * ((b[0] - q[0]) ** 2 + (b[1] - q[1]) ** 2)
    return e


def grad_hess(P0, P3, p, ts, Q):
    """Gradiente e Hessiana exata em blocos: g_p(4), g_t(m), H_pp(4x4), H_pt(m x 4), D(m)."""
    P = ctrl(P0, P3, p)
    gp = [0.0] * 4
    Hpp = [[0.0] * 4 for _ in range(4)]
    gt, Hpt, D = [], [], []
    for t, q in zip(ts, Q):
        b, db = bern(t), dbern(t)
        B = bez(P, t); B1 = bez(P, t, dbern); B2 = bez(P, t, d2bern)
        r = (B[0] - q[0], B[1] - q[1])
        Jp = [(b[1], 0.0), (0.0, b[1]), (b[2], 0.0), (0.0, b[2])]  # colunas dr/dp_a
        for a in range(4):
            gp[a] += Jp[a][0] * r[0] + Jp[a][1] * r[1]
            for c in range(4):
                Hpp[a][c] += Jp[a][0] * Jp[c][0] + Jp[a][1] * Jp[c][1]
        gt.append(r[0] * B1[0] + r[1] * B1[1])
        d2pt = [(db[1], 0.0), (0.0, db[1]), (db[2], 0.0), (0.0, db[2])]  # d2r/dp_a dt
        Hpt.append([Jp[a][0] * B1[0] + Jp[a][1] * B1[1] + r[0] * d2pt[a][0] + r[1] * d2pt[a][1]
                    for a in range(4)])
        D.append(B1[0] ** 2 + B1[1] ** 2 + r[0] * B2[0] + r[1] * B2[1])
    return gp, gt, Hpp, Hpt, D


def solve4(A, b):
    # Cholesky 4x4 (SPD) - o mesmo que dpotrf_/dpotrs_ fariam.
    n = 4
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 0:
                    return None
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i]
    return x


def least_squares_p(P0, P3, ts, Q):
    a11 = a12 = a22 = 0.0
    bx1 = by1 = bx2 = by2 = 0.0
    for t, q in zip(ts, Q):
        b = bern(t)
        rx = q[0] - b[0] * P0[0] - b[3] * P3[0]
        ry = q[1] - b[0] * P0[1] - b[3] * P3[1]
        a11 += b[1] * b[1]; a12 += b[1] * b[2]; a22 += b[2] * b[2]
        bx1 += b[1] * rx; by1 += b[1] * ry; bx2 += b[2] * rx; by2 += b[2] * ry
    det = a11 * a22 - a12 * a12
    return [(a22 * bx1 - a12 * bx2) / det, (a22 * by1 - a12 * by2) / det,
            (a11 * bx2 - a12 * bx1) / det, (a11 * by2 - a12 * by1) / det]


# pontos de teste: arco ruidoso
m = 25
Q = []
for k in range(m + 1):
    s = k / m
    Q.append((10 * s + 0.05 * random.gauss(0, 1),
              4 * math.sin(math.pi * s) + 1.5 * s * s + 0.05 * random.gauss(0, 1)))
P0, P3 = Q[0], Q[-1]
QI = Q[1:-1]
# parametrizacao por comprimento de corda
L = [0.0]
for a, b in zip(Q, Q[1:]):
    L.append(L[-1] + math.dist(a, b))
ts0 = [L[k] / L[-1] for k in range(1, m)]
p0 = least_squares_p(P0, P3, ts0, QI)

# verificacao gradiente/Hessiana por DF
gp, gt, Hpp, Hpt, D = grad_hess(P0, P3, p0, ts0, QI)
hh = 1e-6
wg = wh = 0.0
for a in range(4):
    pp = list(p0); pp[a] += hh; pm = list(p0); pm[a] -= hh
    fd = (fit_energy(P0, P3, pp, ts0, QI) - fit_energy(P0, P3, pm, ts0, QI)) / (2 * hh)
    wg = max(wg, abs(fd - gp[a]))
    gpp = grad_hess(P0, P3, pp, ts0, QI); gpm = grad_hess(P0, P3, pm, ts0, QI)
    for c in range(4):
        wh = max(wh, abs((gpp[0][c] - gpm[0][c]) / (2 * hh) - Hpp[a][c]))
    for k in range(len(ts0)):
        wh = max(wh, abs((gpp[1][k] - gpm[1][k]) / (2 * hh) - Hpt[k][a]))
for k in range(len(ts0)):
    tp = list(ts0); tp[k] += hh; tm = list(ts0); tm[k] -= hh
    fd = (fit_energy(P0, P3, p0, tp, QI) - fit_energy(P0, P3, p0, tm, QI)) / (2 * hh)
    wg = max(wg, abs(fd - gt[k]))
    g_p = grad_hess(P0, P3, p0, tp, QI); g_m = grad_hess(P0, P3, p0, tm, QI)
    wh = max(wh, abs((g_p[1][k] - g_m[1][k]) / (2 * hh) - D[k]))
print(f"[4] Bezier: max |grad - DF| = {wg:.3e}, max |Hess - DF| = {wh:.3e}")


def newton_schur(p, ts, iters, lam=1e-3):
    p, ts = list(p), list(ts)
    hist = [fit_energy(P0, P3, p, ts, QI)]
    for _ in range(iters):
        gp, gt, Hpp, Hpt, D = grad_hess(P0, P3, p, ts, QI)
        while True:
            Dl = [d + lam for d in D]
            if min(Dl) <= 0:
                lam *= 10; continue
            S = [[Hpp[a][c] + (lam if a == c else 0.0)
                  - sum(Hpt[k][a] * Hpt[k][c] / Dl[k] for k in range(len(ts)))
                  for c in range(4)] for a in range(4)]
            rhs = [-gp[a] + sum(Hpt[k][a] * gt[k] / Dl[k] for k in range(len(ts))) for a in range(4)]
            dp = solve4(S, rhs)
            if dp is None:
                lam *= 10; continue
            dt = [-(gt[k] + sum(Hpt[k][a] * dp[a] for a in range(4))) / Dl[k] for k in range(len(ts))]
            pn = [p[a] + dp[a] for a in range(4)]
            tn = [min(1.0, max(0.0, ts[k] + dt[k])) for k in range(len(ts))]
            en = fit_energy(P0, P3, pn, tn, QI)
            if en < hist[-1]:
                p, ts = pn, tn
                lam = max(lam * 0.1, 1e-12)
                break
            lam *= 10
            if lam > 1e12:
                return p, ts, hist
        hist.append(en)
    return p, ts, hist


def alternating(p, ts, iters):
    """Schneider (Graphics Gems 1990): LS em p, 1 passo de Newton por t_k."""
    p, ts = list(p), list(ts)
    hist = [fit_energy(P0, P3, p, ts, QI)]
    for _ in range(iters):
        P = ctrl(P0, P3, p)
        nt = []
        for t, q in zip(ts, QI):
            B = bez(P, t); B1 = bez(P, t, dbern); B2 = bez(P, t, d2bern)
            r = (B[0] - q[0], B[1] - q[1])
            f1 = r[0] * B1[0] + r[1] * B1[1]
            f2 = B1[0] ** 2 + B1[1] ** 2 + r[0] * B2[0] + r[1] * B2[1]
            nt.append(min(1.0, max(0.0, t - f1 / f2)) if f2 > 0 else t)
        ts = nt
        p = least_squares_p(P0, P3, ts, QI)
        hist.append(fit_energy(P0, P3, p, ts, QI))
    return p, ts, hist


_, _, h_newton = newton_schur(p0, ts0, 8)
_, _, h_alt = alternating(p0, ts0, 8)
print("    energia por iteracao (Newton+Schur vs. alternancia LS/reparam.):")
for k in range(9):
    print(f"      it {k}: {h_newton[k]:.10f}   {h_alt[k]:.10f}")


# ---------------------------------------------------------------- teste 5
def turn_angle(u, v):
    return math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])


def angle_energy(pl):
    n = len(pl)
    e = 0.0
    for k in range(n):
        a, b, c = pl[k - 1], pl[k], pl[(k + 1) % n]
        th = turn_angle((b[0] - a[0], b[1] - a[1]), (c[0] - b[0], c[1] - b[1]))
        e += th * th
    return e


def angle_grad(pl):
    """E_ang = sum_k theta_k^2; d(theta)/du = (u_y, -u_x)/|u|^2, d(theta)/dv = (-v_y, v_x)/|v|^2."""
    n = len(pl)
    g = [[0.0, 0.0] for _ in range(n)]
    for k in range(n):
        a, b, c = pl[k - 1], pl[k], pl[(k + 1) % n]
        u = (b[0] - a[0], b[1] - a[1])
        v = (c[0] - b[0], c[1] - b[1])
        th = turn_angle(u, v)
        uu, vv = u[0] ** 2 + u[1] ** 2, v[0] ** 2 + v[1] ** 2
        du = (u[1] / uu, -u[0] / uu)
        dv = (-v[1] / vv, v[0] / vv)
        for comp in (0, 1):
            g[(k - 1) % n][comp] += 2 * th * (-du[comp])
            g[k][comp] += 2 * th * (du[comp] - dv[comp])
            g[(k + 1) % n][comp] += 2 * th * dv[comp]
    return g


ga = angle_grad(poly)
worst_a = 0.0
for node in range(len(poly)):
    for comp in (0, 1):
        pp = list(poly); q = list(pp[node]); q[comp] += h; pp[node] = tuple(q)
        pm = list(poly); q = list(pm[node]); q[comp] -= h; pm[node] = tuple(q)
        fd = (angle_energy(pp) - angle_energy(pm)) / (2 * h)
        worst_a = max(worst_a, abs(fd - ga[node][comp]))
print(f"[5] prior angular dE_ang/dP_k: max |analitico - DF| = {worst_a:.3e}")

# ---------------------------------------------------------------- criterios (codigo de saida p/ CTest)
criteria = {
    "cobertura por acumulacao": err_fwd < 1e-12,
    "dA_p/dP_i": worst < 1e-6,
    "soma vs. shoelace": worst_sl < 1e-12,
    "dE_gen/dP_i": worst_e < 1e-6,
    "gradiente Bezier": wg < 1e-6,
    "Hessiana Bezier": wh < 1e-5,
    "Newton converge mais que a alternancia": h_newton[-1] < h_alt[-1],
    "prior angular": worst_a < 1e-6,
}
failed = [name for name, ok in criteria.items() if not ok]
if failed:
    print("FALHOU:", ", ".join(failed))
    raise SystemExit(1)
print("OK: todas as formulas conferem")
