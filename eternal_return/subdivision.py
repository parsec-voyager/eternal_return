"""Smooth charts for surface maps: a mesh's subdivision surface, the limit its faces tend to when split again and
again, as the shape the lattice stands on instead of the faces' flat (bilinear) patches (phase 6 of the unified
engines plan, 2026-11-23).

CatmullClarkChart: an all-quad mesh's Catmull-Clark limit surface. Away from extraordinary vertices (corners where
other than four faces meet) a face's limit is the uniform bicubic B-spline patch of the 4 x 4 control points round it
(the face, its eight neighbors' corners), exactly: a polynomial of degree 3 in each of u and v, whose coefficients are
the mesh's coordinates (ints over 2^E, as surface.BilinearChart reads them) times whole numbers over 36. So every
lattice point I / M of every level is a rational point, exactly, however deep, and two faces sharing an edge give the
same point on it to the last bit (their patches meet there with matching values and slopes: the surface is smooth,
C2, across them). Extraordinary vertices aren't taken yet (step 2b: exact subdivision toward them).

The chart answers what surface.BilinearChart answers (see there): points and Jacobians in floats, in bulk; exact
points (point(), point_q5()); the expansion about a placed point that deep levels' points are offsets along
(grid_offsets(), nearest(), step()), from the patch's Taylor coefficients there worked out exactly, so they're good to
a float's precision however small the offsets are next to the face; normals, corners and bounds. The patch lies in
its 16 control points' hull (B-spline weights are positive and sum to 1), which gives each face's ball (middle,
radius).

The map object's surface settings choose it: "chart": "catmull_clark" (surface.DEFAULTS, "bilinear" by default).
"""

import math
from fractions import Fraction

import numpy as np

from .cyclotomic import q_add, q_mul, q_scale

# The uniform cubic B-spline's basis in powers of u, times 6: B_i(u) = sum_a BASIS[a][i] u^a / 6
BASIS = ((1, 4, 1, 0), (-3, 0, 3, 0), (3, -6, 3, 0), (-1, 3, -3, 1))
# How many points a side of the grid the step bounds are sampled on (surface.SIGMA_GRID's way)
SIGMA_GRID = 9
# Newton steps nearest() takes at most, and how close its steps get before it stops
NEWTON_STEPS = 12
BINOMIAL = [[math.comb(i, a) for a in range(4)] for i in range(4)]


class ExtraordinaryError(ValueError):
    pass


def control_grids(top):
    """Each face's 4 x 4 control points (vertex indices, (F, 4, 4)): grid[f][i][j], i along the face's u (corner 0 to
    1), j along its v (corner 0 to -1), the face's own corners at [1][1], [2][1], [2][2], [1][2]. Only for a closed
    all-quad mesh whose every vertex has four faces (ExtraordinaryError otherwise)."""
    if np.any(top.sides != 4):
        raise ExtraordinaryError("a Catmull-Clark chart needs an all-quad mesh")
    bad = [v for v, faces in enumerate(top.vertex_faces) if len(faces) != 4]
    if bad:
        raise ExtraordinaryError(f"{len(bad)} vertices where other than four faces meet (or on an open edge)")
    loops = top._corners
    faces_of = [{f for f, _s in vf} for vf in top.vertex_faces]

    def beside(loop, x, not_y):
        """The vertex next to x in loop that isn't not_y"""
        k = loop.index(x)
        a, b = loop[(k + 1) % 4], loop[(k - 1) % 4]
        return b if a == not_y else a

    def other_face(f, x, y):
        """The face other than f with the edge x-y"""
        found = (faces_of[x] & faces_of[y]) - {f}
        return found.pop()

    grids = np.zeros((len(loops), 4, 4), dtype=np.int64)
    for f, (c0, c1, c2, c3) in enumerate(loops):
        g = grids[f]
        g[1, 1], g[2, 1], g[2, 2], g[1, 2] = c0, c1, c2, c3
        # Across each side, the neighbor's two other corners, each next to the side's corner it's beside
        for (x, y), (px, py) in (((c0, c1), ((1, 0), (2, 0))), ((c1, c2), ((3, 1), (3, 2))),
                                 ((c2, c3), ((2, 3), (1, 3))), ((c3, c0), ((0, 2), (0, 1)))):
            h = loops[other_face(f, x, y)]
            g[px] = beside(h, x, y)
            g[py] = beside(h, y, x)
        # The diagonal corners: the fourth vertex of the face round each corner opposite f
        for corner, a, b, out in ((c0, (1, 0), (0, 1), (0, 0)), (c1, (2, 0), (3, 1), (3, 0)),
                                  (c2, (3, 2), (2, 3), (3, 3)), (c3, (1, 3), (0, 2), (0, 3))):
            h = (faces_of[corner] & faces_of[g[a]] & faces_of[g[b]]) - {f}
            loop = loops[h.pop()]
            g[out] = next(v for v in loop if v not in (corner, g[a], g[b]))
    return grids


def exact_ints(positions):
    """The positions' float32 coordinates as ints over 2^E, exactly: (E, [(x, y, z) per vertex])"""
    ratios = [v.as_integer_ratio() for v in positions.ravel().tolist()]
    E = max(den.bit_length() - 1 for _, den in ratios)
    ints = [num << (E - (den.bit_length() - 1)) for num, den in ratios]
    return E, [tuple(ints[3 * k:3 * k + 3]) for k in range(len(positions))]


def powers(t, derivative=0):
    """(1, t, t^2, t^3) differentiated derivative times, on t's last axis (..., 4)"""
    t = np.asarray(t, dtype=float)
    one, zero = np.ones_like(t), np.zeros_like(t)
    if derivative == 0:
        return np.stack([one, t, t * t, t * t * t], axis=-1)
    if derivative == 1:
        return np.stack([zero, one, 2.0 * t, 3.0 * t * t], axis=-1)
    return np.stack([zero, zero, 2.0 * one, 6.0 * t], axis=-1)


def shifted(coefficients, u0, v0):
    """A polynomial's coefficients (4, 4, 3: of u^a v^b, exact numbers) about (u0, v0): the coefficients of du^a dv^b,
    exactly (the binomial expansion)"""
    out = [[[0, 0, 0] for _b in range(4)] for _a in range(4)]
    up = [u0 ** k for k in range(4)]
    vp = [v0 ** k for k in range(4)]
    for i in range(4):
        for j in range(4):
            c = coefficients[i][j]
            if not any(c):
                continue
            for a in range(i + 1):
                ka = BINOMIAL[i][a] * up[i - a]
                for b in range(j + 1):
                    k = ka * BINOMIAL[j][b] * vp[j - b]
                    o = out[a][b]
                    o[0] += k * c[0]
                    o[1] += k * c[1]
                    o[2] += k * c[2]
    return out


class CatmullClarkChart:
    """An all-quad mesh's Catmull-Clark limit surface, a bicubic patch a face (see the module's docstring)"""

    degree = 3
    # A lattice cell isn't its corners' bilinear blend: the floor splits cells on the surface itself (surface.refine_smooth())
    flat_cells = False

    def __init__(self, top):
        self.sides = top.sides
        grids = control_grids(top)
        self.E, ints = exact_ints(top.positions)
        F = len(grids)
        # Each face's coefficients of u^a v^b: in floats for all of them at once, and exactly (whole numbers over
        # 36 2^E, exact_of()) only for the faces asked exactly, a few near the drone (all 720 of the torus's in Python's
        # ints took 0.2 s a load: the float32 coordinates as ints over a common 2^E are too big for int64)
        self.grids, self.ints = grids, ints
        self.denominator = 36 << self.E
        self._exact = {}
        self.C = np.einsum("ai,bj,fijk->fabk", np.array(BASIS, dtype=float) / 6.0, np.array(BASIS, dtype=float) / 6.0,
                           top.positions[grids])
        # Each face's coefficients as plain floats too, C[a][b][k], for the questions asked a point at a time (normal():
        # through numpy, the 40k a top level's parts ask took 4 s, 2026-11-23)
        self._C = self.C.tolist()
        # A ball round each face: its control points' (the patch lies in their hull)
        control = top.positions[grids].reshape(F, 16, 3)
        self.middle = control.mean(axis=1)
        self.radius = np.linalg.norm(control - self.middle[:, None], axis=2).max(axis=1)
        # How far a step of 1 in u or v moves the point, at least and at most: the Jacobian's singular values on a grid,
        # give or take what it can change between the grid's points (by its derivatives' bounds times the way there)
        grid = np.linspace(0.0, 1.0, SIGMA_GRID)
        u, v = (g.ravel() for g in np.meshgrid(grid, grid, indexing="ij"))
        # (Every face at every grid point at once: indexing C by face for each point copied it 81 times over)
        ju = np.einsum("ga,gb,fabk->fgk", powers(u, 1), powers(v), self.C)
        jv = np.einsum("ga,gb,fabk->fgk", powers(u), powers(v, 1), self.C)
        a, b, c = np.sum(ju * ju, -1), np.sum(ju * jv, -1), np.sum(jv * jv, -1)
        mean, gap = (a + c) / 2.0, np.sqrt(((a - c) / 2.0) ** 2 + b * b)
        small, big = np.sqrt(np.maximum(mean - gap, 0.0)), np.sqrt(mean + gap)
        uu, uv, vv = (self.bound(f) for f in ((2, 0), (1, 1), (0, 2)))
        slack = (np.maximum(uu, vv) + uv) * math.sqrt(0.5) / (SIGMA_GRID - 1)
        self.sigma_lo = np.maximum(small.min(axis=1) - slack, 0.0)
        self.sigma_hi = big.max(axis=1) + slack
        self._expansions = {}

    def exact_of(self, f):
        """Face f's coefficients of u^a v^b, exactly: whole numbers over 36 2^E, C[a][b][k]"""
        found = self._exact.get(f)
        if found is None:
            P = [[self.ints[self.grids[f, i, j]] for j in range(4)] for i in range(4)]
            found = self._exact[f] = [[[sum(BASIS[a][i] * BASIS[b][j] * P[i][j][k] for i in range(4) for j in range(4))
                                        for k in range(3)] for b in range(4)] for a in range(4)]
        return found

    def bound(self, derivative):
        """Each face's bound on |d^(a+b) S / du^a dv^b| over it (a, b = derivative): its derivative's coefficients'
        lengths summed (u^i v^j is at most 1 on the face)"""
        da, db = derivative
        total = np.zeros(len(self.C))
        for i in range(da, 4):
            for j in range(db, 4):
                k = math.perm(i, da) * math.perm(j, db)
                total += k * np.linalg.norm(self.C[:, i, j], axis=1)
        return total

    # ----- in floats -----

    def _evaluate(self, f, u, v, du=0, dv=0):
        f, u, v = np.broadcast_arrays(np.asarray(f), np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        return np.einsum("...a,...b,...abk->...k", powers(u, du), powers(v, dv), self.C[f])

    def points(self, f, u, v):
        return self._evaluate(f, u, v)

    def jacobians(self, f, u, v):
        return self._evaluate(f, u, v, 1, 0), self._evaluate(f, u, v, 0, 1)

    def normal(self, f, u, v):
        """Face f's unit normal at (u, v), as three floats (in plain floats, a point at a time)"""
        C = self._C[f]
        up, vp = (1.0, u, u * u, u * u * u), (1.0, v, v * v, v * v * v)
        dup, dvp = (0.0, 1.0, 2.0 * u, 3.0 * u * u), (0.0, 1.0, 2.0 * v, 3.0 * v * v)
        ax = ay = az = bx = by = bz = 0.0
        for i in range(4):
            Ci = C[i]
            for j in range(4):
                c = Ci[j]
                ka, kb = dup[i] * vp[j], up[i] * dvp[j]
                ax += ka * c[0]
                ay += ka * c[1]
                az += ka * c[2]
                bx += kb * c[0]
                by += kb * c[1]
                bz += kb * c[2]
        nx, ny, nz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
        length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1e-300
        return nx / length, ny / length, nz / length

    def frame(self, f, I, J, M):
        """Face f's Jacobian's columns at its point (I, J) of a level of M cells across, and its mixed second derivative
        there (see surface.BilinearChart.frame(); the full expansion is grid_offsets())"""
        u, v = Fraction(I, M), Fraction(J, M)
        T = self.expansion(f, u, v)
        return T[1][0], T[0][1], T[1][1]

    def expansion(self, f, u, v):
        """Face f's Taylor coefficients about exact (u, v), as floats (4, 4, 3): S(u + du, v + dv) = sum T[a][b] du^a dv^b,
        worked out exactly first. Kept for the points asked again (a walk asks at the same base point tick after tick)."""
        key = (f, u, v)
        found = self._expansions.get(key)
        if found is None:
            T = shifted(self.exact_of(f), Fraction(u), Fraction(v))
            found = np.array([[[float(Fraction(c) / self.denominator) for c in cell] for cell in row] for row in T])
            if len(self._expansions) > 4096:
                self._expansions.clear()
            self._expansions[key] = found
        return found

    def grid_offsets(self, f, I, J, M, gi, gj):
        """Face f's points (I + gi, J + gj) of a level of M cells across from its point (I, J), in the mesh's units
        times M (see surface.BilinearChart.grid_offsets())"""
        T = self.expansion(f, Fraction(I, M), Fraction(J, M))
        inv_m = 1 / M
        out = np.zeros(np.shape(gi) + (3,))
        for a in range(4):
            for b in range(4):
                if a + b == 0:
                    continue
                scale = inv_m ** (a + b - 1)
                if scale == 0.0:
                    continue
                out += ((gi ** a) * (gj ** b) * scale)[..., None] * T[a][b]
        return out

    def nearest(self, f, I, J, N, r):
        """Face f's point nearest r (units from its point (I, J) of level N), as surface.BilinearChart.nearest() gives
        it: Newton's steps on the patch's expansion there (its Hessian's terms with the residual too, while that keeps
        the step a descent), then, if that's past the face's sides, the best point along the sides it's past."""
        T = self.expansion(f, Fraction(I, N), Fraction(J, N))
        inv_n = 1 / N
        # The expansion in level-N steps: X(d) = sum E[a][b] d0^a d1^b, E = T N^(1 - a - b)
        E = np.zeros((4, 4, 3))
        for a in range(4):
            for b in range(4):
                if a + b:
                    E[a][b] = T[a][b] * inv_n ** (a + b - 1)
        box = (-big(I), big(N - I), -big(J), big(N - J))
        d = newton(E, np.asarray(r, dtype=float), np.zeros(2))
        free = d.copy()
        a_len, b_len = float(np.linalg.norm(E[1][0])), float(np.linalg.norm(E[0][1]))
        lo_u, hi_u, lo_v, hi_v = box
        past = [(lo_v - d[1]) * b_len, (d[0] - hi_u) * a_len, (d[1] - hi_v) * b_len, (lo_u - d[0]) * a_len]
        order = [s for s in sorted(range(4), key=lambda s: -past[s]) if past[s] > 0.0]
        if order:
            best, best_square = None, math.inf
            for s in range(4):
                fixed = (lo_v, hi_u, hi_v, lo_u)[s]
                if not math.isfinite(fixed):
                    continue
                lo, hi = (lo_u, hi_u) if s in (0, 2) else (lo_v, hi_v)
                t = side_newton(E, r, s in (0, 2), fixed, lo, hi, float(np.clip(free[0] if s in (0, 2) else free[1],
                                                                             lo, hi)))
                candidate = np.array([t, fixed]) if s in (0, 2) else np.array([fixed, t])
                square = float(np.sum((evaluate(E, candidate) - r) ** 2))
                if math.isfinite(square) and square < best_square:
                    best, best_square = candidate, square
            if best is not None:
                d = best
        return d, free, order, evaluate(E, d)

    def step(self, f, ur, vr, du, dv, small):
        """Faces f's points at (ur + small du, vr + small dv) from their points at (ur, vr), over small (see
        surface.BilinearChart.step()): from each face's expansion at (ur, vr), so a small step isn't lost to rounding"""
        extra = (1,) * (du.ndim - 1)
        out = np.zeros(du.shape + (3,))
        for t, face in enumerate(np.asarray(f).tolist()):
            T = self.expansion(face, Fraction(float(ur[t])), Fraction(float(vr[t])))
            for a in range(4):
                for b in range(4):
                    if a + b:
                        out[t] += ((du[t] ** a) * (dv[t] ** b) * small ** (a + b - 1))[..., None] * T[a][b]
        return out

    def mixed(self, f):
        raise NotImplementedError("a curved patch's mixed derivative isn't the same all over it (phase 6 step 5)")

    def corners(self):
        """Each face's corner points on the limit surface (F, 4, 3), in its loop's order"""
        F = len(self.C)
        every = np.arange(F)
        return np.stack([self.points(every, u, v) for u, v in ((0, 0), (1, 0), (1, 1), (0, 1))], axis=1)

    def edge_sizes(self):
        """Each face's size: the geometric mean of its Jacobian's columns at its corner 0"""
        a, b = self.jacobians(np.arange(len(self.C)), 0.0, 0.0)
        return np.sqrt(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))

    def twist_norms(self):
        return self.bound((1, 1))

    # ----- exactly -----

    def point(self, f, I, J, M):
        """Face f's point (I, J) of a level of M cells across, exactly: (ints, d), the place being the ints over d"""
        C = self.exact_of(f)
        ip = [I ** a * M ** (3 - a) for a in range(4)]
        jp = [J ** b * M ** (3 - b) for b in range(4)]
        out = [0, 0, 0]
        for a in range(4):
            for b in range(4):
                k = ip[a] * jp[b]
                c = C[a][b]
                out[0] += k * c[0]
                out[1] += k * c[1]
                out[2] += k * c[2]
        return tuple(out), self.denominator * M ** 6

    def point_q5(self, f, u, v):
        """Face f's point at (u, v), Q(sqrt 5) pairs, exactly: three such pairs, the mesh's units"""
        C = self.exact_of(f)
        up, vp = [(Fraction(1), Fraction(0))], [(Fraction(1), Fraction(0))]
        for _ in range(3):
            up.append(q_mul(up[-1], u))
            vp.append(q_mul(vp[-1], v))
        scale = Fraction(1, self.denominator)
        out = []
        for k in range(3):
            total = (Fraction(0), Fraction(0))
            for a in range(4):
                for b in range(4):
                    if C[a][b][k]:
                        total = q_add(total, q_scale(q_mul(up[a], vp[b]), C[a][b][k]))
            out.append(q_scale(total, scale))
        return out


def big(value):
    """A whole number as a float, inf past a float's range"""
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def evaluate(E, d, du=0, dv=0):
    """An expansion E (4, 4, 3) at the offset d (2,), or its derivative"""
    total = np.zeros(3)
    for a in range(du, 4):
        for b in range(dv, 4):
            if a + b == 0:
                continue
            k = math.perm(a, du) * math.perm(b, dv) * d[0] ** (a - du) * d[1] ** (b - dv)
            if k:
                total = total + k * E[a][b]
    return total


def newton(E, r, d):
    """The offset d nearest r on an expansion's whole patch: Newton's steps from d (Gauss-Newton's when the full step
    wouldn't descend)"""
    for _ in range(NEWTON_STEPS):
        e = evaluate(E, d) - r
        u, v = evaluate(E, d, 1, 0), evaluate(E, d, 0, 1)
        uu, uv, vv = evaluate(E, d, 2, 0), evaluate(E, d, 1, 1), evaluate(E, d, 0, 2)
        g = np.array([u @ e, v @ e])
        G = np.array([[u @ u, u @ v], [u @ v, v @ v]])
        H = G + np.array([[uu @ e, uv @ e], [uv @ e, vv @ e]])
        step = None
        if np.linalg.det(H) > 0.0 and H[0, 0] > 0.0:
            step = -np.linalg.solve(H, g)
        if step is None or not np.all(np.isfinite(step)):
            step = -np.linalg.lstsq(G, g, rcond=None)[0]
        d = d + step
        if not (abs(step[0]) > 1e-15 * (1.0 + abs(d[0])) or abs(step[1]) > 1e-15 * (1.0 + abs(d[1]))):
            break
    return d


def side_newton(E, r, along_u, fixed, lo, hi, t):
    """The point nearest r along a side of the face (d1 = fixed with d0 free if along_u, else d0 = fixed): its
    parameter t, Newton's steps from t within [lo, hi]"""
    for _ in range(NEWTON_STEPS):
        d = np.array([t, fixed]) if along_u else np.array([fixed, t])
        e = evaluate(E, d) - r
        first = evaluate(E, d, 1, 0) if along_u else evaluate(E, d, 0, 1)
        second = evaluate(E, d, 2, 0) if along_u else evaluate(E, d, 0, 2)
        g, h = float(first @ e), float(first @ first + second @ e)
        step = -g / h if h > 0.0 else -g / max(float(first @ first), 1e-300)
        new = min(max(t + step, lo), hi)
        if not abs(new - t) > 1e-15 * (1.0 + abs(t)):
            return new
        t = new
    return t
