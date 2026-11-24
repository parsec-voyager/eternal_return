"""Smooth charts for surface maps: a mesh's subdivision surface, the limit its faces tend to when split again and
again, as the shape the lattice stands on instead of the faces' flat (bilinear) patches (phase 6 of the unified
engines plan, 2026-11-23).

CatmullClarkChart: a closed all-quad mesh's Catmull-Clark limit surface, exactly.
- A regular face (all four corners where four faces meet) is the uniform bicubic B-spline patch of the 4 x 4 control
  points round it: a polynomial of degree 3 in each of u and v whose coefficients are the mesh's coordinates (ints
  over 2^E, as surface.BilinearChart reads them) times whole numbers over 36. Every lattice point I / M of every level
  is a rational point, exactly, however deep, and faces sharing an edge give the same point on it to the last bit
  (the surface is C2 across them).
- A face with an extraordinary vertex (a corner where 3, 5 or more faces meet) is split by Catmull-Clark's own rules,
  exactly in rationals, on a local copy of the faces round it (LocalMesh: its faces within two rings, all a split
  needs): of the four children, those away from the extraordinary corners are regular bicubic patches again, and the
  one at such a corner is split again, toward it, for as long as a point asked for is nearer the corner (the face
  then a tree of leaves: each a parameter box of the face and its bicubic patch, its 16 control points exact). The
  extraordinary vertex's own point is its limit position, (n^2 v + 4 sum of its edge neighbors + sum of its diagonal
  ones) / (n (n + 5)), rational too. Every weight is rational, so this is exact too, at any depth (each level nearer a
  corner costs a split of some dozens of faces in Fractions, kept).
- Near an extraordinary vertex a split shrinks the surface by its subdominant eigenvalue (0.41 at 3 faces, 0.55 at 5,
  0.58 at 6), not by the 1/2 the parameters shrink by, so the lattice's cells there don't keep their size: diving
  exactly at such a corner, its cells drift, ~18% smaller a loop at 3 faces, ~10% bigger at 5. Anywhere else the
  corner's neighborhood shrinks out of the view and the distortion fades. The user chose that over remapping the
  parameters there (irrational exponents, no exact points) or flat corners (2026-11-23). The bounds (sigma_lo,
  sigma_hi) are sampled away from the corners, so they don't hold right at them.

The chart answers what surface.BilinearChart answers (see there): points and Jacobians in floats, in bulk; exact
points (point(), point_q5()); points as offsets from one placed exactly (grid_offsets(), nearest(), step()), from the
patches' Taylor coefficients about it worked out exactly, so they're good to a float's precision however small they
are next to the face, each point from its own leaf's patch, with the exact difference between that patch and the
surface at the base point added (so a leaf's polynomial is never stretched past its box); normals, corners and
bounds. A patch lies in its control points' hull (all of Catmull-Clark's weights are positive), which gives each
face's ball (middle, radius).

The map object's surface settings choose it: "chart": "catmull_clark" (surface.DEFAULTS, "bilinear" by default).
"""

import math
from fractions import Fraction

import numpy as np

from .cyclotomic import q_add, q_mul, q_scale, q_sub

# The uniform cubic B-spline's basis in powers of u, times 6: B_i(u) = sum_a BASIS[a][i] u^a / 6
BASIS = ((1, 4, 1, 0), (-3, 0, 3, 0), (3, -6, 3, 0), (-1, 3, -3, 1))
# How many points a side of the grid the step bounds are sampled on (surface.SIGMA_GRID's way)
SIGMA_GRID = 9
# Newton steps nearest() takes at most, and how close its steps get before it stops
NEWTON_STEPS = 12
BINOMIAL = [[math.comb(i, a) for a in range(4)] for i in range(4)]
# A quad's children, their corner 0's place in its parameters (halves), in the order LocalMesh.subdivide() lists them
CHILDREN = ((0, 0), (1, 0), (1, 1), (0, 1))
# How many levels toward an extraordinary corner floats look (a point nearer it than 2^-FLOAT_DEPTH of the face is
# evaluated from the patch beside it), and exact questions go at most (past it, a point at the corner is its limit)
FLOAT_DEPTH = 40
# Up to how many points of a face are walked down its tree one at a time in plain floats (_float_leaves())
FEW_POINTS = 8
# How many levels toward an extraordinary corner a float point right at it looks for its derivatives' stand-in
# (_corner_leaf())
CORNER_DEPTH = 8
# Up to how many irregular points in all _evaluate() walks down to one at a time
FEW_LEAVES = 64
EXACT_DEPTH = 1 << 16


class ExtraordinaryError(ValueError):
    pass


def grid_of(loops, faces_of, f, regular):
    """Quad f's 4 x 4 control points (vertex ids, a list of 4 rows): grid[i][j], i along its u (corner 0 to 1), j along
    its v (corner 0 to -1), its own corners at [1][1], [2][1], [2][2], [1][2]; None unless every corner is regular
    (four faces, all there: regular(vertex))"""
    c0, c1, c2, c3 = loops[f]
    if not all(regular(c) for c in (c0, c1, c2, c3)):
        return None
    g = [[None] * 4 for _ in range(4)]
    g[1][1], g[2][1], g[2][2], g[1][2] = c0, c1, c2, c3

    def beside(loop, x, not_y):
        k = loop.index(x)
        a, b = loop[(k + 1) % 4], loop[(k - 1) % 4]
        return b if a == not_y else a

    # Across each side, the neighbor's two other corners, each next to the side's corner it's beside
    for (x, y), (px, py) in (((c0, c1), ((1, 0), (2, 0))), ((c1, c2), ((3, 1), (3, 2))),
                             ((c2, c3), ((2, 3), (1, 3))), ((c3, c0), ((0, 2), (0, 1)))):
        h = (faces_of[x] & faces_of[y]) - {f}
        if len(h) != 1:
            return None
        loop = loops[h.pop()]
        g[px[0]][px[1]] = beside(loop, x, y)
        g[py[0]][py[1]] = beside(loop, y, x)
    # The diagonal corners: the fourth vertex of the face round each corner opposite f
    for corner, a, b, out in ((c0, (1, 0), (0, 1), (0, 0)), (c1, (2, 0), (3, 1), (3, 0)),
                              (c2, (3, 2), (2, 3), (3, 3)), (c3, (1, 3), (0, 2), (0, 3))):
        h = (faces_of[corner] & faces_of[g[a[0]][a[1]]] & faces_of[g[b[0]][b[1]]]) - {f}
        if len(h) != 1:
            return None
        loop = loops[h.pop()]
        g[out[0]][out[1]] = next(v for v in loop if v not in (corner, g[a[0]][a[1]], g[b[0]][b[1]]))
    return g


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


def bicubic(P):
    """16 control points (4 x 4 rows of points: Fractions, exactly, or floats) as a bicubic patch's coefficients of
    s^a t^b"""
    return [[[sum(BASIS[a][i] * BASIS[b][j] * P[i][j][k] for i in range(4) for j in range(4)) / 36
              for k in range(3)] for b in range(4)] for a in range(4)]


def bicubic_ints(P, scale):
    """bicubic() of whole numbers over scale, as (D, whole numbers): the coefficients times D (as Fractions, building
    an irregular face's exact leaves took most of quadico's first floor, 2026-11-24)"""
    # One axis at a time: rows[a][j] = sum_i BASIS[a][i] P[i][j], then along j
    rows = [[[sum(BASIS[a][i] * P[i][j][k] for i in range(4)) for k in range(3)] for j in range(4)] for a in range(4)]
    return 36 * scale, [[[sum(BASIS[b][j] * rows[a][j][k] for j in range(4)) for k in range(3)] for b in range(4)]
                        for a in range(4)]


def q_sign(x):
    """The sign of a + b sqrt 5, exactly"""
    a, b = x
    if a >= 0 and b >= 0:
        return 0 if a == 0 and b == 0 else 1
    if a <= 0 and b <= 0:
        return -1
    # Opposite signs: whichever of a^2 and 5 b^2 is bigger wins
    big = a * a - 5 * b * b
    return (1 if a > 0 else -1) if big > 0 else (1 if b > 0 else -1) if big < 0 else 0


class LocalMesh:
    """A few quads of a mesh: their corners' points (id -> three numbers over scale: whole numbers over a whole scale,
    exactly, or floats over 1.0), each vertex's valence in the whole mesh (how many faces it has there), and the quads
    (loops of ids). A vertex has its whole ring here when as many of its faces are. (In Fractions, every sum reduced
    by a gcd: quadico's first floor split its faces' trees for 3 s, 2026-11-24.)"""

    def __init__(self, points, valence, loops, scale):
        self.points, self.valence, self.loops, self.scale = points, valence, loops, scale
        self.exact = isinstance(scale, int)
        self.faces_of = {}
        for t, loop in enumerate(loops):
            for x in loop:
                self.faces_of.setdefault(x, set()).add(t)

    def complete(self, x):
        return len(self.faces_of.get(x, ())) == self.valence[x]

    def regular(self, x):
        return self.valence[x] == 4 and self.complete(x)

    def grid(self, t):
        return grid_of(self.loops, self.faces_of, t, self.regular)

    def around(self, t, rings=2):
        """The faces within rings of t (sharing a corner, again and again) as a mesh of their own, and t's index there"""
        keep = {t}
        for _ in range(rings):
            keep |= {g for f in keep for x in self.loops[f] for g in self.faces_of[x]}
        order = sorted(keep)
        loops = [self.loops[f] for f in order]
        used = {x for loop in loops for x in loop}
        return (LocalMesh({x: self.points[x] for x in used}, {x: self.valence[x] for x in used}, loops, self.scale),
                order.index(t))

    def limit(self, x):
        """Vertex x's point on the limit surface (its ring must be whole here): three Fractions, or floats"""
        n = self.valence[x]
        P = self.points
        edge, diagonal = set(), set()
        for t in self.faces_of[x]:
            loop = self.loops[t]
            k = loop.index(x)
            edge |= {loop[(k + 1) % 4], loop[(k - 1) % 4]}
            diagonal.add(loop[(k + 2) % 4])
        sums = [n * n * P[x][k] + 4 * sum(P[y][k] for y in edge) + sum(P[y][k] for y in diagonal) for k in range(3)]
        under = self.scale * n * (n + 5)
        return tuple(Fraction(c, under) for c in sums) if self.exact else tuple(c / under for c in sums)

    def subdivide(self):
        """One Catmull-Clark split: (the finer mesh, for each face its four children's indices there in CHILDREN's
        order, None for a child some of whose corners can't be worked out here: a vertex without its whole ring, an
        edge without both its faces). Exactly, its points are whole numbers over this scale times every rule's
        denominator (16 and 4 n^2)."""
        P = self.points
        sums = [tuple(P[a][k] + P[b][k] + P[c][k] + P[d][k] for k in range(3)) for a, b, c, d in self.loops]
        whole = [x for x, faces in self.faces_of.items() if len(faces) == self.valence[x]]
        if self.exact:
            L = math.lcm(16, *(4 * self.valence[x] ** 2 for x in whole))
            weight, scale = (lambda under: L // under), self.scale * L
        else:
            weight, scale = (lambda under: 1.0 / under), 1.0
        edge_faces = {}
        for t, loop in enumerate(self.loops):
            for s in range(4):
                a, b = loop[s], loop[(s + 1) % 4]
                edge_faces.setdefault((min(a, b), max(a, b)), []).append(t)
        points, valence, ids = {}, {}, {}

        def add(key, point, n):
            ids[key] = len(points)
            points[ids[key]] = point
            valence[ids[key]] = n

        w = weight(4)
        for t, S in enumerate(sums):
            add(("f", t), (S[0] * w, S[1] * w, S[2] * w), 4)
        w = weight(16)
        for (a, b), faces in edge_faces.items():
            if len(faces) == 2:
                S0, S1 = sums[faces[0]], sums[faces[1]]
                add(("e", a, b), tuple((4 * (P[a][k] + P[b][k]) + S0[k] + S1[k]) * w for k in range(3)), 4)
        for x in whole:
            n = self.valence[x]
            faces = self.faces_of[x]
            ring = set()
            for t in faces:
                loop = self.loops[t]
                k = loop.index(x)
                ring |= {loop[(k + 1) % 4], loop[(k - 1) % 4]}
            # (Q + 2 R + (n - 3) P) / n, Q the faces' points' mean, R the edges' middles': over 4 n^2
            w = weight(4 * n * n)
            add(("v", x), tuple((sum(sums[t][k] for t in faces) + 4 * sum(P[x][k] + P[y][k] for y in ring)
                                 + 4 * n * (n - 3) * P[x][k]) * w for k in range(3)), n)
        loops, children = [], []
        for t, (a, b, c, d) in enumerate(self.loops):
            e = lambda x, y: ids.get(("e", min(x, y), max(x, y)))
            v = lambda x: ids.get(("v", x))
            F = ids[("f", t)]
            row = []
            for quad in ((v(a), e(a, b), F, e(d, a)), (e(a, b), v(b), e(b, c), F), (F, e(b, c), v(c), e(c, d)),
                         (e(d, a), F, e(c, d), v(d))):
                if None in quad:
                    row.append(None)
                else:
                    row.append(len(loops))
                    loops.append(quad)
            children.append(row)
        return LocalMesh(points, valence, loops, scale), children


class Leaf:
    """A regular piece of a face: its parameter box (u0, v0 its corner, size, Fractions), its bicubic patch's
    coefficients of s^a t^b exactly (Q: Fractions, the mesh's units; s = (u - u0) / size, t likewise) and as floats"""

    def __init__(self, u0, v0, size, Q=None, ints=None):
        """Q: the coefficients (floats, or Fractions); or ints: (D, whole numbers), the coefficients times D"""
        self.u0, self.v0, self.size = u0, v0, size
        self._Q, self._ints = Q, ints
        if ints is not None:
            D, C = ints
            # (A whole number over a whole number, rounded once however big they are)
            self.Qf = np.array([[[c / D for c in cell] for cell in row] for row in C])
        else:
            self.Qf = np.array([[[float(c) for c in cell] for cell in row] for row in Q])
        self.fu0, self.fv0, self.fsize = float(u0), float(v0), float(size)
        self.Ql = self.Qf.tolist()
        self._expansions = {}

    @property
    def Q(self):
        """The coefficients exactly (Fractions), worked out when first asked for"""
        if self._Q is None:
            D, C = self._ints
            self._Q = [[[Fraction(c, D) for c in cell] for cell in row] for row in C]
        return self._Q

    def expansion(self, U, V, h, S0, anchor=(0.0, 0.0)):
        """About the face's point (U, V) + h anchor (Fractions, h a Fraction; anchor floats, a point of this leaf's near
        the points asked: a Taylor expansion stretched far past its box cancels in floats), in steps of h: (E (4, 4, 3),
        c (3,)), so the point h (gi, gj) from (U, V) is S0 + h (c + sum E[a][b] (gi - anchor_0)^a (gj - anchor_1)^b):
        this patch's Taylor coefficients at the anchor, and the exact difference of its value there from the surface's
        at (U, V) (S0), both over h, as floats"""
        key = (U, V, h, anchor)
        found = self._expansions.get(key)
        if found is None:
            Ua, Va = U + h * Fraction(anchor[0]), V + h * Fraction(anchor[1])
            s, t = (Ua - self.u0) / self.size, (Va - self.v0) / self.size
            # The derivatives' terms in floats, about a point inside the box (s, t in [0, 1]: nothing cancels, and
            # each is a few steps' worth); only the value there exactly, its difference from S0 being a few steps of
            # h, as small as 2^-1000 of the face (all of it exactly, the 4100 shifts a floor build of quadico asked
            # for took 0.7 s, 2026-11-24)
            T = shifted(self.Ql, float(s), float(t))
            k = np.array([[float(h) ** (a + b - 1) / self.fsize ** (a + b) if a + b else 0.0 for b in range(4)]
                          for a in range(4)])
            E = np.array(T) * k[:, :, None]
            value = self.value(s, t)
            c = np.array([float((value[n] - S0[n]) / h) for n in range(3)])
            found = (E, c)
            if len(self._expansions) > 256:
                self._expansions.clear()
            self._expansions[key] = found
        return found

    def value(self, s, t):
        """The patch's point at exact (s, t) (Fractions), three Fractions: in whole numbers over one denominator (as
        Fractions, a floor build's few thousand took most of its time)"""
        if self._ints is None:
            D = 1
            for row in self.Q:
                for cell in row:
                    for c in cell:
                        D = D * c.denominator // math.gcd(D, c.denominator)
            self._ints = (D, [[[int(c * D) for c in cell] for cell in row] for row in self.Q])
        D, Q = self._ints
        ps, qs, pt, qt = s.numerator, s.denominator, t.numerator, t.denominator
        sp = [ps ** a * qs ** (3 - a) for a in range(4)]
        tp = [pt ** b * qt ** (3 - b) for b in range(4)]
        total = [0, 0, 0]
        for a in range(4):
            for b in range(4):
                c = Q[a][b]
                k = sp[a] * tp[b]
                total[0] += k * c[0]
                total[1] += k * c[1]
                total[2] += k * c[2]
        under = D * qs ** 3 * qt ** 3
        return [Fraction(x, under) for x in total]


class CornerLeaf:
    """A float-only stand-in for a leaf (CatmullClarkChart._corner_leaf()): its box and float coefficients"""

    def __init__(self, fu0, fv0, fsize, Qf):
        self.fu0, self.fv0, self.fsize, self.Qf = fu0, fv0, fsize, Qf
        self.Ql = Qf.tolist()


class CatmullClarkChart:
    """A closed all-quad mesh's Catmull-Clark limit surface (see the module's docstring)"""

    degree = 3
    # A lattice cell isn't its corners' bilinear blend: the floor splits cells on the surface itself (surface.refine_smooth())
    flat_cells = False

    def __init__(self, top):
        self.sides = top.sides
        if np.any(top.sides != 4):
            raise ExtraordinaryError("a Catmull-Clark chart needs an all-quad mesh")
        if any(len(sides) != 2 for sides in top.edge_sides):
            raise ExtraordinaryError("a Catmull-Clark chart needs a closed mesh (every edge between two faces)")
        loops = top._corners
        F = len(loops)
        self._loops = loops
        self._faces_of = [{f for f, _s in vf} for vf in top.vertex_faces]
        self._valence = [len(vf) for vf in top.vertex_faces]
        grids = [grid_of(loops, self._faces_of, f, lambda x: self._valence[x] == 4) for f in range(F)]
        self.regular = np.array([g is not None for g in grids])
        self.grids = np.array([g if g is not None else [[loops[f][0]] * 4] * 4 for f, g in enumerate(grids)])
        self.E, self.ints = exact_ints(top.positions)
        self.denominator = 36 << self.E
        # A regular face's coefficients of u^a v^b: in floats for all of them at once, and exactly (whole numbers over
        # 36 2^E, exact_of()) only for the faces asked exactly, a few near the drone (all 720 of the torus's in
        # Python's ints took 0.2 s a load: the float32 coordinates as ints over a common 2^E are too big for int64)
        self._exact = {}
        self.C = np.einsum("ai,bj,fijk->fabk", np.array(BASIS, dtype=float) / 6.0, np.array(BASIS, dtype=float) / 6.0,
                           top.positions[self.grids])
        self.C[~self.regular] = 0.0
        self._C = self.C.tolist()
        self._expansions = {}
        # An irregular face's tree: (exactly, face, path of children) -> (LocalMesh, the node's face there), and its
        # leaves: one in Fractions for the exact questions, one in floats for the float ones (in Fractions, sampling the
        # bounds of quadico's 60 faces took 4.5 s)
        self._positions = top.positions
        self._nodes, self._leaves = {}, {}
        self._float_leaf_of = {}
        self._corners = {}
        self._splits = {}
        self._corner_leaves = {}
        self._frames = {}
        # A ball round each face: its control points' (a regular face's 16; an irregular one's, its faces' within one
        # ring: its children's control points all lie in their hull)
        self.middle = np.zeros((F, 3))
        self.radius = np.zeros(F)
        for f in range(F):
            if self.regular[f]:
                control = top.positions[self.grids[f]].reshape(16, 3)
            else:
                ring = {x for c in loops[f] for g in self._faces_of[c] for x in loops[g]}
                control = top.positions[sorted(ring)]
            self.middle[f] = control.mean(axis=0)
            self.radius[f] = float(np.linalg.norm(control - self.middle[f], axis=1).max())
        # How far a step of 1 in u or v moves the point, at least and at most: the Jacobian's singular values on a grid,
        # give or take what it can change between the grid's points (a regular face's derivatives' bounds times the way
        # there; an irregular one's sampled a little in from its corners, half again either way)
        grid = np.linspace(0.0, 1.0, SIGMA_GRID)
        u, v = (g.ravel() for g in np.meshgrid(grid, grid, indexing="ij"))
        # (Every regular face at every grid point at once: indexing C by face for each point copied it 81 times over)
        ju = np.einsum("ga,gb,fabk->fgk", powers(u, 1), powers(v), self.C)
        jv = np.einsum("ga,gb,fabk->fgk", powers(u), powers(v, 1), self.C)
        inner = np.clip(u, 1.0 / 64.0, 1.0 - 1.0 / 64.0), np.clip(v, 1.0 / 64.0, 1.0 - 1.0 / 64.0)
        for f in np.flatnonzero(~self.regular).tolist():
            ju[f], jv[f] = self.jacobians(np.full(len(u), f), *inner)
        a, b, c = np.sum(ju * ju, -1), np.sum(ju * jv, -1), np.sum(jv * jv, -1)
        mean, gap = (a + c) / 2.0, np.sqrt(((a - c) / 2.0) ** 2 + b * b)
        small, big = np.sqrt(np.maximum(mean - gap, 0.0)), np.sqrt(mean + gap)
        uu, uv, vv = (self.bound(d) for d in ((2, 0), (1, 1), (0, 2)))
        slack = (np.maximum(uu, vv) + uv) * math.sqrt(0.5) / (SIGMA_GRID - 1)
        self.sigma_lo = np.where(self.regular, np.maximum(small.min(axis=1) - slack, 0.0), 0.5 * small.min(axis=1))
        self.sigma_hi = np.where(self.regular, big.max(axis=1) + slack, 1.5 * big.max(axis=1))
        # How much each face bends at most (its mixed derivative's length): a regular one's bound, an irregular one's
        # sampled away from its corners, half again
        self._twist = uv.copy()
        for f in np.flatnonzero(~self.regular).tolist():
            mixed = self._evaluate(np.full(len(u), f), *inner, du=1, dv=1)
            self._twist[f] = 1.5 * float(np.linalg.norm(mixed, axis=1).max())

    def exact_of(self, f):
        """Regular face f's coefficients of u^a v^b, exactly: whole numbers over 36 2^E, C[a][b][k]"""
        found = self._exact.get(f)
        if found is None:
            P = [[self.ints[self.grids[f, i, j]] for j in range(4)] for i in range(4)]
            found = self._exact[f] = [[[sum(BASIS[a][i] * BASIS[b][j] * P[i][j][k] for i in range(4) for j in range(4))
                                        for k in range(3)] for b in range(4)] for a in range(4)]
        return found

    def bound(self, derivative):
        """Each regular face's bound on |d^(a+b) S / du^a dv^b| over it (a, b = derivative): its derivative's
        coefficients' lengths summed (u^i v^j is at most 1 on the face); 0 for an irregular one"""
        da, db = derivative
        total = np.zeros(len(self.C))
        for i in range(da, 4):
            for j in range(db, 4):
                k = math.perm(i, da) * math.perm(j, db)
                total += k * np.linalg.norm(self.C[:, i, j], axis=1)
        return total

    # ----- an irregular face's tree -----

    def _node(self, f, path, exact=True):
        """The node of irregular face f's tree down path (children's indices): (LocalMesh, its face there), in
        Fractions or floats"""
        found = self._nodes.get((exact, f, path))
        if found is None:
            if not path:
                keep = {f}
                for _ in range(2):
                    keep |= {g for h in keep for x in self._loops[h] for g in self._faces_of[x]}
                order = sorted(keep)
                loops = [tuple(self._loops[g]) for g in order]
                used = {x for loop in loops for x in loop}
                points = ({x: self.ints[x] for x in used} if exact else
                          {x: tuple(self._positions[x].tolist()) for x in used})
                mesh = LocalMesh(points, {x: self._valence[x] for x in used}, loops, 1 << self.E if exact else 1.0)
                found = (mesh, order.index(f))
            else:
                mesh, t = self._node(f, path[:-1], exact)
                # The parent's split, kept for its other children (each split it again: quadico's first floor did
                # every subdivision four times, 2026-11-24)
                split = self._splits.get((exact, f, path[:-1]))
                if split is None:
                    split = self._splits[(exact, f, path[:-1])] = mesh.subdivide()
                finer, children = split
                child = children[t][path[-1]]
                if child is None:
                    raise ExtraordinaryError(f"face {f}: a child down {path} can't be worked out from its rings")
                found = finer.around(child, 2)
            self._nodes[(exact, f, path)] = found
        return found

    def _leaf(self, f, path, exact=True):
        """Face f's leaf down path (a regular node), or None if the node there isn't regular"""
        key = (exact, f, path)
        if key in self._leaves:
            return self._leaves[key]
        mesh, t = self._node(f, path, exact)
        grid = mesh.grid(t)
        leaf = None
        if grid is not None:
            u0 = v0 = Fraction(0)
            size = Fraction(1)
            for c in path:
                size /= 2
                u0 += CHILDREN[c][0] * size
                v0 += CHILDREN[c][1] * size
            P = [[mesh.points[grid[i][j]] for j in range(4)] for i in range(4)]
            leaf = Leaf(u0, v0, size, ints=bicubic_ints(P, mesh.scale)) if exact else Leaf(u0, v0, size, bicubic(P))
        self._leaves[key] = leaf
        return leaf

    def _extraordinary_corner(self, f, path):
        """Which corner (0..3) of the node down path is an extraordinary vertex, if one is (the child there is the one
        that isn't regular), and the vertex's limit point (kept: every exact point of the face walks past it)"""
        key = (f, path)
        found = self._corners.get(key)
        if found is None:
            mesh, t = self._node(f, path)
            found = next(((k, mesh.limit(x)) for k, x in enumerate(mesh.loops[t]) if mesh.valence[x] != 4),
                         (None, None))
            self._corners[key] = found
        return found

    def _exact_at(self, f, u, v, less=None):
        """Face f's point at exact (u, v) (Fractions, or Q(sqrt 5) pairs with less(a, b) their a < b): (the leaf it's
        in or None, its point if it's an extraordinary vertex's)"""
        less = less or (lambda a, b: a < b)
        path = ()
        u0 = v0 = Fraction(0)
        size = Fraction(1)
        while len(path) < EXACT_DEPTH:
            leaf = self._leaf(f, path)
            if leaf is not None:
                return leaf, None
            corner, point = self._extraordinary_corner(f, path)
            if corner is not None:
                cu, cv = ((0, 0), (1, 0), (1, 1), (0, 1))[corner]
                if not less(u, u0 + cu * size) and not less(u0 + cu * size, u) and \
                        not less(v, v0 + cv * size) and not less(v0 + cv * size, v):
                    return None, point
            half = size / 2
            i, j = int(not less(u, u0 + half)), int(not less(v, v0 + half))
            path += (CHILDREN.index((i, j)),)
            u0, v0, size = u0 + i * half, v0 + j * half, half
        raise ExtraordinaryError(f"face {f}: no leaf within {EXACT_DEPTH} levels")

    def _leaf_near(self, f, U, V, h):
        """The leaf face f's point (U, V) is in, or at an extraordinary vertex, the one half a step h in toward the
        face's middle (its patch stands in for the vertex's expansion, which there isn't)"""
        leaf, _point = self._exact_at(f, U, V)
        if leaf is None:
            inward = lambda x: x + h / 2 if x < Fraction(1, 2) else x - h / 2
            leaf, _point = self._exact_at(f, inward(U), inward(V))
        return leaf

    def _exact_point(self, f, u, v):
        """Face f's point at exact (u, v) (Fractions): three Fractions"""
        if self.regular[f]:
            C = self.exact_of(f)
            up, vp = [u ** k for k in range(4)], [v ** k for k in range(4)]
            return tuple(sum(C[a][b][k] * up[a] * vp[b] for a in range(4) for b in range(4)) / self.denominator
                         for k in range(3))
        leaf, point = self._exact_at(f, u, v)
        if leaf is None:
            return point
        return tuple(leaf.value((u - leaf.u0) / leaf.size, (v - leaf.v0) / leaf.size))

    def _float_leaf(self, f, u, v):
        """Irregular face f's leaf for float (u, v), walked down in plain floats, a point at a time (kept: a walk's or
        a search's Newton steps clipped to a face's corner ask for the same one again and again, there 40 levels down)"""
        key = (f, u, v)
        leaf = self._float_leaf_of.get(key)
        if leaf is not None:
            return leaf
        if u in (0.0, 1.0) and v in (0.0, 1.0):
            leaf = self._corner_leaf(f, CHILDREN.index((int(u), int(v))))
            if leaf is not None:
                self._float_leaf_of[key] = leaf
                return leaf
        path, u0, v0, size = (), 0.0, 0.0, 1.0
        while True:
            leaf = self._leaf(f, path, exact=False)
            if leaf is not None:
                break
            half = size / 2.0
            i, j = int(u >= u0 + half), int(v >= v0 + half)
            c = CHILDREN.index((i, j))
            if len(path) >= FLOAT_DEPTH and self._leaf(f, path + (c,), exact=False) is None:
                # Right by the corner: the patch beside it, the child across from it
                leaf = self._leaf(f, path + (CHILDREN.index((1 - i, 1 - j)),), exact=False)
                break
            path, u0, v0, size = path + (c,), u0 + i * half, v0 + j * half, half
        if len(self._float_leaf_of) > 4096:
            self._float_leaf_of.clear()
        self._float_leaf_of[key] = leaf
        return leaf

    def _corner_leaf(self, f, corner):
        """For a float point right at irregular face f's corner (0..3, CHILDREN's order), if it's an extraordinary
        vertex: the leaf across from it CORNER_DEPTH levels toward it, moved to meet the vertex's limit point there
        (its value there exact to a float, its derivatives a stand-in: the surface has none there in u and v); else
        None. A lattice's corner points walked FLOAT_DEPTH levels down each (quadico's first floor: 6250 nodes, 5 s,
        2026-11-24)."""
        found = self._corner_leaves.get((f, corner))
        if found is None:
            mesh, t = self._node(f, (), exact=False)
            x = mesh.loops[t][corner]
            if mesh.valence[x] == 4:
                found = False
            else:
                limit = mesh.limit(x)
                path = ()
                while len(path) < CORNER_DEPTH and self._leaf(f, path, exact=False) is None:
                    path += (corner,)
                leaf = self._leaf(f, path, exact=False)
                if leaf is None:
                    leaf = self._leaf(f, path[:-1] + ((corner + 2) % 4,), exact=False)
                cu, cv = CHILDREN[corner]
                s, t = (cu - leaf.fu0) / leaf.fsize, (cv - leaf.fv0) / leaf.fsize
                at = np.einsum("a,b,abk->k", powers(np.array(s)), powers(np.array(t)), leaf.Qf)
                Qf = leaf.Qf.copy()
                Qf[0, 0] += np.array(limit, dtype=float) - at
                found = CornerLeaf(leaf.fu0, leaf.fv0, leaf.fsize, Qf)
            self._corner_leaves[(f, corner)] = found
        return found or None

    def _point_frame(self, f, u, v):
        """Face f's point at float (u, v) and its Jacobian's columns, in plain floats (a point at a time: through numpy's
        arrays, a walk's Newton steps on an irregular face took 3.4 ms a call)"""
        if self.regular[f]:
            Q, s, t, scale = self._C[f], u, v, 1.0
        else:
            leaf = self._float_leaf(f, u, v)
            Q, s, t, scale = leaf.Ql, (u - leaf.fu0) / leaf.fsize, (v - leaf.fv0) / leaf.fsize, 1.0 / leaf.fsize
        sp, tp = (1.0, s, s * s, s * s * s), (1.0, t, t * t, t * t * t)
        dsp, dtp = (0.0, 1.0, 2.0 * s, 3.0 * s * s), (0.0, 1.0, 2.0 * t, 3.0 * t * t)
        x = [0.0, 0.0, 0.0]
        a = [0.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0]
        for i in range(4):
            Qi = Q[i]
            for j in range(4):
                c = Qi[j]
                k, ka, kb = sp[i] * tp[j], dsp[i] * tp[j], sp[i] * dtp[j]
                for n in range(3):
                    x[n] += k * c[n]
                    a[n] += ka * c[n]
                    b[n] += kb * c[n]
        return x, [scale * c for c in a], [scale * c for c in b]

    def _float_leaves(self, f, u, v):
        """Irregular face f's leaves for float (u, v) (1-D arrays): [(leaf, rows)], a point nearer an extraordinary
        corner than 2^-FLOAT_DEPTH in the leaf beside it"""
        if len(u) <= FEW_POINTS:
            # A point or two (a search's Newton steps, one point a face): walked down in plain floats, as numpy's
            # arrays a level at a time took 0.15 ms a point, 40 levels down at a corner (2026-11-24: a point below on
            # quadico took 50-120 ms, the drone's few a tick held it in the teens of frames a second)
            groups = {}
            for row, (x, y) in enumerate(zip(u.tolist(), v.tolist())):
                leaf = self._float_leaf(f, x, y)
                groups.setdefault(id(leaf), (leaf, []))[1].append(row)
            return [(leaf, np.array(rows)) for leaf, rows in groups.values()]
        out = []
        rest = np.ones(len(u), dtype=bool)
        for c, (cu, cv) in enumerate(CHILDREN):
            # Right at an extraordinary corner: its corner leaf (_corner_leaf()), not FLOAT_DEPTH levels down
            rows = np.flatnonzero((u == cu) & (v == cv))
            if len(rows):
                leaf = self._corner_leaf(f, c)
                if leaf is not None:
                    out.append((leaf, rows))
                    rest[rows] = False
        stack = [((), np.flatnonzero(rest), 0.0, 0.0, 1.0)]
        while stack:
            path, rows, u0, v0, size = stack.pop()
            if not len(rows):
                continue
            leaf = self._leaf(f, path, exact=False)
            if leaf is not None:
                out.append((leaf, rows))
                continue
            half = size / 2.0
            i = (u[rows] >= u0 + half).astype(int)
            j = (v[rows] >= v0 + half).astype(int)
            for c, (ci, cj) in enumerate(CHILDREN):
                sub = rows[(i == ci) & (j == cj)]
                if len(path) >= FLOAT_DEPTH and self._leaf(f, path + (c,), exact=False) is None:
                    # Right by the corner: the patch beside it, the child across from it
                    corner = CHILDREN.index((1 - ci, 1 - cj))
                    out.append((self._leaf(f, path + (corner,), exact=False), sub))
                    continue
                stack.append((path + (c,), sub, u0 + ci * half, v0 + cj * half, half))
        return out

    def _step_leaves(self, f, U, V, h, gi, gj):
        """Irregular face f's leaves for the points (U, V) + h (gi, gj) (exact base and step, small float offsets in
        steps, 1-D): [(leaf or None, rows)], None for the points at an extraordinary vertex (to within h / 8)"""
        out = []
        stack = [((), np.arange(len(gi)), Fraction(0), Fraction(0), Fraction(1))]
        while stack:
            path, rows, u0, v0, size = stack.pop()
            if not len(rows):
                continue
            leaf = self._leaf(f, path)
            if leaf is not None:
                out.append((leaf, rows))
                continue
            half = size / 2
            bu, bv = float((u0 + half - U) / h), float((v0 + half - V) / h)
            i = (gi[rows] >= bu).astype(int)
            j = (gj[rows] >= bv).astype(int)
            for c, (ci, cj) in enumerate(CHILDREN):
                sub = rows[(i == ci) & (j == cj)]
                if half < h / 8 and self._leaf(f, path + (c,)) is None:
                    out.append((None, sub))
                    continue
                stack.append((path + (c,), sub, u0 + ci * half, v0 + cj * half, half))
        return out

    def _offsets(self, f, U, V, h, gi, gj):
        """Irregular face f's points (U, V) + h (gi, gj) from its point at (U, V), over h (see Leaf.expansion())"""
        shape = np.shape(gi)
        gi, gj = np.asarray(gi, dtype=float).ravel(), np.asarray(gj, dtype=float).ravel()
        S0 = self._exact_point(f, U, V)
        out = np.zeros((len(gi), 3))
        for leaf, rows in self._step_leaves(f, U, V, h, gi, gj):
            if leaf is None:
                # Within h / 8 of an extraordinary vertex (at it, on a lattice): exactly, a point at a time
                for row in rows.tolist():
                    p = self._exact_point(f, U + h * Fraction(float(gi[row])), V + h * Fraction(float(gj[row])))
                    out[row] = [float((p[k] - S0[k]) / h) for k in range(3)]
                continue
            # Expanded about a point of the leaf's own among these (the middle one asked, in whole steps)
            middle = rows[len(rows) // 2]
            anchor = (float(np.round(gi[middle])), float(np.round(gj[middle])))
            E, c = leaf.expansion(U, V, h, S0, anchor)
            out[rows] = c + sum_terms(E, gi[rows] - anchor[0], gj[rows] - anchor[1])
        return out.reshape(shape + (3,))

    # ----- in floats -----

    def _evaluate(self, f, u, v, du=0, dv=0, orders=None):
        """Faces f's points at float (u, v), or their derivatives d^(du + dv) / du^du dv^dv; with orders, a list of them,
        one for each (du, dv) there, from one walk down the irregular faces' trees"""
        f, u, v = np.broadcast_arrays(np.asarray(f), np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        wanted = orders or ((du, dv),)
        outs = [np.einsum("...a,...b,...abk->...k", powers(u, a), powers(v, b), self.C[f]) for a, b in wanted]
        irregular = ~self.regular[f]
        if irregular.any():
            fl, ul, vl = f[irregular], u[irregular], v[irregular]
            # Each point's leaf (an index into leaves), then all of them at once: a search's points, one a face, one
            # at a time through numpy cost more than walking down to them
            if len(fl) <= FEW_LEAVES:
                leaves = [self._float_leaf(g, x, y) for g, x, y in zip(fl.tolist(), ul.tolist(), vl.tolist())]
                index = np.arange(len(fl))
            else:
                leaves, index = [], np.zeros(len(fl), dtype=int)
                for g in np.unique(fl).tolist():
                    rows = np.flatnonzero(fl == g)
                    for leaf, sub in self._float_leaves(g, ul[rows], vl[rows]):
                        index[rows[sub]] = len(leaves)
                        leaves.append(leaf)
            Q = np.array([leaf.Qf for leaf in leaves])[index]
            u0, v0, size = (np.array([getattr(leaf, name) for leaf in leaves])[index] for name in ("fu0", "fv0", "fsize"))
            s, t = (ul - u0) / size, (vl - v0) / size
            for (a, b), out in zip(wanted, outs):
                out[irregular] = (size ** -(a + b))[:, None] * np.einsum("na,nb,nabk->nk", powers(s, a), powers(t, b), Q)
        return outs if orders else outs[0]

    def points(self, f, u, v):
        return self._evaluate(f, u, v)

    def jacobians(self, f, u, v):
        return tuple(self._evaluate(f, u, v, orders=((1, 0), (0, 1))))

    def normal(self, f, u, v):
        """Face f's unit normal at (u, v), as three floats (in plain floats, a point at a time)"""
        if not self.regular[f]:
            _x, (ax, ay, az), (bx, by, bz) = self._point_frame(f, float(u), float(v))
            nx, ny, nz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
            length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1e-300
            return nx / length, ny / length, nz / length
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

    def expansion(self, f, u, v):
        """Regular face f's Taylor coefficients about exact (u, v), as floats (4, 4, 3): S(u + du, v + dv) = sum T[a][b]
        du^a dv^b, worked out exactly first. Kept for the points asked again (a walk asks at the same base point tick
        after tick)."""
        key = (f, u, v)
        found = self._expansions.get(key)
        if found is None:
            T = shifted(self.exact_of(f), Fraction(u), Fraction(v))
            found = np.array([[[float(Fraction(c) / self.denominator) for c in cell] for cell in row] for row in T])
            if len(self._expansions) > 4096:
                self._expansions.clear()
            self._expansions[key] = found
        return found

    def frame(self, f, I, J, M):
        """Face f's Jacobian's columns at its point (I, J) of a level of M cells across, and its mixed second derivative
        there (see surface.BilinearChart.frame(); the full expansion is grid_offsets()). Right at an extraordinary
        vertex, from the leaf beside it a step of the level out."""
        u, v = Fraction(I, M), Fraction(J, M)
        if self.regular[f]:
            T = self.expansion(f, u, v)
            return T[1][0], T[0][1], T[1][1]
        # Kept: the 3D parts size their levels from the anchor's frame a few times a tick, an exact walk down the
        # tree each (2026-11-24)
        key = (f, I, J, M)
        found = self._frames.get(key)
        if found is None:
            h = Fraction(1, M)
            E, _c = self._leaf_near(f, u, v, h).expansion(u, v, h, self._exact_point(f, u, v))
            found = (E[1][0], E[0][1], E[1][1] * M)
            if len(self._frames) > 4096:
                self._frames.clear()
            self._frames[key] = found
        return found

    def grid_offsets(self, f, I, J, M, gi, gj):
        """Face f's points (I + gi, J + gj) of a level of M cells across from its point (I, J), in the mesh's units
        times M (see surface.BilinearChart.grid_offsets())"""
        if not self.regular[f]:
            return self._offsets(f, Fraction(I, M), Fraction(J, M), Fraction(1, M), gi, gj)
        T = self.expansion(f, Fraction(I, M), Fraction(J, M))
        inv_m = 1 / M
        steps = np.array([[inv_m ** (a + b - 1) if a + b else 0.0 for b in range(4)] for a in range(4)])
        # (In one einsum: term by term, a smooth torus's floor build spent 0.2 s here, 2026-11-24)
        return np.einsum("...a,...b,abk->...k", powers(np.asarray(gi, dtype=float)), powers(np.asarray(gj, dtype=float)),
                         T * steps[:, :, None])

    def _steps_expansion(self, f, I, J, N, d):
        """Face f's expansion in level-N steps about its point (I, J), from the patch the point d steps further is in:
        (E, c) with the point d' steps out c + sum E[a][b] d'^a (units)"""
        if self.regular[f]:
            T = self.expansion(f, Fraction(I, N), Fraction(J, N))
            inv_n = 1 / N
            E = np.zeros((4, 4, 3))
            for a in range(4):
                for b in range(4):
                    if a + b:
                        E[a][b] = T[a][b] * inv_n ** (a + b - 1)
            return E, np.zeros(3)
        U, V, h = Fraction(I, N), Fraction(J, N), Fraction(1, N)
        leaves = self._step_leaves(f, U, V, h, np.array([d[0]]), np.array([d[1]]))
        leaf = leaves[0][0] or self._leaf_near(f, U, V, h)
        return leaf.expansion(U, V, h, self._exact_point(f, U, V))

    def nearest(self, f, I, J, N, r):
        """Face f's point nearest r (units from its point (I, J) of level N), as surface.BilinearChart.nearest() gives
        it: Newton's steps on the patch's expansion there (its Hessian's terms with the residual too, while that keeps
        the step a descent), then, if that's past the face's sides, the best point along the sides it's past. An
        irregular face: _nearest_irregular()."""
        r = np.asarray(r, dtype=float)
        if not self.regular[f]:
            return self._nearest_irregular(f, I, J, N, r)
        E, c = self._steps_expansion(f, I, J, N, np.zeros(2))
        d = newton(E, r - c, np.zeros(2))
        box = (-big(I), big(N - I), -big(J), big(N - J))
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
                t = side_newton(E, r - c, s in (0, 2), fixed, lo, hi,
                                float(np.clip(free[0] if s in (0, 2) else free[1], lo, hi)))
                candidate = np.array([t, fixed]) if s in (0, 2) else np.array([fixed, t])
                square = float(np.sum((c + evaluate(E, candidate) - r) ** 2))
                if math.isfinite(square) and square < best_square:
                    best, best_square = candidate, square
            if best is not None:
                d = best
        return d, free, order, c + evaluate(E, d)

    def _nearest_irregular(self, f, I, J, N, r):
        """nearest() on an irregular face, whose patch is piecewise (a leaf's polynomial only holds in its box): from
        the surface itself in the face's parameters (Newton's steps kept on the face, from the base and from its
        middle), where its levels' steps are big enough for floats to tell them apart (N < 2^40), else from the base's
        leaf on (deep down, the leaves round a point are vast next to the drone's sight), then the exact expansion of the
        leaf the point is in, in steps. Past a side: the sides the point found sits on with the surface still nearer
        across them (its gradient pointing out)."""
        U, V, h = Fraction(I, N), Fraction(J, N), Fraction(1, N)
        box = (-big(I), big(N - I), -big(J), big(N - J))
        lo_u, hi_u, lo_v, hi_v = box
        if N < (1 << 40):
            base = np.array(self._point_frame(f, float(U), float(V))[0])
            target = (base + r / N).tolist()
            best = None
            for start in ((float(U), float(V)), (0.5, 0.5)):
                u, v = min(max(start[0], 0.0), 1.0), min(max(start[1], 0.0), 1.0)
                for _ in range(24):
                    X, ju, jv = self._point_frame(f, u, v)
                    e = [X[n] - target[n] for n in range(3)]
                    g11, g12, g22 = (sum(p * q for p, q in zip(x_, y_)) for x_, y_ in ((ju, ju), (ju, jv), (jv, jv)))
                    c1, c2 = sum(p * q for p, q in zip(ju, e)), sum(p * q for p, q in zip(jv, e))
                    det = g11 * g22 - g12 * g12
                    if not det > 1e-300:
                        break
                    du, dv = -(g22 * c1 - g12 * c2) / det, -(g11 * c2 - g12 * c1) / det
                    nu, nv = min(max(u + du, 0.0), 1.0), min(max(v + dv, 0.0), 1.0)
                    done = not max(abs(nu - u), abs(nv - v)) > 1e-15
                    u, v = nu, nv
                    if done:
                        break
                square = sum((p - q) ** 2 for p, q in zip(self._point_frame(f, u, v)[0], target))
                if best is None or square < best[0]:
                    best = (square, np.array([u, v]))
                # From the base it came out on the surface's own point, with the target over it: no need of another
                if square <= 1e-24 * (1.0 + float(np.sum(np.asarray(target) ** 2))) or \
                        (0.0 < u < 1.0 and 0.0 < v < 1.0 and start == (float(U), float(V))):
                    break
            uv = best[1]
            d = np.array([float((Fraction(float(uv[0])) - U) * N), float((Fraction(float(uv[1])) - V) * N)])
        else:
            d = np.zeros(2)
        # Deep down, polished on the leaf it's in (again if a step crosses into the next leaf); shallower, the surface's
        # own point stands (a leaf's polynomial, stretched past its box by a step, pulled points near an
        # extraordinary corner off, 2026-11-23)
        free = d.copy()
        for _ in range(3 if N >= (1 << 40) else 0):
            E, c = self._steps_expansion(f, I, J, N, d)
            free = newton(E, r - c, d.copy())
            moved = np.clip(free, [lo_u, lo_v], [hi_u, hi_v])
            if np.allclose(moved, d, rtol=0.0, atol=1e-12 * (1.0 + float(np.abs(d).max()))):
                d = moved
                break
            d = moved
        E, c = self._steps_expansion(f, I, J, N, d)
        if N < (1 << 40):
            # From the surface's own point (the leaf's expansion about the base, stretched to a far leaf, cancels)
            P, ju, jv = self._point_frame(f, float(uv[0]), float(uv[1]))
            X = (np.array(P) - base) * N
            gu, gv = float(np.array(ju) @ (X - r)), float(np.array(jv) @ (X - r))
        else:
            X = c + evaluate(E, d)
            gu, gv = float(evaluate(E, d, 1, 0) @ (X - r)), float(evaluate(E, d, 0, 1) @ (X - r))
        past = {}
        if d[1] <= lo_v and gv > 0.0:
            past[0] = gv
        if d[0] >= hi_u and gu < 0.0:
            past[1] = -gu
        if d[1] >= hi_v and gv < 0.0:
            past[2] = -gv
        if d[0] <= lo_u and gu > 0.0:
            past[3] = gu
        order = sorted(past, key=lambda s: -past[s])
        return d, free, order, X

    def step(self, f, ur, vr, du, dv, small):
        """Faces f's points at (ur + small du, vr + small dv) from their points at (ur, vr), over small (see
        surface.BilinearChart.step()): from each face's expansion at (ur, vr), so a small step isn't lost to rounding"""
        out = np.zeros(du.shape + (3,))
        h = Fraction(float(small))
        for t, face in enumerate(np.asarray(f).tolist()):
            U, V = Fraction(float(ur[t])), Fraction(float(vr[t]))
            if not self.regular[face]:
                out[t] = self._offsets(face, U, V, h, du[t], dv[t])
                continue
            T = self.expansion(face, U, V)
            for a in range(4):
                for b in range(4):
                    if a + b:
                        out[t] += ((du[t] ** a) * (dv[t] ** b) * small ** (a + b - 1))[..., None] * T[a][b]
        return out

    def mixed(self, f):
        raise NotImplementedError("a curved patch's mixed derivative isn't the same all over it (phase 6 step 5)")

    def corners(self):
        """Each face's corner points on the limit surface (F, 4, 3), in its loop's order (exactly, then rounded)"""
        F = len(self.C)
        out = np.zeros((F, 4, 3))
        for f in range(F):
            for k, (u, v) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
                out[f, k] = [float(c) for c in self._exact_point(f, Fraction(u), Fraction(v))]
        return out

    def edge_sizes(self):
        """Each face's size: the geometric mean of its sides' lengths at its corner 0 (as chords)"""
        corners = self.corners()
        return np.sqrt(np.linalg.norm(corners[:, 1] - corners[:, 0], axis=1)
                       * np.linalg.norm(corners[:, 3] - corners[:, 0], axis=1))

    def twist_norms(self):
        return self._twist

    # ----- exactly -----

    def point(self, f, I, J, M):
        """Face f's point (I, J) of a level of M cells across, exactly: (ints, d), the place being the ints over d"""
        if self.regular[f]:
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
        p = self._exact_point(f, Fraction(I, M), Fraction(J, M))
        d = math.lcm(*(c.denominator for c in p))
        return tuple(c.numerator * (d // c.denominator) for c in p), d

    def point_q5(self, f, u, v):
        """Face f's point at (u, v), Q(sqrt 5) pairs, exactly: three such pairs, the mesh's units"""
        one = (Fraction(1), Fraction(0))
        if self.regular[f]:
            C, scale = self.exact_of(f), Fraction(1, self.denominator)
            s, t = u, v
        else:
            as_q = lambda x: x if isinstance(x, tuple) else (Fraction(x), Fraction(0))
            less = lambda a, b: q_sign(q_sub(as_q(a), as_q(b))) < 0
            leaf, point = self._exact_at(f, u, v, less)
            if leaf is None:
                return [(c, Fraction(0)) for c in point]
            C, scale = leaf.Q, Fraction(1)
            s = q_scale(q_sub(u, (leaf.u0, Fraction(0))), 1 / leaf.size)
            t = q_scale(q_sub(v, (leaf.v0, Fraction(0))), 1 / leaf.size)
        up, vp = [one], [one]
        for _ in range(3):
            up.append(q_mul(up[-1], s))
            vp.append(q_mul(vp[-1], t))
        out = []
        for k in range(3):
            total = (Fraction(0), Fraction(0))
            for a in range(4):
                for b in range(4):
                    if C[a][b][k]:
                        total = q_add(total, q_scale(q_mul(up[a], vp[b]), C[a][b][k]))
            out.append(q_scale(total, scale))
        return out


def sum_terms(E, gi, gj):
    """sum over a + b >= 1 of E[a][b] gi^a gj^b, for arrays of offsets (n,): (n, 3)"""
    E = np.array(E, dtype=float)
    E[0, 0] = 0.0
    return np.einsum("na,nb,abk->nk", powers(np.asarray(gi, dtype=float)), powers(np.asarray(gj, dtype=float)), E)


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
