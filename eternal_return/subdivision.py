"""Smooth charts for surface maps: a mesh's subdivision surface, the limit its faces tend to when split again and
again, as the shape the lattice stands on instead of the faces' flat (bilinear) patches (phase 6 of the unified
engines plan).

Two schemes, one machinery (SubdivisionChart), exactly:
- CatmullClarkChart: a closed all-quad mesh's Catmull-Clark limit surface. A regular face (all four corners where four
  faces meet) is the uniform bicubic B-spline patch of the 4 x 4 control points round it.
- LoopChart: a closed all-triangle mesh's Loop limit surface, with Warren's weights (a vertex's new point 5/8 itself
  and 3/(8n) each of its n neighbors, 7/16 and 3/16 where three triangles meet; an edge's 3/8 its ends, 1/8 the two
  corners across): rational, where Loop's own cos(2 pi / n) weights aren't. A regular face (all three corners where
  six triangles meet) is the quartic box-spline patch of the 12 control points round it, whose coefficients
  (loop_patch()) are worked out from the scheme itself, exactly: a regular lattice split twice, the patch through its
  limit points at the quarters.
A regular face's patch is a polynomial whose coefficients are the mesh's coordinates (ints over 2^E, as
surface.BilinearChart reads them) times whole numbers over a whole denominator: every lattice point I / M of every
level is a rational point, exactly, however deep, and faces sharing an edge give the same point on it to the last
bit.

A face with an extraordinary vertex (a corner where other than 4 quads, or 6 triangles, meet) is split by its scheme's
own rules, exactly in rationals, on a local copy of the faces round it (LocalMesh, LoopMesh: its faces within two
rings, all a split needs): of the four children, those away from the extraordinary corners are regular patches again,
and the one at such a corner is split again, toward it, for as long as a point asked for is nearer the corner (the
face then a tree of leaves: each a piece of the face's parameters and its polynomial, exact). A triangle's children
are its three corners' and the middle one, turned half round (its parameters run backwards: a leaf's sign). The
extraordinary vertex's own point is its limit position, rational too: Catmull-Clark's (n^2 v + 4 sum of its edge
neighbors + sum of its diagonal ones) / (n (n + 5)), Warren's (n v + sum of its neighbors) / (2 n), (2 v + sum) / 5
where three meet. Every weight is rational, so this is exact too, at any depth.

Near an extraordinary vertex a split shrinks the surface by its subdominant eigenvalue, not by the 1/2 the parameters
shrink by, so the lattice's cells there don't keep their size: diving exactly at such a corner, its cells drift
(Catmull-Clark: ~18% smaller a loop at 3 faces, ~10% bigger at 5). Anywhere else the corner's neighborhood shrinks
out of the view and the distortion fades. The user chose that over remapping the parameters there (irrational
exponents, no exact points) or flat corners. The bounds (sigma_lo, sigma_hi) are sampled away from the corners, so they
don't hold right at them.

The chart answers what surface.BilinearChart answers (see there): points and Jacobians in floats, in bulk; exact
points (point(), point_q5()); points as offsets from one placed exactly (grid_offsets(), nearest(), step()), from the
patches' Taylor coefficients about it worked out exactly, so they're good to a float's precision however small they
are next to the face, each point from its own leaf's patch, with the exact difference between that patch and the
surface at the base point added (so a leaf's polynomial is never stretched past its piece); normals, corners and
bounds. A patch lies in its control points' hull (both schemes' weights are positive), which gives each face's ball
(middle, radius).

The map object's surface settings choose it: "chart": "catmull_clark" or "loop" (surface.DEFAULTS, "bilinear" by
default).
"""

import math
import operator
import threading
from fractions import Fraction

import numpy as np

from .cyclotomic import q_add, q_mul, q_scale, q_sub

# The engine in Rust (the eternal_return_core wheel's), for the patches' arithmetic that's been moved there (poly_at(),
# taylor_steps(), derivatives(), evaluate(), newton(), side_newton()): the same numbers to a float's last few bits
# (numpy's matrix products round their sums in another order). A core that doesn't have a function yet (an older
# wheel, or none) leaves the Python here to do it.
try:
    from eternal_return_core import _native
except ImportError:
    _native = None


def _floats(a):
    """a as the core takes arrays: float64, in a row"""
    return np.ascontiguousarray(a, dtype=np.float64)

# The uniform cubic B-spline's basis in powers of u, times 6: B_i(u) = sum_a BASIS[a][i] u^a / 6
BASIS = ((1, 4, 1, 0), (-3, 0, 3, 0), (3, -6, 3, 0), (-1, 3, -3, 1))
# How many points a side of the grid the step bounds are sampled on (surface.SIGMA_GRID's way)
SIGMA_GRID = 9
# Newton steps nearest() takes at most, and how close its steps get before it stops
NEWTON_STEPS = 12
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
# How many levels down floats look for a regular piece by a border between quads and triangles, past which a point is
# its piece's corners' blend (a piece 2^-24 of the face across: within 2^-48 of it times the curvature)
BORDER_DEPTH = 24
COARSE_DEPTH = 10
NORMAL_DEPTH = 14
# By a border, a step's float points come from pieces this many levels finer than its tiles (BORDER_DEPTH at most):
# within some 4e-6 of a tile by the UV sphere's poles (4 levels: 0.5% for a tile half a face across)
STEP_EXTRA = 6
# A point is past a face's side by more than this (in its parameters, times the side's length), and goes across at most
# this many sides to the face it's on (relocated())
PAST_SIDE = 1e-12
MOST_HOPS = 8
# How many levels a normal at a piece's corner by a border goes on toward it, and to how deep at least (its rough
# piece's slope, from a piece 2^-3 of that one across, and at most 2^-8 of the face: a coarse level's corners' normals,
# from pieces three levels down, were a degree off)
CORNER_NORMAL = (3, 8)
# A normal asked nearer an extraordinary corner than this share of its face is taken this far from it (off_corners()):
# at a corner of fewer faces than regular (or on a rim), where the derivatives go to 0 (the faces round one agree on
# its normal to 0.3 degrees there, and nearer in the rounding shows: 0.4 degrees at 2^-20); at one of more, where they
# grow and the faces' normals close in slowly (round a corner of 10 triangles they're 4.6 degrees apart at 2^-10,
# 0.4 at 2^-20; of 16, 19 and 8)
CORNER_NUDGE = 2.0 ** -10
CROWDED_NUDGE = 2.0 ** -24
# The least step of a lattice whose offsets by a border are worked out in floats (_offsets())
FLOAT_STEP = Fraction(1, 1 << 24)
# Deeper than that, a tile of an irregular face is stepped from its float leaf's patch (_leaf_steps()) if it's at most
# this share of the leaf across, the leaf at least LEAF_LEAST of the face across, and its point in the leaf to within
# LEAF_SLACK of its size; else exactly. The float tree's pieces toward an extraordinary corner lose digits level by
# level, a corner of 3 triangles (a tetrahedron's) fastest: its leaves' steps were 1e-9 off at 2^-14 of the face,
# 1.5e-6 at 2^-20, 8e-5 at 2^-22; a cube's 7.5e-12 at 2^-14
LEAF_SHARE = 1.0 / 64.0
LEAF_LEAST = 2.0 ** -14
LEAF_SLACK = 1e-9
# How many nodes the trees keep before the float ones are let go of
NODES_KEPT = 60000


class ExtraordinaryError(ValueError):
    pass


# ----- Catmull-Clark -----

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


def quad_control(loops, faces_of, f, regular):
    """grid_of()'s 16 control points, row by row (i * 4 + j), or None"""
    g = grid_of(loops, faces_of, f, regular)
    return None if g is None else [g[i][j] for i in range(4) for j in range(4)]


class LocalMesh:
    """A few quads of a mesh: their corners' points (id -> numbers over scale: whole numbers over a whole scale,
    exactly, or floats over 1.0; any number of coordinates: a Shape's are unit vectors), each vertex's valence in the whole mesh (how many faces it has there), and the quads
    (loops of ids). A vertex has its whole ring here when as many of its faces are. (In Fractions, every sum reduced
    by a gcd: quadico's first floor split its faces' trees for 3 s.)"""

    def __init__(self, points, valence, loops, scale, boundary=frozenset()):
        self.points, self.valence, self.loops, self.scale = points, valence, loops, scale
        self.exact = isinstance(scale, int)
        self.faces_of = {}
        for t, loop in enumerate(loops):
            for x in loop:
                self.faces_of.setdefault(x, set()).add(t)
        # The whole mesh's open edges here (a local mesh's own rim isn't one), as (least, greatest) ids, and each
        # vertex on one its two neighbors along them (rim): split and limit by the rim's cubic B-spline
        self.boundary = frozenset(boundary)
        self.rim = {}
        for a, b in self.boundary:
            self.rim.setdefault(a, []).append(b)
            self.rim.setdefault(b, []).append(a)

    def _finer_boundary(self, ids):
        """The boundary of this mesh's split: each open edge's halves (ids: what made each new vertex -> its id)"""
        out = set()
        for a, b in self.boundary:
            m = ids.get(("e", a, b))
            for end in (a, b):
                x = ids.get(("v", end))
                if m is not None and x is not None:
                    out.add((min(m, x), max(m, x)))
        return frozenset(out)

    def _rim_limit(self, x):
        """A vertex on the boundary's limit point: its rim's cubic B-spline's, (p + 4 x + q) / 6"""
        p, q = self.rim[x]
        P = self.points
        return self._over([P[p][k] + 4 * P[x][k] + P[q][k] for k in range(len(P[x]))], 6)

    def complete(self, x):
        return len(self.faces_of.get(x, ())) == self.valence[x]

    def regular(self, x):
        return self.valence[x] == 4 and self.complete(x) and x not in self.rim

    def kind_regular(self, x, sides):
        """Whether vertex x is a regular corner for a face of so many sides (here: its scheme's valence; never on the
        boundary)"""
        return self.valence[x] == 4 and x not in self.rim

    def control(self, t):
        return quad_control(self.loops, self.faces_of, t, self.regular)

    def around(self, t, rings=2):
        """The faces within rings of t (sharing a corner, again and again) as a mesh of their own, and t's index there"""
        keep = {t}
        for _ in range(rings):
            keep |= {g for f in keep for x in self.loops[f] for g in self.faces_of[x]}
        order = sorted(keep)
        loops = [self.loops[f] for f in order]
        used = {x for loop in loops for x in loop}
        return (type(self)({x: self.points[x] for x in used}, {x: self.valence[x] for x in used}, loops, self.scale,
                           {e for e in self.boundary if e[0] in used and e[1] in used}),
                order.index(t))

    def _over(self, sums, under):
        """Sums over a whole number, as the point they are: Fractions, or floats"""
        under = self.scale * under
        return tuple(Fraction(c, under) for c in sums) if self.exact else tuple(c / under for c in sums)

    def limit(self, x):
        """Vertex x's point on the limit surface (its ring must be whole here): Fractions, or floats"""
        if x in self.rim:
            return self._rim_limit(x)
        n = self.valence[x]
        P = self.points
        # Each face's next corner after x (each edge once round a whole ring) and its far corner, as lists: where two
        # quads meet on two edges (Suzanne's chin, a corner of two faces), the corner beside one is two quads' far
        # corner, and sets counted it once (a limit point 0.2 off)
        edge, diagonal = [], []
        for t in self.faces_of[x]:
            loop = self.loops[t]
            k = loop.index(x)
            edge.append(loop[(k + 1) % 4])
            diagonal.append(loop[(k + 2) % 4])
        sums = [n * n * P[x][k] + 4 * sum(P[y][k] for y in edge) + sum(P[y][k] for y in diagonal)
                for k in range(len(P[x]))]
        return self._over(sums, n * (n + 5))

    def subdivide(self):
        """One Catmull-Clark split: (the finer mesh, for each face its four children's indices there in
        CATMULL_CLARK.children' order, None for a child some of whose corners can't be worked out here: a vertex
        without its whole ring, an edge without both its faces). Exactly, its points are whole numbers over this scale
        times every rule's denominator (16 and 4 n^2)."""
        P = self.points
        dims = len(next(iter(P.values())))
        sums = [tuple(P[a][k] + P[b][k] + P[c][k] + P[d][k] for k in range(dims)) for a, b, c, d in self.loops]
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
            add(("f", t), tuple(c * w for c in S), 4)
        w = weight(16)
        for (a, b), faces in edge_faces.items():
            if len(faces) == 2:
                S0, S1 = sums[faces[0]], sums[faces[1]]
                add(("e", a, b), tuple((4 * (P[a][k] + P[b][k]) + S0[k] + S1[k]) * w for k in range(dims)), 4)
            elif (a, b) in self.boundary:
                # An open edge's middle (the rim's B-spline's), between its quad's two children
                add(("e", a, b), tuple((P[a][k] + P[b][k]) * weight(2) for k in range(dims)), 2)
        for x in whole:
            n = self.valence[x]
            if x in self.rim:
                p, q = self.rim[x]
                add(("v", x), tuple((P[p][k] + 6 * P[x][k] + P[q][k]) * weight(8) for k in range(dims)), n)
                continue
            faces = self.faces_of[x]
            # (Each face's next corner after x: each edge once, as in limit())
            ring = [self.loops[t][(self.loops[t].index(x) + 1) % 4] for t in faces]
            # (Q + 2 R + (n - 3) P) / n, Q the faces' points' mean, R the edges' middles': over 4 n^2
            w = weight(4 * n * n)
            add(("v", x), tuple((sum(sums[t][k] for t in faces) + 4 * sum(P[x][k] + P[y][k] for y in ring)
                                 + 4 * n * (n - 3) * P[x][k]) * w for k in range(dims)), n)
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
        return LocalMesh(points, valence, loops, scale, self._finer_boundary(ids)), children


# ----- Loop, with Warren's weights -----

# A regular triangle's 12 control points, as places on its triangular lattice (its corners at (0, 0), (1, 0), (0, 1),
# the point across an edge from a corner its ends' sum less the corner): its corners, then their other neighbors
CONTROL12 = ((0, 0), (1, 0), (0, 1), (-1, 0), (-1, 1), (0, -1), (1, -1), (2, -1), (2, 0), (1, 1), (0, 2), (-1, 2))


def loop_control(loops, faces_of, f, regular):
    """Triangle f's 12 control points (vertex ids, in CONTROL12's order), or None unless every corner is regular (six
    triangles, all there): its corners' fans placed on the triangular lattice, a triangle across an edge at a time"""
    a, b, c = loops[f]
    if not all(regular(x) for x in (a, b, c)):
        return None
    fan = faces_of[a] | faces_of[b] | faces_of[c]
    place = {a: (0, 0), b: (1, 0), c: (0, 1)}
    queue, seen = [f], {f}
    while queue:
        g = queue.pop()
        loop = loops[g]
        for s in range(3):
            x, y, z = loop[s], loop[(s + 1) % 3], loop[(s + 2) % 3]
            for h in (faces_of[x] & faces_of[y]) - {g}:
                if h in seen or h not in fan:
                    continue
                w = next(v for v in loops[h] if v not in (x, y))
                at = (place[x][0] + place[y][0] - place[z][0], place[x][1] + place[y][1] - place[z][1])
                if place.setdefault(w, at) != at:
                    return None
                seen.add(h)
                queue.append(h)
    by_place = {at: v for v, at in place.items()}
    if len(by_place) != len(place) or any(at not in by_place for at in CONTROL12):
        return None
    return [by_place[at] for at in CONTROL12]


class LoopMesh(LocalMesh):
    """A few triangles of a mesh (LocalMesh's, split by Loop's rules with Warren's weights). Its points may have any
    number of coordinates (loop_patch() splits 14 at once)."""

    def regular(self, x):
        return self.valence[x] == 6 and self.complete(x) and x not in self.rim

    def kind_regular(self, x, sides):
        return self.valence[x] == 6 and x not in self.rim

    def control(self, t):
        return loop_control(self.loops, self.faces_of, t, self.regular)

    def ring(self, x):
        return {y for t in self.faces_of[x] for y in self.loops[t] if y != x}

    def limit(self, x):
        """Vertex x's point on the limit surface (its ring must be whole here): (n v + its neighbors) / 2n, or
        (2 v + its neighbors) / 5 where three triangles meet (Warren's weights' limit); on the boundary, its rim's"""
        if x in self.rim:
            return self._rim_limit(x)
        n = self.valence[x]
        P = self.points
        ring = self.ring(x)
        own, under = (2, 5) if n == 3 else (n, 2 * n)
        return self._over([own * P[x][k] + sum(P[y][k] for y in ring) for k in range(len(P[x]))], under)

    def subdivide(self):
        """One Loop split with Warren's weights: (the finer mesh, for each triangle its four children's indices there in
        LOOP.children' order, None for one that can't be worked out here). Exactly, its points are whole numbers over
        this scale times every rule's denominator (8, 16 and 8 n)."""
        P = self.points
        dims = len(next(iter(P.values())))
        whole = [x for x, faces in self.faces_of.items() if len(faces) == self.valence[x]]
        if self.exact:
            L = math.lcm(16, *(8 * self.valence[x] for x in whole))
            weight, scale = (lambda under: L // under), self.scale * L
        else:
            weight, scale = (lambda under: 1.0 / under), 1.0
        across = {}
        for loop in self.loops:
            for s in range(3):
                a, b, c = loop[s], loop[(s + 1) % 3], loop[(s + 2) % 3]
                across.setdefault((min(a, b), max(a, b)), []).append(c)
        points, valence, ids = {}, {}, {}

        def add(key, point, n):
            ids[key] = len(points)
            points[ids[key]] = point
            valence[ids[key]] = n

        w = weight(8)
        for (a, b), corners in across.items():
            if len(corners) == 2:
                c, d = corners
                add(("e", a, b), tuple((3 * (P[a][k] + P[b][k]) + P[c][k] + P[d][k]) * w for k in range(dims)), 6)
            elif (a, b) in self.boundary:
                # An open edge's middle (the rim's B-spline's), among its triangle's three children
                add(("e", a, b), tuple((P[a][k] + P[b][k]) * weight(2) for k in range(dims)), 3)
        for x in whole:
            n = self.valence[x]
            if x in self.rim:
                p, q = self.rim[x]
                add(("v", x), tuple((P[p][k] + 6 * P[x][k] + P[q][k]) * weight(8) for k in range(dims)), n)
                continue
            ring = self.ring(x)
            own, others, w = (7, 3, weight(16)) if n == 3 else (5 * n, 3, weight(8 * n))
            add(("v", x), tuple((own * P[x][k] + others * sum(P[y][k] for y in ring)) * w for k in range(dims)), n)
        loops, children = [], []
        for a, b, c in self.loops:
            e = lambda x, y: ids.get(("e", min(x, y), max(x, y)))
            v = lambda x: ids.get(("v", x))
            row = []
            for triangle in ((v(a), e(a, b), e(c, a)), (e(a, b), v(b), e(b, c)), (e(c, a), e(b, c), v(c)),
                             (e(b, c), e(c, a), e(a, b))):
                if None in triangle:
                    row.append(None)
                else:
                    row.append(len(loops))
                    loops.append(triangle)
            children.append(row)
        return LoopMesh(points, valence, loops, scale, self._finer_boundary(ids)), children


# ----- quads and triangles: Stam and Loop's scheme -----

# A vertex of a split on the mesh's boundary, smoothed along it (QuadTriMesh.split())
RIM = "rim"


class QuadTriMesh(LocalMesh):
    """A few quads and triangles of a mesh, split by Stam and Loop's quad/triangle rules ("Quad/Triangle Subdivision",
    2003): each edge halved, a quad into four round its middle, a triangle into four (Loop's way), then every vertex
    moved to a weighted average of itself (alpha), its neighbors along edges (beta) and across its quads (gamma):
    - among quads only, Catmull-Clark's: (n - 3) / n, 2 / n^2, 1 / n^2 (n its edges), which makes this Catmull-Clark
    - among triangles only, Warren's Loop: 1/4 and 3 / 4n (-1/8 and 3/8 where three meet), which makes it LoopMesh's
    - among both, the paper's: alpha = 1 / (1 + n_e / 2 + n_q / 4), beta = alpha / 2, gamma = alpha / 4 (n_e its
      edges, n_q its quads)
    All rational, so exact. (The paper's correction of corners of four or fewer edges among both, by factors found
    numerically, isn't applied.) Its points may have any number of coordinates (mixed_mask() splits unit vectors)."""

    def kind_regular(self, x, sides):
        """Whether x is a regular corner for a face of so many sides: four quads, or six triangles, and no other kind (and
        not on the boundary)"""
        return (self.valence[x] == (4 if sides == 4 else 6) and self.complete(x) and x not in self.rim
                and all(len(self.loops[t]) == sides for t in self.faces_of[x]))

    def control(self, t):
        sides = len(self.loops[t])
        regular = lambda x: self.kind_regular(x, sides)
        if sides == 4:
            return quad_control(self.loops, self.faces_of, t, regular)
        return loop_control(self.loops, self.faces_of, t, regular)

    def ring(self, x):
        """Vertex x's neighbors along edges (LoopMesh.limit()'s ring, among triangles)"""
        return self.neighbors(x)[0]

    def neighbors(self, x):
        """Vertex x's neighbors along edges, and across its quads: lists, each face's next corner after x (each edge once
        round a whole ring) and a quad's far corner (as LocalMesh.limit() counts them)"""
        edge, diagonal = [], []
        for t in self.faces_of[x]:
            loop = self.loops[t]
            k, n = loop.index(x), len(loop)
            edge.append(loop[(k + 1) % n])
            if n == 4:
                diagonal.append(loop[(k + 2) % 4])
        return edge, diagonal

    def weights(self, x):
        """Vertex x's smoothing weights (alpha, beta, gamma) as whole numbers over a whole number"""
        sizes = [len(self.loops[t]) for t in self.faces_of[x]]
        quads, triangles = sizes.count(4), sizes.count(3)
        n = self.valence[x]
        if not triangles:
            return n * (n - 3), 2, 1, n * n
        if not quads:
            return (-1, 3, 0, 8) if n == 3 else (n, 3, 0, 4 * n)
        return 4, 2, 1, 4 + 2 * n + quads

    def limit(self, x):
        """Vertex x's point on the limit surface (its ring must be whole here): Catmull-Clark's or Warren's among one
        kind of face, else its subdivision matrix's left eigenvector (mixed_mask()); on the boundary, its rim's"""
        if x in self.rim:
            return self._rim_limit(x)
        sizes = {len(self.loops[t]) for t in self.faces_of[x]}
        if sizes == {4}:
            return LocalMesh.limit(self, x)
        if sizes == {3}:
            return LoopMesh.limit(self, x)
        mask, roles = mixed_mask(self, x)
        P = self.points
        if self.exact:
            under = math.lcm(*(m.denominator for m in mask))
            sums = [sum(int(m * under) * P[y][k] for m, y in zip(mask, roles)) for k in range(len(P[x]))]
            return self._over(sums, under)
        return tuple(sum(float(m) * P[y][k] for m, y in zip(mask, roles)) for k in range(len(P[x])))

    def split(self):
        """subdivide(), and the new vertices' ids by what they were made from: ("v", x), ("e", a, b), ("f", t). Each
        new vertex's neighbors in the linear split straight from what made it (an old vertex's: its edges' middles,
        across its quads their middles; an edge's middle's: its ends, and in each of its faces a quad's middle, or a
        triangle's other edges' middles, across a quad the middles of its sides beside it; a quad's middle's: its
        sides' middles, across them its corners): building the linear split as a mesh and looking them up took most
        of the time of a mixed mesh's trees."""
        P = self.points
        dims = len(next(iter(P.values())))
        loops = self.loops
        sizes = [len(loop) for loop in loops]
        edge_faces = {}
        for t, loop in enumerate(loops):
            n = sizes[t]
            for s_ in range(n):
                a, b = loop[s_], loop[(s_ + 1) % n]
                edge_faces.setdefault((min(a, b), max(a, b)), []).append(t)
        key_e = lambda a, b: ("e", min(a, b), max(a, b))
        # The linear split's points, four times over (whole numbers: halves and quarters), and who smooths
        lin = {}
        for x in self.faces_of:
            lin[("v", x)] = tuple(4 * c for c in P[x])
        for (a, b) in edge_faces:
            lin[("e", a, b)] = tuple(2 * (P[a][k] + P[b][k]) for k in range(dims))
        for t, loop in enumerate(loops):
            if sizes[t] == 4:
                lin[("f", t)] = tuple(sum(P[x][k] for x in loop) for k in range(dims))
        smooth = {}            # new vertex -> (edge neighbors, diagonals, edges, quads, kinds of faces; RIM on the boundary)
        for x, faces in self.faces_of.items():
            if len(faces) != self.valence[x]:
                continue
            if x in self.rim:
                # On the boundary: smoothed along the rim alone (a quarter of each of its two rim edges' middles and half
                # itself: the rim's B-spline, (p + 6 x + q) / 8)
                p, q = self.rim[x]
                smooth[("v", x)] = ([key_e(x, p), key_e(x, q)], [], self.valence[x], 0, RIM)
                continue
            edge, diagonal, kinds = [], [], set()
            for t in faces:
                loop, n = loops[t], sizes[t]
                k = loop.index(x)
                kinds.add(n)
                edge.append(key_e(x, loop[(k + 1) % n]))
                if n == 4:
                    diagonal.append(("f", t))
            smooth[("v", x)] = (edge, diagonal, self.valence[x], len(diagonal), kinds)
        for (a, b), faces in edge_faces.items():
            if len(faces) != 2:
                if (a, b) in self.boundary:
                    # An open edge's middle stays one (a quarter of each end, half itself); its faces in the whole
                    # mesh: its quad's two children or its triangle's three
                    smooth[("e", a, b)] = ([("v", a), ("v", b)], [], 2 if sizes[faces[0]] == 4 else 3, 0, RIM)
                continue
            edge, diagonal, kinds, quads = [("v", a), ("v", b)], [], set(), 0
            for t in faces:
                loop, n = loops[t], sizes[t]
                kinds.add(n)
                if n == 4:
                    edge.append(("f", t))
                    ka, kb = loop.index(a), loop.index(b)
                    # The quad's other corners: the one beside b, and the one beside a
                    step = 1 if (kb - ka) % 4 == 1 else -1
                    beside_b, beside_a = loop[(kb + step) % 4], loop[(ka - step) % 4]
                    diagonal += [key_e(b, beside_b), key_e(a, beside_a)]
                    quads += 2
                else:
                    c = next(y for y in loop if y not in (a, b))
                    edge += [key_e(b, c), key_e(c, a)]
            smooth[("e", a, b)] = (edge, diagonal, len(edge), quads, kinds)
        for t, loop in enumerate(loops):
            if sizes[t] == 4:
                smooth[("f", t)] = ([key_e(loop[i], loop[(i + 1) % 4]) for i in range(4)],
                                    [("v", y) for y in loop], 4, 4, {4})

        def weights(n, quads, kinds):
            if kinds is RIM:
                return 2, 1, 0, 4
            if kinds == {4}:
                return n * (n - 3), 2, 1, n * n
            if kinds == {3}:
                return (-1, 3, 0, 8) if n == 3 else (n, 3, 0, 4 * n)
            return 4, 2, 1, 4 + 2 * n + quads

        found = {key: weights(*value[2:]) for key, value in smooth.items()}
        if self.exact:
            L = math.lcm(*(w[3] for w in found.values())) if found else 1
            scale = self.scale * 4 * L
        else:
            L, scale = None, 1.0
        ids, points, valence = {}, {}, {}
        for key, (edge, diagonal, n, quads, kinds) in smooth.items():
            a, b, c, under = found[key]
            if self.exact:
                k = L // under
                a, b, c = a * k, b * k, c * k
            else:
                a, b, c = a / (4.0 * under), b / (4.0 * under), c / (4.0 * under)
            own = lin[key]
            E = [lin[z] for z in edge]
            D = [lin[z] for z in diagonal]
            ids[key] = len(points)
            points[ids[key]] = tuple(a * own[i] + b * sum(p[i] for p in E) + c * sum(p[i] for p in D)
                                     for i in range(dims))
            valence[ids[key]] = n
        e = lambda x, y: ids.get(key_e(x, y))
        v = lambda x: ids.get(("v", x))
        out, children = [], []
        for t, loop in enumerate(loops):
            if sizes[t] == 4:
                a, b, c, d = loop
                F = ids.get(("f", t))
                pieces = ((v(a), e(a, b), F, e(d, a)), (e(a, b), v(b), e(b, c), F), (F, e(b, c), v(c), e(c, d)),
                          (e(d, a), F, e(c, d), v(d)))
            else:
                a, b, c = loop
                pieces = ((v(a), e(a, b), e(c, a)), (e(a, b), v(b), e(b, c)), (e(c, a), e(b, c), v(c)),
                          (e(b, c), e(c, a), e(a, b)))
            row = []
            for piece in pieces:
                if None in piece:
                    row.append(None)
                else:
                    row.append(len(out))
                    out.append(piece)
            children.append(row)
        return QuadTriMesh(points, valence, out, scale, self._finer_boundary(ids)), children, ids

    def subdivide(self):
        finer, children, _ids = self.split()
        return finer, children


_mixed_masks = {}


def mixed_mask(mesh, x):
    """Vertex x's limit mask among quads and triangles both: (weights, Fractions, the vertices they weigh): its
    subdivision matrix's left eigenvector for 1, over x, its neighbors along edges and across its quads, worked out
    once for each arrangement of faces round a vertex (round it in order, from the turn whose sizes read least) on a
    fan of its own whose points are unit vectors (mesh.split() of them: a row of the matrix each new vertex)"""
    # The faces round x in order: each one's corner after x (its edge to the next face) and its diagonal
    after = {}
    for t in mesh.faces_of[x]:
        loop = mesh.loops[t]
        k, n = loop.index(x), len(loop)
        after[loop[(k - 1) % n]] = (t, loop[(k + 1) % n], loop[(k + 2) % 4] if n == 4 else None)
    start = min(after)
    ring = []
    y = start
    for _ in range(len(after)):
        t, nxt, diagonal = after[y]
        ring.append((y, len(mesh.loops[t]), diagonal))
        y = nxt
    sizes = tuple(size for _y, size, _d in ring)
    turn = min(range(len(ring)), key=lambda r: sizes[r:] + sizes[:r])
    ring = ring[turn:] + ring[:turn]
    key = tuple(size for _y, size, _d in ring)
    roles = [x] + [y for y, _size, _d in ring] + [d for _y, size, d in ring if size == 4]
    found = _mixed_masks.get(key)
    if found is None:
        m = len(ring)
        # The fan: center 0, edge neighbors 1..m, diagonals after; face i between edges i and i + 1
        quads = [i for i in range(m) if key[i] == 4]
        diag = {i: m + 1 + j for j, i in enumerate(quads)}
        loops = []
        for i in range(m):
            e0, e1 = 1 + i, 1 + (i + 1) % m
            loops.append((0, e1, diag[i], e0) if key[i] == 4 else (0, e1, e0))
        R = 1 + m + len(quads)
        points = {r: tuple(int(r == c) for c in range(R)) for r in range(R)}
        valence = {r: 99 for r in range(R)}
        valence[0] = m
        finer, _children, ids = QuadTriMesh(points, valence, loops, 1).split()
        new = [ids[("v", 0)]] + [ids[("e", 0, 1 + i)] for i in range(m)] + [ids[("f", i)] for i in quads]
        S = [[Fraction(c, finer.scale) for c in finer.points[y]] for y in new]
        # l S = l, sum l = 1: (S^T - I) l = 0, the last row swapped for the sum
        A = [[S[j][i] - (1 if i == j else 0) for j in range(R)] for i in range(R)]
        A[-1] = [Fraction(1)] * R
        b = [[Fraction(0)] for _ in range(R)]
        b[-1] = [Fraction(1)]
        found = _mixed_masks[key] = [row[0] for row in solve_exactly(A, b)]
    return found, roles


_loop_patch = None


def loop_patch():
    """The regular Loop patch: (M, D), its coefficient of s^a t^b being sum_i M[a][b][i] P_i / D over its 12 control
    points P (CONTROL12's order), a + b <= 4 (M (5, 5, 12) whole numbers). Worked out from the scheme itself, exactly
    and once: a regular triangular lattice whose points are the control points' indicator vectors (and their lattice
    places, which Loop's rules keep: they reproduce what's linear) split twice, and the quartic through the limit
    points of the 15 quarter points of the triangle (0, 0), (1, 0), (0, 1)."""
    global _loop_patch
    if _loop_patch is None:
        ids = {}
        vid = lambda i, j: ids.setdefault((i, j), len(ids))
        loops = []
        for i in range(-4, 7):
            for j in range(-4, 7):
                loops.append((vid(i, j), vid(i + 1, j), vid(i, j + 1)))
                loops.append((vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)))
        points = {}
        for (i, j), x in ids.items():
            p = [i, j] + [0] * len(CONTROL12)
            if (i, j) in CONTROL12:
                p[2 + CONTROL12.index((i, j))] = 1
            points[x] = tuple(p)
        mesh = LoopMesh(points, {x: 6 for x in ids.values()}, loops, 1)
        for _ in range(2):
            mesh = mesh.subdivide()[0]
        by_place = {(Fraction(p[0], mesh.scale), Fraction(p[1], mesh.scale)): x for x, p in mesh.points.items()}
        monomials = [(a, b) for a in range(5) for b in range(5 - a)]
        rows, values = [], []
        for i in range(5):
            for j in range(5 - i):
                s, t = Fraction(i, 4), Fraction(j, 4)
                rows.append([s ** a * t ** b for a, b in monomials])
                values.append(list(mesh.limit(by_place[(s, t)])))
        solved = solve_exactly(rows, values)
        D = math.lcm(*(c.denominator for row in solved for c in row))
        M = [[[0] * len(CONTROL12) for _b in range(5)] for _a in range(5)]
        for (a, b), row in zip(monomials, solved):
            # The lattice places' own rows must come out as s and t
            assert row[0] == (1 if (a, b) == (1, 0) else 0) and row[1] == (1 if (a, b) == (0, 1) else 0)
            M[a][b] = [int(c * D) for c in row[2:]]
        _loop_patch = (M, D)
    return _loop_patch


def solve_exactly(A, B):
    """A X = B for square A (lists of Fractions), B with any number of columns: X, exactly (Gaussian elimination)"""
    n = len(A)
    rows = [list(map(Fraction, a)) + list(map(Fraction, b)) for a, b in zip(A, B)]
    for col in range(n):
        pivot = next(r for r in range(col, n) if rows[r][col] != 0)
        rows[col], rows[pivot] = rows[pivot], rows[col]
        p = rows[col][col]
        rows[col] = [x / p for x in rows[col]]
        for r in range(n):
            if r != col and rows[r][col] != 0:
                k = rows[r][col]
                rows[r] = [x - k * y for x, y in zip(rows[r], rows[col])]
    return [row[n:] for row in rows]


# ----- the schemes -----

class Scheme:
    """What a subdivision scheme's faces are: sides; corners (their places in a face's own parameters, in its loop's
    order); children (each child's corner 0's place in its parent's parameters, in halves, and its sign: +1 a shrunk
    copy, -1 turned half round), in the order its mesh's subdivide() lists them; the valence its regular vertices
    have; its local mesh; the regular patch, (M, D), a coefficient being sum_i M[a][b][i] P_i / D over the control
    points (order: one more than the patch's degree in either parameter); a face's middle"""

    def child(self, s, t):
        """Which child the point (s, t) of a face's own parameters is in (floats)"""
        raise NotImplementedError

    def children_of(self, ds, dt, half):
        """child() for arrays of points given as their distance from the face's middle (s - 1/2, t - 1/2), all in any
        one scale, half being the 1/2 in it (in a node's steps, deep down, 1/2 may be a vast number of them: adding it
        lost the small ones' sign)"""
        raise NotImplementedError

    def child_exactly(self, s, t, less, add, half):
        """child() for exact numbers, with less(a, b) their a < b"""
        raise NotImplementedError

    def clamp(self, u, v):
        raise NotImplementedError

    def sample(self, u, v, inset=0.0):
        """A grid's points of the unit square (u, v arrays) onto the face, inset from its sides"""
        raise NotImplementedError


class CatmullClarkScheme(Scheme):
    name = "catmull_clark"
    sides = 4
    corners = ((0, 0), (1, 0), (1, 1), (0, 1))
    children = (((0, 0), 1), ((1, 0), 1), ((1, 1), 1), ((0, 1), 1))
    valence = 4
    mesh = LocalMesh
    control = staticmethod(quad_control)
    order = 4
    middle = (0.5, 0.5)

    @property
    def patch(self):
        M = [[[BASIS[a][i] * BASIS[b][j] for i in range(4) for j in range(4)] for b in range(4)] for a in range(4)]
        return M, 36

    def across(self, corner):
        """The child standing in for the one at an extraordinary corner: the one across from it"""
        return (corner + 2) % 4

    def child(self, s, t):
        return ((0, 3), (1, 2))[int(s >= 0.5)][int(t >= 0.5)]

    def children_of(self, ds, dt, half):
        i, j = ds >= 0.0, dt >= 0.0
        return np.where(i, np.where(j, 2, 1), np.where(j, 3, 0))

    def child_exactly(self, s, t, less, add, half):
        return ((0, 3), (1, 2))[int(not less(s, half))][int(not less(t, half))]

    def holds(self, s, t, less, add, number):
        """Whether the face holds the exact point (s, t) of its parameters"""
        zero, one = number(0), number(1)
        return not (less(s, zero) or less(t, zero) or less(one, s) or less(one, t))

    def clamp(self, u, v):
        return min(max(u, 0.0), 1.0), min(max(v, 0.0), 1.0)

    def inside(self, u, v):
        return 0.0 < u < 1.0 and 0.0 < v < 1.0

    def sample(self, u, v, inset=0.0):
        return np.clip(u, inset, 1.0 - inset), np.clip(v, inset, 1.0 - inset)


class LoopScheme(Scheme):
    name = "loop"
    sides = 3
    corners = ((0, 0), (1, 0), (0, 1))
    children = (((0, 0), 1), ((1, 0), 1), ((0, 1), 1), ((1, 1), -1))
    valence = 6
    mesh = LoopMesh
    control = staticmethod(loop_control)
    order = 5
    middle = (1.0 / 3.0, 1.0 / 3.0)

    @property
    def patch(self):
        return loop_patch()

    def across(self, corner):
        """The child standing in for the one at an extraordinary corner: the next corner's (touching it at a point; not
        the middle one, whose control points wrap round a corner where three triangles meet, never regular there)"""
        return (corner + 1) % 3

    def child(self, s, t):
        return 1 if s >= 0.5 else 2 if t >= 0.5 else 0 if s + t < 0.5 else 3

    def children_of(self, ds, dt, half):
        return np.where(ds >= 0.0, 1, np.where(dt >= 0.0, 2, np.where(ds + dt < -half, 0, 3)))

    def child_exactly(self, s, t, less, add, half):
        return 1 if not less(s, half) else 2 if not less(t, half) else 0 if less(add(s, t), half) else 3

    def holds(self, s, t, less, add, number):
        zero = number(0)
        return not (less(s, zero) or less(t, zero) or less(number(1), add(s, t)))

    def clamp(self, u, v):
        u, v = max(u, 0.0), max(v, 0.0)
        over = u + v - 1.0
        if over > 0.0:
            u, v = max(u - over / 2.0, 0.0), max(v - over / 2.0, 0.0)
            over = u + v - 1.0
            if over > 0.0:
                u, v = (u - over, v) if u > v else (u, v - over)
        return u, v

    def inside(self, u, v):
        return u > 0.0 and v > 0.0 and u + v < 1.0

    def sample(self, u, v, inset=0.0):
        return inset + (1.0 - 3.0 * inset) * u * (1.0 - v), inset + (1.0 - 3.0 * inset) * v


CATMULL_CLARK = CatmullClarkScheme()
LOOP = LoopScheme()

# Exact numbers' arithmetic for _exact_at(): Fractions', and Q(sqrt 5) pairs' (a + b sqrt 5)
FRACTIONS = (operator.lt, operator.add, operator.sub, operator.mul, Fraction)


def q_less(a, b):
    return q_sign(q_sub(as_q(a), as_q(b))) < 0


def as_q(x):
    return x if isinstance(x, tuple) else (Fraction(x), Fraction(0))


Q5 = (q_less, lambda a, b: q_add(as_q(a), as_q(b)), lambda a, b: q_sub(as_q(a), as_q(b)),
      lambda a, k: q_scale(as_q(a), k), lambda x: (Fraction(x), Fraction(0)))


# ----- polynomials -----

def exact_ints(positions):
    """The positions' float32 coordinates as ints over 2^E, exactly: (E, [(x, y, z) per vertex])"""
    ratios = [v.as_integer_ratio() for v in positions.ravel().tolist()]
    E = max(den.bit_length() - 1 for _, den in ratios)
    ints = [num << (E - (den.bit_length() - 1)) for num, den in ratios]
    return E, [tuple(ints[3 * k:3 * k + 3]) for k in range(len(positions))]


def powers(t, derivative=0, order=4):
    """(1, t, t^2, ...) to t^(order - 1), differentiated derivative times, on t's last axis (..., order)"""
    t = np.asarray(t, dtype=float)
    one, zero = np.ones_like(t), np.zeros_like(t)
    if order == 4:
        if derivative == 0:
            return np.stack([one, t, t * t, t * t * t], axis=-1)
        if derivative == 1:
            return np.stack([zero, one, 2.0 * t, 3.0 * t * t], axis=-1)
        if derivative == 2:
            return np.stack([zero, zero, 2.0 * one, 6.0 * t], axis=-1)
    terms = []
    for k in range(order):
        terms.append(zero if k < derivative else math.perm(k, derivative) * t ** (k - derivative))
    return np.stack(terms, axis=-1)


def shifted(coefficients, u0, v0):
    """A polynomial's coefficients ((n, n, 3): of u^a v^b, exact numbers or floats) about (u0, v0): the coefficients of
    du^a dv^b (the binomial expansion)"""
    n = len(coefficients)
    out = [[[0, 0, 0] for _b in range(n)] for _a in range(n)]
    up = [u0 ** k for k in range(n)]
    vp = [v0 ** k for k in range(n)]
    for i in range(n):
        for j in range(n):
            c = coefficients[i][j]
            if not any(c):
                continue
            for a in range(i + 1):
                ka = math.comb(i, a) * up[i - a]
                for b in range(j + 1):
                    k = ka * math.comb(j, b) * vp[j - b]
                    o = out[a][b]
                    o[0] += k * c[0]
                    o[1] += k * c[1]
                    o[2] += k * c[2]
    return out


def patch_ints(patch, P, scale):
    """A regular patch's coefficients from its control points P (whole numbers over scale): (D, whole numbers), the
    coefficients times D"""
    M, D = patch
    n = len(M)
    return D * scale, [[[sum(m * p[k] for m, p in zip(M[a][b], P) if m) for k in range(3)] for b in range(n)]
                       for a in range(n)]


def padded(patch, order):
    """A patch (M, D) with M grown to order x order with zeros (a mixed mesh's quads' and triangles' coefficients in
    one array)"""
    M, D = patch
    n, count = len(M), len(M[0][0])
    return [[M[a][b] if a < n and b < n else [0] * count for b in range(order)] for a in range(order)], D


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


class Leaf:
    """A regular piece of a face: where it is in the face's parameters (u0, v0: its corner 0, size, sign: Fractions;
    its own s = sign (u - u0) / size, t likewise), its patch's coefficients of s^a t^b exactly (Q: Fractions, the
    mesh's units) and as floats"""

    def __init__(self, u0, v0, size, sign, Q=None, ints=None):
        """Q: the coefficients (floats, or Fractions); or ints: (D, whole numbers), the coefficients times D. A float
        tree's leaves: Q an array, the place floats."""
        self.u0, self.v0, self.size, self.sign = u0, v0, size, sign
        self._Q, self._ints = Q, ints
        if isinstance(Q, np.ndarray):
            self.Qf = Q
        elif ints is not None:
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

    def local(self, u, v):
        """The face's exact (u, v) in the leaf's own parameters"""
        return self.sign * (u - self.u0) / self.size, self.sign * (v - self.v0) / self.size

    def expansion(self, U, V, h, S0, anchor=(0.0, 0.0)):
        """About the face's point (U, V) + h anchor (Fractions, h a Fraction; anchor floats, a point of this leaf's near
        the points asked: a Taylor expansion stretched far past its piece cancels in floats), in steps of h: (E (n, n,
        3), c (3,)), so the point h (gi, gj) from (U, V) is S0 + h (c + sum E[a][b] (gi - anchor_0)^a (gj -
        anchor_1)^b): this patch's Taylor coefficients at the anchor, and the exact difference of its value there from
        the surface's at (U, V) (S0), both over h, as floats"""
        key = (U, V, h, anchor)
        found = self._expansions.get(key)
        if found is None:
            s, t = self.local(U + h * Fraction(anchor[0]), V + h * Fraction(anchor[1]))
            # The derivatives' terms in floats, about a point inside the piece (s, t in it: nothing cancels, and each
            # is a few steps' worth); only the value there exactly, its difference from S0 being a few steps of h, as
            # small as 2^-1000 of the face (all of it exactly, the 4100 shifts a floor build of quadico asked for took
            # 0.7 s)
            T = shifted(self.Ql, float(s), float(t))
            n = len(T)
            step = self.sign / self.fsize
            # h^(a + b - 1) step^(a + b), as step times (h step)^(a + b - 1): h step, the lattice's step in the piece's
            # own parameters, is a number of ordinary size, where step's powers alone overflowed (a piece 2^-130 of
            # its face across, a quartic's eighth power: every deep lattice by a border, whose pieces are its steps'
            # size, raised OverflowError)
            ratio = float(h * self.sign / self.size)
            k = np.array([[step * ratio ** (a + b - 1) if a + b else 0.0 for b in range(n)] for a in range(n)])
            E = np.array(T) * k[:, :, None]
            value = self.value(s, t)
            c = np.array([float((value[i] - S0[i]) / h) for i in range(3)])
            found = (E, c)
            if len(self._expansions) > 256:
                self._expansions.clear()
            self._expansions[key] = found
        return found

    def value(self, s, t):
        """The patch's point at exact (s, t) of its own (Fractions), three Fractions: in whole numbers over one
        denominator (as Fractions, a floor build's few thousand took most of its time)"""
        if self._ints is None:
            D = 1
            for row in self.Q:
                for cell in row:
                    for c in cell:
                        D = D * c.denominator // math.gcd(D, c.denominator)
            self._ints = (D, [[[int(c * D) for c in cell] for cell in row] for row in self.Q])
        D, Q = self._ints
        n = len(Q)
        ps, qs, pt, qt = s.numerator, s.denominator, t.numerator, t.denominator
        sp = [ps ** a * qs ** (n - 1 - a) for a in range(n)]
        tp = [pt ** b * qt ** (n - 1 - b) for b in range(n)]
        total = [0, 0, 0]
        for a in range(n):
            for b in range(n):
                c = Q[a][b]
                k = sp[a] * tp[b]
                total[0] += k * c[0]
                total[1] += k * c[1]
                total[2] += k * c[2]
        under = D * qs ** (n - 1) * qt ** (n - 1)
        return [Fraction(x, under) for x in total]


class CornerLeaf:
    """A float-only stand-in for a leaf (SubdivisionChart._corner_leaf()): where it is and its float coefficients"""

    def __init__(self, fu0, fv0, fsize, sign, Qf):
        self.fu0, self.fv0, self.fsize, self.sign, self.Qf = fu0, fv0, fsize, sign, Qf
        self.Ql = Qf.tolist()


# ----- node shapes: the float trees' splits as matrices -----

class Shape:
    """A tree's node as a pattern: its local mesh's faces (loops) and its vertices' valences, the vertices numbered
    from the node's own face (face 0, its corners 0, 1, ...: canonical()), the same for every node alike, anywhere in
    any mesh and at any depth. What a node's points make is linear in them, so a child's points are the shape's matrix
    for it times the node's (child()), and its corners' limit points likewise (limits()), each worked out once from a
    split of unit vectors (the shape as a mesh whose vertex k is the k-th unit vector). By a border between quads and
    triangles every node is a split: the UV sphere's float trees made 38,000 nodes in a start and a flight, each a split
    of its own (a millisecond of Python each, 12 s of its start), of 126 shapes. control: its regular patch's control
    points (vertex numbers), or None if it isn't one."""

    def __init__(self, mesh_class, loops, valence, boundary=()):
        self.mesh_class, self.loops, self.valence, self.boundary = mesh_class, loops, valence, boundary
        self.n = len(valence)
        self._unit = None
        self._children = None
        self._limits = None
        self.control = self.unit().control(0)

    def unit(self):
        """The shape as a mesh of unit vectors"""
        if self._unit is None:
            n = self.n
            eye = {x: tuple(1.0 if y == x else 0.0 for y in range(n)) for x in range(n)}
            self._unit = self.mesh_class(eye, dict(enumerate(self.valence)), list(self.loops), 1.0, self.boundary)
        return self._unit

    def child(self, c):
        """Child c of the node: (its Shape, the matrix (its n, this n) taking this node's points to its), or (None,
        None) if it can't be worked out from the node's rings"""
        if self._children is None:
            finer, children = self.unit().subdivide()
            found = []
            for index in children[0]:
                if index is None:
                    found.append((None, None))
                    continue
                mesh, t = finer.around(index, 2)
                shape, order = shape_of(mesh, t)
                found.append((shape, np.array([mesh.points[x] for x in order])))
            # (Whole, at once: a worker thread may be asking too)
            self._children = found
        return self._children[c]

    def limits(self):
        """The node's corners' limit masks (its corners, n): a corner's limit point is its row times the points"""
        if self._limits is None:
            unit = self.unit()
            self._limits = np.array([unit.limit(x) for x in self.loops[0]])
        return self._limits


_shapes = {}         # (mesh class, canonical()'s key) -> Shape


def canonical(mesh, t):
    """Mesh's vertices numbered from its face t: t's corners first, in its loop's order, then each face's new corners as
    the faces are reached across edges, a face's loop read from its corner numbered least. (key: the faces as loops of
    those numbers, the vertices' valences and the open edges (the mesh's boundary), the same for any mesh alike; the
    vertices in that order)"""
    loops = mesh.loops
    edge_faces = {}
    for g, loop in enumerate(loops):
        n = len(loop)
        for i in range(n):
            a, b = loop[i], loop[(i + 1) % n]
            edge_faces.setdefault((min(a, b), max(a, b)), []).append(g)
    number, order = {}, []
    seen, queue, out = {t}, [t], []
    for g in queue:
        loop = loops[g]
        n = len(loop)
        first = 0 if g == t else min(range(n), key=lambda i: number.get(loop[i], math.inf))
        turned = [loop[(first + i) % n] for i in range(n)]
        for x in turned:
            if x not in number:
                number[x] = len(order)
                order.append(x)
        out.append(tuple(number[x] for x in turned))
        for i in range(n):
            a, b = turned[i], turned[(i + 1) % n]
            for h in edge_faces[(min(a, b), max(a, b))]:
                if h not in seen:
                    seen.add(h)
                    queue.append(h)
    if len(queue) != len(loops):
        raise ExtraordinaryError("a node's faces aren't joined by their edges")
    boundary = tuple(sorted((min(number[a], number[b]), max(number[a], number[b])) for a, b in mesh.boundary))
    return (tuple(out), tuple(mesh.valence[x] for x in order), boundary), order


def shape_of(mesh, t):
    """The Shape of mesh about its face t (made the first time), and mesh's vertices in its order"""
    key, order = canonical(mesh, t)
    found = _shapes.get((type(mesh), key))
    if found is None:
        found = _shapes[(type(mesh), key)] = Shape(type(mesh), *key)
    return found, order


def side_map(P, k, G, j, number):
    """The affine map from a face's parameters to those of the face beyond its side k (corners P, in its loop's order),
    whose side j that is (its corners G): ((L, a 2 x 2), t), q = L p + t, as numbers (float, or Fraction: exact for
    corners at whole numbers). The two joined as the regular lattice joins them: a quad's neighbor the square beside it,
    a triangle's the triangle completing a parallelogram; a triangle beside a quad stands on its side, as far out as
    the quad's other side is in."""
    P = [tuple(number(c) for c in corner) for corner in P]
    G = [tuple(number(c) for c in corner) for corner in G]
    m, n = len(P), len(G)
    a, b = P[k], P[(k + 1) % m]
    half = number(1) / 2
    if m == 4:
        out = (a[0] - P[k - 1][0], a[1] - P[k - 1][1])
    else:
        out = ((a[0] + b[0]) * half - P[(k + 2) % m][0], (a[1] + b[1]) * half - P[(k + 2) % m][1])
    far = ((a[0] + out[0], a[1] + out[1]) if n == 4 else
           ((a[0] + b[0]) * half + out[0], (a[1] + b[1]) * half + out[1]))
    # Columns: the side back (a - b) and out to the far corner (far - b), here; their images there
    f00, f10, f01, f11 = a[0] - b[0], a[1] - b[1], far[0] - b[0], far[1] - b[1]
    g0, g1, g2 = G[j], G[(j + 1) % n], G[(j + 2) % n]
    h00, h10, h01, h11 = g1[0] - g0[0], g1[1] - g0[1], g2[0] - g0[0], g2[1] - g0[1]
    det = f00 * f11 - f01 * f10
    i00, i01, i10, i11 = f11 / det, -f01 / det, -f10 / det, f00 / det
    L = ((h00 * i00 + h01 * i10, h00 * i01 + h01 * i11), (h10 * i00 + h11 * i10, h10 * i01 + h11 * i11))
    t = (g0[0] - L[0][0] * b[0] - L[0][1] * b[1], g0[1] - L[1][0] * b[0] - L[1][1] * b[1])
    return L, t


# ----- the chart -----

class SubdivisionChart:
    """A closed mesh's limit surface by its schemes (see the module's docstring): schemes_by_sides, each face's by how
    many sides it has; mesh_class, the local meshes its trees split"""

    name = None
    schemes_by_sides = {}
    mesh_class = None
    # Whether its surface has pieces no split makes regular (along the borders between quads and triangles, Stam and
    # Loop's): its exact points only where a split puts a vertex (halves, quarters...), its float points below
    # BORDER_DEPTH from the corners of the piece they're in
    border = False
    # A lattice cell isn't its corners' blend: the floor splits cells on the surface itself (surface.refine_smooth())
    flat_cells = False

    def __init__(self, top):
        by_sides = self.schemes_by_sides
        self.sides = top.sides
        names = " and ".join(f"{k}" for k in sorted(by_sides))
        if any(int(k) not in by_sides for k in np.unique(top.sides).tolist()):
            raise ExtraordinaryError(f"a {self.name} chart needs faces of {names} sides only")
        if any(len(sides) > 2 for sides in top.edge_sides):
            raise ExtraordinaryError(f"a {self.name} chart needs a manifold mesh (no edge of three faces or more)")
        # Its open edges (an eye hole's rim, say): there the surface ends in its rim's cubic B-spline. No piece along one
        # is ever regular, so as by a border between quads and triangles, float points come from the pieces' corners
        # past a depth and exact ones only where a split puts a vertex (border).
        self._boundary = frozenset(top.edges[e] for e, sides in enumerate(top.edge_sides) if len(sides) == 1)
        rim = {}
        for a, b in self._boundary:
            rim[a] = rim.get(a, 0) + 1
            rim[b] = rim.get(b, 0) + 1
        if any(count != 2 for count in rim.values()):
            raise ExtraordinaryError(f"a {self.name} chart needs a boundary of simple loops (a vertex on it on two of "
                                     f"its edges)")
        if self._boundary:
            self.border = True
        loops = top._corners
        F = len(loops)
        self.schemes = [by_sides[k] for k in top.sides.tolist()]
        used = sorted(set(self.schemes), key=lambda scheme: scheme.sides)
        self.order = max(scheme.order for scheme in used)
        self.patches = {scheme: padded(scheme.patch, self.order) for scheme in used}
        self._patch_floats = {scheme: (np.array(M, dtype=float), D) for scheme, (M, D) in self.patches.items()}
        self._loops = loops
        self._faces_of = [{f for f, _s in vf} for vf in top.vertex_faces]
        self._valence = [len(vf) for vf in top.vertex_faces]
        self._sides_table(loops)

        def regular_for(scheme):
            # Its kind's valence, among faces of its kind only (on a mixed mesh, a vertex of four faces some of which
            # are triangles isn't a quad's regular corner)
            return lambda x: (self._valence[x] == scheme.valence and x not in rim
                              and all(len(loops[g]) == scheme.sides for g in self._faces_of[x]))

        regular = {scheme: regular_for(scheme) for scheme in used}
        # For off_corners(): each face's corners' places in its parameters (a triangle's last twice), how far off each
        # a normal is taken (0: a regular corner), and its middle
        self._corner_uv = np.array([(list(scheme.corners) + [scheme.corners[-1]])[:4] for scheme in self.schemes], dtype=float)
        self._middle_uv = np.array([scheme.middle for scheme in self.schemes], dtype=float)
        self._corner_nudge = np.zeros((F, 4))
        for f, loop in enumerate(loops):
            scheme = self.schemes[f]
            self._corner_nudge[f, :len(loop)] = [0.0 if regular[scheme](x) else CROWDED_NUDGE if self._valence[x] > scheme.valence
                                                 and x not in rim else CORNER_NUDGE for x in loop]
        self.controls = [self.schemes[f].control(loops, self._faces_of, f, regular[self.schemes[f]]) for f in range(F)]
        self.regular = np.array([c is not None for c in self.controls])
        self.E, self.ints = exact_ints(top.positions)
        self.denominators = {scheme: self.patches[scheme][1] << self.E for scheme in used}
        # A regular face's coefficients of u^a v^b: in floats for all of them at once, and exactly (whole numbers over
        # D 2^E, exact_of()) only for the faces asked exactly, a few near the drone (all 720 of the torus's in Python's
        # ints took 0.2 s a load: the float32 coordinates as ints over a common 2^E are too big for int64)
        self._exact = {}
        n = self.order
        self.C = np.zeros((F, n, n, 3))
        for scheme in used:
            rows = [f for f in range(F) if self.regular[f] and self.schemes[f] is scheme]
            if rows:
                M, D = self.patches[scheme]
                self.C[rows] = np.einsum("abi,fik->fabk", np.array(M, dtype=float) / D,
                                         top.positions[np.array([self.controls[f] for f in rows])])
        self._C = self.C.tolist()
        self._expansions = {}
        # An irregular face's tree: (exactly, face, path of children) -> (its local mesh, the node's face there), and
        # its leaves: one in Fractions for the exact questions, one in floats for the float ones (in Fractions,
        # sampling the bounds of quadico's 60 faces took 4.5 s)
        self._positions = top.positions
        self._nodes, self._leaves = {}, {}
        self._float_nodes, self._float_roots = {}, {}
        self._corner_points = None
        self._edge_sizes = None      # edge_sizes(), once asked for
        self._float_leaf_of = {}
        self._corners = {}
        self._splits = {}
        self._corner_leaves = {}
        self._frames = {}
        self._rough = {}
        # A ball round each face: its control points' (a regular face's; an irregular one's, its faces' within one
        # ring: its children's control points all lie in their hull)
        self.middle = np.zeros((F, 3))
        self.radius = np.zeros(F)
        for f in range(F):
            if self.regular[f]:
                control = top.positions[self.controls[f]]
            else:
                ring = {x for c in loops[f] for g in self._faces_of[c] for x in loops[g]}
                control = top.positions[sorted(ring)]
            self.middle[f] = control.mean(axis=0)
            self.radius[f] = float(np.linalg.norm(control - self.middle[f], axis=1).max())
        # How far a step of 1 in u or v moves the point, at least and at most: the Jacobian's singular values on a grid
        # over the face, give or take what it can change between the grid's points (a regular face's derivatives'
        # bounds times the way there; an irregular one's sampled a little in from its corners, half again either way)
        grid = np.linspace(0.0, 1.0, SIGMA_GRID)
        square = tuple(g.ravel() for g in np.meshgrid(grid, grid, indexing="ij"))
        G = len(square[0])
        ju, jv = np.zeros((F, G, 3)), np.zeros((F, G, 3))
        inner = {}
        # An irregular face's samples: by a border between quads and triangles, fewer and further in (its pieces along
        # the border are never regular: the 81 a face, 1/64 in, walked 15000 pieces down the UV sphere's, 5 s)
        coarse = np.linspace(0.0, 1.0, SIGMA_GRID // 2 + 1) if self.border else grid
        coarse = tuple(np.resize(g.ravel(), G) for g in np.meshgrid(coarse, coarse, indexing="ij"))
        for scheme in used:
            u, v = scheme.sample(*square)
            inner[scheme] = scheme.sample(*coarse, inset=1.0 / 16.0 if self.border else 1.0 / 64.0)
            rows = [f for f in range(F) if self.regular[f] and self.schemes[f] is scheme]
            # (Every regular face at every grid point at once: indexing C by face for each point copied it 81 times)
            if rows:
                ju[rows] = np.einsum("ga,gb,fabk->fgk", powers(u, 1, n), powers(v, 0, n), self.C[rows])
                jv[rows] = np.einsum("ga,gb,fabk->fgk", powers(u, 0, n), powers(v, 1, n), self.C[rows])
        # (The same walks give the search patches, _search_patches(), their points fitted)
        search = self.C.copy()
        for f in np.flatnonzero(~self.regular).tolist():
            u, v = inner[self.schemes[f]]
            P, ju[f], jv[f] = self._evaluate(np.full(G, f), u, v, orders=((0, 0), (1, 0), (0, 1)))
            terms = self._terms(f)
            fit = np.linalg.lstsq(np.stack([u ** a * v ** b for a, b in terms], axis=1), P, rcond=None)[0]
            search[f] = 0.0
            for (a, b), row in zip(terms, fit):
                search[f, a, b] = row
        self._search = search
        a, b, c = np.sum(ju * ju, -1), np.sum(ju * jv, -1), np.sum(jv * jv, -1)
        mean, gap = (a + c) / 2.0, np.sqrt(((a - c) / 2.0) ** 2 + b * b)
        small, big_ = np.sqrt(np.maximum(mean - gap, 0.0)), np.sqrt(mean + gap)
        uu, uv, vv = (self.bound(d) for d in ((2, 0), (1, 1), (0, 2)))
        slack = (np.maximum(uu, vv) + uv) * math.sqrt(0.5) / (SIGMA_GRID - 1)
        # (A regular face's bound is rigorous, unless its bend between the grid's points outweighs its least stretch:
        # then it's 0, which says nothing, and it takes an irregular face's estimate. On Suzanne a third of the faces,
        # small and much bent, had 0, and a level's reach in cells came out infinite.)
        rigorous = small.min(axis=1) - slack
        self.sigma_lo = np.where(self.regular & (rigorous > 0.0), rigorous, 0.5 * small.min(axis=1))
        self.sigma_hi = np.where(self.regular, big_.max(axis=1) + slack, 1.5 * big_.max(axis=1))
        # A regular face's ball from its own points on the grid, and the most a point between them can be from one
        # (a step's way times sigma_hi): its control points' hull spans the faces round it too, three times as wide,
        # and the search for the face below Newton-stepped over 54 faces where the flat torus had 14
        for scheme in used:
            rows = [f for f in range(F) if self.regular[f] and self.schemes[f] is scheme]
            if rows:
                u, v = scheme.sample(*square)
                P = np.einsum("ga,gb,fabk->fgk", powers(u, 0, n), powers(v, 0, n), self.C[rows])
                middle = P.mean(axis=1)
                gap = self.sigma_hi[rows] * math.sqrt(2.0) / (SIGMA_GRID - 1)
                self.middle[rows] = middle
                self.radius[rows] = np.linalg.norm(P - middle[:, None], axis=2).max(axis=1) + gap
        # How much each face bends at most: its mixed derivative's length (twist), and the longest of its second
        # derivatives (bend): a regular one's bounds, an irregular one's sampled away from its corners, half again
        self._twist = uv.copy()
        self._bend = np.maximum(np.maximum(uu, vv), uv)
        for f in np.flatnonzero(~self.regular).tolist():
            seconds = self._evaluate(np.full(G, f), *inner[self.schemes[f]], orders=((1, 1), (2, 0), (0, 2)))
            lengths = [float(np.linalg.norm(d, axis=1).max()) for d in seconds]
            self._twist[f] = 1.5 * lengths[0]
            self._bend[f] = 1.5 * max(lengths)

    def exact_of(self, f):
        """Regular face f's coefficients of u^a v^b, exactly: whole numbers over D 2^E (denominator), C[a][b][k]"""
        found = self._exact.get(f)
        if found is None:
            found = self._exact[f] = patch_ints(self.patches[self.schemes[f]], [self.ints[x] for x in self.controls[f]],
                                                1)[1]
        return found

    def _denominator(self, f):
        """Regular face f's coefficients' denominator, D 2^E"""
        return self.denominators[self.schemes[f]]

    def bound(self, derivative):
        """Each regular face's bound on |d^(a+b) S / du^a dv^b| over it (a, b = derivative): its derivative's
        coefficients' lengths summed (u^i v^j is at most 1 on the face); 0 for an irregular one"""
        da, db = derivative
        total = np.zeros(len(self.C))
        for i in range(da, self.order):
            for j in range(db, self.order):
                k = math.perm(i, da) * math.perm(j, db)
                total += k * np.linalg.norm(self.C[:, i, j], axis=1)
        return total

    # ----- an irregular face's tree -----

    def _root(self, f, exact):
        """Face f's tree's root: the faces within two rings of it as a local mesh (Fractions or floats), and f's index
        there"""
        keep = {f}
        for _ in range(2):
            keep |= {g for h in keep for x in self._loops[h] for g in self._faces_of[x]}
        order = sorted(keep)
        loops = [tuple(self._loops[g]) for g in order]
        used = {x for loop in loops for x in loop}
        points = ({x: self.ints[x] for x in used} if exact else {x: tuple(self._positions[x].tolist()) for x in used})
        mesh = self.mesh_class(points, {x: self._valence[x] for x in used}, loops, 1 << self.E if exact else 1.0,
                               {e for e in self._boundary if e[0] in used and e[1] in used})
        return mesh, order.index(f)

    def _node(self, f, path, exact=True):
        """The node of irregular face f's exact tree down path (children's indices): (its local mesh, its face there),
        in Fractions; in floats, _float_node()'s"""
        if not exact:
            return self._float_node(f, path)
        found = self._nodes.get((exact, f, path))
        if found is None:
            if not path:
                found = self._root(f, exact)
            else:
                mesh, t = self._node(f, path[:-1], exact)
                # The parent's split, kept for its other children (each split it again: quadico's first floor did
                # every subdivision four times)
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

    def _float_node(self, f, path):
        """The node of irregular face f's float tree down path: (its Shape, its points (n, 3) in the shape's order), a
        child's its parent's points times the shape's matrix for it. The roots are kept apart, never let go of (the
        smooth Penrose torus's 1990 faces' trees filled the cache, and every clearing rebuilt their roots: 12,500 roots
        in its start)."""
        if not path:
            found = self._float_roots.get(f)
            if found is None:
                mesh, t = self._root(f, exact=False)
                shape, order = shape_of(mesh, t)
                found = self._float_roots[f] = (shape, np.array([mesh.points[x] for x in order]))
            return found
        found = self._float_nodes.get((f, path))
        if found is None:
            shape, P = self._float_node(f, path[:-1])
            child, W = shape.child(path[-1])
            if child is None:
                raise ExtraordinaryError(f"face {f}: a child down {path} can't be worked out from its rings")
            found = (child, W @ P)
            if len(self._float_nodes) > NODES_KEPT:
                # The float tree's nodes let go of now and then (by a border every float point walks its own levels),
                # and their leaves: the exact ones stay, what an exact point costs being far more, and the corner
                # leaves, one a face's corner
                self._float_nodes.clear()
                # (A snapshot of the keys: a worker thread's walk adds to them meanwhile)
                for k in [k for k in list(self._leaves) if not k[0]]:
                    self._leaves.pop(k, None)
                self._float_leaf_of.clear()
                self._rough.clear()
            self._float_nodes[(f, path)] = found
        return found

    def _place(self, f, path, exact=True):
        """Where the node down face f's path is in its parameters: (u0, v0, size, sign), Fractions, or floats (exactly:
        halves, quarters, ... to 53 levels; in Fractions, a float tree's leaves' places were most of what was left of
        the UV sphere's start)"""
        u0 = v0 = Fraction(0) if exact else 0.0
        size, sign = Fraction(1) if exact else 1.0, 1
        for c in path:
            (ou, ov), turn = self.schemes[f].children[c]
            u0 += sign * size * ou / 2
            v0 += sign * size * ov / 2
            size /= 2
            sign *= turn
        return u0, v0, size, sign

    def _leaf(self, f, path, exact=True):
        """Face f's leaf down path (a regular node), or None if the node there isn't regular"""
        key = (exact, f, path)
        if key in self._leaves:
            return self._leaves[key]
        leaf = None
        if exact:
            mesh, t = self._node(f, path)
            control = mesh.control(t)
            if control is not None:
                P = [mesh.points[x] for x in control]
                leaf = Leaf(*self._place(f, path), ints=patch_ints(self.patches[self.schemes[f]], P, mesh.scale))
        else:
            shape, P = self._float_node(f, path)
            if shape.control is not None:
                M, D = self._patch_floats[self.schemes[f]]
                leaf = Leaf(*self._place(f, path, exact=False), Q=np.einsum("abi,ik->abk", M, P[shape.control]) / D)
        self._leaves[key] = leaf
        return leaf

    def _extraordinary_corner(self, f, path):
        """Which corner of the node down path is an extraordinary vertex, if one is (the child there is the one that
        isn't regular), and the vertex's limit point (kept: every exact point of the face walks past it)"""
        key = (f, path)
        found = self._corners.get(key)
        if found is None:
            mesh, t = self._node(f, path)
            found = next(((k, mesh.limit(x)) for k, x in enumerate(mesh.loops[t])
                          if mesh.valence[x] != self.schemes[f].valence), (None, None))
            self._corners[key] = found
        return found

    def _exact_at(self, f, u, v, numbers=FRACTIONS):
        """Face f's point at exact (u, v) (Fractions, or Q(sqrt 5) pairs with numbers=Q5): (the leaf it's in or None,
        its point if it's an extraordinary vertex's)"""
        less, add, sub, mul, number = numbers
        scheme = self.schemes[f]
        half = number(Fraction(1, 2))
        same = lambda a, b: not less(a, b) and not less(b, a)
        path = ()
        s, t = u, v
        if not scheme.holds(s, t, less, add, number):
            # Off the face, the walk toward an extraordinary corner beside it never ends (a triangle's point past its
            # long side split on with ever bigger numbers, 14 GB)
            raise ExtraordinaryError(f"face {f}: ({float(u) if not isinstance(u, tuple) else u}, "
                                     f"{float(v) if not isinstance(v, tuple) else v}) is off it")
        deepest = EXACT_DEPTH
        if self.border:
            # A split puts a vertex at every multiple of its pieces' size: a point a whole number over 2^k is a corner
            # by k levels down or in a regular piece before; any other may walk along a border for ever (one over
            # 3^n may need some n log2 3 levels by an extraordinary corner, which its denominator's bits allow)
            deepest = max(den_bits(u), den_bits(v)) + 64
        while len(path) < deepest:
            leaf = self._leaf(f, path)
            if leaf is not None:
                return leaf, None
            if self.border:
                for k, (cs, ct) in enumerate(scheme.corners):
                    if same(s, number(cs)) and same(t, number(ct)):
                        return None, self._corner_point(f, path, k)
            else:
                corner, point = self._extraordinary_corner(f, path)
                if corner is not None:
                    cs, ct = scheme.corners[corner]
                    if same(s, number(cs)) and same(t, number(ct)):
                        return None, point
            c = scheme.child_exactly(s, t, less, add, half)
            (ou, ov), turn = scheme.children[c]
            s, t = mul(sub(mul(s, 2), number(ou)), turn), mul(sub(mul(t, 2), number(ov)), turn)
            path += (c,)
        if self.border:
            raise ExtraordinaryError(f"face {f}: no exact point at ({u}, {v}): near a border between quads and "
                                     f"triangles, only at halves, quarters, ... of the face")
        raise ExtraordinaryError(f"face {f}: no leaf within {EXACT_DEPTH} levels")

    def _corner_point(self, f, path, k):
        """The limit point of corner k of the node down face f's path, exactly"""
        key = (f, path, k)
        found = self._corners.get(key)
        if found is None:
            mesh, t = self._node(f, path)
            found = self._corners[key] = mesh.limit(mesh.loops[t][k])
        return found

    def _leaf_near(self, f, U, V, h):
        """The leaf face f's point (U, V) is in, or at an extraordinary vertex, the one half a step h in toward the
        face's middle (its patch stands in for the vertex's expansion, which there isn't)"""
        leaf, _point = self._exact_at(f, U, V)
        if leaf is None:
            mu, mv = self.schemes[f].middle
            inward = lambda x, m: x + h / 2 if x < m else x - h / 2
            leaf, _point = self._exact_at(f, inward(U, mu), inward(V, mv))
        return leaf

    def _exact_point(self, f, u, v):
        """Face f's point at exact (u, v) (Fractions): three Fractions (past its sides, the surface's there:
        relocated_exactly())"""
        core = self.core()
        if core is not None and getattr(core, "exact_point", None) is not None:
            # (Rust: the same walk and the same whole numbers, without the lock)
            u, v = Fraction(u), Fraction(v)
            return tuple(Fraction(a, b) for a, b in self._core_ask(core, lambda: core.exact_point(int(f), u, v)))
        f, u, v = self.relocated_exactly(f, u, v)
        if self.regular[f]:
            C = self.exact_of(f)
            n = self.order
            up, vp = [u ** k for k in range(n)], [v ** k for k in range(n)]
            return tuple(sum(C[a][b][k] * up[a] * vp[b] for a in range(n) for b in range(n)) / self._denominator(f)
                         for k in range(3))
        leaf, point = self._exact_at(f, u, v)
        if leaf is None:
            return point
        return tuple(leaf.value(*leaf.local(u, v)))

    def _float_leaf(self, f, u, v, depth=None, corner=None):
        """Irregular face f's leaf for float (u, v), walked down in plain floats, a point at a time (kept: a walk's or
        a search's Newton steps clipped to a face's corner ask for the same one again and again, there 40 levels down)"""
        depth = self._depth(depth)
        key = (f, u, v, depth, corner)
        leaf = self._float_leaf_of.get(key)
        if leaf is not None:
            return leaf
        scheme = self.schemes[f]
        if (u, v) in scheme.corners:
            leaf = self._corner_leaf(f, scheme.corners.index((u, v)))
            if leaf is not None:
                self._float_leaf_of[key] = leaf
                return leaf
        path, s, t = (), u, v
        while True:
            leaf = self._leaf(f, path, exact=False)
            if leaf is not None:
                break
            if corner is not None and self.border and (s, t) in scheme.corners:
                # Right at a corner of this piece (a lattice point of level 2^k is one k levels down): its rough piece
                # corner levels further toward it, its value there the corner's limit point (a point of a lattice by a
                # border walked BORDER_DEPTH levels of its own: 30 s a level's 3D parts by the UV sphere's pole)
                k = scheme.corners.index((s, t))
                extra, least = corner if isinstance(corner, tuple) else (corner, 0)
                for _ in range(max(extra, least - len(path))):
                    if self._leaf(f, path, exact=False) is not None:
                        break
                    path += (k,)
                leaf = self._leaf(f, path, exact=False) or self._rough_leaf(f, path)
                break
            c = scheme.child(s, t)
            if len(path) >= depth and self._leaf(f, path + (c,), exact=False) is None:
                # Right by an extraordinary corner: the patch beside it (by a border: the piece's corners')
                leaf = self._fallback(f, path, c)
                break
            (ou, ov), turn = scheme.children[c]
            path, s, t = path + (c,), turn * (2.0 * s - ou), turn * (2.0 * t - ov)
        if len(self._float_leaf_of) > 4096:
            self._float_leaf_of.clear()
        self._float_leaf_of[key] = leaf
        return leaf

    def _depth(self, depth=None):
        """How deep a float walk goes before its fallback (_fallback()): FLOAT_DEPTH, or by a border (where pieces are
        never regular, and every level is a split) as deep as asked: BORDER_DEPTH by default (a point of its own),
        COARSE_DEPTH for points in bulk (the coarse levels' lattices, the bounds' samples: within some 1e-6 of the face;
        every point walked 24 levels each, seconds a start on the UV sphere), NORMAL_DEPTH for normals, and for the
        drone's height as deep as its level's steps need"""
        if not self.border:
            return FLOAT_DEPTH
        return BORDER_DEPTH if depth is None else depth

    def _fallback(self, f, path, c):
        """The float leaf for points in child c of the node down path, which isn't regular, that deep down"""
        if self.border:
            return self._rough_leaf(f, path + (c,))
        return self._stand_in(f, path, c)

    def _rough_leaf(self, f, path):
        """The node down path as its corners' limit points' blend (bilinear, or a triangle's linear), floats: by a
        border, BORDER_DEPTH levels down, within its size squared times the curvature of the surface"""
        found = self._rough.get((f, path))
        if found is None:
            shape, P = self._float_node(f, path)
            c = shape.limits() @ P
            Q = np.zeros((self.order, self.order, 3))
            Q[0, 0], Q[1, 0] = c[0], c[1] - c[0]
            if len(c) == 4:
                Q[0, 1], Q[1, 1] = c[3] - c[0], c[0] - c[1] + c[2] - c[3]
            else:
                Q[0, 1] = c[2] - c[0]
            u0, v0, size, sign = self._place(f, path, exact=False)
            if len(self._rough) > 65536:
                self._rough.clear()
            found = self._rough[(f, path)] = CornerLeaf(float(u0), float(v0), float(size), sign, Q)
        return found

    def _stand_in(self, f, path, c):
        """The float leaf standing in for child c of the node down path, which isn't regular (right by an extraordinary
        corner): the scheme's across(c), or any regular child"""
        order = [self.schemes[f].across(c)] + [d for d in range(len(self.schemes[f].children)) if d != c]
        for d in order:
            leaf = self._leaf(f, path + (d,), exact=False)
            if leaf is not None:
                return leaf
        raise ExtraordinaryError(f"face {f}: no regular child down {path}")

    def _corner_leaf(self, f, corner):
        """For a float point right at irregular face f's corner (its index in the loop), if it's an extraordinary
        vertex: the leaf beside it CORNER_DEPTH levels toward it, moved to meet the vertex's limit point there (its
        value there exact to a float, its derivatives a stand-in: the surface has none there in u and v); else None. A
        lattice's corner points walked FLOAT_DEPTH levels down each (quadico's first floor: 6250 nodes, 5 s)."""
        found = self._corner_leaves.get((f, corner))
        if found is None:
            shape, P = self._float_node(f, ())
            if shape.unit().kind_regular(corner, self.sides[f]):
                found = False
            else:
                limit = shape.limits()[corner] @ P
                path = ()
                while len(path) < CORNER_DEPTH and self._leaf(f, path, exact=False) is None:
                    path += (corner,)
                leaf = self._leaf(f, path, exact=False)
                if leaf is None:
                    leaf = (self._rough_leaf(f, path) if self.border else self._stand_in(f, path[:-1], corner))
                cu, cv = self.schemes[f].corners[corner]
                s, t = leaf.sign * (cu - leaf.fu0) / leaf.fsize, leaf.sign * (cv - leaf.fv0) / leaf.fsize
                n = self.order
                at = np.einsum("a,b,abk->k", powers(np.array(s), 0, n), powers(np.array(t), 0, n), leaf.Qf)
                Qf = leaf.Qf.copy()
                Qf[0, 0] += np.array(limit, dtype=float) - at
                found = CornerLeaf(leaf.fu0, leaf.fv0, leaf.fsize, leaf.sign, Qf)
            self._corner_leaves[(f, corner)] = found
        return found or None

    def _point_frame(self, f, u, v, depth=None, corner=None):
        """Face f's point at float (u, v) and its Jacobian's columns, in plain floats (a point at a time: through numpy's
        arrays, a walk's Newton steps on an irregular face took 3.4 ms a call)"""
        if self.regular[f]:
            Q, s, t, scale = self._C[f], u, v, 1.0
        else:
            leaf = self._float_leaf(f, u, v, depth, corner)
            scale = leaf.sign / leaf.fsize
            Q, s, t = leaf.Ql, (u - leaf.fu0) * scale, (v - leaf.fv0) * scale
        n = self.order
        sp, tp = [1.0], [1.0]
        for _ in range(n - 1):
            sp.append(sp[-1] * s)
            tp.append(tp[-1] * t)
        dsp = [0.0] + [k * sp[k - 1] for k in range(1, n)]
        dtp = [0.0] + [k * tp[k - 1] for k in range(1, n)]
        x = [0.0, 0.0, 0.0]
        a = [0.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0]
        for i in range(n):
            Qi = Q[i]
            for j in range(n):
                c = Qi[j]
                k, ka, kb = sp[i] * tp[j], dsp[i] * tp[j], sp[i] * dtp[j]
                for m in range(3):
                    x[m] += k * c[m]
                    a[m] += ka * c[m]
                    b[m] += kb * c[m]
        return x, [scale * c for c in a], [scale * c for c in b]

    def _float_leaves(self, f, u, v, depth=COARSE_DEPTH):
        """Irregular face f's leaves for float (u, v) (1-D arrays): [(leaf, rows)], a point nearer an extraordinary
        corner than 2^-FLOAT_DEPTH in the leaf beside it"""
        if len(u) <= FEW_POINTS:
            # A point or two (a search's Newton steps, one point a face): walked down in plain floats, as numpy's
            # arrays a level at a time took 0.15 ms a point, 40 levels down at a corner (a point below on quadico took
            # 50-120 ms, the drone's few a tick held it in the teens of frames a second)
            groups = {}
            for row, (x, y) in enumerate(zip(u.tolist(), v.tolist())):
                leaf = self._float_leaf(f, x, y, depth)
                groups.setdefault(id(leaf), (leaf, []))[1].append(row)
            return [(leaf, np.array(rows)) for leaf, rows in groups.values()]
        scheme = self.schemes[f]
        out = []
        rest = np.ones(len(u), dtype=bool)
        for c, (cu, cv) in enumerate(scheme.corners):
            # Right at an extraordinary corner: its corner leaf (_corner_leaf()), not FLOAT_DEPTH levels down
            rows = np.flatnonzero((u == cu) & (v == cv))
            if len(rows):
                leaf = self._corner_leaf(f, c)
                if leaf is not None:
                    out.append((leaf, rows))
                    rest[rows] = False
        rows = np.flatnonzero(rest)
        stack = [((), rows, u[rows], v[rows])]
        while stack:
            path, rows, s, t = stack.pop()
            if not len(rows):
                continue
            leaf = self._leaf(f, path, exact=False)
            if leaf is not None:
                out.append((leaf, rows))
                continue
            which = scheme.children_of(s - 0.5, t - 0.5, 0.5)
            for c, ((ou, ov), turn) in enumerate(scheme.children):
                mask = which == c
                sub = rows[mask]
                if len(path) >= self._depth(depth) and self._leaf(f, path + (c,), exact=False) is None:
                    # Right by an extraordinary corner: the patch beside it (by a border: the piece's corners')
                    out.append((self._fallback(f, path, c), sub))
                    continue
                stack.append((path + (c,), sub, turn * (2.0 * s[mask] - ou), turn * (2.0 * t[mask] - ov)))
        return out

    def _step_leaves(self, f, U, V, h, gi, gj):
        """Irregular face f's leaves for the points (U, V) + h (gi, gj) (exact base and step, small float offsets in
        steps, 1-D): [(leaf or None, rows)], None for the points at an extraordinary vertex (to within h / 8). A node's
        own parameters of them are A + B (gi, gj), A exact, B its step: which child each is in, from the middle's
        place in steps, (1/2 - A) / B, exactly, then against the floats."""
        scheme = self.schemes[f]
        out = []
        stack = [((), np.arange(len(gi)), U, V, h)]
        while stack:
            path, rows, As, At, B = stack.pop()
            if not len(rows):
                continue
            leaf = self._leaf(f, path)
            if leaf is not None:
                out.append((leaf, rows))
                continue
            # Each one's own parameters' distance from the node's middle, in steps: (s - 1/2) / |B|, the middle's
            # place in steps exactly first
            sign = 1.0 if B > 0 else -1.0
            ds = sign * (gi[rows] - float((Fraction(1, 2) - As) / B))
            dt = sign * (gj[rows] - float((Fraction(1, 2) - At) / B))
            which = scheme.children_of(ds, dt, float(Fraction(1, 2) / abs(B)))
            for c, ((ou, ov), turn) in enumerate(scheme.children):
                sub = rows[which == c]
                # (A child under h / 8: |B|, h over the node's size, past 4)
                if abs(B) > 4 and self._leaf(f, path + (c,)) is None:
                    out.append((None, sub))
                    continue
                stack.append((path + (c,), sub, turn * (2 * As - ou), turn * (2 * At - ov), 2 * turn * B))
        return out

    def _offsets(self, f, U, V, h, gi, gj):
        """Irregular face f's points (U, V) + h (gi, gj) from its point at (U, V), over h (see Leaf.expansion())"""
        shape = np.shape(gi)
        gi, gj = np.asarray(gi, dtype=float).ravel(), np.asarray(gj, dtype=float).ravel()
        out = np.zeros((len(gi), 3))
        if h >= FLOAT_STEP:
            # On a level whose steps floats hold (2^-24 of the face and more: some 1e-8 of a step), every point in
            # floats, all at once, from the float tree (by a border, its pieces as deep as the steps need, their
            # corners' limit points at the lattice's own points), none from the exact tree: its walks to the regular
            # pieces took 10 s of a level's 3D parts by the UV sphere's pole, and on the smooth Penrose torus, nearly
            # every face irregular, 8,000 exact splits for its first floor
            depth = min(BORDER_DEPTH, max(COARSE_DEPTH, h.denominator.bit_length() // 2 + 12))
            step, u, v = float(h), float(U), float(V)
            points = self._evaluate(np.full(len(gi), f), u + step * gi, v + step * gj, depth=depth)
            base = self._evaluate(np.array([f]), np.array([u]), np.array([v]), depth=depth)[0]
            return ((points - base) / step).reshape(shape + (3,))
        core = self.core()
        if core is not None and getattr(core, "offsets", None) is not None and len(gi):
            # (Rust: the walk down the exact tree, each leaf's expansion and the exact differences, without the lock
            # or Python's Fractions: a seventh of a smooth Penrose map's tick, and a tenth of its worker threads' work)
            U, V, h = Fraction(U), Fraction(V), Fraction(h)
            found = self._core_ask(core, lambda: core.offsets(int(f), U, V, h, _floats(gi), _floats(gj)))
            return np.frombuffer(found, dtype=np.float64).reshape(shape + (3,))
        S0 = self._exact_point(f, U, V)
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

    def _evaluate(self, f, u, v, du=0, dv=0, orders=None, depth=COARSE_DEPTH):
        """Faces f's points at float (u, v), or their derivatives d^(du + dv) / du^du dv^dv (in f's own parameters, at
        most second ones); with orders, a list of them, one for each (du, dv) there, from one walk down the irregular
        faces' trees. A point past its face's sides is the surface's beyond them (relocated())."""
        f, u, v = np.broadcast_arrays(np.asarray(f), np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        wanted = orders or ((du, dv),)
        shape = f.shape
        f, u, v = f.ravel(), u.ravel(), v.ravel()
        core = self.core()
        if core is not None:
            outs = [out.reshape(shape + (3,)) for out in self._core_evaluate(core, f, u, v, wanted, depth)]
            return outs if orders else outs[0]
        if not self._past_sides(f, u, v).any():
            outs = [out.reshape(shape + (3,)) for out in self._evaluate_at(f, u, v, wanted, depth)]
            return outs if orders else outs[0]
        g, u, v, L = self.relocated(f, u, v)
        degrees = {a + b for a, b in wanted}
        if max(degrees) > 2:
            raise ValueError("subdivision: derivatives past a face's sides only to the second")
        needed = sorted({(a, d - a) for d in degrees for a in range(d + 1)})
        found = dict(zip(needed, self._evaluate_at(g, u, v, needed, depth)))
        # In f's parameters: d/dp_i = sum_x d/dq_x L[x][i] (q the new face's parameters, L = dq/dp)
        first = lambda i: L[:, 0, i, None] * found[(1, 0)] + L[:, 1, i, None] * found[(0, 1)]
        second = {(0, 0): found.get((2, 0)), (1, 1): found.get((0, 2)), (0, 1): found.get((1, 1)),
                  (1, 0): found.get((1, 1))}
        outs = []
        for a, b in wanted:
            if a + b == 0:
                out = found[(0, 0)]
            elif a + b == 1:
                out = first(0 if a else 1)
            else:
                i, j = (0, 0) if a == 2 else (1, 1) if b == 2 else (0, 1)
                out = sum(L[:, x, i, None] * L[:, y, j, None] * second[(x, y)] for x in (0, 1) for y in (0, 1))
            outs.append(out.reshape(shape + (3,)))
        return outs if orders else outs[0]

    # ----- by the core -----

    def core(self):
        """The core's chart for float points (Rust: every face's patch, the sides' table, and the irregular faces'
        trees as they're asked for), made once from these arrays; None without a core that has it"""
        found = self.__dict__.get("_core", False)
        if found is False:
            found = None
            make = getattr(_native, "SmoothChart", None)
            if make is not None:
                schemes = {scheme.sides: self._patch_floats[scheme] for scheme in set(self.schemes)}
                patch_of = lambda sides: None if sides not in schemes else (_floats(schemes[sides][0]), float(schemes[sides][1]))
                whole = lambda a: np.ascontiguousarray(a, dtype=np.int64)
                try:
                    found = make(self.order, bool(self.border), whole(self.sides), np.ascontiguousarray(self.regular, dtype=bool).view(np.uint8),
                                 _floats(self.C), _floats(self._side_normal), _floats(self._side_point), whole(self._beyond),
                                 _floats(self._beyond_map), whole(self._beyond_side), patch_of(4), patch_of(3))
                except (TypeError, ValueError):
                    found = None
            # The trees' shapes as the core numbers them (kept here, so they stay), and one giver at a time
            self._core_shapes, self._core_shape_of, self._core_lock = [], {}, threading.Lock()
            self._core = found
        return found

    def _core_evaluate(self, core, f, u, v, wanted, depth):
        """_evaluate() of points (1-D arrays) by the core: a list, one (t, 3) for each (du, dv) in wanted. What the
        core's trees lack it asks for by name, a root, a shape's child or its limit masks (the splits of meshes that
        make them stay here: Shape), and is asked again: a few dozen shapes a mesh, and from then on every point is
        its own, without Python's lock. A build's float points were half of what a smooth map's builds did."""
        faces, u, v = np.ascontiguousarray(f, dtype=np.int64), _floats(u), _floats(v)
        wanted = [(int(a), int(b)) for a, b in wanted]
        depth = None if depth is None else int(depth)
        if not len(faces):
            return [np.zeros((0, 3)) for _ in wanted]
        try:
            while True:
                found = core.evaluate(faces, u, v, wanted, depth)
                if type(found) is not tuple:
                    return [np.frombuffer(out, dtype=np.float64).reshape(-1, 3) for out in found]
                self._core_give(core, found)
        except ValueError as error:
            raise ExtraordinaryError(str(error)) from None

    def _core_shape(self, core, shape):
        """A Shape's number in the core (given to it the first time)"""
        number = self._core_shape_of.get(id(shape))
        if number is None:
            number = core.add_shape(shape.n, None if shape.control is None else [int(x) for x in shape.control])
            assert number == len(self._core_shapes)
            self._core_shapes.append(shape)
            self._core_shape_of[id(shape)] = number
        return number

    def _core_give(self, core, need):
        """What the core's trees asked for: ("root", face), ("child", shape, child) or ("limits", shape)"""
        with self._core_lock:
            if need[0] == "root":
                f = need[1]
                shape, P = self._float_node(f, ())
                unit = shape.unit()
                regular = [bool(unit.kind_regular(corner, self.sides[f])) for corner in range(int(self.sides[f]))]
                core.set_root(f, self._core_shape(core, shape), _floats(P), regular)
            elif need[0] == "child":
                child, W = self._core_shapes[need[1]].child(need[2])
                if child is None:
                    core.set_child(need[1], need[2])
                else:
                    core.set_child(need[1], need[2], self._core_shape(core, child), _floats(W))
            elif need[0] == "limits":
                core.set_limits(need[1], _floats(self._core_shapes[need[1]].limits()))
            elif need[0] == "patch":
                # A regular face's patch exactly: whole numbers over D 2^E
                f = need[1]
                core.set_patch(f, self._denominator(f), [c for row in self.exact_of(f) for cell in row for c in cell])
            elif need[0] == "node":
                # What's at a node of a face's exact tree: a regular piece's patch (whole numbers over one), or a split
                # (and without borders, its corner that's an extraordinary vertex, with the vertex's limit point)
                f, path = need[1], tuple(need[2])
                leaf = self._leaf(f, path)
                if leaf is not None:
                    D, C = leaf._ints
                    core.set_node(f, need[2], D, [c for row in C for cell in row for c in cell])
                else:
                    corner, point = (None, None) if self.border else self._extraordinary_corner(f, path)
                    if corner is None:
                        core.set_node(f, need[2])
                    else:
                        core.set_node(f, need[2], corner=corner, point=point)
            else:
                f, path, k = need[1], tuple(need[2]), need[3]
                core.set_corner(f, need[2], k, self._corner_point(f, path, k))

    def _core_ask(self, core, ask):
        """An exact question of the core's chart (ask()), what its trees lack given as it's named and the question
        asked again; its answer"""
        try:
            while True:
                found = ask()
                if not (type(found) is tuple and found and type(found[0]) is str):
                    return found
                self._core_give(core, found)
        except ValueError as error:
            raise ExtraordinaryError(str(error)) from None

    def exact_offset(self, f, u, v, anchor, n, rho):
        """Face f's point at exact (u, v) (Q(sqrt 5) pairs) from the anchor's (surface.World.anchor_point(): (whole
        numbers, their denominator)), times rho 2^n, as three floats, by the core (Rust: the walk down the face's
        tree, the patch's polynomial in Q(sqrt 5) and each difference rounded once, without Python's lock or its
        Fractions: placing its faces' roots was 85% of a smooth Penrose map's giant frames); None without it"""
        core = self.core()
        if core is None or getattr(core, "exact_offset", None) is None:
            return None
        return np.array(self._core_ask(core, lambda: core.exact_offset(int(f), as_q(u), as_q(v), anchor, int(n), float(rho))))

    def _evaluate_at(self, f, u, v, wanted, depth):
        """_evaluate() of points in their faces (1-D arrays): a list, one for each (du, dv) in wanted"""
        n = self.order
        irregular = ~self.regular[f]
        outs = []
        for a, b in wanted:
            out = np.zeros(np.shape(u) + (3,))
            if not irregular.all():
                rows = ~irregular
                out[rows] = poly_at(u[rows], v[rows], a, b, self.C[f[rows]])
            outs.append(out)
        if irregular.any():
            fl, ul, vl = f[irregular], u[irregular], v[irregular]
            # Each point's leaf (an index into leaves), then all of them at once: a search's points, one a face, one
            # at a time through numpy cost more than walking down to them
            if len(fl) <= FEW_LEAVES:
                leaves = [self._float_leaf(g, x, y, depth) for g, x, y in zip(fl.tolist(), ul.tolist(), vl.tolist())]
                index = np.arange(len(fl))
            else:
                leaves, index = [], np.zeros(len(fl), dtype=int)
                for g in np.unique(fl).tolist():
                    rows = np.flatnonzero(fl == g)
                    for leaf, sub in self._float_leaves(g, ul[rows], vl[rows], depth):
                        index[rows[sub]] = len(leaves)
                        leaves.append(leaf)
            Q = np.array([leaf.Qf for leaf in leaves])[index]
            u0, v0, size, sign = (np.array([getattr(leaf, name) for leaf in leaves])[index]
                                  for name in ("fu0", "fv0", "fsize", "sign"))
            step = sign / size
            s, t = (ul - u0) * step, (vl - v0) * step
            for (a, b), out in zip(wanted, outs):
                out[irregular] = (step ** (a + b))[:, None] * poly_at(s, t, a, b, Q)
        return outs

    # ----- past a face's sides -----

    def _sides_table(self, loops):
        """For relocated(): each face's sides (padded to 4) as half-planes of its parameters (outward normal, a point),
        the face beyond each, which side of its that is, and the affine map from this face's parameters to that one's
        (side_map())"""
        F = len(loops)
        self._side_normal = np.zeros((F, 4, 2))
        self._side_point = np.zeros((F, 4, 2))
        self._beyond = np.full((F, 4), -1)
        self._beyond_side = np.full((F, 4), -1)
        self._beyond_map = np.zeros((F, 4, 2, 3))
        self._beyond_exact = {}
        place = {}
        for f, loop in enumerate(loops):
            for k in range(len(loop)):
                place[(loop[k], loop[(k + 1) % len(loop)])] = (f, k)
        for f, loop in enumerate(loops):
            m = len(loop)
            P = self.schemes[f].corners
            for k in range(m):
                a, b = P[k], P[(k + 1) % m]
                self._side_normal[f, k] = (b[1] - a[1], a[0] - b[0])
                self._side_point[f, k] = a
                found = place.get((loop[(k + 1) % m], loop[k]))
                if found is None:
                    continue
                g, j = found
                L, t = side_map(P, k, self.schemes[g].corners, j, float)
                self._beyond[f, k], self._beyond_side[f, k] = g, j
                self._beyond_map[f, k] = [[L[0][0], L[0][1], t[0]], [L[1][0], L[1][1], t[1]]]

    def relocated_exactly(self, f, u, v, numbers=FRACTIONS):
        """relocated() exactly (numbers: FRACTIONS, or Q5 for Q(sqrt 5) pairs): face f's point (u, v) as (the face it's
        on, its u and v there). The float points past a face's sides are the surface beyond them, so an exact one placed
        on the face's own patch carried on past them (a face's root tile's, which floats step from) was a face off."""
        less, add, sub, mul, number = numbers
        zero = number(0)
        for _ in range(MOST_HOPS + 1):
            scheme = self.schemes[f]
            if scheme.holds(u, v, less, add, number):
                return f, u, v
            P = [tuple(Fraction(c) for c in corner) for corner in scheme.corners]
            m = len(P)
            mu, mv = sum(c[0] for c in P) / m, sum(c[1] for c in P) / m
            best = None
            for k in range(m):
                a, b = P[k], P[(k + 1) % m]
                n0, n1 = b[1] - a[1], a[0] - b[0]
                toward = add(mul(sub(u, number(mu)), n0), mul(sub(v, number(mv)), n1))
                if not less(zero, toward):
                    continue
                room = n0 * (a[0] - mu) + n1 * (a[1] - mv)
                # The side the way from the middle leaves by: the least room / toward
                if best is None or less(mul(best[1], room), mul(toward, best[2])):
                    best = (k, toward, room)
            k = best[0]
            g = int(self._beyond[f, k])
            if g < 0:
                break
            L, t = self._beyond_exact.get((f, k)) or self._beyond_exact.setdefault(
                (f, k), side_map(scheme.corners, k, self.schemes[g].corners, int(self._beyond_side[f, k]), Fraction))
            u, v = (add(add(mul(u, L[0][0]), mul(v, L[0][1])), number(t[0])),
                    add(add(mul(u, L[1][0]), mul(v, L[1][1])), number(t[1])))
            f = g
        raise ExtraordinaryError(f"face {f}: a point past its sides too far to place")

    def _past_sides(self, f, u, v):
        """Whether each point (u, v) is past one of its face f's sides (1-D arrays)"""
        d = np.stack([u, v], axis=1)[:, None, :] - self._side_point[f]
        return np.any(np.einsum("tkc,tkc->tk", self._side_normal[f], d) > PAST_SIDE, axis=1)

    def relocated(self, f, u, v):
        """Points (u, v) of faces f (1-D) past their faces' sides as points of the faces they're on: across the side the
        way from the face's middle leaves by, into the face beyond, again while past one of its (MOST_HOPS at most,
        then clamped to the face). (faces, u, v, L): L (t, 2, 2) the map from the first face's parameters to the
        last's, d(u', v') / d(u, v). A face's own patch carried on past its sides (a polynomial's, or deep down a
        piece's next to a corner) was thousands of times the map's size a face off (the smooth Darb torus's girih
        tiles straddle their faces)."""
        f, u, v = f.copy(), u.astype(float).copy(), v.astype(float).copy()
        L = np.broadcast_to(np.eye(2), (len(f), 2, 2)).copy()
        for _ in range(MOST_HOPS):
            rows = np.flatnonzero(self._past_sides(f, u, v))
            if not len(rows):
                return f, u, v, L
            fr = f[rows]
            middle = np.array([self.schemes[g].middle for g in fr.tolist()])
            p = np.stack([u[rows], v[rows]], axis=1)
            normal = self._side_normal[fr]
            toward = np.einsum("tkc,tc->tk", normal, p - middle)
            room = np.einsum("tkc,tkc->tk", normal, self._side_point[fr] - middle[:, None, :])
            with np.errstate(divide="ignore", invalid="ignore"):
                at = np.where(toward > 0.0, room / toward, np.inf)
            k = np.argmin(at, axis=1)
            g = self._beyond[fr, k]
            joined = g >= 0
            rows, k, g, p = rows[joined], k[joined], g[joined], p[joined]
            A = self._beyond_map[f[rows], k]
            q = np.einsum("tij,tj->ti", A[:, :, :2], p) + A[:, :, 2]
            L[rows] = A[:, :, :2] @ L[rows]
            f[rows], u[rows], v[rows] = g, q[:, 0], q[:, 1]
        for row in np.flatnonzero(self._past_sides(f, u, v)).tolist():
            u[row], v[row] = self.schemes[f[row]].clamp(u[row], v[row])
        return f, u, v, L

    def points(self, f, u, v, search=False):
        """Faces f's points at (u, v); search: only good enough to tell which face is nearest (by a border: from the
        search patches, _search_patches())"""
        if search:
            return self._search_evaluate(f, u, v, ((0, 0),))[0]
        return self._evaluate(f, u, v)

    def jacobians(self, f, u, v, search=False):
        if search:
            return tuple(self._search_evaluate(f, u, v, ((1, 0), (0, 1))))
        return tuple(self._evaluate(f, u, v, orders=((1, 0), (0, 1))))

    def _search_evaluate(self, f, u, v, orders):
        f, u, v = np.broadcast_arrays(np.asarray(f), np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        n = self.order
        C = self._search_patches()
        return [poly_at(u, v, a, b, C[f]) for a, b in orders]

    def _search_patches(self):
        """Every face's polynomial for the search for the face below (surface.Below.global_face()): a regular face's
        own, an irregular one's fitted to its points on a grid (least squares, its kind's monomials), good to some
        1e-3: the search's Newton steps over some 40 faces walked their trees every tick, 2 ms a point below on the
        smooth icosphere, 12 ms by the UV sphere's borders"""
        return self._search

    def _terms(self, f):
        """Face f's kind's monomials (a, b): a quad's bicubic ones, a triangle's quartic"""
        n = self.order
        if self.sides[f] == 4:
            return [(a, b) for a in range(4) for b in range(4)]
        return [(a, b) for a in range(n) for b in range(n) if a + b <= 4]

    def off_corners(self, f, u, v):
        """Points (u, v) of faces f (1-D arrays) as a normal is taken at them: one nearer an extraordinary corner of
        its face than CORNER_NUDGE of the face is moved that far from it, the way it lies from the corner (right at
        it: toward the face's middle). At an extraordinary vertex the surface has a normal, but its derivatives in a
        face's own (u, v) are 0 or unbounded there (its pieces shrink toward it by another factor than a half): where
        they go to 0, a corner of three, their cross product was rounding and the normal any direction at all (the
        smooth Darb torus's corners of three triangles: posts there leant by up to 65 degrees, another way each
        build). A little way off, the normal is the vertex's to within the surface's bend over that way; nearer still
        at a corner of more faces than regular (CROWDED_NUDGE)."""
        u, v = np.array(u, dtype=float), np.array(v, dtype=float)
        nudge = self._corner_nudge[f]
        if not nudge.any():
            return u, v
        corners = self._corner_uv[f]
        way = np.stack([u, v], axis=1)[:, None, :] - corners                    # (t, 4, 2)
        reach = np.abs(way).max(axis=2)
        near = reach < nudge
        rows = np.flatnonzero(near.any(axis=1))
        if len(rows):
            k = np.argmax(near[rows], axis=1)
            corner, off, length, share = corners[rows, k], way[rows, k], reach[rows, k], nudge[rows, k]
            inward = self._middle_uv[f[rows]] - corner
            inward /= np.abs(inward).max(axis=1, keepdims=True)
            toward = np.where((length > 0.0)[:, None], off / np.maximum(length, 1e-300)[:, None], inward)
            u[rows], v[rows] = (corner + share[:, None] * toward).T
        return u, v

    def normal(self, f, u, v):
        """Face f's unit normal at (u, v), as three floats (in plain floats, a point at a time)"""
        if self._corner_nudge[f].any():
            (u,), (v,) = self.off_corners(np.array([f]), np.array([u]), np.array([v]))
        _x, (ax, ay, az), (bx, by, bz) = self._point_frame(f, float(u), float(v), NORMAL_DEPTH, CORNER_NORMAL)
        nx, ny, nz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
        length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1e-300
        return nx / length, ny / length, nz / length

    def normals(self, f, u, v):
        """normal() of many points at once (arrays): their faces' derivatives in one batch (_evaluate()), as normal() has
        them (a mesh without borders: there every float leaf is the same either way; by a border, a point at a time)"""
        f, u, v = np.asarray(f), np.asarray(u, dtype=float), np.asarray(v, dtype=float)
        if self.border:
            return np.array([self.normal(g, x, y) for g, x, y in zip(f.tolist(), u.tolist(), v.tolist())]).reshape(
                f.shape + (3,))
        shape = f.shape
        u, v = self.off_corners(f.ravel(), u.ravel(), v.ravel())
        du, dv = self._evaluate(f.ravel(), u, v, orders=((1, 0), (0, 1)), depth=NORMAL_DEPTH)
        du, dv = du.reshape(shape + (3,)), dv.reshape(shape + (3,))
        n = np.cross(du, dv)
        length = np.linalg.norm(n, axis=-1, keepdims=True)
        return n / np.where(length > 0.0, length, 1e-300)

    def expansion(self, f, u, v):
        """Regular face f's Taylor coefficients about (u, v) (exact numbers), as floats (n, n, 3): S(u + du, v + dv) =
        sum T[a][b] du^a dv^b, its constant term aside (the point itself is placed exactly elsewhere). In floats, from
        the float coefficients at (u, v) rounded: the derivatives there are good to a float's precision relative to
        themselves, however deep (a point 2^-53 off has the same derivatives to as many bits); worked out exactly, with
        numbers growing with every level, a smooth map's point below cost some 2 ms more a tick. Kept for the points
        asked again (a walk asks at the same base point tick after tick)."""
        key = (f, u, v)
        found = self._expansions.get(key)
        if found is None:
            found = np.array(shifted(self._C[f], float(u), float(v)))
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
        # tree each
        key = (f, I, J, M)
        found = self._frames.get(key)
        if found is None:
            h = Fraction(1, M)
            leaf = self._leaf_near(f, u, v, h)
            E = leaf.expansion(u, v, h, self._exact_point(f, u, v))[0] if leaf is not None else \
                self._difference_expansion(f, u, v, h)[0]
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
        n = self.order
        steps = np.array([[inv_m ** (a + b - 1) if a + b else 0.0 for b in range(n)] for a in range(n)])
        # (In one einsum: term by term, a smooth torus's floor build spent 0.2 s here)
        return poly_at(np.asarray(gi, dtype=float), np.asarray(gj, dtype=float), 0, 0,
                       np.asarray(T * steps[:, :, None], dtype=float))

    def _steps_expansion(self, f, I, J, N, d):
        """Face f's expansion in level-N steps about its point (I, J), from the patch the point d steps further is in:
        (E, c) with the point d' steps out c + sum E[a][b] d'^a (units)"""
        if self.regular[f]:
            T = self.expansion(f, Fraction(I, N), Fraction(J, N))
            inv_n = 1 / N
            n = self.order
            E = np.zeros((n, n, 3))
            for a in range(n):
                for b in range(n):
                    if a + b:
                        E[a][b] = T[a][b] * inv_n ** (a + b - 1)
            return E, np.zeros(3)
        U, V, h = Fraction(I, N), Fraction(J, N), Fraction(1, N)
        leaves = self._step_leaves(f, U, V, h, np.array([d[0]]), np.array([d[1]]))
        leaf = leaves[0][0] or self._leaf_near(f, U, V, h)
        if leaf is None:
            return self._difference_expansion(f, U, V, h)
        return leaf.expansion(U, V, h, self._exact_point(f, U, V))

    def _difference_expansion(self, f, U, V, h):
        """By a border between quads and triangles (no polynomial piece there): an expansion in steps of h about the
        point (U, V), (E, 0), its terms up to the second from the exact points a step or two round it (a quadratic fitted
        to them), on the face (in toward it at a side)"""
        key = ("difference", f, U, V, h)
        found = self._frames.get(key)
        if found is not None:
            return found
        scheme = self.schemes[f]
        stencils = [[(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]]
        for d1 in ((1, 0), (-1, 0), (-1, 1), (1, -1)):
            for d2 in ((0, 1), (0, -1), (-1, 1), (1, -1)):
                if d1 != d2 and (d1[0] * d2[1] - d1[1] * d2[0]) != 0:
                    stencils.append([(i * d1[0] + j * d2[0], i * d1[1] + j * d2[1]) for i in range(3) for j in range(3)
                                     if i + j <= 2])
        # Steps of h, or halves of it till a stencil fits on the face (a coarse level's step is the face's side)
        delta, points = h, None
        while points is None:
            holds = lambda i, j: scheme.holds(U + i * delta, V + j * delta, operator.lt, operator.add, Fraction)
            points = next((st for st in stencils if all(holds(i, j) for i, j in st)), None)
            if points is None:
                delta /= 2
        S0 = self._exact_point(f, U, V)
        rows, values = [], []
        for i, j in points:
            if (i, j) == (0, 0):
                continue
            p = self._exact_point(f, U + i * delta, V + j * delta)
            gi, gj = float(i * delta / h), float(j * delta / h)
            rows.append([gi, gj, gi * gi, gi * gj, gj * gj])
            values.append([float((p[k] - S0[k]) / h) for k in range(3)])
        fit = np.linalg.lstsq(np.array(rows, dtype=float), np.array(values), rcond=None)[0]
        E = np.zeros((self.order, self.order, 3))
        E[1, 0], E[0, 1], E[2, 0], E[1, 1], E[0, 2] = fit
        found = (E, np.zeros(3))
        if len(self._frames) > 4096:
            self._frames.clear()
        self._frames[key] = found
        return found

    def _sides(self, f, I, J, N):
        """The face's sides in level-N steps from its point (I, J): each as (a point on it, its direction, the range of
        the parameter along it), in the order surface's walks number them (a quad's v = 0, u = 1, v = 1, u = 0; a
        triangle's v = 0, u + v = 1, u = 0), and the box: lo_u, hi_u, lo_v, hi_v (a triangle's hi_u, hi_v its long
        side's room, N - I - J)"""
        lo_u, lo_v = -big(I), -big(J)
        if self.sides[f] == 4:
            hi_u, hi_v = big(N - I), big(N - J)
            sides = (((0.0, lo_v), (1.0, 0.0), lo_u, hi_u), ((hi_u, 0.0), (0.0, 1.0), lo_v, hi_v),
                     ((0.0, hi_v), (1.0, 0.0), lo_u, hi_u), ((lo_u, 0.0), (0.0, 1.0), lo_v, hi_v))
            return sides, (lo_u, hi_u, lo_v, hi_v)
        room = big(N - I - J)
        with np.errstate(invalid="ignore", over="ignore"):
            sides = (((0.0, lo_v), (1.0, 0.0), lo_u, room - lo_v), ((0.0, room), (1.0, -1.0), lo_u, room - lo_v),
                     ((lo_u, 0.0), (0.0, 1.0), lo_v, room - lo_u))
        return sides, (lo_u, room, lo_v, room)

    def _clip_steps(self, f, d, box):
        """Offsets d (steps) onto the face"""
        lo_u, hi_u, lo_v, hi_v = box
        d = np.array([min(max(d[0], lo_u), hi_u), min(max(d[1], lo_v), hi_v)])
        if self.sides[f] == 3 and d[0] + d[1] > hi_u:
            over = d[0] + d[1] - hi_u
            d = np.array([max(d[0] - over / 2.0, lo_u), max(d[1] - over / 2.0, lo_v)])
            over = d[0] + d[1] - hi_u
            if over > 0.0:
                d = d - (np.array([over, 0.0]) if d[0] - lo_u > d[1] - lo_v else np.array([0.0, over]))
        return d

    def _past(self, f, d, box, gu, gv, ju, jv):
        """Which sides offsets d (steps) is past, how far (units): {side: how far}. gu, gv: the step's derivatives of
        half the distance squared to the target (with a side's offsets, the target nearer across it if they point
        out), or None to judge by d alone; ju, jv: units a step"""
        lo_u, hi_u, lo_v, hi_v = box
        past = {}
        if self.sides[f] == 4:
            if gu is None:
                values = ((lo_v - d[1]) * jv, (d[0] - hi_u) * ju, (d[1] - hi_v) * jv, (lo_u - d[0]) * ju)
                return {s: x for s, x in enumerate(values) if x > 0.0}
            if d[1] <= lo_v and gv > 0.0:
                past[0] = gv
            if d[0] >= hi_u and gu < 0.0:
                past[1] = -gu
            if d[1] >= hi_v and gv < 0.0:
                past[2] = -gv
            if d[0] <= lo_u and gu > 0.0:
                past[3] = gu
            return past
        if gu is None:
            values = ((lo_v - d[1]) * jv, (d[0] + d[1] - hi_u) * min(ju, jv), (lo_u - d[0]) * ju)
            return {s: x for s, x in enumerate(values) if x > 0.0}
        if d[1] <= lo_v and gv > 0.0:
            past[0] = gv
        if d[0] + d[1] >= hi_u and gu + gv < 0.0:
            past[1] = -(gu + gv)
        if d[0] <= lo_u and gu > 0.0:
            past[2] = gu
        return past

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
        sides, box = self._sides(f, I, J, N)
        free = d.copy()
        past = self._past(f, d, box, None, None, float(np.linalg.norm(E[1][0])), float(np.linalg.norm(E[0][1])))
        order = sorted(past, key=lambda s: -past[s])
        if order:
            best, best_square = None, math.inf
            for origin, direction, lo, hi in sides:
                origin, direction = np.array(origin), np.array(direction)
                if not (np.all(np.isfinite(origin)) and math.isfinite(lo) and math.isfinite(hi)):
                    continue
                along = float((free - origin) @ direction) / float(direction @ direction)
                t = side_newton(E, r - c, origin, direction, lo, hi, float(np.clip(along, lo, hi)))
                candidate = origin + t * direction
                square = float(np.sum((c + evaluate(E, candidate) - r) ** 2))
                if math.isfinite(square) and square < best_square:
                    best, best_square = candidate, square
            if best is not None:
                d = best
        return d, free, order, c + evaluate(E, d)

    def _nearest_irregular(self, f, I, J, N, r):
        """nearest() on an irregular face, whose patch is piecewise (a leaf's polynomial only holds in its piece): from
        the surface itself in the face's parameters (Newton's steps kept on the face, from the base and from its
        middle), where its levels' steps are big enough for floats to tell them apart (N < 2^40), else from the base's
        leaf on (deep down, the leaves round a point are vast next to the drone's sight), then the exact expansion of the
        leaf the point is in, in steps. Past a side: the sides the point found sits on with the surface still nearer
        across them (its gradient pointing out)."""
        scheme = self.schemes[f]
        U, V = Fraction(I, N), Fraction(J, N)
        _sides, box = self._sides(f, I, J, N)
        # By a border, its float walks as deep as the level's steps need (the rough pieces' error falls as their size
        # squared): half the level's bits and some
        depth = min(BORDER_DEPTH, max(COARSE_DEPTH, N.bit_length() // 2 + 12))
        if N < (1 << 40):
            base = np.array(self._point_frame(f, float(U), float(V), depth)[0])
            target = (base + r / N).tolist()
            best = None
            for start in ((float(U), float(V)), scheme.middle):
                u, v = scheme.clamp(*start)
                for _ in range(24):
                    X, ju, jv = self._point_frame(f, u, v, depth)
                    e = [X[n] - target[n] for n in range(3)]
                    g11, g12, g22 = (sum(p * q for p, q in zip(x_, y_)) for x_, y_ in ((ju, ju), (ju, jv), (jv, jv)))
                    c1, c2 = sum(p * q for p, q in zip(ju, e)), sum(p * q for p, q in zip(jv, e))
                    det = g11 * g22 - g12 * g12
                    if not det > 1e-300:
                        break
                    du, dv = -(g22 * c1 - g12 * c2) / det, -(g11 * c2 - g12 * c1) / det
                    nu, nv = scheme.clamp(u + du, v + dv)
                    done = not max(abs(nu - u), abs(nv - v)) > 1e-15
                    u, v = nu, nv
                    if done:
                        break
                square = sum((p - q) ** 2 for p, q in zip(self._point_frame(f, u, v, depth)[0], target))
                if best is None or square < best[0]:
                    best = (square, np.array([u, v]))
                # From the base it came out on the surface's own point, with the target over it: no need of another
                if square <= 1e-24 * (1.0 + float(np.sum(np.asarray(target) ** 2))) or \
                        (scheme.inside(u, v) and start == (float(U), float(V))):
                    break
            uv = best[1]
            d = np.array([float((Fraction(float(uv[0])) - U) * N), float((Fraction(float(uv[1])) - V) * N)])
        else:
            d = np.zeros(2)
        # Deep down, polished on the leaf it's in (again if a step crosses into the next leaf); shallower, the surface's
        # own point stands (a leaf's polynomial, stretched past its piece by a step, pulled points near an
        # extraordinary corner off)
        free = d.copy()
        for _ in range(3 if N >= (1 << 40) else 0):
            E, c = self._steps_expansion(f, I, J, N, d)
            free = newton(E, r - c, d.copy())
            moved = self._clip_steps(f, free, box)
            if np.allclose(moved, d, rtol=0.0, atol=1e-12 * (1.0 + float(np.abs(d).max()))):
                d = moved
                break
            d = moved
        if N < (1 << 40):
            # From the surface's own point (the leaf's expansion about the base, stretched to a far leaf, cancels)
            P, ju, jv = self._point_frame(f, float(uv[0]), float(uv[1]), depth)
            X = (np.array(P) - base) * N
            gu, gv = float(np.array(ju) @ (X - r)), float(np.array(jv) @ (X - r))
        else:
            # (Only deep down: worked out shallower too and not used, an exact walk and point a call)
            E, c = self._steps_expansion(f, I, J, N, d)
            X = c + evaluate(E, d)
            gu, gv = float(evaluate(E, d, 1, 0) @ (X - r)), float(evaluate(E, d, 0, 1) @ (X - r))
        past = self._past(f, d, box, gu, gv, None, None)
        order = sorted(past, key=lambda s: -past[s])
        return d, free, order, X

    def step(self, f, ur, vr, du, dv, small):
        """Faces f's points at (ur + small du, vr + small dv) from their points at (ur, vr), over small (see
        surface.BilinearChart.step()): ur and vr (t,), du and dv (t, ...), as (t, ..., 3). A regular face's from its
        patch's Taylor expansion about (ur, vr), every tile's at once (one product: a tile at a time, the motif levels
        of a smooth icosphere took 15 s to start); an irregular face's, where floats hold small (2^-24 of the face and
        more), from its points in floats, all at once, else a tile at a time from its leaves' exact expansions
        (_offsets())."""
        f = np.asarray(f)
        ur, vr = np.asarray(ur, dtype=float), np.asarray(vr, dtype=float)
        du, dv = np.asarray(du, dtype=float), np.asarray(dv, dtype=float)
        out = np.zeros(du.shape + (3,))
        if not len(f):
            return out
        core = self.core()
        if core is not None and getattr(core, "step", None) is not None and du.size:
            # (Rust: every tile's steps, whatever its face's kind, and what's left for the exact way too, without the
            # lock: the float leaves' walks and the tiles by odd corners were a fifth of a smooth Penrose map's worker
            # threads' Python)
            per = du.size // len(f)
            found = self._core_ask(core, lambda: core.step(np.ascontiguousarray(f, dtype=np.int64), _floats(ur), _floats(vr),
                                                           _floats(du).reshape(-1), _floats(dv).reshape(-1), per, float(small)))
            return np.frombuffer(found, dtype=np.float64).reshape(du.shape + (3,))
        extra = (1,) * (du.ndim - 1)
        regular = self.regular[f]
        rows = np.flatnonzero(regular)
        if len(rows):
            d_u, d_v = du[rows], dv[rows]
            out[rows] = total = taylor_steps(self.C[f[rows]], ur[rows], vr[rows], d_u, d_v, small)
            if small >= FLOAT_STEP:
                # Ends past the face's sides: the surface's there (relocated()), not the patch's carried on
                faces = np.broadcast_to(f[rows].reshape((len(rows),) + extra), d_u.shape)
                ends_u = ur[rows].reshape((len(rows),) + extra) + small * d_u
                ends_v = vr[rows].reshape((len(rows),) + extra) + small * d_v
                past = self._past_sides(faces.ravel(), ends_u.ravel(), ends_v.ravel()).reshape(d_u.shape)
                if past.any():
                    which = np.nonzero(past)
                    ends = self._evaluate(faces[which], ends_u[which], ends_v[which])
                    starts = self._evaluate(f[rows][which[0]], ur[rows][which[0]], vr[rows][which[0]])
                    total[which] = (ends - starts) / small
                    out[rows] = total
        rows = np.flatnonzero(~regular)
        if len(rows):
            if small >= FLOAT_STEP:
                # (By a border, every piece is a split: as deep as the tiles need, STEP_EXTRA levels under them. A
                # coarse level's bars over the whole UV sphere walked 10 levels split its pole's borders 12,500 times,
                # 12 s of its start, while 10 levels were too few for tiles 2^-12 of a face and finer, 0.5-2% off.)
                depth = min(BORDER_DEPTH, max(0, math.ceil(-math.log2(small))) + STEP_EXTRA)
                faces = np.broadcast_to(f[rows].reshape((len(rows),) + extra), du[rows].shape)
                ends = self._evaluate(faces, ur[rows].reshape((len(rows),) + extra) + small * du[rows],
                                      vr[rows].reshape((len(rows),) + extra) + small * dv[rows], depth=depth)
                starts = self._evaluate(f[rows], ur[rows], vr[rows], depth=depth)
                out[rows] = (ends - starts.reshape((len(rows),) + extra + (3,))) / small
            else:
                h = Fraction(float(small))
                for t in self._leaf_steps(f, rows, ur, vr, du, dv, small, out).tolist():
                    U, V = Fraction(float(ur[t])), Fraction(float(vr[t]))
                    out[t] = self._offsets(int(f[t]), U, V, h, du[t], dv[t])
        return out

    def _leaf_steps(self, f, rows, ur, vr, du, dv, small, out):
        """step() of irregular faces' tiles deeper than floats hold their points (small under FLOAT_STEP), each far
        smaller than its float leaf (a regular piece of its face, no more than 14 levels toward an extraordinary corner:
        LEAF_LEAST), from
        that piece's patch, as a regular face's tiles are from its own, all at once (taylor_steps(), in the leaf's own
        parameters), into out; returns the rest's rows, for _offsets(). Exactly, a tile at a time, the tiles deep in a
        face by a corner of 3 or 5 took a millisecond each, a second a level of a laid-out map's floor (a third of its
        faces are irregular), and a dive there outran its builds (the Tile Selected Object button's Suzanne). The
        patch's polynomial carried a tile's size past its piece is the surface's there to within the tile's size squared
        over the piece's (the pieces meet smoothly: C2)."""
        left = []
        tiles = len(rows)
        reach = small * np.maximum(np.abs(du[rows]).reshape(tiles, -1).max(axis=1),
                                   np.abs(dv[rows]).reshape(tiles, -1).max(axis=1))
        groups = {}
        for face in np.unique(f[rows]).tolist():
            mine = np.flatnonzero(f[rows] == face)
            scheme = self.schemes[face]
            for leaf, sub in self._float_leaves(face, ur[rows[mine]], vr[rows[mine]]):
                sub = mine[sub]
                if not isinstance(leaf, Leaf):
                    left.append(rows[sub])
                    continue
                q = leaf.sign / leaf.fsize
                s_, t_ = q * (ur[rows[sub]] - leaf.fu0), q * (vr[rows[sub]] - leaf.fv0)
                # In the piece (not a stand-in's, right by the corner) and far bigger than the tile
                inside = (s_ >= -LEAF_SLACK) & (t_ >= -LEAF_SLACK) & (reach[sub] <= LEAF_SHARE * leaf.fsize)
                inside &= leaf.fsize >= LEAF_LEAST
                inside &= ((s_ + t_ <= 1.0 + LEAF_SLACK) if isinstance(scheme, LoopScheme)
                           else (s_ <= 1.0 + LEAF_SLACK) & (t_ <= 1.0 + LEAF_SLACK))
                left.append(rows[sub[~inside]])
                if inside.any():
                    groups.setdefault(id(leaf), (leaf, []))[1].append((sub[inside], s_[inside], t_[inside]))
        n = self.order
        for leaf, parts in groups.values():
            sub = np.concatenate([p[0] for p in parts])
            s_, t_ = np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts])
            Q = leaf.Qf
            if len(Q) < n:
                Q = np.pad(Q, ((0, n - len(Q)), (0, n - len(Q)), (0, 0)))
            q = leaf.sign / leaf.fsize
            C = np.broadcast_to(Q, (len(sub),) + Q.shape)
            out[rows[sub]] = q * taylor_steps(C, s_, t_, du[rows[sub]], dv[rows[sub]], q * small)
        return np.concatenate(left) if left else np.zeros(0, dtype=np.int64)

    def hessians(self, f, u, v):
        """Faces f's second derivatives at (u, v): (d2/du2, d2/du dv, d2/dv2), each (..., 3)"""
        return tuple(self._evaluate(f, u, v, orders=((2, 0), (1, 1), (0, 2))))

    def corners(self):
        """Each face's corner points on the limit surface (F, 4, 3), in its loop's order (exactly, then rounded), a
        triangle's last twice: each vertex's limit point, once (from a face of its's root; a face's point at each of its
        corners walked its tree, 0.2 s of the UV sphere's start), kept"""
        if self._corner_points is None:
            limits = {}
            out = np.zeros((len(self.C), 4, 3))
            for f, loop in enumerate(self._loops):
                for k in range(4):
                    x = loop[min(k, len(loop) - 1)]
                    if x not in limits:
                        mesh, _t = self._node(f, ())
                        limits[x] = [float(c) for c in mesh.limit(x)]
                    out[f, k] = limits[x]
            out.flags.writeable = False
            self._corner_points = out
        return self._corner_points

    def limit_masks(self):
        """Each vertex's limit point as a blend of the mesh's points, its Shape's mask (from a face of its): (rows,
        columns, weights), row x's limit point being the sum of its weights times the points of its columns, the rows
        in order (each vertex's entries together; a vertex on no face has none). For layout.fit(), which moves the
        points so the limit points land where it wants them."""
        rows, columns, weights = [], [], []
        for x, faces in enumerate(self._faces_of):
            if not faces:
                continue
            f = min(faces)
            mesh, t = self._root(f, exact=False)
            shape, order = shape_of(mesh, t)
            mask = shape.limits()[list(self._loops[f]).index(x)]
            used = np.flatnonzero(mask)
            rows.extend([x] * len(used))
            columns.extend(order[i] for i in used.tolist())
            weights.extend(mask[used].tolist())
        return np.array(rows, dtype=np.int64), np.array(columns, dtype=np.int64), np.array(weights)

    def edge_sizes(self):
        """Each face's size: the geometric mean of its sides' lengths at its corner 0 (as chords). Worked out once
        (not to be written to), like the flat chart's (surface.BilinearChart.edge_sizes())."""
        if self._edge_sizes is None:
            corners = self.corners()
            sizes = np.sqrt(np.linalg.norm(corners[:, 1] - corners[:, 0], axis=1)
                            * np.linalg.norm(corners[:, 3] - corners[:, 0], axis=1))
            sizes.flags.writeable = False
            self._edge_sizes = sizes
        return self._edge_sizes

    def twist_norms(self):
        return self._twist

    def bend_norms(self):
        """How much each face bends at most: the longest of its second derivatives, a bound (over it)"""
        return self._bend

    # ----- exactly -----

    def point(self, f, I, J, M):
        """Face f's point (I, J) of a level of M cells across, exactly: (ints, d), the place being the ints over d"""
        if self.regular[f]:
            C = self.exact_of(f)
            n = self.order
            ip = [I ** a * M ** (n - 1 - a) for a in range(n)]
            jp = [J ** b * M ** (n - 1 - b) for b in range(n)]
            out = [0, 0, 0]
            for a in range(n):
                for b in range(n):
                    k = ip[a] * jp[b]
                    c = C[a][b]
                    out[0] += k * c[0]
                    out[1] += k * c[1]
                    out[2] += k * c[2]
            return tuple(out), self._denominator(f) * M ** (2 * (n - 1))
        p = self._exact_point(f, Fraction(I, M), Fraction(J, M))
        d = math.lcm(*(c.denominator for c in p))
        return tuple(c.numerator * (d // c.denominator) for c in p), d

    def point_q5(self, f, u, v):
        """Face f's point at (u, v), Q(sqrt 5) pairs, exactly: three such pairs, the mesh's units (past its sides, the
        surface's there: relocated_exactly())"""
        core = self.core()
        if core is not None and getattr(core, "point_q5", None) is not None:
            u, v = as_q(u), as_q(v)
            return [(Fraction(*a), Fraction(*b)) for a, b in self._core_ask(core, lambda: core.point_q5(int(f), u, v))]
        f, u, v = self.relocated_exactly(f, u, v, Q5)
        one = (Fraction(1), Fraction(0))
        if self.regular[f]:
            C, scale = self.exact_of(f), Fraction(1, self._denominator(f))
            s, t = u, v
        else:
            leaf, point = self._exact_at(f, u, v, Q5)
            if leaf is None:
                return [(c, Fraction(0)) for c in point]
            C, scale = leaf.Q, Fraction(1)
            s = q_scale(q_sub(as_q(u), (leaf.u0, Fraction(0))), leaf.sign / leaf.size)
            t = q_scale(q_sub(as_q(v), (leaf.v0, Fraction(0))), leaf.sign / leaf.size)
        n = len(C)
        s, t = as_q(s), as_q(t)
        up, vp = [one], [one]
        for _ in range(n - 1):
            up.append(q_mul(up[-1], s))
            vp.append(q_mul(vp[-1], t))
        out = []
        for k in range(3):
            total = (Fraction(0), Fraction(0))
            for a in range(n):
                for b in range(n):
                    if C[a][b][k]:
                        total = q_add(total, q_scale(q_mul(up[a], vp[b]), C[a][b][k]))
            out.append(q_scale(total, scale))
        return out


class CatmullClarkChart(SubdivisionChart):
    """A closed all-quad mesh's Catmull-Clark limit surface (see the module's docstring)"""
    name = "catmull_clark"
    schemes_by_sides = {4: CATMULL_CLARK}
    mesh_class = LocalMesh


class LoopChart(SubdivisionChart):
    """A closed all-triangle mesh's Loop limit surface, with Warren's weights (see the module's docstring)"""
    name = "loop"
    schemes_by_sides = {3: LOOP}
    mesh_class = LoopMesh


class QuadTriChart(SubdivisionChart):
    """A closed mesh of quads and triangles' limit surface by Stam and Loop's rules (QuadTriMesh): Catmull-Clark among
    quads, Loop's among triangles; along their borders no polynomial pieces (see SubdivisionChart.border)"""
    name = "stam_loop"
    schemes_by_sides = {3: LOOP, 4: CATMULL_CLARK}
    mesh_class = QuadTriMesh
    border = True


def sum_terms(E, gi, gj):
    """sum over a + b >= 1 of E[a][b] gi^a gj^b, for arrays of offsets (n,): (n, 3)"""
    E = np.array(E, dtype=float)
    E[0, 0] = 0.0
    n = len(E)
    return poly_at(np.asarray(gi, dtype=float), np.asarray(gj, dtype=float), 0, 0, E)


def den_bits(x):
    """How many halvings an exact number's denominator holds at most (a Q(sqrt 5) pair's: its parts')"""
    if isinstance(x, tuple):
        return max(den_bits(x[0]), den_bits(x[1]))
    return Fraction(x).denominator.bit_length()


def big(value):
    """A whole number as a float, inf past a float's range"""
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def evaluate(E, d, du=0, dv=0):
    """An expansion E (n, n, 3) at the offset d (2,), or its derivative (evaluate_here(), by the core when it has it)"""
    at = getattr(_native, "patch_evaluate", None)
    if at is None:
        return evaluate_here(E, d, du, dv)
    return np.array(at(_floats(E), len(E), float(d[0]), float(d[1]), du, dv))


def evaluate_here(E, d, du=0, dv=0):
    """An expansion E (n, n, 3) at the offset d (2,), or its derivative"""
    total = np.zeros(3)
    n = len(E)
    for a in range(du, n):
        for b in range(dv, n):
            if a + b == 0:
                continue
            k = math.perm(a, du) * math.perm(b, dv) * d[0] ** (a - du) * d[1] ** (b - dv)
            if k:
                total = total + k * E[a][b]
    return total


_shift_tables = {}


def poly_at(u, v, a, b, C):
    """Polynomials' (d/du)^a (d/dv)^b at the points (u, v) (arrays of one shape): C one patch's coefficients (n, n, 3)
    or each point's own (..., n, n, 3); (..., 3). By the core when it has it (one pass over the points, no arrays of
    powers), else poly_sum() of the points' powers."""
    n = C.shape[-3]
    at = getattr(_native, "patch_values", None)
    if at is not None and np.shape(u) == np.shape(v) and (C.ndim == 3 or C.shape[:-3] == np.shape(u)):
        out = np.frombuffer(at(_floats(C), n, _floats(u), _floats(v), a, b), dtype=np.float64)
        return out.reshape(np.shape(u) + (3,))
    return poly_sum(powers(u, a, n), powers(v, b, n), C)


def poly_sum(pu, pv, C):
    """sum over a, b of pu[..., a] pv[..., b] C[a, b, k]: polynomials' values at many points, pu and pv the points'
    powers (..., n), C one patch's coefficients (n, n, 3) or each point's own (..., n, n, 3); (..., 3). As two einsums
    of two operands (or a matrix product): one einsum of three ran a plain loop over every index, 6 times as long (13 ms
    for 50,000 points; irregular faces' float points were half a laid-out map's level builds)"""
    if C.ndim == 3:
        n = C.shape[0]
        by_b = (pu.reshape(-1, n) @ C.reshape(n, -1)).reshape(pu.shape[:-1] + C.shape[1:])
    else:
        by_b = np.einsum("...a,...abk->...bk", pu, C)
    return np.einsum("...b,...bk->...k", pv, by_b)


def taylor_steps(C, ur, vr, du, dv, small):
    """taylor_steps_here() by the core when it has it (each patch's expansion, then each offset's sum, in one pass:
    no arrays of the offsets' powers)"""
    steps = getattr(_native, "patch_taylor_steps", None)
    if steps is None or not len(C):
        return taylor_steps_here(C, ur, vr, du, dv, small)
    m = len(C)
    per = int(np.prod(du.shape[1:], dtype=np.int64))
    out = steps(_floats(C), C.shape[1], _floats(ur), _floats(vr), _floats(du.reshape(m, -1)),
                _floats(dv.reshape(m, -1)), per, float(small))
    return np.frombuffer(out, dtype=np.float64).reshape(du.shape + (3,))


def taylor_steps_here(C, ur, vr, du, dv, small):
    """Polynomial patches' points at (ur + small du, vr + small dv) from their points at (ur, vr), over small: C (t, n,
    n, 3) each patch's coefficients of u^i v^j, ur and vr (t,), du and dv (t, ...); (t, ..., 3). By the patch's Taylor
    expansion about (ur, vr), T[t, a, b] = sum over i >= a, j >= b of C[i, j] binom(i, a) ur^(i - a) binom(j, b)
    vr^(j - b) (in numpy alone: a worker thread's Python loops here kept the main thread waiting for Python's lock), as
    two matrix products, over i, then over j (as one einsum of three, numpy looped over all six indices one by one, 20
    times as long, most of a smooth map's level builds); then the sum over a, b of d_u^a d_v^b small^(a + b - 1) T[a, b]
    (not the constant term), as two matrix products too: the terms with a >= 1,
    small^(a - 1) d_u^a times small^b d_v^b, and those with a = 0, small^(b - 1) d_v^b (term by term, 16 products of
    every tile's offsets, it was half a level's build), never over small, which a level deep enough makes subnormal
    or 0."""
    m, n = len(C), C.shape[1]
    by_i = (shift_matrices(ur, n) @ np.reshape(C, (m, n, n * 3))).reshape(m, n, n, 3)
    T = (shift_matrices(vr, n) @ by_i.transpose(0, 2, 1, 3).reshape(m, n, n * 3)).reshape(m, n, n, 3).transpose(
        0, 2, 1, 3)
    flat_u, flat_v = du.reshape(m, -1), dv.reshape(m, -1)
    pu = np.zeros(flat_u.shape + (n,))
    pv = np.empty(flat_v.shape + (n,))
    pv1 = np.zeros(flat_v.shape + (n,))
    pu[..., 1] = flat_u
    pv[..., 0] = 1.0
    pv1[..., 1] = flat_v
    for j in range(1, n):
        pv[..., j] = pv[..., j - 1] * flat_v * small
        if j >= 2:
            pu[..., j] = pu[..., j - 1] * flat_u * small
            pv1[..., j] = pv1[..., j - 1] * flat_v * small
    by_a = (pv @ T.transpose(0, 2, 1, 3).reshape(m, n, n * 3)).reshape(flat_u.shape + (n, 3))
    return (np.einsum("tqa,tqac->tqc", pu, by_a) + pv1 @ T[:, 0]).reshape(du.shape + (3,))


def shift_matrices(x, n):
    """For each x (t,): the matrix (n, n) taking a polynomial's coefficients of powers of t to those of powers of
    (t - x): [a][i] = binom(i, a) x^(i - a), i >= a"""
    found = _shift_tables.get(n)
    if found is None:
        i, a = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
        found = _shift_tables[n] = (np.where(i >= a, [[math.comb(int(ii), int(aa)) if ii >= aa else 0 for ii in range(n)]
                                                      for aa in range(n)], 0), np.clip(i - a, 0, None), i >= a)
    comb, power, keep = found
    x = np.asarray(x, dtype=float)
    powers_ = x[:, None] ** np.arange(n)[None, :]                     # (t, n)
    return np.where(keep[None], comb[None] * powers_[:, power], 0.0)


def derivatives(E, d):
    """derivatives_here() by the core when it has it"""
    at = getattr(_native, "patch_derivatives", None)
    if at is None:
        return derivatives_here(E, d)
    return np.frombuffer(at(_floats(E), len(E), float(d[0]), float(d[1])), dtype=np.float64).reshape(3, 3, 3)


def derivatives_here(E, d):
    """An expansion E (n, n, 3) at the offset d: its value (less its constant term) and its derivatives up to the
    second, D[i, j] = d^(i + j) / du^i dv^j, (3, 3, 3), in one product (term by term, a smooth map's Newton steps for
    the point below took most of its tick)"""
    n = len(E)
    E = np.asarray(E, dtype=float)
    if E[0, 0].any():
        E = E.copy()
        E[0, 0] = 0.0
    return np.einsum("ia,jb,abk->ijk", _powers3(d[0], n), _powers3(d[1], n), E)


def _powers3(t, n):
    """(1, t, t^2, ...) and its first and second derivatives, (3, n)"""
    out = np.zeros((3, n))
    p = 1.0
    for k in range(n):
        out[0, k] = p
        p *= t
    for k in range(1, n):
        out[1, k] = k * out[0, k - 1]
    for k in range(2, n):
        out[2, k] = k * (k - 1) * out[0, k - 2]
    return out


def newton(E, r, d):
    """newton_here() by the core when it has it (a dozen steps of a few dozen products each: numpy's calls for them
    were most of the point below's time on a smooth map)"""
    nearest = getattr(_native, "patch_newton", None)
    if nearest is None:
        return newton_here(E, r, d)
    return np.array(nearest(_floats(E), len(E), tuple(float(x) for x in r), float(d[0]), float(d[1])))


def newton_here(E, r, d):
    """The offset d nearest r on an expansion's whole patch: Newton's steps from d (Gauss-Newton's when the full step
    wouldn't descend)"""
    for _ in range(NEWTON_STEPS):
        D = derivatives_here(E, d)
        e = D[0, 0] - r
        u, v = D[1, 0], D[0, 1]
        uu, uv, vv = D[2, 0], D[1, 1], D[0, 2]
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


def side_newton(E, r, origin, direction, lo, hi, t):
    """side_newton_here() by the core when it has it"""
    nearest = getattr(_native, "patch_side_newton", None)
    if nearest is None:
        return side_newton_here(E, r, origin, direction, lo, hi, t)
    return nearest(_floats(E), len(E), tuple(float(x) for x in r), tuple(float(x) for x in origin),
                   tuple(float(x) for x in direction), float(lo), float(hi), float(t))


def side_newton_here(E, r, origin, direction, lo, hi, t):
    """The point nearest r along a side of the face (offsets origin + t direction, t within [lo, hi]): its parameter
    t, Newton's steps from t"""
    du, dv = direction
    for _ in range(NEWTON_STEPS):
        d = origin + t * direction
        D = derivatives_here(E, d)
        e = D[0, 0] - r
        first = du * D[1, 0] + dv * D[0, 1]
        second = du * du * D[2, 0] + 2 * du * dv * D[1, 1] + dv * dv * D[0, 2]
        g, h = float(first @ e), float(first @ first + second @ e)
        step = -g / h if h > 0.0 else -g / max(float(first @ first), 1e-300)
        new = min(max(t + step, lo), hi)
        if not abs(new - t) > 1e-15 * (1.0 + abs(t)):
            return new
        t = new
    return t
