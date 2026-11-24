"""Endless surface maps: any mesh of quads and triangles, its faces the biggest tiles of a lattice, each splitting k x k
at every level (a quad into k^2 quads, a triangle into k^2 triangles; k the map's factor, 2 by default), that the drone
can dive into anywhere forever and climb back out of to see the whole object (2026-11-12: step 2 of tiling any surface,
quads; 2026-11-13, step 3: triangles, and corners where any number of faces meet; 2026-11-21, phase 2 of the unified
engines: any factor, the Inferno's 3 x 3 too; see ~/blender_tests/surface).

The faces are flat tiles, folded at the mesh's edges (sphere.py's chart is conformal instead, made for one surface). A
quad's lattice at level L is its corners' bilinear blend at u, v = I / M, J / M, M = k^L cells across: the points its
repeated k x k subdivision makes; a triangle's is the linear map's at I + J <= M, a triangular lattice, 6 around a point.
Either way a face's side splits into k at every level, so a level's points on an edge are the same from both of its
faces (a quad's and a triangle's alike), and the lattice runs on across the fold. A point of a level's lattice is a point of every finer one,
and its key (Topology.key()) names it the same way from every face it's on: one inside a face by the face, one on an
edge by the edge, a corner by its vertex. Flat faces need no corner charts: a corner where any number of faces meet (3
at a cube's, 5 at an icosahedron's, 24 at a UV sphere's pole) is a vertex of every level, with a neighbor along each of
its edges, and only a level's corners are other than 4 (or 6) around. A quad that isn't flat is its bilinear patch,
curved: the point below the drone is found on it with Newton's steps, and the floor's cells near the drone are split
until they're as flat as it (refine()).

The world is kept in two parts, like a sphere map's: n, the depth (a level-n cell is a unit), and the anchor A, a point
of the level-n lattice, Python integers that grow without limit; and everything else in map units around A's point, the
map's origin, in floats that stay within the view. Map units are the mesh's own times rho k^n.
- A dive loop by k^j: n grows by j and A's integers grow k times j times, a change of units about the map's origin, so
  the view doesn't change at all (dive()).
- A climb loop first moves the anchor to the nearest point of the k^j times coarser lattice, then n shrinks by j
  (climb()). At n = 0 the mesh's faces are the units: climb loops stop there, and the drone grows freely to see the
  whole object (at_top()).
- The anchor follows the point below the drone (recenter_move()), so the drone stays near the map's origin.
Every difference from A is worked out exactly: the mesh's corners are float32, so dyadic, and a lattice point is a
blend of them with weights I / M, so it's integers over M^2 2^E (Topology.point()). A point a thousand loops deep
is placed from A as exactly as one at the top, and rounded once.

What's drawn:
- the surface near the drone, the floor (build_floor()): the cells of the level about a quarter of the window across,
  the cells in the window (window times the drone camera's clip end), each corner from the camera, exactly, then
  rounded. Like the giant parts' window (parts.Exact): loops and recenters only move and scale its object, and it's
  built again when the view reaches past it or it's grown far wider than wanted (and with a bent quad's cells split,
  as the drone moves on). Its faces have the mesh's colors (its face "Col" attribute) and the map's material.
- the 3D parts, pillars and arches at every level (parts.py, the map's "parts" property), through the lattice here
  (MeshLattice): the parts' instanced levels from the lattice near the drone in floats, and the giant ones exactly.
The mesh itself, the map's source, isn't drawn: it's k^n times too big for float32 once the drone has dived.

The map's object (an empty at the map's origin, its axes the mesh's, scaled uniformly) holds:
- "surface": its settings (see DEFAULTS), written by the map's build script
- "surface_source": the mesh's object (its mesh, untransformed, is the surface)
- "surface_state": "n kind id i j", n and the anchor's key (Topology.key()) in hex
- "surface_version": counts the state's writes, so an undo, which puts an older one back, is noticed
"""

import math
import random
import time
from fractions import Fraction

import numpy as np

try:
    from . import background
    from . import loops
    from . import parts
    from . import schedule
    from . import sphere
    from . import tiling
    from .cyclotomic import q_add, q_mul, q_scale
except ImportError:          # tests import it by itself
    import background
    import loops
    import parts
    import schedule
    import sphere
    import tiling
    from cyclotomic import q_add, q_mul, q_scale

PROPERTY = "surface"
STATE = "surface_state"
VERSION = "surface_version"
SOURCE = "surface_source"
FLOOR_KEY = "surface_floor_of"       # on the floor object: its map's object's name

# The "surface" property's keys:
# - rho: map units per unit of the mesh at n = 0 (the mesh's own size, times rho, is the object's at the top)
# - window: how far the floor is built, in drone camera clip ends
# - floor_cells: the floor's cells are about the window over this, across (more: more, smaller cells)
# - floor_color: linear RGBA for faces without a "Col" of the mesh's own
# - rule: "square" (quads and triangles split factor x factor: this module's lattice) or a substitution rule
#   (SUBSTITUTION_RULES: a labeled mesh, the rule's strapwork drawn on the floor at every level by surface_substitution.py,
#   whose mesh lattice splits 2 x 2)
# - factor: how many ways a level's cell's sides split at the next, a whole number from 2 (the map loops by its powers)
# - endless: an endless plane's ({"seed": a whole number}), None for a map with a top: see retop()
# - chart: the shape the lattice stands on: "bilinear" (each face its corners' bilinear patch, BilinearChart) or
#   "catmull_clark" (the mesh's smooth Catmull-Clark limit surface, subdivision.CatmullClarkChart)
# The rules whose floors surface_substitution.py draws (substitution.RULES' names)
SUBSTITUTION_RULES = ("penrose", "darb_imam")
DEFAULTS = {"rho": 1.0, "window": 1.25, "floor_cells": 4.0, "floor_color": [0.04, 0.03, 0.10, 1.0], "rule": "square",
            "factor": 2, "endless": None, "chart": "bilinear"}

# A key's kinds: a corner of the mesh (its vertex), a point inside an edge (the edge, k along it from its smaller vertex
# to its bigger one), a point inside a face (the face, I and J)
VERTEX, EDGE, FACE = 0, 1, 2
# A face of k sides (4: a quad, 3: a triangle) has its corners at these (I, J), in its loop's order, at a level of M
# cells across: a quad's at (0, 0), (M, 0), (M, M), (0, M), its points the bilinear blend's at u, v = I / M, J / M; a
# triangle's at (0, 0), (M, 0), (0, M), its points those with I + J <= M, of the linear map. Its side s runs from
# corner s to corner s + 1. Inside a face a point's neighbors are these steps away: a quad's cells split into quads,
# so its lattice is square, 4 around; a triangle's into triangles, so its lattice is triangular, 6 around.
STEPS = {4: ((1, 0), (0, 1), (-1, 0), (0, -1)), 3: ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))}
# A point inside side s steps into the face by these (a quad's one, a triangle's two)
INWARD = {4: (((0, 1),), ((-1, 0),), ((0, -1),), ((1, 0),)),
          3: (((0, 1), (-1, 1)), ((-1, 0), (0, -1)), ((1, 0), (1, -1)))}

# The point below the drone (Below): Newton steps in a face at most (until they stop moving it: a flat face needs one, a
# curved quad a few), faces crossed at most
NEWTON_STEPS = 12
WALK_HOPS = 12
# Further than this many units from the anchor's point, or this share of its face's size (whichever is less), and a
# face under GLOBAL_UNITS units across, the face below is first found among them all (the source's own coordinates in float64 tell faces
# apart that shallow), else by walking from the anchor's face: over a mesh's creases (Suzanne's eyes), a walk from the
# anchor's face can end in a hollow while a face across the crease is nearer
WALK_REACH = 2.0
WALK_SHARE = 0.25
GLOBAL_UNITS = 1 << 40
# Recenter once the drone is this many units from the anchor's point, or this many of its heights over the surface
RECENTER_UNITS = 4.0
RECENTER_HEIGHTS = 2.0
# The floor: at most this many cells (the whole mesh, at the top), and at most this many from a face's first, either way
FLOOR_MAX_CELLS = 20000
FLOOR_REACH = 64
# A quad cell of the floor is split 2 x 2 while it bulges off the two flat triangles it's drawn as by more than this
# share of its distance from the camera (about a pixel), or of the drone's height nearer than that, this many times at
# most (refine()); a floor with split cells is built again once the camera has moved FLOOR_FOLLOW drone heights from
# where it was built for, or the drone's height has halved
FLOOR_FLATNESS = 1e-3
FLOOR_SPLITS = 24
FLOOR_FOLLOW = 16.0
# The grid a face's Jacobian is bounded on (Topology)
SIGMA_GRID = 9
# A point's faces' normals cancel (Topology.key_normal()) when their sum is shorter than this share of their count
NORMALS_CANCEL = 1e-6


def setting(obj, name):
    settings = obj.get(PROPERTY)
    value = settings.get(name) if settings is not None else None
    return DEFAULTS[name] if value is None else value


def side_point(k, s, t, M):
    """The point t along side s of a face of k sides (from its corner s), as (I, J) at a level of M cells across"""
    if k == 4:
        return ((t, 0), (M, t), (M - t, M), (0, M - t))[s]
    return ((t, 0), (M - t, t), (0, M - t))[s]


def corner_point(k, c, M):
    return ((0, 0), (M, 0), (M, M), (0, M))[c] if k == 4 else ((0, 0), (M, 0), (0, M))[c]


def sides_on(k, I, J, M):
    """The sides of a face of k sides its point (I, J) is on, at a level of M cells across"""
    if k == 4:
        return [s for s, on in enumerate((J == 0, I == M, J == M, I == 0)) if on]
    return [s for s, on in enumerate((J == 0, I + J == M, I == 0)) if on]


def nearest_point(k, u, v, M):
    """The point of a face of k sides' lattice of M cells across nearest (u, v) in its parameters (in cells, numbers of
    any type), (I, J): rounded, kept in the face"""
    I = min(max(round(u), 0), M)
    J = min(max(round(v), 0), M)
    while k == 3 and I + J > M:
        # Past the long side: back off along whichever was rounded up the more
        if (I - u >= J - v and I > 0) or J == 0:
            I -= 1
        else:
            J -= 1
    return I, J


def big_float(value):
    """An int of any size as a float, inf past a float's range"""
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def quotient(a, b):
    """a / b for ints of any size, correctly rounded, inf past a float's range"""
    try:
        return a / b
    except OverflowError:
        return math.inf if (a > 0) == (b > 0) else -math.inf


def power(k, e):
    """k^e as a float, for a whole number k and any int e: inf past a float's range, 0 under it"""
    try:
        return float(k) ** e
    except OverflowError:
        return math.inf


def log_k(x, k):
    """log x to base k (a power of 2's exactly, for k = 2)"""
    return math.log2(x) / math.log2(k)


# ----- exact points: three ints over a whole number -----

def d_sub(a, b):
    """a - b for exact points (ints, d): the ints over d, a whole number (a lattice point's M^2 2^E, Topology.point())"""
    (ax, ad), (bx, bd) = a, b
    if ad == bd:
        return tuple(x - y for x, y in zip(ax, bx)), ad
    g = math.gcd(ad, bd)
    sa, sb = bd // g, ad // g
    return tuple(x * sa - y * sb for x, y in zip(ax, bx)), ad * sa


def d_add(a, b):
    (ax, ad), (bx, bd) = a, b
    if ad == bd:
        return tuple(x + y for x, y in zip(ax, bx)), ad
    g = math.gcd(ad, bd)
    sa, sb = bd // g, ad // g
    return tuple(x * sa + y * sb for x, y in zip(ax, bx)), ad * sa


def d_float(a, scale=1):
    """An exact point times scale (a whole number: N, a map's units in the mesh's), as floats, each correctly rounded
    (inf past a float's range)"""
    x, d = a
    return np.array([quotient(v * scale, d) for v in x])


def d_of(values, scale=1):
    """Floats over scale (a whole number), exactly, as an exact point"""
    ratios = [float(v).as_integer_ratio() for v in values]
    k = max(den.bit_length() - 1 for _, den in ratios)
    return tuple(num << (k - (den.bit_length() - 1)) for num, den in ratios), scale << k


def d_fractions(a):
    x, d = a
    return [Fraction(v, d) for v in x]


# ----- the mesh -----

class BilinearChart:
    """The shape a mesh's faces stand for, as the lattice sees it: each face a bilinear patch through its corners (a
    triangle the plane through its three), P0 + u eu + v ev + u v w with P0 its corner 0, eu and ev its sides from
    there (to corners 1 and -1) and the twist w = P0 - P1 + P2 - P3 (0 for a triangle). As floats and exactly, as ints
    over 2^E (the float32 coordinates as they are). Everything the engine asks of the shape is asked here (phase 6 of
    the unified engines plan, 2026-11-23): its points, in bulk in floats (points(), jacobians()) and exactly (point(),
    point_q5()), how a level's points step from one placed exactly (frame(), step(): exact for a bilinear patch), its
    normals, its corners, and bounds (a ball round each face, middle and radius; sigma_lo and sigma_hi, how far a step
    of a face's parameters moves its point at least and at most; edge_sizes(); twist_norms()). A smooth chart (a
    subdivision surface's limit) answers the same questions.

    Each method does its arithmetic in the order its callers did when they did it themselves, so the floats come out
    the same to the last bit."""

    # A lattice cell's points are its corners' bilinear blend (refine() splits cells by averaging their corners)
    flat_cells = True

    def __init__(self, positions, faces, sides):
        first, second, third, last = (positions[[c[k] for c in faces]] for k in (0, 1, 2, -1))
        self.sides = sides
        self.P0 = first
        self.eu = second - first
        self.ev = last - first
        self.w = np.where((sides == 4)[:, None], first - second + third - last, 0.0)
        # A ball round each face (its corners' mean, the furthest corner): a patch lies in its corners' hull, so no point
        # of it is nearer a point than that point's distance to the middle less the radius (Below.global_face())
        self.middle = np.where((sides == 4)[:, None], (first + second + third + last) / 4.0,
                               (first + second + third) / 3.0)
        self.radius = np.max(np.stack([np.linalg.norm(corner - self.middle, axis=1)
                                       for corner in (first, second, third, last)]), axis=0)
        self._eu, self._ev, self._w = self.eu.tolist(), self.ev.tolist(), self.w.tolist()
        # Exactly: every coordinate is an int over 2^E
        ratios = [v.as_integer_ratio() for v in positions.ravel().tolist()]
        self.E = max(den.bit_length() - 1 for _, den in ratios)
        ints = [num << (self.E - (den.bit_length() - 1)) for num, den in ratios]
        vi = [tuple(ints[3 * k:3 * k + 3]) for k in range(len(positions))]
        sub = lambda p, q: tuple(x - y for x, y in zip(p, q))
        self.exact = []
        for c in faces:
            p0, p1, p2, pl = vi[c[0]], vi[c[1]], vi[c[2]], vi[c[-1]]
            twist = tuple(a - b + c_ - d for a, b, c_, d in zip(p0, p1, p2, pl)) if len(c) == 4 else (0, 0, 0)
            self.exact.append((p0, sub(p1, p0), sub(pl, p0), twist))
        # A step of 1 in u or v moves a face's point by its Jacobian's columns, eu + v w and ev + u w: their singular
        # values on a grid of SIGMA_GRID x SIGMA_GRID over the face, give or take what the Jacobian can change between
        # the grid's points (by |w| times the way to the nearest, Weyl's inequality), bound it over the face (a
        # triangle's is the same everywhere)
        grid = np.linspace(0.0, 1.0, SIGMA_GRID)
        u, v = (g.ravel() for g in np.meshgrid(grid, grid, indexing="ij"))
        ju, jv = self.jacobians(np.arange(len(faces))[:, None], u[None], v[None])
        # A 3 x 2 matrix's singular values, from its 2 x 2 Gram matrix's eigenvalues (closed form)
        a, b, c = np.sum(ju * ju, -1), np.sum(ju * jv, -1), np.sum(jv * jv, -1)
        mean, gap = (a + c) / 2.0, np.sqrt(((a - c) / 2.0) ** 2 + b * b)
        small, big = np.sqrt(np.maximum(mean - gap, 0.0)), np.sqrt(mean + gap)
        slack = np.linalg.norm(self.w, axis=1) * math.sqrt(0.5) / (SIGMA_GRID - 1)
        self.sigma_lo = np.maximum(small.min(axis=1) - slack, 0.0)
        self.sigma_hi = big.max(axis=1) + slack

    # ----- in floats -----

    def points(self, f, u, v):
        """Faces f's points at (u, v) (arrays that broadcast together; f indexes the faces), (..., 3)"""
        u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
        return self.P0[f] + u[..., None] * self.eu[f] + v[..., None] * self.ev[f] + (u * v)[..., None] * self.w[f]

    def jacobians(self, f, u, v):
        """The columns of faces f's Jacobians at (u, v): how a step of u, and of v, moves the point, ((..., 3), (..., 3))"""
        u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
        return self.eu[f] + v[..., None] * self.w[f], self.ev[f] + u[..., None] * self.w[f]

    def mixed(self, f):
        """Faces f's mixed second derivatives (d2 / du dv): a bilinear patch's twist, the same all over it"""
        return self.w[f]

    def normal(self, f, u, v):
        """Face f's unit normal at (u, v), as three floats (a point at a time: plain floats are quicker)"""
        (ex, ey, ez), (fx, fy, fz), (wx, wy, wz) = self._eu[f], self._ev[f], self._w[f]
        ax, ay, az = ex + v * wx, ey + v * wy, ez + v * wz
        bx, by, bz = fx + u * wx, fy + u * wy, fz + u * wz
        nx, ny, nz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
        length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1e-300
        return nx / length, ny / length, nz / length

    def frame(self, f, I, J, M):
        """Face f's patch about its point (I, J) of a level of M cells across: (a, b, w), so its point (I + i, J + j)
        is i a + j b + i j w / M from there (the mesh's units over M), exactly for a bilinear patch: the Jacobian's
        columns there in the level's cells, and the twist"""
        return self.eu[f] + quotient(J, M) * self.w[f], self.ev[f] + quotient(I, M) * self.w[f], self.w[f]

    def grid_offsets(self, f, I, J, M, gi, gj):
        """Face f's points (I + gi, J + gj) of a level of M cells across (gi, gj: small offsets, floats, any shape) from its
        point (I, J), in the mesh's units times M: (..., 3). I, J and M can be any size (Python ints)."""
        a, b, w = self.frame(f, I, J, M)
        inv_m = 1 / M                     # (int / int: for an M past a float's range too)
        return gi[..., None] * a + gj[..., None] * b + (gi * gj * inv_m)[..., None] * w

    def nearest(self, f, I, J, N, r):
        """Face f's point nearest r (units: the mesh's times N, from its point (I, J) of level N): (d, the offset of it
        from (I, J) in level-N steps; free, that on the face's whole patch; past, the sides free is past, the furthest
        first; X, its place from (I, J), units). See nearest_in_face() and nearest_in_triangle()."""
        a, b, w = self.frame(f, I, J, N)
        inv_n = 1 / N
        if self.sides[f] == 4:
            box = (-big_float(I), big_float(N - I), -big_float(J), big_float(N - J))
            d, free, past = nearest_in_face(a, b, w, inv_n, r, box)
        else:
            d, free, past = nearest_in_triangle(a, b, r, -big_float(I), -big_float(J), big_float(N - I - J))
        return d, free, past, d[0] * a + d[1] * b + d[0] * d[1] * w * inv_n

    def step(self, f, ur, vr, du, dv, small):
        """Faces f's points at (ur + small du, vr + small dv) from their points at (ur, vr), over small: ur and vr (t,),
        du and dv (t, ...), as (t, ..., 3). Exact for a bilinear patch: du eu + dv ev + (ur dv + vr du + small du dv) w."""
        extra = (1,) * (du.ndim - 1)
        shape = lambda a: a.reshape(a.shape[:1] + extra)
        vec = lambda a: a[f].reshape((len(f),) + extra + (3,))
        twist = shape(ur) * dv + shape(vr) * du + small * du * dv
        return du[..., None] * vec(self.eu) + dv[..., None] * vec(self.ev) + twist[..., None] * vec(self.w)

    def corners(self):
        """Each face's corner points (F, 4, 3), in its loop's order, a triangle's last twice"""
        quad = (self.sides == 4)[:, None, None]
        P0, eu, ev, w = self.P0, self.eu, self.ev, self.w
        corners = np.stack([P0, P0 + eu, P0 + eu + ev + w, P0 + ev], axis=1)
        triangle = np.stack([P0, P0 + eu, P0 + ev, P0 + ev], axis=1)
        return np.where(quad, corners, triangle)

    def edge_sizes(self):
        """Each face's size: the geometric mean of its sides' lengths at its corner 0"""
        return np.sqrt(np.linalg.norm(self.eu, axis=1) * np.linalg.norm(self.ev, axis=1))

    def twist_norms(self):
        """How much each face's patch bends at most (its mixed second derivative's length)"""
        return np.linalg.norm(self.w, axis=1)

    # ----- exactly -----

    def point(self, f, I, J, M):
        """Face f's point (I, J) of a level of M cells across, exactly: (ints, d), the place in the mesh's units being
        the ints over d. (I M eu + J M ev + I J w + M^2 P0) / (M^2 2^E)."""
        p0, eu, ev, w = self.exact[f]
        MM, IM, JM, IJ = M * M, I * M, J * M, I * J
        return tuple(MM * p + IM * x + JM * y + IJ * z for p, x, y, z in zip(p0, eu, ev, w)), MM << self.E

    def point_q5(self, f, u, v):
        """Face f's point at (u, v), Q(sqrt 5) pairs (a, b: a + b sqrt 5), exactly: three such pairs, the mesh's units"""
        p0, eu, ev, tw = self.exact[f]
        uv = q_mul(u, v)
        scale = Fraction(1, 1 << self.E)
        return [q_scale(q_add(q_add((Fraction(p0[k]), Fraction(0)), q_scale(u, eu[k])),
                              q_add(q_scale(v, ev[k]), q_scale(uv, tw[k]))), scale) for k in range(3)]


class Topology:
    """A mesh of quads and triangles as the lattice sees it: each face's corners (vertex indices, in its loop's order)
    and their places (float64, the float32 coordinates as they are), its number of sides, each face's sides' edges (and
    whether a side runs from the edge's smaller vertex to its bigger one), each edge's faces and each vertex's, the
    faces' blends (P0, the u and v edges eu and ev, and the twist w = P0 - P1 + P2 - P3, as floats and as exact ints over
    2^E; a triangle's eu and ev are its sides from its corner 0, and it has no twist), the bounds on how far a step of a
    face's parameters moves its point (sigma_lo, sigma_hi), and each face's color. Any number of faces may meet at a
    corner, and quads and triangles may share an edge: both split it into k alike.

    From a Blender mesh, or without one from positions (float64, (vertices, 3)) and faces (vertex index tuples): the
    icosahedron sphere's 20 faces, whose keys and neighbors its 3D parts use (parts_ico.py), its places not."""

    def __init__(self, mesh, color=(0.04, 0.03, 0.10, 1.0), positions=None, faces=None, chart="bilinear"):
        if mesh is not None:
            count = len(mesh.vertices)
            co = np.empty(count * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", co)
            self.positions = co.reshape(-1, 3).astype(np.float64)
            faces = [tuple(p.vertices) for p in mesh.polygons]
        else:
            self.positions = np.asarray(positions, dtype=np.float64)
            count = len(self.positions)
            faces = [tuple(int(v) for v in f) for f in faces]
        others = sum(len(f) not in (3, 4) for f in faces)
        if others or not faces:
            raise ValueError(f"surface: {getattr(mesh, 'name', 'a mesh')} has {others} faces that aren't triangles "
                             f"or quads (of {len(faces)})")
        self.corners = faces
        F = len(faces)
        self.sides = np.array([len(c) for c in faces], dtype=np.int64)
        self.edges, self.edge_sides = [], []
        edge_index = {}
        self._side_edge = [[0] * len(c) for c in faces]
        self._side_forward = [[False] * len(c) for c in faces]
        self.vertex_faces = [[] for _ in range(count)]
        for f, c in enumerate(faces):
            k = len(c)
            for s in range(k):
                a, b = c[s], c[(s + 1) % k]
                key = (min(a, b), max(a, b))
                e = edge_index.get(key)
                if e is None:
                    e = edge_index[key] = len(self.edges)
                    self.edges.append(key)
                    self.edge_sides.append([])
                self.edge_sides[e].append((f, s, a < b))
                self._side_edge[f][s] = e
                self._side_forward[f][s] = a < b
                self.vertex_faces[c[s]].append((f, s))
        self.vertex_edges = [sorted({self._side_edge[f][c] for f, c in vf}
                                    | {self._side_edge[f][(c - 1) % len(faces[f])] for f, c in vf})
                             for vf in self.vertex_faces]
        # The shape over the faces: its points, exactly and in floats, and the bounds the lattice's walks use (a ball
        # round each face, middle and radius, and how far a step of a face's parameters moves its point at least and
        # at most, sigma_lo and sigma_hi)
        # As plain lists too, for the lookups made a point at a time
        self._corners = [list(c) for c in faces]
        self._sides = self.sides.tolist()
        if chart == "catmull_clark":
            try:
                from . import subdivision
            except ImportError:
                import subdivision
            self.chart = subdivision.CatmullClarkChart(self)
        else:
            self.chart = BilinearChart(self.positions, faces, self.sides)
        self.E = self.chart.E
        self.middle, self.radius = self.chart.middle, self.chart.radius
        self.sigma_lo, self.sigma_hi = self.chart.sigma_lo, self.chart.sigma_hi
        self._levels = {}
        self.colors = np.tile(np.asarray(color, dtype=np.float64), (F, 1))
        attribute = mesh.attributes.get("Col") if mesh is not None else None
        if attribute is not None and attribute.data_type in ('FLOAT_COLOR', 'BYTE_COLOR'):
            values = np.empty(len(attribute.data) * 4, dtype=np.float32)
            attribute.data.foreach_get("color", values)
            values = values.reshape(-1, 4)
            if attribute.domain == 'FACE':
                self.colors = values.astype(np.float64)
            elif attribute.domain == 'CORNER':
                starts = np.array([p.loop_start for p in mesh.polygons])
                self.colors = np.add.reduceat(values, starts, axis=0) / self.sides[:, None]

    # ----- the lattice's points: keys, and the faces they're on -----

    def key(self, f, I, J, M):
        """The key of the point (I, J) of face f at a level of M cells across, the same from every face it's on"""
        if self._sides[f] == 4:
            on_u, on_v = I == 0 or I == M, J == 0 or J == M
            if on_u and on_v:
                return VERTEX, self._corners[f][(0, 1, 3, 2)[(I == M) + 2 * (J == M)]], 0, 0
            if not (on_u or on_v):
                return FACE, f, I, J
            s = 0 if J == 0 else 1 if I == M else 2 if J == M else 3
            t = (I, J, M - I, M - J)[s]
        else:
            on0, on1, on2 = J == 0, I + J == M, I == 0
            if on0 + on1 + on2 >= 2:
                return VERTEX, self._corners[f][0 if on2 and on0 else 1 if on0 else 2], 0, 0
            if not (on0 or on1 or on2):
                return FACE, f, I, J
            s = 0 if on0 else 1 if on1 else 2
            t = (I, J, M - J)[s]
        return EDGE, self._side_edge[f][s], t if self._side_forward[f][s] else M - t, 0

    def grid_keys(self, M):
        """Every face's points' keys at a level of M cells across, as key() gives them: (faces, M + 1, M + 1, 4) ints,
        the kind -1 where a triangle has no point (I + J > M)"""
        F = len(self.corners)
        grid = np.arange(M + 1)
        I, J = np.meshgrid(grid, grid, indexing="ij")
        keys = np.zeros((F, M + 1, M + 1, 4), dtype=np.int64)
        for k in set(self._sides):
            faces = np.flatnonzero(self.sides == k)
            corners = np.array([self._corners[f] for f in faces])
            side_edge = np.array([self._side_edge[f] for f in faces])
            side_forward = np.array([self._side_forward[f] for f in faces])
            # One face's grid: each point's kind, and its corner or its side and how far along it
            if k == 4:
                on = [J == 0, I == M, J == M, I == 0]
                corner = np.array((0, 1, 3, 2))[(I == M) + 2 * (J == M)]
                along = (I, J, M - I, M - J)
                kind = np.where((on[1] | on[3]) & (on[0] | on[2]), VERTEX, np.where(on[0] | on[1] | on[2] | on[3], EDGE, FACE))
            else:
                on = [J == 0, I + J == M, I == 0]
                corner = np.where(on[2] & on[0], 0, np.where(on[0], 1, 2))
                along = (I, J, M - J)
                kind = np.where(on[0].astype(int) + on[1] + on[2] >= 2, VERTEX, np.where(on[0] | on[1] | on[2], EDGE, FACE))
                kind = np.where(I + J > M, -1, kind)
            side = np.argmax(np.stack(on), axis=0)
            t = np.choose(side, along)
            edge_t = np.where(side_forward[:, side], t, M - t)
            is_vertex, is_edge, is_face = kind == VERTEX, kind == EDGE, kind == FACE
            keys[faces, :, :, 0] = kind
            keys[faces, :, :, 1] = np.where(is_vertex, corners[:, corner],
                                            np.where(is_edge, side_edge[:, side], faces[:, None, None]))
            keys[faces, :, :, 2] = np.where(is_edge, edge_t, np.where(is_face, I, 0))
            keys[faces, :, :, 3] = np.where(is_face, J, 0)
        return keys

    def params(self, key, M):
        """Every (face, I, J) a key is, at a level of M cells across"""
        kind, a, b, c = key
        if kind == FACE:
            return [(a, b, c)]
        sides = self._sides
        if kind == EDGE:
            return [(f, *side_point(sides[f], s, b if forward else M - b, M)) for f, s, forward in self.edge_sides[a]]
        return [(f, *corner_point(sides[f], c_, M)) for f, c_ in self.vertex_faces[a]]

    def param_in(self, key, M, face):
        """The (I, J) of a key in a face it's on"""
        for f, I, J in self.params(key, M):
            if f == face:
                return I, J
        raise KeyError((key, face))

    def neighbors(self, key, M):
        """A key's neighbors in the lattice of M cells across, a cell away along it: inside a face, along each of its
        lattice's directions (STEPS); inside an edge, along it either way and a step into each of its faces (INWARD);
        at a corner, along each of its edges. Each as (its key, a face both are on, the key's (I, J) there, the
        neighbor's)."""
        kind, a, b, c = key
        out = []
        if kind == FACE:
            for dI, dJ in STEPS[self._sides[a]]:
                out.append((self.key(a, b + dI, c + dJ, M), a, (b, c), (b + dI, c + dJ)))
            return out
        sides = self._sides
        if kind == EDGE:
            f, s, forward = self.edge_sides[a][0]
            here = side_point(sides[f], s, b if forward else M - b, M)
            for step in (1, -1):
                t = (b + step) if forward else M - (b + step)
                there = side_point(sides[f], s, t, M)
                out.append((self.key(f, *there, M), f, here, there))
            for f, s, forward in self.edge_sides[a]:
                here = side_point(sides[f], s, b if forward else M - b, M)
                for dI, dJ in INWARD[sides[f]][s]:
                    there = (here[0] + dI, here[1] + dJ)
                    out.append((self.key(f, *there, M), f, here, there))
            return out
        for e in self.vertex_edges[a]:
            f, s, forward = self.edge_sides[e][0]
            # From this corner one step along the edge, in the first face of the edge
            at_start = self._corners[f][s] == a
            here = side_point(sides[f], s, 0 if at_start else M, M)
            there = side_point(sides[f], s, 1 if at_start else M - 1, M)
            out.append((self.key(f, *there, M), f, here, there))
        return out

    def across(self, f, s):
        """The face on the other side of face f's side s, or None at the mesh's boundary"""
        for g, _s, _forward in self.edge_sides[self._side_edge[f][s]]:
            if g != f:
                return g
        return None

    # ----- places -----

    def point(self, f, I, J, M):
        """Face f's point (I, J) of a level of M cells across, exactly: (ints, d), the place in the mesh's units being
        the ints over d. (I M eu + J M ev + I J w + M^2 P0) / (M^2 2^E)."""
        return self.chart.point(f, I, J, M)

    def key_point(self, key, M):
        f, I, J = self.params(key, M)[0]
        return self.point(f, I, J, M)

    def frame(self, f, I, J, M):
        """The columns of face f's Jacobian at (I, J) of a level of M cells across: how a step of its parameters, in
        the level's cells, moves its point (the mesh's units over M), (a, b); and the twist w (BilinearChart.frame())"""
        return self.chart.frame(f, I, J, M)

    def _normal(self, f, u, v):
        """Face f's normal at (u, v), as three floats"""
        return self.chart.normal(f, u, v)

    def normal(self, f, u, v):
        return np.array(self.chart.normal(f, u, v))

    def key_normal(self, key, M):
        """The surface's normal at a key: its face's, or on an edge or a corner, the mean of its faces'. Where those
        cancel (the mesh folds back on itself there: Suzanne's has such a corner), its first face's."""
        x = y = z = 0.0
        normals = [self._normal(f, quotient(I, M), quotient(J, M)) for f, I, J in self.params(key, M)]
        for a, b, c in normals:
            x, y, z = x + a, y + b, z + c
        length = math.sqrt(x * x + y * y + z * z)
        if not length > NORMALS_CANCEL * len(normals):
            return np.array(normals[0])
        return np.array((x / length, y / length, z / length))

    def key_slope(self, key, M):
        """How steeply the surface falls away from a key's tangent plane (its normal's, key_normal()): the tangent of the
        most any of its faces is tilted from it, at most 5 (a fold back on itself); 0 inside a face"""
        if key[0] == FACE:
            return 0.0
        n = self.key_normal(key, M)
        steepest = 0.0
        for f, I, J in self.params(key, M):
            c = float(n @ self.normal(f, quotient(I, M), quotient(J, M)))
            steepest = max(steepest, math.sqrt(max(1.0 - c * c, 0.0)) / max(c, 0.2))
        return steepest

    def level_lattice(self, M):
        """A coarse level's whole lattice (LevelLattice), of M cells across, made once"""
        found = self._levels.get(M)
        if found is None:
            found = self._levels[M] = LevelLattice(self, M)
        return found


# A level with at most this many points over the whole mesh is kept whole (Topology.level_lattice()), once made: a
# coarse level's points near the drone span many faces, and finding them face by face costs as much as the faces
GLOBAL_POINTS = 20000


def is_whole(top, M):
    """Whether a level of M cells across is kept whole"""
    return len(top.corners) * M * M <= GLOBAL_POINTS


# How far the coarsest level of parts is built at most, in its cells, when it isn't kept whole (a mesh of more than
# GLOBAL_POINTS faces): its pieces grow as the square of this
FIRST_REACH = 32


class LevelLattice:
    """A coarse level's lattice over the whole mesh: its points' keys (and rows by key), places (the mesh's units,
    float64), normals and slopes (Topology.key_slope()), each one's neighbors (rows, -1 for none, in
    Topology.neighbors()' order; these two made when first asked for), each key's rank
    among them (sorted), and every face's cells (cells: groups of the cells with as many corners, each group (their
    corners' rows, in the face's order, and their faces); a quad's cells are quads, a triangle's triangles)"""

    def __init__(self, top, M):
        F = len(top.corners)
        grid = np.arange(M + 1)
        I, J = np.meshgrid(grid, grid, indexing="ij")
        u, v = I / M, J / M
        every = np.arange(F)[:, None, None]
        places = top.chart.points(every, u[None], v[None])
        normals = np.cross(*top.chart.jacobians(every, u[None], v[None]))
        normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-300)
        # Every face's grid of (M + 1) x (M + 1), a triangle's points the ones with I + J <= M (-1: not a point): each
        # key's row is its place in the order the keys first come up in, face by face. All faces' keys at once
        # (Topology.grid_keys()), and the slopes and neighbors left for whoever needs them (_around()): everything
        # here a point at a time, as until 2026-11-14, a level of 15k points took 100 to 230 ms in the floor's build,
        # a dozen frames gone the first time a dive reached it.
        grid_keys = top.grid_keys(M).reshape(-1, 4)
        valid = np.flatnonzero(grid_keys[:, 0] >= 0)
        span = max(F, len(top.edges), len(top.positions), M + 1)
        codes = ((grid_keys[valid, 0] * span + grid_keys[valid, 1]) * (M + 1) + grid_keys[valid, 2]) * (M + 1) \
            + grid_keys[valid, 3]
        _codes, found, inverse = np.unique(codes, return_index=True, return_inverse=True)
        count = len(found)
        order = np.argsort(found)
        row_of = np.empty(count, dtype=np.int64)
        row_of[order] = np.arange(count)
        flat = np.full(len(grid_keys), -1, dtype=np.int64)
        flat[valid] = row_of[inverse]
        rows = flat.reshape(F, M + 1, M + 1)
        first = valid[found[order]]
        key_rows = grid_keys[first]
        keys = [tuple(key) for key in key_rows.tolist()]
        index = dict(zip(keys, range(count)))
        self.places = places.reshape(-1, 3)[first]
        total = np.zeros((count, 3))
        np.add.at(total, flat[valid], normals.reshape(-1, 3)[valid])
        self.normals = total / np.maximum(np.linalg.norm(total, axis=1, keepdims=True), 1e-300)
        # Where the faces' normals cancel, as key_normal() has it
        counts = np.bincount(flat[valid], minlength=count)
        for row in np.flatnonzero(~(np.linalg.norm(total, axis=1) > NORMALS_CANCEL * counts)):
            self.normals[row] = top.key_normal(keys[row], M)
        self.keys, self.index = keys, index
        rank = np.empty(count, dtype=np.int64)
        rank[np.lexsort(key_rows.T[::-1])] = np.arange(count)
        self.rank = rank
        self._top, self._M, self._key_rows, self._rows = top, M, key_rows, rows
        self._slopes = self._neighbors = None
        self.cells = []
        quads, triangles = np.flatnonzero(top.sides == 4), np.flatnonzero(top.sides == 3)
        if len(quads):
            r = rows[quads]
            corners = np.stack([r[:, :-1, :-1], r[:, 1:, :-1], r[:, 1:, 1:], r[:, :-1, 1:]], axis=-1)
            self.cells.append((corners.reshape(-1, 4), np.repeat(quads, M * M)))
        if len(triangles):
            r = rows[triangles]
            # A triangle's cells: each (I, J) with I + J < M has one pointing out (to I + 1 and J + 1), and with
            # I + J < M - 1 one pointing back too, between the next ones
            ui, uj = np.nonzero(I[:-1, :-1] + J[:-1, :-1] + 1 <= M)
            di, dj = np.nonzero(I[:-1, :-1] + J[:-1, :-1] + 2 <= M)
            out = np.stack([r[:, ui, uj], r[:, ui + 1, uj], r[:, ui, uj + 1]], axis=-1)
            back = np.stack([r[:, di + 1, dj], r[:, di + 1, dj + 1], r[:, di, dj + 1]], axis=-1)
            self.cells.append((np.concatenate([out.reshape(-1, 3), back.reshape(-1, 3)]),
                               np.concatenate([np.repeat(triangles, len(ui)), np.repeat(triangles, len(di))])))

    def _around(self):
        """The points' neighbors, worked out when first asked for (by the 3D parts: the floor needs none). Inside a
        face they're the grid's, along the face's lattice's directions; only the points on edges and corners are asked
        for one at a time."""
        top, M, key_rows, keys = self._top, self._M, self._key_rows, self.keys
        count = len(keys)
        inside = key_rows[:, 0] == FACE
        around = {row: top.neighbors(keys[row], M) for row in np.flatnonzero(~inside).tolist()}
        inside_sides = top.sides[np.where(inside, key_rows[:, 1], 0)]
        inside_rows = {k: np.flatnonzero(inside & (inside_sides == k)) for k in set(top._sides)}
        width = max([len(a) for a in around.values()] + [len(STEPS[k]) for k, at in inside_rows.items() if len(at)])
        neighbors = np.full((count, width), -1, dtype=np.int64)
        for row, nbs in around.items():
            for slot, (nb, *_rest) in enumerate(nbs):
                neighbors[row, slot] = self.index.get(nb, -1)
        for k, at in inside_rows.items():
            # (None at a coarse level of triangles: level 0's points are all corners, level 1's all on edges)
            for slot, (dI, dJ) in enumerate(STEPS[k] if len(at) else ()):
                neighbors[at, slot] = self._rows[key_rows[at, 1], key_rows[at, 2] + dI, key_rows[at, 3] + dJ]
        self._neighbors = neighbors

    @property
    def slopes(self):
        """The points' slopes (Topology.key_slope()): 0 inside a face, the others' asked for one at a time, when first
        needed (by the 3D parts on a mesh: a sphere's stand on its chart, which has no folds)"""
        if self._slopes is None:
            self._slopes = np.zeros(len(self.keys))
            for row in np.flatnonzero(self._key_rows[:, 0] != FACE).tolist():
                self._slopes[row] = self._top.key_slope(self.keys[row], self._M)
        return self._slopes

    @property
    def neighbors(self):
        if self._neighbors is None:
            self._around()
        return self._neighbors


_topologies = {}     # (mesh name, vertices, faces) -> Topology


def factor(obj):
    """The map's factor: a level's cells' sides split this many ways at the next (a substitution map's mesh lattice, 2)"""
    return 2 if is_substitution(obj) else int(setting(obj, "factor"))


def draws_levels(obj):
    """Whether surface_substitution.py draws the map's levels over its floor: a substitution rule's strapwork, or a
    square map's motif (surface_cells.py; its settings under the same "surface_substitution" property)"""
    return is_substitution(obj) or obj.get("surface_substitution") is not None


def is_substitution(obj):
    """Whether the map's floor is a substitution tiling's (surface_substitution.py), not a lattice's"""
    return setting(obj, "rule") in SUBSTITUTION_RULES


def topology_of(obj):
    """The map's mesh as a Topology (made again for another mesh, or one with other counts: an edited mesh is read again
    after a file loads, forget())"""
    source = obj.get(SOURCE)
    if source is None or getattr(source, "type", None) != 'MESH':
        raise ValueError(f"surface: {obj.name} has no source mesh")
    mesh = source.data
    chart = setting(obj, "chart")
    key = (mesh.name, len(mesh.vertices), len(mesh.polygons), chart)
    topology = _topologies.get(key)
    if topology is None:
        topology = _topologies[key] = Topology(mesh, tuple(setting(obj, "floor_color")), chart=chart)
    return topology


# ----- the world -----

class World:
    """n, the anchor A (a key of the level-n lattice: Topology.key()) and rho, on a mesh's Topology whose faces split
    k x k at every level, so level n's are N = k^n cells across. A's place in its first face: face, U, V (level
    n's)."""

    layout = "mesh"
    level_shift = 0                  # a level's cells are k^(n - level) units

    def __init__(self, n, A, rho, topology, k):
        self.n, self.A, self.rho, self.topology, self.k = int(n), tuple(int(c) for c in A), float(rho), topology, int(k)
        assert self.n >= 0 and self.k >= 2
        self.N = self.k ** self.n
        self.face, self.U, self.V = topology.params(self.A, self.N)[0]
        self._point = None

    def across(self, level):
        """A level's cells across a face of the mesh, k^level"""
        return self.k ** level

    def text(self):
        return " ".join([str(self.n)] + [format(c, "x") for c in self.A])

    @staticmethod
    def from_text(text, rho, topology, k):
        n, *A = text.split()
        return World(int(n), [int(c, 16) for c in A], rho, topology, k)

    def anchor_point(self):
        """A's place in the mesh's units, exactly (Topology.point())"""
        if self._point is None:
            self._point = self.topology.point(self.face, self.U, self.V, self.N)
        return self._point

    def units(self, point):
        """An exact point's offset from A's, in units (the mesh's times N)"""
        return d_float(d_sub(point, self.anchor_point()), self.N)

    def key_units(self, key, level):
        return self.units(self.topology.key_point(key, self.across(level)))

    def camera_point(self, q):
        """The exact point at q units from A's (q floats)"""
        return d_add(self.anchor_point(), d_of(q, self.N))


def rescaled(key, j, k):
    """A key of a level as a key of the level j finer (j >= 0) or -j coarser (it must be a point of that one), in a
    lattice of factor k"""
    kind, a, b, c = key
    if kind == VERTEX:
        return key
    if j >= 0:
        grow = k ** j
        return kind, a, b * grow, c * grow
    shrink = k ** -j
    assert b % shrink == 0 and c % shrink == 0, "not a point of the coarser lattice"
    return kind, a, b // shrink, c // shrink


def dive(world, j):
    """A dive loop by k^j: the unit shrinks k times, j times; everything in map units grows as much, the mesh too"""
    world.__init__(world.n + j, rescaled(world.A, j, world.k), world.rho, world.topology, world.k)


def can_climb(world, j):
    return world.n >= j


def climb(world, j, p):
    """A climb loop by k^j: first the anchor moves to the nearest point of the k^j times coarser lattice below map point
    p, then the unit grows k^j times. Returns the move of the map's origin before the loop (map units, at the old
    scale: the drone moves by minus it first)."""
    assert can_climb(world, j)
    key, offset = Below(world, p).lattice_point(world.n - j)
    world.__init__(world.n - j, key, world.rho, world.topology, world.k)
    return world.rho * offset


def recenter(world, p, reach):
    """If the drone at map point p is more than reach units from the anchor's point, move the anchor to the level-n
    point nearest the point below it. Returns the move of the map's origin (map units: the drone moves by minus it),
    or None."""
    if float(np.linalg.norm(p)) / world.rho < reach:
        return None
    key, offset = Below(world, p).lattice_point(world.n)
    if key == world.A:
        return None
    world.__init__(world.n, key, world.rho, world.topology, world.k)
    return world.rho * offset


# ----- the point below -----

def nearest_in_face(a, b, w, inv_n, r, box):
    """In a face whose points from a base are X(d) = d0 a + d1 b + d0 d1 w / 2^n (d in units, inv_n = 2^-n), within box
    (lo_u, hi_u, lo_v, hi_v, any of them infinite), the d of its point nearest r: (d, the d nearest r on the face's
    whole patch, which box sides that one's past (0..3, -1 for none, the furthest first)). Newton's steps: Gauss-Newton
    converges by the height over the patch's curvature radius a step, slowly on a curved quad (a sphere's, Suzanne's)
    with the drone high over it, so a twisted patch's steps take its one second derivative (d^2 X / d0 d1 = w / 2^n)
    against the residual too, while that keeps the step a descent."""
    tw = w * inv_n
    d0 = d1 = 0.0
    # The steps in plain floats: a 2 x 2 system a step, which numpy's calls took 0.5 ms a face for (2026-11-13)
    (ax, ay, az), (bx, by, bz), (tx, ty, tz), (rx, ry, rz) = a.tolist(), b.tolist(), tw.tolist(), r.tolist()
    for _ in range(NEWTON_STEPS):
        ux, uy, uz = ax + d1 * tx, ay + d1 * ty, az + d1 * tz
        vx, vy, vz = bx + d0 * tx, by + d0 * ty, bz + d0 * tz
        dd = d0 * d1
        ex = d0 * ax + d1 * bx + dd * tx - rx
        ey = d0 * ay + d1 * by + dd * ty - ry
        ez = d0 * az + d1 * bz + dd * tz - rz
        g11, g12, g22 = ux * ux + uy * uy + uz * uz, ux * vx + uy * vy + uz * vz, vx * vx + vy * vy + vz * vz
        c1, c2 = ux * ex + uy * ey + uz * ez, vx * ex + vy * ey + vz * ez
        twist = ex * tx + ey * ty + ez * tz
        h12 = g12 + twist
        det = g11 * g22 - h12 * h12
        if abs(twist) > 1e-12 * (g11 + g22) and g11 > 0.0 and det > 0.0:
            s0, s1 = -(g22 * c1 - h12 * c2) / det, -(g11 * c2 - h12 * c1) / det
        else:
            det = g11 * g22 - g12 * g12
            if det > 1e-24 * (g11 + g22) ** 2:
                s0, s1 = -(g22 * c1 - g12 * c2) / det, -(g11 * c2 - g12 * c1) / det
            else:
                J = np.array([[ux, vx], [uy, vy], [uz, vz]])
                s0, s1 = np.linalg.lstsq(J, -np.array([ex, ey, ez]), rcond=None)[0].tolist()
        d0, d1 = d0 + s0, d1 + s1
        if not (abs(s0) > 1e-15 * (1.0 + abs(d0)) or abs(s1) > 1e-15 * (1.0 + abs(d1))):
            break
    d = np.array([d0, d1])
    free = d.copy()
    lo_u, hi_u, lo_v, hi_v = box
    # How far past each side, in units along the surface (side 0: v = lo_v, 1: u = hi_u, 2: v = hi_v, 3: u = lo_u)
    past = [(lo_v - d[1]) * np.linalg.norm(b), (d[0] - hi_u) * np.linalg.norm(a), (d[1] - hi_v) * np.linalg.norm(b),
            (lo_u - d[0]) * np.linalg.norm(a)]
    order = [s for s in sorted(range(4), key=lambda s: -past[s]) if past[s] > 0.0]
    if not order:
        return d, free, order
    # On the box's boundary: each side's nearest point, the best of them (a side 2^1000 units off overflows: not it)
    best, best_square = None, math.inf
    for s in range(4):
        fixed = (lo_v, hi_u, hi_v, lo_u)[s]
        if not math.isfinite(fixed):
            continue
        if s in (0, 2):
            origin, direction, lo, hi = fixed * b, a + fixed * tw, lo_u, hi_u
        else:
            origin, direction, lo, hi = fixed * a, b + fixed * tw, lo_v, hi_v
        t = float((r - origin) @ direction) / max(float(direction @ direction), 1e-300)
        t = min(max(t, lo), hi)
        if not math.isfinite(t):
            continue
        candidate = np.array([t, fixed]) if s in (0, 2) else np.array([fixed, t])
        with np.errstate(over="ignore", invalid="ignore"):
            X = candidate[0] * a + candidate[1] * b + candidate[0] * candidate[1] * tw
            square = float(np.sum((X - r) ** 2))
        if not math.isfinite(square):
            continue
        if square < best_square:
            best, best_square = candidate, square
    return (best if best is not None else d), free, order


def nearest_in_triangle(a, b, r, lo_u, lo_v, room):
    """In a triangle face whose points from a base are X(d) = d0 a + d1 b (d in units), where d0 >= lo_u, d1 >= lo_v and
    d0 + d1 <= room (any of them infinite), the d of its point nearest r, like nearest_in_face(): (d, the d nearest r on
    the face's whole plane, which sides that one's past (0: d1 = lo_v, 1: d0 + d1 = room, 2: d0 = lo_u), the furthest
    first)"""
    jacobian = np.stack([a, b], axis=1)
    d = np.linalg.lstsq(jacobian, r, rcond=None)[0]
    free = d.copy()
    # How far past each side, in units: across the long side, d0 + d1 changes by the length of its gradient on the
    # face's plane per unit
    slope = float(np.linalg.norm(np.linalg.pinv(jacobian).T @ np.ones(2)))
    with np.errstate(invalid="ignore"):
        past = [(lo_v - d[1]) * np.linalg.norm(b), (d[0] + d[1] - room) / max(slope, 1e-300),
                (lo_u - d[0]) * np.linalg.norm(a)]
    order = [s for s in sorted(range(3), key=lambda s: -past[s]) if past[s] > 0.0]
    if not order:
        return d, free, order
    # On the boundary: each side's nearest point, the best of them (a side 2^1000 units off overflows: not it)
    best, best_square = None, math.inf
    for s, fixed in enumerate((lo_v, room, lo_u)):
        if not math.isfinite(fixed):
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            # Each side as (its point at t = 0, its direction, t's range, (d0, d1) at t)
            origin, direction, lo, hi, at = (
                (lo_v * b, a, lo_u, room - lo_v, lambda t: (t, lo_v)),
                (room * b, a - b, lo_u, room - lo_v, lambda t: (t, room - t)),
                (lo_u * a, b, lo_v, room - lo_u, lambda t: (lo_u, t)))[s]
            t = float((r - origin) @ direction) / max(float(direction @ direction), 1e-300)
            t = min(max(t, lo), hi)
            if not math.isfinite(t):
                continue
            candidate = np.array(at(t), dtype=float)
            X = candidate[0] * a + candidate[1] * b
            square = float(np.sum((X - r) ** 2))
        if not math.isfinite(square):
            continue
        if square < best_square:
            best, best_square = candidate, square
    return (best if best is not None else d), free, order


class Below:
    """The point of the surface nearest map point p (from the anchor's point): its face, a lattice point of level n
    there (base: its I, J, exactly, and its offset from the anchor's point in units, base_offset) and its offset from
    that in units (d, floats); the point itself from the anchor's (foot, units), the surface's normal there, and p's
    height over it (units, negative under: on the side away from the faces' normals). lattice_point() rounds it to any
    level's lattice.

    Found by a walk over the faces down to the nearest point: from the anchor's point, or from start (face, U, V: a
    point of level n there; stay()), down to the nearest point of the dip it starts in."""

    def __init__(self, world, p, start=None):
        self.world = world
        top = world.topology
        N = world.N
        inv_n = 1 / N
        q = np.asarray(p, dtype=float) / world.rho
        if start is not None:
            f, Ub, Vb = start
            base_offset = world.units(top.point(f, Ub, Vb, N))
        else:
            f, Ub, Vb = world.face, world.U, world.V
            base_offset = np.zeros(3)
            # The anchor's face's size in units (its Jacobian's least singular value, a face being N units of it across)
            reach = min(WALK_REACH, WALK_SHARE * float(top.sigma_lo[f]) * N) if N < GLOBAL_UNITS else math.inf
            if float(np.linalg.norm(q)) > reach:
                f, Ub, Vb = self.global_face(world, q)
                base_offset = world.units(top.point(f, Ub, Vb, N))
        best = None
        visited = set()
        for _hop in range(WALK_HOPS):
            visited.add(f)
            r = q - base_offset
            k = top._sides[f]
            d, free, past, X = top.chart.nearest(f, Ub, Vb, N, r)
            square = float(np.sum((X - r) ** 2))
            if best is None or square < best[0]:
                best = (square, f, Ub, Vb, d, base_offset, X)
            if not past:
                break
            # Across the side the patch's nearest point is furthest past, if there's a face there not yet looked at,
            # from the level-n point of that side nearest where it crosses
            s = past[0]
            g = top.across(f, s)
            if g is None or g in visited:
                break
            # Where the patch's nearest point crosses the side, along it from its corner s
            if k == 4:
                t_float, t0 = (d[0], d[1], -d[0], -d[1])[s], (Ub, Vb, N - Ub, N - Vb)[s]
            else:
                t_float, t0 = (d[0], d[1], -d[1])[s], (Ub, Vb, N - Vb)[s]
            t = min(max(t0 + round(t_float), 0), N)
            I, J = side_point(k, s, t, N)
            key = top.key(f, I, J, N)
            Ub, Vb = top.param_in(key, N, g)
            f = g
            base_offset = world.units(top.point(f, Ub, Vb, N))
        _square, f, Ub, Vb, d, base_offset, X = best
        self.face, self.base, self.d, self.base_offset = f, (Ub, Vb), d, base_offset
        self.foot = base_offset + X
        u = quotient(Ub, N) + d[0] * inv_n
        v = quotient(Vb, N) + d[1] * inv_n
        self.normal = top.normal(f, min(max(u, 0.0), 1.0), min(max(v, 0.0), 1.0))
        out = q - self.foot
        distance = float(np.linalg.norm(out))
        self.height = distance if float(out @ self.normal) >= 0.0 else -distance
        self.q = q

    @staticmethod
    def global_face(world, q):
        """The face nearest q (units from the anchor's point), among them all, in the mesh's own coordinates in
        float64, and the level-n point of it nearest there"""
        top = world.topology
        N = world.N
        s = d_float(world.anchor_point()) + q * (1 / N)
        # Only the faces whose balls reach nearer than the nearest middle: a handful of a mesh's thousands, so a tick's
        # few points below cost a fraction of a millisecond (all 1990 of the Penrose torus's were 1 ms each, 2026-11-13)
        reach = np.linalg.norm(top.middle - s, axis=1)
        faces = np.flatnonzero(reach - top.radius <= float(np.min(reach)))
        f, (U, V) = Below.nearest_face(top, faces, s)
        return (f, *nearest_point(top._sides[f], U * N, V * N, N))

    @staticmethod
    def nearest_face(top, faces, s):
        """Among faces (indices), the one whose blend comes nearest the mesh point s, and (u, v) there, by a few
        Gauss-Newton steps from each one's middle"""
        chart = top.chart
        triangle = top.sides[faces] == 3
        u = np.where(triangle, 1.0 / 3.0, 0.5)
        v = np.where(triangle, 1.0 / 3.0, 0.5)
        for _ in range(4):
            X = chart.points(faces, u, v)
            ju, jv = chart.jacobians(faces, u, v)
            r = s - X
            # The normal equations of the 2 x 2 least squares step, per face
            a11, a12, a22 = np.sum(ju * ju, 1), np.sum(ju * jv, 1), np.sum(jv * jv, 1)
            b1, b2 = np.sum(ju * r, 1), np.sum(jv * r, 1)
            det = np.maximum(a11 * a22 - a12 * a12, 1e-300)
            u = np.clip(u + (a22 * b1 - a12 * b2) / det, 0.0, 1.0)
            v = np.clip(v + (a11 * b2 - a12 * b1) / det, 0.0, 1.0)
            # Back into a triangle across its long side
            over = np.where(triangle, np.maximum(u + v - 1.0, 0.0), 0.0) / 2.0
            u, v = np.clip(u - over, 0.0, 1.0), np.clip(v - over, 0.0, 1.0)
        X = chart.points(faces, u, v)
        k = int(np.argmin(np.sum((X - s) ** 2, axis=1)))
        return int(faces[k]), (float(u[k]), float(v[k]))

    def cell_place(self, level):
        """Where the point is in the level's cells of its face, (u, v) in cells as Fractions: the base's exactly, plus
        d"""
        world = self.world
        shift = world.n - level
        Ub, Vb = self.base
        scale = Fraction(1, world.k ** shift) if shift >= 0 else Fraction(world.k ** -shift)
        return ((Fraction(Ub) + Fraction(float(self.d[0]))) * scale, (Fraction(Vb) + Fraction(float(self.d[1]))) * scale)

    def stay(self, world, p):
        """The Below of p walked to from this one's point (the dip it's in), as world has it now"""
        u, v = self.cell_place(world.n)
        start = (self.face, *nearest_point(world.topology._sides[self.face], u, v, world.N))
        return Below(world, p, start)

    def lattice_point(self, level):
        """The point of the level's lattice nearest it (in its face's parameters), and that one's offset from the
        anchor's point in units: (key, offset)"""
        M = self.world.across(level)
        u, v = self.cell_place(level)
        top = self.world.topology
        I, J = nearest_point(top._sides[self.face], u, v, M)
        key = top.key(self.face, I, J, M)
        return key, self.world.units(top.point(self.face, I, J, M))


# ----- the map's object -----

def is_surface(obj):
    return (obj is not None and hasattr(obj, "get") and obj.get(STATE) is not None and obj.get(PROPERTY) is not None
            and obj.get(SOURCE) is not None)


def surface_objects(scene):
    return [obj for obj in scene.objects if is_surface(obj)]


_worlds = {}         # map object name -> (version, World)


def world_of(obj):
    """obj's World, read again if its properties changed under it (an undo)"""
    cached = _worlds.get(obj.name)
    version = obj.get(VERSION, 0)
    topology = topology_of(obj)
    k = factor(obj)
    if cached is None or cached[0] != version or cached[1].topology is not topology or cached[1].k != k:
        world = World.from_text(obj[STATE], float(setting(obj, "rho")), topology, k)
        _worlds[obj.name] = cached = (version, world)
    return cached[1]


def save(obj, world):
    obj[STATE] = world.text()
    version = obj.get(VERSION, 0) + 1
    obj[VERSION] = version
    _worlds[obj.name] = (version, world)


map_point = sphere.map_point
world_vector = sphere.world_vector
map_vector = sphere.map_vector


# map object name -> [(world text, map point, Below)]: the last few, asked again in the same tick (a tick asks for the
# camera's before and after its move, and the end of the move, each two or three times; and the parts, for where the
# drone will be, a point or two more)
_below = {}
BELOW_KEPT = 6
BELOW_SAME = 4 * 2.0 ** -24
# How much further (a share of the height) the point below in the dip it was in may be than the nearest before it
# moves to the nearest's (below_of())
BELOW_STICK = 0.05


def below_of(obj, world, p):
    p = np.asarray(p, dtype=float)
    text = world.text()
    kept = _below.setdefault(obj.name, [])
    # The same point to within the camera's own float32 rounding: scaled about itself, or taken along with its map,
    # the camera comes back a step or two off, and each of those was another point below, 2.6 a tick (2026-11-14)
    near = BELOW_SAME * float(np.abs(p).max())
    for cached in kept:
        if cached[0] == text and float(np.abs(cached[1] - p).max()) <= near:
            return cached[2]
    below = Below(world, p)
    # Over a saddle, a corner with more than 360 degrees of faces round it (the Darb and Penrose tori's inner special
    # points, 432), the faces fold up and down round the corner, and a point above it is nearer a face folded up than
    # the corner, one dip on each side. Diving at such a corner, the nearest point jumped from one dip to the other,
    # some 0.6 heights, at a point of each loop's dive, and back and forth tick after tick in a mirror plane of the
    # tiling, where the two are as near to the last digits; and every engine that builds round it with it: the 3D
    # parts' finest level, a few tenths of a height across, was rebuilt round one and then the other, missing from 47%
    # of the frames of such a dive (2026-11-23). So the point below stays in the dip it was in (walked down to from the
    # last one) until that's BELOW_STICK of the height further than the nearest: 19% (most of it the climb back out,
    # as on any face). Walked from a point below in a face next to it, the same point, so elsewhere nothing changes.
    if kept:
        same = [cached for cached in kept if cached[0] == text] or kept
        last = min(same, key=lambda cached: float(np.abs(cached[1] - p).max()))[2]
        if last.face != below.face and last.world.k == world.k:
            stay = last.stay(world, p)
            if abs(stay.height) <= abs(below.height) * (1.0 + BELOW_STICK):
                below = stay
    kept.insert(0, (text, p.copy(), below))
    del kept[BELOW_KEPT:]
    return below


# ----- what the flight asks (ER_OT_Fly) -----

def surface_vector(obj, camera_world):
    """From a camera (world space) to the nearest point of the surface (world space): what auto scale measures to, with
    or without its Mesh setting (the map's origin moves with the anchor, and the mesh is k^n times its size), and the
    way down for tidal lock and the horizon"""
    world = world_of(obj)
    p = map_point(obj, camera_world)
    below = below_of(obj, world, p)
    return world_vector(obj, world.rho * below.foot - p)


def center_vector(obj, camera_world):
    """From a camera (world space) to the mesh's origin (world space), a torus's or a planet's center: what zoom and
    orbit scale and turn the drone about, as they do about an ordinary object's origin, since the map's object's origin
    is the anchor's point, which follows the drone. inf past a float's range."""
    world = world_of(obj)
    p = map_point(obj, camera_world)
    with np.errstate(over="ignore", invalid="ignore"):
        return world_vector(obj, world.rho * world.units(((0, 0, 0), 1)) - p)


def crossing(obj, start_world, end_world):
    """If a move from start to end (world space) crosses the surface, the heights of its ends over it (world units,
    signed), else None (see ER_OT_Fly.stop_at_surface())"""
    world = world_of(obj)
    scale = float(np.linalg.norm(world_vector(obj, (1.0, 0.0, 0.0))))
    h0 = below_of(obj, world, map_point(obj, start_world)).height
    h1 = below_of(obj, world, map_point(obj, end_world)).height
    h0, h1 = h0 * world.rho * scale, h1 * world.rho * scale
    return (h0, h1) if h0 > 0.0 >= h1 else None


def loop(obj, scale, camera_world):
    """A loop by scale (a power of the map's factor, see loop_factor()) about the map's origin: a dive, or after the
    anchor moves to the coarser lattice under the camera, a climb (an endless plane's re-topped first if that would
    leave it too few levels above the drone, see retop()). Returns the move of the map's origin the climb made first
    (world units, before the loop: the drone moves by minus it), or None."""
    world = world_of(obj)
    j = round(log_k(scale, world.k))
    move = None
    if j > 0:
        dive(world, j)
    elif j < 0:
        if is_endless(obj) and world.n + j < TOP_ROOM:
            retop(obj, world, RETOP_LEVELS)
            world = world_of(obj)
        move = world_vector(obj, climb(world, -j, map_point(obj, camera_world)))
    save(obj, world)
    return move


def loop_factor(obj, scale):
    """The loop the map makes for a loop by scale: the nearest power of its factor, a climb no higher than the mesh's
    own size (n = 0), 1.0 if there's no climbing left; an endless plane's, any (it's re-topped as it climbs)"""
    world = world_of(obj)
    most = math.inf if is_endless(obj) else world.n
    return float(world.k) ** loops.limited_power(scale, most, log_base=math.log(world.k))


def at_top(obj):
    """Whether the map is at the mesh's own size, n = 0, with nothing coarser to climb into (never, for an endless
    plane)"""
    return not is_endless(obj) and world_of(obj).n == 0


# ----- an endless plane: no top -----

# An endless plane (the "endless" setting) is a map whose mesh is one square face, the plane's top cell: whatever's
# biggest so far. It's kept at least TOP_ROOM levels over the drone's (n, the depth of the units), and when a climb would
# leave it fewer, it's re-topped: RETOP_LEVELS levels are made up above it (retop()). Re-topping forgets every build,
# a pop, so the top is made up so far above (some 500 climbing loops) that a flight never gets there: the exact
# arithmetic is as good a thousand levels down as at the top.
TOP_ROOM = 60
RETOP_LEVELS = 1000


def is_endless(obj):
    return bool(setting(obj, "endless"))


def retop(obj, world, levels, clear=None):
    """Make up levels levels above an endless plane's top: each time, the face (the top cell) becomes one of the k x k
    cells of a cell k times its size, a digit along each side, so the world goes a level deeper (n + 1) and the anchor's
    whole numbers shift by where the old top is in the new: the same place, the same units, nothing on screen moves
    (endless.py's digit stack, as the surface's anchor). The digits are random (seeded by the setting and the depth, so
    a map makes up the same ones every time) among those that keep the anchor's point at least 1/(2k) of the new top's
    side in from its sides (there's always one: the one nearest its middle), and of those, the ones where it's clear
    of the motif's pieces at that level (clear(x, y): the point's place in the new top's cell; none clear, any), so
    a made-up level never puts its lines or frames under the drone, and climbing never comes back to the top's edge.

    Every level is relabeled (level d of the old top is d + 1 of the new), so whatever's built for the map is
    forgotten and built again: one hitch every RETOP_LEVELS - TOP_ROOM levels of climbing."""
    settings = setting(obj, "endless")
    seed = int(settings.get("seed", 0)) if hasattr(settings, "get") else 0
    if clear is None and draws_levels(obj):
        try:
            from . import surface_substitution
        except ImportError:
            import surface_substitution
        clear = lambda x, y: surface_substitution.clear_point(obj, x, y)
    k, top = world.k, world.topology
    N = world.N
    kind, a, b, c = world.A
    if kind == FACE:
        I, J = b, c
    else:
        I, J = top.param_in(world.A, N, 0)
    n = world.n
    least = Fraction(1, 2 * k)
    for _ in range(levels):
        rng = random.Random(f"{seed}:{n}")
        x, y = Fraction(I, N), Fraction(J, N)
        options = []
        for dx in range(k):
            for dy in range(k):
                u, v = (dx + x) / k, (dy + y) / k
                if min(u, 1 - u, v, 1 - v) >= least:
                    options.append((dx, dy, u, v))
        clear_ones = [o for o in options if clear is None or clear(float(o[2]), float(o[3]))]
        dx, dy, _u, _v = rng.choice(clear_ones or options)
        I, J = dx * N + I, dy * N + J
        N *= k
        n += 1
    world.__init__(n, (FACE, 0, I, J), world.rho, top, k)
    save(obj, world)
    forget()
    parts.forget()


def recenter_move(obj, camera_world):
    """Move the anchor under the drone camera if it has moved off it (see recenter()): the move of the map's origin in
    world units (the drone moves by minus it), or None"""
    world = world_of(obj)
    p = map_point(obj, camera_world)
    below = below_of(obj, world, p)
    reach = max(RECENTER_UNITS, abs(below.height) * RECENTER_HEIGHTS)
    move = recenter(world, p, reach)
    if move is None:
        return None
    save(obj, world)
    return world_vector(obj, move)


def drone_clip_end():
    return sphere.drone_clip_end()


# ----- the floor: the surface near the drone -----

class FloorBuilt(schedule.Window):
    """The floor as built into its object (kept in memory, by map): its window (schedule.Window: the mesh is from the
    camera then), the level of its cells, and if any were split near the camera (refine()), the drone's height then
    (map units then), else None. placed: the world text the object's transform is for."""

    def __init__(self, world_text, n, center, window, level, split_height=None):
        super().__init__(world_text, n, center, window)
        self.level, self.split_height = level, split_height


_floors = {}         # map object name -> FloorBuilt
# The floor's cells are cut near the camera into pieces about their distance from it across, down to this share of
# the drone's height (tiling.split_near())
FLOOR_SPLIT_TILE = 0.5


def floor_object(obj):
    """The map's floor's object, made the first time: a plain mesh, a child of the map's object, which clicking selects
    the map through (scaling_target)"""
    import bpy
    found = next((o for o in obj.children if o.get(FLOOR_KEY) == obj.name), None)
    if found is not None:
        parts.no_shadow(found)
        return found
    name = f"{obj.name}.floor"
    floor_obj = bpy.data.objects.new(name, bpy.data.meshes.new(name))
    for c in obj.users_collection:
        c.objects.link(floor_obj)
    floor_obj.parent = obj
    floor_obj[FLOOR_KEY] = obj.name
    floor_obj["scaling_target"] = obj
    floor_obj.data.materials.append(parts.parts_material())
    # Built again as the drone flies, like the parts: no shadow (parts.no_shadow())
    parts.no_shadow(floor_obj)
    return floor_obj


def transform(world, text, built):
    """Where something built for another world (built.world_text) is in this one's map (world, its text): (offset,
    scale), its map's origin then (the anchor it was built for) from this map's origin, and how much bigger map units
    are now. Exact from the two anchors' places (World.anchor_point())."""
    if built.transform is not None and built.transform[0] == text:
        return built.transform[1], built.transform[2]
    # The same pair of worlds for everything built in one: a Penrose level's patches (surface_substitution.patch_bars()),
    # a hundred a build, each worked its transform out exactly again at every loop (2026-11-20)
    key = (text, built.world_text, world.rho, id(world.topology), world.k)
    found = _transforms.get(key)
    if found is None:
        then = World.from_text(built.world_text, world.rho, world.topology, world.k)
        found = (world.rho * world.units(then.anchor_point()), power(world.k, world.n - then.n))
        if len(_transforms) > 256:
            _transforms.clear()
        _transforms[key] = found
    offset, scale = found
    built.transform = (text, offset, scale)
    return offset, scale


_transforms = {}     # (text, then's text, rho, topology, factor) -> transform()'s (offset, scale)


def floor_level(world, below, window, cells):
    """The level whose cells, where the drone is, are about window / cells across (map units), no coarser than 0"""
    top = world.topology
    f = below.face
    a, b, _w = top.frame(f, *below.base, world.N)
    cell = world.rho * math.sqrt(float(np.linalg.norm(a)) * float(np.linalg.norm(b)))     # a level-n cell, map units
    wanted = window / cells
    if cell <= 0.0 or not math.isfinite(cell):
        return world.n
    return max(0, world.n - math.floor(log_k(wanted / cell, world.k)))


def face_grid(top, f, IR, JR, M, ref, reach, cap, scale=1.0):
    """The grid of face f's level points (M of the level's cells a mesh edge) within reach of the origin, found from one of them,
    (IR, JR), placed at ref: their offsets from it (gi, gj, floats, (a, b)) and places (a, b, 3), from ref along the
    face's patch, scale times the level's own units (the chart's grid_offsets()). Out to the face's sides, and no more
    than cap steps either way. Also the steps it reaches and, for a triangle, rem = M - IR - JR (how far its long side
    is, in steps of I + J; past 2^60, out of reach anyway). The parameters can be any size (Python ints); the offsets
    are small."""
    sigma = max(float(top.sigma_lo[f]), 1e-12 * float(top.sigma_hi[f]), 1e-300) * scale
    # (A face some 1e154 units off overflows the distance: as far as the cap, its points then out of reach)
    with np.errstate(over="ignore"):
        far = (float(np.linalg.norm(ref)) + reach) / sigma
    D = min(int(math.ceil(far)) + 1, cap) if math.isfinite(far) else cap
    dI = np.arange(max(0, IR - D) - IR, min(M, IR + D) - IR + 1, dtype=np.float64)
    dJ = np.arange(max(0, JR - D) - JR, min(M, JR + D) - JR + 1, dtype=np.float64)
    gi, gj = np.meshgrid(dI, dJ, indexing="ij")
    places = ref + scale * top.chart.grid_offsets(f, IR, JR, M, gi, gj)
    return D, gi, gj, places, min(M - IR - JR, 1 << 60)


def face_cells(k, grid, offsets, rem):
    """A face's cells in a window of its lattice: grid (its points, (a, b, 3), from (I0, J0) on), offsets (their (I, J)
    from there, (a, b, 2)) and, for a triangle, rem = M - I0 - J0 (how far its long side is, in steps of I + J): a
    quad's cells, or a triangle's (pointing out and back), as (corners (m, k, 3), their offsets (m, k, 2))"""
    if k == 4:
        corner = lambda g: np.stack([g[:-1, :-1], g[1:, :-1], g[1:, 1:], g[:-1, 1:]], axis=-2)
        return corner(grid).reshape(-1, 4, 3), corner(offsets).reshape(-1, 4, 2)
    total = offsets[:-1, :-1, 0] + offsets[:-1, :-1, 1]
    out, back = total + 1 <= rem, total + 2 <= rem
    corners = np.concatenate([np.stack([grid[:-1, :-1], grid[1:, :-1], grid[:-1, 1:]], axis=-2)[out],
                              np.stack([grid[1:, :-1], grid[1:, 1:], grid[:-1, 1:]], axis=-2)[back]])
    at = np.concatenate([np.stack([offsets[:-1, :-1], offsets[1:, :-1], offsets[:-1, 1:]], axis=-2)[out],
                         np.stack([offsets[1:, :-1], offsets[1:, 1:], offsets[:-1, 1:]], axis=-2)[back]])
    return corners, at


def refine(corners, faces, near):
    """Quad cells of the floor (corners (m, 4, 3) from the camera, their faces), each split 2 x 2 while it bulges off the
    two triangles it's drawn as by more than FLOOR_FLATNESS of its distance from the camera, or of near (the drone's
    height) if that's more: a bent quad's cells are
    its bilinear patch's, and a cell's sides lie on the patch (a bilinear patch is straight along u and along v) but its
    middle doesn't, off by a quarter of its corners' twist. Cells the floor's level makes hundreds of drone heights
    across bulge by more than the drone's height over a bent quad (Suzanne's), so near the camera they're split, into
    their corners' blend at the halves (exactly the patch's points), and their sides still meet the cells next to them
    unsplit on a straight line: no cracks. A flat face's cells never split (a trapezoid's corners twist, but within its
    plane, which the triangles are in too: only the twist off the cell's plane counts)."""
    out_corners, out_faces = [], []
    for _ in range(FLOOR_SPLITS):
        # Off the cell's plane (its diagonals'): a flat trapezoid's corners twist too, within its plane
        across = np.cross(corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 1])
        across /= np.maximum(np.linalg.norm(across, axis=1, keepdims=True), 1e-300)
        twist = corners[:, 0] - corners[:, 1] + corners[:, 2] - corners[:, 3]
        bulge = np.abs(np.sum(twist * across, axis=1)) / 4.0
        middle = corners.mean(axis=1)
        radius = np.linalg.norm(corners - middle[:, None], axis=-1).max(axis=1)
        distance = np.maximum(np.linalg.norm(middle, axis=1) - radius, near)
        split = bulge > FLOOR_FLATNESS * distance
        out_corners.append(corners[~split])
        out_faces.append(faces[~split])
        if not split.any():
            break
        c0, c1, c2, c3 = (corners[split, k] for k in range(4))
        m01, m12, m23, m30 = (c0 + c1) / 2.0, (c1 + c2) / 2.0, (c2 + c3) / 2.0, (c3 + c0) / 2.0
        mid = (c0 + c1 + c2 + c3) / 4.0
        corners = np.concatenate([np.stack(c, axis=1) for c in ((c0, m01, mid, m30), (m01, c1, m12, mid),
                                                                (mid, m12, c2, m23), (m30, mid, m23, c3))])
        faces = np.tile(faces[split], 4)
    else:
        out_corners.append(corners)
        out_faces.append(faces)
    return np.concatenate(out_corners), np.concatenate(out_faces), len(out_corners) > 1


def refine_smooth(corners, faces, near, chart, M, scale, g0, bases):
    """refine() for a curved chart (one whose cells aren't their corners' bilinear blend: a subdivision surface's): a
    cell splits 2 x 2 while its true middle, or its corners' twist, bulges off the two triangles it's drawn as by more
    than FLOOR_FLATNESS of its distance (or near), and its new corners are the surface's own, from its face's base
    point: g0 (m, 2) is each cell's first corner's offset from its face's base (floats, level steps), bases[f] = (I, J,
    place from the camera) that base, a point of the level of M cells across, scale map units its steps' units. Split
    by averaging, its corners were on the big cell's chords, off the surface by its sag (the smooth torus's floor, 0.14
    off by a cell 4 across, curving at 12: the user, 2026-11-23). A split cell's side meets an unsplit neighbor's
    chord within the flatness (about a pixel)."""

    def place(faces_, gi, gj):
        shape = gi.shape
        faces_, gi, gj = faces_.ravel(), gi.ravel(), gj.ravel()
        out = np.empty((len(gi), 3))
        for f in np.unique(faces_).tolist():
            # (On a worker thread, out of the main thread's way: a curved face's offsets are mostly Python)
            background.give_way()
            rows = np.flatnonzero(faces_ == f)
            I, J, ref = bases[f]
            out[rows] = ref + scale * chart.grid_offsets(f, I, J, M, gi[rows], gj[rows])
        return out.reshape(shape + (3,))

    out_corners, out_faces = [], []
    h = 1.0
    for _ in range(FLOOR_SPLITS):
        across = np.cross(corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 1])
        across /= np.maximum(np.linalg.norm(across, axis=1, keepdims=True), 1e-300)
        twist = corners[:, 0] - corners[:, 1] + corners[:, 2] - corners[:, 3]
        middle = corners.mean(axis=1)
        true = place(faces, g0[:, 0] + h / 2.0, g0[:, 1] + h / 2.0)
        bulge = np.abs(np.sum(twist * across, axis=1)) / 4.0 + np.abs(np.sum((true - middle) * across, axis=1))
        radius = np.linalg.norm(corners - middle[:, None], axis=-1).max(axis=1)
        distance = np.maximum(np.linalg.norm(middle, axis=1) - radius, near)
        split = bulge > FLOOR_FLATNESS * distance
        out_corners.append(corners[~split])
        out_faces.append(faces[~split])
        if not split.any():
            break
        faces, g0, h = faces[split], g0[split], h / 2.0
        # The split cells' 3 x 3 points, (m, 3, 3, 3): [i][j] at g0 + (i, j) h
        steps = np.arange(3) * h
        gi = g0[:, 0, None, None] + steps[None, :, None] + 0.0 * steps[None, None, :]
        gj = g0[:, 1, None, None] + 0.0 * steps[None, :, None] + steps[None, None, :]
        grid = place(np.repeat(faces, 9).reshape(-1, 3, 3), gi, gj)
        quarter = lambda i, j: np.stack([grid[:, i, j], grid[:, i + 1, j], grid[:, i + 1, j + 1], grid[:, i, j + 1]],
                                        axis=1)
        corners = np.concatenate([quarter(0, 0), quarter(1, 0), quarter(1, 1), quarter(0, 1)])
        g0 = np.concatenate([g0, g0 + (h, 0.0), g0 + (h, h), g0 + (0.0, h)])
        faces = np.tile(faces, 4)
    else:
        out_corners.append(corners)
        out_faces.append(faces)
    return np.concatenate(out_corners), np.concatenate(out_faces), len(out_corners) > 1


def build_floor(world, camera, below, window, settings_cells):
    """The cells of the floor's level within window of the camera (map units), each corner from the camera, as
    (corners (m, 3), corners per face, colors, level, whether any cell was split). A coarse level's from its whole lattice (LevelLattice), at a
    depth where float64 places the mesh to far under a float32 step of the window; a finer one's face by face from the
    one below the camera out: in each face, one of its points exactly from the camera, rounded once, and the grid of
    its cells within reach from that, all at once. A quad's cells are quads, a triangle's triangles, and a bent quad's
    near the camera are split finer (refine())."""
    top = world.topology
    level = floor_level(world, below, window, settings_cells)
    M = world.across(level)
    N = world.N
    camera_point = world.camera_point(np.asarray(camera, dtype=float) / world.rho)
    out = []                         # (corners (m, k, 3), faces (m,))
    height = abs(below.height) * world.rho
    split = False

    smooth = not top.chart.flat_cells
    bases = {}                       # a curved chart's: face -> its cells' base point (refine_smooth())

    def keep(corners, face_of, g0=None, scale=None):
        nonlocal split
        middle = corners.mean(axis=-2)
        radius = np.linalg.norm(corners - middle[..., None, :], axis=-1).max(axis=-1)
        near = (np.linalg.norm(middle, axis=-1) - radius <= window) & np.all(np.isfinite(corners), axis=(-1, -2))
        cells, faces = corners[near], np.broadcast_to(face_of, near.shape)[near]
        if cells.shape[-2] == 4 and len(cells):
            if smooth:
                cells, faces, refined = refine_smooth(cells, faces, height, top.chart, M, scale, g0[near], bases)
            else:
                cells, faces, refined = refine(cells, faces, height)
            split |= refined
        out.append((cells, faces))
        return near

    if is_whole(top, M) and N < GLOBAL_UNITS:
        whole = top.level_lattice(M)
        from_camera = (whole.places - d_float(camera_point)) * (world.rho * N)
        for rows, faces in whole.cells:
            if smooth and rows.shape[-1] == 4:
                # Each quad's cells in order, (I, J) row by row: each one's first corner from its face's (0, 0)
                cell = np.arange(len(rows)) % (M * M)
                g0 = np.stack([cell // M, cell % M], axis=1).astype(float)
                for f in np.unique(faces).tolist():
                    bases[f] = (0, 0, (top.chart.points(f, 0.0, 0.0) - d_float(camera_point)) * (world.rho * N))
                keep(from_camera[rows], faces, g0, world.rho * N / M)
            else:
                keep(from_camera[rows], faces)
    else:
        scale = world.rho * power(world.k, world.n - level)      # map units per the level's own (the mesh's times M)
        u, v = below.cell_place(level)
        queue = [(below.face, *nearest_point(top._sides[below.face], u, v, M))]
        done = set()
        while queue and sum(len(c) for c, _ in out) < FLOOR_MAX_CELLS:
            f, IR, JR = queue.pop(0)
            if f in done:
                continue
            background.give_way()
            done.add(f)
            k = top._sides[f]
            ref = world.rho * d_float(d_sub(top.point(f, IR, JR, M), camera_point), N)
            if not np.all(np.isfinite(ref)):
                continue
            D, gi, gj, grid, rem = face_grid(top, f, IR, JR, M, ref, window, FLOOR_REACH, scale)
            if min(gi.shape) < 2:
                continue
            offsets = np.stack([gi, gj], axis=-1).astype(np.int64)
            cells, at = face_cells(k, grid, offsets, rem)
            bases[f] = (IR, JR, ref)
            near = keep(cells, f, at[:, 0].astype(float), scale)
            if not near.any():
                continue
            # The faces across the sides its kept cells reach, from a kept cell's corner on that side: each side as
            # an axis of the offsets from (IR, JR) (None: I + J, a triangle's long side) and where it is along it
            kept = at[near].reshape(-1, 2)
            if k == 4:
                sides = ((1, -JR), (0, M - IR), (1, M - JR), (0, -IR))
            else:
                sides = ((1, -JR), (None, M - IR - JR), (0, -IR))
            for s_, (axis, where) in enumerate(sides):
                if abs(where) > D + 1:
                    continue
                g = top.across(f, s_)
                if g is None or g in done:
                    continue
                along = kept.sum(axis=1) if axis is None else kept[:, axis]
                rows = np.flatnonzero(along == where)
                if not len(rows):
                    continue
                I, J = IR + int(kept[rows[0], 0]), JR + int(kept[rows[0], 1])
                queue.append((g, *top.param_in(top.key(f, I, J, M), M, g)))
    out = [(c, f) for c, f in out if len(c)]
    if not out:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64), np.zeros((0, 4)), level, False
    corners = np.concatenate([c.reshape(-1, 3) for c, _ in out])
    sizes = np.concatenate([np.full(len(c), c.shape[1], dtype=np.int64) for c, _ in out])
    colors = top.colors[np.concatenate([f for _, f in out])]
    return corners, sizes, colors, level, split


def floor_job(world, camera, below, window, settings_cells):
    """The floor built round camera, ready to write (write_floor()): on a worker thread once the floor is drawn
    (settle_floor()), touching no bpy"""
    corners, sizes, colors, level, split = build_floor(world, camera, below, window, settings_cells)
    # Cells far bigger than their distance from the camera cut smaller near it (tiling.split_near(): a cell a
    # quarter of the window across had its depth by the camera off by as much as the levels over it are apart)
    if len(sizes):
        points, counts, rows = tiling.split_near(*tiling.polygons_of(corners, sizes), np.zeros(3),
                                                 FLOOR_SPLIT_TILE * abs(below.height) * world.rho)
        corners, sizes = tiling.corners_of(points, counts)
        colors = np.asarray(colors)[rows]
    height = abs(below.height) * world.rho if split else None
    return (FloorBuilt(world.text(), world.n, np.array(camera, dtype=float), window, level, height), corners, sizes,
            colors)


def floor_jobs(obj):
    """Whose jobs a map's floor builds are on the worker thread (background.py)"""
    return "floor:" + obj.name


# The floor's builds go before every other build waiting (background.submit()'s priority, lower first)
FLOOR_PRIORITY = -1e30
_floor_failed = set()        # map object names whose floor's last build on the worker thread failed


def settle_floor(obj, world, camera, below, clip_end):
    """The floor in today's map: built again when the camera's view (clip_end) reaches past its window or it's grown far
    wider than wanted (the drone has shrunk: its corners are float32 from its middle), or with cells split for where the
    camera was (a bent quad's, refine()) once it's moved on FLOOR_FOLLOW of its heights or come down to half its height,
    else only moved and scaled. Once there's a floor, it's built again on a worker thread (background.py) and stays as
    it was until that's written in a later tick: built here, a smooth chart's floor took 0.1 to 0.3 s, a freeze at every
    loop of a dive (the user, 2026-11-24)."""
    window = float(setting(obj, "window")) * clip_end
    floor_obj = floor_object(obj)
    text = world.text()
    owner = floor_jobs(obj)
    for _key, result in background.finished(owner):
        if result is background.FAILED:
            _floor_failed.add(obj.name)
        else:
            write_floor(obj, floor_obj, *result)
    built = _floors.get(obj.name)
    rebuild = built is None
    if not rebuild:
        offset, scale = transform(world, text, built)
        local = (camera - offset) / scale
        moved = float(np.linalg.norm(local - built.center))
        rebuild = not built.covers(moved, clip_end / scale) or built.overbuilt(window / scale)
        if built.split_height is not None:
            height = abs(below.height) * world.rho / scale
            rebuild |= moved > FLOOR_FOLLOW * height or height < 0.5 * built.split_height
    if rebuild:
        job = (World.from_text(text, world.rho, world.topology, world.k), np.array(camera, dtype=float), below, window,
               float(setting(obj, "floor_cells")))
        key = (owner, 0)
        if built is not None and background.enabled and obj.name not in _floor_failed:
            # Handed in again round the newest camera while it waits; left alone once it's under way
            if not background.running(key):
                background.submit(key, floor_job, *job, priority=FLOOR_PRIORITY, replace=True)
        else:
            _floor_failed.discard(obj.name)
            background.cancel(key)
            write_floor(obj, floor_obj, *floor_job(*job))
    built = _floors[obj.name]
    offset, scale = transform(world, text, built)
    if built.placed != text:
        built.placed = text
        if "sub_push" in floor_obj:
            # Under a Penrose map's levels (surface_substitution.py): its push drivers place it from its unpushed place
            for key, value in zip(("sub_bx", "sub_by", "sub_bz"), offset + scale * built.center):
                floor_obj[key] = float(value)
            floor_obj["sub_bs"] = floor_obj["camera_push_scale"] = float(scale)
            floor_obj.update_tag(refresh={'OBJECT'})
        else:
            from mathutils import Matrix, Vector
            floor_obj.matrix_basis = Matrix.Translation(Vector(offset + scale * built.center)) @ Matrix.Scale(scale, 4)


def write_floor(obj, floor_obj, built, corners, sizes, colors):
    """A floor build (floor_job()) into the floor's object, placed in the next settle_floor()"""
    _floors[obj.name] = built
    tiling.write_mesh(floor_obj, corners, sizes, colors)


def settle(obj, camera_world):
    """At the end of a tick (and when a flight starts): the floor where the camera ended up"""
    if not is_surface(obj):
        return
    world = world_of(obj)
    camera = map_point(obj, camera_world)
    unit = float(np.linalg.norm(world_vector(obj, (1.0, 0.0, 0.0))))
    clip_end = drone_clip_end() / unit
    if clip_end <= 0.0:
        return
    settle_floor(obj, world, camera, below_of(obj, world, camera), clip_end)
    if draws_levels(obj):
        try:
            from . import surface_substitution
        except ImportError:
            import surface_substitution
        surface_substitution.settle(obj, camera_world)


# ----- the lattice for the 3D parts (parts.py) -----

class LocalLattice:
    """A level's lattice points, M cells across a face, near a point of it (center, a key) out to radius cells (the mean
    distance from center to its neighbors) and a ring more: their keys, places from center in the level's own units (the
    mesh's times M), normals, slopes (Topology.key_slope()), whether each is within radius (inner), each one's
    neighbors (rows, -1 for none) and the edges between inner points (rows, the smaller key first). A coarse level's
    are picked out of its whole lattice (LevelLattice); a finer one's, found face by face: in each face reached, the
    grid of its points within reach of center, from a point of it already placed, all at once."""

    def __init__(self, top, M, center, radius):
        self.M = M
        if is_whole(top, M):
            rank = self.from_level(top, center, radius)
        else:
            rank = self.by_faces(top, center, radius)
        # The edges between inner points, each once, the smaller key first
        inner = np.flatnonzero(self.inner)
        pairs = []
        for slot in range(self.neighbors.shape[1]):
            j = self.neighbors[inner, slot]
            ok = j >= 0
            i, j = inner[ok], j[ok]
            ok = (j > i) & self.inner[j]
            i, j = i[ok], j[ok]
            swap = rank[i] > rank[j]
            pairs.append(np.stack([np.where(swap, j, i), np.where(swap, i, j)], axis=1))
        self.edges = np.concatenate(pairs).reshape(-1, 2) if pairs else np.zeros((0, 2), dtype=np.int64)

    @property
    def slopes(self):
        """How steeply the surface falls away from each point (Topology.key_slope()), worked out when first asked for
        (a mesh's 3D parts; a sphere's have none) and kept"""
        if callable(self._slopes):
            self._slopes = self._slopes()
        return self._slopes

    @staticmethod
    def reach(top, M, center, radius):
        """The radius in the level's own units (radius times the mean step from center to its neighbors) and the
        reach a point has to be within to be a neighbor of one inside it"""
        steps = []
        for _key, f, (I0, J0), (I1, J1) in top.neighbors(center, M):
            step = top.chart.grid_offsets(f, I0, J0, M, np.array(float(I1 - I0)), np.array(float(J1 - J0)))
            steps.append(float(np.linalg.norm(step)))
        r_in = radius * sum(steps) / len(steps)
        return r_in, r_in + 2.0 * float(top.sigma_hi.max())

    def from_level(self, top, center, radius):
        whole = top.level_lattice(self.M)
        r_in, r_out = self.reach(top, self.M, center, radius)
        c = whole.index[center]
        places = (whole.places - whole.places[c]) * self.M
        rows = np.flatnonzero(np.linalg.norm(places, axis=1) <= r_out)
        self.keys = [whole.keys[r] for r in rows]
        self.places = places[rows]
        self.normals = whole.normals[rows]
        self._slopes = lambda: whole.slopes[rows]
        new = np.full(len(whole.keys) + 1, -1, dtype=np.int64)
        new[rows] = np.arange(len(rows))
        # -1 (none) indexes the extra -1 at the end
        self.neighbors = new[whole.neighbors[rows]]
        self.center_row = int(new[c])
        self.inner = np.linalg.norm(self.places, axis=1) <= r_in
        return whole.rank[rows]

    def by_faces(self, top, center, radius):
        M = self.M
        inv_m = 1 / M
        r_in, r_out = self.reach(top, M, center, radius)
        keys, places, normals = [], [], []
        index = {}
        grids = []                    # (rows grid, whether each of its points is inside the face, its sides)
        queue = [(f, I, J, np.zeros(3)) for f, I, J in top.params(center, M)]
        done = set()
        while queue:
            f, IR, JR, PR = queue.pop(0)
            if f in done:
                continue
            background.give_way()
            done.add(f)
            k = top._sides[f]
            _D, gi, gj, pos, rem = face_grid(top, f, IR, JR, M, PR, r_out, 4 * int(math.ceil(radius)) + 16)
            if not gi.size:
                continue
            dI, dJ = gi[:, 0], gj[0, :]
            keep = np.linalg.norm(pos, axis=-1) <= r_out
            rows = np.full(gi.shape, -1, dtype=np.int64)
            # Inside the face: its own keys, no others' (vectorized); on its sides, the shared keys. A triangle's
            # points are those with I + J <= M, its long side where that's M.
            inner = np.ones(gi.shape, dtype=bool)
            inner[0, :] &= IR + int(dI[0]) != 0
            inner[:, 0] &= JR + int(dJ[0]) != 0
            if k == 4:
                inner[-1, :] &= IR + int(dI[-1]) != M
                inner[:, -1] &= JR + int(dJ[-1]) != M
            else:
                rem = float(rem)
                keep &= gi + gj <= rem
                inner &= gi + gj < rem
            inside = keep & inner
            ii, jj = np.nonzero(inside)
            start = len(keys)
            keys.extend((FACE, f, IR + int(x), JR + int(y)) for x, y in zip(dI[ii], dJ[jj]))
            places.append(pos[ii, jj])
            uvals = quotient(IR, M) + dI[ii] * inv_m
            vvals = quotient(JR, M) + dJ[jj] * inv_m
            nrm = np.cross(*top.chart.jacobians(f, uvals, vvals))
            normals.append(nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-300))
            rows[ii, jj] = np.arange(start, start + len(ii))
            for kk in range(start, start + len(ii)):
                index[keys[kk]] = kk
            boundary = keep & ~inside
            for x, y in zip(*np.nonzero(boundary)):
                I, J = IR + int(dI[x]), JR + int(dJ[y])
                key = top.key(f, I, J, M)
                row = index.get(key)
                if row is None:
                    row = index[key] = len(keys)
                    keys.append(key)
                    places.append(pos[x, y][None, :])
                    normals.append(top.key_normal(key, M)[None, :])
                rows[x, y] = row
                # Across the side it's on, to the faces not reached yet, from this point
                for s in sides_on(k, I, J, M):
                    g = top.across(f, s)
                    if g is not None and g not in done:
                        Ig, Jg = top.param_in(key, M, g)
                        queue.append((g, Ig, Jg, pos[x, y].copy()))
            grids.append((rows, inner, k))
        self.keys = keys
        self.places = np.concatenate(places) if places else np.zeros((0, 3))
        self.normals = np.concatenate(normals) if normals else np.zeros((0, 3))
        count = len(keys)
        # How steeply the surface falls away from each (Topology.key_slope()): only on a face's sides
        self._slopes = lambda: np.array([top.key_slope(key, M) if key[0] != FACE else 0.0 for key in keys])
        self.center_row = index[center]
        self.inner = np.linalg.norm(self.places, axis=1) <= r_in
        # Neighbors: inside a face from its grid, shifted along its lattice's steps; a shared point's (on an edge or
        # a corner) by key
        shared = {row: top.neighbors(key, M) for row, key in enumerate(keys) if key[0] != FACE}
        width = max([len(STEPS[k]) for _rows, _inner, k in grids] + [len(nbs) for nbs in shared.values()] + [4])
        neighbors = np.full((count, width), -1, dtype=np.int64)
        for rows, inner, k in grids:
            padded = np.full((rows.shape[0] + 2, rows.shape[1] + 2), -1, dtype=np.int64)
            padded[1:-1, 1:-1] = rows
            mask = (rows >= 0) & inner
            for slot, (sx, sy) in enumerate(STEPS[k]):
                shifted = padded[1 + sx:padded.shape[0] - 1 + sx, 1 + sy:padded.shape[1] - 1 + sy]
                neighbors[rows[mask], slot] = shifted[mask]
        for row, nbs in shared.items():
            for slot, (nb, *_rest) in enumerate(nbs):
                neighbors[row, slot] = index.get(nb, -1)
        self.neighbors = neighbors
        rank = np.empty(count, dtype=np.int64)
        rank[sorted(range(count), key=keys.__getitem__)] = np.arange(count)
        return rank


class MeshLocalPieces(parts.LaidOut):
    """A level's pieces around a point of it, in the level's own units (the mesh's times M, its cells across a face, from
    the point): what doesn't depend on the map's anchor, kept for every point looked at (parts._local)"""

    def __init__(self, top, M, center, radius, settings, shape):
        lattice = self.lattice = LocalLattice(top, M, center, radius)
        self.lay_out(lattice.inner, lattice.neighbors, lattice.edges, lattice.center_row, lattice.places,
                     lattice.normals, settings, shape, lattice.slopes)
        self.keys = lattice.keys


class MeshPieces:
    """A level's pieces around center (a key of the level's lattice) out to radius cells: placed in the level's own
    units from center (MeshLocalPieces), each `scale` of the map's units (rho k^(n - level)), and center itself
    origin() in the map, exactly from the anchor"""

    def __init__(self, world, level, center, radius, settings, shape):
        self.world, self.level, self.center = world, level, center
        self.M = M = world.across(level)
        key = ("mesh", id(world.topology), M, center, radius, parts._has(settings, "vertex"),
               parts._has(settings, "edge"), tuple(shape.heights))
        local = parts._local.get(key)
        if local is None:
            if len(parts._local) > parts.LOCAL_KEPT:
                parts._local.clear()
            local = parts._local[key] = MeshLocalPieces(world.topology, M, center, radius, settings, shape)
        self.__dict__.update({k: v for k, v in local.__dict__.items()})
        self.local = local
        self._origin = None
        self._point = None

    def piece_rows(self, row):
        return self.local.piece_rows(row)

    def scale(self, rho):
        return rho * power(self.world.k, self.world.n - self.level)

    def center_point(self):
        if self._point is None:
            self._point = self.world.topology.key_point(self.center, self.M)
        return self._point

    def origin(self, placer=None):
        """center in the map (map units from its origin), rounded once from its exact place"""
        if self._origin is None:
            self._origin = self.world.rho * self.world.units(self.center_point())
        return self._origin

    def camera(self, camera, placer=None):
        """A camera (map units, float64) in the pieces' units from center, from its exact place: (that, 0.0)"""
        world = self.world
        point = world.camera_point(np.asarray(camera, dtype=float) / world.rho)
        return d_float(d_sub(point, self.center_point()), self.M), 0.0


def cell_sizes(world, levels):
    """Each level's cells at the anchor (map units): its face's Jacobian there, the geometric mean of its columns, a
    level's cells k^(n - level) units"""
    top = world.topology
    a, b, _w = top.frame(world.face, world.U, world.V, world.N)
    unit = world.rho * math.sqrt(float(np.linalg.norm(a)) * float(np.linalg.norm(b)))
    return {level: unit * power(world.k, world.n - level) for level in levels}


def vertex_size(top, key, M):
    """A vertex piece's size on a key of a level of M cells across: the mean distance to its neighbors, in the level's
    own units (the mesh's times M), from their exact differences (as LaidOut sizes them, from places)"""
    P = top.key_point(key, M)
    gaps = [np.linalg.norm(d_float(d_sub(top.key_point(nb, M), P), M)) for nb, *_ in top.neighbors(key, M)]
    return float(np.mean(gaps))


def exact_piece(world, level, kind, keys, camera_point, window, shape, settings, phases):
    """A giant level's piece, the one of a kind on lattice keys (a vertex piece's vertex then its neighbors, an edge
    piece's two vertices, the smaller key first) worked out for the window: its frame and size from the keys' places
    in the level's own units (their exact differences, rounded once, so the same whatever the anchor), its place and
    the camera's exactly, its faces cut to the window in Fractions (a cube 2 window / size of its own units across,
    around the camera), each corner then from the camera in map units (parts.exact_piece()'s, for flat faces). None if
    no face reaches the window."""
    top = world.topology
    M = world.across(level)
    points = [top.key_point(k, M) for k in keys]
    P = points[0]
    # The frame from the keys' places in the level's own units from P, and their normals (parts.piece_axes(): as the
    # instanced pieces are laid out)
    offsets = [tuple(d_float(d_sub(p, P), M).tolist()) for p in points]
    normals = [tuple(top.key_normal(k, M).tolist()) for k in keys]
    stretch = parts.foot_stretch(shape, top.key_slope(keys[0], M)) if kind == 0 else None
    # An edge piece's ends at their vertex pieces' scales: each one's size (the mean distance to its neighbors) over
    # the edge's length
    scales = None
    if kind == 1:
        length = float(np.linalg.norm(offsets[1]))
        scales = tuple(vertex_size(top, k, M) / length for k in keys)
    return exact_piece_at(world, level, kind, offsets, normals, stretch, scales, d_fractions(d_sub(camera_point, P)),
                          Fraction(1, M), window, shape, settings, parts.level_phase(level, phases))


def exact_piece_at(world, level, kind, offsets, normals, stretch, scales, relative, unit, window, shape, settings,
                   phase, scale=None):
    """exact_piece()'s piece from what it's worked out of, for any lattice on the mesh's faces (a Penrose map's too,
    parts_substitution.py): the offsets of the points it stands on from the first, P, in the level's own units (floats), their
    normals, a vertex piece's stretch and an edge piece's scales (parts.piece_axes()), the camera's exact offset from P
    (relative: three Fractions, the mesh's units) and the level's unit in the mesh's (unit, a Fraction: 1 / M, or a
    Penrose level's phi^-d to as many bits as the depth needs), in the colors of phase (parts.level_phase()). With
    scale (a level unit in map units) instead of unit, relative is the camera's offset from P in the level's own units,
    floats, and it's all worked out in float64, as parts.add_levels() does where that's exact to well under the drone's
    height (parts.EXACT_SHARE): with Fractions, some 7 ms a piece, and a fast climb's hundreds of pieces took seconds
    a tick (2026-11-19)."""
    place, size, axes = parts.piece_axes(kind, offsets, normals, shape.meet, stretch, shape.heights[0], scales)
    frame = np.array([[axes[c][i] for c in range(3)] for i in range(3)])
    # The camera in the piece's own units, exactly: frame^-1 (camera - place) 2^level / size. Its exact inverse, not
    # its transpose (an edge piece's frame isn't a rotation, and a vertex piece's of floats is orthonormal only to a
    # float's step, and the camera's way back through it then misses by that share of the way from P, a cell, which a
    # hundred loops down is far past the drone's height).
    if scale is None:
        number = Fraction
        relative = [r / unit - Fraction(o) for r, o in zip(relative, place)]
        own = Fraction(size) * unit                # the piece's unit, in the mesh's
        size_map = own * Fraction(world.rho) * world.N
    else:
        number = float
        relative = [float(r) - float(o) for r, o in zip(relative, place)]
        size_map = float(size) * scale
    inverse = parts.inverse_3x3([[number(float(frame[r, c])) for c in range(3)] for r in range(3)])
    u = [sum(inverse[c][r] * relative[r] for r in range(3)) / number(size) for c in range(3)]
    half = number(window) / size_map
    corners = parts.window_faces(shape, kind, u, half, size_map, number)
    if not corners:
        return None
    u_float = np.array([float(c) for c in u])
    distance = min(prism.distance_one(u_float.tolist()) for prism in shape.prisms[kind])
    faces = [face @ frame.T for face, _ in corners]
    colors = [parts.piece_color(settings["pieces"][index], phase) for _, index in corners]
    return parts.ExactPiece(level, kind, frame, float(size_map), tuple(u_float), None, faces, colors,
                            float(distance) * float(size_map))


# A giant level is ruled in or out from where the drone is in its cells while the window's reach is under this share of
# a cell; past it (the finest giant levels) its pieces near the drone are placed in float64 to see
QUICK_LIMIT = 0.45


def cell_at(k, u, v, M):
    """The cell of a face of k sides' lattice of M cells across that (u, v) is in (in cells, Fractions or floats), with how far (u, v)
    is from each of its corners and sides in cells of its parameters: ([((I, J), distance)], [((I, J), (I, J),
    distance)]), or None off the face's cells"""
    I0, J0 = min(max(math.floor(u), 0), M - 1), min(max(math.floor(v), 0), M - 1)
    fa, fb = float(u - I0), float(v - J0)
    if not (0.0 <= fa <= 1.0 and 0.0 <= fb <= 1.0):
        return None
    if k == 4:
        corners = [((I0 + x, J0 + y), math.hypot(fa - x, fb - y)) for x in (0, 1) for y in (0, 1)]
        sides = (((I0, J0), (I0 + 1, J0), fb), ((I0 + 1, J0), (I0 + 1, J0 + 1), 1.0 - fa),
                 ((I0, J0 + 1), (I0 + 1, J0 + 1), 1.0 - fb), ((I0, J0), (I0, J0 + 1), fa))
        return corners, sides
    # A triangle's: pointing out from (I0, J0), or back, past its diagonal (none past the long side)
    if fa + fb <= 1.0 or I0 + J0 + 2 > M:
        if I0 + J0 + 1 > M or fa + fb > 1.0 + 1e-9:
            return None
        a, b, c = (I0, J0), (I0 + 1, J0), (I0, J0 + 1)
        corners = [(a, math.hypot(fa, fb)), (b, math.hypot(fa - 1.0, fb)), (c, math.hypot(fa, fb - 1.0))]
        sides = ((a, b, fb), (b, c, (1.0 - fa - fb) / math.sqrt(2.0)), (c, a, fa))
        return corners, sides
    a, b, c = (I0 + 1, J0), (I0 + 1, J0 + 1), (I0, J0 + 1)
    corners = [(a, math.hypot(fa - 1.0, fb)), (b, math.hypot(fa - 1.0, fb - 1.0)), (c, math.hypot(fa, fb - 1.0))]
    sides = ((a, b, 1.0 - fa), (b, c, 1.0 - fb), (c, a, (fa + fb - 1.0) / math.sqrt(2.0)))
    return corners, sides


def add_levels(world, camera, q, h, window, levels, settings, shape, phases, placer=None):
    """The giant levels' pieces within window of the camera (map units), as parts.ExactPieces, like parts.add_levels():
    - where the point below the drone is in the level's cells of its face, against how far each kind of piece reaches,
      the face's Jacobian bounding how far a cell is: far from every piece, or deep inside a vertex piece, the level
      has nothing in the window; else the pieces on the cell's corners and sides that might reach it
    - once the window is a fair share of a cell, the level's pieces around the vertex nearest the drone, placed in
      float64 (MeshPieces), the camera against each one's solid
    - and those worked out again exactly (exact_piece()) and cut to the window"""
    found = []
    if not levels:
        return found
    top = world.topology
    below = Below(world, q)
    f = below.face
    camera_point = world.camera_point(np.asarray(camera, dtype=float) / world.rho)
    has_vertex, has_edge = bool(shape.prisms[0]), bool(shape.prisms[1])
    reach_v, reach_e = shape.plan[0], shape.plan[1]
    core = shape.core[0]
    sigma_lo, sigma_hi = float(top.sigma_lo[f]), float(top.sigma_hi[f])
    done = set()

    def add(kind, keys, level):
        if (level, kind, keys) in done:
            return
        done.add((level, kind, keys))
        background.give_way()
        piece = exact_piece(world, level, kind, keys, camera_point, window, shape, settings, phases)
        if piece is not None:
            found.append(piece)

    def vertex_keys(key, M):
        return (key,) + tuple(nb for nb, *_rest in top.neighbors(key, M))

    # How far a step of the face's parameters moves its point, at most, against at least (Topology.sigma_lo, sigma_hi)
    spread = sigma_hi / sigma_lo if sigma_lo > 0.0 else math.inf
    for level in reversed(levels):
        M = world.across(level)
        # The window and the drone's height in the level's cells (the least a cell is across here, so at most): all
        # in cells, since a coarse level's cells deep down are past a float's range (3^1000 units), and in map units
        # the test was inf against inf
        per_cell = power(world.k, level - world.n) / (world.rho * sigma_lo) if sigma_lo > 0.0 else math.inf
        w, hh = window * per_cell, h * per_cell
        reach = w + max(reach_v, reach_e, shape.past) * spread
        u, v = below.cell_place(level)
        cell = cell_at(top._sides[f], u, v, M) if reach <= QUICK_LIMIT else None
        if cell is not None:
            # Edge pieces start above the floor (an arch's springing): a drone lower than that by the window, none
            edges = has_edge and not shape.floor[1] > hh + w
            corners, sides = cell
            for (I, J), d in corners:
                if d > reach:
                    continue
                key = top.key(f, I, J, M)
                deep = (core is not None and d * spread + w < core[0] and core[1] + w < 0.0 and hh + w < core[2])
                if has_vertex and not deep:
                    add(0, vertex_keys(key, M), level)
                if edges:
                    for nb, *_rest in top.neighbors(key, M):
                        add(1, tuple(sorted((key, nb))), level)
            if edges:
                for p0, p1, d in sides:
                    if d <= w + reach_e * spread:
                        add(1, tuple(sorted((top.key(f, *p0, M), top.key(f, *p1, M)))), level)
            continue
        # The window a fair share of a cell (or the point below off its face's cells): every piece near the vertex
        # nearest it, placed in float64, against the camera
        center = top.key(f, *nearest_point(top._sides[f], u, v, M), M)
        radius = int(min(math.ceil(1.5 + 1.25 * reach + 1.0), 64))
        pieces = MeshPieces(world, level, center, radius, settings, shape)
        local, rounding = pieces.camera(camera)
        rows, _distances, _tolerance = parts.window_rows(shape, pieces, local, rounding, pieces.scale(world.rho), window)
        for row in rows:
            rows = pieces.piece_rows(int(row))
            keys = tuple(pieces.keys[r] for r in rows)
            kind = int(pieces.kinds[row])
            if kind == 1:
                keys = tuple(sorted(keys))
            add(kind, keys, level)
    return found


class MeshLattice:
    """parts.py's view of a surface map's lattice (see parts_sphere.SphereLattice): one per factor, its ratio, how many
    times smaller a level's cells are than the next coarser's"""

    def __init__(self, ratio=2.0):
        self.ratio = ratio

    def world(self, obj):
        return world_of(obj)

    def unit_level(self, world):
        return world.n + world.level_shift

    def camera(self, obj, world, camera_world):
        camera = map_point(obj, camera_world)
        below = below_of(obj, world, camera)
        return camera, camera, max(abs(below.height) * world.rho, 1e-300)

    world_vector = staticmethod(world_vector)

    def placer(self, world, h):
        return None

    def below(self, obj, world, q):
        return below_of(obj, world, q)

    def center(self, world, below, level):
        return below.lattice_point(level)[0]

    def ground(self, obj, world, camera, q):
        """What a level's center is measured from: the point of the surface below the camera (map units). Not the
        camera itself: the faces turn by up to 33 degrees at their folds, and over a fold (or off to the side of a
        center whose normal is a fold's mean) the camera's offset from the center along the center's surface was a
        share of its height, many of a fine level's cells, so those levels were built again every tick."""
        return world.rho * below_of(obj, world, q).foot

    def first_radius(self, world, level, build, wanted):
        """How far the coarsest level is built, in its cells, wanted to reach the window: nothing coarser goes on past
        it. A level kept whole (is_whole()) is built whole, once, and stays as the drone moves; a finer one, out to
        the next power of 2 of the cells wanted (so it's rarely built again for a new radius), at most FIRST_REACH."""
        if is_whole(world.topology, world.across(level)):
            return math.inf
        if not wanted > build:
            return build
        return float(min(2 ** math.ceil(math.log2(min(wanted, 2.0 ** 30))), max(FIRST_REACH, build)))

    def pieces(self, world, level, center, radius, settings, shape):
        return MeshPieces(world, level, center, radius, settings, shape)

    def pieces_job(self, world, level, center, radius, settings, shape):
        """A level's build for the worker thread (parts.SphereLattice.pieces_job()); None: built here. A curved chart's
        (a subdivision surface's) on the worker: 0.2 to 0.6 s a level on quadico's, built here they froze the flight.
        A flat one's here: a few ms each, and on the worker, in a dive at a plane (some 46 a second, beside the giant
        levels' window builds), the finest level was missing in 31% of the frames, against 3.5% (2026-11-24)."""
        if world.topology.chart.flat_cells:
            return None
        own = World.from_text(world.text(), world.rho, world.topology, world.k)
        return built_mesh_pieces, own, level, center, radius, settings, shape

    def cell_sizes(self, world, levels):
        return cell_sizes(world, levels)

    def transform(self, world, text, built):
        return transform(world, text, built)

    def add_levels(self, world, camera, q, h, window, levels, settings, shape, phases, placer):
        return add_levels(world, camera, q, h, window, levels, settings, shape, phases, placer)

    def add_levels_job(self, world, camera, q, h, window, levels, settings, shape, phases, placer):
        """add_levels() for the worker thread (parts.settle_exact()), on a world of its own: (function, args)"""
        own = World.from_text(world.text(), world.rho, world.topology, world.k)
        return levels_pieces, (own, np.array(camera, dtype=float), np.array(q, dtype=float), h, window, list(levels),
                               settings, shape, phases)


def levels_pieces(world, camera, q, h, window, levels, settings, shape, phases):
    """MeshLattice.add_levels_job()'s pieces: (add_levels(), nothing for the main thread to do then)"""
    return add_levels(world, camera, q, h, window, levels, settings, shape, phases), None


def built_mesh_pieces(world, level, center, radius, settings, shape):
    """A level's pieces (MeshPieces) built on the worker thread, as parts.write_level() takes them"""
    pieces = MeshPieces(world, level, center, radius, settings, shape)
    return world, pieces, pieces.origin(), radius


_lattices = {}       # factor -> MeshLattice


def lattice_of(obj):
    """parts.py's view of the map's lattice, for its factor"""
    k = factor(obj)
    found = _lattices.get(k)
    if found is None:
        found = _lattices[k] = MeshLattice(float(k))
    return found


def forget(_dummy=None):
    """After loading a file: another file's floors, worlds and meshes don't apply"""
    _floors.clear()
    _worlds.clear()
    _below.clear()
    _topologies.clear()
    import sys
    substitution_module = sys.modules.get(__name__.rpartition(".")[0] + ".surface_substitution")
    if substitution_module is not None:
        substitution_module.forget()


def register():
    import bpy
    handler = bpy.app.handlers.persistent(forget)
    if not any(getattr(h, "__name__", "") == "forget" and getattr(h, "__module__", "") == __name__
               for h in bpy.app.handlers.load_post):
        bpy.app.handlers.load_post.append(handler)


def unregister():
    import bpy
    for h in list(bpy.app.handlers.load_post):
        if getattr(h, "__name__", "") == "forget" and getattr(h, "__module__", "") == __name__:
            bpy.app.handlers.load_post.remove(h)
