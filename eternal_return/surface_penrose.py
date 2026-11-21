"""Endless Penrose on any surface (2026-11-13, step 4 of tiling any surface: "i want a general way of tiling a plane
with penrose tiles regardless of the curvature of the plane, just like with square and triangle tilings").

A surface map (surface.py) whose settings say "rule": "penrose" has a rhomb mesh as its source: every face a Penrose
rhomb, its loop (B, A, C, A') counterclockwise from outside, and its kind (0 thin, 1 thick) in the face attribute
"rhomb_kind". The two Robinson halves (A, B, C) and (A', B, C) split by the substitution forever (penrose.py), in the
face's ideal rhomb (B = 0, A = 1, A' at 144 degrees for a thin one, 72 for a thick one), and each point goes onto the
face's bilinear patch at its ideal (u, v): the way surface.py puts a square's lattice on any quad. A half's leg AB is
split and its leg AC turns into the next level's diagonals, so the faces' labels must agree at every edge (the matching
rule): then each level's points on an edge are the same from both faces, and the tiling runs on across it at every
depth. Corners whose ideal angles don't add up to 360 degrees are the surface's special points, like a square mesh's
odd corners. Any polygon mesh can be labeled so (a thick rhomb per face corner: B at the vertex, C at the face's middle,
A and A' at its sides' middles), and the conformal Penrose sphere and torus are rhomb meshes whose faces are true
Penrose rhombs (~/blender_tests/penrose_surface, the build scripts and checks; ~/blender_tests/penrose_torus).

Everything about where the drone is stays surface.py's: its world (n and the exact anchor A), the point below, auto
scale, loops (which only change units) and the floor, drawn dark under the tiles. What's drawn here is each level's
strapwork, like substitution.py's endless girih floors (its Motif: Hankin lines at 54 degrees by default, each bar cut
to its own half, so two halves' bars never overlap in one plane), on the surface:
- Level d's tiles are the faces' halves split d times, edges phi^-d of a face's. The drawn levels are those whose bars
  are between min_bar_px and the fade band's far end wide under the drone (fade(), substitution.py's), each within
  min(sight, radius tiles) of the point below, in its own object, finer levels on top by the push layering (drivers
  scale each level about the drone camera by 1 + its push), the floor under them all.
- Each level's tiles near the drone are kept between ticks (Level), found from the level above's (its tiles' children,
  those within reach), for a window a little wider than wanted, until the drone leaves it. Level 0's are the faces'
  halves, found by walking out from the face below.
- Exact at any depth: a tile is penrose.py's (kind, rot, mirror, tau), tau in Z[zeta_10] in its level's units, so a
  thousand loops deep its numbers have a thousand bits. Per level and face one root point is placed exactly (Q(sqrt 5)
  over powers of 2: the face's float32 corners are dyadic, and the ideal rhomb's coordinates are in Q(sqrt 5)) from the
  anchor and rounded once; every other tile of that level and face is placed from the root by its small exact difference
  in floats, through the patch's (affine plus twist) map.
- A level's color is the arch palette's in turn (palette[d mod 4]), the same from every face: rotation-based colors
  (substitution.py's classes) would change across a fold, since each face has its own frame.

The map's object holds "surface_penrose": its settings (DEFAULTS); the level objects are its children, made as needed
(LEVEL_KEY names the map), with "substitution_pool" so the flight harness treats them as substitution levels (rebuilt
around the drone, not moved).
"""

import functools
import math
import threading
import time
from fractions import Fraction

import numpy as np

from . import background
from . import penrose
from . import schedule
from . import substitution
from . import surface
from . import tiling
from .cyclotomic import RING10 as RING, float_sqrt

PROPERTY = "surface_penrose"
KIND = "rhomb_kind"
LEVEL_KEY = "surface_penrose_level_of"
POOL = "substitution_pool"
SQRT5 = math.sqrt(5.0)
PHI = (1.0 + SQRT5) / 2.0
LOG_PHI = math.log(PHI)

# Settings ("surface_penrose" on the map's object):
# - style, half_width: the strapwork (substitution.Motif: "hankin54", or another contact angle), its bars' half width
#   in tiles
# - radius: a level is drawn within this many of its tiles of the point below (and at most sight: window clip ends)
# - min_bar_px, focal: the finest level drawn has bars this wide under the drone, on a screen focal px per unit of
#   distance
# - fade: [in from, in to] px: the finest levels fade in from under a pixel (substitution.fade()); none ever fades out
# - outline: the share of a bar's width that's outline_color on each side (substitution.outlined()), 0 for none
# - outline_color: linear RGBA, the floor's if None: diving into an outline comes back to the dark floor
# - palette: the levels' colors in turn (linear RGBA)
DEFAULTS = {"style": "hankin54", "half_width": 0.007, "radius": 12.0, "min_bar_px": 0.3, "focal": 1000.0,
            "fade": [0.3, 1.5], "outline": 0.25, "outline_color": None,
            "palette": [[0.03, 0.10, 0.45, 1.0], [0.0, 0.42, 0.40, 1.0], [0.95, 0.55, 0.06, 1.0],
                        [0.80, 0.16, 0.22, 1.0]]}
# A level's tiles are kept for a window this much wider than wanted, plus this many of its tiles
SLACK, SLACK_TILES = 1.5, 2.0
# A level object is built for a window this much wider than its level's, and again once the view reaches past it
BUILD_SLACK = 1.4
# With builds on the worker thread, a level is drawn as it was built until its new build is written, some 50 to 300 ms
# later: by then the drone has flown on by its speed times that. So a build is asked for once the point below, that
# far ahead, would see past the level's edge, and is built that much wider (at least twice it) and centered ahead of
# the drone along its flight (by the wait and half its slack beyond). Asked for only once the drone was past the edge,
# every build came late, and the finest levels, a fifth of a drone height wide, were drawn short of it in three
# quarters of the ticks at a cruise of a height a second (2026-11-17).
# The drone's speed over the surface and the builds' wait: the map's schedule.Motion, shared with its 3D parts.
# 2026-11-19: the build's slack grew with the speed (twice the flight in the wait, up to 1.5 windows) and its center
# went at most that slack ahead, so flying faster than 1.5 windows in a wait (the user's low glides: hundreds of the
# finest levels' tiles a second), every build was behind the drone when it was drawn: rebuilt again and again, drawn
# short (bare floor), and its kept tiles six windows across, cut in the tick (a quarter of each frame, 13 to 18 frames a
# second). Now a build is centered where the drone will be when it's drawn, plus its slack along the flight, its size
# the same at any speed: faster flight asks for more builds, not bigger ones.
# The worker threads' priority: what a build is worth (schedule.value(), by the same rule as the 3D parts'): the share
# of the view it puts right (its level's disc, worth()), how much of its colors it shows there (its fade), both as they
# will be when the build lands, and how soon it's wanted, for its cost. Before 2026-11-20 a table: a level short of the
# drone first, the coarser first (but a level not drawn at all, the finest first), then one built again before it's
# short, then the next finer one built ahead, each tuned against the last failure (2026-11-19: a fast glide kept the
# workers on bars a pixel wide while the levels between were short; a fast dive built each new level just as it turned
# coarse, the view "a solid color").
# Rated as it is when asked for, and its bars' width counted again on top of its fade, a new fine level in a fast dive
# was worth a hundredth of the next 3D parts build; it landed half shown or more after it was wanted, the floor near
# the drone plain, in two ticks of three (2026-11-20, the user: "the floor is a solid color").
# How much of the view round them a level's bars fill, against the 3D parts' pieces (parts.INK): the same, both are
# what's missing where they'd be (0.25 until 2026-11-20)
INK = 1.0
# A floor build's cost until one's measured (s): some 20 ms on a worker (2026-11-19)
COST_FIRST = 0.02
# A build still waiting is handed in again round its new middle once that's moved this share of its reach
FRESH = 0.25
# At most this many tiles in a kept level (the rest dropped, the furthest first): a guard, not a design limit
MAX_TILES = 60000
PUSH_STEP = 100000.0 / 2 ** 24
# Levels whose tiles are small enough that radius of them fit in the sight are "plain": drawn within radius of their
# tiles. Bigger ones are "coarse": drawn out to the sight, cut to a box round the drone (their bars reach far past
# it), and the first whose bar holds the whole box covers everything coarser (coarse_level()). (Until 2026-11-14 a level
# was coarse from bars 200 px wide, still thousands of tiles in its window, all of them cut to a box that cut none.)
# Levels whose tiles are more than this many of their windows across are "giant": never drawn (their places, from a
# root rounded once, are only good to ~1e-16 of their size), only the floor tinted by the bar the drone is in, if any
# (giant_tint())
GIANT = 1e6
# A coarse level is re-rooted at a tile near the drone once its tiles are further than this from their root (its
# places are good to ~1e-16 of the distance to the root)
COARSE_COEF = 16
# Tiles whose outline strips are at least this wide where they are (px, seen square on) are drawn outlined, the rest
# plain (level_bars()); a level with outlines near the drone is built again once the camera has moved half that reach
OUTLINE_PX = 2.0


def setting(obj, name):
    settings = obj.get(PROPERTY)
    value = settings.get(name) if settings is not None else None
    return DEFAULTS[name] if value is None else value


def is_penrose(obj):
    settings = obj.get(surface.PROPERTY)
    return settings is not None and settings.get("rule") == "penrose"


# ----- exact numbers of Q(sqrt 5): (a, b) Fractions, a + b sqrt 5 -----

def q_add(x, y):
    return x[0] + y[0], x[1] + y[1]


def q_sub(x, y):
    return x[0] - y[0], x[1] - y[1]


def q_mul(x, y):
    return x[0] * y[0] + 5 * x[1] * y[1], x[0] * y[1] + x[1] * y[0]


def q_scale(x, k):
    return x[0] * k, x[1] * k


Q_PHI_INV = (Fraction(-1, 2), Fraction(1, 2))          # 1 / phi = (sqrt 5 - 1) / 2
Q_COS = {penrose.THIN: (Fraction(-1, 4), Fraction(-1, 4)),     # cos 144 = -(1 + sqrt 5) / 4
         penrose.THICK: (Fraction(-1, 4), Fraction(1, 4))}     # cos 72 = (sqrt 5 - 1) / 4
_phi_inv_powers = {0: (Fraction(1), Fraction(0))}


def q_phi_inv(d):
    """phi^-d, exactly"""
    found = _phi_inv_powers.get(d)
    if found is None:
        nearest = max(k for k in list(_phi_inv_powers) if k <= d)
        found = _phi_inv_powers[nearest]
        for k in range(nearest + 1, d + 1):
            found = q_mul(found, Q_PHI_INV)
            if k % 16 == 0 or k == d:
                _phi_inv_powers[k] = found
    return found


@functools.lru_cache(maxsize=4096)
def phi_power_float(d, shift=0):
    """2^shift phi^-d as a float, accurate to the last bit however big d and shift are (each alone overflows past
    1000 or so): phi^d = (L + F sqrt 5) / 2, Lucas and Fibonacci numbers, in whole numbers. Kept: a settle asks for
    the same few dozen some 200 times (1 ms of every tick deep down, 2026-11-14)."""
    a, b = 2, 0                          # 2 phi^k = a + b sqrt 5
    x, y, k = 1, 1, d                    # 2 phi = 1 + sqrt 5, squared as we go
    while k:
        if k & 1:
            a, b = (a * x + 5 * b * y) // 2, (a * y + b * x) // 2
        x, y = (x * x + 5 * y * y) // 2, (x * y + y * x) // 2
        k >>= 1
    # 2 phi^d = a + b sqrt 5 > 0; its float as mantissa and exponent
    bits = 64
    shift_bits = bits + 2 * max(a.bit_length(), b.bit_length())
    big = (a << shift_bits) + math.isqrt(5 * b * b << (2 * shift_bits))          # (2 phi^d) 2^shift_bits
    e = big.bit_length()
    mantissa = (big >> max(e - 64, 0)) / 2.0 ** min(e, 64)                     # in [0.5, 1)
    exponent = e - shift_bits - 1                                              # phi^d = mantissa 2^exponent
    return math.ldexp(1.0 / mantissa, shift - exponent)


# ----- each face's ideal rhomb -----

class Faces:
    """A rhomb mesh's labels: each face's kind, its ideal rhomb's A' direction (cos, sin of 144 or 72 degrees), and its
    two halves at level 0 as penrose.py tiles (in the face's own units: legs 1)"""

    def __init__(self, top, mesh):
        attribute = mesh.attributes.get(KIND)
        if attribute is None or attribute.domain != 'FACE':
            raise ValueError(f"surface_penrose: {mesh.name} has no face attribute {KIND}")
        kinds = np.empty(len(mesh.polygons), dtype=np.int32)
        attribute.data.foreach_get("value", kinds)
        if np.any(top.sides != 4):
            raise ValueError(f"surface_penrose: {mesh.name} isn't all quads")
        self.kind = kinds
        angle = np.where(kinds == penrose.THIN, math.radians(144.0), math.radians(72.0))
        self.cos, self.sin = np.cos(angle), np.sin(angle)
        e2 = {penrose.THIN: RING.power(4), penrose.THICK: RING.power(2)}
        # How far a step in the ideal rhomb moves the face's point at least (its Jacobian from ideal coordinates, (u, v)'s
        # times the inverse of (u, v) -> u + v e2, on a 5 x 5 grid, its least singular value, less a tenth for between)
        grid = np.linspace(0.0, 1.0, 5)
        gu, gv = (g.ravel() for g in np.meshgrid(grid, grid, indexing="ij"))
        inverse = np.zeros((len(kinds), 2, 2))
        inverse[:, 0, 0] = 1.0
        inverse[:, 0, 1] = -self.cos / self.sin
        inverse[:, 1, 1] = 1.0 / self.sin
        ju = top.eu[:, None, :] + gv[None, :, None] * top.w[:, None, :]
        jv = top.ev[:, None, :] + gu[None, :, None] * top.w[:, None, :]
        jacobian = np.stack([ju, jv], axis=-1) @ inverse[:, None]                   # (F, grid, 3, 2)
        values = np.linalg.svd(jacobian, compute_uv=False)
        self.sigma = 0.9 * values[..., -1].min(axis=1)
        self.sigma_hi = 1.1 * values[..., 0].max(axis=1)
        # Each face's corners' middle and how far its corners are from it (mesh units), for which faces a window reaches
        corners = np.stack([top.P0, top.P0 + top.eu, top.P0 + top.eu + top.ev + top.w, top.P0 + top.ev], axis=1)
        self.middle = corners.mean(axis=1)
        self.radius = np.linalg.norm(corners - self.middle[:, None], axis=2).max(axis=1)
        self.halves = []
        for f, kind in enumerate(kinds.tolist()):
            c = RING.add(RING.one, e2[kind])
            self.halves.append([penrose.solve(kind, RING.one, RING.zero, c), penrose.solve(kind, e2[kind], RING.zero, c)])

    def uv_exact(self, f, z, d):
        """A ring point z of level d (units phi^-d of the face's) as the face's (u, v), exactly: Q(sqrt 5) pairs"""
        (xa, xb), (ya, yb) = RING.exact_xy(z)
        kind = int(self.kind[f])
        v = (ya, yb) if kind == penrose.THIN else q_mul((ya, yb), Q_PHI_INV)     # sin 72 = sin 36 phi
        u = q_sub((xa, xb), q_mul(v, Q_COS[kind]))
        scale = q_phi_inv(d)
        return q_mul(u, scale), q_mul(v, scale)

    def uv_float(self, f, dx, dy):
        """Ideal offsets (floats, any units) as offsets of the face's (u, v) in the same units"""
        dv = dy / self.sin[f]
        return dx - dv * self.cos[f], dv


_faces = {}


def faces_of(obj, top):
    source = obj.get(surface.SOURCE)
    mesh = source.data
    key = (mesh.name, len(mesh.vertices), len(mesh.polygons), id(top))
    found = _faces.get(key)
    if found is None:
        found = _faces[key] = Faces(top, mesh)
    return found


def exact_offset(world, f, u, v):
    """The face's point at exact (u, v) (Q(sqrt 5) pairs) from the anchor's, in map units, as floats"""
    top = world.topology
    p0, eu, ev, tw = top.exact[f]
    uv = q_mul(u, v)
    ax, ae = world.anchor_point()
    scale = 1 << top.E
    out = []
    for k in range(3):
        coordinate = q_add(q_add((Fraction(p0[k]), Fraction(0)), q_scale(u, eu[k])),
                           q_add(q_scale(v, ev[k]), q_scale(uv, tw[k])))
        coordinate = q_sub(q_scale(coordinate, Fraction(1, scale)), (Fraction(ax[k], 1 << ae), Fraction(0)))
        out.append(float_sqrt(coordinate[0] * (1 << world.n), coordinate[1] * (1 << world.n), 5))
    return world.rho * np.array(out)


# ----- a level's tiles near the drone -----

# Each kind's half: its corners in prototile coordinates, its centroid, and its corners' furthest from it
_PROTO = {k: np.array([RING.xy_small(c) for c in penrose.PROTOTILE[k]]) for k in penrose.KINDS}
_CENTROID = {k: p.mean(axis=0) for k, p in _PROTO.items()}
_RADIUS = {k: float(np.linalg.norm(p - p.mean(axis=0), axis=1).max()) for k, p in _PROTO.items()}
RULE = substitution.RULES["penrose"]
# A kept level keeps at least the tiles within this many of its tiles of the point below (its biggest levels: their
# window is a speck in a tile, and the drone stays in what's kept for long)
MIN_KEEP = 0.25
# Shallower than this many loops, levels are placed in float64 from the mesh's own coordinates and tested by their map
# places, all faces at once (a whole object in view has thousands); deeper, exactly, a face at a time (a few are near)
FLOAT_N = 24


def side_uv(s, t):
    """The point t along a quad's side s (from its corner s), as (u, v)"""
    return ((t, 0), (1, t), (1 - t, 1), (0, 1 - t))[s]


class Centers:
    """The point below (foot, map units) as each face near it sees it, (u, v) exactly (Fractions): in its own face,
    from below; in a face across a side from one already placed, the side's point nearest the foot, exactly, moved
    to the foot along the face's patch carried past its side (a least-squares step on its Jacobian there, in floats:
    the step is small). extra is what that step leaves over (map units: the foot's height off the carried patch, ~0
    unless the surface folds there). Faces whose side's point is further than window aren't placed."""

    def __init__(self, world, faces, below, foot, window):
        top = world.topology
        n = world.n
        f0 = below.face
        Ub, Vb = below.base
        scale = Fraction(1, 1 << n)
        u0 = (Fraction(Ub) + Fraction(float(below.d[0]))) * scale
        v0 = (Fraction(Vb) + Fraction(float(below.d[1]))) * scale
        self.uv, self.extra = {f0: (u0, v0)}, {f0: 0.0}
        clamp = lambda x: min(max(x, Fraction(0)), Fraction(1))
        queue = [f0]
        while queue:
            background.give_way()
            f = queue.pop(0)
            u, v = self.uv[f]
            corners = top._corners[f]
            for s in range(4):
                g = top.across(f, s)
                if g is None or g in self.uv:
                    continue
                t = clamp((u, v, 1 - u, 1 - v)[s])
                e = top._side_edge[f][s]
                s2 = next(k for k in range(4) if top._side_edge[g][k] == e)
                same = top._corners[g][s2] == corners[s]
                ue, ve = side_uv(s2, t if same else 1 - t)
                ue, ve = Fraction(ue), Fraction(ve)
                place = exact_offset(world, g, (ue, Fraction(0)), (ve, Fraction(0)))
                gap = foot - place
                if not float(np.linalg.norm(gap)) <= window:
                    continue
                jacobian = world.rho * math.ldexp(1.0, n) * np.stack(
                    [top.eu[g] + float(ve) * top.w[g], top.ev[g] + float(ue) * top.w[g]], axis=1)
                step = np.linalg.lstsq(jacobian, gap, rcond=None)[0]
                self.uv[g] = (ue + Fraction(float(step[0])), ve + Fraction(float(step[1])))
                self.extra[g] = float(np.linalg.norm(jacobian @ step - gap))
                queue.append(g)
        # Each one's ideal point (x, and y over sin 36: both Q(sqrt 5) pairs)
        self.ideal = {}
        for f, (u, v) in self.uv.items():
            uq, vq = (Fraction(u), Fraction(0)), (Fraction(v), Fraction(0))
            kind = int(faces.kind[f])
            x = q_add(uq, q_mul(vq, Q_COS[kind]))
            y = vq if kind == penrose.THIN else q_mul(vq, (Fraction(1, 2), Fraction(1, 2)))      # v phi
            self.ideal[f] = (x, y)


_phi_powers = {0: (Fraction(1), Fraction(0))}


def q_phi(d):
    """phi^d, exactly"""
    found = _phi_powers.get(d)
    if found is None:
        nearest = max(k for k in list(_phi_powers) if k <= d)
        found = _phi_powers[nearest]
        for k in range(nearest + 1, d + 1):
            found = q_mul(found, (Fraction(1, 2), Fraction(1, 2)))
            if k % 16 == 0 or k == d:
                _phi_powers[k] = found
    return found


# A level's tiles' places are whole-number coefficients from their face's root (int64): re-rooted at a tile near the
# drone once any is bigger than this (a root stays where it is as levels split, so tiles near the drone get further
# from it in their own units by phi a level)
COEF_LIMIT = 2 ** 40


class Level:
    """Level d's tiles near where the drone was (center, map units of the world world_text, like surface.FloorBuilt),
    covering window of it (map units): each face's root, an exact ring point of the level (and once drawn, its map place
    and (u, v)), and each tile as arrays: its face, kind, rot, mirror and tau from its face's root as whole-number ring
    coefficients (coef)"""

    def __init__(self, d, world, center):
        self.d, self.world_text, self.n = d, world.text(), world.n
        self.center, self.window = np.asarray(center, dtype=float), 0.0
        self.transform = None        # for surface.transform()
        self.roots = {}              # face -> [root tau, map place or None, u_r, v_r]
        self.scale = world.rho * phi_power_float(d, world.n)       # map units per (u, v) unit of level d: rho 2^n phi^-d
        self.set_tiles(np.zeros(0, np.int64), np.zeros(0, np.int8), np.zeros(0, np.int8), np.zeros(0, bool),
                       np.zeros((0, 4), np.int64))

    def set_tiles(self, face, kind, rot, mirror, coef):
        self.face, self.kind, self.rot, self.mirror, self.coef = face, kind, rot, mirror, coef
        self.rel = np.stack([coef @ RULE.basis_x, coef @ RULE.basis_y], axis=1) if len(coef) else np.zeros((0, 2))
        self.params = None           # face_params() of every tile, until the tiles or their roots change

    def take(self, rows):
        self.set_tiles(self.face[rows], self.kind[rows], self.rot[rows], self.mirror[rows], self.coef[rows])

    def copy(self):
        """The same level for a build on another thread, which trims and re-roots what it's given: its own
        coefficients and roots (and what it's a copy of, for keep_copy())"""
        fresh = Level.__new__(Level)
        fresh.__dict__.update(self.__dict__)
        fresh.coef = self.coef.copy()
        fresh.roots = {f: list(r) for f, r in self.roots.items()}
        fresh.source = (self, self.window)
        return fresh

    def place_root(self, world, faces, f):
        """Face f's root's map place and (u, v): exactly from the anchor deep down, in float64 from the mesh's own
        coordinates shallow (FLOAT_N: then float64 places the mesh's points to far under a level-n unit)"""
        root = self.roots[f]
        if root[1] is None:
            if world.n < FLOAT_N:
                x, y = RING.xy_float(root[0])
                scale = phi_power_float(self.d)
                du, dv = faces.uv_float(f, x, y)
                ur, vr = du * scale, dv * scale
                top = world.topology
                mesh = top.P0[f] + ur * top.eu[f] + vr * top.ev[f] + ur * vr * top.w[f]
                root[1] = world.rho * math.ldexp(1.0, world.n) * (mesh - surface.d_float(world.anchor_point()))
                root[2], root[3] = ur, vr
            else:
                # Some 2 ms of Fractions a face: on a worker thread, the main thread's Python goes first
                background.give_way()
                u, v = faces.uv_exact(f, root[0], self.d)
                root[1] = exact_offset(world, f, u, v)
                root[2], root[3] = float_sqrt(u[0], u[1], 5), float_sqrt(v[0], v[1], 5)
        return root

    def face_params(self, world, faces, rows):
        """For tiles rows: each one's face's root's map place (t, 3), u_r and v_r (t,). Every tile's are worked out
        the first time and kept (a build asks some 15 times, and sorting the tiles' faces each time was a tenth of it,
        2026-11-14)."""
        if self.params is None:
            fs, index = np.unique(self.face, return_inverse=True)
            params = [self.place_root(world, faces, int(f)) for f in fs]
            place = np.array([p[1] for p in params]).reshape(-1, 3)
            ur = np.array([p[2] for p in params])
            vr = np.array([p[3] for p in params])
            self.params = (place[index], ur[index], vr[index])
        place, ur, vr = self.params
        return place[rows], ur[rows], vr[rows]

    def tile_frames(self, world, faces, rows, turn):
        """For tiles rows, turned by turn (t, 2, 2): each one's map place (t, 3) and frame G (t, 3, 2), so the point p
        of its prototile is at place + G p + bend (G p's (u, v) parts multiplied), bend (t, 3) and the (u, v) parts'
        maker B (t, 2, 2: (du, dv) = B p, level units). What map_points() works out a point at a time, taken apart
        into the tile's own share and its prototile's, so a kind's bars' corners are one product for all its tiles:
        with (du0, dv0) the tile's place from its face's root and (du, dv) a point's from the tile, the patch's
        u eu + v ev + u v w gives the point place + du (eu + (vr + s dv0) w) + dv (ev + (ur + s du0) w) + s du dv w,
        s = phi^-d."""
        top = world.topology
        face = self.face[rows]
        _place, ur, vr = self.face_params(world, faces, rows)
        small = phi_power_float(self.d)
        sin, cos = faces.sin[face], faces.cos[face]
        # Ideal (x, y) to (du, dv): dv = y / sin, du = x - dv cos
        to_uv = np.zeros((len(rows), 2, 2))
        to_uv[:, 0, 0] = 1.0
        to_uv[:, 0, 1] = -cos / sin
        to_uv[:, 1, 1] = 1.0 / sin
        rel = self.rel[rows]
        dv0 = rel[:, 1] / sin
        du0 = rel[:, 0] - dv0 * cos
        w = top.w[face]
        axes = np.stack([top.eu[face] + (vr + small * dv0)[:, None] * w,
                         top.ev[face] + (ur + small * du0)[:, None] * w], axis=2)          # (t, 3, 2)
        B = to_uv @ turn
        return self.map_points(world, faces, rows, rel), self.scale * (axes @ B), (self.scale * small) * w, B

    def map_points(self, world, faces, rows, ideal):
        """Ideal points of tiles rows (level units, from each one's face's root: (t, ..., 2)) as map points (this
        world's)"""
        top = world.topology
        face = self.face[rows]
        place, ur, vr = self.face_params(world, faces, rows)
        extra = (1,) * (ideal.ndim - 2)
        shape = lambda a: a.reshape(a.shape[:1] + extra)
        dv = ideal[..., 1] / shape(faces.sin[face])
        du = ideal[..., 0] - dv * shape(faces.cos[face])
        small = phi_power_float(self.d)
        vec = lambda a: a[face].reshape((len(face),) + extra + (3,))
        twist = shape(ur) * dv + shape(vr) * du + small * du * dv
        offset = du[..., None] * vec(top.eu) + dv[..., None] * vec(top.ev) + twist[..., None] * vec(top.w)
        return place.reshape((len(face),) + extra + (3,)) + self.scale * offset

    def reroot(self, limit=COEF_LIMIT):
        """Faces whose tiles have grown big coefficients (limit) get a tile near their middle as their root"""
        for f in list(self.roots):
            rows = np.flatnonzero(self.face == f)
            if not len(rows) or np.abs(self.coef[rows]).max() < limit:
                continue
            middle = rows[np.argmin(np.linalg.norm(self.rel[rows] - self.rel[rows].mean(axis=0), axis=1))]
            shift = self.coef[middle].copy()
            self.roots[f] = [RING.add(self.roots[f][0], tuple(int(c) for c in shift)), None, 0.0, 0.0]
            self.coef[rows] -= shift
        self.set_tiles(self.face, self.kind, self.rot, self.mirror, self.coef)

    def keep_near(self, world, faces, centers, window, center):
        """Keep the tiles whose halves come within window (map units) of the point below (center). Shallow, by their
        map places (float64 has them to far under a unit); deep, each face's in its ideal coordinates in the level's
        units, from the point below as that face sees it (exactly, once a face: Centers), which never overflows however
        big the level's tiles are in map units, then by their map places too where those are good enough"""
        if not len(self.face):
            self.window = 0.0
            return
        turn = RULE.orient[self.rot, self.mirror.astype(np.int64)]                     # (t, 2, 2)
        cents = self.rel + np.einsum("tij,tj->ti", turn, _CENTROIDS[self.kind])
        if world.n < FLOAT_N:
            rows = np.arange(len(self.face))
            places = self.map_points(world, faces, rows, cents)
            reach = _RADII[self.kind] * self.scale * faces.sigma_hi[self.face]
            with np.errstate(over="ignore", invalid="ignore"):
                away = np.linalg.norm(places - center, axis=1) - reach
                keep = away <= window
            covered = window
        else:
            keep = np.zeros(len(self.face), dtype=bool)
            away = np.full(len(self.face), math.inf)        # map units at least
            covered = math.inf
            grow = q_phi(self.d)
            for f in np.unique(self.face).tolist():
                if f not in centers.ideal:
                    continue
                background.give_way()
                rows = np.flatnonzero(self.face == f)
                (xa, xb), (ya, yb) = RING.exact_xy(self.roots[f][0])
                cx, cy = centers.ideal[f]
                dx = q_sub(q_mul(cx, grow), (xa, xb))
                dy = q_sub(q_mul(cy, grow), (ya, yb))
                here = np.array([float_sqrt(dx[0], dx[1], 5), float_sqrt(dy[0], dy[1], 5) * RING.stretch])
                per = self.scale * max(float(faces.sigma[f]), 1e-300)      # map units an ideal level unit, at least
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    margin = max(window / per if per > 0.0 else 0.0, MIN_KEEP)
                    extra = centers.extra[f] / per if per > 0.0 else 0.0
                if not math.isfinite(margin):
                    margin = math.inf
                gap = np.linalg.norm(cents[rows] - here, axis=1) - _RADII[self.kind[rows]]
                near = gap <= margin + extra
                with np.errstate(over="ignore", invalid="ignore"):
                    away[rows] = gap * per
                keep[rows[near]] = True
                if near.any():
                    covered = min(covered, margin * per)
            # A face's ideal units are at least per map units, and up to its greatest stretch: on a stretched face
            # (a Catmull-Clark icosahedron's, 3 to 1) that kept ten times the tiles the window reaches, each gone
            # through again by every level under this one and every build (2026-11-14). Of those kept, the ones whose
            # map places are out of the window's reach go too, unless the window is a speck in a tile (MIN_KEEP: a
            # big level's places are only good to a ten-thousandth of a tile).
            rows = np.flatnonzero(keep)
            if len(rows):
                size = self.scale * faces.sigma_hi[self.face[rows]]
                with np.errstate(over="ignore", invalid="ignore"):
                    places = self.map_points(world, faces, rows, cents[rows])
                    out = np.linalg.norm(places - center, axis=1) - _RADII[self.kind[rows]] * size
                    far = np.isfinite(out) & (out > np.maximum(window, MIN_KEEP * size))
                if far.any():
                    keep[rows[far]] = False
                    covered = min(covered, max(window, MIN_KEEP * float(size.min())))
        rows = np.flatnonzero(keep)
        if len(rows) > MAX_TILES:
            # A guard: the nearest kept, and the level said to cover only as far as the first one left out (by their
            # coefficients' size, it kept tiles anywhere in the window and still said it covered all of it)
            order = np.argsort(away[rows], kind="stable")
            dropped = away[rows[order[MAX_TILES]]]
            rows = rows[order[:MAX_TILES]]
            if math.isfinite(dropped):
                covered = min(covered, max(float(dropped), 0.0))
        self.take(rows)
        # What's kept is round center now: a build that trimmed a level round the drone left its old center, and the
        # level then claimed tiles it no longer had (covers())
        self.center = np.asarray(center, dtype=float)
        self.__dict__.pop("_at", None)       # covers()'s
        kept = set(self.face.tolist())
        self.roots = {f: r for f, r in self.roots.items() if f in kept}
        self.window = covered if len(rows) else 0.0
        self.reroot()


_CENTROIDS = np.array([_CENTROID[k] for k in penrose.KINDS])
_RADII = np.array([_RADIUS[k] for k in penrose.KINDS])


def near_faces(world, faces, center, window):
    """The faces whose corners' sphere comes within window of center (map units), shallow (float64)"""
    top = world.topology
    mesh_center = surface.d_float(world.anchor_point()) + center / (world.rho * math.ldexp(1.0, world.n))
    reach = window / (world.rho * math.ldexp(1.0, world.n))
    return np.flatnonzero(np.linalg.norm(faces.middle - mesh_center, axis=1) - faces.radius <= reach).tolist()


def level_zero(world, faces, centers, center, window):
    """Level 0: the halves of the faces near the point below, each face's root its first half's apex"""
    level = Level(0, world, center)
    face, kind, rot, mirror, coef = [], [], [], [], []
    for f in (near_faces(world, faces, center, window) if world.n < FLOAT_N else centers.uv):
        halves = faces.halves[f]
        root = halves[0][3]
        level.roots[f] = [root, None, 0.0, 0.0]
        for k, r, m, tau in halves:
            face.append(f)
            kind.append(k)
            rot.append(r)
            mirror.append(m)
            coef.append(RING.sub(tau, root))
    level.set_tiles(np.array(face, np.int64), np.array(kind, np.int8), np.array(rot, np.int8), np.array(mirror, bool),
                    np.array(coef, np.int64).reshape(-1, 4))
    return level


def level_below(world, parent, center):
    """The parent level's tiles' children, all of them (the next level's roots phi times the parent's: the same points,
    which are corners of the children)"""
    level = Level(parent.d + 1, world, center)
    level.roots = {f: [RING.mul(penrose.PHI, r[0]), None, 0.0, 0.0] for f, r in parent.roots.items()}
    grown = parent.coef @ RULE.lambda_matrix.T
    parts = []
    for kind in penrose.KINDS:
        rows = np.flatnonzero(parent.kind == kind)
        if not len(rows):
            continue
        table = RULE.child[kind]
        r, m = parent.rot[rows].astype(np.int64), parent.mirror[rows].astype(np.int64)
        for slot in range(len(table.kind)):
            parts.append((parent.face[rows], np.full(len(rows), table.kind[slot], np.int8),
                          table.rot[slot][r, m].astype(np.int8), table.mirror[slot][r, m],
                          grown[rows] + table.place[slot][r, m]))
    if parts:
        level.set_tiles(*(np.concatenate([p[i] for p in parts]) for i in range(5)))
    return level


# ----- per map: the kept levels and the drawn ones -----

class State:
    def __init__(self, key, name):
        self.levels = {}             # d -> Level: never changed once here (the worker threads read them), only replaced
        self.lock = threading.Lock()     # held to go through levels or needs, or change needs
        self.cutting = {}            # d -> a threading.Event set once tiles()'s cut of d under way is in levels
        self.needs = {}              # d -> {engine: (a Drawn: what its latest build of d needs, when)}: tiles()
        self.drawn = {}              # d -> (level object or None (a coarse level with nothing in sight), Drawn)
        self.giant = None            # giant_tint()'s answers: d -> (color, Drawn whose window is how far it holds)
        self.giant_walk = None       # its last walk's: (top, color, the level it's from, Drawn: how far it holds)
        self.no_pieces = {}          # parts_penrose's giant levels with no 3D piece near the camera: d -> (shape, Drawn)
        self.giant_tiles = {}        # parts_penrose's giant levels' tiles: d -> (the kept level's origin_of(), its own
                                     # copy, trimmed)
        self.jobs = {}               # d -> a build under way on a worker thread: (when, priority, its middle, the
                                     # world's text, whether coarse, whether wanted by the time it lands)
        self.ready = {}              # d -> its latest finished build still to write (built_level()'s)
        self.failed = set()          # levels whose build failed there: built here, where the error shows
        self.motion = schedule.motion_of(name)       # shared with the map's 3D parts
        self.tint = None             # the floor's tint as written (None: its own colors), and for which floor build
        self.tinted_floor = None
        self.key = key


_states = {}


def state_of(obj, top):
    st = _states.get(obj.name)
    if st is None or st.key != id(top):
        st = _states[obj.name] = State(id(top), obj.name)
    return st


def forget():
    """After loading a file (surface.forget()): another file's levels and labels don't apply"""
    _states.clear()
    _faces.clear()
    schedule.forget()
    background.forget()


def covers(world, text, record, center, window):
    """Whether something built for record's world, window and center (Level, Drawn) covers window around center (map
    units, this world: an array or a tuple). Its place here is kept on it as plain floats until the world changes
    (asked of every level twice a tick)."""
    at = record.__dict__.get("_at")
    if at is None or at[0] != text:
        offset, scale = surface.transform(world, text, record)
        with np.errstate(over="ignore", invalid="ignore"):
            place = tuple((offset + scale * np.asarray(record.center, dtype=float)).tolist())
            reach = float(scale * record.window)
        at = record._at = (text, place, reach, record.window)
    elif at[3] != record.window:
        # Its window was cut since (keep_near()): the same place, a new reach
        _offset, scale = surface.transform(world, text, record)
        at = record._at = (text, at[1], float(scale * record.window), record.window)
    with np.errstate(over="ignore", invalid="ignore"):
        return bool(math.dist(at[1], tuple(center)) + window <= at[2])


# kept_levels() asks this many levels coarser than the finest by default (each tick, whether each covers)
KEPT_ASKED = 48


def kept_levels(world, faces, below, st, finest, center, need, lowest=None):
    """Levels 0 to finest, each covering need(d) around center (map units), cut here and now: each that doesn't (or
    isn't kept yet) cut again from the one above it, which does by then (needs shrink down the levels). Every level from
    lowest (finest - KEPT_ASKED by default) is asked: a need that grows with the drone's speed (settle()) left a coarse
    level short while the finer ones covered, and its builds, short of its window, were stale as soon as they were
    written, built again and again (2026-11-17). Coarser ones (giant: only the floor's tint asks for them) only up to
    the first that still covers: deep down there are a thousand and more.

    Since 2026-11-19 only for the levels read in the tick: the giant ones (the floor's tint, the 3D parts' giant
    pieces). The levels the builds draw are cut where they're built (tiles()): cut here, ahead of the drone, they were a
    quarter of the main thread's time in a fast dive, and the 3D parts' builds cut them again for their own windows."""
    text = world.text()
    now = []

    def below_at(point):
        """The point below a cut's own center: deep down, keep_near() keeps a level's tiles round the point its
        Centers were worked out from (below), and says they're round the center it's given. Cut round where the drone
        will be (foot_soon) from the point below it now, a level said it held tiles it didn't, and every level cut from
        it came out empty; the 3D parts' builds failed, in the tick too, which ended the flight (2026-11-19)"""
        point = np.asarray(point, dtype=float)
        return below if np.array_equal(point, world.rho * below.foot) else surface.Below(world, point)
    lowest = max(0, finest - KEPT_ASKED) if lowest is None else max(0, min(lowest, finest))
    for k in range(lowest - 1, -1, -1):
        level = st.levels.get(k)
        if level is not None and covers(world, text, level, center, need(k)):
            break
        now.insert(0, k)
    for k in range(lowest, finest + 1):
        level = st.levels.get(k)
        if level is None or not covers(world, text, level, center, need(k)):
            now.append(k)
    if now:
        cut_below = below_at(center)
        window = lambda k: need(k) * SLACK + min(SLACK_TILES * surface_tile(world, faces, cut_below, k), 8.0 * need(k))
        for k in now:
            parents = {} if k == 0 else {k - 1: parent_of(world, faces, st, k)}
            cut = cut_levels(world, faces, cut_below, parents, [(k, window(k))], center, need(0) * SLACK * 2.0)
            with st.lock:
                st.levels.update(cut)


def let_go(st, finest):
    """The kept levels finer than finest (and what builds needed of them) let go"""
    with st.lock:
        for table in (st.levels, st.needs, st.giant_tiles):
            for k in [k for k in table if k > finest]:
                del table[k]


# A build's need of a level's tiles is widened by tiles() to the other builds' needs of it (want_tiles()) asked for
# within this long (s), as far as this many times the wider of the two
NEED_TIME = 0.5
NEED_MOST = 4.0


def want_tiles(st, engine, d, world, center, window, now):
    """A build of engine's ("floor", "parts") round center out to window (map units, this world) needs level d's
    tiles: for NEED_TIME, every cut of the level (tiles()) is that wide too"""
    record = Drawn(world.text(), world.n, np.array(center, dtype=float), float(window), 1.0)
    with st.lock:
        st.needs.setdefault(d, {})[engine] = (record, now)


def wanted_window(st, world, text, d, center, window, now):
    """How far round center (map units) level d's tiles are wanted: window, or as far as the builds asking for the
    level lately need (want_tiles())"""
    with st.lock:
        records = [r for r, when in st.needs.get(d, {}).values() if now - when <= NEED_TIME]
    widest = window
    for record in records:
        offset, scale = surface.transform(world, text, record)
        with np.errstate(over="ignore", invalid="ignore"):
            reach = scale * record.window
            far = float(np.linalg.norm(offset + scale * record.center - center)) + reach
        if math.isfinite(far):
            widest = max(widest, min(far, NEED_MOST * max(window, reach)))
    return widest


def kept(st, world, faces, text, d, center, window):
    """Kept level d if it covers window round center (map units, this world), in this world (rebased()), else None"""
    level = st.levels.get(d)
    if level is None or not covers(world, text, level, center, window):
        return None
    if level.world_text != text:
        fresh = rebased(world, faces, level)
        with st.lock:
            if st.levels.get(d) is level:
                st.levels[d] = fresh
        level = fresh
    return level


def tiles(st, world, faces, d, center, window):
    """Level d's tiles covering window round center (map units, this world), as a Level of its own (the caller's to trim
    and re-root): a kept level's, or cut from the finest kept level that covers the window, with the levels between,
    each round center and as wide as the builds asking for it lately need (want_tiles()), and kept for the next build,
    the floor's or the 3D parts'. A level being cut is waited for: two builds of a new level (the floor's and the
    parts') cut it once. Levels apart are cut at once (one cut at a time, a worker thread often waited on the other's,
    and the floor's builds were drawn late). Nothing here touches Blender, so it runs where the builds do, on the worker
    threads, or in the tick when a build is done there.

    2026-11-19: before, the floor's levels were cut in the tick round where the drone would be (a quarter of the main
    thread's time in a fast dive), and each 3D parts' build cut its level again on the worker thread for its own
    window (some 40% of its time there)."""
    text = world.text()
    center = np.asarray(center, dtype=float)
    while True:
        found = kept(st, world, faces, text, d, center, window)
        if found is not None:
            return found.copy()
        k, parent = d - 1, None
        while k >= 0:
            parent = kept(st, world, faces, text, k, center, window)
            if parent is not None:
                break
            k -= 1
        with st.lock:
            busy = next((st.cutting[j] for j in range(k + 1, d + 1) if j in st.cutting), None)
            if busy is None:
                done = threading.Event()
                for j in range(k + 1, d + 1):
                    st.cutting[j] = done
                break
        busy.wait()
    try:
        # Each level cut as wide as it's wanted, and as any finer one cut from it (its children reach only as far)
        now = time.perf_counter()
        windows, widest = {}, window
        for j in range(d, k, -1):
            widest = max(widest, wanted_window(st, world, text, j, center, window, now))
            windows[j] = widest
        below = surface.Below(world, center)
        keep = lambda j: windows[j] * SLACK + min(SLACK_TILES * surface_tile(world, faces, below, j), 8.0 * windows[j])
        cut = cut_levels(world, faces, below, {} if parent is None else {k: parent},
                         [(j, keep(j)) for j in range(k + 1, d + 1)], center, widest * SLACK * 2.0)
        with st.lock:
            st.levels.update(cut)
    finally:
        with st.lock:
            for j in range(k + 1, d + 1):
                if st.cutting.get(j) is done:
                    del st.cutting[j]
        done.set()
    return cut[d].copy()


def keep_copy(st, level):
    """A copy of a kept level that a build re-rooted, and placed its roots for (exactly, deep down), kept in its place,
    if it still holds what it was copied from (not trimmed) and that's still kept: the same tiles, worked out once.
    Thrown away, every coarse build re-rooted its copy and placed its roots again, some 2 ms a face deep down, in the
    tick for a level never drawn (a climb's new coarse levels, 2026-11-19). The level isn't changed after this."""
    source = getattr(level, "source", None)
    if source is None or level.window < source[1] or len(level.face) != len(source[0].face):
        return
    with st.lock:
        if st.levels.get(level.d) is source[0]:
            level.source = None
            st.levels[level.d] = level


def parent_of(world, faces, st, k):
    """Level k - 1 as kept, in this world (rebased())"""
    parent = st.levels[k - 1]
    if parent.world_text != world.text():
        parent = st.levels[k - 1] = rebased(world, faces, parent)
    return parent


def cut_levels(world, faces, below, parents, cuts, center, centers_window):
    """Levels cut round center: cuts = [(k, window)] in order, each kept for its window from the level above it (one
    cut here before it, or parents[k - 1]; level 0 from the faces), and holding what its parent's tiles' children
    reach, no more (keep_near() would say the whole window it was cut for); below is the point below center. Returns
    {k: Level}. Nothing here touches Blender or the levels given (it reads them), so it can run on the worker thread."""
    centers = Centers(world, faces, below, center, centers_window) if world.n >= FLOAT_N else None
    out = {}
    for k, window in cuts:
        if k == 0:
            level = level_zero(world, faces, centers, center, window)
            level.keep_near(world, faces, centers, window, center)
        else:
            parent = out.get(k - 1) or parents[k - 1]
            level = level_below(world, parent, center)
            level.keep_near(world, faces, centers, window, center)
            with np.errstate(over="ignore", invalid="ignore"):
                held = float(parent.window - np.linalg.norm(parent.center - center))
            if math.isfinite(held):
                level.window = min(level.window, held)
        out[k] = level
    return out


def rebased(world, faces, level):
    """A kept level with its map places worked out in this world (its tiles and roots are the same)"""
    offset, scale = surface.transform(world, world.text(), level)
    fresh = Level(level.d, world, offset + scale * level.center)
    fresh.window = scale * level.window
    fresh.set_tiles(level.face, level.kind, level.rot, level.mirror, level.coef)
    fresh.roots = {f: [r[0], None, 0.0, 0.0] for f, r in level.roots.items()}
    fresh.origin = origin_of(level)
    return fresh


def origin_of(level):
    """The level a kept one's tiles were cut as: itself, or what it was rebased or copied from (the same tiles)"""
    return level.__dict__.get("origin") or level


def surface_tile(world, faces, below, d):
    """Level d's tile edge where the drone is, map units: the faces' edges blended across the face below from its
    corners' (corner_edges()), so it changes smoothly over the mesh. The face's own edge jumped at every side, and
    every level's window, bars' width on screen and fade with it: crossing onto a face a sixth bigger, every level was
    short of its window at once, sixteen builds were asked for in a tick, and the worker thread fell up to 2.5 s behind
    (2026-11-17)."""
    top = world.topology
    f = below.face
    corner = corner_edges(top)[np.asarray(top.corners[f])]
    N = world.N
    u = min(max(float(below.base[0]) / N + float(below.d[0]) * math.ldexp(1.0, -world.n), 0.0), 1.0)
    v = min(max(float(below.base[1]) / N + float(below.d[1]) * math.ldexp(1.0, -world.n), 0.0), 1.0)
    if len(corner) == 4:
        edge = ((1.0 - u) * (1.0 - v) * corner[0] + u * (1.0 - v) * corner[1] + u * v * corner[2]
                + (1.0 - u) * v * corner[3])
    else:
        edge = (1.0 - u - v) * corner[0] + u * corner[1] + v * corner[2]
    return world.rho * float(edge) * phi_power_float(d, world.n)


def corner_edges(top):
    """Each mesh vertex's tile edge: the mean of its faces' (each face's the geometric mean of its sides' lengths at
    its first corner), kept on the topology"""
    found = getattr(top, "_corner_edges", None)
    if found is None:
        edge = np.sqrt(np.linalg.norm(top.eu, axis=1) * np.linalg.norm(top.ev, axis=1))
        found = np.array([float(np.mean([edge[f] for f, _c in vf])) if vf else 0.0 for vf in top.vertex_faces])
        top._corner_edges = found
    return found


# ----- drawing -----

_motifs = {}


class EdgeMotif(substitution.Motif):
    """The "edges" style: each half's legs as bars (its rhomb's edges, cut to the half; the base, inside the rhomb,
    isn't drawn), so each level's rhombs show, split by the next level's"""

    def __init__(self, rule, style="edges", half_width=0.012, outline=0.0):
        lines = rule.module.lines
        rule.module.lines = lambda kind, theta: [(_PROTO[kind][0], _PROTO[kind][1]), (_PROTO[kind][0], _PROTO[kind][2])]
        try:
            super().__init__(rule, "hankin54", half_width, outline)
        finally:
            rule.module.lines = lines
        self.style = style


def motif_of(obj, outline=None, half_width=None):
    """The map's motif: outlined as its settings say, or with outline (0 for plain bars) and half_width"""
    outline = float(setting(obj, "outline")) if outline is None else outline
    half_width = float(setting(obj, "half_width")) if half_width is None else half_width
    key = (str(setting(obj, "style")), half_width, outline)
    if key not in _motifs:
        _motifs[key] = (EdgeMotif if key[0] == "edges" else substitution.Motif)(RULE, *key)
    return _motifs[key]


def motifs_of(obj):
    """(outlined, cores) motifs for level_bars(): the bars, and their cores alone (as wide as an outlined bar's core)"""
    core = float(setting(obj, "half_width")) * (1.0 - 2.0 * float(setting(obj, "outline")))
    return motif_of(obj), motif_of(obj, 0.0, core)


# Face codes (level_bars()): a bar's core, an outline strip, and a far bar drawn whole in its core and strips' mean color
CORE, STRIP, BLEND = 0, 1, 2


def motif_bars(motif, kind, mirror):
    """A motif's bars for a kind and mirroring, laid out for level_bars(): (polygons (p, k, 2), counts (p,), codes
    (p,), their corners in one row (p k, 2), their middles (p, 2), and {corners a polygon: which polygons})"""
    kept = motif.__dict__.setdefault("_laid_out", {})
    found = kept.get((kind, mirror))
    if found is None:
        polygons, counts = (motif.bars_reversed if mirror else motif.bars)[kind]
        codes = np.where(motif.edges[kind], STRIP, CORE).astype(np.int8)
        valid = np.arange(polygons.shape[1]) < counts[:, None]
        middles = (polygons * valid[..., None]).sum(axis=1) / np.maximum(counts, 1)[:, None]
        by_size = {int(size): np.flatnonzero(counts == size) for size in np.unique(counts)}
        found = kept[(kind, mirror)] = (polygons, counts, codes, polygons.reshape(-1, 2), middles, by_size)
    return found


def level_bars(world, faces, level, motifs, center, window, box=False, camera=None, outline_within=0.0, height=0.0):
    """Level's bars within window of center, as polygons of map points (this world: level must be of it): (corners
    (m, 3), sizes, codes (CORE, STRIP, BLEND), whether any bar was cut into pieces: its geometry then holds for a
    camera this near, SPLIT_FOLLOW). Each kind and mirroring's bars for all its tiles at once. With box, every bar
    that comes within the box of half side window round center, cut to it (a coarse level's).

    motifs = (outlined, cores): the tiles within outline_within of the camera (map units) are drawn outlined, the rest
    as their cores alone, at a fifth of the faces (outlines everywhere were 42k faces a level, 2026-11-13). The strips
    are the floor's color, so on the floor the two look the same; only on a coarser level's bar do the strips show, and
    they're narrow by then. (Far bars whole in the mean color looked the same only under a pixel: on the floor, a bar
    switching between the two grew or shrank by its strips, and the flight harness saw it at loops.)

    On a bent face (its blend twisted) a bar is cut across into pieces short enough that each, flat between its
    corners, stays within BAR_FLATNESS of its distance from the camera (at least the drone's height) of the face: drawn
    whole, a long bar's middle sank under the floor, which split finer near the drone, and the floor showed through
    its side, differently after every loop (2026-11-13: Catmull-Clark icosahedron).

    Only the tiles that can reach the window (or the box) are gone through at all, and where no bar is cut (no box,
    and the face flat enough for whole bars, as every one is a few loops down), a kind's bars' corners are one product
    for all its tiles, each tile's place and frame times its prototile's bars (Level.tile_frames()): gathering each
    polygon's face and working its corners out one polygon at a time took three times as long (2026-11-14)."""
    out_corners, out_sizes, out_codes = [], [], []
    split_any = False
    if not len(level.face):
        return np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros(0, np.int8), False
    outlined, cores = motifs
    rots = level.rot.astype(np.int64)
    top = world.topology
    twist = np.linalg.norm(top.w, axis=1)
    small = phi_power_float(level.d)
    for kind in penrose.KINDS:
        for mirror in (False, True):
            every = np.flatnonzero((level.kind == kind) & (level.mirror == mirror))
            if not len(every):
                continue
            # Each tile's middle, and how far its corners can be from it (map units): only those that can reach the
            # window, or the box, go on
            turn = RULE.orient[rots[every], int(mirror)]
            cents = level.rel[every] + turn @ _CENTROIDS[kind]
            places = level.map_points(world, faces, every, cents)
            reach = _RADII[kind] * level.scale * faces.sigma_hi[level.face[every]]
            with np.errstate(over="ignore", invalid="ignore"):
                if box:
                    near = np.all(np.abs(places - center) - reach[:, None] <= window, axis=1)
                else:
                    near = np.linalg.norm(places - center, axis=1) - reach <= window
            every, places, reach = every[near], places[near], reach[near]
            if not len(every):
                continue
            # Each tile's distance from the camera, for its outlines and its bars' pieces
            distance = np.full(len(every), math.inf)
            if camera is not None:
                with np.errstate(over="ignore", invalid="ignore"):
                    distance = np.linalg.norm(places - camera, axis=1)
                    close = distance <= outline_within
                    distance = np.maximum(distance - reach, height)
                groups = ((close, outlined), (~close, cores)) if outlined is not cores else ((close | ~close, outlined),)
            else:
                groups = ((np.ones(len(every), bool), outlined),)
            # The longest piece (level units) where the camera is nearest: a straight piece of length L across a blend
            # with twist w strays from it by at most |w| (L phi^-d in (u, v), times 2 for a rhomb's slant)^2 / 4. Each
            # tile's camera point in its level units (its blend taken as affine there) and how far along a bar the
            # camera's distance has surely grown fourfold (distance over the face's least stretch), for split_along()
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                bend = twist[level.face[every]] * level.scale * small
                longest = np.where(bend > 0.0, np.sqrt(BAR_FLATNESS * distance / bend), math.inf)
                step = 4.0 * distance / (level.scale * np.maximum(faces.sigma[level.face[every]], 1e-300))
            seen = np.zeros((len(every), 2))
            span = max(bar_span(motif, kind, mirror) for _pick, motif in groups)
            if camera is not None and (longest < span).any():
                frame = level.rel[every][:, None, :] + np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
                p = level.map_points(world, faces, every, frame)                 # (t, 3, 3)
                axes = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]], axis=2)  # (t, 3, 2)
                gram = np.einsum("tki,tkj->tij", axes, axes)
                rhs = np.einsum("tki,tk->ti", axes, camera - p[:, 0])
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    det = gram[:, 0, 0] * gram[:, 1, 1] - gram[:, 0, 1] ** 2
                    a = (gram[:, 1, 1] * rhs[:, 0] - gram[:, 0, 1] * rhs[:, 1]) / det
                    b = (gram[:, 0, 0] * rhs[:, 1] - gram[:, 0, 1] * rhs[:, 0]) / det
                seen = level.rel[every] + np.nan_to_num(np.stack([a, b], axis=1))
            for pick, motif in groups:
                sel = every[pick]
                polygons, counts, codes, flat, middles, by_size = motif_bars(motif, kind, mirror)
                if not len(sel) or not len(polygons):
                    continue
                # The tiles whose bars are all short enough to stay whole (far enough from the camera, or on a face
                # flat enough), unless they're cut to a box: each one's place and frame times the prototile's corners
                tile_long, tile_seen, tile_step = longest[pick], seen[pick], step[pick]
                whole = np.zeros(len(sel), dtype=bool) if box else ~(tile_long < bar_span(motif, kind, mirror))
                if whole.any():
                    turn = RULE.orient[rots[sel[whole]], int(mirror)]            # (t, 2, 2)
                    place, G, bent, B = level.tile_frames(world, faces, sel[whole], turn)
                    turned = G.transpose(0, 2, 1)
                    corners = place[:, None, :] + flat @ turned                  # (t, p k, 3)
                    inside = place[:, None, :] + middles @ turned                # (t, p, 3)
                    # The blend's twist within a tile, unless it's under a float32 step of the tile by far (deep down)
                    if np.abs(bent).max() > 1e-12 * np.abs(G).max():
                        uv = flat @ B.transpose(0, 2, 1)
                        corners += (uv[..., 0] * uv[..., 1])[..., None] * bent[:, None, :]
                        uv = middles @ B.transpose(0, 2, 1)
                        inside += (uv[..., 0] * uv[..., 1])[..., None] * bent[:, None, :]
                    with np.errstate(over="ignore", invalid="ignore"):
                        inside = np.linalg.norm(inside - center, axis=2) <= window
                    corners = corners.reshape(len(place), len(polygons), polygons.shape[1], 3)
                    for size, columns in by_size.items():
                        kept = inside[:, columns]
                        if kept.any():
                            out_corners.append(corners[:, columns, :size][kept].reshape(-1, 3))
                            out_sizes.append(np.full(int(kept.sum()), size, dtype=np.int64))
                            out_codes.append(np.broadcast_to(codes[columns], kept.shape)[kept])
                    if whole.all():
                        continue
                    sel, tile_long, tile_seen, tile_step = sel[~whole], tile_long[~whole], tile_seen[~whole], tile_step[~whole]
                # The rest a polygon at a time: cut to the box, or across into pieces
                turn = RULE.orient[rots[sel], int(mirror)]
                x, y = polygons[..., 0], polygons[..., 1]
                pts = np.stack([turn[:, 0, 0, None, None] * x + turn[:, 0, 1, None, None] * y,
                                turn[:, 1, 0, None, None] * x + turn[:, 1, 1, None, None] * y], axis=-1)
                pts += level.rel[sel][:, None, None, :]
                # One row a bar polygon: its tile, its corners in level units, how many, its code, its pieces' length
                rows = np.repeat(sel, len(polygons))
                row_counts = np.tile(counts, len(sel))
                row_codes = np.tile(codes, len(sel))
                row_long = np.repeat(tile_long, len(polygons))
                row_seen = np.repeat(tile_seen, len(polygons), axis=0)
                row_step = np.repeat(tile_step, len(polygons))
                ideal = pts.reshape(-1, pts.shape[2], 2)
                if box:
                    # Cut to the box first (in map units, with the level units carried along as two more coordinates:
                    # the cut's new corners are where the same mix of the corners is, on the face), then split what's
                    # left: a coarse level's bars are long, and split whole, most of the pieces were cut away after
                    ideal, row_counts, keep = box_cut(level, world, faces, rows, ideal, row_counts, center, window)
                    rows, row_codes, row_long = rows[keep], row_codes[keep], row_long[keep]
                    row_seen, row_step = row_seen[keep], row_step[keep]
                corners2, row_counts, parent = split_along(ideal, row_counts, row_long, row_seen, row_step, span)
                split_any |= len(parent) > len(ideal)
                rows, row_codes = rows[parent], row_codes[parent]
                places2 = level.map_points(world, faces, rows, corners2)           # (r, k, 3)
                for size in np.unique(row_counts):
                    which = row_counts == size
                    block, flags = places2[which, :size], row_codes[which]
                    if not box:
                        with np.errstate(over="ignore", invalid="ignore"):
                            near = np.linalg.norm(block.mean(axis=1) - center, axis=-1) <= window
                        block, flags = block[near], flags[near]
                    if len(block):
                        out_corners.append(block.reshape(-1, 3))
                        out_sizes.append(np.full(len(block), size, dtype=np.int64))
                        out_codes.append(flags)
    if not out_corners:
        return np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros(0, np.int8), split_any
    return np.concatenate(out_corners), np.concatenate(out_sizes), np.concatenate(out_codes), split_any


# A bar piece strays from its face by at most this share of its distance from the camera: with the floor's own
# (surface.FLOOR_FLATNESS) well inside the push between layers (PUSH_STEP relative, 0.6%)
BAR_FLATNESS = 2e-3
# A level whose bars were cut into pieces is built again once the drone's height is this share of what it was: its
# pieces were cut for that distance, and nearer, their sag grew past the push between layers, so a coarse bar poked
# up through finer levels until its next build (2026-11-13, harness dive on the icosphere: 4.5% of pixels at once)
SPLIT_FOLLOW = 0.7
# ... and its bars are cut for a drone this much lower than it is: diving, a level cut for the drone's height was due
# again at every 30% of it, most of a dive's builds (2026-11-17: 532 of 790 on the Penrose sphere), and cut for half
# of it, each holds three times as far down
SPLIT_AHEAD = 0.5
# inside_bar() looks at the tiles within this many level units of the point
INSIDE_REACH = 2.5
# A drawn level is built again once its window is this many times wider than wanted (the drone has shrunk under it):
# kept, a level built when it was small on screen kept all its bars out to far past the view as the drone dived on,
# every level ever drawn (2026-11-13: 580k faces 36 loops down, and every line gone further down, its objects scaled
# by every loop since past what float32 and the push drivers can place)
OVERBUILT = 4.0
# The finest levels fade in in this many steps: each step writes every corner's color again and has Blender send the
# mesh to the GPU again, and with substitution.fade()'s 32, three or four levels fading in at once in a dive came to
# 70 a second, more than one a tick (2026-11-14)
FADE_STEPS = 8
# A level's fade step changes only once its fade is this share of a step past the step's edge (settle())
FADE_HOLD = 0.25
# A tick writes at most this many faces of finished builds and recolored levels, and one mesh however big: writing a
# level's mesh is half a millisecond here, and then Blender's turn, which evaluates it and sends it to the GPU before
# the next picture (a few ms for a level of 10,000 faces). A dive writes some 30 a second (builds and fade steps), and
# the frames two of them fell on were the ones dropped (2026-11-14). The rest wait a tick or two. 2026-11-19, 20,000:
# builds a fixed size at any speed (settle()) are some 6,000 faces, so 2,000 wrote one a tick, and in a fast glide up to
# 31 finished builds waited to be written, drawn 0.4 to 1 s after they were asked for; 20,000 drew none of them short by
# more than 2.3 windows (5.6 before) at the same frame rates, gliding and cruising.
WRITE_FACES = 20000
# A tick builds levels for at most this long (s); a level already drawn whose rebuild comes due after that waits for
# a later tick (it was due with slack to spare: BUILD_SLACK, half its outlines' reach). Several levels coming due in
# one tick took it to 150-270 ms on bent maps (2026-11-13).
BUILD_BUDGET = 0.012
def box_cut(level, world, faces, rows, ideal, counts, center, window):
    """Polygons of level units (tiles rows) cut to the box of half side window round center (map units): (their
    corners in level units, counts, which rows are left). Those wholly in the box stay as they are; those across its
    sides are cut in map units with their level units carried along (a cut corner is the same mix of the corners in
    both, so it's mapped back onto the face)."""
    places = level.map_points(world, faces, rows, ideal) - center                   # (r, k, 3)
    valid = np.arange(ideal.shape[1]) < counts[:, None]
    with np.errstate(over="ignore", invalid="ignore"):
        lo = np.where(valid[..., None], places, np.inf).min(axis=1)
        hi = np.where(valid[..., None], places, -np.inf).max(axis=1)
        near = np.all((hi >= -window) & (lo <= window), axis=1)
        inside = near & np.all((lo >= -window) & (hi <= window), axis=1)
    across = np.flatnonzero(near & ~inside)
    whole = np.flatnonzero(inside)
    if not len(across):
        return ideal[whole], counts[whole], whole
    mixed = np.concatenate([places[across], ideal[across]], axis=2)                 # (a, k, 5)
    cut_counts = counts[across]
    for axis in range(3):
        for sign in (1.0, -1.0):
            normal = np.zeros((len(mixed), 5))
            normal[:, axis] = sign
            mixed, cut_counts = substitution.clip_half_planes(mixed, cut_counts, normal, np.full(len(mixed), window))
    good = cut_counts >= 3
    mixed, cut_counts, across = mixed[good], cut_counts[good], across[good]
    width = max(ideal.shape[1], mixed.shape[1])
    out = np.zeros((len(whole) + len(across), width, 2))
    out[:len(whole), :ideal.shape[1]] = ideal[whole]
    out[len(whole):, :mixed.shape[1]] = mixed[..., 3:]
    return out, np.concatenate([counts[whole], cut_counts]), np.concatenate([whole, across])


# A coarse level's bar covers its box if it holds a disk this share of the box's half side (its cut corners are mapped
# back onto the face, a hair off the box's planes)
COVER_SHARE = 0.999
# split_along(): at most this many pieces a polygon in a band, and bands
MAX_PIECES = 64
MAX_BANDS = 8


def bar_span(motif, kind, mirror):
    """The longest any of a motif's bar polygons is for a kind and mirroring (level units): no piece is cut from a bar
    whose pieces may be longer"""
    spans = motif.__dict__.setdefault("_spans", {})
    found = spans.get((kind, mirror))
    if found is None:
        polygons, counts = (motif.bars_reversed if mirror else motif.bars)[kind]
        found = 0.0
        for polygon, count in zip(polygons, counts):
            c = polygon[:count]
            if count:
                found = max(found, float(np.linalg.norm(c[:, None] - c[None], axis=2).max()))
        spans[(kind, mirror)] = found
    return found


def split_along(corners, counts, spacing, seen, step, span=math.inf):
    """Convex polygons (r, k, 2) cut across their long way (their longest side's direction) into pieces spacing long
    (r,) where the camera is nearest (seen (r, 2), its point in their units, taken onto that direction), each band
    further out twice as long: band j reaches 4^j step (r,) from there (the camera's distance has grown fourfold by
    then, and a piece may be twice as long). The cuts are whole multiples of each band's length along the direction,
    so a bar's core and its strips (parallel, the same tile's camera point and lengths) are cut in the same places and
    meet without cracks: (corners, counts, the row each piece came from). Rows whose spacing is infinite (or longer
    than they are) stay whole: those whose spacing is at least span (no polygon is longer) are left out before
    anything is worked out for them, which is every bar of a level a few loops down, where the surface is flat to
    far under the pieces' allowance (finding that out a polygon at a time was a tenth of a deep build, 2026-11-14)."""
    r, k = corners.shape[:2]
    with np.errstate(invalid="ignore"):
        maybe = np.flatnonzero(np.isfinite(spacing) & (spacing > 0.0) & (spacing < span) & np.isfinite(step))
    if not len(maybe):
        return corners, counts, np.arange(r)
    if len(maybe) < r:
        out, out_counts, parent = split_along(corners[maybe], counts[maybe], spacing[maybe], seen[maybe], step[maybe])
        rest = np.ones(r, dtype=bool)
        rest[maybe] = False
        rest = np.flatnonzero(rest)
        joined = np.zeros((len(rest) + len(out), max(k, out.shape[1]), 2))
        joined[:len(rest), :k] = corners[rest]
        joined[len(rest):, :out.shape[1]] = out
        return joined, np.concatenate([counts[rest], out_counts]), np.concatenate([rest, maybe[parent]])
    order = np.arange(k)
    valid = order < counts[:, None]
    following = tiling.take_corners(corners, (order + 1) % np.maximum(counts, 1)[:, None])
    edge = following - corners
    length = np.where(valid, np.hypot(edge[..., 0], edge[..., 1]), -1.0)
    direction = edge[np.arange(r), np.argmax(length, axis=1)]
    direction /= np.maximum(np.hypot(direction[:, 0], direction[:, 1]), 1e-300)[:, None]
    flip = (direction[:, 0] < 0.0) | ((direction[:, 0] == 0.0) & (direction[:, 1] < 0.0))
    direction[flip] *= -1.0
    along = np.where(valid, np.einsum("rkj,rj->rk", corners, direction), np.nan)
    lo, hi = np.nanmin(along, axis=1), np.nanmax(along, axis=1)
    mid = np.einsum("rj,rj->r", seen, direction)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        split = np.isfinite(spacing) & (spacing > 0.0) & (spacing < hi - lo) & np.isfinite(step)
        spacing = np.where(split, np.maximum(spacing, (hi - lo) / MAX_PIECES), np.inf)
    rows = np.flatnonzero(split)
    if not len(rows):
        return corners, counts, np.arange(r)
    # Each band's cuts: band 0 within step of mid (spacing), band j from 4^(j-1) step to 4^j step out (spacing 2^j)
    cut_rows, cut_at = [], []
    s0, st, m, l0, h0 = spacing[rows], step[rows], mid[rows], lo[rows], hi[rows]
    for j in range(MAX_BANDS):
        s = s0 * 2.0 ** j
        inner = np.where(j == 0, 0.0, st * 4.0 ** (j - 1))
        outer = st * 4.0 ** j if j < MAX_BANDS - 1 else np.full(len(rows), np.inf)
        for sign in (1.0, -1.0):
            a = m + sign * inner if sign > 0 else m - outer
            b = m + outer if sign > 0 else m - inner
            a, b = np.maximum(a, l0), np.minimum(b, h0)
            with np.errstate(over="ignore", invalid="ignore"):
                first = np.ceil(a / s)
                last = np.floor(b / s)
                n = np.where((b > a) & np.isfinite(first) & np.isfinite(last), last - first + 1.0, 0.0)
            n = np.clip(np.nan_to_num(n), 0, MAX_PIECES).astype(np.int64)
            if not n.any():
                continue
            which = np.repeat(np.arange(len(rows)), n)
            index = np.arange(len(which)) - np.repeat(np.cumsum(n) - n, n)
            cut_rows.append(which)
            cut_at.append((first[which] + index) * s[which])
    if not cut_rows:
        return corners, counts, np.arange(r)
    which, at = np.concatenate(cut_rows), np.concatenate(cut_at)
    inside = (at > l0[which]) & (at < h0[which])
    which, at = which[inside], at[inside]
    if not len(which):
        return corners, counts, np.arange(r)
    order = np.lexsort((at, which))
    which, at = which[order], at[order]
    fresh = np.r_[True, (which[1:] != which[:-1]) | (at[1:] != at[:-1])]
    which, at = which[fresh], at[fresh]
    # Pieces: each row's cuts in order, from -inf to the first, between each two, and the last to +inf
    cuts = np.bincount(which, minlength=len(rows))
    split_rows = rows[cuts > 0]
    pieces = cuts[cuts > 0] + 1
    parent = np.repeat(split_rows, pieces)
    position = np.arange(len(parent)) - np.repeat(np.cumsum(pieces) - pieces, pieces)
    starts = np.cumsum(cuts) - cuts                                            # each row's first cut in `at`
    base = np.repeat(starts[cuts > 0], pieces)
    last = np.repeat(pieces, pieces) - 1
    low = np.where(position == 0, -np.inf, at[np.minimum(base + position - 1, len(at) - 1)])
    high = np.where(position == last, np.inf, at[np.minimum(base + position, len(at) - 1)])
    out, out_counts = corners[parent], counts[parent]
    with np.errstate(invalid="ignore", over="ignore"):
        out, out_counts = substitution.clip_half_planes(out, out_counts, -direction[parent], -low)
        out, out_counts = substitution.clip_half_planes(out, out_counts, direction[parent], high)
    good = out_counts >= 3
    out, out_counts, parent = out[good], out_counts[good], parent[good]
    whole = np.setdiff1d(np.arange(r), split_rows)
    width = max(k, out.shape[1])
    joined = np.zeros((len(whole) + len(out), width, 2))
    joined[:len(whole), :k] = corners[whole]
    joined[len(whole):, :out.shape[1]] = out
    return joined, np.concatenate([counts[whole], out_counts]), np.concatenate([whole, parent])


class Drawn:
    """What a level was built for (world, center, window: see covers()), how much of its colors it shows, which of
    its faces are outline strips (edges), whether it was built coarse, and if so, whether a bar holds the whole window
    (cover: its code, level_bars(), or None), and whether it came out empty"""

    def __init__(self, world_text, n, center, window, shown, cover=None, edges=None, coarse=False, camera=None,
                 split_height=None, empty=False, sizes=None):
        self.world_text, self.n, self.center, self.window, self.shown = world_text, n, center, window, shown
        self.cover, self.edges, self.coarse = cover, edges, coarse
        self.empty = empty           # built coarse, with no bar in its box (stale())
        self.split_height = split_height     # the drone's height if bars were cut into pieces for it (map units then)
        self.camera = camera         # where the camera was, if outlines were drawn for it (map units then), else None
        self.transform = None
        self.placed = None           # placed()'s
        self.sizes = sizes           # each face's corners, for recoloring
        self.corner_codes = None     # each corner's code, the first time it's recolored


def placed(world, text, record):
    """Where a drawn level's center and camera are in today's map, and how much bigger map units are than when it was
    built: (text, center, camera or None, scale), as plain floats, kept until the world changes (stale() asks for
    every drawn level every tick, some 50 of them)"""
    kept = record.placed
    if kept is None or kept[0] != text:
        offset, scale = surface.transform(world, text, record)
        with np.errstate(over="ignore", invalid="ignore"):
            center = tuple((offset + scale * record.center).tolist())
            camera = None if record.camera is None else tuple((offset + scale * record.camera).tolist())
        kept = record.placed = (text, center, camera, scale)
    return kept


def stale(world, text, record, foot, window, camera, outline_within, height, reach_wanted=None):
    """Whether a drawn level needs building again: its window no longer covering or far wider than a build now would
    be (OVERBUILT times reach_wanted, window by default), its bars cut into pieces for a drone much higher
    (SPLIT_FOLLOW), or its outlines drawn for a camera half their reach away (or none drawn, and the drone now near
    enough for some). foot and camera as tuples: where they'll be once a build asked for now is drawn.

    Not for having been built the other way (plain or coarse): a level's tiles outgrow the sight with all of it still
    drawn, and it's built coarse the next time it's due (every level was built again right at the switch, 31 ms each,
    1.4 a second diving, 2026-11-14). Nor an empty one for anything but its cover: no bar in a box is none in a
    smaller one (the coarse levels with no bar in sight, some 30 deep down, were each built again every 1.5 s as the
    sight shrank: 12 builds a second of nothing)."""
    _text, center, record_camera, scale = placed(world, text, record)
    reach = scale * record.window
    if not math.dist(center, foot) + window <= reach:
        return True
    if record.empty:
        return False
    if not reach <= OVERBUILT * (window if reach_wanted is None else reach_wanted):
        return True
    if record.split_height is not None and height < SPLIT_FOLLOW * scale * record.split_height:
        return True
    if height >= outline_within:
        return False
    if record_camera is None:
        return True
    return not math.dist(record_camera, camera) <= 0.5 * outline_within


def covering(corners, sizes, codes, center, reach):
    """Whether one of the polygons (corners (m, 3) map points, sizes, codes) holds the disk of radius reach round
    center (in its plane): which, as its code, or None. A polygon's normal by Newell's sum (it's flat but may be cut);
    all polygons with as many corners at once."""
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]) if len(sizes) else np.zeros(0, np.int64)
    for size in np.unique(sizes).tolist():
        rows = np.flatnonzero(sizes == size)
        c = corners[starts[rows, None] + np.arange(size)] - center                  # (r, size, 3), from the center
        nxt = np.roll(c, -1, axis=1)
        normal = np.cross(c, nxt).sum(axis=1)
        length = np.linalg.norm(normal, axis=1)
        ok = length > 0.0
        normal = normal / np.maximum(length, 1e-300)[:, None]
        inward = np.cross(normal[:, None, :], nxt - c)
        inward /= np.maximum(np.linalg.norm(inward, axis=2, keepdims=True), 1e-300)
        # The center's distance inside each side's line (the center is at 0)
        inside = np.einsum("rkj,rkj->rk", -c, inward)
        hold = np.flatnonzero(ok & np.all(inside >= reach, axis=1))
        if len(hold):
            return int(codes[rows[hold[0]]])
    return None


def add_push_drivers(obj):
    """Scale obj about the drone camera by 1 + its "sub_push" (finer levels on top): its location (1 + p) b - p
    (camera - map), b its unpushed place ("sub_bx", "sub_by", "sub_bz", in the map's axes: its empty is neither turned
    nor scaled), and its scale (1 + p) times its unpushed one ("sub_bs"). Made once, by the add-on."""
    if obj.animation_data is not None and obj.animation_data.drivers:
        return
    import sys
    import bpy
    # The flight's camera if one is flying (flight.py's), else the rig's (a build script's first settle)
    flight = sys.modules.get(__package__ + ".flight")
    camera = getattr(flight, "drone_camera", None) or bpy.data.objects.get("Drone_Camera")
    empty = obj.parent
    for key, value in (("sub_push", 0.0), ("sub_bx", 0.0), ("sub_by", 0.0), ("sub_bz", 0.0), ("sub_bs", 1.0)):
        if key not in obj:
            obj[key] = value

    def prop(driver, name, key):
        variable = driver.variables.new()
        variable.name = name
        variable.type = 'SINGLE_PROP'
        variable.targets[0].id = obj
        variable.targets[0].data_path = f'["{key}"]'

    for index, axis in enumerate("xyz"):
        driver = obj.driver_add("location", index).driver
        driver.type = 'SCRIPTED'
        prop(driver, "p", "sub_push")
        prop(driver, "b", f"sub_b{axis}")
        for name, target in ((axis, camera), ("o" + axis, empty)):
            variable = driver.variables.new()
            variable.name = name
            variable.type = 'TRANSFORMS'
            variable.targets[0].id = target
            variable.targets[0].transform_type = 'LOC_' + axis.upper()
            variable.targets[0].transform_space = 'WORLD_SPACE'
        driver.expression = f"(1 + p) * b - p * min(1e9, max(-1e9, {axis} - o{axis}))"
        scale = obj.driver_add("scale", index).driver
        scale.type = 'SCRIPTED'
        prop(scale, "p", "sub_push")
        prop(scale, "s", "sub_bs")
        scale.expression = "(1 + p) * s"


def place_pushed(obj, location, scale):
    """Put a pushed object's unpushed place and scale (its drivers do the rest)"""
    changed = False
    for key, value in zip(("sub_bx", "sub_by", "sub_bz", "sub_bs"), (*location, scale)):
        if obj.get(key) != float(value):
            obj[key] = float(value)
            changed = True
    if changed:
        obj["camera_push_scale"] = float(scale)
        obj.update_tag(refresh={'OBJECT'})


def level_object(obj, st, d):
    """A level object for level d: one of the map's free ones, or a new one"""
    import bpy
    used = {o.name for o, _r in st.drawn.values() if o is not None}
    for child in obj.children:
        if child.get(LEVEL_KEY) == obj.name and child.name not in used:
            return child
    name = f"{obj.name}.penrose"
    level_obj = bpy.data.objects.new(name, bpy.data.meshes.new(name))
    for c in obj.users_collection:
        c.objects.link(level_obj)
    level_obj.parent = obj
    level_obj[LEVEL_KEY] = obj.name
    level_obj[POOL] = True
    level_obj["scaling_target"] = obj
    from . import parts
    level_obj.data.materials.append(parts.parts_material())
    parts.no_shadow(level_obj)
    add_push_drivers(level_obj)
    return level_obj


def level_color(obj, d, shown):
    palette = np.array([list(c) for c in setting(obj, "palette")], dtype=float)
    color = palette[d % len(palette)]
    floor_color = np.array(list(surface.setting(obj, "floor_color")), dtype=float)
    return floor_color + (color - floor_color) * shown


def outline_color(obj):
    color = setting(obj, "outline_color")
    return np.array(list(color if color is not None else surface.setting(obj, "floor_color")), dtype=float)


def face_colors(obj, d, shown, codes):
    """Each face's linear RGBA by its code (level_bars()): the level's color, the outline color, or their mean as a bar
    shows them (each strip outline of the width), all faded in alike"""
    color = level_color(obj, d, shown)
    floor_color = np.array(list(surface.setting(obj, "floor_color")), dtype=float)
    edge = floor_color + (outline_color(obj) - floor_color) * shown
    share = 2.0 * float(setting(obj, "outline"))
    table = np.array([color, edge, color + (edge - color) * share])
    return table[np.asarray(codes, dtype=np.int64)]


def giant_tint(obj, world, faces, below, st, top, foot, sight):
    """The color of the finest giant level (coarser than top) whose bar the drone is in, None if none: walked down
    from top until one is, each level's answer kept (st.giant: d -> (color, a Drawn whose window is its clearance)) for
    as long as the sight round the drone stays clear of that level's bars' edges. Deep only (giant levels are only
    ever deep).

    The walk's answer is kept whole too (st.giant_walk: top, the color, the level it's from, and a Drawn round the
    foot for as far as every answer it went by holds), so a tick asks once however deep the map is: with the drone
    over no bar at all, the walk goes by every giant level there is, 1.4 more of them every loop down, and every tick
    took longer than the last (2026-11-14)."""
    if world.n < FLOAT_N or top <= 0:
        return None
    text = world.text()
    if st.giant is None:
        st.giant = {}
    walk = st.giant_walk
    stop = 0
    if walk is not None and covers(world, text, walk[3], foot, sight):
        if walk[0] == top:
            return walk[1]
        # Deeper: only the levels new to the giant ones are asked (a finer one's bar wins), the rest answered as before
        if walk[0] < top:
            stop = walk[0]
    else:
        walk = None
    centers = None
    room = math.inf
    color, source = None, None
    for d in range(top - 1, stop - 1, -1):
        answer = st.giant.get(d)
        if answer is None or not covers(world, text, answer[1], foot, sight):
            if centers is None:
                centers = Centers(world, faces, below, foot, sight)
            found, clearance = inside_bar(obj, world, faces, below, st.levels.get(d), centers, d)
            answer = st.giant[d] = (found, Drawn(text, world.n, foot.copy(), clearance, 1.0))
        # How much further the foot's sight can go before this answer stops holding
        offset, scale = surface.transform(world, text, answer[1])
        with np.errstate(over="ignore", invalid="ignore"):
            left = scale * answer[1].window - float(np.linalg.norm(offset + scale * answer[1].center - foot)) - sight
        room = min(room, left if math.isfinite(left) else 0.0)
        if answer[0] is not None:
            color, source = answer[0], d
            break
    else:
        if stop:
            # No bar among the new ones: the kept walk's answer, for as far as that still holds
            offset, scale = surface.transform(world, text, walk[3])
            with np.errstate(over="ignore", invalid="ignore"):
                left = scale * walk[3].window - float(np.linalg.norm(offset + scale * walk[3].center - foot)) - sight
            room = min(room, left if math.isfinite(left) else 0.0)
            color, source = walk[1], walk[2]
    if not math.isfinite(room):
        room = 0.0
    st.giant_walk = (top, color, source, Drawn(text, world.n, foot.copy(), sight + max(room, 0.0), 1.0))
    return color


def inside_bar(obj, world, faces, below, level, centers, d):
    """Whether the point below is in one of a giant level's bars (in its face's ideal coordinates, exact from the
    anchor however deep), as (its color or None, how far it is from any of its level's bars' edges, map units at least)"""
    f = below.face
    if level is None or f not in centers.ideal or not len(level.face):
        return None, 0.0
    rows = np.flatnonzero(level.face == f)
    per = level.scale * max(float(faces.sigma[f]), 1e-300)
    # How far the point is from its face's sides: past them, other faces' bars aren't looked at
    u, v = (float(x) for x in centers.uv[f])
    side = min(u, 1.0 - u, v, 1.0 - v) * world.rho * math.ldexp(float(world.topology.sigma_lo[f]), world.n)
    clearance = min(side, level.window)
    if not len(rows) or f not in level.roots:
        return None, clearance
    (xa, xb), (ya, yb) = RING.exact_xy(level.roots[f][0])
    cx, cy = centers.ideal[f]
    grow = q_phi(d)
    dx = q_sub(q_mul(cx, grow), (xa, xb))
    dy = q_sub(q_mul(cy, grow), (ya, yb))
    here = np.array([float_sqrt(dx[0], dx[1], 5), float_sqrt(dy[0], dy[1], 5) * RING.stretch])
    # Only the tiles round the point (level units): further bars are at least that far, which caps the clearance
    turn = RULE.orient[level.rot[rows].astype(np.int64), level.mirror[rows].astype(np.int64)]
    cents = level.rel[rows] + np.einsum("tij,tj->ti", turn, _CENTROIDS[level.kind[rows]])
    reach = INSIDE_REACH
    rows = rows[np.linalg.norm(cents - here, axis=1) <= reach + _RADII[level.kind[rows]]]
    clearance = min(clearance, (reach - float(_RADII.max())) * per)
    motif = motif_of(obj)
    found = None
    for kind in penrose.KINDS:
        for mirror in (False, True):
            sel = rows[(level.kind[rows] == kind) & (level.mirror[rows] == mirror)]
            polygons, counts = (motif.bars_reversed if mirror else motif.bars)[kind]
            if not len(sel) or not len(polygons):
                continue
            turns = RULE.orient[level.rot[sel].astype(np.int64), int(mirror)]
            for size in np.unique(counts).tolist():
                which = np.flatnonzero(counts == size)
                c = (np.einsum("tij,pkj->tpki", turns, polygons[which, :size]) + level.rel[sel][:, None, None, :]
                     - here).reshape(-1, size, 2)
                e = np.roll(c, -1, axis=1) - c
                # Signed distance inside each side (counterclockwise polygons; mirrored ones are listed reversed)
                inside = (e[..., 0] * -c[..., 1] - e[..., 1] * -c[..., 0]) / np.maximum(np.hypot(e[..., 0], e[..., 1]),
                                                                                       1e-300)
                least = inside.min(axis=1)
                # Inside, the least is the distance to its edge; outside, the most a side is passed by bounds the
                # distance to it from below (it's convex)
                clearance = min(clearance, float(np.abs(least).min()) * per)
                hits = np.flatnonzero(least >= 0.0)
                if len(hits) and found is None:
                    found = bool(motif.edges[kind][which[hits[0] % len(which)]])
    if found is None:
        return None, clearance
    return (outline_color(obj) if found else level_color(obj, d, 1.0)), clearance


def tint_floor(obj, st, color):
    """The floor all one color (a giant level's bar the drone is in), or its own colors again (None)"""
    floor = surface.floor_object(obj)
    built = surface._floors.get(obj.name)
    if built is None:
        return
    key = None if color is None else tuple(np.round(color, 6))
    if key == st.tint and st.tinted_floor is built:
        return
    attribute = floor.data.color_attributes.get("Col")
    if attribute is None or not len(attribute.data):
        return
    if color is None:
        if st.tint is None or st.tinted_floor is not built:
            st.tint, st.tinted_floor = None, built
            return
        colors = built.own_colors
    else:
        if st.tinted_floor is not built or st.tint is None:
            own = np.empty(len(attribute.data) * 4, dtype=np.float32)
            attribute.data.foreach_get("color", own)
            built.own_colors = own
        colors = np.tile(np.asarray(color, dtype=np.float32), len(attribute.data))
    attribute.data.foreach_set("color", colors)
    floor.data.update()
    st.tint, st.tinted_floor = key, built


def settle(obj, camera_world):
    """At the end of a tick: the levels to draw where the camera ended up, built where they need to be, placed and
    pushed (finer on top, the floor under them all). Nothing fades out: from the finest (bars min_bar_px wide under the
    drone, fading in from there) every level is drawn, the plain ones within radius of their tiles, the coarse ones
    out to the sight, up to the first whose bar holds the whole view; past the giant ones' threshold, the floor takes
    the color of the bar the drone is in, if any (2026-11-13: "i like the patterns to remain static"). With builds on
    the worker thread, each is asked for and built ahead of the drone's flight by the time builds take (BUILD_SLACK)."""
    world = surface.world_of(obj)
    top = world.topology
    faces = faces_of(obj, top)
    st = state_of(obj, top)
    text = world.text()
    camera = surface.map_point(obj, camera_world)
    below = surface.below_of(obj, world, camera)
    height = max(abs(below.height) * world.rho, 1e-300)
    foot = world.rho * below.foot
    unit = float(np.linalg.norm(surface.world_vector(obj, (1.0, 0.0, 0.0))))
    clip_end = surface.drone_clip_end() / unit
    if clip_end <= 0.0:
        return
    sight = float(surface.setting(obj, "window")) * clip_end
    motifs = motifs_of(obj)
    # Only the fade in: maps built before 2026-11-13 hold the old [in, in, out, out] band, and nothing fades out now
    settings = {"fade": list(setting(obj, "fade") or [])[:2]}
    focal = float(setting(obj, "focal"))
    bar = 2.0 * float(setting(obj, "half_width"))
    radius = float(setting(obj, "radius"))
    tile0 = surface_tile(world, faces, below, 0)
    px0 = focal * bar * tile0 / height                     # level 0's bars' width under the drone, px
    strip = float(setting(obj, "outline")) * bar
    outline_within = functools.lru_cache(None)(lambda d: focal * strip * tile0 * phi_power_float(d) / OUTLINE_PX)
    log_level = lambda x: math.log(max(x, 1e-300)) / LOG_PHI
    # Level d's bars are px0 phi^-d wide and its tiles tile0 phi^-d: the finest drawn has bars min_bar_px wide; plain
    # while radius of its tiles fit in the sight; giant past GIANT
    finest = max(0, math.ceil(log_level(px0 / float(setting(obj, "min_bar_px")))))
    plain = min(finest, max(0, math.ceil(log_level(radius * tile0 / sight))))
    drawable = min(plain, max(0, math.ceil(log_level(tile0 / (GIANT * sight)))))
    drawn_window = functools.lru_cache(None)(lambda d: min(sight, radius * tile0 * phi_power_float(d)))
    # With nothing drawn yet (a flight's start, a file just loaded), everything is built before the first picture
    first = not st.drawn
    # Where the drone will be when a build asked for now is drawn (soon), and each level's build (plan()): its middle,
    # how far it reaches, and how far ahead of the drone it is
    now = time.perf_counter()
    velocity = st.motion.velocity(world, text, foot, height, now, surface.transform)
    wait = st.motion.wait("floor") if background.enabled and not first else 0.0
    speed = float(np.linalg.norm(velocity))
    ahead = velocity * wait
    foot_soon = tuple((foot + ahead).tolist())
    camera_soon = tuple((np.asarray(camera, dtype=float) + ahead).tolist())

    @functools.lru_cache(None)
    def plan(d):
        """Level d's build: its middle (where the drone will be when it's drawn, and up to its slack further along the
        flight), how far it reaches (its window and slack) and how far its middle is past where the drone will be"""
        window = drawn_window(d)
        margin = (BUILD_SLACK - 1.0) * window
        past = min(margin, speed * wait)
        lead = speed * wait + past
        return (foot + velocity * (lead / speed) if lead else foot), window + margin, past

    # What the worker threads have finished, and the giant levels, read here (giant_tint()), kept round where the drone
    # will be; the levels drawn are cut by their builds (tiles())
    collect(obj, st)
    if drawable > 0:
        kept_levels(world, faces, below, st, drawable - 1, np.asarray(foot_soon), lambda d: plan(d)[1] + plan(d)[2],
                    lowest=drawable - 1)
    let_go(st, finest + 2)

    # How much of its colors level d shows (None: not drawn at all), in fade steps (asked some 60 times a tick)
    fade_steps = functools.lru_cache(None)(lambda d: substitution.fade(settings, px0 * phi_power_float(d)) * FADE_STEPS)

    def shown_of(d):
        if not drawable <= d <= finest:
            return None
        if d < plain:
            return 1.0
        steps = fade_steps(d)
        # A drawn level keeps its step until the fade is FADE_HOLD of a step past its edge: a drone holding its height
        # near one switched a level between two steps and back within a tick or two, a flicker (2026-11-17)
        entry = st.drawn.get(d)
        if entry is not None and abs(steps - entry[1].shown * FADE_STEPS) < 0.5 + FADE_HOLD:
            return entry[1].shown
        return round(steps) / FADE_STEPS

    def due(d, record):
        """Whether a build of level d drawn as record needs doing again by the time one asked for now is drawn"""
        return stale(world, text, record, foot_soon, drawn_window(d), camera_soon, outline_within(d), height,
                     plan(d)[1])

    def short(d):
        """Whether level d is drawn short of the drone now (or not drawn at all)"""
        entry = st.drawn.get(d)
        return entry is None or not covers(world, text, entry[1], foot, drawn_window(d))

    floor_wait = st.motion.wait("floor")
    foot_now = tuple(foot.tolist())

    def worth(d, shown, quality=False):
        """What a build of level d asked for now puts right: (stake, how soon it's wanted, s; see schedule.value()).
        Its disc's share of the view times how much of its colors it shows (shown), times how much of the
        disc it would leave missing by the time it lands: all of it if it's not drawn, the part past its drawn window
        if that's short then, else nothing but how it looks (REFRESH: cut finer, outlines, trimmed, built coarse). Not
        short yet but short by then: wanted once it's short. Seen as it will be when it lands: diving, the drone lower
        (by the map's descent over the floor's wait), the level's disc more of the view and its bars wider, more shown."""
        window = drawn_window(d)
        sinking = max(0.0, -st.motion.descent) * floor_wait
        height_then = height * math.exp(-sinking)
        if d >= plain:
            shown = max(shown, substitution.fade(settings, px0 * phi_power_float(d) * math.exp(sinking)))
        stake = INK * schedule.view_share(window, height_then) * shown
        entry = st.drawn.get(d)
        if entry is None:
            return stake, 0.0
        _text, center, _camera, scale = placed(world, text, entry[1])
        reach = scale * entry[1].window
        with np.errstate(over="ignore", invalid="ignore"):
            gap_now = math.dist(center, foot_now) + window - reach
            gap_soon = math.dist(center, foot_soon) + window - reach
        if quality or not gap_soon > 0.0:
            return stake * schedule.REFRESH, 0.0
        missing = min(1.0, gap_soon / (2.0 * window))
        return stake * missing, (0.0 if gap_now > 0.0 or speed <= 0.0 else -gap_now / speed)

    def ask(d, shown, coarse, wait_here, stake, needed_in=0.0):
        """Level d built (build_level()), unless a finished build of it still to write does for now, at what it's
        worth (worth()'s stake and how soon it's wanted, for what such builds take)"""
        kind = "floor coarse" if coarse else "floor plain"
        priority = schedule.priority(stake, needed_in, floor_wait, st.motion.costs.guess(kind, 1.0, COST_FIRST))
        if not wait_here:
            ready = st.ready.get(d)
            if ready is not None and ready[1].coarse >= coarse and not due(d, ready[1]):
                return
        center, reach, _lead = plan(d)
        # Its tiles, as far as a coarse level's box reaches (its corners), kept for the 3D parts' builds too
        want_tiles(st, "floor", d, world, center, reach * (BOX_CORNER if coarse else 1.0), now)
        build_level(obj, world, faces, st, motifs, d, shown, coarse, foot, center, reach, text, camera_soon,
                    outline_within(d), height, below, wait=wait_here, priority=priority,
                    urgent=needed_in <= floor_wait)

    # Writing, at most WRITE_FACES a tick (and one mesh however big): the finished builds of levels drawn short of the
    # drone first, then a fade step (the level furthest behind its colors), then the rest. A rebuilt level shows the
    # same picture a tick later, but one drawn short shows the floor through, and a level kept waiting for its colors
    # (builds written every tick of a fast dive) jumped several steps at once (2026-11-14).
    wrote = write_ready(obj, st, shown_of, short, 0, urgent=True)
    behind = [(abs(record.shown - (shown_of(d) or 0.0)), d) for d, (level_obj, record) in st.drawn.items()
              if level_obj is not None and (shown_of(d) or 0.0) > 0.0]
    if behind and max(behind)[0] > 1e-9 and wrote < WRITE_FACES:
        d = max(behind)[1]
        wrote += recolor(obj, d, *st.drawn[d], shown_of(d))
    wrote += write_ready(obj, st, shown_of, short, wrote, urgent=False)
    # What to draw: (d, shown, coarse)
    wanted = {}
    for d in range(plain, finest + 1):
        shown = shown_of(d)
        if shown > 0.0:
            wanted[d] = (shown, False)
    covered = False
    fresh = set()                    # the levels known to be up to date this tick
    for d in range(plain - 1, drawable - 1, -1):
        wanted[d] = (1.0, True)
        entry = st.drawn.get(d)
        # One never drawn is built at once: whether its bar holds the view says where the walk ends. One drawn from a
        # build of when it was plain (diving, every level turns coarse in turn) is built coarse: its plain build reaches
        # past the sight, so it's never due, and it says nothing of a bar holding the view, so the walk went on to the
        # giant levels, every one drawn as it was when plain: 50 levels of 8,500 bars where 20 were wanted
        # (2026-11-19, the user: "the framerate drops off after awhile")
        if entry is None or due(d, entry[1]):
            ask(d, 1.0, True, first or entry is None, *worth(d, 1.0))
        elif not entry[1].coarse:
            ask(d, 1.0, True, False, *worth(d, 1.0, quality=True))
        fresh.add(d)
        if st.drawn[d][1].cover is not None:
            covered = True
            break
    # Free the levels not wanted, and empty any of the map's level objects not in use (after a file loads, the levels
    # it was saved with, which nothing here knows of)
    for d in [d for d in st.drawn if d not in wanted]:
        st.drawn.pop(d)
    # And their builds still waiting on the worker thread (all but the finest's, built ahead of need below): diving,
    # those of the levels left behind stayed in line, some 230 ten seconds into a fast dive (2026-11-19)
    for d in [d for d in st.jobs if d not in wanted and d != finest]:
        background.cancel((obj.name, d))
        st.jobs.pop(d)
    used = {o.name for o, _r in st.drawn.values() if o is not None}
    for child in obj.children:
        if child.get(LEVEL_KEY) == obj.name and child.name not in used and len(child.data.polygons):
            tiling.write_mesh(child, np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros((0, 4)))
    started = time.perf_counter()
    # The finest first: they're the ones under the drone. Handed to the worker thread, a level stays as it is (or, new,
    # undrawn: its bars are a fraction of a pixel wide) until its build is done; built here (no worker thread), those
    # due wait their turn once a tick has spent BUILD_BUDGET.
    for d, (shown, coarse) in sorted(wanted.items(), reverse=True):
        entry = st.drawn.get(d)
        if d not in fresh and (entry is None or due(d, entry[1])):
            if first or not background.enabled:
                if entry is None or time.perf_counter() - started < BUILD_BUDGET:
                    ask(d, shown, coarse, True, 1.0)
            else:
                ask(d, shown, coarse, False, *worth(d, shown))
        level_obj, record = st.drawn.get(d, (None, None))
        if level_obj is None:
            continue
        if abs(record.shown - shown) > 1e-9 and (wrote < WRITE_FACES or first):
            wrote += recolor(obj, d, level_obj, record, shown)
        offset, scale = surface.transform(world, text, record)
        place_pushed(level_obj, offset + scale * record.center, scale)
    # The next finer level built ahead of need, kept until it fades in (write_ready()): diving, a new level comes in
    # every quarter of a second or so, and asked for only once it showed, it came in half faded in (2026-11-17: the
    # Penrose sphere's dive, a level that should have been drawn half shown missing in two thirds of the ticks)
    if background.enabled and not first and plain <= finest and finest not in wanted:
        ready = st.ready.get(finest)
        if ready is None or due(finest, ready[1]):
            # Worth what it shows once it fades in (its first step), wanted once its bars are min_bar_px wide: diving
            # (schedule.Motion's rate), the drone's height falls and its bars widen
            px = px0 * phi_power_float(finest)
            rate = st.motion.descent
            needed_in = (math.log(float(setting(obj, "min_bar_px")) / px) / -rate
                         if rate < 0.0 and 0.0 < px < float(setting(obj, "min_bar_px")) else 1.0)
            ask(finest, 0.0, False, False, worth(finest, 1.0 / FADE_STEPS)[0], max(needed_in, 0.0))
    tint_floor(obj, st, None if covered else giant_tint(obj, world, faces, below, st, drawable, foot, sight))
    # Finer levels on top: the finest unpushed, each coarser a step under; the floor a step under them all
    push = 0.0
    for d in sorted(st.drawn, reverse=True):
        level_obj = st.drawn[d][0]
        if level_obj is None:
            continue
        if abs(level_obj.get("sub_push", -1.0) - push) > 1e-12:
            level_obj["sub_push"] = push
            level_obj.update_tag(refresh={'OBJECT'})
        push += PUSH_STEP
    floor = surface.floor_object(obj)
    add_push_drivers(floor)
    if abs(floor.get("sub_push", -1.0) - push) > 1e-12:
        floor["sub_push"] = push
        floor.update_tag(refresh={'OBJECT'})


def recolor(obj, d, level_obj, record, shown):
    """A drawn level's colors rewritten showing shown of them (a fade step): one attribute written, from each
    corner's color code, not the level built again. Returns how many faces it wrote."""
    colors = level_obj.data.color_attributes.get("Col")
    if colors is not None and len(colors.data):
        if record.corner_codes is None:
            record.corner_codes = np.repeat(record.edges, record.sizes).astype(np.int64)
        table = face_colors(obj, d, shown, np.arange(3)).astype(np.float32)
        colors.data.foreach_set("color", table[record.corner_codes].ravel())
        level_obj.data.update()
    record.shown = shown
    return len(record.sizes)


def built_level(world, faces, level, motifs, coarse, foot, center, reach, camera, outline_within, height, below):
    """A level's bars built round center out to reach (map units), the drone's point below at foot, as write_level()
    takes them: plain, within reach; coarse, cut to a box that size, re-rooted near the drone first, and whether a bar
    holds it all (Drawn.cover). Outlined within outline_within of the camera (level_bars()). level is trimmed and
    re-rooted as the build needs it. The build is said to reach only as far as the level's tiles do (it's drawn from
    them): one said to reach further was kept on with a gap in it (its tiles held what a build reaches, but they were
    only kept for the level's window, 2026-11-17). Nothing here touches Blender, so it can run on the worker
    thread (background.py), given its own world and level."""
    # A level kept far wider than this build wants (cut when the drone was bigger) is cut down first: kept, it held every
    # tile it had when it was small on screen, and coarse builds went through tens of thousands of them deep down
    keep = (reach + float(np.linalg.norm(center - foot))) * SLACK
    with np.errstate(over="ignore", invalid="ignore"):
        if level.window > OVERBUILT * keep and len(level.face) > TRIM_MIN:
            centers = Centers(world, faces, below, foot, keep) if world.n >= FLOAT_N else None
            level.keep_near(world, faces, centers, keep, foot)
    if coarse and len(level.coef) and np.abs(level.coef).max() > COARSE_COEF:
        level.reroot(COARSE_COEF)
    near = height < outline_within
    cut_for = height * SPLIT_AHEAD
    corners, sizes, edges, split = level_bars(world, faces, level, motifs, center, reach, box=coarse,
                                              camera=camera, outline_within=outline_within if near else 0.0,
                                              height=cut_for)
    with np.errstate(over="ignore", invalid="ignore"):
        held = float(level.window - np.linalg.norm(level.center - center))
    window = min(reach, held) if math.isfinite(held) else reach
    cover = covering(corners, sizes, edges, center, window * COVER_SHARE) if coarse and len(sizes) else None
    record = Drawn(level.world_text, world.n, center.copy(), window, 1.0, cover, edges, coarse,
                   np.array(camera, dtype=float) if near else None, cut_for if split else None,
                   empty=coarse and not len(sizes), sizes=sizes)
    return (corners - center).astype(np.float32), record, level


# A build trims a level only if it has more tiles than this (built_level()): trimming a few is more work than it saves,
# and the trimmed copy can't be kept (keep_copy())
TRIM_MIN = 4000
# A coarse level's box (level_bars()) reaches this many times its half side, at its corners
BOX_CORNER = math.sqrt(2.0)


def built_from_tiles(st, world, faces, d, motifs, coarse, foot, center, reach, camera, outline_within, height, below):
    """built_level() of level d's tiles as kept, or cut here (tiles())"""
    level = tiles(st, world, faces, d, center, reach * (BOX_CORNER if coarse else 1.0))
    built = built_level(world, faces, level, motifs, coarse, foot, center, reach, camera, outline_within, height, below)
    keep_copy(st, level)
    return built


def write_level(obj, st, d, shown, built):
    """A built level (built_level()'s) written into a level object and drawn from now on (st.drawn[d]); a coarse
    level with nothing in its box gets no object"""
    corners, record, _level = built
    entry = st.drawn.get(d)
    level_obj = entry[0] if entry is not None else None
    if record.empty:
        if level_obj is not None:
            tiling.write_mesh(level_obj, np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros((0, 4)))
        level_obj = None
    else:
        if level_obj is None:
            level_obj = level_object(obj, st, d)
        tiling.write_mesh(level_obj, corners, record.sizes, face_colors(obj, d, shown, record.edges), convex=True)
    record.shown = shown
    st.drawn[d] = (level_obj, record)


def build_level(obj, world, faces, st, motifs, d, shown, coarse, foot, center, reach, text, camera, outline_within,
                height, below, wait=True, priority=0.0, urgent=True):
    """Level d built round center out to reach (settle()'s plan(): ahead of the drone's point below, foot) and drawn
    (st.drawn[d]): here and now with wait, or if the worker thread can't be used (background.py); otherwise handed to
    it at priority (settle()'s worth of it), in place of a build of it still waiting there (unless one's under way),
    and written in a later tick (write_ready()), the level drawn as it is until then; urgent: wanted by the time it can
    land (its wait is the floor's latency). Its tiles are cut where it's built (built_from_tiles())."""
    key = (obj.name, d)
    if wait or not background.enabled or d in st.failed:
        st.failed.discard(d)
        st.jobs.pop(d, None)
        st.ready.pop(d, None)
        background.cancel(key)
        write_level(obj, st, d, shown, built_from_tiles(st, world, faces, d, motifs, coarse, foot, center, reach,
                                                        camera, outline_within, height, below))
        return
    job = st.jobs.get(d)
    # A build still waiting is handed in again, in its place in line, if its middle has moved FRESH of its reach since
    # it was asked for: waiting behind others in a fast glide, builds asked round where the drone had been were short
    # of it when drawn (2026-11-19); or it's plain and a coarse one is asked for. Else it's only rated again.
    if background.pending(key) and (job is None or background.running(key) or (
            job[3] == text and job[4] >= coarse and np.linalg.norm(center - job[2]) < FRESH * reach)):
        background.rate(key, priority)
        if job is not None and not background.running(key):
            st.jobs[d] = job[:1] + (priority,) + job[2:5] + (urgent or job[5],)
        return
    # Its own world: this thread loops on
    if background.submit(key, built_from_tiles, st, surface.World.from_text(text, world.rho, world.topology), faces, d,
                         motifs, coarse, foot.copy(), center.copy(), reach, np.array(camera, dtype=float),
                         outline_within, height, below, priority=priority, replace=True):
        st.jobs[d] = (time.perf_counter(), priority, center.copy(), text, coarse, urgent)


def collect(obj, st):
    """The worker threads' finished builds, each level's latest kept to write (st.ready), how long they took to land
    (the map's motion's latency: those wanted by then), and what they cost (the workers' time, by kind)"""
    now = time.perf_counter()
    for key, result in background.finished(obj.name):
        d = key[1]
        job = st.jobs.pop(d, None)
        if result is background.FAILED:
            st.failed.add(d)
        elif job is not None:
            st.ready[d] = result
            st.motion.costs.took("floor coarse" if job[4] else "floor plain", background.ran(key))
            if job[5]:
                st.motion.took("floor", now - job[0])


def write_ready(obj, st, shown_of, short, wrote, urgent):
    """The finished builds still to write (st.ready) of the levels shown, those of levels drawn short of the drone
    (short(d)) with urgent, else the rest, the most shown first and the coarser of those (bigger on the screen), while
    the tick has written under WRITE_FACES faces (wrote of them already); returns how many it wrote. A level not shown
    yet (built ahead, settle()) keeps its build until it is; a level no longer drawn at all loses it."""
    start = wrote
    for d in sorted(st.ready, key=lambda d: (-(shown_of(d) or 0.0), d)):
        if wrote >= WRITE_FACES:
            break
        shown = shown_of(d)
        if shown is None:
            del st.ready[d]
            continue
        if shown <= 0.0 or short(d) != urgent:
            continue
        built = st.ready.pop(d)
        write_level(obj, st, d, shown, built)
        wrote += max(len(built[1].sizes), 1)
    return wrote - start
