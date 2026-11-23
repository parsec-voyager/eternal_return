"""Endless motifs on square-lattice surface maps (2026-11-23, phase 4 of the unified engines: one motif language). A
surface map (surface.py) whose rule is "square" splits every quad face k x k at every level, forever; with a motif
(motif.Pieces: the plane maps' per-cell pieces, the Inferno's star outlines and gold lines, say) its cells are drawn the
way surface_substitution.py draws a substitution rule's tiles: the same levels (each its own object, finer ones on top,
the floor under them all), outlines, builds on the worker threads, coarse levels cut to a box. A square's k x k split is
a substitution rule too, a periodic one, so all that's needed here is where the tiles are:
- CellRule: the rule, three kinds (the unit square, a cell, and its halves, a triangle face's), each level k times
  finer
- CellFaces: what the drawing asks of the faces (surface_substitution.Faces holds a labeled mesh's): a face's (u, v)
  are a cell's own frame times k^d, so no labels
- CellLevel: a level's cells near a point, each face's root an exact point of the level's lattice (Topology.point(),
  a key of any depth), the cells by their (I, J) from it in whole numbers
- tiles(): a level's cells within a window, found face by face from the face below (surface.build_floor()'s walk)

The cells' places in the window are exact at any depth, as the floor's are: a root is placed from the anchor exactly
and rounded once, and the cells near it in floats from it. A triangle face splits into k^2 triangles, up and down, in
its own (u, v) the halves of square cells (motif.CELLS): its up cells are kind UP and its down ones DOWN, each placed by
its square cell's corner.
"""

import math
from fractions import Fraction

import numpy as np

from . import surface
from . import surface_substitution as SS

# The cells' kinds (motif.CELLS): a quad face's square cells, a triangle face's up and down halves of them
SQUARE, UP, DOWN = 0, 1, 2
# How many cells either way of a root a face's grid may reach (surface.face_grid()'s cap)
CELL_REACH = 1024


class CellRule:
    """A square lattice's k x k split, as a substitution rule's tables (substitution.Rule) are asked for by the
    drawing: three kinds, a cell of unit side and its halves (a triangle face's), never turned or mirrored"""

    periodic = True
    straddle = False
    pad = 0.0

    def __init__(self, k):
        self.k = int(k)
        self.name = f"cells{self.k}"
        self.kinds = (SQUARE, UP, DOWN)
        self.log_loop = math.log(self.k)
        self.most_children = self.k * self.k
        self.middles = np.array([[0.5, 0.5], [1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]])
        self.radii = np.array([math.sqrt(0.5), math.sqrt(5.0) / 3.0, math.sqrt(5.0) / 3.0])
        self.orient = np.array([[np.eye(2), np.diag([1.0, -1.0])]])          # (rot 1, mirror 2, 2, 2)

    def shrink_float(self, d, shift=0):
        """k^(shift - d) as a float, correctly rounded, saturating past a float's range: level d's cell in level 0's,
        times k^shift (a world's n: map units per level unit over rho)"""
        e = shift - d
        if e * self.log_loop > 709.78:
            return math.inf
        if e * self.log_loop < -745.2:
            return 0.0
        return float(self.k ** e) if e >= 0 else float(Fraction(1, self.k ** -e))

    def log_shrink(self, d, shift=0):
        """The log of shrink_float(d, shift), past a float's range too"""
        return (shift - d) * self.log_loop


_rules = {}


def rule_of(obj):
    k = surface.factor(obj)
    found = _rules.get(k)
    if found is None:
        found = _rules[k] = CellRule(k)
    return found


class CellFaces:
    """What surface_substitution.py's drawing asks of a mesh's faces, for its cells: the rule, each face's least and
    greatest stretch (mesh units a unit of (u, v)), its ideal frame (a cell's own: v straight up, ev_x 0 and ev_y 1),
    and that cells aren't cut to their faces (they tile them)"""

    clip = False

    def __init__(self, top, rule):
        self.rule = rule
        self.sigma = np.asarray(top.sigma_lo, dtype=float)
        self.sigma_hi = np.asarray(top.sigma_hi, dtype=float)
        count = len(self.sigma)
        self.ev_x = np.zeros(count)
        self.ev_y = np.ones(count)


_faces = {}


def faces_of(obj, top):
    key = (id(top), surface.factor(obj))
    found = _faces.get(key)
    if found is None:
        found = _faces[key] = CellFaces(top, rule_of(obj))
    return found


class CellLevel(SS.Level):
    """Level d's cells near center: each face's root an exact point of the level's lattice, roots[f] = [(I, J), its
    map place, u, v], and each cell (its corner nearest the face's (0, 0)) by its whole-number offsets from it, coef
    (t, 2), which are its place in the level's units (rel)"""

    def set_tiles(self, face, kind, rot, mirror, coef):
        self.face, self.kind, self.rot, self.mirror = face, kind, rot, mirror
        self.coef = np.asarray(coef, dtype=np.int64).reshape(-1, 2)
        self.rel = self.coef.astype(float)
        self.params = None

    def place_root(self, world, faces, f):
        root = self.roots[f]
        if root[1] is None:
            I, J = root[0]
            M = world.across(self.d)
            root[1] = world.rho * world.units(world.topology.point(f, I, J, M))
            root[2], root[3] = float(Fraction(I, M)), float(Fraction(J, M))
        return root

    def reroot(self, limit=SS.COEF_LIMIT):
        """Faces whose cells are far from their root get a cell near their middle as their root"""
        for f in list(self.roots):
            rows = np.flatnonzero(self.face == f)
            if not len(rows) or np.abs(self.coef[rows]).max() < limit:
                continue
            middle = rows[np.argmin(np.linalg.norm(self.rel[rows] - self.rel[rows].mean(axis=0), axis=1))]
            shift = self.coef[middle].copy()
            I, J = self.roots[f][0]
            self.roots[f] = [(I + int(shift[0]), J + int(shift[1])), None, 0.0, 0.0]
            self.coef[rows] -= shift
        self.set_tiles(self.face, self.kind, self.rot, self.mirror, self.coef)

    def keep_near(self, world, faces, centers, window, center):
        """Keep the cells within window (map units) of center: by their map places, which are good near the drone at
        any depth (from roots near it)"""
        if len(self.face):
            rows = np.arange(len(self.face))
            places = self.map_points(world, faces, rows, self.rel + self.rule.middles[self.kind])
            reach = self.rule.radii[self.kind] * self.scale * faces.sigma_hi[self.face]
            with np.errstate(over="ignore", invalid="ignore"):
                keep = np.linalg.norm(places - center, axis=1) - reach <= window
            self.take(np.flatnonzero(keep))
        self.center = np.asarray(center, dtype=float)
        self.__dict__.pop("_at", None)
        kept = set(self.face.tolist())
        self.roots = {f: r for f, r in self.roots.items() if f in kept}
        self.window = window if len(self.face) else 0.0
        self.reroot()


def tiles(world, faces, d, center, window):
    """Level d's cells within window (map units) of center (this world's map units), as a CellLevel: face by face from
    the one below center, in each a root (the level's lattice point nearest where the walk came in) placed exactly and
    the grid of its cells within reach (surface.face_grid()), then the faces across the sides its cells reach, as
    surface.build_floor() walks the floor's"""
    top = world.topology
    rule = faces.rule
    M = world.across(d)
    level = CellLevel(d, world, center, rule)
    center = np.asarray(center, dtype=float)
    below = surface.Below(world, center)
    u, v = below.cell_place(d)
    queue = [(below.face, *surface.nearest_point(top._sides[below.face], u, v, M))]
    done = set()
    out_face, out_kind, out_coef = [], [], []
    while queue:
        f, IR, JR = queue.pop(0)
        if f in done:
            continue
        done.add(f)
        sides = top._sides[f]
        root = [(IR, JR), None, 0.0, 0.0]
        level.roots[f] = root
        ref = level.place_root(world, faces, f)[1] - center
        if not np.all(np.isfinite(ref)):
            continue
        D, gi, gj, grid, rem = surface.face_grid(top, f, IR, JR, M, ref, window, CELL_REACH, level.scale)
        if min(gi.shape) < 2:
            continue
        offsets = np.stack([gi, gj], axis=-1).astype(np.int64)
        corners, at = surface.face_cells(sides, grid, offsets, rem)
        if sides == 4:
            kinds, cell = np.full(len(at), SQUARE, np.int8), at[:, 0]
        else:
            # face_cells() lists a triangle's up cells (corner 0 the square cell's) then its down ones (corner 0 the
            # square cell's (I + 1, J))
            total = offsets[:-1, :-1].sum(axis=-1)
            ups = int((total + 1 <= rem).sum())
            kinds = np.where(np.arange(len(at)) < ups, UP, DOWN).astype(np.int8)
            cell = at[:, 0] - np.where(kinds == DOWN, 1, 0)[:, None] * np.array([1, 0])
        middle = corners.mean(axis=1)
        radius = np.linalg.norm(corners - middle[:, None], axis=-1).max(axis=1)
        with np.errstate(over="ignore", invalid="ignore"):
            near = np.linalg.norm(middle, axis=1) - radius <= window
        if not near.any():
            continue
        out_face.append(np.full(int(near.sum()), f, dtype=np.int64))
        out_kind.append(kinds[near])
        out_coef.append(cell[near])
        # The faces across the sides its kept cells reach (their corners on that side), as build_floor() goes on: a
        # side as an axis of the offsets from (IR, JR) (None: I + J, a triangle's long side) and where it is along it
        kept = at[near].reshape(-1, 2)
        if sides == 4:
            reach = ((1, -JR), (0, M - IR), (1, M - JR), (0, -IR))
        else:
            reach = ((1, -JR), (None, M - IR - JR), (0, -IR))
        for s, (axis, where) in enumerate(reach):
            if abs(where) > D + 1:
                continue
            g = top.across(f, s)
            if g is None or g in done:
                continue
            along = kept.sum(axis=1) if axis is None else kept[:, axis]
            rows = np.flatnonzero(along == where)
            if not len(rows):
                continue
            I, J = IR + int(kept[rows[0], 0]), JR + int(kept[rows[0], 1])
            queue.append((g, *top.param_in(top.key(f, I, J, M), M, g)))
    if out_face:
        face = np.concatenate(out_face)
        coef = np.concatenate(out_coef)
        count = len(face)
        level.set_tiles(face, np.concatenate(out_kind), np.zeros(count, np.int8), np.zeros(count, bool), coef)
        level.roots = {f: r for f, r in level.roots.items() if f in set(face.tolist())}
    level.window = window
    return level


def inside(world, faces, below, d, motif):
    """Whether the point below is in one of level d's pieces (a giant level's, surface_substitution.giant_tint()), as
    (None, or (whether it's an outline strip, its paint), how far it is from any piece's edge in its cell or its face's
    sides, map units at least). Its place in the level's cells is exact however deep (Below.cell_place()), and since
    every piece is cut to its cell, the cell it's in holds all that can hold it."""
    top = world.topology
    f = below.face
    N, M = world.N, world.across(d)
    u, v = below.cell_place(d)
    I, J = min(max(math.floor(u), 0), M - 1), min(max(math.floor(v), 0), M - 1)
    here = np.array([float(u - I), float(v - J)])
    kind = SQUARE if top._sides[f] == 4 else (UP if here.sum() <= 1.0 else DOWN)
    with np.errstate(over="ignore"):
        per = min(world.rho * faces.rule.shrink_float(d, world.n) * max(float(faces.sigma[f]), 1e-300), 1e300)
        unit = min(world.rho * faces.rule.shrink_float(0, world.n) * float(faces.sigma[f]), 1e300)
    fu = surface.quotient(below.base[0], N) + float(below.d[0]) * (1 / N)
    fv = surface.quotient(below.base[1], N) + float(below.d[1]) * (1 / N)
    sides = (fu, 1.0 - fu, fv, 1.0 - fv) if kind == SQUARE else (fu, fv, 1.0 - fu - fv)
    clearance = max(min(sides), 0.0) * unit
    polygons, counts = motif.bars[kind]
    found = None
    for size in np.unique(counts).tolist():
        which = np.flatnonzero(counts == size)
        c = polygons[which, :size] - here
        e = np.roll(c, -1, axis=1) - c
        # Signed distance inside each side (counterclockwise): the least is the distance to the edge inside, and
        # outside, the most a side is passed by bounds the distance from below (convex pieces)
        least = ((e[..., 0] * -c[..., 1] - e[..., 1] * -c[..., 0]) / np.maximum(np.hypot(e[..., 0], e[..., 1]),
                                                                              1e-300)).min(axis=1)
        clearance = min(clearance, float(np.abs(least).min()) * per)
        hits = np.flatnonzero(least >= 0.0)
        if len(hits) and found is None:
            i = which[hits[0]]
            found = (bool(motif.edges[kind][i]), int(motif.paints[kind][i]))
    return found, clearance


def forget():
    """After loading a file (surface_substitution.forget()): another file's meshes don't apply"""
    _faces.clear()
