"""Endless digit fractals: endless maps (see endless.py) of a fractal that fills space, like a Sierpinski tetrahedron or
Cantor dust, built cell by cell around the drone, which auto scale can measure the distance to in all three axes.

The fractal lives on a cube lattice. Every cell of level k splits into factor x factor x factor cells of level k + 1,
and only some of them, the "kept" digits, are part of it: the 4 of 8 on alternate corners for the Sierpinski
tetrahedron, the 8 corners of 27 for Cantor dust. A cell is part of the fractal if it and every bigger cell holding it
is kept, which depends on its index's digits only, at every level alike. So the fractal is the same after a loop about
any point, once the anchor scales with it (which endless.py does), and moving by whole steps only moves it. Unlike the
Inferno, whose small levels repeat over a step, no part of it repeats: what's near the drone depends on the anchor
all the way up. So every level is built here, from the anchor, exactly.

What's drawn: each cell opens up as the drone gets near it, by its distance in its own cells, r. Beyond r_out cells
it's closed, one solid shape (a tetrahedron, a gem). Within r_in it's fully open, and its kept children draw
themselves instead. In between, its children are drawn and the cell's middle shrinks away (the Sierpinski
tetrahedron's octahedron between its 4 corner tetrahedra), or its children bud out of it as it shrinks (Cantor dust's
gems). Openness goes by r alone, so a loop, which scales the drone and every cell together, doesn't change it, and
nothing ever pops: a closed shape looks the same as the children and middle that replace it. Each child starts opening
only once its parent is fully open, so a cell opens, then its children, and so on, down to cells about a pixel wide.
The map's geometry nodes do this per cell from where its eye object is, the drone camera's place in the map and its
scale; this module only works out which cells there are, and moves the eye (follow()). It moves it once a frame (the
frame change handler follow_eyes()) and when a loop or the drone moving back by steps relabels the level objects, with
them. The eye used to be driven to the camera, and then every time the add-on brought the camera's matrix up to date
(view_layer.update(), two to five times a tick), every level's geometry nodes ran again, about 8 ms each time for the
Sierpinski tetrahedron: ticks took 12 ms and more, the viewport ran in the teens, and ticks with a loop and a recenter
stalled. For the same reason, the level objects aren't moved where a loop, a move back or a rebuild puts them mid-tick:
settle() puts them there, and the eye with them, at the tick's end, so their geometry nodes run once, for what's drawn.

Each level's cells are in one of the level objects, children of the map's object: a point per cell within reach of
the drone (in the level's own cells, from a corner near the drone, so float32 is exact), with its parent's center and
both their colors. A level object's mesh is rebuilt once the drone is "margin" cells from where it was built for. A
loop turns every level into the next, the same cells only a factor bigger or smaller, so it only relabels the level
objects and builds the one new level at the end (scale_frames()); the drone moving back by whole steps only moves
them (shift_frames()).

Auto scale measures to what's drawn: in mesh mode, an endless fractal's selected object reports the closest point on
its visible shapes (closest_point()), found by searching down from the anchor's cells. It's continuous as the drone
flies, since the shapes change continuously with where it is, and the same after a loop, so the drone's scale is too.

The map's "endless" settings hold "fractal" (written by its build script):
- kept: the kept digits, [x, y, z] each from 0 to factor - 1
- cell: level 0's cell size. Level k's is cell / factor^k.
- open: [r_in, r_out]. Children must not open before their parent is fully open: r_out <= factor * (r_in - 0.58).
- closed, middle: the shapes a closed cell and an opening cell's middle draw, lists of {"kind", "size"} in cells,
  centered on the cell: kind "tetra", the regular tetrahedron on alternate corners of a cube size to each side; "octa",
  the octahedron with corners size out along the axes; "box", a cube size to each side, or with size [x, y, z], a box
  that far to each side along each axis (the slabs of Cantor cubes' cross).
- bud: closed children grow out of their parent's center as it opens, rather than being there all along
- margin: how many of its cells the drone moves before a level is rebuilt
- window: how far from the drone cells are built, past what the fog lets you see
- finest: the smallest cells built. Closer to the fractal than that (flying into it without its config), what's
  closest stays closed.
- palette: a color (linear RGBA) per kept digit
- eye: the empty the geometry nodes draw from, a child of the map's object, which follow() moves. Maps built before
  2026-10-30 have none, their eye driven to the camera instead.
"""

import math
import random
from fractions import Fraction

import bpy
import numpy as np
from bpy.app.handlers import persistent

from . import endless

POOL = "fractal_pool"       # on a level object: it's one of the map's level objects
LEVEL = "fractal_level"     # on a level object: the level it holds, absent while it's free
FRAME = "fractal_frame"     # on a level object: its corner from the map's origin, exactly, as text
SQRT3 = math.sqrt(3.0)

# alive() by (factor, kept digits, index): the same at every level and for every anchor, so kept between calls
_alive_cache = {}
# Lattice() by object and anchor, and the last visible distance found by object (see closest_point()), and after a
# loop, what it is now (scale_frames())
_lattices = {}
_distances = {}
_guesses = {}
# Level objects whose frame or points changed, to be put in place, and maps whose eye must move with them (settle())
_unplaced = set()
_eye_due = set()


def is_fractal(obj):
    return endless.is_endless(obj) and "fractal" in obj[endless.PROPERTY]


def openness(r, r_in, r_out):
    """How open a cell is, 0 (closed) to 1, r cells from it: smoothstep from r_out down to r_in. The map's geometry
    nodes work it out the same way."""
    t = np.clip((r_out - r) / (r_out - r_in), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class Lattice:
    """An endless fractal map's lattice with its anchor as it is: exact cell sizes, and where the map's origin is in
    each level's cells. Cell indices are kept relative to the cell holding the origin (the level's "base"), so they
    stay small near the drone, wherever the anchor is."""

    def __init__(self, obj):
        settings = obj[endless.PROPERTY]
        fractal = settings["fractal"]
        self.factor = int(settings["factor"])
        self.step = endless.exact(settings["step"])
        self.cell = endless.exact(fractal["cell"])
        self.kept = np.array([[int(v) for v in d] for d in fractal["kept"]], dtype=np.int64)
        self.kept_index = {tuple(d): i for i, d in enumerate(self.kept.tolist())}
        self.r_in, self.r_out = (float(v) for v in fractal["open"])
        self.window = float(fractal["window"])
        self.finest = float(fractal["finest"])
        self.margin = float(fractal["margin"])
        self.bud = bool(fractal.get("bud", False))
        self.closed = [shape_setting(shape) for shape in fractal["closed"]]
        self.middle = [shape_setting(shape) for shape in fractal["middle"]]
        self.palette = np.array([[float(c) for c in color] for color in fractal["palette"]], dtype=np.float32)
        self.origin = endless.anchor_position(obj, endless.get_anchor(obj))
        self.key = (self.factor, tuple(map(tuple, self.kept.tolist())))
        self._bases = {}

    def size(self, level):
        return self.cell / Fraction(self.factor) ** level

    def base(self, level):
        """The index of level's cell that holds the map's origin, along each axis, and how far into it the origin is,
        in cells: exactly, and as floats"""
        if level not in self._bases:
            size = self.size(level)
            index, into = [], []
            for o in self.origin:
                q = o / size
                index.append(math.floor(q))
                into.append(q - math.floor(q))
            self._bases[level] = (index, into, np.array([float(v) for v in into]))
        return self._bases[level]

    def alive(self, index):
        """Whether the cell with this index (an absolute one, Python integers) is part of the fractal: its digit, and
        every bigger cell's holding it, is kept. Far enough up, the index is 0 or -1 along each axis, and its digit
        stays the same from there on (0 or factor - 1)."""
        key = (self.key, tuple(index))
        if key not in _alive_cache:
            if len(_alive_cache) > 100000:
                _alive_cache.clear()
            n, f, alive = list(index), self.factor, True
            while True:
                if tuple(v % f for v in n) not in self.kept_index:
                    alive = False
                    break
                if all(v in (0, -1) for v in n):
                    break
                n = [v // f for v in n]
            _alive_cache[key] = alive
        return _alive_cache[key]

    def top_level(self, extent):
        """The finest level whose cells are at least 2 * extent across: at most 2 x 2 x 2 of them reach within extent
        of any point"""
        level = math.floor(math.log(float(self.cell) / (2.0 * extent), self.factor))
        while self.size(level) < 2 * Fraction(repr(extent)):
            level -= 1
        return level

    def top(self, level, point, radius):
        """The cells of level that are part of the fractal and whose cubes come within radius of point (from the map's
        origin): their indices from the base, their digits and their parents' (as indices into kept)"""
        size = float(self.size(level))
        base, _into, into = self.base(level)
        lo = np.floor((point - radius) / size + into).astype(np.int64)
        hi = np.floor((point + radius) / size + into).astype(np.int64)
        f = self.factor
        cells, digits, parents = [], [], []
        for i in range(lo[0], hi[0] + 1):
            for j in range(lo[1], hi[1] + 1):
                for k in range(lo[2], hi[2] + 1):
                    index = [b + int(v) for b, v in zip(base, (i, j, k))]
                    if self.alive(index):
                        cells.append((i, j, k))
                        digits.append(self.kept_index[tuple(v % f for v in index)])
                        parents.append(self.kept_index[tuple((v // f) % f for v in index)])
        cells = np.array(cells, dtype=np.int64).reshape(-1, 3)
        digits, parents = np.array(digits, dtype=np.int64), np.array(parents, dtype=np.int64)
        near = self.cube_distance(level, cells, point) <= radius
        return cells[near], digits[near], parents[near]

    def children(self, level, cells, digits):
        """The kept children, at level + 1, of cells of level: their indices, digits and parents' digits"""
        f = self.factor
        base, _into, _ = self.base(level)
        child_base, _into, _ = self.base(level + 1)
        # Where the children's base is in the parents' base, times the factor: 0 unless the origin isn't on a corner
        delta = np.array([f * b - cb for b, cb in zip(base, child_base)], dtype=np.int64)
        count = len(self.kept)
        children = (f * cells[:, None, :] + self.kept[None, :, :] + delta).reshape(-1, 3)
        return children, np.tile(np.arange(count), len(cells)), np.repeat(digits, count)

    def centers(self, level, cells):
        """The cells' centers from the map's origin"""
        _base, _into, into = self.base(level)
        return (cells + 0.5 - into) * float(self.size(level))

    def parent_offsets(self, digits):
        """From a cell's center to its parent's, in the cell's own cells"""
        return (self.factor - 1) / 2.0 - self.kept[digits]

    def cube_distance(self, level, cells, point):
        """How far point is from each cell's cube, 0 inside it"""
        size = float(self.size(level))
        _base, _into, into = self.base(level)
        low = (cells - into) * size
        gap = np.maximum(np.maximum(low - point, point - (low + size)), 0.0)
        return np.linalg.norm(gap, axis=1)


def shape_setting(shape):
    """A "closed" or "middle" shape as (kind, size): size a float, or for a box with a size per axis, an array (3,)"""
    size = shape["size"]
    if hasattr(size, "__len__"):
        return str(shape["kind"]), np.array([float(v) for v in size])
    return str(shape["kind"]), float(size)


def lattice(obj):
    key = (obj.name, obj.get(endless.ANCHOR, ""), str(obj[endless.PROPERTY]["fractal"].to_dict()))
    if key not in _lattices:
        if len(_lattices) > 8:
            _lattices.clear()
        _lattices[key] = Lattice(obj)
    return _lattices[key]


def local_point(obj, world_location):
    """world_location from the map's origin, in its own coordinates, as a float64 array"""
    return np.array(obj.matrix_world.inverted() @ world_location, dtype=np.float64)


# --- The shapes cells draw, and the distance to them --------------------------------------------------------------

def tetra_corners(half):
    """The regular tetrahedron on alternate corners of a cube half to each side (the kept corners of the Sierpinski
    tetrahedron's 2 x 2 x 2 cells), as (n, 4, 3) for halves of shape (n,)"""
    corners = np.array([(-1, -1, -1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)], dtype=np.float64)
    return corners[None, :, :] * half[:, None, None]


def closest_on_triangles(p, a, b, c):
    """The closest point on each triangle abc to each point p, all (n, 3) (Ericson, Real-Time Collision Detection 5.1.5)"""
    dot = lambda u, v: np.einsum('ij,ij->i', u, v)
    ab, ac, ap, bp, cp = b - a, c - a, p - a, p - b, p - c
    d1, d2, d3, d4, d5, d6 = dot(ab, ap), dot(ac, ap), dot(ab, bp), dot(ac, bp), dot(ab, cp), dot(ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide='ignore', invalid='ignore'):
        total = va + vb + vc
        result = a + ab * (vb / total)[:, None] + ac * (vc / total)[:, None]
        # From the face's inside out to the edges and corners, each overriding the last
        bc = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        result = np.where(bc[:, None], b + (c - b) * w[:, None], result)
        on_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        result = np.where(on_ac[:, None], a + ac * (d2 / (d2 - d6))[:, None], result)
        result = np.where(((d6 >= 0) & (d5 <= d6))[:, None], c, result)
        on_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        result = np.where(on_ab[:, None], a + ab * (d1 / (d1 - d3))[:, None], result)
        result = np.where(((d3 >= 0) & (d4 <= d3))[:, None], b, result)
        result = np.where(((d1 <= 0) & (d2 <= 0))[:, None], a, result)
    return result


def tetra_closest(p, half):
    """The closest point on each tetrahedron (see tetra_corners()) centered on the origin to each point p (n, 3): the
    point itself when it's inside"""
    corners = tetra_corners(half)
    best, best_distance = p.copy(), np.full(len(p), np.inf)
    for face in ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)):
        q = closest_on_triangles(p, *(corners[:, i] for i in face))
        d = np.linalg.norm(p - q, axis=1)
        better = d < best_distance
        best[better], best_distance[better] = q[better], d[better]
    # Inside: on the inner side of all four faces, x + y + z <= h and the like
    signs = np.array([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)], dtype=np.float64)
    inside = np.all(p @ signs.T <= half[:, None], axis=1)
    best[inside] = p[inside]
    return best


def box_closest(p, half):
    """The closest point on each box centered on the origin to each point p (n, 3): half (n,) for cubes, (n, 3) for a
    half size per axis"""
    half = half if half.ndim == 2 else half[:, None]
    return np.clip(p, -half, half)


def octa_closest(p, size):
    """The closest point on each octahedron |x| + |y| + |z| <= size to each point p (n, 3): projected onto its face's
    plane, then onto the face's triangle"""
    signs = np.where(p < 0, -1.0, 1.0)
    q = np.abs(p)
    corners = np.eye(3)[None, :, :] * size[:, None, None]
    closest = closest_on_triangles(q, corners[:, 0], corners[:, 1], corners[:, 2])
    inside = q.sum(axis=1) <= size
    closest[inside] = q[inside]
    return closest * signs


CLOSEST = {"tetra": tetra_closest, "box": box_closest, "octa": octa_closest}


def shapes_closest(shapes, p, sizes):
    """The closest point on a cell's shapes (see the "closed" and "middle" settings) to each point p, from the shapes'
    center, for cells sizes across: (closest points, distances)"""
    best, best_distance = None, None
    for kind, size in shapes:
        q = CLOSEST[kind](p, size[None, :] * sizes[:, None] if isinstance(size, np.ndarray) else size * sizes)
        d = np.linalg.norm(p - q, axis=1)
        if best is None:
            best, best_distance = q, d
        else:
            better = d < best_distance
            best[better], best_distance[better] = q[better], d[better]
    return best, best_distance


def drawn(lat, level, cells, digits, eye):
    """What each of a level's cells draws, seen from eye: (closed, middle), each (selection, centers, sizes), and every
    cell's openness. Like the map's geometry nodes."""
    size = float(lat.size(level))
    centers = lat.centers(level, cells)
    o = openness(np.linalg.norm(centers - eye, axis=1) / size, lat.r_in, lat.r_out)
    parents = centers + lat.parent_offsets(digits) * size
    o_parent = openness(np.linalg.norm(parents - eye, axis=1) / (lat.factor * size), lat.r_in, lat.r_out)
    closed = (o_parent > 0) & (o <= 0)
    if lat.bud:
        closed_at, closed_size = parents + (centers - parents) * o_parent[:, None], o_parent * size
    else:
        closed_at, closed_size = centers, np.full(len(cells), size)
    middle = (o > 0) & (o < 1)
    return (closed, closed_at, closed_size), (middle, centers, (1.0 - o) * size), o


def search(lat, eye, bound):
    """The closest point to eye on any shape drawn within bound of it, and its distance, or (None, bound): from the
    biggest cells within bound down, opening only cells that are open and may hold something closer than the closest
    found yet"""
    level = lat.top_level(bound)
    cells, digits, _parents = lat.top(level, eye, bound)
    best, best_distance = None, bound
    while len(cells):
        (closed, closed_at, closed_size), (middle, centers, middle_size), o = drawn(lat, level, cells, digits, eye)
        for selection, at, sizes, shapes in ((closed, closed_at, closed_size, lat.closed),
                                             (middle, centers, middle_size, lat.middle)):
            if selection.any():
                q, d = shapes_closest(shapes, eye - at[selection], sizes[selection])
                i = int(np.argmin(d))
                if d[i] < best_distance:
                    best, best_distance = q[i] + at[selection][i], float(d[i])
        # Only open cells have children drawn, and only within the closest found can they be closer
        expand = o > 0
        cells, digits = cells[expand], digits[expand]
        near = lat.cube_distance(level, cells, eye) < best_distance
        cells, digits = cells[near], digits[near]
        if not len(cells):
            break
        cells, digits, _parents = lat.children(level, cells, digits)
        level += 1
    return best, best_distance


def search_near(lat, eye, guess):
    """search() within twice guess (a distance it's likely about), then further out until something's found. Searching
    the whole window, every open cell in it is opened down to where shapes are drawn: tens of milliseconds."""
    bound = min(2.0 * guess, lat.window) if guess and guess > 0 else lat.window
    while True:
        point, distance = search(lat, eye, bound)
        if point is not None or bound >= lat.window:
            return point, distance
        bound = min(4.0 * bound, lat.window)


def visible_distance(obj, eye):
    """The closest point on the shapes the fractal draws to eye (from the map's origin) and its distance. Searched near
    the last distance found, since the drone only moves a little between ticks, scaled by a loop if one came between
    (scale_frames()). Guessed a factor further every time, the search covered 4 times the distance, 64 times the cells,
    every tick, and 8 times after a climbing loop."""
    lat = lattice(obj)
    key = (obj.name, obj.get(endless.ANCHOR, ""), tuple(eye))
    cached = _distances.get(obj.name)
    if cached and cached[0] == key:
        return cached[1], cached[2]
    guess = _guesses.pop(obj.name, cached[2] if cached else None)
    point, distance = search_near(lat, eye, guess)
    _distances[obj.name] = (key, point, distance)
    return point, distance


def distance_at_most(obj, eye):
    """A distance from eye to the shapes drawn, never more than the true one, which is all choosing the levels to build
    needs (level_range()): the last one found (by auto scale, earlier in the tick) less how far the eye has moved since,
    if that's from the anchor as it is and little has; otherwise found anew. A search of its own every tick, from where
    the drone ended up, took a third of the tick."""
    cached = _distances.get(obj.name)
    if cached and cached[0][1] == obj.get(endless.ANCHOR, "") and cached[2] > 0:
        moved = float(np.linalg.norm(eye - np.array(cached[0][2])))
        if moved < 0.5 * cached[2]:
            return cached[2] - moved
    return visible_distance(obj, eye)[1]


def closest_point(obj, world_location):
    """The closest point to world_location on what the fractal draws, in world space, for auto scale's mesh mode: what
    auto scale keeps the drone's scale proportional to the distance to, like a mesh's surface. The map's origin if
    nothing is drawn within its window."""
    eye = local_point(obj, world_location)
    point, _distance = visible_distance(obj, eye)
    if point is None:
        return obj.matrix_world.translation.copy()
    return obj.matrix_world @ type(world_location)(point.tolist())


# The most steps crossing() marches. A flight is only marched at all when it's longer than half the distance to the
# nearest shape, and then for about as many steps as it's longer: up to 64 at first, every tick, for a drone scaled up
# by hand near the fractal (which auto scale keeps), 37 searches and 80 ms a tick on average, and Blender froze for
# seconds. A drone that far off the usual ratio has everything that near inside its clip start, out of sight.
MARCH_STEPS = 8


def crossing(obj, start_world, end_world):
    """If flying from start_world to end_world goes through a drawn shape, how far along it that is from each end (in
    the map's units), for stop_at_surface(). Otherwise None. Marched from the start by the distance to the closest
    shape each time, which nothing drawn is nearer than, so it can't step past one. Each step's shapes are what's
    drawn seen from there. A flight shorter than half the distance at its start can't reach anything, which is nearly
    every tick's."""
    start, end = local_point(obj, start_world), local_point(obj, end_world)
    length = float(np.linalg.norm(end - start))
    if length == 0.0:
        return None
    lat = lattice(obj)
    _point, first = visible_distance(obj, start)
    if length < 0.5 * first:
        return None
    direction = (end - start) / length
    # Within a thousandth of the distance it started at, it's hit: from much closer, auto scale would never get it
    # there flying normally. Passing a shape at a grazing angle, the steps shrink without reaching it: let it pass.
    travelled, distance = first, first
    safe = 0.0
    for _step in range(MARCH_STEPS):
        if travelled >= length:
            return None
        _point, distance = search_near(lat, start + direction * travelled, distance)
        if distance <= 1e-3 * first:
            return travelled, length - travelled
        safe = travelled
        travelled += distance
    if travelled >= length:
        return None
    # Not settled in MARCH_STEPS: unless the flight ends inside a shape, let it pass; if it does, stop where the march
    # had got to, clear of everything
    _point, distance = search_near(lat, end, distance)
    if distance <= 1e-3 * first:
        return safe, length - safe
    return None


# --- Level objects ------------------------------------------------------------------------------------------------

def level_objects(obj):
    return [child for child in obj.children if POOL in child]


def frame_of(level_obj):
    return [Fraction(v) for v in level_obj[FRAME].split()]


def set_frame(level_obj, frame):
    level_obj[FRAME] = " ".join(str(v) for v in frame)


def level_range(lat, distance, near=0.0, pool=None):
    """The levels that can draw something: from the coarsest whose cells open within the window (drawn from r_in of
    their cells out) to the finest whose active cells (within factor * (r_out + 1) of their cells) can reach the
    closest shape, distance away, or the camera's near clip, near away (in the map's units), whichever is further,
    and one more for the drone getting closer. None finer than the finest setting, and no more than the pool of
    level objects holds, the finest left out first.

    By the distance alone, a drone scaled up by hand near a surface (which auto scale keeps) had it far inside its
    clip start, where nothing is drawn: levels down to the finest, 15 micrometers, were built and drawn for nothing,
    24 levels and 240,000 faces, rebuilt as fast as the drone moved, and Blender froze for seconds at a time. The pool
    ran out, and the coarsest levels, the biggest shapes, were left out."""
    f = lat.factor
    coarse = math.ceil(math.log(float(lat.cell) * lat.r_in / lat.window, f))
    reach = f * (lat.r_out + 1.0) + 2.0
    distance = max(distance, near, lat.finest * reach)
    fine = math.floor(math.log(float(lat.cell) * reach / distance, f)) + 1
    fine = min(fine, math.floor(math.log(float(lat.cell) / lat.finest, f)))
    if pool is not None:
        fine = min(fine, coarse + pool - 1)
    return coarse, max(coarse, fine)


def build_level(lat, level_obj, level, eye):
    """Build level's cells into level_obj: every one part of the fractal whose center is within reach of the drone
    for as long as it stays within margin cells of eye, as points in the level's own cells from the corner of the cell
    eye is in. Each point has its parent's center ("pc", in the same units) and both their colors ("col", "pcol")."""
    f = lat.factor
    size = lat.size(level)
    base, into_exact, into = lat.base(level)
    corner = np.floor(eye / float(size) + into).astype(np.int64)
    frame = [(int(c) - i) * size for c, i in zip(corner, into_exact)]
    frame_float = np.array([float(v) for v in frame])
    # Children are active while their parent is opening, within r_out of its cells, so within f * (r_out + 1) of
    # theirs
    radius_cells = f * (lat.r_out + 1.0) + lat.margin
    radius = radius_cells * float(size)
    top = lat.top_level(radius)
    cells, digits, parents = lat.top(top, frame_float, radius)
    for k in range(top, level):
        near = lat.cube_distance(k, cells, frame_float) <= radius
        cells, digits = cells[near], digits[near]
        cells, digits, parents = lat.children(k, cells, digits)
    points = (cells - corner + 0.5).astype(np.float64)
    near = np.linalg.norm(points, axis=1) <= radius_cells
    points, digits, parents = points[near], digits[near], parents[near]
    parent_points = points + lat.parent_offsets(digits)

    level_obj[LEVEL] = level
    set_frame(level_obj, frame)
    write_points(level_obj.data, points, parent_points, lat.palette[digits], lat.palette[parents])
    _unplaced.add(level_obj.name)


def write_points(mesh, points, parent_points, colors, parent_colors):
    mesh.clear_geometry()
    count = len(points)
    mesh.vertices.add(count)
    mesh.vertices.foreach_set("co", points.astype(np.float32).ravel())
    for name, kind, values, key in (("pc", 'FLOAT_VECTOR', parent_points, "vector"), ("col", 'FLOAT_COLOR', colors, "color"),
                                    ("pcol", 'FLOAT_COLOR', parent_colors, "color")):
        attribute = mesh.attributes.get(name) or mesh.attributes.new(name, kind, 'POINT')
        attribute.data.foreach_set(key, np.asarray(values, dtype=np.float32).ravel())
    mesh.update()


def clear_level(level_obj):
    if LEVEL in level_obj:
        del level_obj[LEVEL]
    if len(level_obj.data.vertices):
        level_obj.data.clear_geometry()
        level_obj.data.update()


def place(lat, level_obj):
    """Put a level object where its frame is, scaled to its level's cells"""
    size = float(lat.size(level_obj[LEVEL]))
    level_obj.location = [float(v) for v in frame_of(level_obj)]
    level_obj.scale = (size, size, size)


def refresh(obj, world_location, rebuild=False):
    """Bring the level objects up to date for the drone camera at world_location: a level object for every level that
    can draw something there, each rebuilt if the drone has gone margin cells from where it was built (or rebuild),
    the rest freed. Cheap when nothing needs building."""
    lat = lattice(obj)
    eye = local_point(obj, world_location)
    pool = level_objects(obj)
    # Nothing nearer than the drone camera's clip start is drawn, in the map's units
    camera = drone_camera(bpy.context.scene)
    near = camera.data.clip_start / obj.matrix_world.to_scale()[0] if camera else 0.0
    coarse, fine = level_range(lat, distance_at_most(obj, eye), near, len(pool))
    by_level = {o[LEVEL]: o for o in pool if LEVEL in o and coarse <= o[LEVEL] <= fine}
    free = [o for o in pool if o not in by_level.values()]
    for o in free:
        clear_level(o)
    for level in range(coarse, fine + 1):
        level_obj = by_level.get(level)
        if level_obj is None:
            if not free:
                print(f"{obj.name}: no level object free for level {level}")
                continue
            level_obj = free.pop()
        elif not rebuild:
            size = float(lat.size(level))
            frame = np.array([float(v) for v in frame_of(level_obj)])
            if np.linalg.norm(eye - frame) / size <= lat.margin:
                continue
        build_level(lat, level_obj, level, eye)


def follow(obj, camera_matrix):
    """Put the map's eye (the "eye" setting) where the drone camera is, camera_matrix its world matrix: its place in
    the map and its scale. Only if it moved, since setting it re-runs every level's geometry nodes."""
    eye = obj[endless.PROPERTY]["fractal"].get("eye")
    if eye is None:
        return
    location = obj.matrix_world.inverted() @ camera_matrix.translation
    scale = sum(camera_matrix.to_scale()) / 3.0
    if eye.location != location or eye.scale[0] != scale:
        eye.location = location
        eye.scale = (scale, scale, scale)


def settle(obj, camera_matrix):
    """Put the level objects whose frame or points changed (a loop, a move back by steps, a rebuild) where their
    frames now are, and after a loop or a move back, the eye with them (follow()), camera_matrix the drone camera's
    world matrix. Once, at the end of the tick: the geometry nodes then run once for it, rather than each time the
    add-on brings the camera's matrix up to date mid-tick, and the eye and the levels change together, so loops stay
    seamless."""
    lat = lattice(obj)
    for level_obj in level_objects(obj):
        if level_obj.name in _unplaced:
            _unplaced.discard(level_obj.name)
            if LEVEL in level_obj:
                place(lat, level_obj)
    if obj.name in _eye_due:
        _eye_due.discard(obj.name)
        follow(obj, camera_matrix)


def drone_camera(scene):
    """The drone's camera, the Drone object's camera child, or None"""
    drone = getattr(getattr(scene, "references", None), "drone_obj", None)
    return next((child for child in drone.children if child.type == 'CAMERA'), None) if drone else None


@persistent
def follow_eyes(scene, _depsgraph=None):
    """frame_change_pre: every endless fractal's eye to the drone camera (follow()), and anything left to put in place
    (settle()), once a frame, flying or not, so the frame's own evaluation runs the geometry nodes with it. The camera
    is where the last tick left it."""
    camera = drone_camera(scene)
    if camera is None:
        return
    for obj in scene.objects:
        if is_fractal(obj):
            settle(obj, camera.matrix_world)
            follow(obj, camera.matrix_world)


def scale_frames(obj, power, anchor):
    """A loop by factor^power about the map's origin, with the anchor as it was before: every level object now holds
    the level power bigger (or smaller), the same cells, and its frame scales with them. Scaling down, the drone also
    moves by the anchor's remainder (see endless.scale_anchor()), and so do they."""
    f, step = int(obj[endless.PROPERTY]["factor"]), endless.exact(obj[endless.PROPERTY]["step"])
    for level_obj in level_objects(obj):
        if LEVEL not in level_obj:
            continue
        frame = frame_of(level_obj)
        if power > 0:
            frame = [v * f ** power for v in frame]
        else:
            divisor = f ** -power
            frame = [v / divisor + Fraction(a % divisor) * step / divisor for v, a in zip(frame, anchor)]
        level_obj[LEVEL] = level_obj[LEVEL] - power
        set_frame(level_obj, frame)
    for level_obj in level_objects(obj):
        if LEVEL in level_obj:
            _unplaced.add(level_obj.name)
    _eye_due.add(obj.name)
    # The distance to what's drawn scaled with everything
    if obj.name in _distances:
        _guesses[obj.name] = _distances[obj.name][2] * float(f) ** power


def shift_frames(obj, steps):
    """The map's origin moved by steps along each axis (the drone moving back toward it): every level object's frame
    moves back with the drone"""
    step = endless.exact(obj[endless.PROPERTY]["step"])
    for level_obj in level_objects(obj):
        if LEVEL in level_obj:
            set_frame(level_obj, [v - s * step for v, s in zip(frame_of(level_obj), steps)])
            _unplaced.add(level_obj.name)
    _eye_due.add(obj.name)


def keep_background(anchor, background, last, top, factor):
    """The anchor with its background's digits (the random settings' "background", kept digits, one per axis, see
    endless.py) from just above the made-up ones to top, in the lattice's own digits, 0 to factor - 1: a climb moves
    them all down a place, so the top one is continued from the pattern below it, and an anchor without them gets the
    pattern. Without it, the digits above are the same forever, 0 (or factor - 1 below the lattice's origin), and in
    Cantor dust, where a cell of all 0 digits meets one of all 2 (the lattice's origin, along each axis), both are kept,
    at every level: their cubes touched, mirror images of each other, at every scale. Climbing near it, the digits made
    up stayed as they were (choose_digit(): what's in view mustn't change), so it stayed in view. Alternating 0 and 2
    keeps the drone a quarter of every bigger cell from its sides, and the lattice's origin a dozen levels out of sight.
    (The Sierpinski tetrahedron needs none: its kept cells around the origin are a bigger one's 4 corners.)"""
    period = len(background)
    digit = lambda a, place: (a // factor ** place) % factor
    pattern = lambda place, phase, axis: int(background[(place - last - 1 + phase) % period][axis])
    kept = []
    for axis, a in enumerate(anchor):
        # Below its top (which a climb has just brought down from above), the pattern in either phase. A climb shifts
        # it a place, so writing it afresh by place flipped its phase every time.
        below_top = range(last + 1, top - period + 1)
        if not any(all(digit(a, place) == pattern(place, phase, axis) for place in below_top) for phase in range(period)):
            for place in range(last + 1, top + 1):
                a += (pattern(place, 0, axis) - digit(a, place)) * factor ** place
        for place in range(top - period + 1, top + 1):
            a += (digit(a, place - period) - digit(a, place)) * factor ** place
        kept.append(a)
    return tuple(kept)


def choose_digit(obj, anchor, place_value, drone):
    """The anchor with a random kept digit at place_value (a power of the factor, in steps), along all three axes
    together, if what's in view stays the same: the drone's window is inside its cell of that size, whose contents
    are the same for every kept digit. Otherwise the anchor as it is, which is what's in view now."""
    settings = obj[endless.PROPERTY]
    f, step = int(settings["factor"]), endless.exact(settings["step"])
    # Plus where the drone may really be: on a loop, drone is where the camera was before it, up to factor steps off
    window = endless.exact(settings["fractal"]["window"]) + 2 * f * step
    size = place_value * step
    inside = all(window < (Fraction(a % place_value) * step + Fraction(repr(d))) % size < size - window
                 for a, d in zip(anchor, drone))
    if not inside:
        return anchor
    new = random.choice([[int(v) for v in d] for d in settings["fractal"]["kept"]])
    return tuple(a + (n - (a // place_value) % f) * place_value for a, n in zip(anchor, new))
