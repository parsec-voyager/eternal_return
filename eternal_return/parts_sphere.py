"""The 3D parts' lattice on a round sphere map (sphere.py): one for both of its layouts, the cube's and the
icosahedron's (2026-11-21, phase 1 of the unified engines: a sphere is a mesh plus a chart). parts.py asks a lattice
what it needs (SphereLattice; a surface map's is surface.MeshLattice); here that's worked out the same way for either
layout, from a chart that says what's the layout's own:

- which points are near which: the layout's polyhedron (the cube [-1, 1]^3's 6 squares, the icosahedron's 20
  triangles) as a surface.Topology, whose keys name a lattice point the same from every face it's on, and whose
  LocalLattice finds a level's points near a point and the edges between them, by the flat polyhedron's distances
  (the chart is a similarity over a cell, away from the corners)
- where a lattice point is on the sphere: from the map's anchor through the conformal chart, without cancellation, in
  float64 (placed()) or with as many digits as the depth needs (the chart's placer: sphere_exact.Placer for the cube,
  parts_ico.IcoPlacer for the icosahedron, which works in the folded anchor's frame and leaves its turn to
  parts.exact_piece())
- where the drone is in a level's cells (below(), cell_place()), and how far the chart takes a cell from a similarity
  there (distortion()), which rule most giant levels out at once

A chart's lattice points, in a world's units (world points): the cube's are three ints or Fractions on its surface
(A, sphere.World's), the icosahedron's a face's corners and whole weights on them (parts_ico). A level's cells are
cell(world, level) units: 2N / 2^level on the cube (its faces are 2N across), N / 2^level on the icosahedron.

Every level's pieces around a lattice point are laid out once, in the level's own world (n = level - shift + REDUCED,
anchored at the point, its cells 2^REDUCED units, rho 1), so their places are a few bits from the point, exact as
floats at any depth (LocalPieces, kept for every point looked at: the chart's local_placed()); the point itself is placed in the map from the
anchor (Pieces.origin()). Until 2026-11-21 the cube had its own lattice of cube coordinates (parts.Lattice, its
levels' points out to a square of cells round the point) and the icosahedron a copy of this module's classes.
"""

import math
from fractions import Fraction

import numpy as np

try:
    from . import parts
    from . import sphere
    from . import sphere_exact
    from . import surface
except ImportError:          # tests import it by itself
    import parts
    import sphere
    import sphere_exact
    import surface

# A level's pieces are placed in a world of the level's own, anchored at the vertex they're around, its cells 2^REDUCED
# units: their lattice points' offsets from it are then a few bits, exact as floats, at any depth and in any level
REDUCED = 40
# The vertex's own place, from the map's anchor, is placed in float64 within N >> NEAR_SHIFT of it and a world under
# 2^NEAR_N (float64's rounding is a share of the offset, about 1e-14 of it, and past 2^NEAR_N units the globe's offsets
# near a float's range), else with digits
NEAR_SHIFT = 6
NEAR_N = 990
# How far the chart takes a cell from a square: under this share, the drone's place in a giant level's cells rules
# pieces in and out; over it (the coarsest levels, and near a corner of the polyhedron) the level is placed in float64
# to see
DISTORTION_LIMIT = 0.25
# How far the sphere curves away under a piece's base, at most, in its cells: a pillar 0.065 cells across on a level's
# cells a fifth of the sphere's radius (level 3), with room
SAG = 0.002


# ----- the cube's chart -----

class CubeChart:
    """The cube sphere's chart (sphere.py): its polyhedron the cube [-1, 1]^3, its world points the cube's points (three
    ints or Fractions, on its surface, N units from its middle along one axis at least)"""

    name = "cube"
    shift = 1                    # a level's cells are 2N / 2^level units

    def __init__(self):
        self._topology = None
        self._sizes = {}

    def topology(self):
        """The cube's 6 faces as a surface.Topology: each a square of the corners (+-1, +-1, +-1), its u and v along
        two axes"""
        if self._topology is None:
            corners = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
            index = {c: i for i, c in enumerate(corners)}
            faces = []
            for axis in range(3):
                a, b = [k for k in range(3) if k != axis]
                for side in (-1, 1):
                    loop = []
                    for ua, ub in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                        c = [0, 0, 0]
                        c[axis], c[a], c[b] = side, ua, ub
                        loop.append(index[tuple(c)])
                    # Outward: counterclockwise seen from outside
                    faces.append(tuple(loop) if side > 0 else tuple(reversed(loop)))
            self._topology = surface.Topology(None, positions=np.array(corners, dtype=float), faces=faces)
        return self._topology

    @staticmethod
    def world(n, A, rho):
        return sphere.World(n, A, rho)

    @staticmethod
    def cell(world, level):
        return Fraction(2 * world.N, 1 << level)

    def key_point(self, key, M):
        """A key of a level of M cells across as a point in the level's cells, from the cube's middle"""
        top = self.topology()
        ints, d = top.key_point(key, M)
        # The cube is 2 across, M cells
        return tuple(Fraction(v * M, 2 * d) for v in ints)

    @staticmethod
    def scaled(point, factor):
        return tuple(Fraction(c) * factor for c in point)

    def key_of(self, point, C, M):
        """A world point (a lattice point of the level of cells C units, M cells across) as the level's key"""
        cells = [Fraction(c) / C for c in point]
        half = Fraction(M, 2)
        top = self.topology()
        for f, loop in enumerate(top._corners):
            p0, eu, ev = top.P0[f], top.eu[f], top.ev[f]
            axis = int(np.flatnonzero(np.abs(np.cross(eu, ev)) > 0.5)[0])
            if cells[axis] != half * int(round(p0[axis])):
                continue
            # Along u and v from the face's corner 0, in cells: eu and ev are 2 long, along one axis each
            ua, va = int(np.flatnonzero(eu)[0]), int(np.flatnonzero(ev)[0])
            I = (cells[ua] - half * int(round(p0[ua]))) * int(round(eu[ua] / 2.0))
            J = (cells[va] - half * int(round(p0[va]))) * int(round(ev[va] / 2.0))
            assert I.denominator == 1 and J.denominator == 1 and 0 <= I <= M and 0 <= J <= M, (point, C, M)
            return top.key(f, int(I), int(J), M)
        raise ValueError(f"parts_sphere: {point} isn't on the cube")

    @staticmethod
    def on_level(world, point, C):
        """Whether a world point is a lattice point of the level of cells C units"""
        return all(((Fraction(c) + world.N) / C).denominator == 1 for c in point)

    def local_placed(self, level, center, lattice):
        """A LocalLattice's points round center on the sphere, in the level's own world (n = level - 1 + REDUCED,
        anchored at center, its cells 2^REDUCED units): (places, normals). The lattice's own places are the flat cube's,
        from center in the level's units (a cell is 2 of them), whole numbers, so they're the offsets from center on the
        cube exactly, as floats: no point needs placing on its own (with Fractions, a level 1000 loops down took 1.2 s)."""
        M = 1 << level
        local = self.world(level - 1 + REDUCED, self.scaled(self.key_point(center, M), 1 << REDUCED), 1.0)
        deltas = np.ldexp(lattice.places, REDUCED - 1)
        return sphere.placed(local, deltas, max(1.0, float(np.abs(deltas).max()) * 2.0))

    @staticmethod
    def placed(world, points):
        """World points on the sphere, map coordinates from the map's origin, and the sphere's unit normals there"""
        deltas = np.array([[float(Fraction(c) - a) for c, a in zip(p, world.A)] for p in points])
        return sphere.placed(world, deltas, max(1.0, float(np.abs(deltas).max()) * 2.0))

    @staticmethod
    def near(world, point):
        """Whether a world point is near enough the map's anchor to be placed from it in float64"""
        return world.n < NEAR_N and max(abs(Fraction(c) - a) for c, a in zip(point, world.A)) <= world.N >> NEAR_SHIFT

    @staticmethod
    def placer(world, digits):
        return sphere_exact.placer(world, digits)

    @staticmethod
    def exact_placed(exact, world, points):
        """World points and their normals placed with a placer's digits (Decimals, the map's axes)"""
        return exact.placed([[Fraction(c) - a for c, a in zip(p, world.A)] for p in points])

    @staticmethod
    def below(world, q):
        return sphere.Below(world, q)

    @staticmethod
    def center_point(below, C):
        """The lattice point of cells C units nearest the point below, as a world point"""
        return below.lattice_point(C)

    @staticmethod
    def to_corner(world, below):
        """The point below's distance from the cube's corner its chart is folded to, in units"""
        N = world.N
        return math.hypot(sphere.wall_float(N - below.base[0]) - below.D.real,
                          sphere.wall_float(N - below.base[1]) - below.D.imag)

    @staticmethod
    def cell_place(world, below, C, level):
        """Where the point below is in its cell of the level of cells C units: its distances from the cell's corners
        (sorted) and from the nearest of its sides, in cells"""
        n = world.n
        fa = float((Fraction(below.base[0]) % C) / C) + math.ldexp(below.D.real, level - n - 1)
        fb = float((Fraction(below.base[1]) % C) / C) + math.ldexp(below.D.imag, level - n - 1)
        fa, fb = fa - math.floor(fa), fb - math.floor(fb)
        corners = sorted(math.hypot(x, y) for x in (fa, 1.0 - fa) for y in (fb, 1.0 - fb))
        return corners, min(fa, 1.0 - fa, fb, 1.0 - fb)

    @staticmethod
    def distortion(n, level, to_corner):
        """How far, at most, the chart takes a cell of a level from a square (a share of its size), for a drone
        to_corner units from the cube's corner: the conformal factor's change over the cell, from the corner's pinch
        (|dlog| ~ 1 / 3 the distance from it) and the face's own (~2 / N), twice that for the turn"""
        return _distortion(n + 1 - level, to_corner, level)

    def cell_sizes(self, world, levels):
        """Each level's cells at the anchor (map units), measured, as {level: size} (kept for the world, with every
        level measured for it before): from the level's lattice point nearest the anchor on the cube to the next one
        along the surface, placed exactly"""
        sizes = _kept_sizes(self._sizes, world)
        levels = [k for k in levels if k not in sizes]
        if levels:
            N = world.N
            points = []
            for level in levels:
                C = self.cell(world, level)
                # A level's points are where (P + N) / C is whole; the face's own axis (+-N) always is
                X = [C * round((a + N) / C) - N for a in world.A]
                # The next one along the surface: along an axis off the face, toward the face's middle (at a cube
                # corner, every axis is a face's)
                axis = min(range(3), key=lambda k: abs(X[k]))
                Y = list(X)
                Y[axis] += -C if X[axis] > 0 else C
                points += [X, Y]
            p, _normals = self.placed(world, points)
            sizes.update(zip(levels, np.linalg.norm(p[1::2] - p[0::2], axis=1).tolist()))
        return sizes

    def forget(self):
        self._sizes.clear()


def _distortion(exponent_base, to_corner, level):
    """distortion()'s: the pinch 2^exponent_base / (3 to_corner), and the face's own 6 / 2^level, twice for the turn"""
    if to_corner == 0.0:
        return math.inf
    pinch = 0.0
    if math.isfinite(to_corner):
        exponent = exponent_base - math.log2(3.0 * to_corner)
        pinch = math.inf if exponent > 60 else 2.0 ** exponent
    return 2.0 * (pinch + 6.0 * 2.0 ** -level) + 1e-9


def _kept_sizes(kept, world):
    text = world.text()
    sizes = kept.get(text)
    if sizes is None:
        if len(kept) > 64:
            kept.clear()
        sizes = kept[text] = {}
    return sizes


# ----- a level's pieces round a lattice point -----

class LocalPieces(parts.LaidOut):
    """A level's pieces around a lattice point (a key of the level), in the level's own world: what doesn't depend on
    the map's anchor, kept for every point looked at (parts._local)"""

    def __init__(self, chart, level, center, radius, settings, shape):
        lattice = self.lattice = surface.LocalLattice(chart.topology(), 1 << level, center, radius)
        places, normals = chart.local_placed(level, center, lattice)
        self.lay_out(lattice.inner, lattice.neighbors, lattice.edges, lattice.center_row, places, normals, settings,
                     shape)
        self.keys = lattice.keys


class Pieces:
    """A level's pieces around center (a key of the level's lattice) out to radius cells: its vertices' and edges'
    pieces, each a place (its vertex, or by its edge's first vertex), up, size, frame (axes as columns), kind, height
    and the point its distance is measured from, and the lattice rows it stands on (a vertex piece's vertex then its
    neighbors, an edge piece's two vertices).

    They're placed in float64 in the level's own world (LocalPieces), from center, so their places and sizes are in its
    map units, each `scale` of the map's (scale_log2 its power of 2 times rho), and center itself is origin() in the
    map. That keeps a coarse level's pieces exact whatever the chart does between the map's anchor and them, and a
    giant one's in floats however deep a dive goes (its cells in map units can be far past a float's range)."""

    def __init__(self, chart, world, level, center, radius, settings, shape):
        self.chart, self.world, self.level, self.center = chart, world, level, center
        self.M = 1 << level
        self.C = chart.cell(world, level)
        self.scale_log2 = world.n - (level - chart.shift + REDUCED)
        key = (chart.name, level, center, radius, parts._has(settings, "vertex"), parts._has(settings, "edge"),
               tuple(shape.heights))
        local = parts._local.get(key)
        if local is None:
            if len(parts._local) > parts.LOCAL_KEPT:
                parts._local.clear()
            local = parts._local[key] = LocalPieces(chart, level, center, radius, settings, shape)
        self.__dict__.update(local.__dict__)
        self.local = local
        self._origin = None
        self._folded = None

    def piece_rows(self, row):
        return self.local.piece_rows(row)

    def scale(self, rho):
        """The map's units per unit of the pieces' places and sizes, as a float (inf past a float's range)"""
        try:
            return math.ldexp(rho, self.scale_log2)
        except OverflowError:
            return math.inf

    def world_point(self, key):
        """A key of the level as a world point"""
        return self.chart.scaled(self.chart.key_point(key, self.M), self.C)

    def origin(self, placer):
        """center in the map (map units from its origin): float64 if near the anchor, else with placer's digits (placer()
        makes it, called only then) as Decimals. A placer that works in a frame of its own (the icosahedron's folded
        anchor's, its turn) keeps center there for camera(), and gives the map's axes by its turn's floats: a place to
        build a level object at, not to measure from."""
        if self._origin is None:
            chart, world = self.chart, self.world
            point = self.world_point(self.center)
            if chart.near(world, point):
                self._origin = chart.placed(world, [point])[0][0]
            else:
                exact = placer()
                self._folded = chart.exact_placed(exact, world, [point])[0][0]
                if getattr(exact, "turn", None) is None:
                    self._origin = self._folded
                else:
                    with exact.context():
                        self._origin = exact.unfold(self._folded)
        return self._origin

    def exact_placed(self, rows, exact):
        """The lattice points rows and their normals placed with a placer's digits, in its frame"""
        return self.chart.exact_placed(exact, self.world, [self.world_point(self.keys[r]) for r in rows])

    def camera(self, camera, placer):
        """A camera (map units, float64) in the pieces' units from center, as floats, and float64's rounding of center's
        place in them; with digits, measured in the placer's frame, the difference turned back"""
        origin = self.origin(placer)
        if isinstance(origin, np.ndarray):
            scale = self.scale(self.world.rho)
            return (camera - origin) / scale, parts.FLOAT_ERROR * float(np.linalg.norm(origin)) / scale
        from decimal import Decimal
        exact = placer()
        turn = getattr(exact, "turn", None)
        seen = np.asarray(camera, dtype=float) if turn is None else turn @ np.asarray(camera, dtype=float)
        with exact.context():
            scale = Decimal(self.world.rho) * Decimal(2) ** self.scale_log2
            relative = np.array([float((Decimal(float(c)) - o) / scale) for c, o in zip(seen, self._folded)])
        return (relative if turn is None else turn.T @ relative), 0.0


# ----- the giant levels -----

def add_levels(chart, world, camera, q, h, window, levels, settings, shape, phases, placer):
    """The giant levels' pieces within window of the camera (map units), as parts.ExactPieces, finest level first:
    - the drone's place in the level's cells from the anchor's digits and the point below it (cell_place()), its
      distances from the cell's corners and sides, against how far each kind of piece reaches (parts.Shape), with room
      for how far the chart is from a similarity over a cell (distortion()): far from every piece, or deep inside a
      vertex piece, the level has nothing in the window
    - otherwise the level's pieces around the vertex nearest the drone, placed in float64 (Pieces), the camera against
      each one's solid: those within the window (or float64's rounding of it) are kept (parts.window_rows())
    - and those worked out again exactly (parts.exact_piece()) and cut to the window
    Deep inside a vertex piece, the drone is deep inside that vertex's piece at every coarser level the vertex is on
    (the same vertex, a piece at least twice as big), so those aren't looked at again."""
    found = []
    if not levels:
        return found
    below = chart.below(world, q)
    to_corner = chart.to_corner(world, below)
    # Cells next to the drone, from the finest instanced level's: twice as big every level up
    reference = levels[-1] + 1
    c_reference = chart.cell_sizes(world, [reference])[reference]
    has_vertex, has_edge = bool(shape.prisms[0]), bool(shape.prisms[1])
    reach_v, reach_e = shape.plan[0], shape.plan[1]
    core = shape.core[0]
    nested = None                    # the vertex (a world point) the drone is deep inside the piece of
    for level in reversed(levels):
        C = chart.cell(world, level)
        k = reference - level
        rho = math.ldexp(window / c_reference, -k)
        hc = math.ldexp(h / c_reference, -k)
        eps = chart.distortion(world.n, level, to_corner)
        on_nested = nested is not None and chart.on_level(world, nested, C)
        uncertain = eps > DISTORTION_LIMIT
        corners, side = chart.cell_place(world, below, C, level)
        check = False
        if has_vertex and not on_nested:
            if uncertain or rho + reach_v > 0.45:
                check = True
            else:
                deep = (core is not None and corners[0] < core[0] * (1.0 - eps) - rho - eps
                        and core[1] + rho + eps + SAG < 0.0 and hc * (1.0 + eps) + rho + eps < core[2])
                if deep:
                    nested = chart.center_point(below, C)
                    on_nested = True
                    corners = corners[1:]
                check = any(d < (reach_v + rho) * (1.0 + eps) + eps for d in corners)
        if has_edge and not check:
            if shape.floor[1] * (1.0 - eps) - eps > hc * (1.0 + eps) + rho:
                pass
            elif uncertain or rho + reach_e > 0.45:
                check = True
            else:
                check = side < (reach_e + rho) * (1.0 + eps) + eps
        if not check:
            continue
        M = 1 << level
        point = chart.center_point(below, C)
        center = chart.key_of(point, C, M)
        # The drone is within half a cell of the center on each axis; a vertex piece in reach is within its plan reach
        # and the window of it, an edge piece's edge within that, past the edge's ends, of a cell next to it
        radius = math.ceil(1.5 + 1.25 * rho + max(reach_v, reach_e, shape.past))
        pieces = Pieces(chart, world, level, center, radius, settings, shape)
        # In the pieces' own units (their level's world), which stay floats however deep
        local, rounding = pieces.camera(camera, placer)
        scale = pieces.scale(world.rho)
        rows, distances, tolerance = parts.window_rows(shape, pieces, local, rounding, scale, window)
        for row in rows:
            # float64 is enough while its rounding is well under the drone's height
            exact = not tolerance[row] * scale < parts.EXACT_SHARE * h
            piece = parts.exact_piece(pieces, int(row), level, camera, local, window, shape, settings, phases,
                                      placer if exact else None)
            if piece is not None:
                found.append(piece)
        here = pieces.middle_piece
        if (not on_nested and core is not None and here is not None
                and distances[here] < -(window / scale + tolerance[here])):
            nested = point
    return found


# ----- parts.py's view of a sphere map's lattice -----

class SphereLattice:
    """What parts.settle() asks of the lattice the parts stand on, for a sphere map's (sphere.py), either layout's
    (chart): its world, the camera in map units (and lifted, for 3D tiles) and its height, the point below it and any
    level's lattice point nearest that, a level's pieces around a lattice point (Pieces), each level's cells' size,
    where something built for another world is in today's map (transform()), and the giant levels' pieces in the
    window (add_levels(), with digits from placer()). A surface map's is surface.MeshLattice."""

    # How many times smaller a level's cells are than the next coarser's
    ratio = 2.0
    # Parts on every level_step-th level only (parts.drawn())
    level_step = 1

    def __init__(self, chart):
        self.chart = chart

    def world(self, obj):
        return sphere.world_of(obj)

    def unit_level(self, world):
        """A level whose cells are about a unit at the anchor (parts.wanted_levels() works out from it)"""
        return world.n + world.level_shift

    def camera(self, obj, world, camera_world):
        camera = sphere.map_point(obj, camera_world)
        q = sphere.lifted(world, camera, sphere.lift_of(obj))
        return camera, q, max(abs(float(sphere.height(world, q))), 1e-300)

    world_vector = staticmethod(sphere.world_vector)

    def placer(self, world, h):
        """What's placed with more digits than float64's has them for the world and the drone's height (made when
        first needed, and kept)"""
        return lambda: self.chart.placer(world, sphere_exact.digits_for(world.n, h / world.rho))

    def below(self, obj, world, q):
        return self.chart.below(world, q)

    def center(self, world, below, level):
        C = self.chart.cell(world, level)
        return self.chart.key_of(self.chart.center_point(below, C), C, 1 << level)

    def ground(self, obj, world, camera, q):
        """What a level's center is measured from: the camera (the sphere's normals turn smoothly, so its offset along
        the surface is the point below's)"""
        return camera

    def first_radius(self, world, level, build, wanted):
        """The coarsest level is built whole once, if its points are few (surface.is_whole()), else to "build" of its
        cells"""
        return math.inf if surface.is_whole(self.chart.topology(), 1 << level) else build

    def pieces(self, world, level, center, radius, settings, shape):
        return Pieces(self.chart, world, level, center, radius, settings, shape)

    def pieces_job(self, world, level, center, radius, settings, shape):
        """A level's build for the worker thread (background.py), as (function, *args); None: built here (a sphere's
        levels are a few ms, mostly kept, see Pieces)"""
        return None

    def cell_sizes(self, world, levels):
        return self.chart.cell_sizes(world, levels)

    def transform(self, world, text, built):
        """Where a level built for another world is in this one's map (world, its text): (offset, scale), its map's
        origin then (the anchor it was built for) from this map's origin, and how much bigger map units are now. Exact
        from the two anchors: the one it was built for, scaled to today's units (2^(n - n then) times, a Fraction after
        a climb), placed from today's."""
        if built.transform is not None and built.transform[0] == text:
            return built.transform[1], built.transform[2]
        then = sphere.World.from_text(built.world_text, world.rho)
        shift = world.n - then.n
        if shift == 0 and then.A == world.A:
            offset = np.zeros(3)
        else:
            offset = self.chart.placed(world, [self.chart.scaled(then.A, Fraction(2) ** shift)])[0][0]
        scale = 2.0 ** shift
        built.transform = (text, offset, scale)
        return offset, scale

    def add_levels(self, world, camera, q, h, window, levels, settings, shape, phases, placer):
        return add_levels(self.chart, world, camera, q, h, window, levels, settings, shape, phases, placer)


CUBE = CubeChart()
SPHERE = SphereLattice(CUBE)


def forget():
    CUBE.forget()
