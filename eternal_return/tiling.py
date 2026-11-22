"""Endless tilings: the parts an endless map's lattice draws in every cell of every level, at their full size forever.

An endless map (endless.py) stands on a lattice of square cells on its floor plane, each level's cells a whole loop
factor bigger than the next one's, so every cell splits factor x factor, at every level, forever. Square in the map
object's own coordinates, that is: everything here and in endless.py is worked out in them, so a map object with a shear
in its matrix (a parent inverse can hold one) has parallelogram cells, and with a motif of two triangles a cell, an
endless plane of equilateral triangles, each splitting factor x factor (2026-11-21: build_point_line_plane.py
--triangles; a window or reach in its local units covers as far in the world as long as the shear's inverse has rows
of unit length). Its motif is a few
parts, each drawn in every cell of every level in cell units, so it scales with its level: flat ones on the floor
plane, like the Inferno's star outlines and gold lines, or solid ones standing on it, like its pillars or 2d_time's
octagon pillars. A part is convex pieces (polygons in plan, extruded from low to high for a solid part), colored by the
cell's index modulo the lattice's period. They have to be thin, outlines, bars and posts: a filled shape covering a
share of its cell would cover the whole view at some level, and there's always a bigger one.

The map draws the levels whose pattern repeats within reach as patches that follow the drone by whole periods. Bigger
levels can't be patches: where their pieces are near the drone depends on the anchor's digits, more of them every
diving loop, and in float32 a piece of a cell 1e20 m across would be meters off near the drone. So every part's bigger
levels are built here, all in the same way, whenever the anchor changes (endless.set_anchor(), on every loop and wrap):
- levels(): the levels from the part's levels_from up with a piece in sight, within the window around the map's origin
  (which the drone is within a step of), walked up until no bigger level can have one
- level_pieces(): where a level's pieces are from the map's origin, placed exactly from the anchor
- build(): their meshes, cut to the window and colored by "Col", which Solid mode shows
- in_window(): whether a random digit would move a piece in sight (see endless.choose_digit())

Flat parts: each level is built into its own object, a slot, filled finest first, which the map layers (with drivers),
so where levels overlap one of them shows: by default the bigger one, over the finer ones, or with on_top "finer", the
finer one. The map object's "tiling_<name>_slots" says how many slots are filled, so the map can layer only those.
Layered by slot rather than by the level's own size, the layers stay bounded however deep a dive goes. Once the window
is all inside one of a level's pieces, the levels under it can't show: that level isn't built, the floor is tinted its
piece's color instead, and the map object's "tiling_<name>_covered" goes to 1. Bigger levels on top, nothing finer
shows, so the patches' drivers read that to hide them, and only the bigger levels with a piece in sight are built.
Finer levels on top, nothing bigger shows, and the finer levels go on showing over its color, the patches too: diving
into a piece, there's always a finer level to dive into, however deep, where with bigger levels on top, the dive ended
in one piece's flat color.

Solid parts: every level is built into one object, each piece's sides cut to the window and its caps (top and bottom)
if they're within it. From inside a prism bigger than the window, there's nothing to see, as there wouldn't be. Shaded
flat, or for boxes (axis-aligned rectangles in plan), smooth like the map's own patch boxes (see box_mesh()).

Whether a piece is in sight is the separating axis test of a convex polygon and the square window, exact (gaps()). Where
the map's origin is in a level's cells is kept as an integer and a float fraction while going up the levels, since exact
fractions for every level took milliseconds deep into a dive. The float decides which levels have a piece in sight,
except when a piece is within its rounding of where the decision changes, as when diving straight at an edge: then
it's decided exactly.

The endless settings' "tiling" key (see endless.py):
- cell: level 0's cell size. Level k's is cell / factor^k.
- offset: where cell (i, j)'s motif is: its point (0, 0) at ((i + offset) * cell, (j + offset) * cell) in the lattice,
  whose origin is the map's origin at anchor 0
- period: the colors repeat every period cells
- parts: a list, each:
  - name: a flat part's names the map object's properties "tiling_<name>_covered" and "tiling_<name>_slots"
  - pieces: convex polygons, counterclockwise, each a list of [u, v] in cells from the cell's motif point. No piece may
    have an edge through the lattice's origin, or every level would have one there, nor may a flat one cover it.
  - colors: colors[(i mod period) + period * (j mod period)] is a list of each piece's linear RGBA, for cell (i, j)
  - low, high: optional: how high a solid part stands, from and to, in cells up from the floor plane. Without them, or
    both the same, the part is flat, at low.
  - open: a solid part's caps not built: "-z" for the bottoms of pillars standing on a floor, "+z" for their tops
  - shading: a solid part's, "flat" (the default) or "smooth", for boxes only (see box_mesh())
  - levels_from: the finest level built here, the first bigger than its patches
  - window: how far from the map's origin (half a square's side) pieces count as in sight and are built: as far as the
    drone can see at its biggest scale, plus a step, since it's within one of the origin
  - object: a solid part's mesh object; slots: a flat part's, [{"object": o}, ...], finest first. Children of the map's
    object with no transform of their own (but the transform a flat part's slots are layered with).
  - floor, floor_color: optional, for a flat part: the floor's mesh object, its "Col" set to a covering piece's color,
    and to floor_color otherwise
  - on_top: optional, for a flat part: which level shows where two overlap, as the map layers them, "bigger" (the
    default) or "finer"
  - tile: optional, for a flat part, default 100: its faces are split on lines at 0 and +-tile * 2^k from the map's
    origin, so near the drone (within a step of it) no corner is far away. A polygon's depth is worked out in float32
    from its corners, so with them 50 km off it came out millimeters wrong near the camera, more than the levels'
    layers were apart, and two giant bands flickered across the whole floor (the floor itself is built in tiles for the
    same reason).
"""

import functools
import math
from fractions import Fraction

import numpy as np

# Exact arithmetic places a level's pieces from this cell size up; smaller ones, in float64 from each cell's exact place,
# are within 1e-10 m. Bigger, a piece's corner can be far beyond the window while its edge passes right by the drone.
EXACT_FROM = 1e6
# How far a float distance (in cells) can be from the exact one: the fraction where the origin is in a cell has the
# float64 rounding of a number under 1, a few times over going up the levels
ROUNDING = 1e-12
# The cells around the one the origin is in whose pieces are measured: two either side, since a piece can stick out of
# its cell a little and the motif point needn't be the cell's corner
NEAR = np.arange(-2, 3)
# A box's faces, by axis and side, as "open" names them
FACES = ("-x", "+x", "-y", "+y", "-z", "+z")
# The directions a piece's bounding box separates it from a window along
BOX_SIDES = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])


def exact(value):
    """A setting as an exact fraction, as written: 0.035 as 7/200, not the binary float nearest it."""
    return Fraction(repr(float(value)))


def cell_size(part, factor, level):
    """A level's cell size as a float, saturating instead of overflowing, 600-odd loops into a dive"""
    return float(part.cell) * math.exp(min(-level * math.log(factor), 700.0))


class Part:
    """A part's settings, ready to measure and build: its pieces' corners as floats and exact fractions, and the
    directions that can separate a piece from a square window (see gaps()): its edges' outward normals and its bounding
    box's sides."""

    def __init__(self, settings, lattice):
        self.settings = settings
        self.name = settings.get("name", "tiling")
        self.cell = exact(lattice["cell"])
        self.offset = exact(lattice.get("offset", 0.0))
        self.variants = int(lattice.get("period", 1))
        self.pieces = [np.array([[float(u), float(v)] for u, v in piece]) for piece in settings["pieces"]]
        self.colors = [[[float(c) for c in color] for color in variant] for variant in settings["colors"]]
        # All one color, a part repeats every cell, which tells which levels a move changes (level_for())
        self.period = self.variants if any(variant != self.colors[0] for variant in self.colors) else 1
        self.low, self.high = exact(settings.get("low", 0.0)), exact(settings.get("high", settings.get("low", 0.0)))
        self.solid = self.high > self.low
        self.open = set(settings.get("open", []))
        self.smooth = settings.get("shading") == "smooth"
        self.levels_from = int(settings["levels_from"])
        self.window = exact(settings["window"])
        self.finer_on_top = settings.get("on_top", "bigger") == "finer"
        # Each piece's directions and how far along each it reaches: its edges' outward normals, then its bounding
        # box's sides, all pieces' in one array, (pieces, directions, ...), padded by repeating its first edge, so every
        # piece is measured at once (one by one, measuring the levels took three times as long)
        corners_most = max(len(piece) for piece in self.pieces)
        normals, offsets, self.edge_normals, self.spans = [], [], [], []
        start = 0
        for corners in self.pieces:
            edges = np.roll(corners, -1, axis=0) - corners
            outward = np.stack([edges[:, 1], -edges[:, 0]], axis=1) / np.hypot(edges[:, 0], edges[:, 1])[:, None]
            self.edge_normals.append(outward)
            pad = corners_most - len(corners)
            normals.append(np.concatenate([outward, BOX_SIDES, outward[:1].repeat(pad, 0)]))
            reaches = (outward * corners).sum(axis=1)
            offsets.append(np.concatenate([reaches, [corners[:, 0].max(), -corners[:, 0].min(), corners[:, 1].max(),
                                                     -corners[:, 1].min()], reaches[:1].repeat(pad)]))
            self.spans.append((start, start + len(corners)))
            start += len(corners)
        self.normals, self.offsets = np.array(normals), np.array(offsets)
        # How far a square window reaches along each direction, per cell of its half width
        self.spreads = np.abs(self.normals).sum(axis=2)
        # Each piece's bounding box, (pieces, 2) along X and Y (see boxes_near())
        self.box_low = np.array([corners.min(axis=0) for corners in self.pieces])
        self.box_high = np.array([corners.max(axis=0) for corners in self.pieces])
        # Every corner's coordinate along each axis, piece after piece (see axis_places())
        self.float_us = [np.concatenate([piece[:, axis] for piece in self.pieces]) for axis in (0, 1)]
        if self.smooth:
            assert self.solid and all(len(piece) == 4 and len(set(piece[:, 0])) == 2 and len(set(piece[:, 1])) == 2
                                      for piece in self.pieces), "only boxes are shaded smooth"

    # As exact fractions only when they're needed, since most loops don't need them: made for every part on every
    # loop, they took most of a millisecond
    @functools.cached_property
    def exact_pieces(self):
        return [[(exact(u), exact(v)) for u, v in piece] for piece in self.pieces]

    @functools.cached_property
    def exact_us(self):
        return [[corner[axis] for piece in self.exact_pieces for corner in piece] for axis in (0, 1)]

    @functools.cached_property
    def span_groups(self):
        """The pieces grouped by how many corners they have, so level_pieces() can measure each group at once:
        (pieces, columns) for each, columns (pieces, corners) their corners' columns in float_us"""
        groups = {}
        for piece, (start, end) in enumerate(self.spans):
            groups.setdefault(end - start, []).append(piece)
        return [(pieces, np.array([np.arange(*self.spans[piece]) for piece in pieces])) for pieces in groups.values()]

    @functools.cached_property
    def exact_bounds(self):
        """The lowest and highest corner along each axis, exact"""
        return [(min(us), max(us)) for us in self.exact_us]

    @functools.cached_property
    def exact_piece_bounds(self):
        """Each piece's lowest and highest corner along each axis, exact: [piece][axis] = (low, high)"""
        return [[(min(c[axis] for c in corners), max(c[axis] for c in corners)) for axis in (0, 1)]
                for corners in self.exact_pieces]

    @functools.cached_property
    def at_origin(self):
        """How far the lattice's origin is from the pieces around it, in cells, the same at every level: (clear, depth),
        to the nearest piece it's outside of, and to the nearest edge of one it's inside (a solid part's, as a 2 x 2
        lattice's corner is inside its pillar), infinite if there's none"""
        offset = float(self.offset)
        grid_u, grid_v = np.meshgrid(-NEAR - offset, -NEAR - offset, indexing='ij')
        gaps = self.gaps(grid_u, grid_v, 0.0)[0]
        assert (gaps != 0).all(), "a piece with an edge on the lattice's origin has one there at every level"
        assert self.solid or (gaps > 0).all(), "a flat piece over the lattice's origin covers the view at every level"
        return gaps[gaps > 0].min(initial=math.inf), -gaps[gaps < 0].max(initial=-math.inf)

    def boxes_near(self, u, v, reach):
        """For several levels at once, u and v (levels, points) the points' coordinates along each axis (every pair of
        them a point) and reach (levels,) their windows': along each axis, whether each piece's bounding box is within
        ROUNDING of overlapping each point's window there, (pieces, levels, points) for X and for Y. A box and a
        window are apart along X or along Y, so a piece is near a point only where it is along both; anywhere else,
        gaps() has it out of sight by more than ROUNDING, by its box directions alone (these are those, to the bit)."""
        along = []
        for axis, at in ((0, u), (1, v)):
            low, high = self.box_low[:, axis, None, None], self.box_high[:, axis, None, None]
            along.append((np.maximum(at[None] - high, low - at[None]) - reach[None, :, None]) < ROUNDING)
        return along

    def gaps_at(self, pieces, u, v, reach):
        """gaps()' outside and inside for a list of cases, each a piece, a point (u, v) and its window's reach (arrays
        of one length): the same numbers, only for those"""
        base = (self.normals[pieces, :, 0] * u[:, None] + self.normals[pieces, :, 1] * v[:, None]
                - self.offsets[pieces])
        window = self.spreads[pieces] * reach[:, None]
        return (base - window).max(axis=1), (base + window).max(axis=1)

    def gaps(self, u, v, reach):
        """For points (u, v) (arrays of one shape, in cells from a cell's motif point) and a square window reach cells to
        each side of each (broadcasting with them): (distance, outside, inside), each an array (pieces, *shape).
        A piece and the window are apart where outside > 0 (along one of the piece's directions, they don't
        overlap), and the window is all inside the piece where inside <= 0. Both are exact, the separating axis test for
        a convex polygon and a square. distance, outside at reach 0, is never more than the true distance (and inside a
        piece, minus the depth within it), and changes by at most as much as the point moves."""
        shape = (len(self.pieces),) + u.shape
        base = (self.normals[:, :, 0, None] * u.ravel() + self.normals[:, :, 1, None] * v.ravel()
                - self.offsets[:, :, None])
        window = self.spreads[:, :, None] * np.broadcast_to(reach, u.shape).ravel()
        return (base.max(axis=1).reshape(shape), (base - window).max(axis=1).reshape(shape),
                (base + window).max(axis=1).reshape(shape))

    def exact_gap(self, piece, u, v, reach, sign):
        """gaps()'s outside (sign -1) or inside (sign 1) for one piece and exact fractions u, v and reach, exact up to
        the division by each edge's length, so its sign is exact however small it is"""
        corners = self.exact_pieces[piece]
        values = []
        for k, (pu, pv) in enumerate(corners):
            qu, qv = corners[(k + 1) % len(corners)]
            du, dv = qu - pu, qv - pv
            # Outside the edge's line by -cross(edge, point - corner), the polygon being counterclockwise, and the
            # window's half width along its normal is reach * (|du| + |dv|), both times the edge's length
            value = -(du * (v - pv) - dv * (u - pu)) + sign * reach * (abs(du) + abs(dv))
            values.append(float(value) / math.hypot(float(du), float(dv)))
        us, vs = [c[0] for c in corners], [c[1] for c in corners]
        values += [float(u - max(us) + sign * reach), float(min(us) - u + sign * reach),
                   float(v - max(vs) + sign * reach), float(min(vs) - v + sign * reach)]
        return max(values)

    def nearest(self, x, y):
        """How far the point (x, y) is from the part's nearest piece in plan, in cells, negative inside one: x and y in
        cells from the lattice's origin, floats or exact fractions, of any size, since only where the point is in its
        cell counts (worked out exactly from fractions). The true distance, as a map's start wants: gaps()' is never
        more, but near a strip's mitered end it was a quarter less, and a start that far from the floor's lines was
        refused."""
        u, v = (float(c - self.offset - math.floor(c - self.offset)) for c in (x, y))
        points = np.stack(np.meshgrid(u - NEAR, v - NEAR, indexing='ij'), axis=-1).reshape(-1, 2)
        best = math.inf
        for corners, normals in zip(self.pieces, self.edge_normals):
            # Inside a convex piece, the depth is the distance to its nearest edge's line; outside, to its nearest edge
            depth = (points @ normals.T - (normals * corners).sum(axis=1)).max(axis=1)
            edges = np.roll(corners, -1, axis=0) - corners
            t = np.clip(((points[:, None, :] - corners[None]) * edges[None]).sum(axis=2) / (edges ** 2).sum(axis=1),
                        0.0, 1.0)
            apart = np.linalg.norm(points[:, None, :] - corners[None] - t[:, :, None] * edges[None], axis=2).min(axis=1)
            best = min(best, float(np.where(depth < 0, depth, apart).min()))
        return best

    def color(self, piece, i, j):
        return self.colors[(i % self.variants) + self.variants * (j % self.variants)][piece]

    @functools.cached_property
    def color_table(self):
        """The colors as an array, (variants, pieces, 4)"""
        return np.array(self.colors, dtype=float)

    def cell_colors(self, piece, cells):
        """A piece's color in each of the cells (an array (n, 2) of their indices), as an array (n, 4)"""
        cells = np.asarray(cells, dtype=np.int64).reshape(-1, 2)
        return self.color_table[cells[:, 0] % self.variants + self.variants * (cells[:, 1] % self.variants), piece]


_parts = {}         # the settings as written -> their Parts
PARTS_KEPT = 8


def parts(settings):
    """The endless settings' "tiling" (settings) as Parts, kept for as long as the settings read the same: a part's
    directions and exact pieces are the same for every build, and making them again was a quarter of each loop's
    (2026-11-14). Each is given the settings as they are now, for the objects they name. A plain dict works too, as a
    map's build script passes them before they're on the map."""
    key = repr(settings.to_dict() if hasattr(settings, "to_dict") else settings)
    found = _parts.get(key)
    if found is None:
        if len(_parts) >= PARTS_KEPT:
            _parts.clear()
        found = _parts[key] = [Part(part, settings) for part in settings["parts"]]
    else:
        for part, part_settings in zip(found, settings["parts"]):
            part.settings = part_settings
    return found


def levels(part, factor, origin, first=None):
    """The part's levels from first (default: its levels_from) up with a piece in sight, for the map's origin at origin
    in the lattice, finest first, until no bigger level can have one, each as (level, distance, covering): distance, in
    the map's units, from the map's origin to the nearest piece in plan (negative inside one, see gaps()); covering, for
    a flat part whose window is all inside a piece, (i, j, piece) for its cell and piece, else None. A solid part's level
    is in sight if its height reaches the window, and one of its sides does or, with the window all inside a piece, one
    of its caps. The levels are walked first, then all measured at once: one by one, 640 levels deep into a dive took
    140 ms, a hitch on every loop."""
    window, offset = part.window, float(part.offset)
    level = part.levels_from if first is None else first
    cell = part.cell * Fraction(factor) ** -level
    index, into = [], []
    for axis in (0, 1):
        q = origin[axis] / cell
        index.append(math.floor(q))
        into.append(float(q - index[-1]))
    reach = float(window / cell)
    # How far the lattice's origin is from the pieces around it (Part.at_origin): once the window is within that of
    # it, so is every bigger level's. It may be inside a solid part's piece: then every bigger level's window is too,
    # and once their caps are beyond it, nothing's left.
    clear, depth = part.at_origin
    walk = []
    while True:
        walk.append((level, tuple(index), tuple(into), reach))
        if all(i in (0, -1) for i in index):
            spread = math.hypot(*(float(origin[axis] * Fraction(factor) ** level / part.cell) for axis in (0, 1)))
            beyond = spread + reach * math.sqrt(2)
            if beyond < clear and (depth == math.inf or (beyond < depth and caps_beyond(part, factor, level))):
                break
        for axis in (0, 1):
            digit = index[axis] % factor
            index[axis] //= factor
            into[axis] = (digit + into[axis]) / factor
        reach /= factor
        level -= 1
    # Where each level's 5 x 5 points are along each axis, (levels, 5), and their windows' reach
    u = np.array([w[2][0] for w in walk])[:, None] - NEAR[None, :] - offset
    v = np.array([w[2][1] for w in walk])[:, None] - NEAR[None, :] - offset
    reaches = np.array([w[3] for w in walk])
    # Only the pieces whose bounding box is near a point's window are measured along their edges, all at once: every
    # piece at every point of every walked level was a quarter of a loop's build on the strapwork floor's 42 pieces
    # (2026-11-14). The rest are out of sight for certain (boxes_near()).
    near_u, near_v = part.boxes_near(u, v, reaches)
    piece_at, level_at, a_at, b_at = np.nonzero(near_u[:, :, :, None] & near_v[:, :, None, :])
    outside_at, inside_at = part.gaps_at(piece_at, u[level_at, a_at], v[level_at, b_at], reaches[level_at])
    w = float(window)
    # Only the levels with a piece in sight, or too close to call, need a closer look: deep into a dive, nearly all of
    # hundreds walked have none, and looking at each one took most of the time
    close = (outside_at <= 0) | (np.abs(outside_at) < ROUNDING) | (np.abs(inside_at) < ROUNDING)
    grid = (len(part.pieces), len(NEAR), len(NEAR))
    for n in np.unique(level_at[close]):
        level, index, _into, _reach = walk[n]
        # This level's, (pieces, 5, 5): infinitely far wherever a box wasn't near
        rows = np.flatnonzero(level_at == n)
        outside, inside = np.full(grid, np.inf), np.full(grid, np.inf)
        outside[piece_at[rows], a_at[rows], b_at[rows]] = outside_at[rows]
        inside[piece_at[rows], a_at[rows], b_at[rows]] = inside_at[rows]
        unsure = (np.abs(outside) < ROUNDING) | (np.abs(inside) < ROUNDING)
        if unsure.any():
            # Too close to call in floats: measure those exactly
            cell = part.cell * Fraction(factor) ** -level
            q = [origin[axis] / cell for axis in (0, 1)]
            for p, a, b in zip(*np.nonzero(unsure)):
                pu = q[0] - (index[0] + int(NEAR[a])) - part.offset
                pv = q[1] - (index[1] + int(NEAR[b])) - part.offset
                outside[p, a, b] = part.exact_gap(p, pu, pv, window / cell, -1)
                inside[p, a, b] = part.exact_gap(p, pu, pv, window / cell, 1)
        hits, covers = outside <= 0, inside <= 0
        if not hits.any():
            continue
        across = cell_size(part, factor, level)
        low, high = float(part.low) * across, float(part.high) * across
        if low > w or high < -w:
            continue
        covering = None
        if part.solid:
            caps = ("+z" not in part.open and abs(high) <= w) or ("-z" not in part.open and abs(low) <= w)
            if not (hits & (~covers | caps)).any():
                continue
        # How far every piece around is from the origin, for the nearest one and the covering one
        distances = part.gaps(np.ascontiguousarray(np.broadcast_to(u[n][:, None], grid[1:])),
                              np.ascontiguousarray(np.broadcast_to(v[n][None, :], grid[1:])), 0.0)[0]
        if not part.solid and covers.any():
            p, a, b = np.unravel_index(np.where(covers, distances, np.inf).argmin(), covers.shape)
            covering = (index[0] + int(NEAR[a]), index[1] + int(NEAR[b]), int(p))
        yield level, float(distances.min()) * across, covering


def caps_beyond(part, factor, level):
    """Whether a level's caps are out of sight: open, or further from the floor plane than the window"""
    across, w = cell_size(part, factor, level), float(part.window)
    return (("+z" in part.open or float(part.high) * across > w)
            and ("-z" in part.open or float(part.low) * across < -w))


def chosen_levels(part, factor, origin):
    """The levels to build, finest first, and for a flat part, the covering level's (level, i, j, piece) or None. A
    solid part's are every level in sight. A flat part's: once the window is all inside a level's piece, none of the
    levels it's drawn over shows. With bigger levels on top, that's every finer one, so the biggest covering level is
    the one that shows, and only the levels bigger than it are built; with finer levels on top (on_top "finer"), it's
    every bigger one, so the finest covering level shows, and only the levels finer than it are built. With more levels
    than slots still, the ones with pieces nearest the drone, so it's faint ones far off that are left out."""
    found = list(levels(part, factor, origin))
    if part.solid:
        return [level for level, _distance, _cover in found], None
    covering = [n for n, (_level, _distance, cover) in enumerate(found) if cover is not None]
    cover = None
    if covering:
        shown = covering[0] if part.finer_on_top else covering[-1]
        level, _distance, (i, j, piece) = found[shown]
        cover = (level, i, j, piece)
        found = found[:shown] if part.finer_on_top else found[shown + 1:]
    slots = len(part.settings["slots"])
    if len(found) > slots:
        found = sorted(sorted(found, key=lambda f: abs(f[1]))[:slots], key=lambda f: -f[0])
    return [level for level, _distance, _cover in found], cover


def axis_places(part, axis, cell, at):
    """Along one axis (0 for X, 1 for Y), for a level's cells cell across and the map's origin at at in the lattice: the
    indices of the cells whose pieces may reach the window, and where each of the part's corners (every piece's in turn,
    see Part.spans) is in each of them, from the map's origin: (indices, places), places[n][k] for indices[n] and corner
    k, float64 from each cell's exact place. The indices are modulo the lattice's period, all their colors go by: deep
    into a dive, they're far bigger than an int64."""
    first, bases = cell_bases(part, cell, at, *part.exact_bounds[axis])
    indices = (first % part.variants + np.arange(len(bases))) % part.variants
    places = np.array([float(b) for b in bases]).reshape(-1, 1) + part.float_us[axis][None, :] * float(cell)
    return indices, places


def cell_bases(part, cell, at, low, high):
    """Along one axis, for a level's cells cell across and the map's origin at at in the lattice: the first index of
    the cells whose pieces, from low to high in cells from their motif points, may reach the window, and where each
    such cell's motif point is from the map's origin, exactly: (first, bases)"""
    first = math.ceil((at - part.window) / cell - part.offset - high)
    last = math.floor((at + part.window) / cell - part.offset - low)
    base = (first + part.offset) * cell - at
    return first, [base + n * cell for n in range(last - first + 1)]


def level_pieces(part, factor, origin, level):
    """A level's pieces that may reach the window, piece by piece of the part: (whole, cells, cut) for each, whole the
    corners of those all within it, floats from the map's origin, (count, corners, 2), cells their cells' indices modulo
    the lattice's period (count, 2), and cut the rest that overlap it in their bounds, a list of (corners, (i, j)), each a list of (x, y): exact
    fractions for cells of EXACT_FROM and up (their corners can be far beyond what a float holds), else floats. A
    corner's X only depends on its cell's X index, and its Y on its Y index, so they're worked out once per index."""
    cell = part.cell * Fraction(factor) ** -level
    if cell < EXACT_FROM:
        (xi, xs), (yi, ys) = (axis_places(part, axis, cell, origin[axis]) for axis in (0, 1))
        return float_level_pieces(part, xi, xs, yi, ys)
    w = part.window
    cells_along = []
    for axis in (0, 1):
        first, bases = cell_bases(part, cell, origin[axis], *part.exact_bounds[axis])
        cells_along.append(([int(i) for i in (first % part.variants + np.arange(len(bases))) % part.variants], bases))
    result = []
    for piece, corners in enumerate(part.exact_pieces):
        # Along each axis, the cells where the piece's bounds reach the window, and whether they're within it there:
        # a piece is out of sight unless it reaches along both, so only those have their corners placed (every
        # piece's in every cell around, each corner compared as a fraction, was half an exact level's time)
        reaching = []
        for axis, (_indices, bases) in enumerate(cells_along):
            low, high = part.exact_piece_bounds[piece][axis]
            sides = [(n, base + low * cell, base + high * cell) for n, base in enumerate(bases)]
            reaching.append([(n, -w <= lo and hi <= w, [bases[n] + c[axis] * cell for c in corners])
                             for n, lo, hi in sides if lo <= w and hi >= -w])
        whole, cells, cut = [], [], []
        for n, within_x, xs in reaching[0]:
            for m, within_y, ys in reaching[1]:
                index = (cells_along[0][0][n], cells_along[1][0][m])
                if within_x and within_y:
                    whole.append([(float(x), float(y)) for x, y in zip(xs, ys)])
                    cells.append(index)
                else:
                    cut.append((list(zip(xs, ys)), index))
        result.append((np.array(whole, dtype=float).reshape(-1, len(corners), 2),
                       np.array(cells, dtype=np.int64).reshape(-1, 2), cut))
    return result


def float_level_pieces(part, xi, xs, yi, ys):
    """level_pieces() for a level small enough for floats, from axis_places(): every piece with the same number of
    corners measured at once (piece by piece, the numpy calls' overhead was most of a loop's time on the strapwork
    floor's 42 pieces)"""
    w = float(part.window)
    result = [None] * len(part.spans)
    for pieces, columns in part.span_groups:
        # (cells along the axis, pieces, corners)
        x, y = xs[:, columns], ys[:, columns]
        near_x = (x.min(axis=2) <= w) & (x.max(axis=2) >= -w)
        near_y = (y.min(axis=2) <= w) & (y.max(axis=2) >= -w)
        in_x, in_y = (np.abs(x) <= w).all(axis=2), (np.abs(y) <= w).all(axis=2)
        # Piece by piece, its cells in X index then Y index order
        p, a, b = np.nonzero(near_x.T[:, :, None] & near_y.T[:, None, :])
        corners = np.stack([x[a, p], y[b, p]], axis=2)
        cells = np.stack([xi[a], yi[b]], axis=1)
        inside = in_x[a, p] & in_y[b, p]
        bounds = np.searchsorted(p, np.arange(len(pieces) + 1))
        for q, piece in enumerate(pieces):
            s, e = bounds[q], bounds[q + 1]
            cut = [([tuple(c) for c in corners[n]], (int(cells[n, 0]), int(cells[n, 1])))
                   for n in s + np.flatnonzero(~inside[s:e])]
            result[piece] = (corners[s:e][inside[s:e]], cells[s:e][inside[s:e]], cut)
    return result


def past_window(side, window):
    """A side as a float, no further out than just past the window. Nothing beyond the window is built, and the sides
    of the biggest levels' pieces, deep into a dive, are further out than a float can hold: after 640 diving loops, the
    Inferno's pillars were over 1e308 m tall, and converting that to a float overflowed."""
    return float(max(-2 * window, min(2 * window, side)))


class Faces:
    """Faces gathered for a mesh, each with its own corners: corners and their normals (n, 3), corners per face (m,) and
    each face's color (m, 4)"""

    def __init__(self):
        self.corners, self.normals, self.sizes, self.colors = [], [], [], []

    def add(self, corners, normals, sizes, colors):
        self.corners.append(np.asarray(corners, dtype=float).reshape(-1, 3))
        self.normals.append(np.asarray(normals, dtype=float).reshape(-1, 3))
        self.sizes.append(np.asarray(sizes, dtype=np.int64).reshape(-1))
        self.colors.append(np.asarray(colors, dtype=float).reshape(-1, 4))

    def arrays(self):
        return (np.concatenate([np.zeros((0, 3))] + self.corners), np.concatenate([np.zeros((0, 3))] + self.normals),
                np.concatenate([np.zeros(0, dtype=np.int64)] + self.sizes),
                np.concatenate([np.zeros((0, 4))] + self.colors))


def flat_faces(part, factor, origin, level):
    """A flat part's level, its pieces cut to the window and split into tiles (see tiles()), as (corners, sizes,
    colors): every face's corners in a row, (n, 2) floats from the map's origin, how many each face has (m,), and its
    color (m, 4)"""
    exactly = part.cell * Fraction(factor) ** -level >= EXACT_FROM
    window = part.window if exactly else float(part.window)
    batches, colors = [], []
    for piece, (whole, cells, cut) in enumerate(level_pieces(part, factor, origin, level)):
        batches.append(np.asarray(whole, dtype=float))
        colors.append(part.cell_colors(piece, cells))
        for corners, (i, j) in cut:
            clipped = clip(corners, window)
            if len(clipped) >= 3:
                batches.append(np.array([[(float(x), float(y)) for x, y in clipped]]))
                colors.append(np.asarray(part.color(piece, i, j), dtype=float).reshape(1, 4))
    return tiles(batches, colors, float(part.settings.get("tile", 100.0)), float(part.window))


def prism_faces(part, factor, origin, level, faces):
    """A solid part's level, shaded flat, added to faces: each piece's sides, an edge's each, cut to the window, from
    its low to its high (no further out than just past the window), and its caps if they're within it and not open,
    cut to it. Each corner is placed from its exact point, so a face near the drone is as exact as float32 is there."""
    cell = part.cell * Fraction(factor) ** -level
    window = part.window if cell >= EXACT_FROM else float(part.window)
    low, high = part.low * cell, part.high * cell
    z0, z1 = past_window(low, part.window), past_window(high, part.window)
    top = "+z" not in part.open and abs(high) <= part.window
    bottom = "-z" not in part.open and abs(low) <= part.window
    for piece, (whole, cells, cut) in enumerate(level_pieces(part, factor, origin, level)):
        normals = np.concatenate([part.edge_normals[piece], np.zeros((len(part.edge_normals[piece]), 1))], axis=1)
        count = len(normals)
        following = np.roll(np.arange(count), -1)
        n = len(whole)
        colors = part.cell_colors(piece, cells)
        if n:
            quads = np.zeros((n, count, 4, 3))
            quads[:, :, 0, :2], quads[:, :, 1, :2] = whole, whole[:, following]
            quads[:, :, 2, :2], quads[:, :, 3, :2] = whole[:, following], whole
            quads[:, :, :2, 2], quads[:, :, 2:, 2] = z0, z1
            faces.add(quads, np.tile(np.repeat(normals, 4, axis=0), (n, 1)), np.full(n * count, 4),
                      np.repeat(colors, count, axis=0))
            for cap, z, up in ((top, z1, 1.0), (bottom, z0, -1.0)):
                if cap:
                    polygon = whole if up > 0 else whole[:, ::-1]
                    faces.add(np.concatenate([polygon, np.full((n, count, 1), z)], axis=2),
                              np.tile([0.0, 0.0, up], (n * count, 1)), np.full(n, count), colors)
        for corners, (i, j) in cut:
            color = part.color(piece, i, j)
            for k in range(count):
                segment = clip_segment(*corners[k], *corners[(k + 1) % count], window)
                if segment is not None:
                    (a, b), (c, d) = ((float(x), float(y)) for x, y in segment)
                    faces.add([(a, b, z0), (c, d, z0), (c, d, z1), (a, b, z1)], np.tile(normals[k], (4, 1)), [4],
                              [color])
            clipped = clip(corners, window) if top or bottom else []
            if len(clipped) >= 3:
                points = [(float(x), float(y)) for x, y in clipped]
                for cap, z, up in ((top, z1, 1.0), (bottom, z0, -1.0)):
                    if cap:
                        faces.add([(x, y, z) for x, y in (points if up > 0 else points[::-1])],
                                  np.tile([0.0, 0.0, up], (len(points), 1)), [len(points)], [color])


def smooth_box_faces(part, factor, origin, level, faces):
    """A solid part of boxes' level, shaded smooth (see box_mesh()), added to faces. Their sides are worked out exactly
    along each axis, then rounded once: a face a meter from the drone is a meter from the map's origin give or take a
    step, wherever the anchor is."""
    cell = part.cell * Fraction(factor) ** -level
    window = part.window
    for piece, corners in enumerate(part.exact_pieces):
        ranges, indices = [], []
        for axis in (0, 1):
            low, high = min(c[axis] for c in corners), max(c[axis] for c in corners)
            first, bases = cell_bases(part, cell, origin[axis], low, high)
            ranges.append(side_ranges([b + low * cell for b in bases], [b + high * cell for b in bases], window))
            indices.append((first % part.variants + np.arange(len(bases))) % part.variants)
        if not (len(indices[0]) and len(indices[1])):
            continue
        ranges.append(side_ranges([part.low * cell], [part.high * cell], window))
        corners_out, normals, sizes, rows = box_mesh(ranges, float(window), part.open)
        # A face's grid row is (x index, y index) of its box, Z having one
        cells = np.stack([indices[0][rows // len(indices[1])], indices[1][rows % len(indices[1])]], axis=1)
        faces.add(corners_out, normals, sizes, part.cell_colors(piece, cells))


def clip(polygon, window):
    """A convex polygon (a list of (x, y), floats or exact fractions) cut to the square within window of the origin
    (Sutherland-Hodgman), exactly if its corners are exact"""
    for axis in (0, 1):
        for sign in (1, -1):
            if not polygon:
                return polygon
            kept = []
            for k, point in enumerate(polygon):
                previous = polygon[k - 1]
                inside, was_inside = sign * point[axis] <= window, sign * previous[axis] <= window
                if inside != was_inside:
                    # Where the edge crosses the side, at sign * window
                    t = (sign * window - previous[axis]) / (point[axis] - previous[axis])
                    kept.append(tuple(previous[n] + t * (point[n] - previous[n]) for n in (0, 1)))
                if inside:
                    kept.append(point)
            polygon = kept
    # A corner on a side comes out twice
    return [point for k, point in enumerate(polygon) if point != polygon[k - 1]]


def clip_segment(x0, y0, x1, y1, w):
    """The part of the segment from (x0, y0) to (x1, y1) within -w to w along X and Y, as its two ends, or None
    (Liang-Barsky), exactly if they're exact"""
    t0, t1 = 0, 1
    dx, dy = x1 - x0, y1 - y0
    for p, q in ((-dx, x0 + w), (dx, w - x0), (-dy, y0 + w), (dy, w - y0)):
        if p == 0:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return None
    if t0 == 0 and t1 == 1:
        return (x0, y0), (x1, y1)
    return (x0 + t0 * dx, y0 + t0 * dy), (x0 + t1 * dx, y0 + t1 * dy)


def tiles(batches, colors, tile, window, adaptive=False):
    """Convex polygons split on the lines at 0 and +-tile * 2^k, up to the window, with their colors: batches of them,
    each (count, corners, 2) floats, and each batch's colors (count, 4). As (corners, sizes, colors): every face's
    corners in a row (n, 2), how many each has (m,), and its color (m, 4), each polygon's pieces where it was.

    The lines bound every piece's size by its distance from the middle (or tile, near it): that's what depth precision
    needs, the pieces near the camera small. Adaptive, a polygon already within that bound isn't cut at all, only
    those bigger than their distance from the middle (a girih level's bars near it): the lines run the whole window,
    and cut thousands of short bars far from the middle that needed nothing.

    All at once, as arrays: each polygon is cut into slabs between the lines across it along X, then each slab between
    the lines across it along Y (only those: a long diagonal strip's slabs each span a few of the strip's lines). One at
    a time in Python, this took most of a loop's time on the strapwork floor, which froze the drone at every loop.
    Only polygons a line crosses go through that: on a girih level's 11k bars, a tenth of them, and cutting every one
    (passing the rest on as they were, with the same answer) took 13 ms a level, most of a dive's rebuilding time
    (2026-11-05)."""
    cuts = [0.0]
    while cuts[-1] < window:
        cuts.append(tile if cuts[-1] == 0.0 else 2 * cuts[-1])
    cuts = np.array(sorted(set([-c for c in cuts] + cuts)))
    width = max([3] + [b.shape[1] for b in batches])
    points = np.zeros((sum(len(b) for b in batches), width, 2))
    counts = np.zeros(len(points), dtype=np.int64)
    row = 0
    for b in batches:
        points[row:row + len(b), :b.shape[1]] = b
        counts[row:row + len(b)] = b.shape[1]
        row += len(b)
    colors = np.concatenate([np.zeros((0, 4))] + [np.asarray(c, dtype=float).reshape(-1, 4) for c in colors])
    crossed = np.zeros(len(points), dtype=bool)
    valid = np.arange(width) < counts[:, None]
    extents, gaps = [], []
    for axis in (0, 1):
        low = np.where(valid, points[..., axis], np.inf).min(axis=1, initial=np.inf)
        high = np.where(valid, points[..., axis], -np.inf).max(axis=1, initial=-np.inf)
        crossed |= np.searchsorted(cuts, high, side="left") > np.searchsorted(cuts, low, side="right")
        extents.append(high - low)
        gaps.append(np.maximum(0.0, np.maximum(low, -high)))
    if adaptive:
        # How far each polygon is from the middle (Chebyshev, as the lines go), and whether it's bigger than that
        bound = np.maximum(tile, np.maximum(*gaps))
        crossed &= (extents[0] > bound) | (extents[1] > bound)
    rows = np.nonzero(crossed)[0]
    pieces, piece_counts, origin = split_on_lines(points[rows], counts[rows], cuts)
    # Each polygon's faces in its place: itself if no line crosses it, else its pieces, in the order they were cut
    per_row = np.ones(len(points), dtype=np.int64)
    per_row[rows] = np.bincount(origin, minlength=len(rows))
    first = np.cumsum(per_row) - per_row
    out = np.zeros((int(per_row.sum()), max(width, pieces.shape[1]), 2))
    out_counts = np.zeros(len(out), dtype=np.int64)
    whole = np.nonzero(~crossed)[0]
    out[first[whole], :width] = points[whole]
    out_counts[first[whole]] = counts[whole]
    rank = np.arange(len(origin)) - (np.cumsum(per_row[rows]) - per_row[rows])[origin]
    places = first[rows][origin] + rank
    out[places, :pieces.shape[1]] = pieces
    out_counts[places] = piece_counts
    out_colors = colors[np.repeat(np.arange(len(points)), per_row)]
    return out[np.arange(out.shape[1]) < out_counts[:, None]], out_counts, out_colors


def split_on_lines(points, counts, cuts):
    """Convex polygons (n, corners, 2) cut on the lines at cuts along X, then along Y, as (pieces, their counts, which
    polygon each came from), in order"""
    origin = np.arange(len(points))
    for axis in (0, 1):
        valid = np.arange(points.shape[1]) < counts[:, None]
        low = np.where(valid, points[..., axis], np.inf).min(axis=1, initial=np.inf)
        high = np.where(valid, points[..., axis], -np.inf).max(axis=1, initial=-np.inf)
        # The lines strictly across each polygon are cuts[first:first + across], cutting it into across + 1 slabs; a
        # polygon none cross is passed on as it is
        first = np.searchsorted(cuts, low, side="right")
        across = np.maximum(np.searchsorted(cuts, high, side="left") - first, 0)
        index = np.repeat(np.arange(len(counts)), across + 1)
        slab = np.arange(len(index)) - np.repeat(np.cumsum(across + 1) - (across + 1), across + 1)
        lines = across[index]
        cut = lines > 0
        top = len(cuts) - 1
        a = np.where(slab == 0, low[index], cuts[np.clip(first[index] + slab - 1, 0, top)])
        b = np.where(slab == lines, high[index], cuts[np.clip(first[index] + slab, 0, top)])
        sliced = index[cut]
        clipped, clipped_counts = clip_sides(*clip_sides(points[sliced], counts[sliced], axis, a[cut], -1),
                                             axis, b[cut], 1)
        split = np.zeros((len(index), max(points.shape[1], clipped.shape[1]), 2))
        split[~cut, :points.shape[1]] = points[index[~cut]]
        split[cut, :clipped.shape[1]] = clipped
        split_counts = counts[index]
        split_counts[cut] = clipped_counts
        keep = ~cut | (split_counts >= 3)
        points, counts, origin = split[keep], split_counts[keep], origin[index[keep]]
    return points, counts, origin


def clip_sides(points, counts, axis, values, sign):
    """Convex polygons, (n, corners, 2) with how many corners each has (n,), each cut to the side of its own line axis =
    values[i] where sign * (coordinate - value) <= 0 (Sutherland-Hodgman), as (points, counts)"""
    n, width = points.shape[:2]
    if n == 0:
        return np.zeros((0, width + 1, 2)), np.zeros(0, dtype=np.int64)
    k = np.arange(width)
    previous = take_corners(points, (k - 1) % np.maximum(counts, 1)[:, None])
    value = np.asarray(values, dtype=float)[:, None]
    inside = sign * (points[..., axis] - value) <= 0
    was_inside = sign * (previous[..., axis] - value) <= 0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        t = (value - previous[..., axis]) / (points[..., axis] - previous[..., axis])
        crossing = previous + t[..., None] * (points - previous)
    # Each corner gives where the edge to it crosses the line, if it does, then itself, if it's inside
    valid = k < counts[:, None]
    points = np.stack([crossing, points], axis=2).reshape(n, 2 * width, 2)
    points, counts = compact_corners(points, np.stack([valid & (inside != was_inside), valid & inside],
                                                      axis=2).reshape(n, 2 * width))
    # A corner on the line comes out twice
    k = np.arange(points.shape[1])
    previous = take_corners(points, (k - 1) % np.maximum(counts, 1)[:, None])
    return compact_corners(points, (k < counts[:, None]) & (points != previous).any(axis=2))


def take_corners(points, indices):
    """points (n, corners, 2) rearranged along each row by indices (n, m)"""
    return points[np.arange(len(points))[:, None], indices]


def compact_corners(points, flags):
    """The corners of each row of points (n, corners, 2 or 3) that flags (n, corners) keeps, moved to its start in order,
    as (points, counts); past each row's count, zeros. Each kept corner goes straight to its place (how many kept ones
    come before it): a stable argsort of every row and take_along_axis, as until 2026-11-05, took twice as long, and
    most of a girih level's split (tiles())."""
    counts = flags.sum(axis=1)
    out = np.zeros((len(points), max(1, int(counts.max(initial=0))), points.shape[2]))
    rows, columns = np.nonzero(flags)
    out[rows, (np.cumsum(flags, axis=1) - 1)[rows, columns]] = points[rows, columns]
    return out, counts


def side_ranges(lows, highs, window):
    """Along one axis, boxes' exact low and high sides as (lows, highs, starts, ends): the sides as floats no further
    out than just past the window (past_window()), and where the window's part of each box starts and ends, as fractions
    of the box across (0 and 1 when it's all within it). The box's shading goes by those (see box_mesh())."""
    starts, ends = [], []
    for low, high in zip(lows, highs):
        size = high - low
        starts.append(float((max(low, -window) - low) / size) if size > 0 else 0.0)
        ends.append(float((min(high, window) - low) / size) if size > 0 else 1.0)
    return (np.array([past_window(v, window) for v in lows]), np.array([past_window(v, window) for v in highs]),
            np.array(starts), np.array(ends))


# A face's corners in order, by side (low 0 or high 1) along the face's two other axes, for a face on the low and on
# the high side of its own axis: wound to face out, and starting on the diagonal its two triangles split along, from
# low on both to high on both, as in the map's own boxes (the build script's box()), which a quad shades along.
FACE_ORDER = {0: [(1, 1), (1, 0), (0, 0), (0, 1)], 1: [(0, 0), (1, 0), (1, 1), (0, 1)]}


def box_mesh(ranges, w, open_faces):
    """The faces of every box in a grid of them (every combination of each axis's sides), each cut off at the window
    (-w to w along each axis), wound to face out, with the normal at each of their corners: (corners (n, 3), normals
    (n, 3), corners per face (m,), rows (m,)), each face's corners in a row, and rows the grid row of each face's box
    (its axes' indices in turn, the last fastest). A face is built if it's within the window and not open. ranges: per
    axis, (lows, highs, starts, ends) from side_ranges().

    Shaded like the map's own boxes, smooth (the Inferno's round-looking columns), so a pillar looks the same handed
    from its patch to here on a loop: each corner's normal is the average of the box's faces meeting there, set as
    custom normals. With flat faces here, diving at a wall, the loop that handed it over brightened it by 15% (the
    junction flight, 2026-10-29). A face the window cuts gets the normals the whole face would have there, as if the box
    went on, cut again along the diagonal it shades along if that crosses it: with its corners' own normals, the shading
    changed on the loop where a pillar outgrew the window."""
    grid = lambda k: np.stack([a.ravel() for a in np.meshgrid(*(r[k] for r in ranges), indexing='ij')], axis=1)
    lows, highs, starts, ends = grid(0), grid(1), grid(2), grid(3)
    clipped = (np.maximum(lows, -w), np.minimum(highs, w))
    spans = clipped[0] < clipped[1]
    whole = (starts == 0.0) & (ends == 1.0)
    corners, normals, sizes, rows = [np.zeros((0, 3))], [np.zeros((0, 3))], [np.zeros(0, dtype=np.int64)], \
        [np.zeros(0, dtype=np.int64)]
    for axis in range(3):
        # The other two axes, in the order that makes their cross product point along +axis
        b, c = (axis + 1) % 3, (axis + 2) % 3
        for side, sides in ((0, lows), (1, highs)):
            if FACES[2 * axis + side] in open_faces:
                continue
            # Within the window, its plane is the box's side, uncut
            built = (np.abs(sides[:, axis]) <= w) & spans[:, b] & spans[:, c]
            corner_normals = face_corner_normals(axis, side, open_faces)
            # Faces all within the window, as quads
            full = np.flatnonzero(built & whole[:, b] & whole[:, c])
            quads = np.zeros((len(full), 4, 3))
            quads[:, :, axis] = sides[full, axis][:, None]
            for k, (sb, sc) in enumerate(FACE_ORDER[side]):
                quads[:, k, b], quads[:, k, c] = clipped[sb][full, b], clipped[sc][full, c]
            corners.append(quads.reshape(-1, 3))
            normals.append(np.tile([corner_normals[sb, sc] for sb, sc in FACE_ORDER[side]], (len(full), 1)))
            sizes.append(np.full(len(full), 4))
            rows.append(full)
            # Faces the window cuts, piece by piece: only the biggest pillars near the window's edges
            for i in np.flatnonzero(built & ~(whole[:, b] & whole[:, c])):
                for piece, below in face_pieces(starts[i, b], ends[i, b], starts[i, c], ends[i, c], side):
                    points = np.zeros((len(piece), 3))
                    points[:, axis] = sides[i, axis]
                    for k, (s, t, _u, _v) in enumerate(piece):
                        points[k, b] = clipped[0][i, b] + (clipped[1][i, b] - clipped[0][i, b]) * s
                        points[k, c] = clipped[0][i, c] + (clipped[1][i, c] - clipped[0][i, c]) * t
                    corners.append(points)
                    normals.append(np.array([face_field(corner_normals, u, v, below) for _s, _t, u, v in piece]))
                    sizes.append(np.array([len(piece)]))
                    rows.append(np.array([i]))
    return np.concatenate(corners), np.concatenate(normals), np.concatenate(sizes), np.concatenate(rows)


def face_corner_normals(axis, side, open_faces):
    """The normals at a box face's corners, smooth-shaded: (2, 2, 3) by side along its two other axes, each the
    average of the face's normal and those of the box's faces meeting it there (not open ones)"""
    b, c = (axis + 1) % 3, (axis + 2) % 3
    result = np.zeros((2, 2, 3))
    for sb in (0, 1):
        for sc in (0, 1):
            n = np.zeros(3)
            n[axis] = 1.0 if side else -1.0
            for other, s in ((b, sb), (c, sc)):
                if FACES[2 * other + s] not in open_faces:
                    n[other] = 1.0 if s else -1.0
            result[sb, sc] = n / np.linalg.norm(n)
    return result


def face_pieces(u0, u1, v0, v1, side):
    """The window's part [u0, u1] x [v0, v1] of a box face (in fractions of the whole face across, along its two other
    axes), as polygons wound like the face, each with whether it's below the diagonal u = v the face shades along: all
    of it if it's on one side of that, else each side. A polygon's corners are (s, t, u, v): where they are in the part,
    0 to 1 across it (exact at its own corners, so it fills the window's part exactly), and in the whole face, for the
    normals. A part too small to tell where it is (from a pillar a million times bigger than the window) is one piece,
    with the normals at its middle all over: a billion times bigger, the fractions were the same float, and placing its
    corners by them squashed it to nothing."""
    du, dv = u1 - u0, v1 - v0
    rectangle = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    if not side:
        rectangle = [(1.0, 1.0), (1.0, 0.0), (0.0, 0.0), (0.0, 1.0)]
    if du < 1e-9 or dv < 1e-9:
        u, v = (u0 + u1) / 2, (v0 + v1) / 2
        return [([(s, t, u, v) for s, t in rectangle], v <= u)]
    corners = [(s, t, u0 + s * du, v0 + t * dv) for s, t in rectangle]
    if v1 <= u0 or v0 >= u1:
        return [(corners, v1 <= u0)]
    pieces = []
    for below in (True, False):
        piece = []
        for p, q in zip(corners, corners[1:] + corners[:1]):
            dp, dq = p[3] - p[2], q[3] - q[2]
            if (dp <= 0.0) if below else (dp >= 0.0):
                piece.append(p)
            if (dp < 0.0 < dq) or (dq < 0.0 < dp):
                f = dp / (dp - dq)
                piece.append(tuple(a + (b - a) * f for a, b in zip(p, q)))
        if len(piece) >= 3:
            pieces.append((piece, below))
    return pieces


def face_field(corner_normals, u, v, below):
    """The normal a smooth box face shades with at (u, v), interpolated from the whole face's corner normals across
    whichever of its two triangles that's in (split along u = v): below it or above. On the diagonal, both agree."""
    n00, n10, n11, n01 = corner_normals[0, 0], corner_normals[1, 0], corner_normals[1, 1], corner_normals[0, 1]
    if below:
        return n00 + u * (n10 - n00) + v * (n11 - n10)
    return n00 + v * (n01 - n00) + u * (n11 - n01)


def solid_faces(polygon, axes="xy", low=0.0, high=1.0, open_faces=()):
    """A piece's faces in its own units, for a mesh of its own (parts.py instances them): a simple polygon of any shape
    (an arch's outline too), its corners in the two axes named (e.g. "xz" for an outline standing up), extruded along
    the third from low to high: its sides, and its caps unless open ("-y", "+y" and so on), wound to face out whichever
    way the polygon was given. As (corners (n, 3), corners per face (m,))."""
    a, b = ("xyz".index(axes[0]), "xyz".index(axes[1]))
    k = 3 - a - b
    polygon = np.asarray(polygon, dtype=float)
    area = np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) - np.roll(polygon[:, 0], -1) * polygon[:, 1])
    # Counterclockwise looking down the extrusion axis, so its sides face out and its high cap up it
    if (area < 0) == ((a, b, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1))):
        polygon = polygon[::-1]
    count = len(polygon)
    rings = np.zeros((2, count, 3))
    rings[:, :, a], rings[:, :, b] = polygon[:, 0], polygon[:, 1]
    rings[0, :, k], rings[1, :, k] = low, high
    following = np.roll(np.arange(count), -1)
    sides = np.stack([rings[0], rings[0][following], rings[1][following], rings[1]], axis=1).reshape(-1, 3)
    corners, sizes = [sides], [np.full(count, 4)]
    axis = "xyz"[k]
    if "+" + axis not in open_faces:
        corners.append(rings[1])
        sizes.append([count])
    if "-" + axis not in open_faces:
        corners.append(rings[0][::-1])
        sizes.append([count])
    return np.concatenate(corners), np.concatenate(sizes).astype(np.int64)


def write_mesh(obj, corners, sizes, colors, normals=None, convex=False):
    """obj's mesh as faces, each with its own corners (a face's in a row), colored per face by "Col", which Solid mode
    shows, with normals as Blender's free custom normals if given.

    Corners and the corner list go straight into their attribute arrays, and the faces get only their starts (each
    face's corner count follows from the next one's start): through vertices.co, loops.vertex_index and loop_total,
    a girih level of 10k faces took 6.9 ms, which a dive spent some 30 times a second (2026-11-05), and 1.4 ms this way,
    the same mesh (~/blender_tests/girih/perf/write_mesh_equal.py).

    What Blender does with a written mesh before it's drawn is kept short too (2026-11-14: a tenth of the main thread
    in a dive, with some 20 levels written a second):
    - Without normals given, the faces are flat ("sharp_face"): each has its own corners, so it looks the same, but
      for a smooth mesh Blender works out every vertex's normal (and which faces meet at it, for that).
    - With convex, faces of five corners and more are written as quadrilaterals (and a triangle, for an odd one)
      fanned from their first corner, sharing its corners: Blender cuts a polygon of five corners or more into
      triangles by a general method (for any outline), ten times the work of a quadrilateral's. Only for polygons
      known to be convex, and flat.
    - "Col" is a color per vertex (a face's corners are its own vertices), so the colors don't depend on how the
      faces are cut: a caller's array of one value per corner still fits ("Col", or its own attributes on points)."""
    if obj is None or obj.type != 'MESH':
        return
    mesh = obj.data
    count = len(corners)
    sizes = np.asarray(sizes, dtype=np.int64)
    starts = np.cumsum(sizes) - sizes
    corner_vert = np.arange(count, dtype=np.int32)
    face_starts = starts
    if convex and normals is None and len(sizes) and int(sizes.max()) > 4:
        # Polygon f in (sizes - 1) // 2 pieces: piece j has its corners 0, 2j + 1, 2j + 2 and, unless that's past its
        # last, 2j + 3
        pieces = (sizes - 1) // 2
        polygon = np.repeat(np.arange(len(sizes)), pieces)
        j = np.arange(len(polygon)) - np.repeat(np.cumsum(pieces) - pieces, pieces)
        first = starts[polygon]
        four = 2 * j + 3 < sizes[polygon]
        piece_sizes = np.where(four, 4, 3)
        face_starts = np.cumsum(piece_sizes) - piece_sizes
        corner_vert = np.empty(int(piece_sizes.sum()), dtype=np.int32)
        corner_vert[face_starts] = first
        corner_vert[face_starts + 1] = first + 2 * j + 1
        corner_vert[face_starts + 2] = first + 2 * j + 2
        corner_vert[face_starts[four] + 3] = (first + 2 * j + 3)[four]
    mesh.clear_geometry()
    mesh.vertices.add(count)
    mesh.loops.add(len(corner_vert))
    mesh.polygons.add(len(face_starts))
    if count:
        mesh.attributes["position"].data.foreach_set("vector", np.asarray(corners, dtype=np.float32).ravel())
        mesh.attributes[".corner_vert"].data.foreach_set("value", corner_vert)
    mesh.polygons.foreach_set("loop_start", face_starts.astype(np.int32))
    attribute = mesh.color_attributes.get("Col") or mesh.color_attributes.new("Col", 'FLOAT_COLOR', 'POINT')
    if count:
        colors = np.repeat(np.asarray(colors, dtype=np.float32).reshape(-1, 4), sizes, axis=0)
        attribute.data.foreach_set("color", colors.ravel())
    mesh.color_attributes.active_color = attribute
    mesh.color_attributes.render_color_index = 0
    if normals is not None:
        custom = mesh.attributes.get("custom_normal") or mesh.attributes.new("custom_normal", 'FLOAT_VECTOR', 'CORNER')
        if count:
            custom.data.foreach_set("vector", np.asarray(normals, dtype=np.float32).ravel())
    elif len(face_starts):
        sharp = mesh.attributes.get("sharp_face") or mesh.attributes.new("sharp_face", 'BOOLEAN', 'FACE')
        sharp.data.foreach_set("value", np.ones(len(face_starts), dtype=bool))
    mesh.update()


def build(obj, settings, factor, origin):
    """Rebuild the tiling's biggest levels (settings, the endless settings' "tiling") for the map's origin at origin
    (see the module's docstring): each solid part's object, and each flat part's slots, how many are filled, the
    covered flag its patches read and the floor's tint."""
    for part in parts(settings):
        chosen, cover = chosen_levels(part, factor, origin)
        if part.solid:
            faces = Faces()
            for level in chosen:
                (smooth_box_faces if part.smooth else prism_faces)(part, factor, origin, level, faces)
            corners, normals, sizes, colors = faces.arrays()
            write_mesh(part.settings.get("object"), corners, sizes, colors, normals)
            continue
        for slot, entry in enumerate(part.settings["slots"]):
            corners, sizes, colors, z = np.zeros((0, 2)), np.zeros(0, dtype=np.int64), np.zeros((0, 4)), 0.0
            if slot < len(chosen):
                corners, sizes, colors = flat_faces(part, factor, origin, chosen[slot])
                z = past_window(part.low * part.cell * Fraction(factor) ** -chosen[slot], part.window)
            write_mesh(entry["object"], np.column_stack([corners, np.full(len(corners), z)]), sizes, colors)
        obj[f"tiling_{part.name}_slots"] = float(min(len(chosen), len(part.settings["slots"])))
        obj[f"tiling_{part.name}_covered"] = 0.0 if cover is None else 1.0
        floor = part.settings.get("floor")
        if floor is not None and floor.type == 'MESH':
            color = part.settings["floor_color"] if cover is None else part.color(cover[3], cover[1], cover[2])
            attribute = floor.data.color_attributes.get("Col")
            if attribute is not None:
                attribute.data.foreach_set("color", np.tile(np.asarray(color[:], dtype=np.float32),
                                                            len(attribute.data)))
                floor.data.update()


def level_for(part, factor, distance):
    """The finest level of a part a move of distance (exact, in the map's units) changes: it repeats over its period's
    cells"""
    level = 0
    while (distance / (part.period * part.cell * Fraction(factor) ** -level)).denominator == 1:
        level -= 1
    return level


def in_window(part, factor, origin, moved):
    """Whether any piece of the levels a move of moved changes is in sight, for the map's origin at origin"""
    first = min(level_for(part, factor, moved), part.levels_from)
    return next(levels(part, factor, origin, first), None) is not None
