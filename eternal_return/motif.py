"""Motifs: what a floor draws on each of its tiles, the same on every tile of a kind, at every level: one motif language
for every engine. A motif is flat pieces per tile kind, polygons in the kind's prototile frame (the tile's own units),
each a bar's core or one of its outline strips, and painted one of the motif's paints:
- Hankin: girih strapwork, Hankin's method (bars along lines meeting the tiles' edges at a contact angle, 54 degrees
  by default), on a substitution rule's prototiles (substitution.py, surface_substitution.py); and the "edges" style,
  each prototile's edges as bars
- Pieces: convex polygons given in a cell's units, each with a paint, on the unit square of a square lattice (the
  plane maps' tiling.py motifs: the Inferno's star outlines and gold lines, strapwork), cut to the cell, the parts
  sticking out of it drawn by the neighbors they stick into (pieces are periodic), so each piece of a mesh's cell stays
  on its own face

Every motif lays its pieces out alike (Motif.lay_out()): bars[kind] = (polygons (n, corners, 2), counts (n,)), the
same with each polygon's corners the other way round for mirrored tiles (bars_reversed), edges[kind] (whether each
polygon is an outline strip), paints[kind] (each polygon's paint, an index into paint_names) and boxes[kind] (their
bounding boxes)."""

import math

import numpy as np

from .cyclotomic import RING10 as RING

class Motif:
    """Flat pieces per tile kind, laid out (lay_out()) as the module's docstring says"""

    def __init__(self):
        self.bars, self.bars_reversed, self.edges, self.paints, self.boxes = {}, {}, {}, {}, {}
        self.paint_names = ["bar"]

    def lay_out(self, kind, polygons, edges, paints=None):
        """A kind's polygons (each a list of corners), whether each is an outline strip, and each one's paint (all the
        first if None), as bars, bars_reversed, edges, paints and boxes have them"""
        self.edges[kind] = np.array(edges, dtype=bool)
        self.paints[kind] = np.zeros(len(polygons), np.int64) if paints is None else np.array(paints, np.int64)
        width = max([3] + [len(p) for p in polygons])
        array = np.zeros((len(polygons), width, 2))
        reverse = np.zeros((len(polygons), width, 2))
        counts = np.array([len(p) for p in polygons], dtype=np.int64)
        for i, polygon in enumerate(polygons):
            array[i, :len(polygon)] = polygon
            reverse[i, :len(polygon)] = polygon[::-1]
        self.bars[kind] = (array, counts)
        self.bars_reversed[kind] = (reverse, counts)
        # Each polygon's bounding box (lo (n, 2), hi (n, 2)), to pass over those far from the view
        valid = (np.arange(width) < counts[:, None])[..., None]
        self.boxes[kind] = (np.where(valid, array, np.inf).min(axis=1), np.where(valid, array, -np.inf).max(axis=1))


class Strapwork(Motif):
    """Bars along segments per kind (prototile floats, (n, 2, 2)), half_width either side, in tile units, each already
    cut to its own tile (two tiles' bars meet at edge midpoints and would overlap in one plane), the same for every
    tile of a kind. A tile that isn't convex (darb_imam's bowtie) is cut into convex pieces (rule.module.pieces()), and
    a bar across two is cut in two.

    With an outline (a share of the bar's width, 0 to 0.5), each bar is a core in its level's color with a strip of the
    outline color along each side (outlined(): every level's drawing has dark in it, so diving into a bar never leaves
    the dark behind), and edges[kind] says which polygons are outline strips."""

    def __init__(self, rule, segments, half_width=0.007, outline=0.0):
        super().__init__()
        self.rule, self.half_width, self.outline = rule, half_width, outline
        self.segments = {k: np.array(segs, dtype=float).reshape(-1, 2, 2) for k, segs in segments.items()}
        for kind, segs in self.segments.items():
            pieces = rule.module.pieces(kind)
            polygons, edges = [], []
            if outline > 0.0:
                polygons, edges = outlined(segs, pieces, half_width, outline)
            for a, b in (segs if outline <= 0.0 else ()):
                u = (b - a) / np.hypot(*(b - a))
                n = np.array([-u[1], u[0]]) * half_width
                a, b = a - u * half_width, b + u * half_width
                for piece in pieces:
                    polygon = [a - n, b - n, b + n, a + n]
                    for i in range(len(piece)):
                        polygon = clip_to_left(polygon, piece[i], piece[(i + 1) % len(piece)])
                    if len(polygon) >= 3:
                        polygons.append(polygon)
                        edges.append(False)
            self.lay_out(kind, polygons, edges)


class Hankin(Strapwork):
    """Girih strapwork by Hankin's method: a style "hankin<angle>" (54 by default) draws the rule's lines meeting each
    tile's edges at that contact angle (rule.module.lines())"""

    def __init__(self, rule, style="hankin54", half_width=0.007, outline=0.0):
        theta = float(style[len("hankin"):]) if style.startswith("hankin") else 54.0
        super().__init__(rule, {k: rule.module.lines(k, theta) for k in rule.kinds}, half_width, outline)
        self.style = style


class Edges(Strapwork):
    """The "edges" style (Penrose): each half's legs as bars (its rhomb's edges, cut to the half; the base, inside the
    rhomb, isn't drawn), so each level's rhombs show, split by the next level's"""

    def __init__(self, rule, style="edges", half_width=0.012, outline=0.0):
        proto = {kind: [RING.xy_small(c) for c in rule.module.PROTOTILE[kind]] for kind in rule.kinds}
        super().__init__(rule, {kind: [(proto[kind][0], proto[kind][1]), (proto[kind][0], proto[kind][2])]
                                for kind in rule.kinds}, half_width, outline)
        self.style = style


class Pieces(Motif):
    """A square lattice's motif on the unit cell [0, 1] x [0, 1]: pieces, each (convex polygon of [u, v] corners in the
    cell's units, counterclockwise, paint name), as the plane maps' tiling.py motifs are given. Three kinds (CELLS): the
    square cell, and its halves either side of its diagonal, up ((0, 0), (1, 0), (0, 1)) and down ((1, 0), (1, 1),
    (0, 1)), a triangle face's cells (a triangle splits into k^2 of them, up and down, in its own (u, v)): the plane's
    triangle maps' motifs are drawn on square cells sheared into two equilateral triangles. With an
    outline (a share of a strip's width each side), each quad piece whose corners 0 to 1 and 3 to 2 run along it (a
    bar or an outline's quad) is a core between two outline strips (the build scripts' plane_tiling.outlined()), and
    with cores, the cores alone (as a level's far cells are drawn). Every polygon is cut to the cell: what sticks out
    of it is drawn by the cell it sticks into (each piece shifted a cell each way and cut again: the motif repeats), so a
    cell's pieces stay on its own face of a mesh, and two faces' cells meet seamlessly at a fold. A motif that draws its
    own outlines names the paints of its strips (strips): those pieces are outline strips as they are, and no quad is
    outlined again."""

    def __init__(self, pieces, outline=0.0, cores=False, strips=()):
        super().__init__()
        self.outline, self.cores, self.strips = outline, cores, set(strips)
        if self.strips:
            # A motif that draws its own outlines (the pieces painted strips: the triangle motif's bands' borders)
            # isn't outlined again, and with cores its strips are left out
            pieces = [(polygon, paint) for polygon, paint in pieces if not (cores and paint in self.strips)]
        self.paint_names = list(dict.fromkeys(paint for _polygon, paint in pieces))
        for kind, region in enumerate(CELLS):
            self.lay_out(kind, *self.cut(pieces, [np.array(c, dtype=float) for c in region]))

    def cut(self, pieces, region):
        """Every piece's polygons (outlined) cut to region (a convex polygon in the cell), each shifted a cell each way
        too: (polygons, whether each is an outline strip, each one's paint)"""
        outline, cores = (0.0 if self.strips else self.outline), self.cores
        polygons, edges, paints = [], [], []
        for polygon, paint in pieces:
            polygon = np.asarray(polygon, dtype=float)
            if paint in self.strips:
                parts = [(polygon, True)]
            elif outline > 0.0 and len(polygon) == 4:
                parts = [(across(polygon, outline, 1.0 - outline), False)]
                if not cores:
                    parts += [(across(polygon, 0.0, outline), True), (across(polygon, 1.0 - outline, 1.0), True)]
            else:
                parts = [(polygon, False)]
            for part, strip in parts:
                for dx in (-1.0, 0.0, 1.0):
                    for dy in (-1.0, 0.0, 1.0):
                        cut = [np.asarray(c) for c in part + (dx, dy)]
                        for i in range(len(region)):
                            cut = clip_to_left(cut, region[i], region[(i + 1) % len(region)])
                            if len(cut) < 3:
                                break
                        if len(cut) >= 3 and polygon_area(cut) > PIECE_SLIVER:
                            polygons.append(cut)
                            edges.append(strip)
                            paints.append(self.paint_names.index(paint))
        return polygons, edges, paints


# A square lattice's cell kinds (Pieces): the square, and its up and down halves (a triangle face's cells)
CELLS = (((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
         ((1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
# A piece cut to its cell smaller than this (cell units squared) is left out: a sliver where it only touches the cell
PIECE_SLIVER = 1e-12


def across(quad, start, end):
    """The strip of a quad whose corners 0 to 1 and 3 to 2 run along it from start to end of the way across it (0 the
    side through corners 0 and 1, 1 the other): for a mitered outline quad, a narrower one with the same miters, so the
    strips of neighboring quads still meet edge to edge"""
    quad = np.asarray(quad, dtype=float)
    side, other = quad[[0, 1]], quad[[3, 2]]
    low, high = side + start * (other - side), side + end * (other - side)
    return np.array([low[0], low[1], high[1], high[0]])

# outlined(): how close (prototile units) two ends are to be one joint, and a point to be on a piece's edge
JOINT = 1e-9
# A bar's end on its piece's edge reaches this many half widths past it before it's cut there (so a bar that meets the
# edge at a slant still fills to it: 20 is enough down to 3 degrees)
REACH_PAST = 20.0


def outlined(segments, pieces, half_width, outline):
    """A tile's bars with outlines, cut to its pieces: (polygons [[corners]], edges [is an outline strip]). Each bar is
    a core (half width h_c = (1 - 2 outline) h) and two strips from h_c to h, one each side. So that nothing overlaps
    in one plane but polygons of one color:
    - Two bars ending at the same point inside the tile (a star's tip) are cut on the angle's bisector there: the
      cores and both pairs of strips meet in clean miters.
    - A bar ending on a piece's edge (where its line runs on into the next tile, or a half's base) reaches well past it
      and is cut there, so it runs on seamlessly from the next tile's, cut the same way.
    - Where bars cross, the cores win: each strip has every other bar's core cut out of it (convex differences), so
      crossing strands meet in a plain patch of the level's color, the same from both tiles at a crossing on an edge."""
    h, h_c = half_width, half_width * (1.0 - 2.0 * outline)
    segments = [(np.asarray(a, float), np.asarray(b, float)) for a, b in segments]
    edges = [(np.asarray(piece[i], float), np.asarray(piece[(i + 1) % len(piece)], float))
             for piece in pieces for i in range(len(piece))]

    def on_edge(p):
        for a, b in edges:
            e = b - a
            t = float((p - a) @ e) / float(e @ e)
            if -JOINT <= t <= 1.0 + JOINT and abs(e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0])) <= JOINT * math.hypot(*e):
                return True
        return False

    # Each end's treatment: (how far past the end the bar reaches, the half plane it's cut to there or None)
    ends = []
    for i, (a, b) in enumerate(segments):
        treat = []
        for p, away in ((a, b), (b, a)):
            out = (p - away) / np.hypot(*(p - away))                  # along the bar, out past this end
            others = [(q, r) for j, (q0, q1) in enumerate(segments) if j != i
                      for q, r in ((q0, q1), (q1, q0)) if np.hypot(*(q - p)) <= JOINT]
            if on_edge(p):
                treat.append((REACH_PAST * h, None))
            elif len(others) == 1 and abs(float(out @ ((others[0][1] - p) / np.hypot(*(others[0][1] - p))))) < 1.0 - 1e-9:
                # The bisector of the angle between the two bars: this bar keeps the side its own body is on
                mine = -out
                theirs = (others[0][1] - p) / np.hypot(*(others[0][1] - p))
                bisector = mine + theirs
                half_angle = math.acos(max(-1.0, min(1.0, float(mine @ theirs)))) / 2.0
                reach = h / max(math.tan(half_angle), 1e-3) + h
                # clip_to_left keeps the left of a line: the line along the bisector, turned so the body is on its left
                a_line, b_line = p, p + bisector
                side = bisector[0] * mine[1] - bisector[1] * mine[0]
                treat.append((reach, (a_line, b_line) if side >= 0.0 else (b_line, a_line)))
            else:
                treat.append((h, None))
        ends.append(treat)

    def band(i, lo, hi):
        """Bar i's strip between offsets lo and hi (left of its direction), its ends as treated, one convex polygon"""
        a, b = segments[i]
        u = (b - a) / np.hypot(*(b - a))
        n = np.array([-u[1], u[0]])
        (reach_a, cut_a), (reach_b, cut_b) = ends[i]
        a, b = a - u * reach_a, b + u * reach_b
        polygon = [a + n * lo, b + n * lo, b + n * hi, a + n * hi]
        for cut in (cut_a, cut_b):
            if cut is not None and len(polygon) >= 3:
                polygon = clip_to_left(polygon, *cut)
        return polygon

    cores = [band(i, -h_c, h_c) for i in range(len(segments))]
    shapes = [(core, False) for core in cores]
    for i in range(len(segments)):
        for lo, hi in ((-h, -h_c), (h_c, h)):
            parts = [band(i, lo, hi)]
            for j, core in enumerate(cores):
                if j != i and len(core) >= 3:
                    parts = [rest for part in parts for rest in subtract_convex(part, core)]
            shapes += [(part, True) for part in parts]
    polygons, flags = [], []
    for polygon, edge in shapes:
        for piece in pieces:
            cut = polygon
            for k in range(len(piece)):
                if len(cut) < 3:
                    break
                cut = clip_to_left(cut, np.asarray(piece[k], float), np.asarray(piece[(k + 1) % len(piece)], float))
            if len(cut) >= 3 and polygon_area(cut) > 1e-6 * h * h:
                polygons.append(cut)
                flags.append(edge)
    return polygons, flags


def polygon_area(polygon):
    x = np.array([p[0] for p in polygon])
    y = np.array([p[1] for p in polygon])
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def subtract_convex(polygon, hole):
    """A convex polygon less a convex hole (both counterclockwise), as convex polygons: the part outside each of the
    hole's edges in turn, inside the ones before it (the polygon itself if they don't overlap)"""
    common = polygon
    for k in range(len(hole)):
        common = clip_to_left(common, np.asarray(hole[k], float), np.asarray(hole[(k + 1) % len(hole)], float))
        if len(common) < 3:
            return [polygon]
    if polygon_area(common) <= 1e-12 * max(polygon_area(polygon), 1e-300):
        return [polygon]
    out, rest = [], polygon
    for k in range(len(hole)):
        a, b = np.asarray(hole[k], float), np.asarray(hole[(k + 1) % len(hole)], float)
        outside = clip_to_left(rest, b, a)
        if len(outside) >= 3 and polygon_area(outside) > 0.0:
            out.append(outside)
        rest = clip_to_left(rest, a, b)
        if len(rest) < 3:
            return out
    return out


def clip_to_left(polygon, a, b):
    """A convex polygon cut to the left of the line from a to b"""
    e = b - a
    side = lambda p: e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0])
    out = []
    for i, p in enumerate(polygon):
        q = polygon[(i + 1) % len(polygon)]
        sp, sq = side(p), side(q)
        if sp >= 0:
            out.append(p)
        if (sp >= 0) != (sq >= 0):
            out.append(p + (q - p) * (sp / (sp - sq)))
    return out
