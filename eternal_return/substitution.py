"""Endless substitution floors: a substitution tiling with girih strapwork at every level, bigger and smaller forever,
which loops about any point, for the user's "floors with nested girih tiles" (2026-11-05). Two rules: penrose.py's
Penrose rhombs (as Robinson half rhombs), looping by phi, and darb_imam.py's true girih tiles (decagons, elongated
hexagons, bowties), looping by 2 phi^3 = 8.47 (2026-11-06). A map's settings name its "rule" (penrose if none).

A lattice floor (endless.py, tiling.py) repeats, so its small levels are patches that follow the drone and its anchor
is digits. A substitution tiling never repeats: each level's tiles are cut from the next level's by the rule. So the
whole universe is kept as one exact tile, `made` (the biggest ancestor of the origin's tile decided so far), at
`made_level`, everything inside it following from the rule, exactly (cyclotomic.py: every place is four Python ints, or
Fractions for a rule whose factor isn't a unit). Prototyped and tested in ~/blender_tests/girih (universe.py,
test_universe.py: dives, climbs, moves and revisits through a renderer, with no change at any loop).

- Levels are relative to the drone: level 0's tiles are about its size (settings "cell" across). A loop by LAMBDA^p
  about the map's origin O relabels every level p up (loop()), with no remainder: each level's tiles keep their places
  in their own units, only their names change.
- Moving O by an exact ring vector (recenter()) needs no symmetry: `made` just moves by it (divided by LAMBDA^made_level,
  which for darb_imam.py puts powers of 2 under its place), and the drone moves back by the same vector, rounded to
  floats once.
- Above `made`, the universe goes on by a default parent rule (each rule's DEFAULT): always the same slot in the same
  parent kind. That carries O to the same point of every ancestor, clear of any line or edge (0.0885 of its size for
  Penrose halves, 0.95 for darb_imam's concentric decagons), so an ancestor big enough draws nothing in sight. Those
  levels are drawn as they are from the start (the explicit top is the rule's EXTRA default levels above `made`).
  Climbing brings `made` down toward the view, and keep_ahead() fixes the next level up while it's still far bigger
  than the view: a random parent if neither the default levels above `made` nor the new ones draw anything in sight,
  else the default, what was already drawn. So climbing never pops: a lattice map's background digits, for a
  substitution tiling. The random parent is one that doesn't repeat the levels below it (the same choice twice, or two
  in turn), if one can: a Penrose level has only two or three to choose from, so a fair pick alone often ran the same
  short cycle for a few levels, and climbing, the view came back the same.
- darb_imam.py's children straddle their parents' edges (its STRADDLE): a tile decides only what's more than its PAD
  inside its edges, and a child can be listed by two parents. So every test of what a tile decides allows for PAD,
  every square a level's tiles are found for reaches CHILD_REACH further up the levels, and children found twice are
  dropped (exactly: a level's float tiles share one root place, so a tile has the same whole-number place whichever
  parent it came from).

Every decision is made with accurate floats of exact places (cyclotomic.xy_float), and too close to call, the safe way:
a tile drawn, a random parent refused. Levels bigger than GIANT aren't drawn from floats (a bar's place from them is
only good to 1e-16 of its level's size): giant_levels() places their bars near the drone in fixed point from the exact
tiles, and draws those whose edges are in sight; one whose bar fills the sight tints the floor instead.

What's drawn: each level k's bars within min(sight, radius * LAMBDA^k) of the drone's place over the floor (level-0
units), in one of the map's pool of level objects (settings "levels"): its mesh in level-k tile units around O, the
object scaled by cell * LAMBDA^k. Each bar is cut to its own tile, so two tiles' bars never overlap in the same plane,
cut to the level's window, and split into tiles (tiling.tiles(), for depth precision). Finer levels lie on top, by
plane_tiling's push layering: drivers scale each level object about the drone camera by 1 + its push (custom properties
"sub_push", "sub_bs" its unpushed scale, "sub_bx"/"sub_by" its unpushed place). A loop relabels the level objects
(scales them by LAMBDA^p) and a move of O shifts them; a level is only rebuilt once the drone has moved "margin" of its
tiles from where it was built (settle(), at the end of every tick).

A map can outline its bars (settings "outline", a share of a bar's width each side, outlined()): a core in the level's
color between two strips of the outline color, the floor's if none ("outline_color"). Every level then has the dark in
it, so diving into a bar, the view comes back to black, and the finer bars inside it show on its strips ("instead of
shapes fading in and out, the lines have black lines on either side", 2026-11-15, as the Penrose surface maps). Only
the tiles near the camera, where the strips are OUTLINE_PX wide or more, are outlined; the rest are their cores alone.
A map can fade its levels in by how wide their bars are on screen (settings "fade", fade()), from under a pixel, like
anti-aliasing. It can also fade them out into the floor's color as they grow wider ([in from, in to, out from, out
to]): the Darb-i Imam floor's way back to black from 2026-11-06 ("if i fly into an edge, i cant see the smaller edges
of the same color") until outlines; a level wholly faded out isn't drawn.

The map's object holds "substitution" (written by the map's build script): rule, cell, style, half_width, sight,
radius, margin, focal, step, seed, palette (a color a color class, the rule's CLASSES of them), floor, floor_color,
fade, fade_color, outline, outline_color, levels; and the state, "substitution_made" (text) and
"substitution_made_level".
"""

import math
import random
import time
from fractions import Fraction

import numpy as np
from mathutils import Matrix, Vector

from . import background
from . import darb_imam
from . import penrose
from . import tiling

PROPERTY = "substitution"
MADE = "substitution_made"
MADE_LEVEL = "substitution_made_level"
POOL = "substitution_pool"          # on the level objects: the flight harness's pop check leaves them out
# Levels are decided once they're this many times the sight across: a parent then has a line within sight of O
# seldom enough that most climbs get random parents (at 20, only 1 in 6 did: the rest had to stay default; at 200, 1 in
# 10 had to, 150 climbing loops in ~/blender_tests/girih/climb_variety/addon_climb.py, and at 2000 none did)
AHEAD = 2000.0
MARGIN = 1e-9               # float tests closer than this (relative) are too close to call
ROOT_SIZE = 1e4             # levels at least this big (level-0 units) are found exactly, smaller ones in floats
GIANT = 1e12                # bigger levels are drawn from exact tiles (giant_levels()), not floats
LAZY_PX = 1.0               # a level never built, its bars narrower than this under the drone, waits its turn
# A decision with more parents to choose from than this tries them in a random order until one will do, rather than
# trying them all: darb_imam's tiles have 50 to 150, and trying all, each through the levels above it, took a climbing
# loop's tick 100 ms and more. The same odds either way (any that will do, those that don't repeat the levels below
# first), but Penrose's two or three are all tried, as they were, so its universes decide as they always have.
TRY_ALL = 4
# A level whose window is capped at the sight is built at least this many times the sight across (settle()'s
# build_window()): it comes due once the drone has moved the difference, so a smaller slack rebuilt the biggest levels,
# whose bars are the widest on screen and go first, over and over, and the levels in view starved (a bounding square
# for what the finest one needs one climb on, 2026-11-06: Penrose's 60 s harness dive lost its middle levels).
# Penrose's phi, which built all of them so before darb_imam's 8.47.
CAPPED_SLACK = (1.0 + math.sqrt(5.0)) / 2.0
OUTLINE = -1                # an outline strip's color class: colors_of()'s last row
# A level is drawn outlined while its outline strips are at least this wide under the drone (px), as its bars' cores
# alone once they're under half that (in between, as it was): narrower strips can't be seen, and outlined, a level has
# some three times the faces (surface_penrose.py's OUTLINE_PX)
OUTLINE_PX = 2.0
# A giant level whose bars fill this much of its window, in one color class, is the floor's tint (giant_levels())
GIANT_COVER = 1.0 - 1e-9


# --- Rules --------------------------------------------------------------------------------------------------------------

class Rule:
    """A substitution rule (penrose.py, darb_imam.py: a module with the same names), and its tables as arrays, for
    cutting levels in floats: for each kind, each slot's child as the parent's turn and mirror put it (child_pose())"""

    def __init__(self, module):
        self.module = m = module
        self.name, self.ring, self.n = m.NAME, m.RING, m.N
        self.kinds = m.KINDS
        self.loop, self.log_loop = m.LOOP, math.log(m.LOOP)
        self.lambda_, self.lambda_inv = m.LAMBDA, m.LAMBDA_INV
        self.extra, self.pad, self.reach, self.child_reach = m.EXTRA, m.PAD, m.REACH, m.CHILD_REACH
        self.straddle, self.classes = m.STRADDLE, m.CLASSES
        R, N = self.ring, self.n
        self.lambda_matrix = np.array([R.mul(m.LAMBDA, R.power(j)) for j in range(4)], dtype=np.int64).T
        self.basis_x = np.array([R.xy_small(R.power(j))[0] for j in range(4)])
        self.basis_y = np.array([R.xy_small(R.power(j))[1] for j in range(4)])
        self.orient = np.array([[orient_matrix(N, r, mi) for mi in (False, True)] for r in range(N)])     # (N, 2, 2, 2)
        # poses[kind][rot][mirror] = [(kind, rot, mirror, place)] a slot each; a rule without mirrors has the
        # unmirrored ones in both
        self.poses, self.child = {}, {}
        for kind in self.kinds:
            slots = range(len(m.RULE[kind]))
            self.poses[kind] = [[[m.child_pose(kind, r, mi if mi in m.MIRRORS else False, s) for s in slots]
                                 for mi in (False, True)] for r in range(N)]
            poses = self.poses[kind]
            self.child[kind] = ChildTable(
                kind=np.array([c[0] for c in m.RULE[kind]], np.int8),
                rot=np.array([[[poses[r][mi][s][1] for mi in (0, 1)] for r in range(N)] for s in slots], np.int8),
                mirror=np.array([[[poses[r][mi][s][2] for mi in (0, 1)] for r in range(N)] for s in slots], bool),
                place=np.array([[[poses[r][mi][s][3] for mi in (0, 1)] for r in range(N)] for s in slots], np.int64),
                offset=np.array([[[R.xy_small(poses[r][mi][s][3]) for s in slots] for mi in (0, 1)] for r in range(N)]),
                box=np.array([[[bounds(m.corners_float(poses[r][mi][s])) for s in slots] for mi in (0, 1)]
                              for r in range(N)]))
        # Each kind's corners, padded to the most any kind has with copies of its last (for bounding boxes)
        width = max(len(m.PROTOTILE[kind]) for kind in self.kinds)
        self.corners = {}
        for kind in self.kinds:
            pts = [R.xy_small(v) for v in m.PROTOTILE[kind]]
            self.corners[kind] = np.array(pts + [pts[-1]] * (width - len(pts)))
        self.color_class = np.array([[[m.color_class(kind, r, bool(mi)) for mi in (0, 1)] for r in range(N)]
                                     for kind in self.kinds])                     # (kind, rot, mirror)
        self._powers = {}

    def size(self, k):
        """LAMBDA^k as a float, saturating: level k's tile size in level-0 units"""
        return math.exp(max(-700.0, min(700.0, k * self.log_loop)))

    def power(self, k):
        """LAMBDA^k as a ring element, for any whole k"""
        if k not in self._powers:
            R = self.ring
            base = self.lambda_ if k >= 0 else self.lambda_inv
            out, power, n = R.one, base, abs(k)
            while n:
                if n & 1:
                    out = R.mul(out, power)
                power = R.mul(power, power)
                n >>= 1
            if len(self._powers) > 4096:
                self._powers.clear()
            self._powers[k] = R.tidy(out)
        return self._powers[k]


class ChildTable:
    """A kind's children as arrays: kind (slots,), and for each slot and the parent's (rot, mirror), the child's rot,
    mirror and place (slots, N, 2[, 4]), the place as floats, offset (N, 2, slots, 2), and the child's bounding box
    from there, box (N, 2, slots, 4: x from, x to, y from, y to)"""

    def __init__(self, kind, rot, mirror, place, offset, box):
        self.kind, self.rot, self.mirror, self.place, self.offset, self.box = kind, rot, mirror, place, offset, box


def bounds(points):
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return min(xs), max(xs), min(ys), max(ys)


def orient_matrix(n, rotation, mirror):
    angle = rotation * 2 * math.pi / n
    turn = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return turn @ np.diag([1.0, -1.0]) if mirror else turn


RULES = {module.NAME: Rule(module) for module in (penrose, darb_imam)}


def rule_of(settings):
    return RULES[settings.get("rule", "penrose")]


# --- Exact helpers ----------------------------------------------------------------------------------------------------

def meets_square(rule, tile, h, center=(0.0, 0.0)):
    """Whether the tile might meet the square of half side h around center (its level's units), generously"""
    pts = rule.module.corners_float(tile)
    margin = MARGIN * (1.0 + max(abs(c) for p in pts for c in p))
    xs, ys = [p[0] - center[0] for p in pts], [p[1] - center[1] for p in pts]
    return not (max(xs) < -h - margin or min(xs) > h + margin or max(ys) < -h - margin or min(ys) > h + margin)


def children_near(rule, tile, h, center=(0.0, 0.0)):
    """The tile's children, (slot, child), that might meet the square of half side h around center (the children's
    units), in slot order: those whose bounding boxes do, in floats, more generously than meets_square(), and only
    those are worked out exactly. A darb_imam decagon has 211 children, and near the top, where a level's window is a
    speck of its tiles, all but one or two are far out of sight: each exact test took most of a decision's 20 ms."""
    R = rule.ring
    kind, r, m, tau = tile
    grown = R.mul(tau, rule.lambda_)
    gx, gy = R.xy_float(grown)
    box = rule.child[kind].box[r, int(m)]
    reach = h + 2.0 * MARGIN * (1.0 + abs(gx) + abs(gy)) + 1e-6
    x, y = gx - center[0], gy - center[1]
    near = (box[:, 1] + x >= -reach) & (box[:, 0] + x <= reach) & (box[:, 3] + y >= -reach) & (box[:, 2] + y <= reach)
    poses = rule.poses[kind][r][int(m)]
    return [(s, (poses[s][0], poses[s][1], poses[s][2], R.add(grown, poses[s][3]))) for s in np.flatnonzero(near)]


def cut(rule, tiles, h, center=(0.0, 0.0)):
    """The children of tiles (exact) that meet the square of half side h around center (their units), each once"""
    found = [c for t in tiles for _s, c in children_near(rule, t, h, center) if meets_square(rule, c, h, center)]
    return list(dict.fromkeys(found)) if rule.straddle else found


# --- Motif: girih strapwork, Hankin's method at 54 degrees ----------------------------------------------------------

class Motif:
    """A style's segments per kind (prototile floats, (n, 2, 2)) and its bars' half width, in tile units; and its bars
    as polygons, each already cut to its own tile (two tiles' bars meet at edge midpoints and would overlap in one
    plane), the same for every tile of a kind: bars[kind] = (polygons (n, corners, 2), counts (n,)), and the same
    with each polygon's corners the other way round, for mirrored tiles (which turn them clockwise). A tile that isn't
    convex (darb_imam's bowtie) is cut into convex pieces, and a bar across two is cut in two.

    With an outline (a share of the bar's width, 0 to 0.5), each bar is a core in its level's color with a strip of the
    outline color along each side (outlined(): every level's drawing has dark in it, so diving into a bar never leaves
    the dark behind, 2026-11-13), and edges[kind] says which polygons are outline strips."""

    def __init__(self, rule, style="hankin54", half_width=0.007, outline=0.0):
        self.rule, self.style, self.half_width, self.outline = rule, style, half_width, outline
        theta = float(style[len("hankin"):]) if style.startswith("hankin") else 54.0
        self.segments = {k: np.array(rule.module.lines(k, theta), dtype=float).reshape(-1, 2, 2) for k in rule.kinds}
        self.bars, self.bars_reversed, self.edges, self.boxes = {}, {}, {}, {}
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
            self.edges[kind] = np.array(edges, dtype=bool)
            width = max([3] + [len(p) for p in polygons])
            array = np.zeros((len(polygons), width, 2))
            reverse = np.zeros((len(polygons), width, 2))
            counts = np.array([len(p) for p in polygons], dtype=np.int64)
            for i, polygon in enumerate(polygons):
                array[i, :len(polygon)] = polygon
                reverse[i, :len(polygon)] = polygon[::-1]
            self.bars[kind] = (array, counts)
            self.bars_reversed[kind] = (reverse, counts)
            # Each polygon's bounding box (lo (n, 2), hi (n, 2)), for giant_bars() to pass over those far from the view
            valid = (np.arange(width) < counts[:, None])[..., None]
            self.boxes[kind] = (np.where(valid, array, np.inf).min(axis=1), np.where(valid, array, -np.inf).max(axis=1))


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


def shows(tile, level, h_world, motif):
    """Whether the tile's bars, at level, might come within h_world (level-0 units) of O, generously"""
    rule = motif.rule
    kind, r, m, tau = tile
    tx, ty = rule.ring.xy_float(tau)
    h = h_world / rule.size(level) + motif.half_width
    margin = MARGIN * (1.0 + abs(tx) + abs(ty))
    turn = rule.orient[r, int(m)]
    for p, q in motif.segments[kind]:
        a, b = turn @ p + (tx, ty), turn @ q + (tx, ty)
        e = b - a
        t = min(1.0, max(0.0, float(-(a @ e) / (e @ e))))
        if math.hypot(*(a + t * e)) <= h * math.sqrt(2.0) + margin:
            return True
    return False


# --- The universe ---------------------------------------------------------------------------------------------------

_tops = {}


class Universe:
    def __init__(self, rule, made, made_level):
        self.rule, self.made, self.made_level = rule, made, made_level

    def top(self):
        """The explicit top: the rule's EXTRA default parents above `made`, and its level. Kept for each `made`:
        level_tiles() asks for it at every level it finds, and 14 exact parents each time took 5 ms of a settle
        (2026-11-05)."""
        key = self.rule.name, self.made
        tile = _tops.get(key)
        if tile is None:
            tile = self.made
            for _ in range(self.rule.extra):
                tile = self.rule.module.default_parent(tile)
            if len(_tops) > 64:
                _tops.clear()
            _tops[key] = tile
        return tile, self.made_level + self.rule.extra

    def loop(self, power):
        """The view scaled by LAMBDA^power about O (diving, power > 0): every level relabeled power up"""
        self.made_level += power

    def recenter(self, v):
        """O moves by the exact ring vector v (level-0 units): the drone moves back by it"""
        R, rule = self.rule.ring, self.rule
        kind, r, m, tau = self.made
        self.made = (kind, r, m, R.tidy(R.sub(tau, R.mul(v, rule.power(-self.made_level)))))
        # If O left what `made` decides (it was near its edge), make default parents real, which changes nothing,
        # until one holds it
        while rule.module.inside(self.made) < 0:
            self.made = rule.module.default_parent(self.made)
            self.made_level += 1

    def chain_shows(self, made, made_level, sight, motif, include_made=False):
        """Whether any tile of the levels above made_level, up to EXTRA default parents above it, draws a bar within
        sight (level-0 units) of O"""
        rule = self.rule
        tile, level = made, made_level
        for _ in range(rule.extra):
            tile, level = rule.module.default_parent(tile), level + 1
        h = lambda k: sight / rule.size(k) + rule.pad
        current = [t for t in rule.module.tops(tile) if meets_square(rule, t, h(level))]
        lowest = made_level if include_made else made_level + 1
        while level >= lowest:
            if any(shows(t, level, sight, motif) for t in current):
                return True
            current = cut(rule, current, h(level - 1))
            level -= 1
        return False

    def decide(self, rng, sight, motif):
        """Fix the level above `made`: a random parent if that changes nothing within sight, else the default. Returns
        which and why (for the decisions counter): "random", "default by chance", "not inside" (the sight square isn't
        inside what `made` decides), "chain shows" (the default levels above it draw in sight) or "none valid" (every
        other parent would)."""
        rule = self.rule
        inside = rule.module.inside(self.made) * rule.size(self.made_level) > sight * math.sqrt(2.0) * (1 + MARGIN)
        default = rule.module.default_parent(self.made)
        choice = default
        if not inside:
            reason = "not inside"
        elif self.chain_shows(self.made, self.made_level, sight, motif):
            reason = "chain shows"
        else:
            # The candidate's own level too: it's the new level above `made`, and its bars can pass through the
            # view. Checking only the levels above it let a random parent with a band over the drone through now and
            # then (2026-11-05: "when flying out, it ocassionally jumpcuts to a different shape").
            valid = lambda p: (not self.chain_shows(p, self.made_level + 1, sight, motif, include_made=True)
                               and rule.module.inside(p) > 0)
            # Not one that repeats the levels below it, if there's another: each level's choice is one of two or
            # three, so a fair pick ran the same one or two choices over and over now and then (the default
            # chain, thin in thin, ...), and climbing, the same view came back level after level
            recent = recent_choices(rule, self.made, 3)
            options = rule.module.parents(self.made)
            picked = None
            if len(options) <= TRY_ALL:
                entries = [(c, p) for c, p in options if valid(p)]
                if entries:
                    fresh = [e for e in entries if not repeats(e[0], recent)]
                    picked = rng.choice(fresh or entries)[1]
            else:
                order = rng.sample(options, len(options))
                fresh = [e for e in order if not repeats(e[0], recent)]
                picked = next((p for _c, p in fresh if valid(p)), None)
                if picked is None:
                    picked = next((p for c, p in order if repeats(c, recent) and valid(p)), None)
            if picked is not None:
                choice = picked
                reason = "default by chance" if choice == default else "random"
            else:
                reason = "none valid"
        self.made, self.made_level = choice, self.made_level + 1
        return reason

    def keep_ahead(self, rng, sight, motif, ahead=AHEAD):
        """Decide levels until `made` is ahead times bigger than sight: the decisions' reasons (decide())"""
        reasons = []
        while self.rule.size(self.made_level) < ahead * sight:
            reasons.append(self.decide(rng, sight, motif))
        return reasons


def recent_choices(rule, made, count):
    """The parent choices (the rule's choice()) that made the count levels up to `made`: going down, which child of
    each holds O"""
    out, tile = [], made
    for _ in range(count):
        near = children_near(rule, tile, 0.0)
        if not near:
            break
        slot, child = max(near, key=lambda sc: rule.module.inside(sc[1]))
        out.append(rule.module.choice(tile[0], slot))
        tile = child
    return out


def repeats(choice, recent):
    """Whether choice, above the recent ones (nearest first), makes a run of one choice (AA) or of two (ABAB)"""
    return (len(recent) >= 1 and choice == recent[0]) or (len(recent) >= 3 and choice == recent[1]
                                                           and recent[0] == recent[2])


def start(rule, seed, sight, motif, view, below=3, attempts=200):
    """A random universe: O inside a level -below tile (the rule's start_tile()), random parents up to AHEAD times
    sight (any that has O inside what it decides), then decided the safe way. Drawn again while a bar wider than an
    eighth of the view passes within half the view of O."""
    rng = random.Random(seed)
    universe = None
    for _ in range(attempts):
        universe = Universe(rule, rule.module.start_tile(), -below)
        while rule.size(universe.made_level) < AHEAD * sight:
            options = [p for _c, p in rule.module.parents(universe.made) if rule.module.inside(p) > 0]
            universe.made = rng.choice(options)
            universe.made_level += 1
        universe.keep_ahead(rng, sight, motif)
        if not dominated(universe, motif, view):
            break
    return universe


def dominated(universe, motif, view):
    """Whether a bar wider than an eighth of the view passes within half the view of O"""
    rule = universe.rule
    widest = view / 8.0
    finest = math.ceil(math.log(widest / (2 * motif.half_width)) / rule.log_loop)
    for k, level in levels(universe, finest, lambda k: view).items():
        if 2 * motif.half_width * rule.size(k) < widest:
            continue
        ends, _ = level.bars(motif)
        if not len(ends):
            continue
        a, b = ends[:, 0], ends[:, 1]
        e = b - a
        t = np.clip(-(a * e).sum(axis=1) / np.maximum((e * e).sum(axis=1), 1e-300), 0.0, 1.0)
        if (np.hypot(*(a + t[:, None] * e).T) - motif.half_width * rule.size(k) < view / 2).any():
            return True
    return cover(universe, motif) is not None


def cover(universe, motif):
    """(color class, level) of the finest level bigger than GIANT with a bar over O, or None (the floor's tint)"""
    rule = universe.rule
    top, k = universe.top()
    current, found = list(rule.module.tops(top)), None
    while rule.size(k) > GIANT:
        for kind, r, m, tau in current:
            tx, ty = rule.ring.xy_float(tau)
            turn = rule.orient[r, int(m)]
            for p, q in motif.segments[kind]:
                a, b = turn @ p + (tx, ty), turn @ q + (tx, ty)
                e = b - a
                t = min(1.0, max(0.0, float(-(a @ e) / (e @ e))))
                if math.hypot(*(a + t * e)) < motif.half_width:
                    found = rule.module.color_class(kind, r, m), k
        # The tiles with O in them (inside() counts from PAD in, for a tile alone; any tile that has O has it inside)
        current = [c for t in current for _s, c in children_near(rule, t, 0.0)
                   if rule.module.inside(c) + rule.pad > -1e-9]
        if rule.straddle:
            current = list(dict.fromkeys(current))
        k -= 1
    return found


# --- Giant levels, drawn exactly ----------------------------------------------------------------------------------------

def giant_from(rule):
    """The finest giant level: past GIANT, a level's bars near the drone are placed from its exact tiles
    (giant_levels()), not from floats, which are only good to 1e-16 of its size"""
    return math.floor(math.log(GIANT) / rule.log_loop) + 1


def bits_of(rule, k):
    """Fixed-point bits for level k's places in its own units: 2^-64 of a level-0 tile"""
    return 64 + max(0, math.ceil(k * rule.log_loop / math.log(2.0)))


def fixed(x, bits):
    """A float times 2^bits as a whole number (rounded down)"""
    m, e = math.frexp(x)
    mantissa, shift = int(m * 2.0 ** 53), bits + e - 53
    return mantissa << shift if shift >= 0 else mantissa >> -shift


_turns = {}


def turn_fixed(rule, r, bits):
    """cos and sin of r turns of 360 / N degrees, times 2^bits, from zeta^r exactly"""
    key = rule.name, r, bits
    if key not in _turns:
        if len(_turns) > 1024:
            _turns.clear()
        _turns[key] = rule.ring.xy_fixed(rule.ring.power(r), bits)
    return _turns[key]


def giant_bars(rule, motif, tile, k, center, window):
    """A level-k tile's bar polygons (the motif's, cut to it already) within the square of half side window around
    center (level-0 units from O), cut to that square: ([corners (n, 2) from center, level-0 units], [color classes,
    OUTLINE for a strip]). Exact however big the level: where center is in the tile's own prototile units is worked
    out in fixed point (bits_of()) from the tile's exact place, each polygon is cut to the square there, in the tile's
    axes, and only the cut corners, which are small, become floats, turned by the tile's turn and scaled by the
    level's size (each good to a share of itself). In floats, a bar's edge near the drone was off by 1e-16 of the
    level's size: past GIANT, more than the drone. Cut in floats, a long bar's sides through the square, from corners
    1e18 away, were off by a hundred (2026-11-15: a bar over the whole view took it for a bar's edge in it, and back)."""
    R = rule.ring
    kind, r, m, tau = tile
    bits = bits_of(rule, k)
    one = 1 << bits
    tx, ty = R.xy_fixed(tau, bits)
    inverse, _ = R.xy_fixed(rule.power(-k), bits)          # LAMBDA^-k, a real number
    vx = ((fixed(center[0], bits) * inverse) >> bits) - tx
    vy = ((fixed(center[1], bits) * inverse) >> bits) - ty
    cos, sin = turn_fixed(rule, r, bits)
    # Into the tile's axes, turned back but not mirrored (a mirrored tile's polygons are mirrored into them instead:
    # orient_matrix() is the turn after the mirror), so polygons and square both go counterclockwise
    lx, ly = (cos * vx + sin * vy) >> bits, (-sin * vx + cos * vy) >> bits
    mirror = -1 if m else 1
    size = rule.size(k)
    # Only the polygons whose bounding boxes come near center (generously: a float of the place is good to 1e-16)
    here = (lx / one, mirror * ly / one)
    reach = window * math.sqrt(2.0) / size + 1e-9
    lo, hi = motif.boxes[kind]
    near = np.flatnonzero((lo[:, 0] <= here[0] + reach) & (hi[:, 0] >= here[0] - reach)
                          & (lo[:, 1] <= here[1] + reach) & (hi[:, 1] >= here[1] - reach))
    if not len(near):
        return [], []
    polygons, counts = (motif.bars_reversed if m else motif.bars)[kind]
    turn = rule.orient[r, 0] * size
    tile_class = int(rule.color_class[kind, r, int(m)])
    # The square round center in the tile's axes, turned back like the rest
    w = fixed(window / size, bits)
    square = [((cos * x + sin * y) >> bits, (-sin * x + cos * y) >> bits)
              for x, y in ((-w, -w), (w, -w), (w, w), (-w, w))]
    out, classes = [], []
    for i in near:
        corners = [(fixed(float(p[0]), bits) - lx, fixed(mirror * float(p[1]), bits) - ly)
                   for p in polygons[i, :counts[i]]]
        for a in range(4):
            if len(corners) < 3:
                break
            corners = clip_to_left_whole(corners, square[a], square[(a + 1) % 4])
        if len(corners) >= 3:
            corners = [turn @ np.array((x / one, y / one)) for x, y in corners]
            if polygon_area(corners) > 0.0:
                out.append(np.array(corners))
                classes.append(OUTLINE if motif.edges[kind][i] else tile_class)
    return out, classes


def clip_to_left_whole(polygon, a, b):
    """clip_to_left() in whole numbers (fixed point), exactly but for each new corner's rounding"""
    ex, ey = b[0] - a[0], b[1] - a[1]
    side = lambda p: ex * (p[1] - a[1]) - ey * (p[0] - a[0])
    out = []
    for i, p in enumerate(polygon):
        q = polygon[(i + 1) % len(polygon)]
        sp, sq = side(p), side(q)
        if sp >= 0:
            out.append(p)
        if (sp >= 0) != (sq >= 0):
            out.append((p[0] + (q[0] - p[0]) * sp // (sp - sq), p[1] + (q[1] - p[1]) * sp // (sp - sq)))
    return out


class Giant:
    """giant_levels()'s answer for a universe (stamp) and a window (center, half side window, level-0 units from O):
    drawn {level: (polygons from center, classes)} and tint (the floor's color class, or None for its own), and serial,
    which level objects built from it carry ("sub_giant")"""

    serial = random.randrange(1, 2 ** 30)       # not one a level object saved in a file may still carry

    def __init__(self, stamp, center, window, drawn, tint):
        Giant.serial += 1
        self.serial = Giant.serial
        self.stamp, self.center, self.window, self.drawn, self.tint = stamp, tuple(center), window, drawn, tint


def giant_levels(universe, motif, center, window):
    """The giant levels (giant_from() up) seen in the square of half side window around center (level-0 units from
    O): (drawn, tint). From the top down, each level's tiles near the square (exact, as cover() walks them), and their
    bars in it (giant_bars()). A level with a polygon over the whole square hides everything coarser: it's the floor's
    tint (tint, its color class) and the finer levels with bars in the square are drawn (drawn, {level: (polygons,
    classes)}). So a giant level's edge in view is drawn where it is, and a dive at it shows it for good: the floor,
    tinted all or nothing by the bar over O, turned one side of it the other color in a frame once its level passed
    GIANT (2026-11-06), which fading levels out before then hid until 2026-11-15."""
    rule = universe.rule
    top, k = universe.top()
    first = giant_from(rule)
    reach = max(abs(center[0]), abs(center[1])) + window
    h = lambda k: reach / rule.size(k) + rule.pad
    current = [t for t in rule.module.tops(top) if meets_square(rule, t, h(k))]
    drawn, tint = {}, None
    full = (2.0 * window) ** 2
    while k >= first and current:
        polygons, classes = [], []
        for tile in current:
            more, their = giant_bars(rule, motif, tile, k, center, window)
            polygons += more
            classes += their
        whole = [c for p, c in zip(polygons, classes) if polygon_area(p) >= GIANT_COVER * full]
        if whole:
            # Polygons of a level only overlap others of their color (outlined()): the square is all this color
            drawn, tint = {}, whole[0]
        elif polygons:
            drawn[k] = (polygons, classes)
        k -= 1
        if k >= first:
            current = cut(rule, current, h(k))
    return drawn, tint


def giant_faces(polygons, classes, window, tile):
    """A giant level's polygons (from its window's middle, level-0 units) as level_mesh() gives faces, split into
    tiles (tile level-0 units, out to the window) for depth precision"""
    batches, batch_colors = [], []
    counts = np.array([len(p) for p in polygons])
    classes = np.array(classes, float)
    for k in np.unique(counts):
        rows = np.flatnonzero(counts == k)
        batches.append(np.array([polygons[i] for i in rows]))
        batch_colors.append(classes[rows][:, None].repeat(4, axis=1))
    corners, sizes, colors = tiling.tiles(batches, batch_colors, tile, 2.0 * window, adaptive=True)
    return corners, sizes, colors[:, 0].astype(np.int64)


# --- Levels as arrays -------------------------------------------------------------------------------------------------

class FloatLevel:
    """A level's tiles as arrays, placed from roots' accurate world places: world = base + LAMBDA^level rho"""

    def __init__(self, rule, level, kind, rot, mirror, rho, base):
        self.rule, self.level = rule, level
        self.kind, self.rot, self.mirror, self.rho, self.base = kind, rot, mirror, rho, base

    def __len__(self):
        return len(self.kind)

    def extremes(self):
        """The biggest |rho| and |base| coefficients, worked out once (staleness() asks every tick)"""
        if not hasattr(self, "_extremes"):
            self._extremes = (float(np.abs(self.rho).max()), float(np.abs(self.base).max()))
        return self._extremes

    def children(self):
        """The next level down: every tile's children, a slot at a time for all tiles of a kind at once, and for a
        rule whose children straddle (darb_imam), each found once: the same kind, turn and whole-number place from the
        same root place"""
        rule = self.rule
        parts = []
        for kind in rule.kinds:
            index = np.nonzero(self.kind == kind)[0]
            if not len(index):
                continue
            table = rule.child[kind]
            r, m = self.rot[index].astype(np.int64), self.mirror[index].astype(np.int64)
            slots = len(table.kind)
            # (slots, tiles): slot by slot, as each slot's children of every tile in turn
            rho = (self.rho[index] @ rule.lambda_matrix.T)[None] + table.place[:, r, m]
            parts.append((np.repeat(table.kind, len(index)), table.rot[:, r, m].ravel(), table.mirror[:, r, m].ravel(),
                          rho.reshape(-1, 4), np.broadcast_to(self.base[index][None], (slots, len(index), 2))
                          .reshape(-1, 2)))
        if not parts:
            return FloatLevel(rule, self.level - 1, *empty_level_arrays())
        cat = lambda i: np.concatenate([p[i] for p in parts])
        out = FloatLevel(rule, self.level - 1, cat(0).astype(np.int8), cat(1).astype(np.int8), cat(2).astype(bool),
                         cat(3).astype(np.int64).reshape(-1, 4), cat(4).reshape(-1, 2))
        if rule.straddle and len(out):
            keys = np.column_stack([out.kind.astype(np.int64), out.rot.astype(np.int64), out.mirror.astype(np.int64),
                                    out.rho, out.base.view(np.int64)])
            _unique, first = np.unique(keys, axis=0, return_index=True)
            if len(first) < len(out):
                out = out.select(np.sort(first))
        return out

    def select(self, keep):
        return FloatLevel(self.rule, self.level, self.kind[keep], self.rot[keep], self.mirror[keep], self.rho[keep],
                          self.base[keep])

    def places(self):
        rho = np.stack([self.rho @ self.rule.basis_x, self.rho @ self.rule.basis_y], axis=1)
        return self.base + self.rule.size(self.level) * rho

    def turns(self):
        return self.rule.orient[self.rot, self.mirror.astype(int)]

    def corners(self):
        """(n, corners, 2) world corners, counterclockwise (padded with copies of a tile's last one for kinds with
        fewer)"""
        rule = self.rule
        width = next(iter(rule.corners.values())).shape[0]
        out = np.zeros((len(self), width, 2))
        places, turns, size = self.places(), self.turns(), rule.size(self.level)
        for kind in rule.kinds:
            index = np.nonzero(self.kind == kind)[0]
            out[index] = places[index, None, :] + size * np.einsum("nij,kj->nki", turns[index], rule.corners[kind])
        mirrored = self.mirror
        out[mirrored] = out[mirrored][:, ::-1]
        return out

    def bars(self, motif):
        """Every tile's bars as (ends (n, 2, 2) world, owners (n,) indices of their tiles)"""
        ends, owners = [], []
        places, turns, size = self.places(), self.turns(), self.rule.size(self.level)
        for kind in self.rule.kinds:
            index = np.nonzero(self.kind == kind)[0]
            segs = motif.segments[kind]
            if not len(index) or not len(segs):
                continue
            world = places[index, None, None, :] + size * np.einsum("nij,skj->nski", turns[index], segs)
            ends.append(world.reshape(-1, 2, 2))
            owners.append(np.repeat(index, len(segs)))
        if not ends:
            return np.zeros((0, 2, 2)), np.zeros(0, dtype=np.int64)
        return np.concatenate(ends), np.concatenate(owners)

    def colors(self):
        return self.rule.color_class[self.kind.astype(int), self.rot.astype(int), self.mirror.astype(int)]

    def bar_polygons(self, motif):
        """Every tile's bar polygons (Motif.bars, cut to the tile already), placed in the level's own tile units around
        O, counterclockwise: (polygons (n, corners, 2), counts (n,), color classes (n,))"""
        size = self.rule.size(self.level)
        places, turns, classes = self.places() / size, self.turns(), self.colors()
        parts = []
        for kind in self.rule.kinds:
            for mirrored, bars in ((False, motif.bars[kind]), (True, motif.bars_reversed[kind])):
                index = np.nonzero((self.kind == kind) & (self.mirror == mirrored))[0]
                polygons, counts = bars
                if not len(index) or not len(polygons):
                    continue
                # (n, bars * corners, 2): each tile's turn applied to every corner at once (a matmul, faster than
                # einsum), then placed
                flat = polygons.reshape(-1, 2)
                placed = places[index, None, :] + np.matmul(flat[None], turns[index].transpose(0, 2, 1))
                parts.append((placed.reshape(-1, polygons.shape[1], 2), np.tile(counts, len(index)),
                              np.where(np.tile(motif.edges[kind], len(index)), OUTLINE,
                                       np.repeat(classes[index], len(polygons)))))
        if not parts:
            return np.zeros((0, 3, 2)), np.zeros(0, np.int64), np.zeros(0, np.int64)
        width = max(p[0].shape[1] for p in parts)
        polygons = np.concatenate([np.pad(p[0], ((0, 0), (0, width - p[0].shape[1]), (0, 0))) for p in parts])
        return polygons, np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts])


def empty_level_arrays():
    return (np.zeros(0, np.int8), np.zeros(0, np.int8), np.zeros(0, bool), np.zeros((0, 4), np.int64),
            np.zeros((0, 2)))


def square_keep(corners, w, center):
    margin = 1e-9 * max(1.0, w)
    x, y = corners[..., 0] - center[0], corners[..., 1] - center[1]
    return ((x.max(axis=1) >= -w - margin) & (x.min(axis=1) <= w + margin)
            & (y.max(axis=1) >= -w - margin) & (y.min(axis=1) <= w + margin))


def near_square(level, center, half):
    """Which of a FloatLevel's tiles might meet the square of half side half around center (level-0 units): those
    whose place is within the rule's REACH of a tile of it"""
    reach = half + level.rule.reach * level.rule.size(level.level)
    return (np.abs(level.places() - np.array(center)) <= reach).all(axis=1)


def roots(rule, tiles, level):
    """Exact tiles as a FloatLevel of roots: each one's world place, LAMBDA^level tau, as one accurate float. For a
    rule whose children straddle, one place for all, the first's, with the others' whole-number offsets from it as
    their rho (a level's tiles are all on one lattice in its units), so a child two parents have comes out the same."""
    R = rule.ring
    scale = rule.power(level)
    kinds = np.array([t[0] for t in tiles], np.int8)
    rots = np.array([t[1] for t in tiles], np.int8)
    mirrors = np.array([t[2] for t in tiles], bool)
    if rule.straddle and tiles:
        first = tiles[0][3]
        rho = np.array([R.tidy(R.sub(t[3], first)) for t in tiles], dtype=np.int64).reshape(-1, 4)
        base = np.tile(np.array(R.xy_float(R.mul(first, scale))), (len(tiles), 1))
        return FloatLevel(rule, level, kinds, rots, mirrors, rho, base)
    base = np.array([R.xy_float(R.mul(t[3], scale)) for t in tiles]).reshape(-1, 2)
    return FloatLevel(rule, level, kinds, rots, mirrors, np.zeros((len(tiles), 4), np.int64), base)


def levels(universe, finest, window, center=(0.0, 0.0)):
    """{level: FloatLevel} from the top down to finest, each level's tiles meeting the square of half side window(k)
    around center (level-0 units), up to GIANT: exact from the top while tiles are ROOT_SIZE or bigger, then floats.
    Each level is found out to the rule's PAD further, so the levels below it are complete in their windows."""
    rule = universe.rule
    top, k = universe.top()
    local = lambda k: (center[0] / rule.size(k), center[1] / rule.size(k))
    h = lambda k: window(k) / rule.size(k) + rule.pad
    current = [t for t in rule.module.tops(top) if meets_square(rule, t, h(k), local(k))]
    exact = {k: current}
    while k > finest and rule.size(k - 1) >= ROOT_SIZE:
        k -= 1
        current = cut(rule, current, h(k), local(k))
        exact[k] = current
    out = {level: roots(rule, tiles, level) for level, tiles in exact.items()
           if level >= finest and rule.size(level) <= GIANT}
    if k > finest and current:
        level = out.get(k) or roots(rule, current, k)
        while level.level > finest and len(level):
            below = window(level.level - 1) + rule.pad * rule.size(level.level - 1)
            level = level.select(near_square(level, center, below + rule.child_reach * rule.size(level.level - 1)))
            level = level.children()
            level = level.select(square_keep(level.corners(), below, center))
            out[level.level] = level
    return out


# --- Meshes -------------------------------------------------------------------------------------------------------------

def mixed_polygons(level, motif, cores=None, near=None):
    """A level's bar polygons (FloatLevel.bar_polygons()): outlined (motif) for its tiles near the camera, near = (x,
    y, r) (level-0 units: their places within r of (x, y), and the rule's REACH of a tile further), and their cores
    alone (cores) for the rest, where the outline strips are too thin to see; all of them as motif without cores"""
    if cores is None or near is None:
        return level.bar_polygons(motif)
    size = level.rule.size(level.level)
    close = np.hypot(*(level.places() - np.array(near[:2])).T) <= near[2] + level.rule.reach * size
    parts = [level.select(close).bar_polygons(motif), level.select(~close).bar_polygons(cores)]
    width = max(p[0].shape[1] for p in parts)
    return (np.concatenate([np.pad(p[0], ((0, 0), (0, width - p[0].shape[1]), (0, 0))) for p in parts]),
            np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts]))


def level_mesh(level, motif, center, window, tile, cores=None, near=None):
    """A level's bars as faces in its own tile units around O: (corners (n, 2), sizes, color classes). The bar polygons
    (cut to their own tiles already, Motif.bars; outlined near the camera, mixed_polygons()) placed, cut to the level's
    window around center (only those across its edge), and, for levels big enough that a bar can be longer than tile
    (level units), split into tiles (depth precision, tiling.tiles())."""
    size = level.rule.size(level.level)
    polygons, counts, classes = mixed_polygons(level, motif, cores, near)
    if not len(polygons):
        return np.zeros((0, 2)), np.zeros(0, np.int64), np.zeros(0, np.int64)
    cx, cy, h = center[0] / size, center[1] / size, window / size
    valid = np.arange(polygons.shape[1]) < counts[:, None]
    xs = np.where(valid, polygons[..., 0], np.nan)
    ys = np.where(valid, polygons[..., 1], np.nan)
    lo_x, hi_x = np.nanmin(xs, axis=1) - cx, np.nanmax(xs, axis=1) - cx
    lo_y, hi_y = np.nanmin(ys, axis=1) - cy, np.nanmax(ys, axis=1) - cy
    outside = (hi_x < -h) | (lo_x > h) | (hi_y < -h) | (lo_y > h)
    across = ~outside & ((lo_x < -h) | (hi_x > h) | (lo_y < -h) | (hi_y > h))
    keep = ~outside & ~across
    parts = [(polygons[keep], counts[keep], classes[keep])]
    if across.any():
        cut_polygons, cut_counts = polygons[across], counts[across]
        for normal, offset in (((1.0, 0.0), cx + h), ((-1.0, 0.0), -(cx - h)), ((0.0, 1.0), cy + h),
                               ((0.0, -1.0), -(cy - h))):
            normals = np.broadcast_to(np.array(normal), (len(cut_polygons), 2))
            cut_polygons, cut_counts = clip_half_planes(cut_polygons, cut_counts, normals,
                                                        np.full(len(cut_polygons), offset))
        good = cut_counts >= 3
        parts.append((cut_polygons[good], cut_counts[good], classes[across][good]))
    batches, batch_colors = [], []
    for polys, cnts, cls in parts:
        for k in np.unique(cnts):
            rows = cnts == k
            batches.append(polys[rows, :k])
            batch_colors.append(cls[rows].astype(float)[:, None].repeat(4, axis=1))
    if tile < 2.0:
        # Split on lines around where the drone is, not around O: a coarse level is only rebuilt every so often, and
        # by then O can be thousands of km away, and the pieces near the drone thousands of km across (tiling.tiles()
        # doubles them out from the middle), too big for float32 depths: what was drawn changed with rounding, so a
        # loop's exact scaling changed the view (2026-11-05). Only bars longer than their distance from there are cut
        # (adaptive), the few near the drone: cut on every line across the window, 2k of 11k bars were, and the split
        # took four times as long as the rest of the level's build.
        shift = np.array([cx, cy])
        corners, sizes, colors = tiling.tiles([b - shift for b in batches], batch_colors, tile, 2.0 * h, adaptive=True)
        return corners + shift, sizes, colors[:, 0].astype(np.int64)
    corners = np.concatenate([b.reshape(-1, 2) for b in batches]) if batches else np.zeros((0, 2))
    sizes = np.concatenate([np.full(len(b), b.shape[1], np.int64) for b in batches]) if batches else np.zeros(0, np.int64)
    colors = np.concatenate([c[:, 0] for c in batch_colors]).astype(np.int64) if batches else np.zeros(0, np.int64)
    return corners, sizes, colors


def chunked(rule, level, tile):
    """Whether a level is built in kept chunks: unless it's split into tiles (bars longer than tile, level-0 units)"""
    return tile / rule.size(level) >= 2.0


class Chunks:
    """One level object's bars, cut into size-square chunks of its own tile units around O (the map's "chunk", CHUNK
    if none), kept between its builds: {(i, j, outlined): (corners, sizes, color classes)} for the chunk from (i, j)
    size to (i + 1, j + 1) size, outlined or its bars' cores alone (chunked_mesh())"""

    def __init__(self, level, size=None, motif=None):
        self.level, self.size, self.motif, self.parts = level, size or CHUNK, motif, {}

    def copy(self):
        """Its own table of the same chunks (a chunk's arrays are never changed, only kept or dropped)"""
        other = Chunks(self.level, self.size, self.motif)
        other.parts = dict(self.parts)
        return other


def chunked_mesh(chunks, level, motif, center, window, cores=None, near=None):
    """A level's bars over the chunks that cover the square of half side window around center (level-0 units), as
    level_mesh() gives them, with the chunks kept in chunks: only those not kept yet are worked out. Returns the faces
    and the half side of the square round center they cover (level-0 units).

    Flying, a level is rebuilt each time the drone has moved its margin (3 of its tiles) from where it was built, and
    every build cut all of its ~3,500 halves' bars to the window again: 3 of a build's 5 ms, with the mesh written in
    the other 2. Flying backward over the floor, the fine levels came due dozens of times a second, more than the time
    budget allowed, and they fell behind ("when climbing backwards, tiles stop generating after awhile", 2026-11-06).
    A new build now only cuts the chunks that came into range (a row or column of them) and joins the rest."""
    rule = level.rule
    size = rule.size(level.level)
    chunk = chunks.size
    cx, cy, h = center[0] / size, center[1] / size, window / size
    i0, i1 = math.floor((cx - h) / chunk), math.floor((cx + h) / chunk)
    j0, j1 = math.floor((cy - h) / chunk), math.floor((cy + h) / chunk)
    # Near the camera (near = (x, y, r), level-0 units: a chunk's square within r of (x, y)) a chunk is drawn outlined,
    # further as its bars' cores alone (mixed_polygons()), and kept as (i, j, outlined)
    if cores is not None and near is not None:
        nx, ny, nr = near[0] / size, near[1] / size, near[2] / size

        def close(i, j):
            return math.hypot(max(0.0, abs(nx - (i + 0.5) * chunk) - chunk / 2.0),
                              max(0.0, abs(ny - (j + 0.5) * chunk) - chunk / 2.0)) <= nr
    else:
        close = lambda i, j: False
    wanted = [(i, j, close(i, j)) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]
    missing = [key for key in wanted if key not in chunks.parts]

    def fill(group, motif, outlined):
        mi = np.array([key[0] for key in group])
        mj = np.array([key[1] for key in group])
        # Only the tiles near the missing chunks: a tile lies within the rule's REACH of its place, so it can meet the
        # chunks from its place less that to its place plus that, at most 2 x 2 of them (twice reach is under a chunk)
        places = level.places() / size
        reach = rule.reach + 0.5
        assert 2.0 * reach < chunk, "chunks too small for the rule's tiles"
        grid = np.zeros((i1 - i0 + 3, j1 - j0 + 3), bool)          # a chunk's margin all round, never missing
        grid[mi - i0 + 1, mj - j0 + 1] = True
        cell_of = lambda v, lo, top: np.clip(np.floor(v / chunk).astype(np.int64) - lo + 1, 0, top)
        ii = [cell_of(places[:, 0] + d, i0, i1 - i0 + 2) for d in (-reach, reach)]
        jj = [cell_of(places[:, 1] + d, j0, j1 - j0 + 2) for d in (-reach, reach)]
        near = grid[ii[0], jj[0]] | grid[ii[0], jj[1]] | grid[ii[1], jj[0]] | grid[ii[1], jj[1]]
        polygons, counts, classes = level.select(near).bar_polygons(motif)
        found = {key: None for key in group}
        if len(polygons):
            valid = np.arange(polygons.shape[1]) < counts[:, None]
            xs = np.where(valid, polygons[..., 0], np.nan)
            ys = np.where(valid, polygons[..., 1], np.nan)
            lo_i, hi_i = np.floor(np.nanmin(xs, axis=1) / chunk), np.floor(np.nanmax(xs, axis=1) / chunk)
            lo_j, hi_j = np.floor(np.nanmin(ys, axis=1) / chunk), np.floor(np.nanmax(ys, axis=1) / chunk)
            # Each bar with every missing chunk its bounds meet (a bar is far shorter than a chunk: at most 2 x 2)
            rows, ci, cj = [], [], []
            for di in (0, 1):
                for dj in (0, 1):
                    i, j = lo_i + di, lo_j + dj
                    ok = (i <= hi_i) & (j <= hi_j)
                    index = np.nonzero(ok)[0]
                    rows.append(index)
                    ci.append(i[index].astype(np.int64))
                    cj.append(j[index].astype(np.int64))
            rows, ci, cj = np.concatenate(rows), np.concatenate(ci), np.concatenate(cj)
            code = lambda a, b: a * (1 << 32) + b
            keep = np.isin(code(ci, cj), code(mi, mj))
            rows, ci, cj = rows[keep], ci[keep], cj[keep]
            # Bars inside one chunk are kept whole; the rest are cut to their chunk
            whole = (lo_i[rows] == hi_i[rows]) & (lo_j[rows] == hi_j[rows])
            parts = [(polygons[rows[whole]], counts[rows[whole]], classes[rows[whole]], ci[whole], cj[whole])]
            if (~whole).any():
                cut_polygons, cut_counts = polygons[rows[~whole]], counts[rows[~whole]]
                i, j = ci[~whole], cj[~whole]
                for normal, offset in (((1.0, 0.0), (i + 1) * chunk), ((-1.0, 0.0), -i * chunk),
                                       ((0.0, 1.0), (j + 1) * chunk), ((0.0, -1.0), -j * chunk)):
                    normals = np.broadcast_to(np.array(normal), (len(cut_polygons), 2))
                    cut_polygons, cut_counts = clip_half_planes(cut_polygons, cut_counts, normals, offset.astype(float))
                good = cut_counts >= 3
                parts.append((cut_polygons[good], cut_counts[good], classes[rows[~whole]][good], i[good], j[good]))
            # Every new chunk's faces at once: the pieces in one array, sorted by chunk and within one by corner count
            # (stably: whole bars before cut ones, each as found), each chunk kept as its slice. A chunk at a time
            # (np.pad and np.unique for each), a darb_imam level's 100 new chunks took 300 ms (2026-11-06).
            parts = [p for p in parts if len(p[0])]
            if parts:
                width = max(p[0].shape[1] for p in parts)
                polys = np.zeros((sum(len(p[0]) for p in parts), width, 2))
                at = 0
                for p in parts:
                    polys[at:at + len(p[0]), :p[0].shape[1]] = p[0]
                    at += len(p[0])
                cnts, cls, pi, pj = (np.concatenate([p[n] for p in parts]) for n in (1, 2, 3, 4))
                order = np.lexsort((cnts, pj, pi))
                polys, cnts, cls, pi, pj = polys[order], cnts[order], cls[order], pi[order], pj[order]
                corners = polys[np.arange(width) < cnts[:, None]]
                ends = np.concatenate([[0], np.cumsum(cnts)])
                starts = np.flatnonzero(np.r_[True, (pi[1:] != pi[:-1]) | (pj[1:] != pj[:-1])])
                for s, e in zip(starts, np.r_[starts[1:], len(pi)]):
                    found[int(pi[s]), int(pj[s]), outlined] = (corners[ends[s]:ends[e]], cnts[s:e].astype(np.int64),
                                                               cls[s:e].astype(np.int64))
        for key, faces in found.items():
            empty = (np.zeros((0, 2)), np.zeros(0, np.int64), np.zeros(0, np.int64))
            chunks.parts[key] = faces if faces is not None else empty

    for outlined in (False, True):
        group = [key for key in missing if key[2] == outlined]
        if group:
            fill(group, motif if outlined or cores is None else cores, outlined)
    # Chunks out of range go: the drone seldom comes back before they'd be found again anyway
    wanted_set = set(wanted)
    for key in [key for key in chunks.parts if key not in wanted_set]:
        del chunks.parts[key]
    faces = [chunks.parts[key] for key in wanted]
    corners = np.concatenate([f[0] for f in faces])
    sizes = np.concatenate([f[1] for f in faces])
    classes = np.concatenate([f[2] for f in faces])
    # The biggest square round center the chunks cover: at least window, and up to a chunk more
    covered = min(cx - i0 * chunk, (i1 + 1) * chunk - cx, cy - j0 * chunk, (j1 + 1) * chunk - cy) * size
    return corners, sizes, classes, covered


def clip_half_planes(points, counts, normals, offsets):
    """Convex polygons (n, corners, 2) each cut to normal . p <= offset (its own normal and offset), keeping order"""
    n, width = points.shape[:2]
    if n == 0:
        return points, counts
    k = np.arange(width)
    valid = k < counts[:, None]
    previous = tiling.take_corners(points, (k - 1) % np.maximum(counts, 1)[:, None])
    # Padding corners past a polygon's count hold anything (NaN, inf): computed and thrown away, quietly
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        f = (points * normals[:, None, :]).sum(axis=2) - offsets[:, None]
        g = (previous * normals[:, None, :]).sum(axis=2) - offsets[:, None]
        inside, was_inside = f <= 0, g <= 0
        t = g / (g - f)
        crossing = previous + t[..., None] * (points - previous)
    stacked = np.stack([crossing, points], axis=2).reshape(n, 2 * width, points.shape[2])
    flags = np.stack([valid & (inside != was_inside), valid & inside], axis=2).reshape(n, 2 * width)
    return tiling.compact_corners(stacked, flags)


# --- The map ----------------------------------------------------------------------------------------------------------

def is_substitution(obj):
    return obj is not None and hasattr(obj, "keys") and PROPERTY in obj.keys()


def encode(tile):
    """A tile as text: its place's coefficients in hex, a Fraction as numerator/denominator"""
    kind, r, m, tau = tile
    number = lambda c: hex(c) if type(c) is int else f"{hex(c.numerator)}/{hex(c.denominator)}"
    return f"{kind} {r} {int(m)} " + " ".join(number(c) for c in tau)


def decode(text):
    parts = text.split()

    def number(word):
        if "/" in word:
            top, bottom = word.split("/")
            return Fraction(int(top, 16), int(bottom, 16))
        return int(word, 16)
    return int(parts[0]), int(parts[1]), bool(int(parts[2])), tuple(number(c) for c in parts[3:7])


def load(obj):
    return Universe(rule_of(obj[PROPERTY]), decode(obj[MADE]), int(obj[MADE_LEVEL]))


def save(obj, universe):
    obj[MADE] = encode(universe.made)
    obj[MADE_LEVEL] = universe.made_level


_motifs = {}


def palette_of(settings):
    """The settings' palette, a color class's linear RGBA a row"""
    return np.array([list(c) for c in settings["palette"]], dtype=float)


# fade()'s steps: each one a level takes rewrites its colors (recolor()). A step is 3% of a bar's color, under what the
# flight harness counts as a change.
FADE_STEPS = 32


def fade(settings, px):
    """How much of its colors a level shows, 0 to 1, its bars px wide under the drone (px as settle() measures them:
    settings "focal" over the drone's height): settings "fade" = [in from, in to, out from, out to] px, rising from 0
    to 1 between the first two and falling back to 0 between the last two, smoothly in log px (a level's bars grow by
    the same factor every loop), in FADE_STEPS steps. 1 for a map without "fade". Just [in from, in to]: bars fade in
    from under a pixel (like anti-aliasing) and never out, so what's drawn stays put however big it grows."""
    band = settings.get("fade")
    if not band:
        return 1.0
    rise_from, rise_to, fall_from, fall_to = (float(b) for b in (list(band) + [math.inf, math.inf])[:4])

    def ramp(lo, hi):
        if hi <= lo:
            return 1.0 if px >= hi else 0.0
        t = min(1.0, max(0.0, math.log(max(px, 1e-300) / lo) / math.log(hi / lo)))
        return t * t * (3.0 - 2.0 * t)
    shown = (ramp(rise_from, rise_to) if rise_to > 0.0 else 1.0) * (1.0 - ramp(fall_from, fall_to))
    return round(shown * FADE_STEPS) / FADE_STEPS


def colors_of(settings, shown=1.0):
    """Each color class's linear RGBA, and last the outline strips' (OUTLINE, so -1 picks it), showing that much of
    them (fade()), the rest the color levels fade to (settings "fade_color", the floor's if none, so a faded bar is
    lost in the floor)"""
    floor = settings.get("floor_color", (0.0, 0.0, 0.0, 1.0))
    outline = settings.get("outline_color") or floor
    palette = np.vstack([palette_of(settings), np.array(list(outline), float)])
    if shown >= 1.0:
        return palette
    under = np.array(list(settings.get("fade_color", floor)), float)
    return under + (palette - under) * shown


def drawn_levels(settings, rule, motif, universe, height, drawn=()):
    """The levels to draw, (finest, coarsest): from the finest whose bars are min_bar_px wide under the drone (height
    over the floor, level-0 units; finer ones are fainter than that anywhere) up to `made`'s EXTRA default parents,
    short of GIANT, and for a map that fades (fade()), to one past the last level that shows, ready for climbing. A
    level already drawn (drawn) stays until the drone is a whole level past where it's wanted: bobbing about the
    threshold, the finest level was freed and built again and again, its faint lines blinking (2026-11-05: "it
    flickers every now and then")."""
    focal = float(settings.get("focal", 1000.0))
    bar = 2.0 * motif.half_width
    finest = math.ceil(math.log(float(settings.get("min_bar_px", 0.3)) * height / focal / bar) / rule.log_loop)
    coarsest = top = min(universe.made_level + rule.extra, math.floor(math.log(GIANT) / rule.log_loop))
    band = settings.get("fade")
    if band and len(band) >= 4:
        coarsest = min(top, math.floor(math.log(float(band[3]) * height / focal / bar) / rule.log_loop) + 1)
        if coarsest + 1 in drawn and coarsest + 1 <= top:
            coarsest += 1
    if finest - 1 in drawn:
        finest -= 1
    return finest, coarsest


def motif_of(settings, cores=False):
    """The map's motif, its bars outlined as its settings say ("outline", a share of a bar's width each side, 0 for
    none); or with cores, its bars' cores alone, as wide as an outlined bar's core (a level whose outline strips are
    too thin to see under the drone is drawn so, at a fraction of the faces: outlined())"""
    rule = rule_of(settings)
    half_width, outline = float(settings.get("half_width", 0.007)), float(settings.get("outline", 0.0))
    if cores:
        half_width, outline = half_width * (1.0 - 2.0 * outline), 0.0
    key = (rule.name, settings.get("style", "hankin54"), half_width, outline)
    if key not in _motifs:
        _motifs[key] = Motif(rule, *key[1:])
    return _motifs[key]


def pool(obj):
    return [entry["object"] for entry in obj[PROPERTY]["levels"] if entry.get("object") is not None]


# --- Each level's tiles, kept between ticks ------------------------------------------------------------------------

# A level's tiles are kept for a square of half side build_half(k) around where they were found, a little more than
# its drawn window: the next level down is cut from them while the drone stays within build_half(k + 1) - build_half(k)
# of there, which grows with the level (the LAMBDA^k term), so rebuilding a level seldom needs its parent found again.
# Before (2026-11-05) every rebuild found every level again from the top: 200 to 370 ms a tick flying fast.
BUILD_SLACK, BUILD_GROWTH = 1.25, 0.5
# A float level's places are whole-number coefficients (rho) from its exact roots, growing by LAMBDA a level, and a
# dive relabels the kept levels and cuts each new finest one from the kept level above: the chain from the roots grew
# a level every loop. Some 60 loops deep, rho was about 1e18, rho times the basis (in floats) was off by whole tiles,
# every half fell outside its window, and the finest levels drew nothing ("after enough loops downward, smaller shapes
# no longer render", 2026-11-05). Past this, a kept level is found again from exact roots instead. Its roots' places,
# base, grow by LAMBDA with every diving loop too, and they're checked the same way (level_tiles()).
RHO_LIMIT = 2 ** 40
REFRESH = 0.1               # a kept level this far to stale (staleness()) is found again from fresher roots (refresh())
# Level tiles a side of the squares a level's bars are kept in between builds (Chunks), unless the map's settings say
# ("chunk"): darb_imam's floor has windows of 40 tiles, and at 4, a level was 400 chunks, a full build of one 370 ms
CHUNK = 4.0
_caches = {}
_warned = {}
decisions = {}          # by map: how many levels decide() fixed each way, for testing how varied climbs are
_build_time = {}        # settle()'s budget of time for overdue levels: (seconds saved up, when), by map
_jobs = {}              # (map's name, level object's name) -> Job: its build under way on the worker thread
_in_place = set()       # maps whose builds failed on the worker thread: built in the tick from then on
JOBS = 2                # levels of a map built on the worker thread at a time (under way or waiting to be written):
                        # more would be built for where the drone was by the time their turn came
# With the builds' arithmetic on the worker thread, only finding a level's tiles and writing its mesh are left to the
# flight's own, a third of a build: a third of the time budget ("build_share") then rebuilds as many levels as the
# whole budget did (girih in a fast dive: 34 a second before, 2026-11-14), and the rest goes to the frames
WORKER_SHARE = 1.0 / 3.0


class Job:
    """A level's build under way on the worker thread (background.py), as the level object will be once it's
    written: its level, its window's middle (level-0 units), its unpushed place and scale, how much loops have grown
    the map since it was handed over, and the copy of its kept chunks it's filling (or None). loop() and recenter()
    keep these in step with the map, as they do the level objects' own."""

    def __init__(self, cache, level, center, scale, chunks, outlined=True, near=None):
        self.cache, self.level, self.center, self.chunks = cache, level, list(center), chunks
        self.bx, self.by, self.bs, self.grow = 0.0, 0.0, scale, 1.0
        self.outlined = outlined    # built with the outlined motif, or its cores alone
        self.near = near            # where it's outlined ("sub_near", mark_outlines()), or None
        self.faces = None           # level_faces()'s, once the worker has finished: waiting to be written


def jobs_of(obj):
    return [(key, job) for key, job in _jobs.items() if key[0] == obj.name]


def drop_job(key):
    background.cancel(key)
    _jobs.pop(key, None)


class LevelCache:
    """{level: (tiles, center, half side)}, level-0 units: exact tiles (lists) for levels of ROOT_SIZE and up, a
    FloatLevel below. Kept in step with the universe by loop() and recenter(); stamp is the universe it was built for,
    so anything else that changes it (the file reloaded, an undo) starts it over."""

    def __init__(self, rule, stamp):
        self.rule, self.stamp, self.levels = rule, stamp, {}
        self.chunks = {}        # {level object's name: Chunks}, its bars kept in chunks between builds

    def relabel(self, power):
        rule = self.rule
        grow = rule.size(power)
        moved = {}
        for k, (tiles, center, half) in self.levels.items():
            if isinstance(tiles, FloatLevel):
                if rule.size(k + power) >= ROOT_SIZE:
                    continue        # an exact level now, which is cut from exact tiles: found again when needed
                tiles = FloatLevel(rule, tiles.level + power, tiles.kind, tiles.rot, tiles.mirror, tiles.rho,
                                   tiles.base * grow)
            moved[k + power] = (tiles, (center[0] * grow, center[1] * grow), half * grow)
        self.levels = moved
        # A level object's bars are the same in its own tile units, just labeled power up
        for chunks in self.chunks.values():
            chunks.level += power

    def shift(self, v):
        rule = self.rule
        R = rule.ring
        vx, vy = R.xy_small(v)
        moved = {}
        for k, (tiles, center, half) in self.levels.items():
            if isinstance(tiles, FloatLevel):
                tiles = FloatLevel(rule, tiles.level, tiles.kind, tiles.rot, tiles.mirror, tiles.rho,
                                   tiles.base - np.array([vx, vy]))
            else:
                d = R.mul(v, rule.power(-k))
                tiles = [(t[0], t[1], t[2], R.tidy(R.sub(t[3], d))) for t in tiles]
            moved[k] = (tiles, (center[0] - vx, center[1] - vy), half)
        self.levels = moved
        # Kept chunks are around the old O, off the new one's chunk grid: found again as levels come due
        self.chunks = {}


def stamp_of(universe):
    return encode(universe.made), universe.made_level


def cache_of(obj, universe):
    cache = _caches.get(obj.name)
    if cache is None or cache.stamp != stamp_of(universe) or cache.rule is not universe.rule:
        cache = _caches[obj.name] = LevelCache(universe.rule, stamp_of(universe))
    return cache


def level_tiles(cache, universe, k, center, need, build_half, finest=None):
    """Level k's tiles meeting the square of half side need around center (level-0 units): the cached ones if they
    cover it, else found again for build_half(k) around center, cut from level k + 1's (found first if they don't
    cover that, out to the rule's CHILD_REACH of level k's tiles further: a child's parent is the one its middle is
    in), exactly from the top for levels of ROOT_SIZE and up"""
    rule = universe.rule
    entry = cache.levels.get(k)
    if entry is not None:
        tiles, (cx, cy), half = entry
        # A float level far from its exact roots isn't reused (RHO_LIMIT), nor one whose roots' places a dive has
        # grown too big: each loop scales them by LAMBDA, and a tile's place is base + LAMBDA^k rho, the two nearly
        # cancelling near the drone, so base's rounding has to stay well under a tile of the finest level that will
        # be cut from it (finest), not just its own. Checked against its own tiles alone, a coarse level passed while
        # the fine ones cut from it each loop were already off by a growing share of their tiles (1e-3 of a tile 40
        # loops into a dive, 0.4 at 50: metres off in flight, 2026-11-05).
        if staleness(tiles, k, finest) < 1.0 and max(abs(center[0] - cx), abs(center[1] - cy)) + need <= half:
            return tiles
    half = max(need, build_half(k))
    top, top_level = universe.top()
    size = rule.size(k)
    local = (center[0] / size, center[1] / size)
    if k > top_level:
        tiles = []
    elif k == top_level:
        tiles = [t for t in rule.module.tops(top) if meets_square(rule, t, half / size, local)]
    else:
        parent = level_tiles(cache, universe, k + 1, center, half + rule.child_reach * size, build_half,
                             min(size, finest if finest is not None else math.inf))
        if size >= ROOT_SIZE:
            tiles = cut(rule, parent, half / size, local)
        else:
            if not isinstance(parent, FloatLevel):
                parent = roots(rule, parent, k + 1)
            # Only the parents that can have a child in the square: a darb_imam level's window is some 5 of its
            # parents' tiles across, and they're kept for their own, 8.47 times as wide, so cutting them all made a
            # hundred times the children needed (2 s for a view's levels, 2026-11-06)
            children = parent.select(near_square(parent, center, half + rule.child_reach * size)).children()
            # Generously: a tile lies within the rule's REACH of its place
            near = (np.abs(children.places() - np.array(center)) <= half + rule.reach * size).all(axis=1)
            tiles = children.select(near)
    cache.levels[k] = (tiles, center, half)
    return tiles


def staleness(tiles, k, finest=None):
    """How far a kept level is from its exact roots, 1 or more being too far to reuse (level_tiles()): the larger of
    its rho over RHO_LIMIT and its roots' places' rounding over 1e-5 of the finest level cut from it (finest, its size).
    0 for exact levels."""
    if not isinstance(tiles, FloatLevel) or not len(tiles):
        return 0.0
    smallest = min(tiles.rule.size(k), finest if finest is not None else math.inf)
    rho, base = tiles.extremes()
    return max(rho / RHO_LIMIT, base * 2.0 ** -52 / (1e-5 * smallest))


def refresh(cache, universe, center, finest, build_half, need):
    """Find one kept float level again from its parent, ahead of its going stale: the coarsest a tenth of the way
    there (REFRESH) whose parent is fresher by half, for need(k) around center(k) (where its window is, or one
    point for all, the drone's place over the floor). Its children share
    its roots' places, so fresh roots go down the chain a level a tick. Without this, a level found stale was found
    again with every level above it that shared its roots, down from the exact ones, in one tick: some 25 levels, 40 to
    80 ms, every 20 loops or so of a dive (2026-11-05). Not for the square it was kept for: loops scale that with it,
    and a level kept for a few loops asked its exact parents for a square ten times what anything needs, which they
    then kept. Returns whether it found one."""
    kept = {k: entry for k, entry in cache.levels.items() if isinstance(entry[0], FloatLevel) and len(entry[0])}
    for k in sorted(kept, reverse=True):
        own = staleness(kept[k][0], k, finest)
        if own < REFRESH:
            continue
        parent = cache.levels.get(k + 1)
        if parent is not None and staleness(parent[0], k + 1, finest) > 0.5 * own:
            continue
        del cache.levels[k]
        level_tiles(cache, universe, k, center(k) if callable(center) else center, need(k), build_half, finest)
        return True
    return False


def as_float_level(rule, tiles, k):
    if isinstance(tiles, FloatLevel):
        return tiles
    return roots(rule, tiles, k) if tiles else None


def loop(obj, factor):
    """A loop by factor (a whole power of LAMBDA) about the map's origin: relabel the universe, the kept levels and the
    level objects, which are the same bars, LAMBDA^power bigger or smaller. Their custom properties change, so they're
    tagged for their drivers, as transforms only: a plain update_tag() re-evaluated every level's mesh too, and every
    loop re-uploaded all 250k faces to the GPU (frames of 50 to 60 ms at loops, 2026-11-05)."""
    universe = load(obj)
    rule = universe.rule
    power = round(math.log(factor) / rule.log_loop)
    cache = cache_of(obj, universe)
    universe.loop(power)
    save(obj, universe)
    cache.relabel(power)
    cache.stamp = stamp_of(universe)
    grow = rule.size(power)
    for _key, job in jobs_of(obj):
        job.level += power
        job.center = [c * grow for c in job.center]
        job.bx, job.by, job.bs, job.grow = job.bx * grow, job.by * grow, job.bs * grow, job.grow * grow
        if job.near is not None:
            job.near = [c * grow for c in job.near]
        if job.chunks is not None:
            job.chunks.level += power
    for level_obj in pool(obj):
        if level_obj.get("sub_level") is None:
            continue
        level_obj["sub_level"] = int(level_obj["sub_level"]) + power
        # A level not built yet has no center (settle() builds faint new levels in turn)
        for key in ("sub_center", "sub_near"):
            if key in level_obj:
                level_obj[key] = [c * grow for c in level_obj[key]]
        for key in ("sub_bs", "sub_bx", "sub_by", "camera_push_scale", "sub_window"):
            if key in level_obj:
                level_obj[key] = level_obj[key] * grow
        if int(level_obj["sub_level"]) >= giant_from(rule) and not level_obj.get("sub_giant") and "sub_bs" in level_obj:
            as_level_zero(level_obj, float(obj[PROPERTY]["cell"]))
        level_obj.update_tag(refresh={'OBJECT'})


def as_level_zero(level_obj, cell):
    """A level object a diving loop has made giant, its mesh rewritten in level-0 units, the object a cell of level 0,
    drawn the same: scaled by a level's size past ~7e12, an object can't be drawn (its float32 matrix's determinant
    overflows), and from this loop on until settle() builds it from the exact tiles at the end of the tick (or frees
    it), it would be missing, which the flight harness's renders between the two showed (2026-11-15). Marked to be
    built again ("sub_giant" -1, no giant_levels() serial)."""
    factor = float(level_obj["sub_bs"]) / cell
    mesh = level_obj.data
    if len(mesh.vertices):
        position = np.empty(3 * len(mesh.vertices), np.float64)
        mesh.attributes["position"].data.foreach_get("vector", position)
        mesh.attributes["position"].data.foreach_set("vector", (position * factor).astype(np.float32))
        mesh.update()
    level_obj["sub_bs"] = level_obj["camera_push_scale"] = cell
    level_obj["sub_giant"] = -1


def exact_factor(obj, factor):
    """The power of the rule's LAMBDA nearest factor, as the float the drone is scaled by (the floor relabels by the
    power). A factor that isn't near one (more than 1% off) can't loop the floor seamlessly: said once, and the nearest
    power taken anyway, rather than stopping the flight."""
    rule = rule_of(obj[PROPERTY])
    power = round(math.log(factor) / rule.log_loop)
    if power == 0:
        power = 1 if factor > 1.0 else -1
    if abs(math.log(factor) - power * rule.log_loop) > 0.01 and not _warned.get((obj.name, "factor")):
        print(f"substitution: {obj.name} loops by powers of {rule.loop:.6f}; the loop factor {factor} is taken as "
              f"{rule.loop:.6f}^{power}")
        _warned[obj.name, "factor"] = True
    return rule.size(power)


def recenter_vector(obj, camera_world):
    """The exact ring vector to move O by, if the drone camera is more than "step" level-0 tiles from it (map's
    axes), else None: the nearest point of the lattice of 1 and zeta^2 to where the camera is"""
    settings = obj[PROPERTY]
    R = rule_of(settings).ring
    local = obj.matrix_world.inverted() @ camera_world
    cell = float(settings["cell"])
    x, y = local.x / cell, local.y / cell
    if math.hypot(x, y) < float(settings.get("step", 64.0)):
        return None
    c72, s72 = math.cos(math.radians(72)), math.sin(math.radians(72))
    b = round(y / s72)
    a = round(x - b * c72)
    return R.add(R.scale(R.one, a), R.scale(R.power(2), b))


def recenter(obj, v):
    """Move O by v (level-0 units): the universe, the kept levels, and the level objects back by it. Returns the
    drone's move, map axes, meters (the caller moves the drone by minus that)."""
    universe = load(obj)
    cache = cache_of(obj, universe)
    universe.recenter(v)
    save(obj, universe)
    cache.shift(v)
    cache.stamp = stamp_of(universe)
    cell = float(obj[PROPERTY]["cell"])
    vx, vy = universe.rule.ring.xy_small(v)
    for _key, job in jobs_of(obj):
        job.center = [job.center[0] - vx, job.center[1] - vy]
        job.bx, job.by = job.bx - vx * cell, job.by - vy * cell
        job.chunks = None           # around the old O, like the kept ones (LevelCache.shift())
        if job.near is not None:
            job.near = [job.near[0] - vx, job.near[1] - vy, *job.near[2:]]
    for level_obj in pool(obj):
        if level_obj.get("sub_level") is None:
            continue
        for key in ("sub_center", "sub_near"):
            if key in level_obj:
                place = list(level_obj[key])
                level_obj[key] = [place[0] - vx, place[1] - vy, *place[2:]]
        level_obj["sub_bx"] = level_obj.get("sub_bx", 0.0) - vx * cell
        level_obj["sub_by"] = level_obj.get("sub_by", 0.0) - vy * cell
        level_obj.update_tag(refresh={'OBJECT'})
    return Vector((vx * cell, vy * cell, 0.0))


def view_half_angle(settings):
    """Half the drone camera's vertical field of view, radians: the add-on's rig's Drone_Camera with the scene's render
    shape (its sensor fit as Blender's AUTO does it), else "view_half_angle" (degrees, default 45: the rig's 10 mm lens
    on a 16:9 view)"""
    import bpy
    camera = bpy.data.objects.get("Drone_Camera")
    if camera is not None and camera.type == 'CAMERA' and camera.data.type == 'PERSP':
        data, render = camera.data, bpy.context.scene.render
        x, y = render.resolution_x * render.pixel_aspect_x, render.resolution_y * render.pixel_aspect_y
        fit = data.sensor_fit if data.sensor_fit != 'AUTO' else ('HORIZONTAL' if x >= y else 'VERTICAL')
        height = data.sensor_height if fit == 'VERTICAL' else data.sensor_width * y / x
        return math.atan(height / 2.0 / data.lens)
    return math.radians(float(settings.get("view_half_angle", 45.0)))


class View:
    """Where the drone camera is over the floor and which way it looks, for putting each level's window where it's
    seen: along the view's forward direction over the floor, from the nearest floor in view out as far as the level's
    bars are min_bar_px wide.

    Every level's window used to be centered right under the drone. Looking ahead over the floor, the floor in view
    starts some way ahead of that point, so the finest levels, smaller than that, were built out of sight, using most
    of the time budget, and the floor in view only had a strip of each fine level near its bottom edge, ending at its
    window's square edge ("when climbing backwards, tiles stop generating after awhile": the floor ends at an edge,
    the fine detail is missing, 2026-11-06)."""

    def __init__(self, obj, camera_matrix, settings):
        cell = float(settings["cell"])
        local = obj.matrix_world.inverted() @ camera_matrix
        place = local.translation
        self.cx, self.cy, self.height = place.x / cell, place.y / cell, max(abs(place.z) / cell, 1e-6)
        look = local.to_3x3() @ Vector((0.0, 0.0, -1.0))
        if place.z < 0.0:
            look.z = -look.z            # under the floor: the same seen from the other side
        across = math.hypot(look.x, look.y)
        below = math.atan2(-look.z, across)          # how far below the horizon the view's middle is
        half = view_half_angle(settings)
        self.forward = None
        if across > 1e-6 and below + half > 0.0:
            self.forward = (look.x / across, look.y / across)
            low, high = min(below + half, math.pi - 1e-6), below - half
            # How far ahead of the drone's place over the floor the floor in view starts (behind it, less than 0, if
            # the view takes in what's under the drone) and ends (the horizon, if the view's top edge is above it)
            self.near = self.height * math.cos(low) / math.sin(low)
            self.far = self.height * math.cos(high) / math.sin(high) if high > 1e-6 else math.inf

    def center(self, window, reach):
        """The middle (level-0 units) of a window of half side window for a level whose bars are seen out to reach
        over the floor (level-0 units from the drone's place over it): covering all of it in view if it fits, else its
        nearest part, where its bars are widest; under the drone if none of it is in view ahead"""
        if self.forward is None:
            return self.cx, self.cy
        a, b = self.near, min(self.far, reach)
        if b <= a:
            ahead = 0.0
        elif b - a <= 2.0 * window:
            ahead = (a + b) / 2.0
        else:
            ahead = a + window
        return self.cx + self.forward[0] * ahead, self.cy + self.forward[1] * ahead


def level_centers(rule, view, settings, motif, sight, radius):
    """Each level's window's middle, level-0 units, as a function of the level: placed by the view (View.center()),
    except for levels whose window is capped at the sight, which stay round the drone (they reach past anything in
    view, and placed by the view, every turn of the camera would move their middles by up to twice the sight)"""
    focal = float(settings.get("focal", 1000.0))
    min_px = float(settings.get("min_bar_px", 0.3))
    centers = {}

    def center(k):
        if k not in centers:
            size = rule.size(k)
            if radius * size >= sight:
                centers[k] = (view.cx, view.cy)
            else:
                seen = 2.0 * motif.half_width * size * focal / min_px          # bars min_px wide that far away
                reach = math.sqrt(max(seen * seen - view.height * view.height, 0.0))
                centers[k] = view.center(radius * size, reach)
        return centers[k]
    return center


def plan(obj, camera_matrix):
    """Whether settle() would change anything now (decide a level, draw another set of levels, or rebuild one), without
    changing anything: for the flight harness, which renders before and after every settle that does, at the same
    camera, since a rebuild or a decision must never change what's in view"""
    settings = obj[PROPERTY]
    rule = rule_of(settings)
    cell = float(settings["cell"])
    motif = motif_of(settings)
    sight = float(settings["sight"])
    view = View(obj, camera_matrix, settings)
    height = view.height
    universe = load(obj)
    if rule.size(universe.made_level) < AHEAD * sight:
        return True
    radius, margin = float(settings.get("radius", 20.0)), float(settings.get("margin", 4.0))
    center_of = level_centers(rule, view, settings, motif, sight, radius)
    window = lambda k: min(sight, radius * rule.size(k))
    by_level = {int(o["sub_level"]): o for o in pool(obj) if o.get("sub_level") is not None}
    finest, coarsest = drawn_levels(settings, rule, motif, universe, height, by_level)
    giant = giant_of(obj, universe, motif, (view.cx, view.cy), settings, peek=True)
    if giant is None or set(by_level) != set(range(finest, coarsest + 1)) | set(giant.drawn):
        return True
    if any(by_level[level].get("sub_giant") != giant.serial for level in giant.drawn):
        return True
    for level, level_obj in by_level.items():
        if level in giant.drawn:
            continue
        center, built = level_obj.get("sub_center"), level_obj.get("sub_window")
        if not level_obj.get("sub_built") or center is None or built is None:
            return True
        cx, cy = center_of(level)
        moved = max(abs(cx - center[0]), abs(cy - center[1]))
        if moved + window(level) > built:
            return True
        if (float(settings.get("tile", 100.0)) / (cell * rule.size(level)) < 2.0
                and moved * cell > float(settings.get("split_reach", 1000.0))):
            return True
    return False


def settle(obj, camera_matrix, force=False):
    """At the end of a tick: decide levels coming into reach, and rebuild the level objects whose window has moved
    "margin" of their tiles from where they were built (or all of them, force), each level's window where the view
    sees it (level_centers()), coarsest first, each from the kept levels; set every level object's push (finer on
    top) and the floor's tint"""
    settings = obj[PROPERTY]
    rule = rule_of(settings)
    cell = float(settings["cell"])
    motif = motif_of(settings)
    sight = float(settings["sight"])
    view = View(obj, camera_matrix, settings)
    height = view.height
    universe = load(obj)
    cache = cache_of(obj, universe)
    rng = random.Random(hash((int(settings.get("seed", 0)), universe.made_level, universe.made[3])))
    old_made_level = universe.made_level
    decided = universe.keep_ahead(rng, sight, motif)
    for reason in decided:
        counts = decisions.setdefault(obj.name, {})
        counts[reason] = counts.get(reason, 0) + 1
    if decided:
        save(obj, universe)
        # Decided levels above where `made` was may have changed: what's kept of them goes
        cache.levels = {k: entry for k, entry in cache.levels.items() if k <= old_made_level}
        cache.chunks = {name: chunks for name, chunks in cache.chunks.items() if chunks.level <= old_made_level}
        cache.stamp = stamp_of(universe)
        for key, job in jobs_of(obj):
            if job.level > old_made_level:
                drop_job(key)
    radius, margin = float(settings.get("radius", 20.0)), float(settings.get("margin", 4.0))
    focal = float(settings.get("focal", 1000.0))
    objects = pool(obj)
    by_level = {int(o["sub_level"]): o for o in objects if o.get("sub_level") is not None}
    finest, coarsest = drawn_levels(settings, rule, motif, universe, height, by_level)
    window = lambda k: min(sight, radius * rule.size(k))
    # How far the drone may move from where a level was built before it's due: margin of its tiles, but no more than a
    # tenth of the sight (for the levels whose window is capped at it). A tenth of every level's window, as until
    # 2026-11-05, rebuilt the fine levels every 1.4 of their tiles instead of every margin.
    tolerance = lambda k: min(margin * rule.size(k), 0.1 * sight)
    build_half = lambda k: BUILD_SLACK * window(k) + BUILD_GROWTH * rule.size(k)
    center_of = level_centers(rule, view, settings, motif, sight, radius)

    def build_window(k):
        """How far around the drone a level is built: its window and its tolerance, and for a level whose window is
        capped at the sight, LAMBDA times the sight, so a climbing loop (which shrinks it by LAMBDA) still leaves it
        covering the sight, and the 30 or so of them rebuild in turn under the time budget, not all at once. But no
        more than it needs one climb on, its window then, radius of its tiles (in today's units), unless that's under
        CAPPED_SLACK times the sight: with darb_imam's 8.47, LAMBDA times the sight put 670k faces into the finest
        capped level (2026-11-06)."""
        if radius * rule.size(k) < sight:
            return window(k) + tolerance(k)
        return min(rule.loop * sight, max(radius * rule.size(k), CAPPED_SLACK * sight)) + tolerance(k)
    # Nothing kept yet (the flight's first settle, or the file reloaded): every level's tiles found now, down to the
    # finest, rather than all at the first rebuild in flight, which took 80 ms a few seconds into a dive (2026-11-05)
    if not cache.levels and rule.size(finest) <= GIANT:
        level_tiles(cache, universe, finest, center_of(finest), build_window(finest), build_half)
    giant = giant_of(obj, universe, motif, (view.cx, view.cy), settings)
    wanted = list(range(finest, coarsest + 1)) + sorted(giant.drawn)
    # Free the objects whose levels aren't wanted any more, and give wanted levels without one a free one
    for level, level_obj in list(by_level.items()):
        if level not in wanted:
            del level_obj["sub_level"]      # custom properties can't be None
            del by_level[level]
            drop_job((obj.name, level_obj.name))
            clear(level_obj)
    free = [o for o in objects if o.get("sub_level") is None]
    missing = sum(1 for level in wanted if level not in by_level) - len(free)
    if missing > 0 and not _warned.get(obj.name):
        print(f"substitution: {obj.name} wants {len(wanted)} levels but has {len(objects)} level objects; the "
              f"coarsest {missing} aren't drawn")
        _warned[obj.name] = True
    for level in wanted:
        if level not in by_level and free:
            level_obj = free.pop()
            level_obj["sub_level"] = level
            level_obj["sub_built"] = False
            by_level[level] = level_obj
    # Which need rebuilding: never built, decided levels changed above `made` (all of these now), and those whose built
    # window no longer covers the level's window around the drone, the most overdue first, at most "builds" of them a
    # tick (each costs a few milliseconds, mostly writing its mesh). A level is built out to its window plus how far
    # the drone may move before it's due (tolerance()), and a loop scales that with it. For a level whose window is
    # capped at the sight, a climbing loop shrinks what was built below what it needs: the drone moving was the only
    # thing checked until 2026-11-05, so climbing, the big bars' squares (cut to the shrinking window) came into view
    # ("i can see the pink and orange squares after flying upward enough"), and diving, they grew back.
    bar_px = lambda k: 2.0 * motif.half_width * rule.size(k) * focal / height
    outline = float(settings.get("outline", 0.0))
    cores = motif_of(settings, cores=True) if outline > 0.0 else motif

    def outlined(level):
        """Whether a level is to be drawn outlined (OUTLINE_PX, with its hysteresis), or as its bars' cores alone"""
        if outline <= 0.0:
            return True
        now = by_level.get(level) is not None and by_level[level].get("sub_outlined", False)
        return outline * bar_px(level) >= (OUTLINE_PX / 2.0 if now else OUTLINE_PX)

    def outline_reach(level):
        """How far from the camera (level-0 units) a level's outline strips are OUTLINE_PX wide or more, square on"""
        return focal * outline * 2.0 * motif.half_width * rule.size(level) / OUTLINE_PX

    def needed(level):
        """How far round the camera's place over the floor (level-0 units) a level's tiles must be outlined"""
        reach = outline_reach(level)
        return math.sqrt(max(reach * reach - height * height, 0.0))

    def near_of(level):
        """Where an outlined level is drawn so (mixed_polygons()'s near): out to needed() and half its outlines' reach
        further, so it's built again only once the camera has moved that much ("sub_near"); None for all of it. Only
        the tiles near the drone are outlined, the rest drawn as their bars' cores alone: with every bar outlined, the
        Darb-i Imam floor had 480k faces at the start, four times its bars alone (2026-11-15)."""
        if outline <= 0.0:
            return None
        radius = needed(level) + 0.5 * outline_reach(level)
        if radius >= math.sqrt(2.0) * build_window(level):
            return None
        return (view.cx, view.cy, radius)
    rebuild, overdue = [], []
    for level, level_obj in by_level.items():
        if level in giant.drawn:
            continue                    # built from giant below
        if level_obj.get("sub_giant"):
            # A giant level a climbing loop has brought under GIANT: drawn as a level again, at once (it shows)
            level_obj["sub_giant"] = 0
            level_obj["sub_built"] = False
        built_center, built_window = level_obj.get("sub_center"), level_obj.get("sub_window")
        if force or not level_obj.get("sub_built") or built_center is None or built_window is None:
            # Never built: at once if its bars show (a level appearing a few ticks late would pop in), else in turn,
            # all of it missing. That's the new finest level at every diving loop, its bars 0.3 px wide: built at
            # once, outside the time budget, it was one of a fast dive's hitches (2026-11-05).
            if force or bar_px(level) >= LAZY_PX:
                rebuild.append(level)
            else:
                overdue.append((1.0, level))
            continue
        if decided and level > old_made_level:
            # Decided levels draw nothing in sight, before or after (Universe.decide()): rebuilt in turn, last, rather
            # than all at once (some 14 of them at every climbing loop's decision)
            overdue.append((0.0, level))
            continue
        if bool(level_obj.get("sub_outlined", True)) != outlined(level):
            # Its outlines come or go: only strips of a pixel or two change, in turn
            overdue.append((1.0, level))
            continue
        near = level_obj.get("sub_near")
        if near is not None:
            near = list(near)
            # Outlined round where the camera was: due once the strips it needs round where it is now reach past that
            past = math.dist((view.cx, view.cy), near[:2]) + needed(level) - near[2]
            if past > 0.0:
                overdue.append((min(1.0, 0.1 + past / outline_reach(level)), level))
                continue
        wx, wy = center_of(level)
        moved = max(abs(wx - built_center[0]), abs(wy - built_center[1]))
        due = (moved + window(level)) / built_window
        # A level split into tiles (bars longer than "tile") is split around where it was built: past split_reach
        # (map units) from there, the pieces near the drone are too big for float32 depths
        split_due = 0.0
        if float(settings.get("tile", 100.0)) / (cell * rule.size(level)) < 2.0:
            split_due = moved * cell / float(settings.get("split_reach", 1000.0))
        if due > 1.0 or split_due > 1.0:
            # How much of what should show is missing: the share of its window the built one no longer covers (all
            # of it once the drone has left it). Due only for being far from where it was split, a sliver.
            missing = min(1.0, max(0.0, moved + window(level) - built_window) / window(level))
            overdue.append((max(missing, 0.01 * min(split_due, 10.0)), level))
    # Which first: how much of its window is missing, times how wide its bars are on screen under the drone: how much
    # you'd see wrong. Diving fast, every fine level falls behind, the finest the most, since its tiles are smallest.
    # Taken by how overdue (moved over its tolerance, which goes like 1 / LAMBDA^k), the finest (bars 0.3 px wide) took
    # every turn, and levels a few up (bars many pixels wide) drifted dozens of tiles from the drone and vanished from
    # around it ("after enough loops downward, smaller shapes no longer render", 2026-11-05). Overdue times bar width
    # came out about the same for every level (1 / LAMBDA^k times LAMBDA^k). Now what lags is the faintest detail.
    overdue = [(min(1e6, missing * bar_px(level)), level) for missing, level in overdue]
    # Overdue levels are rebuilt within a share of wall time ("build_share", 30%), saved up to "build_burst" (12 ms):
    # ticks come about 1.7 times a frame, so a budget a tick let rebuilding take most of every frame flying fast.
    # Never-built levels that show are built anyway. What's left waits for the next tick, most visible first.
    now = time.perf_counter()
    tokens, last = _build_time.get(obj.name, (0.0, now))
    burst = float(settings.get("build_burst", 0.012))
    share = float(settings.get("build_share", 0.3))
    if background.enabled and obj.name not in _in_place:
        share *= WORKER_SHARE
    tokens = min(burst, tokens + (now - last) * share)
    full = tokens >= burst
    queued = [level for _urgency, level in sorted(overdue, reverse=True)]
    tile_size = float(settings.get("tile", 100.0))
    chunk = float(settings.get("chunk", CHUNK))

    def build(level, wait=False):
        """Build a level for where the drone is now: its tiles found here (from the kept levels, which only this
        thread touches), its bars' faces worked out on the worker thread and written in a later tick (take_built()),
        the level showing what it has meanwhile; or all of it here and now (wait: a level that shows and has nothing
        yet; in a background Blender; after a build failed on the worker)"""
        level_obj = by_level[level]
        key = (obj.name, level_obj.name)
        started = time.perf_counter()
        tiles = None
        center, built_window = center_of(level), build_window(level)
        tile = tile_size / cell
        if rule.size(level) <= GIANT:
            # Chunks reach up to a chunk past the window, and are kept: their tiles are found as far
            need = built_window + (chunk * rule.size(level) if chunked(rule, level, tile) else 0.0)
            tiles = as_float_level(rule, level_tiles(cache, universe, level, center, need, build_half), level)
        if tiles is not None and len(tiles):
            # Outlined near the camera, as cores further (near_of()), or all as cores while its strips are too thin
            drawn, rest, near = (motif, cores, near_of(level)) if outlined(level) else (cores, None, None)
            if rest is motif or near is None:
                rest, near = None, None
            record = None if near is None else [*near, outline_reach(level)]
            kept = kept_chunks(cache.chunks, level_obj.name, rule, level, tile, chunk, (drawn, rest))
            scale = cell * rule.size(level)
            if wait or not background.enabled or obj.name in _in_place:
                drop_job(key)
                faces = level_faces(tiles, drawn, center, built_window, tile, kept, rest, near)
                write_level(level_obj, faces, settings, fade(settings, bar_px(level)), center, 0.0, 0.0, scale)
                mark_outlines(level_obj, drawn is motif, record)
            else:
                # The worker fills its own copy of the kept chunks, which becomes the kept one when it's written
                job = Job(cache, level, center, scale, kept and kept.copy(), drawn is motif, record)
                if background.submit(key, level_faces, tiles, drawn, center, built_window, tile, job.chunks, rest,
                                     near):
                    _jobs[key] = job
        else:
            # Nothing of it within its window (or a giant level, drawn as the floor's tint)
            drop_job(key)
            clear(level_obj)
            level_obj["sub_center"] = list(center)
            level_obj["sub_window"] = built_window
            level_obj["sub_built"] = True
            level_obj["sub_bx"], level_obj["sub_by"] = 0.0, 0.0
            level_obj["sub_bs"] = cell * rule.size(level)
            mark_outlines(level_obj, outlined(level), None)
            level_obj.update_tag(refresh={'OBJECT'})
        # What it took of this thread, to tell next time whether it fits in what's left of the budget
        level_obj["sub_cost"] = time.perf_counter() - started

    def take_built(tokens):
        """Write the levels the worker thread has finished, each as its Job has it now (the loops and moves of O
        since it was handed over applied), the one most missed first, while the budget lasts (tokens, returned less
        what the writing took): the rest wait. One built for a cache that's gone (the file reloaded) or for a level
        its object no longer draws is dropped."""
        for key, faces in background.finished(obj.name):
            job = _jobs.get(key)
            if faces is background.FAILED:
                _in_place.add(obj.name)
                _jobs.pop(key, None)
            elif job is not None:
                job.faces = faces
        by_name = {o.name: o for o in objects}
        urgency = {level: urgent for urgent, level in overdue}
        ready = [(key, job) for key, job in jobs_of(obj) if job.faces is not None]
        for key, job in sorted(ready, key=lambda item: urgency.get(item[1].level, 0.0), reverse=True):
            level_obj = by_name.get(key[1])
            if (job.cache is not cache or level_obj is None or level_obj.get("sub_level") != job.level
                    or job.level in giant.drawn):
                del _jobs[key]
                continue
            if tokens <= 0.0:
                break
            started = time.perf_counter()
            corners, sizes, classes, window, chunks = job.faces
            if chunks is not None and job.chunks is chunks:
                cache.chunks[level_obj.name] = chunks
            write_level(level_obj, (corners, sizes, classes, window * job.grow, None), settings,
                        fade(settings, bar_px(job.level)), job.center, job.bx, job.by, job.bs)
            mark_outlines(level_obj, job.outlined, job.near)
            del _jobs[key]
            tokens -= time.perf_counter() - started
        return tokens

    # One kept level found again from fresher roots, if one's getting stale, its time taken from the budget
    started = time.perf_counter()
    refresh(cache, universe, center_of, rule.size(finest), build_half, lambda k: max(build_half(k), build_window(k)))
    tokens -= time.perf_counter() - started
    # Forced ones first, coarsest first (finer ones are cut from them), then queued ones while time is left, each
    # one's time taken from it as it's spent. A queued one starts only if what its object's last build took fits in
    # what's left, or, the first of a tick, if the budget was full: with any time left at all, a tick with a
    # millisecond to spare started a coarse level that took 25 ms (2026-11-05). (Fits in the whole burst instead, a
    # level that once took longer than the burst never started again, and the drone left it behind.)
    for level in sorted(rebuild, reverse=True):
        build(level, wait=True)
    # Giant levels whenever giant_levels() has worked them out again: at once (a loop changes them), a few polygons each
    for level in sorted(giant.drawn):
        level_obj = by_level.get(level)
        if level_obj is not None and level_obj.get("sub_giant") != giant.serial:
            drop_job((obj.name, level_obj.name))
            write_giant(level_obj, giant, level, settings, cell)
    # What the worker thread has finished is written, its time taken from the budget too; of the queued ones, those
    # already under way there wait for it, and at most JOBS are at a time
    tokens = take_built(tokens)
    under_way = {job.level for _key, job in jobs_of(obj)}
    queued = [level for level in queued if level not in under_way]
    first = True
    while (queued and tokens > 0.0 and len(under_way) < JOBS
           and (tokens >= float(by_level[queued[0]].get("sub_cost", 0.0)) or (full and first))):
        started = time.perf_counter()
        build(queued.pop(0))
        tokens -= time.perf_counter() - started
        under_way = {job.level for _key, job in jobs_of(obj)}
        first = False
    _build_time[obj.name] = (tokens, now)
    # Each level's colors as much as its bars' width shows them, a level whose fade() step has changed rewritten (a
    # loop changes none: a level's bars are as wide on screen, its name and the drone's height both relabeled)
    for level, level_obj in by_level.items():
        shown = fade(settings, bar_px(level))
        if level not in giant.drawn and level_obj.get("sub_built") and level_obj.get("sub_fade", 1.0) != shown:
            recolor(level_obj, settings, shown)
    set_pushes(obj, by_level, window, cell)
    tint(obj, settings, giant.tint)


def kept_chunks(chunks, name, rule, level, tile, size, motif):
    """The chunks kept for a level object (in chunks, {level object's name: Chunks}) if its level is built in chunks
    (chunked()), started over if they're another level's, another size or another motif's (outlined or cores); else
    None, and nothing kept for it"""
    if not chunked(rule, level, tile):
        chunks.pop(name, None)
        return None
    kept = chunks.get(name)
    if kept is None or kept.level != level or kept.size != size or kept.motif != motif:
        kept = chunks[name] = Chunks(level, size, motif)
    return kept


def level_faces(float_level, motif, center, window, tile, chunks=None, cores=None, near=None):
    """One level's bars as faces, in its tile units around O: (corners, sizes, color classes, the half side of the
    square round center they cover, chunks). From the chunks kept for it if given (chunked_mesh(), which fills them),
    else cut to the window, a level split into tiles (those are split around the drone, so nothing of one build fits
    the next). Outlined (motif) near the camera (near), as cores further, if cores is given (mixed_polygons()).
    Arithmetic on arrays only, nothing of Blender's: it runs on the worker thread too (background.py)."""
    if chunks is not None:
        return (*chunked_mesh(chunks, float_level, motif, center, window, cores, near), chunks)
    size = float_level.rule.size(float_level.level)
    return (*level_mesh(float_level, motif, center, window, tile / size, cores, near), window, None)


def write_level(level_obj, faces, settings, shown, center, bx, by, scale):
    """Write a level's faces (level_faces()) into its object, colored showing shown of the palette (fade()), with each
    corner's color class kept ("sub_class") for recolor(); its window's middle (center, level-0 units), and its
    unpushed place and scale: at O, a tile a cell of its level, unless the map has moved or looped since the faces were
    worked out (settle()'s take_built())."""
    corners, sizes, classes, window, _chunks = faces
    colors = colors_of(settings, shown)[classes].reshape(-1, 4)
    tiling.write_mesh(level_obj, np.column_stack([corners, np.zeros(len(corners))]), sizes, colors, convex=True)
    mesh = level_obj.data
    kept = mesh.attributes.get("sub_class") or mesh.attributes.new("sub_class", 'INT', 'POINT')
    if len(corners):
        kept.data.foreach_set("value", np.repeat(classes, sizes).astype(np.int32))
    level_obj["sub_fade"] = shown
    level_obj["sub_center"] = [center[0], center[1]]
    level_obj["sub_window"] = window
    level_obj["sub_built"] = True
    level_obj["sub_bx"], level_obj["sub_by"] = bx, by
    level_obj["sub_bs"] = scale
    level_obj.update_tag(refresh={'OBJECT'})


def mark_outlines(level_obj, outlined, near):
    """A level object's outlines as built: "sub_outlined" whether outlined (or its bars' cores alone), and "sub_near"
    where, [x, y, radius, the outlines' reach] (level-0 units), if only near there (settle()'s near_of())"""
    level_obj["sub_outlined"] = bool(outlined)
    if near is not None:
        level_obj["sub_near"] = [float(c) for c in near]
    elif "sub_near" in level_obj:
        del level_obj["sub_near"]


def recolor(level_obj, settings, shown):
    """A built level's colors rewritten showing shown of the palette (fade()), from each corner's color class
    ("sub_class"): one attribute written, not the level built again. A level built without them (by an older add-on)
    is built again next tick."""
    mesh = level_obj.data
    kept, colors = mesh.attributes.get("sub_class"), mesh.color_attributes.get("Col")
    if len(mesh.loops):
        if kept is None or colors is None or len(kept.data) != len(colors.data):
            level_obj["sub_built"] = False
            return
        classes = np.empty(len(kept.data), np.int32)
        kept.data.foreach_get("value", classes)
        colors.data.foreach_set("color", colors_of(settings, shown).astype(np.float32)[classes].ravel())
        mesh.update()
    level_obj["sub_fade"] = shown


def clear(level_obj):
    tiling.write_mesh(level_obj, np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros((0, 4)))
    level_obj["sub_built"] = False


def set_pushes(obj, by_level, window, cell):
    """Finer levels on top: the finest unpushed, each bigger one a step below the one above it, weighted by how far
    that one reaches over the shortest clip end (plane_tiling.py's push layering), the floor a step below them all"""
    settings = obj[PROPERTY]
    clip_ratio = float(settings.get("clip_ratio", 100000.0))
    step = clip_ratio / 2 ** 24
    clip_end = float(settings.get("clip_end", 50000.0)) / rule_of(settings).loop
    push = 0.0
    for level in sorted(by_level):
        level_obj = by_level[level]
        if abs(level_obj.get("sub_push", -1.0) - push) > 1e-12:
            level_obj["sub_push"] = push
            level_obj.update_tag(refresh={'OBJECT'})
        level_obj["camera_push_scale"] = level_obj.get("sub_bs", 1.0)
        # Only a level that draws something needs the ones below it a step further: with every level counted, the
        # giant empty ones too, the biggest were pushed 15% deeper, and a layer pushed by p ends at 1 / (1 + p) of
        # the clip end, short of the horizon
        if len(level_obj.data.polygons):
            reach = window(level) * cell
            push += step * max(0.05, min(1.0, reach / clip_end))
    floor = settings.get("floor")
    if floor is not None and abs(floor.get("sub_push", -1.0) - (push + step)) > 1e-12:
        floor["sub_push"] = push + step
        floor.update_tag(refresh={'OBJECT'})


_giants = {}


def giant_reach(settings):
    """How far (level-0 units) the view may move from where the giant levels were worked out before they're worked out
    again: a tenth of the sight, and no more than split_reach (their faces are split into tiles around there)"""
    return min(0.1 * float(settings["sight"]), float(settings.get("split_reach", 1000.0)) / float(settings["cell"]))


def giant_of(obj, universe, motif, center, settings, peek=False):
    """The map's Giant for the view's place over the floor (center, level-0 units): the one kept, unless the universe
    has changed since (a loop, a move of O, a decision) or the view has moved giant_reach() from it; then worked out
    again (walking every level bigger than GIANT, a hundred and more deep in a dive), for the sight and that reach
    round center. With peek, None instead of working it out."""
    giant = _giants.get(obj.name)
    reach = giant_reach(settings)
    if giant is None or giant.stamp != stamp_of(universe) or math.dist(giant.center, center) > reach:
        if peek:
            return None
        window = float(settings["sight"]) + reach
        giant = _giants[obj.name] = Giant(stamp_of(universe), center, window,
                                          *giant_levels(universe, motif, center, window))
    return giant


def write_giant(level_obj, giant, level, settings, cell):
    """A giant level's object: its polygons (giant_levels()) in level-0 units around O, the object a cell of level 0
    (loops and moves of O then scale and shift it as they do every level object's), carrying giant's serial"""
    polygons, classes = giant.drawn[level]
    corners, sizes, cls = giant_faces(polygons, classes, giant.window, float(settings.get("tile", 100.0)) / cell)
    write_level(level_obj, (corners + np.array(giant.center), sizes, cls, giant.window, None), settings, 1.0,
                giant.center, 0.0, 0.0, cell)
    level_obj["sub_giant"] = giant.serial
    mark_outlines(level_obj, True, None)


def tint(obj, settings, found):
    """The floor's color: found's, the color class of a giant level's polygon over the whole view (giant_levels()),
    or the floor's own"""
    floor = settings.get("floor")
    if floor is None or floor.type != 'MESH':
        return
    color = settings["floor_color"] if found is None else colors_of(settings)[found]
    key = tuple(round(float(c), 6) for c in color)
    if floor.get("sub_tint") == list(key):
        return
    attribute = floor.data.color_attributes.get("Col")
    if attribute is not None:
        attribute.data.foreach_set("color", np.tile(np.asarray(color[:], dtype=np.float32), len(attribute.data)))
        floor.data.update()
    floor["sub_tint"] = list(key)


def drone_move(obj, move):
    """A move in the map's axes (meters) as a world matrix to put before the drone's"""
    return Matrix.Translation(obj.matrix_world.to_3x3() @ move)
