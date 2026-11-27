"""The Darb-i Imam girih tiles and their subdivision, the second rule substitution.py's endless floors run: decagons,
elongated hexagons and bowties, true girih shapes, with unit edges along multiples of 36 degrees, so every vertex is a
point of Z[zeta_10] (cyclotomic.py). Each tile is cut into 35 to 211 tiles LAMBDA = 2 phi^3 = 8.47 times smaller.

Lu and Steinhardt (Science 2007) found the subdivision in the Darb-i Imam shrine's tilework (Isfahan, 1453), and
Tennant (Bridges 2008) gave the full rules. The layout of the children (darb_imam_rule.json) was read from the Tilings
Encyclopedia's picture of it and checked exactly (~/blender_tests/darb_imam: extract.py, own.py, fill.py, verify.py): no
overlaps, as symmetric as each big tile, and every kind cut three times, 1.1 million tiles, has every inner edge shared
by exactly two tiles: no gaps, no overlaps (debug/edge_check.py). Every big edge carries a decagon on its midpoint and a
hexagon near each end, their middles on the edge. The picture's hexagons and bowties lost some of those end hexagons (at
the hexagon's 144 degree corners, the bowtie's 72 degree ones), so where two big hexagons met, neither listed it: a hole
at every level below, which zooming into its edge never ended ("i can zoom into the edge of this gap seemingly
infinitely but it remains an empty space"). The 12 are now in the rule.

Same interface as penrose.py. Two things set it apart:
- Children straddle their parent's edges. The picture clips them, but each level is a plain edge-to-edge tiling. Each
  big tile lists the children whose middles are in it, edges included, so a child on an edge (every edge has a decagon
  on its midpoint, and some the two hexagons beside it) may be listed by both tiles there: a level's tiles are its
  parents' children with doubles dropped (STRADDLE). A tile alone decides only what's more than PAD inside its edges:
  a child reaches at most phi (a decagon's radius) from its middle, the grandchildren 1 / LAMBDA of that further, and
  so on. Owning only the edge children of edges at 0 to 144 degrees, to have no doubles, lost hexagons: near some
  corners only one of the two tiles lists them.
- LAMBDA isn't a unit of the ring (it's 2 times one). Cutting is exact in whole numbers, but a parent's place is its
  child's divided by LAMBDA, so places may have powers of 2 under them (Fractions; cyclotomic.xy_float() takes them).

A tile is (kind, rot, mirror, tau), as in penrose.py: the prototile turned rot * 36 degrees, with its vertex 0 moved to
tau, in its level's units. mirror is always False: each tile is its own mirror image, and so is the rule. Each tile is
written one way only, with its rot as small as it can be (canonical()). A decagon looks the same turned any multiple of
36 degrees, so its rot is always 0. The hexagon and the bowtie look the same turned 180, so theirs is 0 to 4.
"""

import json
import math
import os
from fractions import Fraction

import numpy as np

from .cyclotomic import RING10 as RING

NAME = "darb_imam"
DECAGON, HEXAGON, BOWTIE = 0, 1, 2
KINDS = (DECAGON, HEXAGON, BOWTIE)
NAMES = ("decagon", "hexagon", "bowtie")
MIRRORS = (False,)
N = RING.n
ZERO, ONE = RING.zero, RING.one
PHI = RING.add(ONE, RING.sub(RING.power(2), RING.power(3)))       # 1 + zeta^2 - zeta^3 = 2 cos 36
PHI_INV = RING.sub(RING.power(2), RING.power(3))
LAMBDA = RING.scale(RING.mul(PHI, RING.mul(PHI, PHI)), 2)           # 2 phi^3 = 4 + 2 sqrt 5
PHI3_INV = RING.mul(PHI_INV, RING.mul(PHI_INV, PHI_INV))
LAMBDA_INV = tuple(Fraction(c, 2) for c in PHI3_INV)
assert RING.tidy(RING.mul(LAMBDA, LAMBDA_INV)) == ONE
LOOP = 4.0 + 2.0 * math.sqrt(5.0)
# LAMBDA as 2^a phi^b, (a, b): its powers exact in Q(sqrt 5) and as floats to the last bit (substitution.Rule)
LAMBDA_POWERS = (1, 3)
PHI_F = (1.0 + math.sqrt(5.0)) / 2.0
# Default levels drawn above `made`: by then O is within LAMBDA^-4 = 2e-4 of the default chain's fixed point (a
# decagon's middle)
EXTRA = 4

# Counterclockwise edge directions (powers of zeta), as the rule picture draws the tiles: the decagon from its left
# vertex, the hexagon from its left 72 degree corner, the bowtie from its top right corner (its top and bottom edges
# horizontal, its 216 degree corners at its waist)
EDGES = {
    DECAGON: (8, 9, 0, 1, 2, 3, 4, 5, 6, 7),
    HEXAGON: (9, 0, 1, 4, 5, 6),
    BOWTIE: (5, 8, 7, 0, 3, 2),
}


def walk(directions):
    """The vertices of the polygon whose unit edges go along zeta^d for each d in turn, starting at 0"""
    points, p = [], ZERO
    for d in directions:
        points.append(p)
        p = RING.add(p, RING.power(d))
    assert p == ZERO, "the polygon doesn't close"
    return points


PROTOTILE = {kind: walk(EDGES[kind]) for kind in KINDS}
CORNERS = {kind: [RING.xy_small(v) for v in PROTOTILE[kind]] for kind in KINDS}
TURN = [(math.cos(k * math.pi / 5), math.sin(k * math.pi / 5)) for k in range(N)]
# Whatever its kind, a tile lies within REACH of its vertex 0 (a decagon's width), and within CHILD_REACH (a decagon's
# radius) of its middle, the point its parent is decided by
REACH = 2.0 * PHI_F
CHILD_REACH = PHI_F
PAD = CHILD_REACH / (LOOP - 1.0)
STRADDLE = True

def vertices(kind, rot, tau):
    return [RING.add(tau, RING.rot(v, rot)) for v in PROTOTILE[kind]]


def symmetries(kind):
    """The (s, j) for which the prototile turned s * 36 degrees with its vertex 0 on its vertex j is the same tile"""
    same = set(PROTOTILE[kind])
    return [(s, j) for s in range(N) for j in range(len(PROTOTILE[kind]))
            if set(vertices(kind, s, PROTOTILE[kind][j])) == same]


SYMMETRIES = {kind: symmetries(kind) for kind in KINDS}
PERIOD = {kind: min(s for s, _j in SYMMETRIES[kind] if s > 0) for kind in KINDS}        # 1, 5, 5


def canonical(kind, rot, tau):
    """The tile (kind, rot, False, tau), written with its rot under PERIOD[kind]"""
    for s, j in SYMMETRIES[kind]:
        if (rot + s) % N < PERIOD[kind]:
            return (kind, (rot + s) % N, False, RING.add(tau, RING.rot(PROTOTILE[kind][j], rot)))
    raise AssertionError("no pose turned under the period")


def load_rule():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "darb_imam_rule.json")) as f:
        raw = json.load(f)
    return {kind: tuple((NAMES.index(c[0]), c[1], False, tuple(c[2:6])) for c in raw[NAMES[kind]]) for kind in KINDS}


# The children each big tile lists, (kind, rot, mirror, place), the big tile turned 0 with its vertex 0 on 0, in the
# children's units (its own vertices are LAMBDA times the prototile's)
RULE = load_rule()
# Where each kind appears in a parent: (parent kind, slot)
INVERSE = {kind: [(pk, j) for pk in KINDS for j, child in enumerate(RULE[pk]) if child[0] == kind] for kind in KINDS}


def middle(kind, rot, tau):
    """A tile's middle as floats (each of the three is symmetric about it)"""
    points = [RING.xy_small(v) for v in vertices(kind, rot, tau)]
    return sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)


def child_pose(kind, rot, mirror, slot):
    """The child in slot of a tile turned rot, relative to it: (kind, rot, mirror, place), the child being at
    LAMBDA tau + place in its own units"""
    ck, cr, _cm, ct = RULE[kind][slot]
    child = canonical(ck, (rot + cr) % N, RING.rot(ct, rot))
    return child[0], child[1], False, child[3]


POSES = {kind: [[child_pose(kind, r, False, slot) for slot in range(len(RULE[kind]))] for r in range(N)]
         for kind in KINDS}


def children(tile):
    kind, r, _m, tau = tile
    grown = RING.mul(tau, LAMBDA)
    return [(ck, cr, False, RING.add(grown, place)) for ck, cr, _cm, place in POSES[kind][r]]


def orbits():
    """For each kind, each slot's orbit under the big tile's turns onto itself (as its smallest slot): a parent's
    choice (parents()) is the same one whichever slot of an orbit holds the child"""
    out = {}
    for kind in KINDS:
        index = {(ck, cr, place): slot for slot, (ck, cr, _cm, place) in enumerate(POSES[kind][0])}
        orbit = list(range(len(RULE[kind])))
        for s, j in SYMMETRIES[kind]:
            # The big tile turned s with its vertex 0 on its vertex j is the same tile, its slots in another order:
            # SYMMETRIES is every such turn, so a slot's orbit is its images under them all
            tau = RING.mul(PROTOTILE[kind][j], LAMBDA)
            for slot, (ck, cr, _cm, place) in enumerate(POSES[kind][s]):
                image = index[(ck, cr, RING.add(tau, place))]
                orbit[image] = min(orbit[image], slot)
        out[kind] = tuple(orbit)
    return out


ORBIT = orbits()


def choice(kind, slot):
    """What a parent of kind holding a child in slot is called, for telling whether climbing repeats itself"""
    return kind, ORBIT[kind][slot]


def parent_steps(kind, rot):
    """How to get from a tile (kind, rot, False, tau) to each tile one level up that has it as a child, in whole
    numbers: (parent kind, parent rot, B, choice), the parent's place being (tau phi^-3 + B) / 2 for a whole-number tau
    (darb_imam's 2 phi^3 = LAMBDA). Every pose of the child is tried: a hexagon parent turned about the decagon it holds
    by 36 degrees is a different parent. parents() took 70 ms with Fractions, these tables 2."""
    steps = []
    for pk, slot in INVERSE[kind]:
        _ck, cr, _cm, ct = RULE[pk][slot]
        for s, j in SYMMETRIES[kind]:
            pr = ((rot + s) % N - cr) % N
            # The child's vertex j is the slot's vertex 0: the parent's place is (tau + that offset - the slot's place
            # turned pr) / LAMBDA, then written as canonical() writes it
            offset = RING.mul(RING.sub(RING.rot(PROTOTILE[kind][j], rot), RING.rot(ct, pr)), PHI3_INV)
            for s2, j2 in SYMMETRIES[pk]:
                if (pr + s2) % N < PERIOD[pk]:
                    turn = RING.scale(RING.rot(PROTOTILE[pk][j2], pr), 2)
                    steps.append((pk, (pr + s2) % N, RING.add(offset, turn), choice(pk, slot)))
                    break
    return steps


PARENT_STEPS = {kind: [parent_steps(kind, rot) for rot in range(PERIOD[kind])] for kind in KINDS}


def over_common(tau):
    """tau as whole numbers T over a common power of 2, D: (T, D)"""
    if all(type(c) is int for c in tau):
        return tau, 1
    common = math.lcm(*(Fraction(c).denominator for c in tau))
    return tuple(int(c * common) for c in tau), common


def from_common(scaled, common):
    """The place scaled / common, its whole coefficients as ints"""
    return tuple(c // common if c % common == 0 else Fraction(c, common) for c in scaled)


def parents(tile):
    """Every tile one level up that has tile as a child, as (choice, parent), in its own units (parent_steps())"""
    kind, rot, _m, tau = canonical(tile[0], tile[1], tile[3])
    whole, common = over_common(tau)
    shrunk = RING.mul(whole, PHI3_INV)
    found = {}
    for pk, pr, b, c in PARENT_STEPS[kind][rot]:
        key = (pk, pr, (shrunk[0] + common * b[0], shrunk[1] + common * b[1], shrunk[2] + common * b[2],
                        shrunk[3] + common * b[3]))
        if key not in found:
            found[key] = c
    return [(c, (pk, pr, False, from_common(scaled, 2 * common))) for (pk, pr, scaled), c in found.items()]


def tops(tile):
    """The tiles a top stands for: only itself (it decides what's more than PAD inside it)"""
    return [tile]


def corners_float(tile):
    """The tile's corners as accurate floats in its level's units, counterclockwise: its place, and the prototile's
    corners turned about it"""
    kind, rot, _m, tau = tile
    x, y = RING.xy_float(tau)
    c, s = TURN[rot]
    return [(x + c * px - s * py, y + s * px + c * py) for px, py in CORNERS[kind]]


def inside(tile):
    """How far O is inside what the tile alone decides (its level's units, accurate floats; negative outside): its
    distance from the tile's edges, less PAD"""
    pts = corners_float(tile)
    nearest, crossings = math.inf, 0
    for i in range(len(pts)):
        (ax, ay), (bx, by) = pts[i], pts[(i + 1) % len(pts)]
        ex, ey = bx - ax, by - ay
        t = min(1.0, max(0.0, (-ax * ex - ay * ey) / (ex * ex + ey * ey)))
        nearest = min(nearest, math.hypot(ax + t * ex, ay + t * ey))
        if (ay > 0.0) != (by > 0.0) and 0.0 < ax - ay * ex / ey:
            crossings += 1
    return (nearest if crossings % 2 else -nearest) - PAD


def start_tile():
    """The start's `made`: a decagon with its middle on O"""
    return DECAGON, 0, False, RING.neg(PHI)


# --- Look -------------------------------------------------------------------------------------------------------------

CLASSES = 11


def color_class(kind, rot, mirror):
    """Which way a tile lies, the same at every level: hexagons 0-4 and bowties 6-10 by their rot (each looks the same
    turned 180 degrees), decagons all 5"""
    return {HEXAGON: rot % 5, DECAGON: 5, BOWTIE: 6 + rot % 5}[kind]


def pieces(kind):
    """The prototile as convex polygons, for cutting its bars to it: the bowtie cut at its waist into two trapezoids"""
    pts = [np.array(p) for p in CORNERS[kind]]
    if kind == BOWTIE:
        return [[pts[0], pts[1], pts[2], pts[5]], [pts[2], pts[3], pts[4], pts[5]]]
    return [pts]


def lines(kind, theta):
    """The kind's girih strapwork in prototile coordinates, as segments: Hankin's method at contact angle theta (54
    degrees is girih's own). From each edge's midpoint two rays leave, each turned theta inward from the edge, one
    toward each end. Rays pair up shortest first (Kaplan 2005): of two rays that meet inside the polygon, the pair
    whose longer leg is shortest goes first. In a convex tile that's always the two rays heading for the same corner;
    at the bowtie's 216 degree waist those two part, and each meets its mirror image from the other side instead
    (~/blender_tests/darb_imam/motif.py). Rays meeting head on (parallel, as at a 72 degree corner) become one
    straight segment. Across every edge the lines carry straight on, so a level's lines are one strapwork."""
    polygon = [np.array(p) for p in CORNERS[kind]]
    n = len(polygon)
    tan = math.tan(math.radians(theta))
    perp = lambda v: np.array([-v[1], v[0]])
    rays = []
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        m = (a + b) / 2
        forward, back = b - a, a - b
        rays.append((m, forward + perp(forward) * tan, i))
        rays.append((m, back - perp(back) * tan, i))
    pairs = []
    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            (m1, d1, e1), (m2, d2, e2) = rays[i], rays[j]
            if e1 == e2:
                continue
            c = d1[0] * d2[1] - d1[1] * d2[0]
            dm = m2 - m1
            if abs(c) < 1e-12:
                if abs(dm[0] * d1[1] - dm[1] * d1[0]) < 1e-9 and np.dot(dm, d1) > 0 and np.dot(dm, d2) < 0:
                    pairs.append((np.hypot(*dm) / 2, i, j, None))
                continue
            s = (dm[0] * d2[1] - dm[1] * d2[0]) / c
            t = (dm[0] * d1[1] - dm[1] * d1[0]) / c
            if s <= 1e-9 or t <= 1e-9:
                continue
            meet = m1 + d1 * s
            if not point_in(meet, polygon):
                continue
            pairs.append((max(s * np.hypot(*d1), t * np.hypot(*d2)), i, j, meet))
    pairs.sort(key=lambda p: p[0])
    used, segments = set(), []
    for _length, i, j, meet in pairs:
        if i in used or j in used:
            continue
        used |= {i, j}
        if meet is None:
            segments.append((rays[i][0], rays[j][0]))
        else:
            segments += [(rays[i][0], meet), (meet, rays[j][0])]
    assert len(used) == len(rays), "a ray found no partner"
    return segments


def point_in(p, polygon):
    x, y, hit = p[0], p[1], False
    for i in range(len(polygon)):
        (ax, ay), (bx, by) = polygon[i], polygon[(i + 1) % len(polygon)]
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            hit = not hit
    return hit
