"""The Penrose rhomb substitution, with Robinson triangles (half rhombs) as its tiles: the 10-fold family girih belongs to.

Each rhomb is two mirror-image triangles across one diagonal: the thin rhomb (36 and 144 degrees) across its short one,
two triangles with a 36 degree apex A and unit legs; the thick rhomb (72 and 108) across its long one, two triangles
with a 108 degree apex. Inflated by phi = (1 + sqrt 5) / 2, a thin half is cut exactly into a thin and a thick half, and
a thick half into two thick and a thin one (the subdivision Jeff Preshing's Penrose tiling script uses): a stone
inflation, every big tile cut into whole small ones, so a loop by phi about a fixed point only relabels levels. phi is a
unit of Z[zeta_10] (1 / phi = phi - 1), so places stay exact both ways (cyclotomic.py).

Same interface as the Ammann-Beenker prototype's ab.py: a tile is (kind, rot, mirror, tau), the prototile mirrored
across x (if mirror), turned rot * 36 degrees, moved by tau, in its level's own units (level k's edges are phi^k long).
Its children are the next level down's, in theirs: tau_child = phi tau + (the child's place in the prototile, turned
with it). The rule's tables are worked out from the subdivision's points, exactly, rather than written by hand.

One of substitution.py's rules, with darb_imam.py: what it asks of a rule is the names from NAME on below.
"""

import math

import numpy as np

from .cyclotomic import RING10 as RING

NAME = "penrose"
THIN, THICK = 0, 1
KINDS = (THIN, THICK)
NAMES = ("thin", "thick")
MIRRORS = (False, True)
N = RING.n
# A stone inflation: every child lies in its parent, so a tile alone decides everything inside it (PAD 0), its
# children reach nothing outside it (CHILD_REACH 0), and no two parents have the same child (STRADDLE)
REACH = 1.0                 # a half lies within its legs' length of its apex, its place
CHILD_REACH = 0.0
PAD = 0.0
STRADDLE = False
# Default levels drawn above `made` (substitution.py): by then O is within phi^-14 = 0.001 of the default chain's
# fixed point
EXTRA = 14
ZERO, ONE = RING.zero, RING.one
PHI = RING.add(ONE, RING.sub(RING.power(2), RING.power(3)))       # 1 + zeta^2 - zeta^3 = 2 cos 36
PHI_INV = RING.sub(RING.power(2), RING.power(3))                    # phi - 1
LAMBDA, LAMBDA_INV = PHI, PHI_INV
LOOP = (1.0 + math.sqrt(5.0)) / 2.0

# Corners A (the apex), B, C, counterclockwise: legs AB and AC of length 1, the base BC the rhomb's diagonal
PROTOTILE = {THIN: (ZERO, ONE, RING.power(1)), THICK: (ZERO, ONE, RING.power(3))}
# Edges as (from, to, kind): the legs are the rhomb's edges, the base its inside
EDGES = {kind: ((0, 1, "leg"), (1, 2, "base"), (2, 0, "leg")) for kind in (THIN, THICK)}
ANGLES = {THIN: (36, 72, 72), THICK: (108, 36, 36)}


def orient(rotation, mirror, z):
    return RING.rot(RING.conj(z) if mirror else z, rotation)


def place(tile, z):
    """A prototile point z where tile puts it, in its level's units"""
    return RING.add(orient(tile[1], tile[2], z), tile[3])


def polygon(tile):
    """The tile's corners, counterclockwise"""
    corners = [place(tile, v) for v in PROTOTILE[tile[0]]]
    return corners[::-1] if tile[2] else corners


def solve(kind, apex, b, c):
    """The tile (kind, rot, mirror, tau) whose corners A, B, C are apex, b, c: mirrored if they go clockwise. Never
    with B and C swapped instead: which corner is B matters, since the next cut lands on leg AB."""
    k = RING.unit_index(RING.sub(b, apex))
    for mirror in (False, True):
        tile = (kind, k, mirror, apex)
        if k is not None and place(tile, PROTOTILE[kind][2]) == c:
            return tile
    raise ValueError(f"no {NAMES[kind]} half with corners {apex}, {b}, {c}")


def subdivision(kind):
    """The inflated prototile's children as (kind, apex, b, c), in the children's units (the parent's legs are phi)"""
    a, b, c = (RING.mul(PHI, v) for v in PROTOTILE[kind])
    toward = lambda start, end: RING.add(start, RING.mul(RING.sub(end, start), PHI_INV))
    if kind == THIN:
        p = toward(a, b)
        return [(THIN, c, p, b), (THICK, p, c, a)]
    q, r = toward(b, a), toward(b, c)
    return [(THICK, r, c, a), (THICK, q, r, b), (THIN, r, q, a)]


# The children of each inflated prototile, (kind, rot, mirror, place), in the children's units
RULE = {kind: tuple(solve(*child) for child in subdivision(kind)) for kind in (THIN, THICK)}
# Where each kind appears in a parent: (parent kind, slot), the choices a made-up parent has
INVERSE = {kind: [(pk, j) for pk in (THIN, THICK) for j, child in enumerate(RULE[pk]) if child[0] == kind]
           for kind in (THIN, THICK)}


def child_pose(kind, rotation, mirror, slot):
    """The child in slot of a tile turned rotation (and mirrored), relative to it: (kind, rot, mirror, place), the
    child being at phi tau + place in its own units"""
    ck, cr, cm, ct = RULE[kind][slot]
    return ck, (rotation + (-cr if mirror else cr)) % N, mirror != cm, orient(rotation, mirror, ct)


def children(tile):
    kind, r, m, tau = tile
    grown = RING.mul(tau, PHI)
    return [(ck, (r + (-cr if m else cr)) % N, m != cm, RING.add(grown, orient(r, m, ct)))
            for ck, cr, cm, ct in RULE[kind]]


def parent(tile, parent_kind, slot):
    """The tile one level up that has tile as its child in slot (an INVERSE entry), in its own units"""
    kind, r, m, tau = tile
    ck, cr, cm, ct = RULE[parent_kind][slot]
    assert ck == kind
    pm = m != cm
    pr = (r - (-cr if pm else cr)) % N
    return (parent_kind, pr, pm, RING.mul(RING.sub(tau, orient(pr, pm, ct)), PHI_INV))


def parents(tile):
    """Every tile one level up that has tile as a child, as (choice, parent): each INVERSE entry"""
    return [((pk, slot), parent(tile, pk, slot)) for pk, slot in INVERSE[tile[0]]]


def choice(kind, slot):
    """What a parent of kind holding a child in slot is called, for telling whether climbing repeats itself"""
    return kind, slot


# The default parent rule (~/blender_tests/girih/default_chain.py): a thick half's parent is the thin half it's slot 1
# of, a thin half's the thick half it's slot 2 of; their cycle's fixed point is 0.0885 of the tile clear of every line
# and edge (hankin54)
DEFAULT = {THICK: (THIN, 1), THIN: (THICK, 2)}


def default_parent(tile):
    return parent(tile, *DEFAULT[tile[0]])


def rotated(tile, k):
    """tile turned k * 36 degrees about the origin"""
    kind, r, m, tau = tile
    return (kind, (r + k) % N, m, RING.rot(tau, k))


def mirrored(tile, a, b):
    """tile mirrored across the line through a and b: z -> a + zeta^j conj(z - a), zeta^j = (b - a) / conj(b - a)"""
    d = RING.sub(b, a)
    x, y = RING.xy_small(d)
    j = round(2 * math.atan2(y, x) / (2 * math.pi / N)) % N
    assert RING.rot(RING.conj(d), j) == d, "the line's angle isn't a multiple of 18 degrees"
    return (tile[0], (j - tile[1]) % N, not tile[2], RING.add(a, RING.rot(RING.conj(RING.sub(tile[3], a)), j)))


def partner(tile):
    """The other half of the tile's rhomb: its mirror image across the base"""
    return mirrored(tile, place(tile, PROTOTILE[tile[0]][1]), place(tile, PROTOTILE[tile[0]][2]))


def tops(tile):
    """The tiles a top stands for: itself and its partner"""
    return [tile, partner(tile)]


def rhomb(tile):
    """The tile's whole rhomb, counterclockwise: A, B, the far apex B + C - A, C"""
    a, b, c = (place(tile, v) for v in PROTOTILE[tile[0]])
    far = RING.sub(RING.add(b, c), a)
    corners = [a, b, far, c]
    return corners[::-1] if tile[2] else corners


coverage = rhomb


def corners_float(tile):
    """The half's corners as accurate floats in its level's units, counterclockwise"""
    return [RING.xy_float(v) for v in polygon(tile)]


def inside(tile):
    """How far O is inside the half (its level's units, accurate floats; negative outside)"""
    pts = corners_float(tile)
    best = math.inf
    for i in range(3):
        (ax, ay), (bx, by) = pts[i], pts[(i + 1) % 3]
        ex, ey = bx - ax, by - ay
        best = min(best, (ex * -ay - ey * -ax) / math.hypot(ex, ey))
    return best


def start_tile():
    """The start's `made`: the thick half with O deepest inside it of those with small places"""
    point = min(((a, b, c, d) for a in range(-2, 3) for b in range(-2, 3) for c in range(-2, 3) for d in range(-2, 3)),
                key=lambda v: -inside((THICK, 0, False, RING.neg(v))))
    return THICK, 0, False, RING.neg(point)


# --- Look -------------------------------------------------------------------------------------------------------------

CLASSES = 10


def color_class(kind, rotation, mirror):
    """Which way a rhomb lies, the same at every level (colors must be, for loops to be seamless), and the same for
    both its halves: its kind, and its axis (apex to far apex) modulo 180 degrees, one of 5 ways. 0-4 thin, 5-9 thick."""
    axis = 18 if kind == THIN else 54                    # the prototile's axis, degrees
    angle = (-axis if mirror else axis) + 36 * rotation
    return 5 * kind + round(((angle - axis) % 180) / 36) % 5


def tile_key(tile):
    """A tile as its kind and corners, the same however its (rot, mirror, tau) write it; apex first"""
    a, b, c = (place(tile, v) for v in PROTOTILE[tile[0]])
    return tile[0], a, frozenset((b, c))


def pieces(kind):
    """The half as convex polygons (floats), for cutting its bars to it: the triangle itself"""
    return [[np.array(RING.xy_small(v)) for v in PROTOTILE[kind]]]


def hankin(polygon, theta):
    """Hankin strapwork of a convex polygon (counterclockwise points), contact angle theta (degrees): from each edge's
    midpoint two rays at theta to the edge, one toward each end, each corner's pair meeting"""
    n = len(polygon)
    tan = math.tan(math.radians(theta))
    perp = lambda v: np.array([-v[1], v[0]])
    mids = [(polygon[i] + polygon[(i + 1) % n]) / 2 for i in range(n)]
    segments = []
    for i in range(n):
        m1, m2 = mids[i - 1], mids[i]
        e1 = polygon[i] - polygon[i - 1]
        d1 = e1 + perp(e1) * tan
        e2 = polygon[i] - polygon[(i + 1) % n]
        d2 = e2 - perp(e2) * tan
        c = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(c) < 1e-12:
            segments.append((m1, m2))
            continue
        s = ((m2 - m1)[0] * d2[1] - (m2 - m1)[1] * d2[0]) / c
        meet = m1 + d1 * s
        segments += [(m1, meet), (meet, m2)]
    return segments


def lines(kind, theta):
    """The half's share of its rhomb's strapwork, in prototile coordinates: what's on its side of the base. Girih
    tiles' lines cross every edge at its midpoint at 54 degrees (Lu and Steinhardt 2007): theta 54 is girih's own."""
    a, b, c = (np.array(RING.xy_small(v)) for v in PROTOTILE[kind])
    rhomb = [a, b, b + c - a, c]
    edge = c - b
    normal = np.array([-edge[1], edge[0]])
    if np.dot(a - b, normal) < 0:
        normal = -normal
    out = []
    for p, q in hankin(rhomb, theta):
        fp, fq = float(np.dot(p - b, normal)), float(np.dot(q - b, normal))
        if fp <= 1e-12 and fq <= 1e-12:
            continue
        if fp < -1e-12 or fq < -1e-12:
            cut = p + (q - p) * (fp / (fp - fq))
            p, q = (p, cut) if fp > 0 else (cut, q)
        out.append((p, q))
    return out
