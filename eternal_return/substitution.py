"""Substitution rules for the endless floors: penrose.py's Penrose rhombs (as Robinson half rhombs, looping by phi) and
darb_imam.py's girih tiles (decagons, elongated hexagons, bowties, looping by 2 phi^3 = 8.47), each a module with the
same names, and their tables as arrays (Rule) for the surface engine that draws them (surface_substitution.py: a
rule's levels on any labeled mesh, the endless planes' too, and parts_substitution.py's frames on them).

Until 2026-11-23 this was also the plane floors' own engine (a universe kept as one exact tile, its levels drawn in
a pool of objects round the drone), with endless.py and tiling.py's lattice floors beside it. Phase 5 of the unified
engines plan moved the planes onto the surface engine (an endless plane is a one-tile surface map made up a thousand
levels up), and that engine went. What's left: the rules' tables, the exact tests of which children of a tile meet a
square (cut(), and shows(), which the endless planes' build script uses to keep its made-up levels' bars clear of the
drone), and fade(), how much of its colors a level shows by how wide its bars are on screen.
"""

import importlib
import math
from fractions import Fraction

import numpy as np

from .cyclotomic import phi_power_float, q_phi, q_scale

MARGIN = 1e-9               # float tests closer than this (relative) are too close to call


# --- Rules --------------------------------------------------------------------------------------------------------------

class Rule:
    """A substitution rule (penrose.py, darb_imam.py: a module with the same names), and its tables as arrays, for
    cutting levels in floats: for each kind, each slot's child as the parent's turn and mirror put it (child_pose())"""

    def __reduce__(self):
        # By name: the module it holds doesn't pickle, and its tables are made again from it (rule_named())
        return rule_named, (self.module_name,)

    def __init__(self, module):
        self.module = m = module
        self.module_name = module.__name__.rpartition(".")[2]
        self.name, self.ring, self.n = m.NAME, m.RING, m.N
        self.kinds = m.KINDS
        self.loop, self.log_loop = m.LOOP, math.log(m.LOOP)
        self.lambda_, self.lambda_inv = m.LAMBDA, m.LAMBDA_INV
        self.extra, self.pad, self.reach, self.child_reach = m.EXTRA, m.PAD, m.REACH, m.CHILD_REACH
        self.straddle, self.classes = m.STRADDLE, m.CLASSES
        self.most_children = max(len(m.RULE[kind]) for kind in m.KINDS)
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
        # Each kind's corners' middle and how far its corners are from it (prototile units), indexed by kind: how far a
        # tile can reach from its middle
        self.middles, self.radii = prototile_reach(m)
        self.two_power, self.phi_power = m.LAMBDA_POWERS
        self._powers = {}

    def size(self, k):
        """LAMBDA^k as a float, saturating: level k's tile size in level-0 units"""
        return math.exp(max(-700.0, min(700.0, k * self.log_loop)))

    def shrink_float(self, d, shift=0):
        """2^shift LAMBDA^-d as a float (d >= 0), accurate to the last bit however big d and shift are: level d's tile
        size in level 0's (the surface maps' levels)"""
        return phi_power_float(self.phi_power * d, shift - self.two_power * d)

    def log_shrink(self, d, shift=0):
        """The log of shrink_float(d, shift), past a float's range too"""
        return shift * math.log(2.0) - d * self.log_loop

    def q_grow(self, d):
        """LAMBDA^d exactly in Q(sqrt 5), d of either sign (a pair: a + b sqrt 5)"""
        found = q_phi(self.phi_power * d)
        two = self.two_power * d
        return found if not two else q_scale(found, Fraction(1 << two) if two > 0 else Fraction(1, 1 << -two))

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


def prototile_reach(m):
    """A rule module's kinds' corners' middles and how far their corners are from them (prototile units), indexed by
    kind: how far a tile can reach from its middle (without building its Rule)"""
    true = {kind: np.array([m.RING.xy_small(v) for v in m.PROTOTILE[kind]]) for kind in m.KINDS}
    middles = np.array([true[kind].mean(axis=0) for kind in m.KINDS])
    radii = np.array([float(np.linalg.norm(true[kind] - true[kind].mean(axis=0), axis=1).max()) for kind in m.KINDS])
    return middles, radii


class Rules:
    """The rules by name (penrose.py's and darb_imam.py's NAME), each built the first time a map asks for it: both built
    when this was imported, the first flight on any substitution or cell map waited 0.7 s for them, the rule it doesn't
    use too (2026-11-24)"""

    MODULES = ("penrose", "darb_imam")

    def __init__(self):
        self._built = {}

    def __getitem__(self, name):
        rule = self._built.get(name)
        if rule is None:
            if name not in self.MODULES:
                raise KeyError(name)
            module = importlib.import_module(f"{__package__}.{name}" if __package__ else name)
            rule = self._built[name] = Rule(module)
        return rule

    def __contains__(self, name):
        return name in self.MODULES

    def __iter__(self):
        return iter(self.MODULES)

    def keys(self):
        return self.MODULES

    def values(self):
        return [self[name] for name in self.MODULES]

    def items(self):
        return [(name, self[name]) for name in self.MODULES]


RULES = Rules()


def rule_named(name):
    """RULES[name]: a pickled Rule made again where it's unpickled (a build process: background.py), from its own
    module, not a copy of its tables"""
    return RULES[name]


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


# --- Fading in ---------------------------------------------------------------------------------------------------------

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
