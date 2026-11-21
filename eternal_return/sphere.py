"""Endless sphere maps: a planet tiled by a cube's lattices, each face splitting 2 x 2 at every level, that the drone can
dive into anywhere forever and climb back out of to see the whole globe (prototyped in ~/blender_tests/sphere).

The lattice is drawn on the sphere through Rančić's conformal cube chart (sphere_chart.py), so its lines run straight
across the cube's edges and pinch at its 8 corners, where three faces' right angles open to 120 degrees. A second
layout, the conformal icosahedron, is sphere_ico.py: the same machinery with its own chart and fold (a map's state says
which). What's drawn on a layout's lattice is its motif (the "motif" setting), every level over the coarser ones:
- "sierpinski" (sphere_sierpinski.py, the default): the cube's binary carpet, Pascal's triangle mod 2 from every face's
  corners, and the icosahedron's Sierpinski globe, every triangle's middle cut out. Filled color, each level's shapes
  coming in on top of the coarser ones' as they grow to a few pixels.
- "stars": on the cube, the endless floors' 8-point star outlines, moved half a cell so they sit on the lattice's
  corners: at a cube corner three quarters of every level's star make a 6-point star, nested at every scale, and a
  16-point sun around it makes a 12-point one (CubeStars). On the icosahedron, 12-point rosettes that close up into
  10-point suns at its corners (sphere_ico.IcoStars).
Under the motif can be a planet (the "planet" setting, planet.py): a whole planet's image, Earth's from NASA's Blue
Marble, on the sphere where it has detail, the motif's levels starting only where the image runs out of it, so from
far away it's the planet and diving into it gives way to the motif.
Over the sphere can be 3D tiles (the "tiles" setting, tiles3d.py): a planet's surface streamed as OGC 3D Tiles, Google's
Photorealistic 3D Tiles for Earth, from the globe down to the street. Where they're drawn, the sphere isn't (it's drawn
only where no tile is yet, behind everything). Auto scale measures the drone's height alone, not its distance to the
ground: its height over the valley it's in, the lowest ground around it (follow_base()), so it skims mountain tops at the
pace of their height over the valley, slows down into the valley, and can use the slopes as ramps. The map's origin is
lifted to the ground's height under the drone (the lift), so the drone stays near it, in float32, flying low over
mountains as anywhere. A floor just beyond the drone camera's lens clip over their ground (the "tiles_floor" setting)
holds the drone off it: flown into it, it glides along it. Dives stop where tiles have no more detail in any case
(tiles3d.DEEPEST_UNIT).

The world is kept in two parts, like an endless map's (endless.py): N = 2^n units from the cube's center to a face, and
the anchor A, a whole point of the cube's surface, both exact Python integers that grow without limit; and everything
else in map units around A's point on the sphere, the map's origin, in floats that stay within the view. The sphere's
radius is R = rho N (rho, map units per unit, a setting), its center at -R n(A) in map coordinates, which are the
cube's axes: the map's object never turns, flying around the globe or not.
- A dive loop by 2^k doubles N and A k times (dive()): a pure change of units, about the map's origin, so the view
  doesn't change at all, at any scale, the whole globe in view or not. Loops are by powers of 2 only (exact_factor()).
- A climb loop first moves the anchor to the nearest point of the 2^k times coarser lattice, then halves (climb()). At
  N = 1, the globe, there's nothing coarser: climb loops stop there (can_climb()).
- The anchor follows the point below the drone (recenter()), so the drone stays near the map's origin.
R is never formed: every difference from A is worked out without cancellation (sphere_chart.py and surface_offset()),
so it's as exact a thousand loops deep as at the globe.

Drawing (draw handlers, so in Solid and Rendered mode alike): each pixel's ray meets the sphere (ray_hit()), its point
is measured from the anchor scaled by N (surface_offset()), folded into the chart's domain and turned into an offset
from the folded anchor in units (sphere_chart.Reference.inverse_difference()), all in float32 on the GPU. The levels to
draw, and each level's cell position of the folded anchor, exactly, come from here (Plan): the finest from the pixel
footprint under the drone, the coarsest from the first level whose piece covers the whole view (uniform()), every level
in between per pixel. Then a little sunlight, so the globe reads as round, and the fog.

The map's object (an empty, at the map's origin) holds:
- "sphere": its settings (see DEFAULTS), written by the map's build script
- "sphere_state": "n a0 a1 a2", n and the anchor's coordinates in hex (they can be any size)
- "sphere_version": counts the state's writes, so an undo, which puts an older one back, is noticed
- "sphere_lift": the map's origin's height over the sphere, a share of its radius R (0 without tiles): the origin is the
  anchor's point lifted R lift along its normal, so the sphere's center is at -R (1 + lift) n(A)
"""

import itertools
import math
import time
from fractions import Fraction

import numpy as np

try:
    from . import background
    from . import planet as planets
    from . import sphere_chart as chart
    from . import sphere_ico
    from . import space
    from . import sphere_sierpinski
    from . import tiles3d
except ImportError:          # tests import it by itself
    import background
    import planet as planets
    import sphere_chart as chart
    import sphere_ico
    import space
    import sphere_sierpinski
    import tiles3d

PROPERTY = "sphere"
STATE = "sphere_state"
VERSION = "sphere_version"
LIFT = "sphere_lift"

# The "sphere" property's keys:
# - rho: map units per unit (the lattice's finest whole step), so the sphere's radius is rho * 2^n
# - resolution: the share of the view's pixels, across and down, the sphere is worked out at, scaled up to the view
# - auto_resolution: lower the resolution, down to MIN_RESOLUTION, while frames miss the display's refresh, and raise it
#   back up to resolution once they've kept up for a while
# - show_info: N, the levels drawn and the frame time in the corner of each view
# - motif: what's drawn on the lattice, "sierpinski" or "stars" (see the module's docstring)
# - planet: a planet's tile pyramid (a folder made by tools/make_planet.py, "//" for the .blend's), the sphere's
#   surface under the motif, or "" for none (planet.py)
# - planet_handoff: the coarsest motif level drawn over the planet has cells this many of its finest texels across
# - planet_level_bias: levels coarser than a texel a pixel the planet is drawn at (fewer tiles to load, a little softer)
# - tiles: 3D tiles over the sphere (tiles3d.py): "google" for Google's Photorealistic 3D Tiles, a tileset's URL or
#   path ("//" for the .blend's folder), or "" for none
# - tiles_error: the screen-space error, in pixels, a tile may have before its finer children are drawn instead
# - tiles_motion: whether detail that the view's motion smears is left unloaded until it slows (tiles3d.MOTION_SHARP):
#   off, every tile in view is loaded to tiles_error however fast it crosses the screen
# - tiles_floor: how far over the tiles' ground the drone is held, as a share of how far the drone camera's lens clip
#   reaches (clip_reach(), floor_of()): flown into it, it glides along it, and the clip never cuts into the ground. 0 for
#   none: the drone stops at the ground instead, like at any surface (crossing())
# - tiles_base: the lowest the valley level auto scale measures the drone's height over can be, in meters
#   (follow_base()): the Dead Sea's shore, the lowest ground the tiles show (their heights are over WGS84's ellipsoid,
#   within about 100 m of sea level; the ocean's floor isn't in them). It's also what the valley's reach is measured from.
# - tiles_valley: how far around the drone the valley level is looked for (follow_base()), a share of its height over
#   tiles_base: on a ridge, far enough to take in the valley floors beside it. 0: the valley level is tiles_base
#   everywhere, so auto scale measures plain height over the lowest ground on Earth
DEFAULTS = {"rho": 1.0, "resolution": 1.0, "auto_resolution": True, "show_info": False,
            "motif": sphere_sierpinski.MOTIF, "planet": "", "planet_handoff": 2.0, "planet_level_bias": 0.5,
            "tiles": "", "tiles_error": tiles3d.ERROR, "tiles_motion": True, "tiles_floor": 1.1, "tiles_base": -430.0,
            "tiles_valley": 2.0}
# auto_resolution, as the Mandelbrot floor's (see mandelbrot.py): a frame slower than MISSED_FRAME missed the display's
# refresh; MISSED_TO_LOWER of the last RECENT_FRAMES lower the resolution a step, FAST_FRAMES_TO_RAISE in a row raise it
MISSED_FRAME = 0.025
RECENT_FRAMES = 12
MISSED_TO_LOWER = 2
FAST_FRAMES_TO_RAISE = 60
MAX_RAISE_WAIT = 480
RESOLUTION_STEP = 0.05
MIN_RESOLUTION = 0.2

# The motif, in cell units of the cells centered on the lattice's corners (fx, fy from a cell's corner, which is the
# middle of four of the lattice's cells), every line at a dyadic fraction with a few bits: the GPU puts a line exactly
# where a cell's position minus the line's is computed exactly in float32, which Sterbenz's lemma gives it when both have
# few bits, at any level, however big (see Plan.level_data()). The prototype's 0.36, 0.025, 0.01 and 0.04, within 2%.
STAR = 23 / 64                  # the 8-point star's diamond: |u| + |w| = STAR (u, w from the cell's middle)
S_HALF = 65 / 256               # its square's half-size, STAR / sqrt 2 to 0.1%
RING = 13 / 512                 # its outline's width
RING_DIAGONAL = 37 / 2048       # the diamond's outline half-width in fx + fy: RING / 2 * sqrt 2 to 0.2%
LINE = 5 / 512                  # the gold lines' half-width
GAP = 5 / 128                   # where they stop short of a cell's corner (the lattice's cell middles)
# The cube corners' suns: 16 points, between this level's star and the next level's sun (half the size)
SUN_POINTS, SUN_OUTER, SUN_INNER = 16, 0.485, 0.39
# Levels whose cells are this many times the view's extent are tested for being one color throughout (exactly, so any
# number above 8 is right; every level below it is drawn per pixel)
COARSE = 16.0
# The finest level drawn: its star outline at least this many pixels wide where the floor is finest
FINEST_PIXELS = 0.3
MAX_LEVELS = 48
# Levels up to 2^FINE_LEVELS times the finest drawn are drawn with the pixel's place in its cell in plain float32, good
# to 6e-8 cells, far below a footprint at the finest's scale: only coarser levels need every line placed exactly
FINE_LEVELS = 16


def hsv(h, s, v):
    """sRGB 0..255"""
    import colorsys
    return tuple(c * 255.0 for c in colorsys.hsv_to_rgb(h % 1.0, s, v))


SQUARE = [hsv(0.63 + 0.25 * p, 0.9, 1.0) for p in range(4)]
TIP = [hsv(0.55 + 0.25 * p, 0.9, 0.75) for p in range(4)]
GOLD = hsv(0.13, 0.8, 1.0)
SUN = hsv(0.12, 0.42, 1.0)
FLOOR = hsv(0.72, 0.6, 0.08)


def float32(x):
    return float(np.float32(x))


def words(value):
    """An exact value (Fraction) as three float32s whose sum is it to about 2^-72 of 1"""
    out = []
    rest = Fraction(value)
    for _ in range(3):
        w = float32(float(rest))
        out.append(w)
        rest -= Fraction(w)
    return out


# ----- the world -----

class World:
    """N = 2^n, the anchor A (three ints, a point of the cube's surface, |A|_max = N) and rho (see the module's
    docstring). The anchor folded into the chart's domain is A_f = G @ A (G a signed permutation): (N, a, b). The
    icosahedron's (sphere_ico.IcoWorld) has the same interface."""

    layout = "cube"
    level_shift = 1                  # a level's cells are 2N / 2^level units
    chart_sign = -1                  # the chart's dz~ runs against the in-face axes (from the face's corner)
    unit_angle = math.pi / 4         # radians N units span on the sphere at a face's middle (a face: 2N, 90 degrees)
    walls = chart.WALLS
    fold_steps = chart.FOLD_STEPS

    def __init__(self, n, A, rho):
        self.n, self.A, self.rho = int(n), [int(c) for c in A], float(rho)
        assert self.n >= 0 and max(abs(c) for c in self.A) == self.N, (self.n, self.A)
        self.A_f, self.G = chart.fold_point(self.A)
        self._references = {}
        # Exact per-level values of each motif, which depend on N and the anchor only (level_data(), uniform())
        self._levels = {}
        self._uniform = {}

    @property
    def N(self):
        return 1 << self.n

    @property
    def kappa(self):
        """1 / R in map units, 0 once it's below any float"""
        return math.ldexp(1.0 / self.rho, -self.n) if self.n < 1070 else 0.0

    def charts(self, view_units=1.0, corner_view_units=None):
        """The chart's references at the folded anchor and the domain's corner (sphere_chart.Charts), scaled for a view
        of view_units (cached by the powers of 2 of the sizes)"""
        key = (math.floor(math.log2(max(view_units, 1e-300))),
               math.floor(math.log2(max(corner_view_units or view_units, 1e-300))))
        charts = self._references.get(key)
        if charts is None:
            if len(self._references) > 8:
                self._references.clear()
            charts = self._references[key] = chart.Charts(self.n, self.A_f[1], self.A_f[2], 2.0 ** key[0],
                                                           2.0 ** key[1])
        return charts

    def normal(self):
        """n(A), cube axes"""
        return self.G.T.astype(float) @ self.charts().anchor.n0

    def text(self):
        return " ".join([str(self.n)] + [format(c, "x") for c in self.A])

    @staticmethod
    def from_text(text, rho):
        """The world a sphere_state holds: the cube's, or the icosahedron's (its text starts with its layout)"""
        if text.startswith(sphere_ico.LAYOUT):
            return sphere_ico.IcoWorld.from_text(text, rho)
        n, *A = text.split()
        return World(int(n), [int(c, 16) for c in A], rho)

    def cell(self, level):
        return Fraction(2 * self.N, 1 << level)

    def to_corner(self):
        """From the folded anchor to the domain's corner (the cube's corner, where the corner's reference is), in units
        along a + i b"""
        return complex(float(self.N - self.A_f[1]), float(self.N - self.A_f[2]))

    def dived_anchor(self, k):
        """The anchor after a dive loop by 2^k"""
        return [c << k for c in self.A]

    @staticmethod
    def climbed_anchor(A, k):
        """An anchor A on the 2^k times coarser lattice after a climb loop by 2^k"""
        return [c >> k for c in A]

    def nearest_vertex(self, p, step=1):
        return nearest_vertex(self, p, step)


# ----- the sphere's geometry, in map coordinates (cube axes, map units, the anchor's point at the origin) -----

def surface_offset(world, p, normal=None):
    """N (n(p) - n(A)) for map points p (..., 3): n(p) the direction of p from the sphere's center, so for a point on
    the sphere, its offset from the anchor, in units of 1 / N (map units / rho for small ones). Without cancellation
    however big R is: with q = kappa p, n(p) = (q + n_A) / L, L = |q + n_A|, and 1 - L = -(2 q.n_A + |q|^2) / (1 + L)"""
    p = np.asarray(p, dtype=float)
    nA = world.normal() if normal is None else normal
    k = world.kappa
    pn = p @ nA
    L = np.sqrt(np.maximum(0.0, 1.0 + k * (2.0 * pn + k * np.sum(p * p, axis=-1))))
    lift = (2.0 * pn + k * np.sum(p * p, axis=-1)) / (1.0 + L)
    return (p - lift[..., None] * nA) / (world.rho * L[..., None])


def height(world, p, normal=None):
    """How far map points p are above the sphere (negative inside), without cancellation: |p + R n_A| - R =
    (2 p.n_A + kappa |p|^2) / (1 + L)"""
    p = np.asarray(p, dtype=float)
    nA = world.normal() if normal is None else normal
    k = world.kappa
    pn = p @ nA
    s = 2.0 * pn + k * np.sum(p * p, axis=-1)
    return s / (1.0 + np.sqrt(np.maximum(0.0, 1.0 + k * s)))


def down(world, p, normal=None):
    """The unit direction from map point p toward the sphere's center"""
    p = np.asarray(p, dtype=float)
    nA = world.normal() if normal is None else normal
    v = world.kappa * p + nA
    return -v / np.linalg.norm(v, axis=-1, keepdims=True)


def ray_hit(world, c, r, normal=None):
    """The nearest t > 0 where the rays c + t r (map coordinates) meet the sphere, or inf. The quadratic scaled by kappa,
    which can be 0: kappa |r|^2 t^2 + 2 (kappa r.c + r.n_A) t + (kappa |c|^2 + 2 c.n_A) = 0, the nearer root as
    q / (-b + sqrt(b^2 - a q)), which doesn't cancel."""
    r = np.asarray(r, dtype=float)
    c = np.asarray(c, dtype=float)
    nA = world.normal() if normal is None else normal
    k = world.kappa
    a = k * np.sum(r * r, axis=-1)
    b = k * (r @ c) + r @ nA
    q = k * float(c @ c) + 2.0 * float(c @ nA)
    disc = b * b - a * q
    hit = (disc >= 0.0) & (b < 0.0) & (q > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(hit, q / (-b + np.sqrt(np.maximum(disc, 0.0))), np.inf)
    return np.where(t > 0.0, t, np.inf)


# A view's size in units is guessed from its points' m (N times their directions' offsets), which is about right where
# the chart's stretch is about 1. Near the domain's corner, though, a point's offset in units grows past its m by a
# power of N (the cube's by N^(1/4), the icosahedron's by N^(1/6)), so there the guess is too small by up to that much,
# and deep enough (past 2^1000) the chart scaled for it overflows. Then the guess grows GUESS_GROWTH times, up to
# GUESS_TRIES times, until the offsets come out finite. A view found more than GUESS_SLACK times past the charts' scale
# has them refit to it: the inverse copes with a view up to 2^200 past its scale, but the chart's forward direction
# (z^4, a move's offset) overflows past 2^256, and in the shallows a corner's guess is short by up to 2^(n / 4).
GUESS_GROWTH = 2.0 ** 64
GUESS_TRIES = 48
GUESS_SLACK = 2.0 ** 64


def inverse_from_guess(world, m, guess, jacobian=False, estimate=False):
    """Map offsets m (..., 3: N times, the map's cube axes) folded into the domain and taken back to D through the
    charts scaled for a view of guess units, grown until every D is finite: the charts, the folded offsets and their
    reflections, and charts.inverse()'s D (, J) and corner mask"""
    G = world.G.astype(float)
    m = np.asarray(m)

    def attempt(view_units, corner_view_units=None):
        charts = world.charts(view_units, corner_view_units)
        mf, R = chart.fold_offsets(m @ G.T, charts.anchor.wall_distances(), world.fold_steps, world.walls)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            out = charts.inverse(mf, jacobian=jacobian, estimate=estimate)
        return charts, mf, R, out

    for _ in range(GUESS_TRIES):
        charts, mf, R, out = attempt(guess)
        if np.all(np.isfinite(out[0])):
            break
        guess *= GUESS_GROWTH
    D, corner = out[0], out[-1]
    if np.all(np.isfinite(D)):
        # The view found, the anchor's and the corner's, as Plan measures them
        from_anchor = np.where(corner, D + world.to_corner(), D) if corner.any() else D
        extent = max(1e-300, float(np.max(np.abs(from_anchor))))
        corner_extent = float(np.max(np.abs(D[corner]))) if corner.any() else None
        if max(extent, corner_extent or 0.0) > guess * GUESS_SLACK:
            charts, mf, R, out = attempt(extent, corner_extent)
    return charts, mf, R, out


# ----- moving the anchor -----

def fold_matrix_exact(R):
    """A fold's reflection matrix (floats from sphere_chart.fold_offsets()) as the exact signed permutation it is"""
    return np.rint(R).astype(np.int64)


def nearest_vertex(world, p, step=1):
    """The lattice point (a multiple of step in every coordinate) nearest the point of the sphere below map point p, and
    its offset from the anchor's point (map units, cube axes), exactly: the point below p folded into the domain, its
    offset there rounded to the lattice, and back through the same fold, an isometry, so the offset is the folded one's
    difference from the point below p, unfolded (no cancellation)."""
    nA = world.normal()
    m = surface_offset(world, p, nA)
    charts, mf, R, (D, corner) = inverse_from_guess(world, m, max(1.0, float(np.linalg.norm(m)) * 4.0))
    ref = charts.anchor
    D = complex(D)
    N, a, b = world.N, world.A_f[1], world.A_f[2]
    base_a, base_b = (N, N) if corner else (a, b)
    Pa = step * round((Fraction(base_a) + Fraction(D.real)) / step)
    Pb = step * round((Fraction(base_b) + Fraction(D.imag)) / step)
    # In the domain: N >= Pa >= Pb >= 0 (a lattice point on a wall is on both sides)
    Pa = max(0, min(N, Pa))
    Pb = max(0, min(Pa, Pb))
    m_new_f = ref.forward_scaled(ref.dz_of(Pa - a, Pb - b))
    unfold = (fold_matrix_exact(R) @ world.G).T
    A_new = [int(v) for v in unfold @ np.array([N, Pa, Pb], dtype=object)]
    offset = m + unfold.astype(float) @ (m_new_f - mf)
    return A_new, world.rho * offset


def recenter(world, p, reach=1.0):
    """If the lattice point nearest the point below map point p isn't the anchor and p is more than reach units from the
    anchor's point (along the surface, roughly), move the anchor there. Returns the move of the map's origin (map units:
    the drone moves by minus it), or None."""
    m = surface_offset(world, p)
    if float(np.linalg.norm(m)) < reach:
        return None
    A_new, offset = world.nearest_vertex(p)
    if A_new == world.A:
        return None
    world.__init__(world.n, A_new, world.rho)
    return offset


def dive(world, k):
    """A dive loop by 2^k: the unit halves k times; everything in map units doubles, the sphere's radius too"""
    world.__init__(world.n + k, world.dived_anchor(k), world.rho)


def can_climb(world, k):
    return world.n >= k


def climb(world, k, p):
    """A climb loop by 2^k: first the anchor moves to the nearest point of the 2^k times coarser lattice below map point
    p, then the unit doubles k times. Returns the move of the map's origin before the loop (map units, at the old
    scale: the drone moves by minus it first), or None."""
    assert can_climb(world, k)
    step = 1 << k
    A_new, offset = world.nearest_vertex(p, step)
    world.__init__(world.n - k, world.climbed_anchor(A_new, k), world.rho)
    return offset


def exact_factor(factor):
    """The power of 2 nearest a loop factor, as the float the drone is scaled by"""
    power = round(math.log2(factor)) if factor > 0 else 1
    if power == 0:
        power = 1 if factor > 1.0 else -1
    return 2.0 ** power


# ----- points of the cube placed on the sphere, exactly: the forward direction, for what's built on it (parts.py) -----

def wall_float(value):
    """A distance from a wall in units (an int of any size) as a float, inf past a float's range"""
    try:
        return float(value)
    except OverflowError:
        return math.inf


def surface_offsets(world, deltas, view_units):
    """N (n(A + d) - n(A)) (cube axes, (..., 3)) for points A + d of the cube's surface, d (..., 3) their offsets from
    the anchor in units along the cube's axes, as floats (exact for the lattice's points near the anchor: they're
    dyadic, with few bits), within about view_units of it. The forward direction of what the GPU does per pixel, with no
    cancellation at any depth:
    - each point, in the anchor's folded frame, is folded into the chart's domain by reflecting it across the domain's
      walls (exactly: on the cube, a reflection is a sign change or a swap of coordinates), noting which walls in turn
    - its offset from the folded anchor there goes through the chart (sphere_chart.Charts: from the anchor, or from the
      cube's corner if it's nearer that)
    - and back through the same reflections on the sphere: a reflection σ of the point's offset m from the anchor is
      σ m - 2 s ν, s the anchor's distance from the wall and ν its normal (as sphere_chart.fold_offsets() folds), so a
      chain of them is S m + c, c summed from the walls' distances (wall_distances()), never from N n(A) itself"""
    d = np.array(deltas, dtype=float).reshape(-1, 3) @ world.G.T.astype(float)
    N, a, b = world.N, world.A_f[1], world.A_f[2]
    charts = world.charts(view_units)
    s = charts.anchor.wall_distances()
    # The folded anchor's distances from the domain's walls, on the cube: z = 0, y = z, x = y (sphere_chart.WALLS). A
    # wall too far from the view for anything in it to cross (FAR_WALL views: as the GPU folds) is infinitely far, so
    # the folds never take it, and never add its stand-in distance on the way back
    w0, w1, w2 = (math.inf if s[k] == chart.FAR_WALL_DISTANCE else wall_float(v)
                  for k, v in enumerate((b, a - b, N - a)))
    codes = np.zeros(len(d), dtype=np.int64)
    for _ in range(chart.FOLD_STEPS):
        # How far each point is inside each wall (negative: past it), and the wall it's furthest past: a point on a
        # wall (a cube edge is on two walls' planes) can be rounded a fraction of a float's step past it, and reflected
        # it rounds back to itself, so taking the walls in order it would never get to the ones it's really past (any
        # wall it's past is a step of its fold, so the order changes nothing else)
        inside = np.stack([d[:, 2] + w0, d[:, 1] - d[:, 2] + w1, d[:, 0] - d[:, 1] + w2], axis=1)
        wall = np.argmin(inside, axis=1)
        x, y, z = d[:, 0], d[:, 1], d[:, 2]
        folded = d.copy()
        on = wall == 0
        folded[on, 2] = -2.0 * w0 - z[on]
        on = wall == 1
        folded[on, 1], folded[on, 2] = z[on] - w1, y[on] + w1
        on = wall == 2
        folded[on, 0], folded[on, 1] = y[on] - w2, x[on] + w2
        # A point its reflection doesn't move is on its walls, to rounding: folded
        has = (inside[np.arange(len(d)), wall] < 0.0) & np.any(folded != d, axis=1)
        if not has.any():
            break
        d = np.where(has[:, None], folded, d)
        codes = np.where(has, codes * 4 + wall + 1, codes)
    # Folded, every point is on the face +x: its offset in the domain, along a + i b
    D = d[:, 1] + 1j * d[:, 2]
    to_corner = complex(wall_float(N - a), wall_float(N - b))
    with np.errstate(invalid="ignore", over="ignore"):
        corner = np.abs(D - to_corner) < 0.5 * np.abs(D)
    m = np.zeros((len(d), 3))
    if (~corner).any():
        m[~corner] = charts.anchor.forward_difference(D[~corner])
    if corner.any():
        m[corner] = charts.m_corner + charts.corner.forward_difference(D[corner] - to_corner)
    # Back through each point's reflections, first ones last: m' = σ_1 ... σ_k m + sum_j σ_1 ... σ_(j-1) (-2 s_j ν_j)
    for code in np.unique(codes[codes > 0]):
        walls = []
        c = int(code)
        while c:
            walls.insert(0, c % 4 - 1)
            c //= 4
        S, shift = np.eye(3), np.zeros(3)
        for wall in walls:
            nu = chart.WALLS[wall]
            shift += S @ (-2.0 * s[wall] * nu)
            S = S @ (np.eye(3) - 2.0 * np.outer(nu, nu))
        on = codes == code
        m[on] = m[on] @ S.T + shift
    return (m @ world.G.astype(float)).reshape(np.shape(deltas))


def placed(world, deltas, view_units, lift=0.0):
    """Points A + d of the cube's surface (see surface_offsets()) on the sphere, in map coordinates from the map's
    origin (R lift over the anchor's point), and the sphere's unit normals there: (points, normals)"""
    m = surface_offsets(world, deltas, view_units)
    nA = world.normal()
    normals = nA + np.ldexp(m, -world.n) if world.n < 1070 else np.broadcast_to(nA, m.shape).copy()
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    points = world.rho * m
    if lift:
        points = points - math.ldexp(lift * world.rho, world.n) * nA
    return points, normals


class Below:
    """The point of the sphere below map point p (from the anchor's point), folded into the chart's domain: where it is
    there, from the folded anchor or the corner (exact base, float offset), and the fold's way back. lattice_point()
    rounds it to any level's lattice: one inverse of the chart for all of them (nearest_vertex()'s, split in two)."""

    def __init__(self, world, p):
        m = surface_offset(world, p, world.normal())
        _charts, _mf, R, (D, corner) = inverse_from_guess(world, m, max(1.0, float(np.linalg.norm(m)) * 4.0))
        self.D = complex(D)
        self.N, a, b = world.N, world.A_f[1], world.A_f[2]
        self.base = (self.N, self.N) if corner else (a, b)
        self.unfold = (fold_matrix_exact(R) @ world.G).T

    def lattice_point(self, step):
        """The point of the lattice of step units (a power of 2: an int, or a Fraction under 1) nearest it, exactly,
        as three Fractions on the cube"""
        step = Fraction(step)
        Pa = step * round((Fraction(self.base[0]) + Fraction(self.D.real)) / step)
        Pb = step * round((Fraction(self.base[1]) + Fraction(self.D.imag)) / step)
        # In the domain: N >= Pa >= Pb >= 0 (a lattice point on a wall is on both sides)
        Pa = max(Fraction(0), min(Fraction(self.N), Pa))
        Pb = max(Fraction(0), min(Pa, Pb))
        return [int(row[0]) * self.N + int(row[1]) * Pa + int(row[2]) * Pb for row in self.unfold]


# ----- the motifs: what each level draws, and its cells at the anchor -----

def level_data(world, motif, level):
    """motif.level_data_exact(), kept by the world"""
    key = (motif.name, level)
    data = world._levels.get(key)
    if data is None:
        data = world._levels[key] = motif.level_data_exact(world, level)
    return data


def uniform(world, motif, level, extent, cuts_offset=0):
    """motif.uniform_exact() for the view's extent rounded up to a power of 2 (bigger is safe: a level one color over a
    bigger square is one color over a smaller one), kept by the world. cuts_offset: cut-outs not counted, the ones
    above a planet's handoff (Plan), which only a motif that counts them (cuts()) is asked about."""
    bucket = math.ceil(math.log2(max(extent, 1e-300)))
    key = (motif.name, level, bucket, cuts_offset)
    if key not in world._uniform:
        if len(world._uniform) > 512:
            world._uniform.clear()
        if cuts_offset:
            world._uniform[key] = motif.uniform_exact(world, level, 2.0 ** bucket, cuts_offset=cuts_offset)
        else:
            world._uniform[key] = motif.uniform_exact(world, level, 2.0 ** bucket)
    return world._uniform[key]


def level_cell(world, level):
    """The folded anchor's cell at a level, exactly: per in-face axis (a, b) the cell index k0 and its place r in the
    cell, 0 <= r < 1, in the cells centered on the lattice's corners (shifted half a cell), C = 2N / 2^level units"""
    N, a, b = world.N, world.A_f[1], world.A_f[2]
    C = Fraction(2 * N, 1 << level)
    out = []
    for coord in (a, b):
        q = (coord + N + C / 2) / C
        k0 = math.floor(q)
        out.append((k0, q - k0))
    return C, out


def level_data_exact(world, level):
    """What the GPU needs of a level (see the shader): its cell position at the anchor as three float32 words each for
    fx, fy, fx + fy and fx - fy, the parities of the anchor's cell, and the offset of the cell centered on the cube's
    corner (in cells, or a big number when it's out of reach)"""
    C, ((ka, ra), (kb, rb)) = level_cell(world, level)
    values = []
    for r in (ra, rb, ra + rb, ra - rb):
        values += words(r)
    top = 1 << level
    corner = [float(top - ka) if abs(top - ka) < 2 ** 20 else 1e9, float(top - kb) if abs(top - kb) < 2 ** 20 else 1e9]
    return {"words": values, "parity": (ka % 2, kb % 2, 1 if level == 0 else 0), "corner": corner, "cell": C}


def corner_level_data(level):
    """level_data() measured from the domain's corner (the cube's corner, (N, N)), which is the middle of a cell of
    every level: r = 1/2 along both axes"""
    odd = 1 if level == 0 else 0
    return {"words": [0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "parity": (odd, odd, odd),
            "corner": [0.0, 0.0], "cell": None}


INSIDE, OUTSIDE, PARTLY = "inside", "outside", "partly"


def interval_side(lo, hi):
    """Where an interval of a signed distance (negative inside) is: wholly inside, outside or across 0"""
    if hi < 0:
        return INSIDE
    if lo >= 0:
        return OUTSIDE
    return PARTLY


def uniform_exact(world, level, extent):
    """Whether a level is one color over the view, the square of offsets up to extent units along each in-face axis
    from the folded anchor: None if a piece's edge may pass through it, else the color (sRGB) of the piece covering it,
    or False for none. Exact: the lines' distances from the anchor in Fractions, the view's extent as an interval."""
    C, ((ka, ra), (kb, rb)) = level_cell(world, level)
    e = Fraction(extent) / C
    if e >= Fraction(1, 8):
        return None
    pieces = []

    def side(value, slope):
        """The interval of a linear function of the view's offsets (its value at the anchor, its slope's L1 norm)"""
        return value - slope * e, value + slope * e

    # Gold lines along the cells' edges (fy near 0 or 1), from GAP to 1 - GAP along them
    for along, across in ((ra, rb), (rb, ra)):
        lo, hi = side(min(across, 1 - across) - Fraction(LINE), 1)          # distance outside the band
        band = interval_side(lo, hi)
        lo2, hi2 = side(max(Fraction(GAP) - along, along - (1 - Fraction(GAP))), 1)
        span = interval_side(lo2, hi2)
        if band == OUTSIDE or span == OUTSIDE:
            where = OUTSIDE
        elif band == INSIDE and span == INSIDE:
            where = INSIDE
        else:
            where = PARTLY
        pieces.append((where, GOLD))
    # The star's outline: the band between its inner and outer boundaries (min of square and diamond distances)
    u, w = ra - Fraction(1, 2), rb - Fraction(1, 2)
    h = Fraction(RING) / 2
    sq = max(abs(u), abs(w)) - Fraction(S_HALF)                             # changes by at most e
    di = abs(u) + abs(w) - Fraction(STAR)                                     # by at most 2e
    outer_lo, outer_hi = min(sq - h - e, di - Fraction(RING_DIAGONAL) - 2 * e), min(sq - h + e, di - Fraction(RING_DIAGONAL) + 2 * e)
    inner_lo, inner_hi = min(sq + h - e, di + Fraction(RING_DIAGONAL) - 2 * e), min(sq + h + e, di + Fraction(RING_DIAGONAL) + 2 * e)
    # In the band: inside the outer boundary and outside the inner
    if outer_lo >= 0 or inner_hi < 0:
        star = OUTSIDE
    elif outer_hi < 0 and inner_lo >= 0:
        star = INSIDE
    else:
        star = PARTLY
    parity = (ka % 2 + kb % 2 + (1 if level == 0 else 0)) % 4
    pieces.append((star, SQUARE[parity] if sq < di / math.sqrt(2.0) else TIP[parity]))
    # The sun, in the cell centered on the cube's corner
    top = 1 << level
    if ka == top and kb == top:
        radius = math.hypot(float(u), float(w))
        if SUN_INNER * 0.9 - 2 * float(e) < radius < SUN_OUTER + 2 * float(e):
            pieces.append((PARTLY, SUN))
    if any(where == PARTLY for where, _ in pieces):
        return None
    for where, color in pieces:
        if where == INSIDE:
            return color
    return False


class CubeStars:
    """The cube's star motif (the module's docstring), as a motif for Plan and the shader. A motif has: its name and
    layout; first_level, the coarsest level with anything on it; base, the color under every level (sRGB);
    finest_footprint, the most cells a pixel spans at the finest level drawn; level_data_exact() and
    corner_level_data(), what a level needs from the anchor and from the domain's corner; level_rows(), those as the
    shader's rows; uniform_exact(), whether a level is one color over the view; cuts(), what the shader's count of a
    pixel's cut-outs starts from below a level that covers the view (sphere_sierpinski.Carpet's); paint_source() and
    defines(), its part of the shader: its Level, levelAt(), paintAt(), tileClear(), foldInfo() and tooFine()."""

    name = "stars"
    layout = "cube"
    first_level = 0
    base = FLOOR
    finest_footprint = RING / FINEST_PIXELS

    def level_data_exact(self, world, level):
        return level_data_exact(world, level)

    def corner_level_data(self, world, level):
        return corner_level_data(level)

    def level_rows(self, world, level, data, kappa_anchor, kappa_corner, fine):
        w = data["words"]
        return [(kappa_anchor, kappa_corner, data["parity"][2], 1.0 if fine else 0.0), w[0:4], w[4:8], w[8:12],
                (data["parity"][0], data["parity"][1], data["corner"][0], data["corner"][1])]

    def uniform_exact(self, world, level, extent):
        return uniform_exact(world, level, extent)

    def cuts(self, world, level):
        return 0

    def paint_source(self):
        return STAR_HELPERS + CUBE_PAINT_SOURCE

    def defines(self, info):
        star_defines(info)


CUBE_STARS = CubeStars()


def motif_of(layout, name):
    """A layout's motif by its name (DEFAULTS' if there's none by that name)"""
    motifs = {("cube", "stars"): CUBE_STARS, ("ico", "stars"): sphere_ico.STARS,
              ("cube", sphere_sierpinski.MOTIF): sphere_sierpinski.CARPET,
              ("ico", sphere_sierpinski.MOTIF): sphere_sierpinski.GASKET}
    return motifs.get((layout, name)) or motifs[(layout, DEFAULTS["motif"])]


def newton_steps(ref, mf):
    """The fewest Newton steps the GPU's inverse of the chart's series needs for the folded offsets mf (the view's
    samples, measured from ref): the first guess is good to the square of the view's size relative to its distance from
    the corner, and each step squares that, so deep down views need none"""
    if not len(mf):
        return GPU_NEWTON
    with np.errstate(all="ignore"):
        best = ref.inverse_difference(mf, newton=GPU_NEWTON + 3, terms=GPU_TERMS)
        size = float(np.max(np.abs(best)))
        if not np.isfinite(size) or size == 0.0:
            return GPU_NEWTON
        for steps in range(GPU_NEWTON):
            D = ref.inverse_difference(mf, newton=steps, terms=GPU_TERMS)
            if float(np.max(np.abs(D - best))) <= NEWTON_ERROR * size:
                return steps
    return GPU_NEWTON


class Plan:
    """What a frame draws of a motif (DEFAULTS' if None), from the drone camera's place: the chart's reference at the
    folded anchor, scaled for the view; the walls' distances; the levels drawn per pixel, coarse to fine, each with its
    level_data(); the color under them (the motif's base, or the color of a coarser level covering the view) and the
    shader's count of cut-outs to start from (the motif's cuts()).
    Over a planet (planet.Planet), the motif starts at its handoff level (planet.handoff_level(), handoff its finest
    texels a cell) and the planet is under it where no coarser level covers the view (on_planet); cut-outs are counted
    from the handoff, so the motif starts afresh on the planet whatever its hidden coarser levels did."""

    def __init__(self, world, camera, pixel_angle, clip_end, motif=None, planet=None, handoff=2.0):
        self.world = world
        self.motif = motif = motif or motif_of(world.layout, DEFAULTS["motif"])
        nA = self.normal = world.normal()
        h = float(height(world, camera, nA))
        self.height = h
        R = world.rho * world.N if world.n < 1000 else math.inf
        # How far the view reaches over the sphere: to the horizon or the clip end
        horizon = math.sqrt(max(h, 0.0) * (2.0 * R + max(h, 0.0))) if math.isfinite(R) else math.inf
        reach = max(h, min(horizon, clip_end))
        # Sample the view around the point below the camera, out to its reach: offsets in units, and units per map unit
        below = camera + h * down(world, camera, nA)
        e1 = np.cross(nA, (1.0, 0.0, 0.0) if abs(nA[0]) < 0.9 else (0.0, 1.0, 0.0))
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(nA, e1)
        radii = np.array([reach * 4.0 ** -k for k in range(0, 20) if reach * 4.0 ** -k > h * 1e-2])
        angles = np.linspace(0.0, 2.0 * math.pi, 6, endpoint=False)
        ring = np.cos(angles)[:, None] * e1 + np.sin(angles)[:, None] * e2
        points = np.concatenate([below[None], (below + radii[:, None, None] * ring[None]).reshape(-1, 3)])
        m = surface_offset(world, points, nA)
        guess = max(1e-300, float(np.max(np.linalg.norm(m, axis=-1))))
        charts, mf, _R, (D, J, corner) = inverse_from_guess(world, m, guess, jacobian=True, estimate=True)
        sample_scale = charts.anchor.scale
        # The view's extent from the anchor (corner points measured from the corner, so moved by the anchor's offset
        # from it), and the part near the corner from the corner
        if corner.any():
            from_anchor = np.where(corner, D + world.to_corner(), D)
            corner_extent = float(np.max(np.abs(D[corner]))) * 1.25 + 1e-300
        else:
            from_anchor = D
            corner_extent = None
        extent = float(np.max(np.abs(from_anchor))) * 1.25 + 1e-300
        self.charts = charts = world.charts(extent, corner_extent)
        self.newton = (newton_steps(charts.anchor, mf[~corner]),
                       newton_steps(charts.corner, mf[corner] - charts.m_corner) if corner.any() else GPU_NEWTON)
        self.reference = ref = charts.anchor
        self.walls = ref.wall_distances()
        self.extent = extent
        # The finest level: the least footprint in units, a pixel's angle times the distance, times units per map
        # unit there (the chart's stretch: dD = J dm / scale, and dm = dmap / rho near the point)
        distance = np.linalg.norm(points - camera, axis=-1)
        stretch = np.linalg.norm(np.abs(J), axis=-1) / (sample_scale * world.rho) / math.sqrt(2.0)
        # At the cube's corner itself the chart's stretch is infinite (cells pinch to nothing): no sample there
        footprint = pixel_angle * distance * stretch
        footprint = footprint[np.isfinite(footprint) & (footprint > 0.0)]
        if len(footprint):
            finest = math.floor(world.n + world.level_shift + math.log2(motif.finest_footprint / float(np.min(footprint))))
        else:
            finest = world.n + world.level_shift
        finest = max(motif.first_level, finest)
        self.finest = finest
        # Levels with cells under FINE_CELLS times the finest's are drawn in plain float32 (paintFine() in the shader)
        self.fine_from = finest - FINE_LEVELS
        first, offset = motif.first_level, 0
        if planet is not None:
            first = max(first, planets.handoff_level(world, motif, planet.pyramid, handoff))
            offset = motif.cuts(world, first - 1) % 2 if first > motif.first_level else 0
        self.first = first
        # Coarse to fine: walk up from the finest until a level covers the view
        base, cuts, per_pixel, on_planet = motif.base, 0, [], planet is not None
        for level in range(finest, first - 1, -1):
            C = world.cell(level)
            if C >= COARSE * Fraction(extent):
                u = uniform(world, motif, level, extent, offset)
                if u is None:
                    per_pixel.append(level)
                elif u is not False:
                    base, cuts, on_planet = u, (motif.cuts(world, level) - offset) % 2, False
                    break
            else:
                per_pixel.append(level)
            if len(per_pixel) >= MAX_LEVELS:
                break
        self.base = base
        self.base_cuts = cuts
        self.on_planet = on_planet
        self.levels = list(reversed(per_pixel))
        self.data = [level_data(world, motif, level) for level in self.levels]
        self.corner_data = [motif.corner_level_data(world, level) for level in self.levels]


# ----- drawing -----

# Terms of Rančić's series the GPU sums: in the domain |Z| <= 1/4, so term 14 is below float32's rounding
GPU_TERMS = 14
# The most Newton steps the GPU takes inverting the series, and how close the fewest it takes must come to many more
# over the view (relative to the view's extent: float32's rounding is 2^-24)
GPU_NEWTON = 3
NEWTON_ERROR = 2.0 ** -32
# The level buffer's rows of 4 per level (see level_buffer(); the cube's take 5)
LEVEL_ROWS = sphere_ico.LEVEL_ROWS
# The tile pass's tiles, TILE x TILE pixels of the main pass (two of the GPU's SIMD groups, which then walk the same
# levels), and how much more a pixel's footprint may be than at its tile's middle, past what the corners show
TILE = 8
TILE_SPREAD = 1.25


def srgb_to_linear(rgb):
    """An sRGB color, 0 to 255, as linear floats: the view's output is color managed, so it takes linear colors"""
    def channel(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return tuple(channel(c) for c in rgb)


def glsl_vec3(rgb):
    return "vec3(%r, %r, %r)" % srgb_to_linear(rgb)


VERTEX_SOURCE = """
void main()
{
    ndc = pos;
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""

SHADER_HEAD = """
vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
// a / b with both scaled by b's larger part first: deep down next to a domain's corner both are tiny, and their parts'
// products fell below float32's normal range, which the GPU flushes to 0 (the icosahedron's corners came out a hole)
vec2 cdiv(vec2 a, vec2 b)
{
    float s = max(abs(b.x), abs(b.y));
    a /= s;
    b /= s;
    return vec2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / dot(b, b);
}
vec2 csq(vec2 a) { return cmul(a, a); }

float seriesA(int k)
{
    SERIES_TABLE
}

// dW/dZ for absolute Z
vec2 seriesSlope(vec2 Z)
{
    vec2 dw = vec2(0.0);
    for (int k = TERMS - 1; k >= 1; k--) {
        dw = cmul(dw, Z) + vec2(float(k) * seriesA(k), 0.0);
    }
    return dw;
}

// (W(Z1) - W(Z0)) / (Z1 - Z0) for absolute Z0, Z1, with no cancellation
vec2 seriesDivided(vec2 Z0, vec2 Z1)
{
    vec2 total = vec2(seriesA(1), 0.0);
    vec2 divided = vec2(1.0, 0.0);
    vec2 power0 = vec2(1.0, 0.0);
    for (int k = 2; k < TERMS; k++) {
        power0 = cmul(power0, Z0);
        divided = cmul(divided, Z1) + power0;
        total += seriesA(k) * divided;
    }
    return total;
}

// One of the chart's references (sphere_chart.Reference), from the uniform buffer
struct Ref {
    vec2 z0, Z0, v0, w0;
    vec3 n0;
    float s0, t, r, rho1, sigma, scale;
    int newton;
};

Ref anchorRef()
{
    Ref f;
    f.z0 = uRefA0.xy; f.Z0 = uRefA0.zw; f.v0 = uRefA1.xy; f.w0 = uRefA1.zw; f.n0 = uRefA2.xyz; f.s0 = uRefA2.w;
    f.t = uRefA3.x; f.r = uRefA3.y; f.rho1 = uRefA3.z; f.sigma = uRefA3.w; f.scale = uRefA4.x;
    f.newton = int(uRefA4.y + 0.5);
    return f;
}

Ref cornerRef()
{
    Ref f;
    f.z0 = uRefC0.xy; f.Z0 = uRefC0.zw; f.v0 = uRefC1.xy; f.w0 = uRefC1.zw; f.n0 = uRefC2.xyz; f.s0 = uRefC2.w;
    f.t = uRefC3.x; f.r = uRefC3.y; f.rho1 = uRefC3.z; f.sigma = uRefC3.w; f.scale = uRefC4.x;
    f.newton = int(uRefC4.y + 0.5);
    return f;
}

"""

# The cube's chart backward (sphere_ico.CHART_SOURCE is the icosahedron's)
CUBE_CHART_SOURCE = """
// z with z^4 = Z in the domain's sector, continued past its uWalls
vec2 fourthRoot(vec2 Z)
{
    float a = atan(Z.y, Z.x);
    if (a <= HALF_PI) {
        a += 2.0 * PI;
    }
    // |Z| from Z scaled by its larger part: length() squares Z's parts, below float32's within 3e-5 of a view's size
    // from the cube's corner (see sphere_ico's sectorRoot())
    float big = max(abs(Z.x), abs(Z.y));
    float r = big > 0.0 ? sqrt(sqrt(big)) * sqrt(sqrt(length(Z / big))) : 0.0;
    return r * vec2(cos(0.25 * a), sin(0.25 * a));
}

// The chart backward (sphere_chart.Reference.inverse_difference()): dz~ for a folded offset m from the reference, and
// d(dz~)/dm along two pixel steps dm1, dm2
vec2 inverseDifference(Ref f, vec3 m, vec3 dm1, vec3 dm2, out vec2 g1, out vec2 g2)
{
    vec3 dn = m / f.scale;
    float X0 = f.n0.y, Y0 = f.n0.z, Z0n = f.n0.x;
    vec2 dw = (vec2(dn.y, dn.z) * (1.0 + Z0n) - vec2(X0, Y0) * dn.x) / ((1.0 + Z0n + f.sigma * dn.x) * (1.0 + Z0n));
    vec2 w1 = f.w0 + f.sigma * dw;
    vec2 du = cdiv(cmul(MOBIUS_DET, dw), cmul(vec2(1.0, 0.0) - cmul(CC, w1), vec2(1.0, 0.0) - cmul(CC, f.w0)));
    vec2 dv = cdiv(du, I13);
    vec2 v1 = f.v0 + f.rho1 * dv;
    vec2 dW = cmul(vec2(0.0, -1.0), cmul(dv, csq(v1) + cmul(v1, f.v0) + csq(f.v0)));
    vec2 Z0abs = f.t * f.Z0;
    vec2 dZ = cdiv(dW, seriesSlope(Z0abs));
    for (int i = 0; i < f.newton; i++) {
        vec2 Z1 = f.Z0 + f.rho1 * dZ;
        dZ -= cdiv(cmul(dZ, seriesDivided(Z0abs, f.t * Z1)) - dW, seriesSlope(f.t * Z1));
    }
    vec2 Z1 = f.Z0 + f.rho1 * dZ;
    vec2 z1 = fourthRoot(Z1);
    vec2 q = cmul(z1 + f.z0, csq(z1) + csq(f.z0));
    vec2 dz = dot(q, q) > 0.0 ? cdiv(dZ, q) : (z1 - f.z0) / max(f.rho1, 1.0e-30);
    // The pixel's steps: the chain's complex derivative, then the stereographic projection's
    vec3 n1 = f.n0 + f.sigma * dn;
    vec2 zz = dot(z1, z1) > 0.0 ? z1 : vec2(1.0e-30, 0.0);
    vec2 chain = cdiv(cmul(vec2(0.0, -3.0), csq(v1)), cmul(cmul(4.0 * cmul(csq(zz), zz), seriesSlope(f.t * Z1)), I13));
    vec2 oneMinus = vec2(1.0, 0.0) - cmul(CC, w1);
    chain = cdiv(cmul(chain, MOBIUS_DET), csq(oneMinus));
    float iz = 1.0 / ((1.0 + n1.x) * (1.0 + n1.x));
    // dw = ((dX + i dY)(1 + Z) - (X + i Y) dZ) / (1 + Z)^2, with n = (Z, X, Y)
    vec2 a1 = (vec2(dm1.y, dm1.z) * (1.0 + n1.x) - vec2(n1.y, n1.z) * dm1.x) * iz / f.scale;
    vec2 a2 = (vec2(dm2.y, dm2.z) * (1.0 + n1.x) - vec2(n1.y, n1.z) * dm2.x) * iz / f.scale;
    g1 = cmul(chain, a1);
    g2 = cmul(chain, a2);
    return dz;
}

"""

SHADER_HELPERS = """
float clip01(float x) { return clamp(x, 0.0, 1.0); }

float band(float loEdge, float hiEdge, float F, float halfWidth)
{
    F = max(F, 1.0e-30);
    return clip01((min(0.5 * F, loEdge) + min(0.5 * F, hiEdge)) / F) * clip01(2.0 * halfWidth / F);
}

// band() and spanCover() for F already kept from 0, and its reciprocal
float bandScaled(float loEdge, float hiEdge, float F, float invF, float halfWidth)
{
    return clip01((min(0.5 * F, loEdge) + min(0.5 * F, hiEdge)) * invF) * clip01(2.0 * halfWidth * invF);
}

float spanScaled(float low, float high, float F, float invF)
{
    return clip01((min(0.5 * F, low) + min(0.5 * F, high)) * invF);
}

float spanCover(float low, float high, float F)
{
    F = max(F, 1.0e-30);
    return clip01((min(0.5 * F, low) + min(0.5 * F, high)) / F);
}

// cellOffset() and lineAt() take hi - (k + f), exact, through OPAQUE_ONE (a uniform 1) before adding the pixel's
// offset d: Metal compiles with fast math, free to make (hi - (k + f)) + d into (hi + d) - (k + f), which rounds d to
// hi's precision (2^-24 of a cell). On a coarse level whose line crosses a view a few millionths of a cell wide, that
// smeared the line over a hundred pixels, and an anchor move shifted it (seen 2026-11-10 on the gasket).

// The pixel's cell from the reference's (an integer), exactly near the cells' edges
float cellOffset(vec3 w, float d)
{
    float k = floor(w.x + d + w.y + w.z);
    float x0 = ((((w.x - k) * OPAQUE_ONE) + d) + w.y) + w.z;
    if (x0 < 0.0) { k -= 1.0; }
    else if (x0 >= 1.0) { k += 1.0; }
    return k;
}

// The pixel's own cell's coordinate minus a line's f: hi - (k + f) is exact for f with few bits, then the rest
float lineAt(vec3 w, float k, float f, float d)
{
    return ((((w.x - (k + f)) * OPAQUE_ONE) + d) + w.y) + w.z;
}

"""

# What both layouts' star motifs share
STAR_HELPERS = """
vec3 squareColor(int p)
{
    return p == 0 ? SQUARE0 : p == 1 ? SQUARE1 : p == 2 ? SQUARE2 : SQUARE3;
}

vec3 tipColor(int p)
{
    return p == 0 ? TIP0 : p == 1 ? TIP1 : p == 2 ? TIP2 : TIP3;
}

// Whether a level is too fine to draw for a pixel of footprint F (its cells): its star's outline under FINEST_PIXELS
bool tooFine(float F)
{
    return RING < FINEST_PIXELS * F;
}

"""

# The cube's star motif, levels and fold colors (sphere_ico.IcoStars's is the icosahedron's)
CUBE_PAINT_SOURCE = """
float starDistance(float u, float w)
{
    float sector = 2.0 * PI / SUN_POINTS;
    float rho = length(vec2(u, w));
    float phi = mod(atan(w, u), sector);
    phi = abs(phi - 0.5 * sector);
    vec2 p = rho * vec2(cos(phi), sin(phi));
    vec2 a = vec2(SUN_INNER, 0.0);
    vec2 b = SUN_OUTER * vec2(cos(0.5 * sector), sin(0.5 * sector));
    vec2 e = b - a;
    float t = clamp(dot(p - a, e) / dot(e, e), 0.0, 1.0);
    return length(p - a - t * e);
}

// Whether a pixel at (fx, fy) in its cell, F its footprint across the cell's axes (the larger), is so far from every
// line of the motif (but the sun's) that it covers none of it: a gold line's band is HALF_LINE either side of it, the
// star's outline is a square's and a diamond's, HALF_RING either side (RING_DIAGONAL / sqrt 2 across the diamond's
// sides, the same to 1%), and a pixel covers a band's share of its footprint, F / 2 across the cell's axes and up to
// F / sqrt 2 across the diagonals. off: how far off fx and fy may be.
bool clearOfLines(float fx, float fy, float F, float off)
{
    float u = abs(fx - 0.5), w = abs(fy - 0.5);
    float edge = min(min(fx, 1.0 - fx), min(fy, 1.0 - fy));
    float star = min(max(u, w) - S_HALF, (u + w - STAR) * SQRT_HALF);
    return edge >= LINE + 0.5 * F + off && abs(star) >= STAR_HALF_WIDTH + F + 2.0 * off;
}

// Whether a pixel at (fx, fy) in its cell, off as in clearOfLines(), is so deep inside one of the motif's bands that it
// covers the pixel whole (paintLevel() and paintFine() would mix its color in at cover 1): 1 a gold line's, 2 the
// star's outline where it's the square's, 3 where it's the diamond's, 0 if that can't be said
int deepInside(float fx, float fy, float F, float off)
{
    float m1 = 0.5 * F + off;                 // across the cell's axes
    float m2 = SQRT_HALF * F + 2.0 * off;     // across the diagonals (the diamond's sides)
    // Gold lines: inside one (along its span) and clear of the other
    float ey = min(fy, 1.0 - fy), ex = min(fx, 1.0 - fx);
    bool inH = ey <= LINE - m1 && fx >= GAP + m1 && fx <= 1.0 - GAP - m1;
    bool outH = ey >= LINE + m1 || fx <= GAP - m1 || fx >= 1.0 - GAP + m1;
    bool inV = ex <= LINE - m1 && fy >= GAP + m1 && fy <= 1.0 - GAP - m1;
    bool outV = ex >= LINE + m1 || fy <= GAP - m1 || fy >= 1.0 - GAP + m1;
    if (2.0 * LINE >= F && (inH || inV) && (inH || outH) && (inV || outV)) {
        return 1;
    }
    if (!(outH && outV)) {
        return 0;
    }
    // The star's outline: inside its outer boundary (the square's or the diamond's) by a pixel's half and outside its
    // inner (both) by one, so its cover is 1; colored by whichever is nearer, which has to be clear
    float u = abs(fx - 0.5), w = abs(fy - 0.5);
    float sqC = max(u, w) - S_HALF;
    float diC = (u + w - STAR) * SQRT_HALF;
    bool inOuter = sqC - HALF_RING <= -m1 || diC - RING_DIAGONAL * SQRT_HALF <= -m2;
    bool outInner = sqC + HALF_RING >= m1 && diC + RING_DIAGONAL * SQRT_HALF >= m2;
    if (2.0 * HALF_RING >= F && inOuter && outInner && abs(sqC - diC) > 3.0 * off) {
        return sqC < diC ? 2 : 3;
    }
    return 0;
}

// Whether a pixel at (u, w) from the middle of the cube corner's cell is so far from its sun's outline that it covers
// none of it: the outline lies between SUN_INNER and SUN_OUTER from the middle, HALF_RING either side
bool clearOfSun(float u, float w, float F, float off)
{
    float rho = length(vec2(u, w));
    float margin = HALF_RING + 0.5 * F + 2.0 * off;
    return rho < SUN_INNER - margin || rho > SUN_OUTER + margin;
}

// One level (sphere.paint_level in the tests): d the pixel's offset from the reference in this level's cells, ga, gb its
// gradients along the in-face axes per pixel step
vec3 paintLevel(vec3 color, vec3 wa, vec3 wb, vec3 ws, vec3 wd, vec3 parity, vec2 cornerCell, vec2 d, vec2 ga,
                vec2 gb, int parity0)
{
    float Fx = abs(ga.x) + abs(ga.y);
    float Fy = abs(gb.x) + abs(gb.y);
    {
        // Most pixels of a coarse level are nowhere near its lines: a rough place in the cell tells, good to float32's
        // rounding of the pixel's offset (and the chart's, far below ROUGH), and the exact lines are only worked out
        // near them
        float xa = (wa.x + d.x) + wa.y, xb = (wb.x + d.y) + wb.y;
        float fx = xa - floor(xa), fy = xb - floor(xb);
        bool sunCell = floor(xa) == cornerCell.x && floor(xb) == cornerCell.y;
        float off = ROUGH * (1.0 + abs(d.x) + abs(d.y));
        float F = max(Fx, Fy);
        bool sunClear = !sunCell || clearOfSun(fx - 0.5, fy - 0.5, F, off);
        if (!sunClear) {
            // Near the sun: its outline's distance from the rough place, which is off by no more than the place
            float sunDistance = starDistance(fx - 0.5, fy - 0.5);
            if (sunDistance <= HALF_RING - 0.5 * F - 2.0 * off && 2.0 * HALF_RING >= F) {
                return mix(color, SUN, 1.0);
            }
            sunClear = sunDistance >= HALF_RING + 0.5 * F + 2.0 * off;
        }
        if (sunClear && clearOfLines(fx, fy, F, off)) {
            return color;
        }
        // And most of the rest are deep inside a band, which covers them whole: its color, as the exact path below
        // would mix it in, cover 1 (a coarse level's band is many pixels wide, so only its edges need the exact lines)
        int inside = sunClear ? deepInside(fx, fy, F, off) : 0;
        if (inside == 1) {
            return mix(color, GOLD, 1.0);
        }
        if (inside >= 2) {
            int p = int(mod(parity.x + floor(xa), 2.0) + mod(parity.y + floor(xb), 2.0) + parity.z + 0.5);
            if (parity.z > 0.5) {
                p = parity0;
            }
            p = p % 4;
            return mix(color, inside == 2 ? squareColor(p) : tipColor(p), 1.0);
        }
    }
    float Fs = (abs(ga.x + gb.x) + abs(ga.y + gb.y)) * SQRT_HALF;
    float Fm = (abs(ga.x - gb.x) + abs(ga.y - gb.y)) * SQRT_HALF;
    float ka = cellOffset(wa, d.x);
    float kb = cellOffset(wb, d.y);
    float ks = ka + kb, kd = ka - kb;
    float ds = d.x + d.y, dd = d.x - d.y;
    // Gold lines along the nearest of the cell's edges, from GAP to 1 - GAP along it
    float x0 = lineAt(wa, ka, 0.0, d.x), y0 = lineAt(wb, kb, 0.0, d.y);
    float loH = y0 < 0.5 ? lineAt(wb, kb, -LINE, d.y) : lineAt(wb, kb, 1.0 - LINE, d.y);
    float hiH = y0 < 0.5 ? -lineAt(wb, kb, LINE, d.y) : -lineAt(wb, kb, 1.0 + LINE, d.y);
    float loV = x0 < 0.5 ? lineAt(wa, ka, -LINE, d.x) : lineAt(wa, ka, 1.0 - LINE, d.x);
    float hiV = x0 < 0.5 ? -lineAt(wa, ka, LINE, d.x) : -lineAt(wa, ka, 1.0 + LINE, d.x);
    float coverH = band(loH, hiH, Fy, LINE) * spanCover(lineAt(wa, ka, GAP, d.x), -lineAt(wa, ka, 1.0 - GAP, d.x), Fx);
    float coverV = band(loV, hiV, Fx, LINE) * spanCover(lineAt(wb, kb, GAP, d.y), -lineAt(wb, kb, 1.0 - GAP, d.y), Fy);
    color = mix(color, GOLD, max(coverH, coverV));
    // The star's outline, between its outer and inner boundaries, each a square and a diamond
    vec4 sidesC = vec4(lineAt(wa, ka, 0.5 + S_HALF, d.x), -lineAt(wa, ka, 0.5 - S_HALF, d.x),
                       lineAt(wb, kb, 0.5 + S_HALF, d.y), -lineAt(wb, kb, 0.5 - S_HALF, d.y));
    float sqC = max(max(sidesC.x, sidesC.y), max(sidesC.z, sidesC.w));
    float FSq = max(sidesC.x, sidesC.y) >= max(sidesC.z, sidesC.w) ? Fx : Fy;
    vec4 diagC = vec4(lineAt(ws, ks, 1.0 + STAR, ds), -lineAt(ws, ks, 1.0 - STAR, ds),
                      lineAt(wd, kd, STAR, dd), -lineAt(wd, kd, -STAR, dd));
    float diC = max(max(diagC.x, diagC.y), max(diagC.z, diagC.w)) * SQRT_HALF;
    float FDi = max(diagC.x, diagC.y) >= max(diagC.z, diagC.w) ? Fs : Fm;
    float sqOut = max(max(lineAt(wa, ka, 0.5 + S_HALF + HALF_RING, d.x), -lineAt(wa, ka, 0.5 - S_HALF - HALF_RING, d.x)),
                      max(lineAt(wb, kb, 0.5 + S_HALF + HALF_RING, d.y), -lineAt(wb, kb, 0.5 - S_HALF - HALF_RING, d.y)));
    float sqIn = max(max(lineAt(wa, ka, 0.5 + S_HALF - HALF_RING, d.x), -lineAt(wa, ka, 0.5 - S_HALF + HALF_RING, d.x)),
                     max(lineAt(wb, kb, 0.5 + S_HALF - HALF_RING, d.y), -lineAt(wb, kb, 0.5 - S_HALF + HALF_RING, d.y)));
    float diOut = max(max(lineAt(ws, ks, 1.0 + STAR + RING_DIAGONAL, ds), -lineAt(ws, ks, 1.0 - STAR - RING_DIAGONAL, ds)),
                      max(lineAt(wd, kd, STAR + RING_DIAGONAL, dd), -lineAt(wd, kd, -STAR - RING_DIAGONAL, dd))) * SQRT_HALF;
    float diIn = max(max(lineAt(ws, ks, 1.0 + STAR - RING_DIAGONAL, ds), -lineAt(ws, ks, 1.0 - STAR + RING_DIAGONAL, ds)),
                     max(lineAt(wd, kd, STAR - RING_DIAGONAL, dd), -lineAt(wd, kd, -STAR + RING_DIAGONAL, dd))) * SQRT_HALF;
    float Fq = max(FSq, 1.0e-30), Fd = max(FDi, 1.0e-30);
    float dOut = min(sqOut / Fq, diOut / Fd);
    float dIn = min(sqIn / Fq, diIn / Fd);
    float cover = max(0.0, clip01(0.5 - dOut) - clip01(0.5 - dIn));
    int p = int(mod(parity.x + ka, 2.0) + mod(parity.y + kb, 2.0) + parity.z + 0.5);
    if (parity.z > 0.5) {
        p = parity0;
    }
    p = p % 4;
    color = mix(color, sqC < diC ? squareColor(p) : tipColor(p), cover);
    // The sun, in the cell centered on the cube's corner
    if (ka == cornerCell.x && kb == cornerCell.y) {
        float su = lineAt(wa, ka, 0.5, d.x), sw = lineAt(wb, kb, 0.5, d.y);
        if (!clearOfSun(su, sw, max(Fx, Fy), 0.0)) {
            float dist = starDistance(su, sw);
            color = mix(color, SUN, band(dist + HALF_RING, HALF_RING - dist, max(Fx, Fy), HALF_RING));
        }
    }
    return color;
}

// paintLevel() for a level whose cells are small next to the view (FINE_LEVEL): the pixel's place in its cell in plain
// float32 is good to far below its footprint there, so every band comes from its center line and half-width
vec3 paintFine(vec3 color, vec3 wa, vec3 wb, vec3 parity, vec2 cornerCell, vec2 d, vec2 ga, vec2 gb, int parity0)
{
    float Fx = abs(ga.x) + abs(ga.y);
    float Fy = abs(gb.x) + abs(gb.y);
    float F = max(Fx, Fy);
    float xa = (wa.x + d.x) + (wa.y + wa.z), xb = (wb.x + d.y) + (wb.y + wb.z);
    float ka = floor(xa), kb = floor(xb);
    float fx = xa - ka, fy = xb - kb;
    bool sunCell = ka == cornerCell.x && kb == cornerCell.y;
    bool sunClear = !sunCell || clearOfSun(fx - 0.5, fy - 0.5, F, 0.0);
    // Each part only near it (clearOfLines(), part by part): most of a pixel's painted levels are near one of them
    float u = fx - 0.5, w = fy - 0.5;
    float au = abs(u), aw = abs(w);
    float ey = min(fy, 1.0 - fy), ex = min(fx, 1.0 - fx);
    float sqC = max(au, aw) - S_HALF;
    float diC = (au + aw - STAR) * SQRT_HALF;
    bool nearGold = min(ex, ey) < LINE + 0.5 * F;
    bool nearStar = abs(min(sqC, diC)) < STAR_HALF_WIDTH + F;
    if (sunClear && !nearGold && !nearStar) {
        return color;
    }
    float Fxs = max(Fx, 1.0e-30), Fys = max(Fy, 1.0e-30);
    float ix = 1.0 / Fxs, iy = 1.0 / Fys;
    if (nearGold) {
        // Gold lines along the nearest of the cell's edges, from GAP to 1 - GAP along it
        float coverH = bandScaled(ey + LINE, LINE - ey, Fys, iy, LINE) * spanScaled(fx - GAP, 1.0 - GAP - fx, Fxs, ix);
        float coverV = bandScaled(ex + LINE, LINE - ex, Fxs, ix, LINE) * spanScaled(fy - GAP, 1.0 - GAP - fy, Fys, iy);
        color = mix(color, GOLD, max(coverH, coverV));
    }
    if (nearStar) {
        // The star's outline
        float Fs = (abs(ga.x + gb.x) + abs(ga.y + gb.y)) * SQRT_HALF;
        float Fm = (abs(ga.x - gb.x) + abs(ga.y - gb.y)) * SQRT_HALF;
        float iq = au >= aw ? ix : iy;
        float id = 1.0 / max(u * w >= 0.0 ? Fs : Fm, 1.0e-30);
        float dOut = min((sqC - HALF_RING) * iq, (diC - RING_DIAGONAL * SQRT_HALF) * id);
        float dIn = min((sqC + HALF_RING) * iq, (diC + RING_DIAGONAL * SQRT_HALF) * id);
        float cover = max(0.0, clip01(0.5 - dOut) - clip01(0.5 - dIn));
        int p = int(mod(parity.x + ka, 2.0) + mod(parity.y + kb, 2.0) + parity.z + 0.5);
        if (parity.z > 0.5) {
            p = parity0;
        }
        p = p % 4;
        color = mix(color, sqC < diC ? squareColor(p) : tipColor(p), cover);
    }
    if (!sunClear) {
        float dist = starDistance(u, w);
        color = mix(color, SUN, band(dist + HALF_RING, HALF_RING - dist, F, HALF_RING));
    }
    return color;
}

// Level i's rows of the level buffer (level_buffer()), measured from the anchor or from the cube's corner
struct Level {
    vec4 t0;
    vec3 wa, wb, ws, wd, par;
    vec2 cornerCell;
    float toCorner;     // how far the anchor's cell is from the cube corner's, in cells along both axes
};

Level levelAt(int i, bool useCorner)
{
    Level L;
    L.t0 = levelValues.rows[LEVEL_ROWS * i];
    vec4 t4 = levelValues.rows[LEVEL_ROWS * i + 4];
    L.toCorner = abs(t4.z) + abs(t4.w);
    if (useCorner) {
        L.wa = vec3(0.5, 0.0, 0.0); L.wb = L.wa; L.ws = vec3(1.0, 0.0, 0.0); L.wd = vec3(0.0);
        L.par = vec3(L.t0.z); L.cornerCell = vec2(0.0);
    }
    else {
        vec4 t1 = levelValues.rows[LEVEL_ROWS * i + 1];
        vec4 t2 = levelValues.rows[LEVEL_ROWS * i + 2];
        vec4 t3 = levelValues.rows[LEVEL_ROWS * i + 3];
        L.wa = t1.xyz; L.wb = vec3(t1.w, t2.xy); L.ws = vec3(t2.zw, t3.x); L.wd = t3.yzw;
        L.par = vec3(t4.xy, L.t0.z); L.cornerCell = t4.zw;
    }
    return L;
}

// One level painted over color, the pixel d from the reference in its cells (along the in-face axes), ga and gb d's
// steps per pixel; info: foldInfo()'s; cuts: the count the Sierpinski motifs keep, not the stars
vec3 paintAt(vec3 color, Level L, vec2 d, vec2 ga, vec2 gb, vec4 info, inout float cuts)
{
    if (L.t0.w > 0.5) {
        return paintFine(color, L.wa, L.wb, L.par, L.cornerCell, d, ga, gb, int(info.x));
    }
    return paintLevel(color, L.wa, L.wb, L.ws, L.wd, L.par, L.cornerCell, d, ga, gb, int(info.x));
}

// tileMain()'s test of a level: whether every pixel of the tile, whose places in the level's cells are within r of d,
// its middle's, and footprints at most F, covers none of the level. Each pixel's rough place is rounded to its own
// offset (as is the middle's), and a pixel measured from the other reference is as far from the anchor as the cube's
// corner is.
bool tileClear(Level L, vec2 d, float F, float r, bool useCorner)
{
    float far = abs(d.x) + abs(d.y) + 2.0 * r + (useCorner ? L.toCorner : 0.0);
    float off = 2.0 * ROUGH * (2.0 + far) + r;
    float xa = (L.wa.x + d.x) + (L.wa.y + L.wa.z), xb = (L.wb.x + d.y) + (L.wb.y + L.wb.z);
    vec2 fromSun = vec2(xa, xb) - (L.cornerCell + 0.5);
    bool nearSun = abs(fromSun.x) < 0.5 + off && abs(fromSun.y) < 0.5 + off
                   && !clearOfSun(fromSun.x, fromSun.y, F, off);
    return !nearSun && clearOfLines(xa - floor(xa), xb - floor(xb), F, off);
}

// What the colors need of a pixel's fold F (folded = F unfolded): the level-0 colors, how many of the unfolded cube
// corner's coordinates are positive
vec4 foldInfo(mat3 F)
{
    vec3 cornerOrig = transpose(F) * vec3(1.0);
    return vec4(float(int(cornerOrig.x > 0.0) + int(cornerOrig.y > 0.0) + int(cornerOrig.z > 0.0)), 0.0, 0.0, 0.0);
}

"""

SHADER_MAIN = """
// The sphere where the ray through clip coordinates at meets it (map coordinates, from the camera, uCamera): the ray's
// direction and the distance along it, or false if it misses (the nearer root as in ray_hit())
bool rayHit(vec2 at, out vec3 dir, out float t)
{
    dir = mat3(uRayMatrix) * vec3(at, 1.0);
    vec3 c = uCamera.xyz;
    float k = uCamera.w;
    vec3 nA = uNormal.xyz;
    float a = k * dot(dir, dir);
    float b = k * dot(dir, c) + dot(dir, nA);
    float q = k * dot(c, c) + 2.0 * dot(c, nA);
    float disc = b * b - a * q;
    t = 0.0;
    if (disc < 0.0 || b >= 0.0 || q <= 0.0) {
        return false;
    }
    t = q / (-b + sqrt(disc));
    return t > 0.0;
}

// N (n(p) - n(A)) for the point t along dir (surface_offset()), its steps per pixel along the surface, and the sphere's
// normal there
void surfaceOffset(vec3 dir, float t, out vec3 m, out vec3 dm1, out vec3 dm2, out vec3 nn)
{
    vec3 c = uCamera.xyz;
    float k = uCamera.w;
    vec3 nA = uNormal.xyz;
    float rho = uNormal.w;
    vec3 p = c + t * dir;
    float pn = dot(p, nA);
    float L = sqrt(max(0.0, 1.0 + k * (2.0 * pn + k * dot(p, p))));
    float lift = (2.0 * pn + k * dot(p, p)) / (1.0 + L);
    m = (p - lift * nA) / (rho * L);
    vec3 nu = k * p + nA;
    nn = nu / length(nu);
    vec3 dr1 = mat3(uRayMatrix)[0] * uPixelSize.x;
    vec3 dr2 = mat3(uRayMatrix)[1] * uPixelSize.y;
    vec3 dp1 = t * (dr1 - dir * (dot(nu, dr1) / dot(nu, dir)));
    vec3 dp2 = t * (dr2 - dir * (dot(nu, dr2) / dot(nu, dir)));
    dm1 = (dp1 - nn * dot(nn, dp1)) / (rho * L);
    dm2 = (dp2 - nn * dot(nn, dp2)) / (rho * L);
}

// m folded into the chart's domain (sphere_chart.fold_offsets()), the anchor first, then across the domain's walls, and
// measured from the anchor, or from the domain's corner when nearer it (sphere_chart.Charts): dz~ and its steps per
// pixel g1, g2, which reference, what the colors need of the fold (foldInfo()) and the folded m
vec2 chartOf(vec3 m, vec3 dm1, vec3 dm2, out vec2 g1, out vec2 g2, out bool useCorner, out vec4 info, out vec3 mf)
{
    mat3 F = mat3(uFold);
    m = F * m; dm1 = F * dm1; dm2 = F * dm2;
    for (int i = 0; i < FOLD_STEPS; i++) {
        // The wall it's furthest past, until its reflection doesn't move it (sphere_chart.fold_offsets())
        vec3 dWall = uWalls.xyz + vec3(dot(m, WALL0), dot(m, WALL1), dot(m, WALL2));
        vec3 nu2 = WALL0;
        float dk = dWall.x;
        if (dWall.y < dk) { nu2 = WALL1; dk = dWall.y; }
        if (dWall.z < dk) { nu2 = WALL2; dk = dWall.z; }
        vec3 folded = m - 2.0 * dk * nu2;
        if (!(dk < 0.0) || (folded.x == m.x && folded.y == m.y && folded.z == m.z)) { break; }
        m = folded;
        dm1 -= 2.0 * dot(dm1, nu2) * nu2;
        dm2 -= 2.0 * dot(dm2, nu2) * nu2;
        F = (mat3(1.0) - 2.0 * mat3(nu2.x * nu2, nu2.y * nu2, nu2.z * nu2)) * F;
    }
    info = foldInfo(F);
    mf = m;
    vec3 fromCorner = m - uCornerOffset.xyz;
    useCorner = dot(fromCorner, fromCorner) < 0.25 * dot(m, m);
    return useCorner ? inverseDifference(cornerRef(), fromCorner, dm1, dm2, g1, g2)
                     : inverseDifference(anchorRef(), m, dm1, dm2, g1, g2);
}

#if MASK
// The tile pass: for each tile of TILE x TILE pixels of the main pass, which levels may have anything of the motif in
// it (the motif's tileClear()), so the main pass walks only those. A level walked past its early-out costs a pixel as
// much as painting one, and most pixels walk twenty or more levels to paint three, so this is most of the main pass's
// time saved. Conservative: worked out at the tile's middle, every pixel's place in a level's cells is within r of it
// (folds never move points apart, but a fold across a wall can turn a's spread into b's, so r is both), its footprint
// within SPREAD of the middle's; where that can't be said (the limb, rays that turn too fast across the tile, the cube's
// corner where the chart's stretch grows without bound) every level is walked. Bits: 16 levels in each of r, g, b.
void tileMain()
{
    vec2 size = uTiles.xy;
    vec2 tile = floor((ndc * 0.5 + 0.5) * uTiles.zw);
    vec2 at = ((tile + 0.5) * TILE) / size * 2.0 - 1.0;
    vec3 every = vec3(65535.0);
    vec3 dir, cornerDir[4];
    float t, cornerT[4];
    bool middle = rayHit(at, dir, t);
    int hits = 0;
    for (int j = 0; j < 4; j++) {
        vec2 corner = vec2(float(j & 1), float(j >> 1)) * 2.0 - 1.0;
        if (rayHit(at + corner * (0.5 * TILE) * uPixelSize.xy, cornerDir[j], cornerT[j])) {
            hits++;
        }
    }
    if (hits == 0 && !middle) {
        fragColor = vec4(0.0);
        return;
    }
    if (hits < 4 || !middle) {
        fragColor = vec4(every, 1.0);
        return;
    }
    vec3 m, dm1, dm2, nn;
    surfaceOffset(dir, t, m, dm1, dm2, nn);
    // How much the tile's corners stray from the middle's linear steps, and their steps from the middle's
    float stretch = 1.0;
    for (int j = 0; j < 4; j++) {
        vec2 corner = vec2(float(j & 1), float(j >> 1)) * 2.0 - 1.0;
        vec3 mj, dm1j, dm2j, nnj;
        surfaceOffset(cornerDir[j], cornerT[j], mj, dm1j, dm2j, nnj);
        vec3 linear = (0.5 * TILE) * (corner.x * dm1 + corner.y * dm2);
        if (length(mj - m - linear) > 0.25 * length(linear)) {
            fragColor = vec4(every, 1.0);
            return;
        }
        stretch = max(stretch, max(length(dm1j) / length(dm1), length(dm2j) / length(dm2)));
    }
    if (stretch > 1.5) {
        fragColor = vec4(every, 1.0);
        return;
    }
    vec2 g1, g2;
    bool useCorner;
    vec4 info;
    vec3 mf;
    vec2 dz = chartOf(m, dm1, dm2, g1, g2, useCorner, info, mf);
    float reach = (0.5 * TILE) * (length(dm1) + length(dm2)) * stretch;
    vec3 toCorner = mf - uCornerOffset.xyz;
    if (dot(toCorner, toCorner) < 16.0 * reach * reach) {
        fragColor = vec4(every, 1.0);
        return;
    }
    float spread = SPREAD * stretch;
    vec3 bits = vec3(0.0);
    int levels = int(uBase.w);
    for (int i = 0; i < levels; i++) {
        Level L = levelAt(i, useCorner);
        float kappa = useCorner ? L.t0.y : L.t0.x;
        vec2 d = kappa * dz;
        float Fx = abs(kappa) * (abs(g1.x) + abs(g2.x));
        float Fy = abs(kappa) * (abs(g1.y) + abs(g2.y));
        // Every pixel of the tile is past its finest level (the main pass's break)
        if (tooFine(max(Fx, Fy) / spread)) {
            break;
        }
        float r = (0.5 * TILE) * (Fx + Fy) * spread;
        if (!tileClear(L, d, max(Fx, Fy) * spread, r, useCorner)) {
            bits[i >> 4] += exp2(float(i & 15));
        }
    }
    fragColor = vec4(bits, 1.0);
}
#endif

void main()
{
#if MASK
    tileMain();
#else
    // The pixel's ray, in map coordinates (uRayMatrix: from the view's clip coordinates, without the camera's place)
    vec3 dir;
    float t;
    if (!rayHit(ndc, dir, t)) {
        discard;
    }
    vec3 worldPoint = mat3(uWorldFromMap) * (t * dir);
    vec4 clip = uViewProjection * vec4(worldPoint, 1.0);
    float depth = clip.z / clip.w * 0.5 + 0.5;
    if (depth > 1.0) {
        discard;
    }
#if WRITE_DEPTH
    // Under 3D tiles, behind everything (uMotif.z): the sphere only fills where no tile is drawn yet
    gl_FragDepth = uMotif.z > 0.5 ? max(depth, BEHIND_DEPTH) : depth;
#endif
#if COMPOSITE
    // The sphere worked out at a lower resolution, scaled up by hand: the offscreen texture
    // is sampled nearest. Its alpha is 1 where the sphere was hit, so the limb's texels next to sky are left out.
    ivec2 size = textureSize(lowColor, 0);
    vec2 at = (ndc * 0.5 + 0.5) * vec2(size) - 0.5;
    ivec2 i0 = ivec2(floor(at));
    vec2 f = at - floor(at);
    ivec2 top = size - 1;
    vec4 c00 = texelFetch(lowColor, clamp(i0, ivec2(0), top), 0);
    vec4 c10 = texelFetch(lowColor, clamp(i0 + ivec2(1, 0), ivec2(0), top), 0);
    vec4 c01 = texelFetch(lowColor, clamp(i0 + ivec2(0, 1), ivec2(0), top), 0);
    vec4 c11 = texelFetch(lowColor, clamp(i0 + ivec2(1, 1), ivec2(0), top), 0);
    vec4 mixed = mix(mix(c00 * vec4(vec3(c00.a), 1.0), c10 * vec4(vec3(c10.a), 1.0), f.x),
                     mix(c01 * vec4(vec3(c01.a), 1.0), c11 * vec4(vec3(c11.a), 1.0), f.x), f.y);
    fragColor = vec4(mixed.a > 0.0 ? mixed.rgb / mixed.a : uFog.rgb, 1.0);
#else
    vec3 m, dm1, dm2, nn;
    surfaceOffset(dir, t, m, dm1, dm2, nn);
    vec2 g1, g2;
    bool useCorner;
    vec4 info;
    vec3 mf;
    vec2 dz = chartOf(m, dm1, dm2, g1, g2, useCorner, info, mf);

    // The levels the pixel's tile may have lines of (tileMain())
    ivec2 tile = ivec2(floor((ndc * 0.5 + 0.5) * uTiles.xy / TILE));
    uvec3 bits = uvec3(texelFetch(tileMask, clamp(tile, ivec2(0), ivec2(uTiles.zw) - 1), 0).rgb + 0.5);
#if PLANET
    vec3 color = uPlanet.x > 0.5 ? planetColor(nn, dm1, dm2) : uBase.rgb;
#else
    vec3 color = uBase.rgb;
#endif
    // The times the pixel's been cut out above the levels drawn (a Sierpinski motif's count, Plan.base_cuts): a level
    // the tile pass leaves out cuts out none of its pixels
    float cuts = uMotif.x;
    int levels = int(uBase.w);
    for (int i = 0; i < levels; i++) {
        if (((bits[i >> 4] >> uint(i & 15)) & 1u) == 0u) {
            continue;
        }
        Level L = levelAt(i, useCorner);
        float kappa = useCorner ? L.t0.y : L.t0.x;
        vec2 d = kappa * dz;
        vec2 h1 = kappa * g1, h2 = kappa * g2;
        vec2 ga = vec2(h1.x, h2.x), gb = vec2(h1.y, h2.y);
        float Fmax = max(abs(ga.x) + abs(ga.y), abs(gb.x) + abs(gb.y));
        if (tooFine(Fmax)) {
            break;
        }
        color = paintAt(color, L, d, ga, gb, info, cuts);
    }

    // Sunlight, so the globe reads as round, then the fog (tiles3d.FOG_SOURCE)
    color *= 0.62 + 0.38 * max(0.0, dot(nn, uSun.xyz));
    fragColor = vec4(fogged(color, worldPoint), 1.0);
#endif
#endif
}
"""


def fragment_source(motif):
    """The sphere's fragment shader for a motif: the shared head, helpers and main, its layout's chart, the motif and
    the planet (planet.py; only the passes that paint compile it)"""
    chart_source = sphere_ico.CHART_SOURCE if motif.layout == sphere_ico.LAYOUT else CUBE_CHART_SOURCE
    return (SHADER_HEAD + chart_source + SHADER_HELPERS + motif.paint_source()
            + "#if PLANET\n" + planets.PLANET_SOURCE + "#endif\n" + tiles3d.FOG_SOURCE + SHADER_MAIN)


# Tests can replace the cube stars' fragment shader whole
FRAGMENT_SOURCE = fragment_source(CUBE_STARS)

# The uniform buffer, rows of 4 (std140) after the matrices
SPHERE_MATRICES = ("uRayMatrix", "uWorldFromMap", "uViewProjection", "uFold")
SPHERE_ROWS = ("uCamera", "uNormal", "uWalls", "uCornerOffset", "uRefA0", "uRefA1", "uRefA2", "uRefA3", "uRefA4",
               "uRefC0", "uRefC1", "uRefC2", "uRefC3", "uRefC4", "uBase", "uFog", "uAir", "uUp", "uHaze",
               "uClearAir", "uSun", "uPixelSize", "uTiles", "uMotif", "uPlanet", "uPlanetGrid")


def star_defines(info):
    """Both layouts' star motifs' shader defines"""
    for name in ("STAR", "S_HALF", "LINE", "GAP", "RING", "RING_DIAGONAL", "SUN_OUTER", "SUN_INNER", "FINEST_PIXELS"):
        info.define(name, repr(float(globals()[name])))
    info.define("HALF_RING", repr(RING / 2))
    info.define("STAR_HALF_WIDTH", repr(max(RING / 2, RING_DIAGONAL * math.sqrt(0.5))))
    info.define("SUN_POINTS", repr(float(SUN_POINTS)))
    for p in range(4):
        info.define(f"SQUARE{p}", glsl_vec3(SQUARE[p]))
        info.define(f"TIP{p}", glsl_vec3(TIP[p]))
    info.define("GOLD", glsl_vec3(GOLD))
    info.define("SUN", glsl_vec3(SUN))


def cube_chart_defines(info):
    """The cube's chart's and fold's shader defines (sphere_ico.chart_defines() are the icosahedron's)"""
    table = " ".join(f"if (k == {k}) return {chart.A_RANCIC[k]!r};" for k in range(1, GPU_TERMS)) + " return 0.0;"
    info.define("SERIES_TABLE", table)
    info.define("TERMS", str(GPU_TERMS))
    info.define("MOBIUS_DET", "vec2(%r, %r)" % (chart.MOBIUS_DET.real, chart.MOBIUS_DET.imag))
    info.define("CC", "vec2(%r, %r)" % (chart.CC.real, chart.CC.imag))
    info.define("I13", "vec2(%r, %r)" % (chart.I13.real, chart.I13.imag))
    for i, wall in enumerate(chart.WALLS):
        info.define(f"WALL{i}", "vec3(%r, %r, %r)" % tuple(float(c) for c in wall))
    info.define("FOLD_STEPS", str(chart.FOLD_STEPS))


def make_shader(composite=False, write_depth=True, tiles=False, motif=CUBE_STARS):
    """The sphere's shader for a motif (on its layout, the cube or the icosahedron), or with composite, the pass that
    scales a lower resolution one up to the view (the same ray and depth, the color from the texture lowColor), or with
    tiles, the pass before either that says which levels each tile of pixels may have anything of (tileMain()).
    Without write_depth, for an offscreen, which has no depth buffer."""
    import gpu
    info = gpu.types.GPUShaderCreateInfo()
    info.define("COMPOSITE", "1" if composite else "0")
    info.define("MASK", "1" if tiles else "0")
    info.define("PLANET", "0" if composite or tiles else "1")
    planets.defines(info)
    info.define("WRITE_DEPTH", "1" if write_depth and not tiles else "0")
    info.define("TILE", repr(float(TILE)))
    info.define("SPREAD", repr(TILE_SPREAD))
    if motif.layout == sphere_ico.LAYOUT:
        sphere_ico.chart_defines(info)
    else:
        cube_chart_defines(info)
    motif.defines(info)
    info.define("PI", repr(math.pi))
    info.define("HALF_PI", repr(math.pi / 2))
    info.define("SQRT_HALF", repr(math.sqrt(0.5)))
    # How far off a rough place in a cell may be, per cell of the pixel's offset from the anchor (and one)
    info.define("ROUGH", repr(2.0 ** -12))
    info.define("OPAQUE_ONE", "uMotif.y")
    info.define("EXTINCTION", "vec3(%r, %r, %r)" % tiles3d.EXTINCTION)
    info.define("BEHIND_DEPTH", "0.9999995")
    # The struct's members are numbered, and each name a define for its member: a define named like a member would
    # rename the member too
    rows = (" ".join(f"mat4 m{i};" for i in range(len(SPHERE_MATRICES))) + " "
            + " ".join(f"vec4 r{i};" for i in range(len(SPHERE_ROWS))))
    # The levels' data in a uniform buffer too, not a texture: a level's fetches from a texture stalled every pixel
    # until they came, level after level (four times the cost of the rest of the shader)
    info.define("LEVEL_ROWS", str(LEVEL_ROWS))
    info.typedef_source(f"struct SphereValues {{ {rows} }};\n"
                        f"struct LevelValues {{ vec4 rows[{LEVEL_ROWS * MAX_LEVELS}]; }};")
    info.uniform_buf(0, "SphereValues", "sphereValues")
    info.uniform_buf(1, "LevelValues", "levelValues")
    for i, name in enumerate(SPHERE_MATRICES):
        info.define(name, f"sphereValues.m{i}")
    for i, name in enumerate(SPHERE_ROWS):
        info.define(name, f"sphereValues.r{i}")
    if composite:
        info.sampler(0, 'FLOAT_2D', "lowColor")
    elif not tiles:
        info.sampler(0, 'FLOAT_2D', "tileMask")
        info.sampler(1, 'FLOAT_2D', "planetAtlas")
        info.sampler(2, 'FLOAT_2D', "planetPages")
    info.vertex_in(0, 'VEC2', "pos")
    interface = gpu.types.GPUStageInterfaceInfo("sphere_interface")
    interface.smooth('VEC2', "ndc")
    info.vertex_out(interface)
    info.fragment_out(0, 'VEC4', "fragColor")
    if write_depth and not tiles:
        info.depth_write('ANY')
    info.vertex_source(VERTEX_SOURCE)
    info.fragment_source(FRAGMENT_SOURCE if motif is CUBE_STARS else fragment_source(motif))
    return gpu.shader.create_from_info(info)


def reference_rows(ref, newton=GPU_NEWTON):
    """A sphere_chart.Reference's values for the shader, five rows of 4, and the Newton steps its inverse takes (an
    IcoReference's xi~, T~ and w~ where the cube's z~, Z~ and v~ go)"""
    if isinstance(ref, chart.IcoReference):
        return [(ref.xi0.real, ref.xi0.imag, ref.T0.real, ref.T0.imag),
                (ref.w0t.real, ref.w0t.imag, ref.w0.real, ref.w0.imag),
                (ref.n0[0], ref.n0[1], ref.n0[2], ref.s0), (ref.t, ref.r, ref.rho1, ref.sigma),
                (ref.scale, float(newton), 0.0, 0.0)]
    return [(ref.z0.real, ref.z0.imag, ref.Z0.real, ref.Z0.imag), (ref.v0.real, ref.v0.imag, ref.w0.real, ref.w0.imag),
            (ref.n0[0], ref.n0[1], ref.n0[2], ref.s0), (ref.t, ref.r, ref.rho1, ref.sigma),
            (ref.scale, float(newton), 0.0, 0.0)]


def clamp32(x):
    """A float that fits a float32 (signs kept, overflow to +-1e30)"""
    return max(-1.0e30, min(1.0e30, float(x)))


def values_buffer(plan, camera, ray_matrix, world_from_map, view_projection, pixel_size, fog, sun_direction, tiles,
                  planet_rows=None, behind=False):
    """The shader's uniform buffer for a frame (see SPHERE_MATRICES, SPHERE_ROWS); fog: its rows (fog_rows());
    tiles: the main pass's size and the tile pass's; planet_rows: uPlanet and uPlanetGrid (planet.Planet.uniforms()), none without a planet; behind: the
    sphere's depth behind everything, under 3D tiles"""
    import gpu
    world = plan.world
    data = np.zeros(16 * len(SPHERE_MATRICES) + 4 * len(SPHERE_ROWS), np.float32)

    def put_matrix(index, matrix):
        m4 = np.eye(4)
        m = np.asarray(matrix, dtype=float)
        m4[:m.shape[0], :m.shape[1]] = m
        data[16 * index:16 * index + 16] = m4.T.ravel()          # GLSL keeps matrices by columns

    put_matrix(0, ray_matrix)
    put_matrix(1, world_from_map)
    put_matrix(2, view_projection)
    put_matrix(3, world.G.astype(float))
    rows = {
        "uCamera": (*camera, world.kappa),
        "uNormal": (*plan.normal, world.rho),
        "uWalls": (*[clamp32(s) for s in plan.walls], 0.0),
        "uCornerOffset": (*[clamp32(s) for s in plan.charts.m_corner], 0.0),
        "uBase": (*srgb_to_linear(plan.base), float(len(plan.levels))),
        "uFog": fog[0],
        "uAir": fog[1],
        "uUp": fog[2],
        "uHaze": fog[3],
        "uClearAir": fog[4],
        "uSun": (*sun_direction, 0.0),
        "uPixelSize": (*pixel_size, 0.0, 0.0),
        "uTiles": tuple(float(v) for v in tiles),
        # The count of cut-outs above the levels drawn, OPAQUE_ONE (cellOffset()), and the depth behind everything
        "uMotif": (float(plan.base_cuts), 1.0, 1.0 if behind else 0.0, 0.0),
        "uPlanet": planet_rows[0] if planet_rows else (0.0, 0.0, 0.0, 0.0),
        "uPlanetGrid": planet_rows[1] if planet_rows else (1.0, 1.0, 1.0, 1.0),
    }
    for i, row in enumerate(reference_rows(plan.charts.anchor, plan.newton[0])):
        rows[f"uRefA{i}"] = row
    for i, row in enumerate(reference_rows(plan.charts.corner, plan.newton[1])):
        rows[f"uRefC{i}"] = row
    start = 16 * len(SPHERE_MATRICES)
    for i, name in enumerate(SPHERE_ROWS):
        data[start + 4 * i:start + 4 * i + 4] = [clamp32(v) for v in rows[name]]
    return gpu.types.GPUUniformBuf(data)


def level_buffer(plan):
    """Each level's data for the shader, up to LEVEL_ROWS rows of 4 a level (the motif's level_rows()): first kappa,
    the reference's dz~ to the level's cells, from the anchor and from the corner (signed: the cube's chart measures
    from the face's corner, against the axes), then the motif's"""
    import gpu
    data = np.zeros((MAX_LEVELS, LEVEL_ROWS, 4), np.float32)
    anchor, corner = plan.charts.anchor, plan.charts.corner
    sign = plan.world.chart_sign
    for i, (level, d) in enumerate(zip(plan.levels, plan.data)):
        kappas = (sign * math.ldexp(1.0, anchor.e_delta + level), sign * math.ldexp(1.0, corner.e_delta + level))
        rows = plan.motif.level_rows(plan.world, level, d, *kappas, level >= plan.fine_from)
        for k, row in enumerate(rows):
            data[i, k] = [clamp32(v) for v in row]
    return gpu.types.GPUUniformBuf(data.ravel())


# ----- the map's object -----

_worlds = {}         # object name -> World, the one loops and moves change and drawing reads
_floor = None        # the Floor drawing (its shader and cached plans)


def is_sphere(obj):
    return obj is not None and hasattr(obj, "get") and obj.get(STATE) is not None and obj.get(PROPERTY) is not None


def planet_of(obj):
    """The sphere's planet (its "planet" setting's pyramid, planet.planet()), or None"""
    folder = setting(obj, "planet")
    if not folder:
        return None
    try:
        import bpy
        folder = bpy.path.abspath(folder)
    except ImportError:
        pass
    return planets.planet(folder)


def tiles_of(obj):
    """The sphere's 3D tiles (its "tiles" setting's tileset, tiles3d.tileset()), or None"""
    source = setting(obj, "tiles")
    if not source:
        return None
    if source != tiles3d.GOOGLE and not tiles3d.is_web(source):
        try:
            import bpy
            source = bpy.path.abspath(source)
        except ImportError:
            pass
    return tiles3d.tileset(source)


def lift_of(obj):
    """The map's origin's height over the sphere, a share of R (the "sphere_lift" property; 0 without tiles)"""
    return float(obj.get(LIFT, 0.0))


def lifted(world, p, lift, normal=None):
    """Map points p (from the map's origin) from the anchor's point on the sphere instead, as the sphere's own
    functions take them: the origin is R lift over it"""
    if not lift:
        return p
    nA = world.normal() if normal is None else normal
    return np.asarray(p, dtype=float) + math.ldexp(lift * world.rho, world.n) * nA


def map_scale(world):
    """Map units a meter on a planet of tiles3d.RADIUS meters (R = rho N map units)"""
    return math.ldexp(world.rho / tiles3d.RADIUS, world.n)


def meters_log2(obj):
    """log2 of the meters a world unit is on obj's Earth (a sphere with a planet or 3D tiles: R = rho N map units are
    tiles3d.RADIUS meters), or None for a sphere that isn't the Earth. In logs, since N = 2^n has no float past 2^1024
    and the map units' world size can be anything."""
    if not (setting(obj, "planet") or setting(obj, "tiles")):
        return None
    world = world_of(obj)
    world_per_map_unit = float(np.linalg.norm(world_vector(obj, (1.0, 0.0, 0.0))))
    if world_per_map_unit <= 0.0:
        return None
    return math.log2(tiles3d.RADIUS / world.rho) - world.n - math.log2(world_per_map_unit)


# The Earth's air, two layers thinning as e^(-h / H) (the user, 2026-11-16, asked whether flying the Alps was still
# true with one layer thin enough to see the globe clearly through: it wasn't, peaks 50 km off were 5% fogged where
# real air fogs them 45 to 85%):
# - the clear air itself (Rayleigh scattering, what makes the sky blue): its extinction at sea level, per meter, red to
#   blue, and its scale height. Over the haze it's what blues and fades far mountains.
# - haze (aerosols: dust, water): the rest of the scene's fog at sea level (fog_distance meters, the fog's EXTINCTION
#   per channel, which was made from the two together), thinning by HAZE_HEIGHT (the scene's "fog_height", if it has
#   one): mostly within the lowest kilometer or two.
# Straight down from space, the ground is seen through blue's optical depth 0.5 (haze 0.25, air 0.26): 60% of its blue
# gets through, as in raw satellite photos.
CLEAR_AIR = (5.8e-6, 13.5e-6, 33.1e-6)
CLEAR_AIR_HEIGHT = 8000.0
HAZE_HEIGHT = 1200.0


def air_view(obj, world, camera):
    """On the Earth (meters_log2()), where a view from camera (map coordinates from the anchor's point) is in its air:
    the camera's height over the sphere and its distance from the center (world units), its way up (world space), and
    world units a meter; else None"""
    meters = meters_log2(obj)
    if meters is None:
        return None
    nA = world.normal()
    unit = float(np.linalg.norm(world_vector(obj, (1.0, 0.0, 0.0))))
    height_world = float(height(world, camera, nA)) * unit
    up = world_vector(obj, -down(world, camera, nA))
    up = up / np.linalg.norm(up)
    try:
        per_meter = 2.0 ** -meters
        center = math.ldexp(world.rho, world.n) * unit + height_world
    except OverflowError:
        per_meter = center = math.inf
    return height_world, center, up, per_meter


def fog_rows(scene, obj, world, camera, drone_scale, pixels_per_tangent):
    """The fog's shader rows (tiles3d.fog_rows()) for a view from camera (map coordinates from the anchor's point, as
    the sphere's own functions take them), pixels_per_tangent across its middle (for how far the 3D tiles' heights are
    true: their error on screen, "tiles_error", over it, per unit of distance). On the Earth (meters_log2()), the
    Earth's own air, whatever the drone's scale (CLEAR_AIR): the scene's fog_distance is meters at sea level, thinning
    with height, so the fog doesn't draw back as the drone grows, and climbing out to see the globe, it's seen through
    the air's thin shell. On any other sphere, fog_distance drone sizes, as on every map."""
    color = tuple(scene.get("fog_color", (0.114, 0.352, 0.913)))[:3]
    distance = float(scene.get("fog_distance", 23000.0))
    air = air_view(obj, world, camera)
    if air is None:
        return tiles3d.fog_rows(color, distance * drone_scale)
    height_world, center, up, per_meter = air
    # The clear air never more than the whole fog, so a fog_distance past the clear air's own still thins it all away
    fog = [1.0 / (distance * e) for e in tiles3d.EXTINCTION]
    clear_air = [min(clear, total) for clear, total in zip(CLEAR_AIR, fog)]
    haze = [total - clear for total, clear in zip(fog, clear_air)]
    error = float(setting(obj, "tiles_error")) / pixels_per_tangent
    return tiles3d.fog_rows(color, distance * per_meter, (
        height_world, center, tuple(float(c) for c in up), error,
        ([x / per_meter for x in haze], float(scene.get("fog_height", HAZE_HEIGHT)) * per_meter),
        ([x / per_meter for x in clear_air], CLEAR_AIR_HEIGHT * per_meter)))


def origin_meters(world, lift):
    """The map's origin from the planet's center, meters (the tiles' coordinates, tiles3d.py)"""
    return tiles3d.RADIUS * (1.0 + lift) * world.normal()


def ground_height(obj, world, p, tiles=None):
    """How far map point p is above the ground of the tiles drawn last, straight down (map units; negative under it),
    or None where no tile is drawn under it"""
    tiles = tiles_of(obj) if tiles is None else tiles
    if tiles is None or not tiles.last_draw:
        return None
    # A flight tick asks several times from the same place (auto scale, tidal lock, the horizon, the recenter)
    key = (obj.name, world.text(), lift_of(obj), tuple(np.asarray(p, dtype=float).tolist()), id(tiles.last_draw),
           len(tiles.tiles))
    if key in _grounds:
        return _grounds[key]
    s = map_scale(world)
    point = origin_meters(world, lift_of(obj)) + np.asarray(p, dtype=float) / s
    # The same place as last time, to a rounding (GROUND_SAME of its height): the tick's moves between its questions
    # (a scaling, a turn, the map recentered or looped under the drone) leave the drone where it is, but not p. Over a
    # hole in the drawn tiles, each question then searched every drawn tile's box before the stand-ins'.
    last = _ground_last.get(obj.name)
    state = (tiles.last_draw, tiles.last_holes, tiles.drops, len(tiles.tiles))
    if (last is not None and last[2] is not None and all(a is b for a, b in zip(last[0][:2], state[:2]))
            and last[0][2:] == state[2:] and float(np.sum((point - last[1]) ** 2)) < (GROUND_SAME * last[2]) ** 2):
        t = last[2]
    else:
        down_m = -point / np.linalg.norm(point)
        t = tiles.ground(tiles.last_draw, point, down_m)
        if t is None and tiles.last_holes:
            # In a hole (a tile not in yet): the ground of the tile standing in for it, which is what's drawn there
            t = tiles.hole_ground(tiles.last_holes, point, down_m)
        if t is None:
            # Off the selection's ground (flown off it since): the finest tile loaded there. None, and auto scale
            # measured to the sphere, kilometers below the Alps' ground: the drone grew 13 times in a tick.
            t = tiles.loaded_ground(point, down_m)
        _ground_last[obj.name] = (state, point, t)
    if len(_grounds) > 32:
        _grounds.clear()
    _grounds[key] = h = None if t is None else t * s
    return h


_grounds = {}        # ground_height()'s last answers
# object name -> ground_height()'s last search: (the tiles' lists and counts), the point (meters), the way down to the
# ground (meters, signed); and how near it a point is the same place, as a share of that
_ground_last = {}
GROUND_SAME = 1.0e-6


def surface_distance(obj, world, p, tiles=None):
    """How far map point p is from the nearest point of the ground of the tiles drawn last (map units): no further
    than the ground straight down (ground_height()), negative under it, None where no tile is drawn under it. What
    the floor (floor_move()), the Altitude readout (ground_vector()) and the ground the tiles select around the drone
    measure: straight down, the ground jumps at every cliff's edge (beside the Matterhorn's face, from millimeters to
    the glacier hundreds of meters below; auto scale, when it measured to the ground, grew the drone 512 times in a
    tick).
    The search costs a few milliseconds high over mountains, so it's made again only once the point has moved
    NEAREST_MOVE of the distance or NEAREST_AGE seconds have passed; until then, the distance is to the nearest point
    found last (a second-order error: exact while it stays the nearest), never more than straight down."""
    import time
    tiles = tiles_of(obj) if tiles is None else tiles
    ground = ground_height(obj, world, p, tiles)
    if ground is None or ground <= 0.0:
        return ground
    s = map_scale(world)
    lift = lift_of(obj)
    point = origin_meters(world, lift) + np.asarray(p, dtype=float) / s
    state = (world.text(), lift)
    last = _nearest.get(obj.name)
    now = time.perf_counter()
    if (last is not None and last[0] == state and now - last[1] < NEAREST_AGE
            and np.linalg.norm(point - last[2]) < NEAREST_MOVE * last[3]):
        return min(ground, float(np.linalg.norm(point - last[4])) * s) if last[4] is not None else ground
    # No further than the nearest point found last, if it's still there: beside a cliff, high over the valley floor,
    # the search within the distance straight down measured thousands of triangles
    d, nearest = math.inf, None
    if last is not None and last[4] is not None:
        d, nearest = tiles.nearest_ground(tiles.last_draw, point, float(np.linalg.norm(point - last[4])) * (1.0 + 1e-9))
    if nearest is None:
        d, nearest = tiles.nearest_ground(tiles.last_draw, point, ground / s)
    d = min(d, ground / s)
    _nearest[obj.name] = (state, now, point, d, nearest)
    return min(ground, d * s)


# object name -> surface_distance()'s last search: (world, lift), time, point, distance, nearest point (meters)
_nearest = {}
# surface_distance() searches again once the point has moved this share of the distance, or after this many seconds
NEAREST_MOVE = 0.05
NEAREST_AGE = 0.1
# The draw takes the flight's measure for the drone's distance to the ground while it's no older than this (seconds)
NEAREST_FRESH = 0.5


def setting(obj, name):
    settings = obj.get(PROPERTY)
    value = settings.get(name) if settings is not None else None
    return DEFAULTS[name] if value is None else value


def world_of(obj):
    """obj's World, read again if its properties changed under it (an undo)"""
    cached = _worlds.get(obj.name)
    version = obj.get(VERSION, 0)
    if cached is None or cached[0] != version:
        world = World.from_text(obj[STATE], float(setting(obj, "rho")))
        _worlds[obj.name] = cached = (version, world)
    return cached[1]


def save(obj, world, lift=None):
    obj[STATE] = world.text()
    if lift is not None:
        obj[LIFT] = float(lift)
    version = obj.get(VERSION, 0) + 1
    obj[VERSION] = version
    _worlds[obj.name] = (version, world)


def sphere_objects(scene):
    return [obj for obj in scene.objects if is_sphere(obj)]


def to_map(obj):
    """(origin, M, M inverse): world point w is map point M_inv (w - origin), as float64 arrays. A flight tick asks
    some fifteen times, the object's matrix the same each time: its inverse is kept for it."""
    matrix = np.array(obj.matrix_world, dtype=np.float64)
    last = _to_map.get(obj.name)
    if last is None or not np.array_equal(last[0], matrix):
        linear = matrix[:3, :3]
        last = _to_map[obj.name] = (matrix, (matrix[:3, 3], linear, np.linalg.inv(linear)))
    return last[1]


_to_map = {}          # object name -> to_map()'s last: the object's matrix, and what it gave


def map_point(obj, world_point):
    origin, _linear, inverse = to_map(obj)
    return inverse @ (np.asarray(world_point, dtype=np.float64) - origin)


def world_vector(obj, map_vector):
    _origin, linear, _inverse = to_map(obj)
    return linear @ np.asarray(map_vector, dtype=np.float64)


def map_vector(obj, world_vec):
    _origin, _linear, inverse = to_map(obj)
    return inverse @ np.asarray(world_vec, dtype=np.float64)


# ----- what the flight asks (ER_OT_Fly) -----

def surface_vector(obj, camera_world):
    """From a camera (world space) to the nearest point of the sphere straight below it (world space), without
    cancellation: what auto scale measures to, with or without its Mesh setting (the map's object's origin is the
    anchor's point, which moves, and the sphere's center can be any distance away), and the way down for tidal lock and
    the horizon. With 3D tiles, the drone's height alone, over the valley it's in (base_height()): measured to their
    nearest ground (ground_vector()), the drone slowed down for every mountain it came near, and crawled over their
    tops, where the user wanted to skim them fast, slow down only into the valleys, and launch off the slopes like
    ramps. It doesn't jump, either, as a finer tile's ground comes in or a cliff's edge goes by."""
    world = world_of(obj)
    p = map_point(obj, camera_world)
    nA = world.normal()
    q = lifted(world, p, lift_of(obj), nA)
    h = base_height(obj, world, p) if setting(obj, "tiles") else float(height(world, q, nA))
    return world_vector(obj, h * down(world, q, nA))


def base_height(obj, world, p):
    """How far map point p is over the valley level (base_of()), in map units, at least BASE_LEAST meters: the height
    auto scale measures with 3D tiles (surface_vector())"""
    nA = world.normal()
    s = map_scale(world)
    h = float(height(world, lifted(world, p, lift_of(obj), nA), nA))
    return max(h - base_of(obj) * s, BASE_LEAST * s)


# base_height()'s least, meters: where dives end, the drone held a floor over the valley's floor (floor_of()) as for
# this height (about a meter up, at the usual drone scale for its height: where the tiles' detail is still sharp), and
# a drone under the valley level (in a coarse tile's sag) isn't measured through it
BASE_LEAST = 10.0


def base_of(obj):
    """The valley level auto scale measures the drone's height over (follow_base()), meters over sea level: tiles_base
    until the drone has flown"""
    kept = _bases.get(obj.name)
    return kept[0] if kept is not None else float(setting(obj, "tiles_base"))


def follow_base(obj, camera_world):
    """Once a flight tick (ER_OT_Fly), the valley level under a drone camera at camera_world (world space), on a
    sphere with 3D tiles: the lowest ground around it, within the "tiles_valley" setting's share of its height over
    tiles_base (valley_ground()), never under tiles_base nor over the ground straight below the drone. On a ridge that
    takes in the valleys beside it, so the drone's height is over the valley, not the ridge, and over a city it's the
    streets' level. Looked for at a point or a few a tick (VALLEY_PER_TICK: all of them in one tick took 5 ms, a missed
    frame every fifth of a second), a round of them giving the level to ease toward over BASE_EASE seconds, so a lower
    valley come into reach grows the drone smoothly; dropped to the ground straight below at once, though, so flown off
    an edge whose lower ground wasn't in reach yet, the drone isn't measured as under the valley level and shrunk to
    BASE_LEAST."""
    if not setting(obj, "tiles"):
        return
    lowest = float(setting(obj, "tiles_base"))
    share = float(setting(obj, "tiles_valley"))
    tiles = tiles_of(obj)
    now = time.perf_counter()
    if share <= 0.0 or tiles is None or not tiles.tiles:
        _bases[obj.name] = [lowest, lowest, None, 0, now]
        return
    world = world_of(obj)
    s = map_scale(world)
    p = map_point(obj, camera_world)
    point = origin_meters(world, lift_of(obj)) + p / s
    altitude = float(np.linalg.norm(point)) - tiles3d.RADIUS
    ground = ground_height(obj, world, p, tiles)
    below = altitude - ground / s if ground is not None else None
    reach = share * max(altitude - lowest, BASE_LEAST)
    kept = _bases.get(obj.name)
    count = len(VALLEY_RINGS) * VALLEY_RAYS
    if kept is None or now - kept[4] > BASE_STALE:
        # A flight starting, or one long after the last: the level where the drone is now, all at once, not eased from
        # where it was
        found = valley_ground(tiles, point, reach, range(count), lazy=False)
        lows = [h for h in (found, below) if h is not None]
        base = target = max(lowest, min(lows)) if lows else lowest
        _bases[obj.name] = [base, target, None, 0, now]
        return
    base, target, round_low, index, _then = kept
    found = valley_ground(tiles, point, reach, range(index, min(index + VALLEY_PER_TICK, count)))
    if found is not None:
        round_low = found if round_low is None else min(round_low, found)
    index += VALLEY_PER_TICK
    if index >= count:
        lows = [h for h in (round_low, below) if h is not None]
        if lows:
            target = max(lowest, min(lows))
        round_low, index = None, 0
    base += (target - base) * (1.0 - math.exp(-(now - kept[4]) / BASE_EASE))
    if below is not None:
        base = min(base, max(lowest, below))
    _bases[obj.name] = [base, target, round_low, index, now]


def valley_ground(tiles, point, reach, samples, lazy=True):
    """The lowest ground (meters over sea level) of the loaded tiles straight down from points around point (meters,
    from the planet's center), on rings out to reach meters across the planet's surface (VALLEY_RINGS, VALLEY_RAYS a
    ring; samples: which of them, numbered ring by ring), or None where none's loaded. The finest tile loaded at each,
    in view or not: behind the drone, the valley it came from is still loaded. lazy: only where that tile has been
    searched before, the others made ready on the worker thread for the next round (Tileset.loaded_ground())."""
    up = point / np.linalg.norm(point)
    side = np.cross(up, (0.0, 0.0, 1.0) if abs(up[2]) < 0.9 else (1.0, 0.0, 0.0))
    side /= np.linalg.norm(side)
    other = np.cross(up, side)
    # The search remembers the triangle it hit last in each tile, for the drone's next tick: kept as it was
    tile = tiles.tiles.get(tiles.ground_tile)
    last_hit = tile.last_hit if tile is not None else None
    lowest = None
    among = tiles.loaded_around(point, max(VALLEY_RINGS) * reach)
    for sample in samples:
        ring, k = divmod(sample, VALLEY_RAYS)
        angle = (k + 0.5 * (ring % 2)) * 2.0 * math.pi / VALLEY_RAYS
        q = point + VALLEY_RINGS[ring] * reach * (math.cos(angle) * side + math.sin(angle) * other)
        r = float(np.linalg.norm(q))
        t = tiles.loaded_ground(q, -q / r, among, lazy=lazy)
        if t is not None:
            h = r - t - tiles3d.RADIUS
            lowest = h if lowest is None else min(lowest, h)
    if tile is not None:
        tile.last_hit = last_hit
    return lowest


# object name -> follow_base()'s [valley level, the level it eases toward (meters over sea level), the lowest ground of
# the round of points so far, the round's next point, the tick's time]
_bases = {}
# follow_base(): the rings looked down from (shares of the reach), the rays on each and how many a tick, how long the
# level takes to ease most of the way (seconds, the time constant), and a gap after which a flight starts over. One a
# tick, a round of them is 0.4 s, well within the easing; each is a third of a millisecond over the Alps (2026-11-17:
# three a tick were a quarter of the tick)
VALLEY_RINGS = (0.25, 0.5, 1.0)
VALLEY_RAYS = 8
VALLEY_PER_TICK = 1
BASE_EASE = 1.0
BASE_STALE = 1.0


def ground_vector(obj, camera_world):
    """From a camera (world space) to the nearest point of the ground straight below it (world space): with 3D tiles,
    as long as the distance to their nearest ground, where they're drawn (surface_distance()), the way down still the
    sphere's; without them, the sphere's (surface_vector()). What the Altitude readout measures."""
    if not setting(obj, "tiles"):
        return surface_vector(obj, camera_world)
    world = world_of(obj)
    p = map_point(obj, camera_world)
    nA = world.normal()
    q = lifted(world, p, lift_of(obj), nA)
    h = surface_distance(obj, world, p)
    if h is None:
        h = float(height(world, q, nA))
        # With 3D tiles, none loaded under it: over the lift (their ground under the drone when last there, within a
        # fraction of its height, recenter_move()), not the sphere, kilometers below mountains
        over_lift = h - math.ldexp(lift_of(obj) * world.rho, world.n)
        if over_lift > 0.0:
            h = over_lift
    return world_vector(obj, h * down(world, q, nA))


def center_vector(obj, camera_world):
    """From a camera (world space) to the sphere's center (world space): what zoom and orbit scale and turn the drone
    about, as they do about an ordinary object's origin, since the map's object's origin is the anchor's point (lifted),
    which follows the drone. The center is at -R (1 + lift) n(A) in map coordinates. inf past a float's range, 2^1024
    units deep."""
    world = world_of(obj)
    p = map_point(obj, camera_world)
    try:
        R = math.ldexp(world.rho * (1.0 + lift_of(obj)), world.n)
    except OverflowError:
        return np.full(3, math.inf)
    with np.errstate(over="ignore", invalid="ignore"):
        return world_vector(obj, -R * world.normal() - p)


def crossing(obj, start_world, end_world):
    """If a move from start to end (world space) crosses the sphere's surface, the heights of its ends above it (world
    units, signed), else None (see ER_OT_Fly.stop_at_surface()). With 3D tiles, their ground under the start,
    where they're drawn, taken as level across the move, and at the bottom of their dives a clip_start of the drone
    camera over it: their dives stop (loop_factor()), and there, flying on into the ground brought the drone within
    its clip_start of it, which hid it."""
    world = world_of(obj)
    nA = world.normal()
    lift = lift_of(obj)
    scale = float(np.linalg.norm(world_vector(obj, (1.0, 0.0, 0.0))))
    start = map_point(obj, start_world)
    h0 = float(height(world, lifted(world, start, lift, nA), nA))
    h1 = float(height(world, lifted(world, map_point(obj, end_world), lift, nA), nA))
    ground = ground_height(obj, world, start)
    standoff = 0.0
    if ground is not None:
        level = h0 - ground
        h0, h1 = ground, h1 - level
        # At the bottom only (no diving left): above it auto scale shrinks the drone as it nears the ground, and
        # within its lower bound too. A standoff there held a drone already within it (popped out of a finer tile's
        # ground by recenter_move(), or grown to its bound) still every tick, and auto scale, seeing it unmoved, never
        # shrank it: stuck on the ground for good.
        if loop_factor(obj, 2.0) == 1.0:
            standoff = drone_clip_start()
    h0, h1 = h0 * scale - standoff, h1 * scale - standoff
    return (h0, h1) if h0 > 0.0 >= h1 else None


def floor_of(obj, p=None):
    """The floor over the sphere's 3D tiles' ground, meters: the "tiles_floor" setting's share of how far the drone
    camera's lens clip reaches (clip_reach()), 0 without tiles or a floor. No nearer than that, the ground is never
    inside the lens clip, whichever way the drone looks. In the clip's reach rather than meters, since auto scale
    doesn't shrink the drone as it nears the ground (surface_vector()): a meter's floor under a drone grown to skim the
    Alps would be inside its clip, and the ground under it cut away. For a drone at map point p, no more than
    FLOOR_MOST of its height over the valley level (base_height()): auto scale grows the drone as the floor lifts it,
    and the floor with it, so a drone whose floor reached its height (the scale control lets the lens clip reach
    further than that) would be lifted ever higher, never coming to rest."""
    if not setting(obj, "tiles"):
        return 0.0
    share = max(0.0, float(setting(obj, "tiles_floor")))
    if share == 0.0:
        return 0.0
    world = world_of(obj)
    s = map_scale(world)
    world_per_map_unit = float(np.linalg.norm(world_vector(obj, (1.0, 0.0, 0.0))))
    floor = share * clip_reach() / (world_per_map_unit * s)
    if p is not None:
        floor = min(floor, FLOOR_MOST * base_height(obj, world, p) / s)
    return floor


# The floor's most, a share of the drone's height over the valley level (floor_of()): held there, the drone is at most
# as far over the ground as the ground is over the valley level
FLOOR_MOST = 0.5


def floor_move(obj, camera_world):
    """The move (world space) that takes a drone camera at camera_world out of the floor over the sphere's 3D tiles,
    floor_of() meters over their ground where drawn, or None if it isn't in it (see ER_OT_Fly.stop_at_surface()).
    The least move out: flown into the floor at a slant, what's left of the flight is its part along it, so the drone
    glides along the ground, up and down slopes and along cliffs. Each pass finds the ground's nearest point and holds
    the drone a floor off the plane through it, square to the way out (from under the ground, flown past it in a tick
    or a finer tile's ground come in higher, the way out through that point; straight up where none is drawn near, a
    coarser tile standing in for one not in yet), and the move is to the nearest place off all the passes' planes at
    once (off_planes()). In a crease, out of one slope is into the other: pushed out of one at a time, three passes
    didn't settle between two at a shallow angle, and pressed into one, the drone shook by a centimeter a frame a meter
    from the rock.

    Auto scale measures the drone's height over the valley (surface_vector()), not to the ground, so the floor is the
    lens clip's reach (floor_of()), which grows with the drone: over the mountains, where it's big for how near the
    ground is, it skims their tops further up, and on the valley's floor it comes down close."""
    p = map_point(obj, camera_world)
    floor = floor_of(obj, p)
    tiles = tiles_of(obj) if floor > 0.0 else None
    if tiles is None or not tiles.last_draw:
        return None
    world = world_of(obj)
    s = map_scale(world)
    lift = lift_of(obj)
    origin = origin_meters(world, lift)
    end = origin + p / s
    # The planes, as n.move >= b for the move from end (meters, small: no cancellation against the planet's radius)
    normals, bounds = [], []
    move = np.zeros(3)
    contact = None
    for _ in range(FLOOR_PASSES):
        ground = ground_height(obj, world, p + move * s, tiles)
        if ground is None:
            break
        ground /= s
        point = end + move
        up = point / np.linalg.norm(point)
        if ground >= 0.0:
            # Far over it by the nearest ground measured last, it isn't searched for again (surface_distance(): it's
            # searched again once the drone has moved a twentieth of the way, twenty times a second at a cruise, 0.6 ms
            # each over the Alps) until the drone may be within FLOOR_CLEAR floors of it, or NEAREST_FRESH s on
            last = _nearest.get(obj.name)
            if (ground > floor and last is not None and last[0][1] == lift
                    and time.perf_counter() - last[1] < NEAREST_FRESH
                    and (1.0 - FLOOR_SLACK) * last[3] - float(np.linalg.norm(point - last[2])) > FLOOR_CLEAR * floor):
                break
            # Nothing's within the floor if the nearest ground found last (surface_distance(), searched again as the
            # drone moves) was further than the floor and the way flown since, with FLOOR_SLACK of it for finer tiles
            # come in since (they move the ground by their parent's error, a few percent of its distance): the search
            # every tick, for the tiles whose boxes the drone is in, was a fifth of the tick, for a drone that's mostly
            # far from the floor
            if ground > floor:
                surface_distance(obj, world, p + move * s, tiles)
            last = _nearest.get(obj.name)
            if (ground > floor and last is not None
                    and (1.0 - FLOOR_SLACK) * last[3] - float(np.linalg.norm(point - last[2])) > floor):
                break
            # The nearest ground can be beside it (a slope, a cliff), no further than straight down
            d, nearest = tiles.nearest_ground(tiles.last_draw, point, min(ground, floor))
            if d >= floor * (1.0 - FLOOR_TOLERANCE):
                break
            out = (point - nearest) / d if nearest is not None and d > 0.0 else up
            depth = floor - d
        else:
            d, nearest = tiles.nearest_ground(tiles.last_draw, point, -ground)
            out = (nearest - point) / d if nearest is not None and d > 0.0 else up
            depth = floor + d
        # Off the plane by the floor: out . (end + move - point) >= depth
        normals.append(out)
        bounds.append(depth + float(out @ move))
        move = off_planes(np.array(normals), np.array(bounds))
        contact = nearest
    if not move.any():
        return None
    if contact is not None:
        # The next measure of the ground (surface_distance()) from the point the floor holds the drone off, so the two
        # agree: from the nearest point it found last, it read the drone a little further (auto scale's jitter, when
        # it measured to the ground)
        _nearest[obj.name] = ((world.text(), lift), time.perf_counter(), end + move, floor, contact)
    return world_vector(obj, move * s)


def off_planes(normals, bounds):
    """The shortest move x with normals[i] . x >= bounds[i] for every i (normals unit): the nearest place off a few
    planes at once. The constraints it rests on are a subset (at most 3 in 3D), where x is a combination of their
    normals with no negative weight: tried in turn, the shortest that keeps every constraint."""
    if (bounds <= 0.0).all():
        return np.zeros(3)
    best = None
    count = len(bounds)
    for size in range(1, min(count, 3) + 1):
        for subset in itertools.combinations(range(count), size):
            N = normals[list(subset)]
            try:
                weights = np.linalg.solve(N @ N.T, bounds[list(subset)])
            except np.linalg.LinAlgError:
                continue
            if (weights < 0.0).any():
                continue
            x = N.T @ weights
            if (normals @ x >= bounds - FLOOR_TOLERANCE * np.abs(bounds).max()).all():
                if best is None or x @ x < best @ best:
                    best = x
        if best is not None:
            return best
    # No subset keeps them all (planes facing each other across a gap narrower than the floor): out of the last one
    return normals[-1] * bounds[-1]


# floor_move()'s passes at most, and how far within the floor counts as on it (a share of it: float64's rounding)
FLOOR_PASSES = 4
FLOOR_TOLERANCE = 1e-6
# The share of the nearest ground's distance that finer tiles may have moved it by since it was measured (floor_move())
FLOOR_SLACK = 0.1
# floor_move() looks for the nearest ground again only once the drone may be within this many floors of it
FLOOR_CLEAR = 2.0


def clip_reach():
    """How far the drone camera's lens clip reaches (world units): its clip_start out to the clip plane's corners, the
    furthest of it from the camera. With the 10 mm lens on a 36 mm sensor, 2.3 times the clip_start, so the ground a
    clip_start away can be inside the clip at the frame's corners. 0 without the camera."""
    data = _camera_frame()
    return 0.0 if data is None else data.clip_start * _corner[1]


def view_tangents():
    """How wide the drone camera sees, as the tangents of its view's half angles across and up (a point d ahead of it
    is in view within d times them of its axis), or None without a perspective camera"""
    data = _camera_frame()
    return None if data is None or data.type != 'PERSP' else _corner[2]


def _camera_frame():
    """The drone camera's data, its frame's corners measured (_corner) for its settings, or None without the camera"""
    import bpy
    camera = bpy.data.objects.get("Drone_Camera")
    if camera is None or camera.type != 'CAMERA':
        return None
    scene = bpy.context.scene
    render = scene.render
    data = camera.data
    key = (data.lens, data.sensor_width, data.sensor_height, data.sensor_fit, data.shift_x, data.shift_y, data.type,
           render.resolution_x * render.pixel_aspect_x, render.resolution_y * render.pixel_aspect_y)
    if _corner[0] != key:
        corners = data.view_frame(scene=scene)
        if data.type == 'PERSP':
            _corner[:] = [key, max(v.length / abs(v.z) for v in corners),
                          (max(abs(v.x / v.z) for v in corners), max(abs(v.y / v.z) for v in corners))]
        else:
            _corner[:] = [key, 1.0, None]
    return data


# The camera settings clip_reach() and view_tangents() were measured for, its clip plane's corners' distance over
# clip_start, and its view's half widths as tangents
_corner = [None, 1.0, None]


def drone_clip_start():
    """The drone camera's clip_start (world units: a camera's scale doesn't scale its clipping), 0 without one"""
    import bpy
    camera = bpy.data.objects.get("Drone_Camera")
    if camera is None or camera.type != 'CAMERA':
        return 0.0
    return camera.data.clip_start


def drone_clip_end():
    """The drone camera's clip_end (world units), 0 without one"""
    import bpy
    camera = bpy.data.objects.get("Drone_Camera")
    if camera is None or camera.type != 'CAMERA':
        return 0.0
    return camera.data.clip_end


def loop(obj, factor, camera_world):
    """A loop by factor (a power of 2, see exact_factor()) about the map's origin: N and A doubled (a dive, factor > 1)
    or, after the anchor moves to the coarser lattice under the camera, halved (a climb). Returns the move of the map's
    origin the climb made first (world units, before the loop: the drone moves by minus it), or None."""
    world = world_of(obj)
    lift = lift_of(obj)
    power = round(math.log2(factor))
    move = None
    if power > 0:
        dive(world, power)
    elif power < 0:
        offset = climb(world, -power, lifted(world, map_point(obj, camera_world), lift))
        # The origin is R lift over the sphere, so it moves (1 + lift) times as far as the anchor's point
        move = world_vector(obj, (1.0 + lift) * np.asarray(offset)) if offset is not None else None
    save(obj, world)
    return move


def loop_factor(obj, factor):
    """The loop a sphere makes for a loop by factor: the nearest power of 2, and a climb no higher than the globe (N = 1,
    the top), 1.0 if there's no climbing left. With 3D tiles, a dive no deeper than map units of tiles3d.DEEPEST_UNIT
    meters, 1.0 if there's no diving left: past it there's no detail, and tiles' places would begin to jitter."""
    world = world_of(obj)
    power = round(math.log2(factor)) if factor > 0 else 0
    if power == 0:
        power = 1 if factor > 1.0 else -1
    if power < 0:
        power = max(power, -world.n)
    elif tiles_of(obj) is not None:
        room = math.floor(math.log2(tiles3d.RADIUS / (world.rho * tiles3d.DEEPEST_UNIT))) - world.n
        if room <= 0:
            return 1.0
        power = min(power, room)
    return 2.0 ** power


def at_top(obj):
    """Whether the sphere is at its globe, N = 1, with nothing coarser to climb into"""
    return world_of(obj).n == 0


def recenter_move(obj, camera_world):
    """Move the anchor under the drone camera if it has moved off it (see recenter()), and with 3D tiles, the lift to
    their ground's height under it if that's moved off it as far: the move of the map's origin in world units (the
    drone moves by minus it), or None. With 3D tiles, a drone under their ground is also lifted out of it (that part
    of the move moves the drone, not the map)."""
    world = world_of(obj)
    p = map_point(obj, camera_world)
    lift = lift_of(obj)
    nA = world.normal()
    q = lifted(world, p, lift, nA)
    tiles = tiles_of(obj)
    ground = ground_height(obj, world, p, tiles) if tiles is not None else None
    over_sphere = float(height(world, q, nA))
    # Only once the drone is a few units from the anchor's point or a fair way relative to its height: every move
    # rebuilds the frame's plan
    h = abs(ground if ground is not None else over_sphere)
    reach = max(RECENTER_UNITS, h / world.rho * RECENTER_HEIGHTS)
    offset = recenter(world, q, reach)
    # The origin is R lift over the sphere, so it moves (1 + lift) times as far as the anchor's point
    move = None if offset is None else (1.0 + lift) * np.asarray(offset)
    new_lift = lift
    pop = None
    if ground is not None:
        # The lift that puts the origin on the ground under the drone (its height over the sphere, a share of R)
        target = math.ldexp((over_sphere - ground) / world.rho, -world.n)
        if abs(math.ldexp(target - lift, world.n)) > reach:
            rise = math.ldexp((target - lift) * world.rho, world.n) * world.normal()
            move = rise if move is None else move + rise
            new_lift = target
        if ground < 0.0:
            # Under the ground: a finer tile's came in higher than the coarser one's the drone was over (by up to the
            # coarser one's geometric error), or flight got past it. Up out of it, to the floor (floor_move()), or
            # without one a clip_start over it, where stopping at the surface (crossing()) would have kept it: not a
            # move of the map, so not saved with it.
            unit = float(np.linalg.norm(world_vector(obj, (1.0, 0.0, 0.0))))
            floor = floor_of(obj, p)
            standoff = floor * map_scale(world) if floor > 0.0 else drone_clip_start() / unit
            pop = -(-ground + standoff) * world.normal()
    if move is None and pop is None:
        return None
    if move is not None:
        save(obj, world, new_lift)
    return world_vector(obj, (move if move is not None else 0.0) + (pop if pop is not None else 0.0))


def select_if_looking_at_it(scene, camera, rig_names=("Drone", "Drone_Camera", "clip_start", "clip_end")):
    """From ER_OT_Fly.click(), after its click at the view's center: if it selected nothing (but the drone rig)
    and the drone camera's view center meets a sphere, select that sphere's object, as a click would have if it were a
    mesh"""
    import bpy
    if camera is None or any(o.name not in rig_names for o in bpy.context.selected_objects):
        return
    for obj in sphere_objects(scene):
        world = world_of(obj)
        c = lifted(world, map_point(obj, camera.matrix_world.translation), lift_of(obj))
        forward = map_vector(obj, np.array(camera.matrix_world.to_3x3())[:, 2] * -1.0)
        if np.isfinite(ray_hit(world, c, forward)):
            for o in bpy.context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            return


# A frame's plan is made again once the camera has moved this share of its height, or its height changed by it
PLAN_MOVE = 0.05
# The planet's tiles in view are found again once the camera has moved this share of its height, or turned about this
# many radians (Floor.planet_tiles())
VIEW_MOVE = 0.02
VIEW_TURN = 0.01
# Recenter once the drone is this many units from the anchor's point, or this many of its heights over the surface
RECENTER_UNITS = 4.0
RECENTER_HEIGHTS = 2.0


# ----- the draw handler -----

class Floor:
    """Draws the scene's spheres: the shader, its full-screen triangle, and the last plan (rebuilt when the world, or the
    camera's place or scale, changed)"""

    def __init__(self):
        import gpu
        from gpu_extras.batch import batch_for_shader
        self.shaders = {}          # (layout, motif) -> its main, composite, lower resolution and tile shaders
        cube = self.shaders_for(motif_of("cube", DEFAULTS["motif"]))
        triangle = {"pos": [(-1.0, -1.0), (3.0, -1.0), (-1.0, 3.0)]}
        self.batch = batch_for_shader(cube[0], 'TRIS', triangle)
        self.composite_batch = batch_for_shader(cube[1], 'TRIS', triangle)
        self.offscreens = {}       # by size, for the lower resolution pass and the tile pass
        self.resolution = None     # auto_resolution's, a multiple of RESOLUTION_STEP
        self.fast_frames = 0
        self.recent_missed = []
        self.raise_wait = FAST_FRAMES_TO_RAISE
        self.raised = False
        self.plans = {}            # object name -> (key, plan, level buffer)
        self.values = None         # kept until the next draw, past this one's use of it
        self.frame_time = None
        self.last_draw = None
        self.last_through = None   # when a view looking through the drone camera last drew
        self.plan_ms = 0.0
        self.empty_textures = None # for the planet's samplers without a planet
        self.planet_views = {}     # object name -> (key, camera, ray matrix, the tiles it needed)

    def shaders_for(self, motif):
        """A motif's shaders (make_shader()): the main pass, the composite, the lower resolution pass, the tile pass;
        made the first time a sphere with it draws"""
        key = (motif.layout, motif.name)
        shaders = self.shaders.get(key)
        if shaders is None:
            shaders = self.shaders[key] = (make_shader(motif=motif), make_shader(composite=True, motif=motif),
                                           make_shader(write_depth=False, motif=motif),
                                           make_shader(tiles=True, motif=motif))
        return shaders

    def plan_for(self, obj, world, motif, camera, pixel_angle, clip_end, planet=None, handoff=2.0):
        """The plan for a view, kept while the camera stays within PLAN_MOVE of its height of where it was made (the
        shader gets the camera's place every frame; the plan only says which levels to draw and how to scale them)"""
        import time
        h = max(abs(float(height(world, camera))), 1e-300)
        step = h * PLAN_MOVE
        key = (world.text(), motif.name, round(math.log2(h) / PLAN_MOVE),
               tuple(np.round(np.asarray(camera) / step).astype(np.int64)),
               round(math.log2(max(pixel_angle, 1e-30)) * 8), round(math.log2(max(clip_end, 1e-30)) * 8),
               planet.pyramid.folder if planet is not None else None, handoff)
        cached = self.plans.get(obj.name)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        start = time.perf_counter()
        plan = Plan(world, camera, pixel_angle, clip_end, motif, planet, handoff)
        levels = level_buffer(plan)
        self.plan_ms = (time.perf_counter() - start) * 1000.0
        self.plans[obj.name] = (key, plan, levels)
        return plan, levels

    def draw(self, context, scene, obj):
        """From the draw handler: this view's matrices and size, and the drone camera's clip end when looking through
        it"""
        import bpy
        region_data = context.region_data
        if not region_data.is_perspective:
            return
        drone_camera = bpy.data.objects.get("Drone_Camera")
        looking_through = region_data.view_perspective == 'CAMERA' and drone_camera is not None
        clip_end = drone_camera.data.clip_end if looking_through else context.space_data.clip_end
        drone_scale = drone_camera.matrix_world.to_scale()[0] if drone_camera is not None else 1.0
        import time
        now = time.perf_counter()
        if looking_through:
            self.last_through = now
        # The drone's ground (3D tiles) is the view's looking through the drone camera, any view's without one
        drone_view = looking_through or self.last_through is None or now - self.last_through > 0.5
        # Frame time, between draws of the view looking through the drone camera (or any view, without one)
        if looking_through or not self.frame_time:
            if self.last_draw is not None:
                interval = now - self.last_draw
                self.frame_time = interval if self.frame_time is None or interval > 0.5 else (
                    0.9 * self.frame_time + 0.1 * interval)
                if interval < 0.1:
                    self.adjust_resolution(obj, interval)
            self.last_draw = now
        self.draw_view(scene, obj, region_data.view_matrix, region_data.window_matrix, context.region.width,
                       context.region.height, clip_end, drone_scale, self.current_resolution(obj),
                       drone_view=drone_view)

    def current_resolution(self, obj):
        top = min(1.0, max(MIN_RESOLUTION, float(setting(obj, "resolution"))))
        if not setting(obj, "auto_resolution"):
            return top
        if self.resolution is None or self.resolution > top:
            self.resolution = top
        return self.resolution

    def adjust_resolution(self, obj, interval):
        """auto_resolution, after each frame of interval seconds: as the Mandelbrot floor's (mandelbrot.py)"""
        if not setting(obj, "auto_resolution"):
            return
        top = min(1.0, max(MIN_RESOLUTION, float(setting(obj, "resolution"))))
        current = self.current_resolution(obj)
        missed = interval > MISSED_FRAME
        self.recent_missed = (self.recent_missed + [missed])[-RECENT_FRAMES:]
        self.fast_frames = 0 if missed else self.fast_frames + 1
        if sum(self.recent_missed) >= MISSED_TO_LOWER and current > MIN_RESOLUTION:
            self.resolution = max(MIN_RESOLUTION, round((current - RESOLUTION_STEP) / RESOLUTION_STEP) * RESOLUTION_STEP)
            self.raise_wait = min(MAX_RAISE_WAIT, self.raise_wait * 2) if self.raised else FAST_FRAMES_TO_RAISE
            self.raised = False
            self.recent_missed = []
            self.fast_frames = 0
        elif self.fast_frames >= self.raise_wait and current < top:
            self.resolution = min(top, current + RESOLUTION_STEP)
            self.raised = True
            self.fast_frames = 0

    def draw_view(self, scene, obj, view_matrix, window_matrix, width, height, clip_end, drone_scale, resolution=1.0,
                  into=None, drone_view=True):
        """Draws the sphere into the bound framebuffer (the view's), or into the offscreen into (tests): the lower
        resolution pass first, while the view's framebuffer is bound, as the Mandelbrot floor does it (an offscreen
        bound inside another doesn't give the outer one back its depth). drone_view: whether the drone's ground (3D
        tiles) is this view's (draw_tiles())."""
        import gpu
        # The view's framebuffer, bound again for the passes drawn into it after an offscreen's: taken before any
        # offscreen is bound (the planet's atlas, below), since after one this would be the window's, with no depth
        view_framebuffer = gpu.state.active_framebuffer_get()
        world = world_of(obj)
        lift = lift_of(obj)
        view = np.array(view_matrix, dtype=np.float64)
        window = np.array(window_matrix, dtype=np.float64)
        camera_world = np.linalg.inv(view)[:3, 3]
        origin, linear, inverse = to_map(obj)
        camera_map = inverse @ (camera_world - origin)
        # The sphere's own functions measure from the anchor's point on it, which is under the origin by the lift
        camera = lifted(world, camera_map, lift)
        rotation = view[:3, :3]
        view_projection = window @ np.block([[rotation, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])
        inv = np.linalg.inv(view_projection)
        # A ray's direction from clip coordinates (x, y), world space without the camera's place: inv (x, y, 1, 1),
        # whose w doesn't change with x and y in a perspective view
        w = inv[3, 2] + inv[3, 3]
        ray_world = np.stack([inv[:3, 0], inv[:3, 1], inv[:3, 2] + inv[:3, 3]], axis=1) / w
        ray_matrix = inverse @ ray_world
        low_width, low_height = max(1, round(width * resolution)), max(1, round(height * resolution))
        pixel_size = (2.0 / low_width, 2.0 / low_height)
        pixel_angle = 2.0 / (window[1][1] * low_height)
        fog = fog_rows(scene, obj, world, camera, drone_scale, 0.5 * window[1][1] * height)

        # A function: an offscreen's bind() binds it at once, not when its context is entered
        def target():
            return into.bind() if into is not None else view_framebuffer.bind()
        # The Earth's sky first, behind everything (space.py)
        air = air_view(obj, world, camera)
        if air is not None:
            height_world, center, up, per_meter = air
            scale_height = CLEAR_AIR_HEIGHT * per_meter
            if not math.isfinite(scale_height):
                # Too deep for floats: on the ground, as far as the sky can tell
                height_world, center, scale_height = 0.0, 1.0e6, 1.0
            star_axes = inverse / np.linalg.norm(inverse, axis=1, keepdims=True)
            with target():
                space.sky().draw(ray_world, star_axes, fog[0][:3], height_world / scale_height, center / scale_height,
                                 up, 2.0 / (window[1][1] * height))
        # 3D tiles first: where they cover the whole view, the sphere isn't drawn; elsewhere it's behind them
        tileset = tiles_of(obj)
        selection = None
        if tileset is not None:
            selection = self.draw_tiles(obj, tileset, world, lift, camera_map, view, window, view_projection, linear,
                                        width, height, fog, target, drone_view)
            if selection.complete:
                gpu.state.depth_mask_set(False)
                gpu.state.depth_test_set('NONE')
                return
        motif = motif_of(world.layout, setting(obj, "motif"))
        planet = planet_of(obj)
        handoff = float(setting(obj, "planet_handoff"))
        plan, levels = self.plan_for(obj, world, motif, camera, pixel_angle, clip_end, planet, handoff)
        planet_rows = None
        if planet is not None:
            # The tiles of the planet the view shows, loaded as far as this draw's budget goes (planet.Planet.update())
            bias = float(setting(obj, "planet_level_bias"))
            needed = self.planet_tiles(obj, planet, plan, camera, ray_matrix, pixel_size, bias)
            planet.update(keeper(), needed)
            planet_rows = planet.uniforms(plan.on_planet, world.n, bias)
        sun = next((o for o in scene.objects if o.type == 'LIGHT' and o.data.type == 'SUN'), None)
        sun_world = np.array(sun.matrix_world.to_3x3(), dtype=np.float64)[:, 2] if sun else np.array((0.35, -0.45, 0.82))
        sun_map = inverse @ sun_world
        sun_map /= np.linalg.norm(sun_map)
        tiles_width, tiles_height = -(-low_width // TILE), -(-low_height // TILE)
        self.values = values_buffer(plan, camera, ray_matrix, linear, view_projection, pixel_size, fog, sun_map,
                                    (low_width, low_height, tiles_width, tiles_height),
                                    planet_rows, behind=selection is not None)
        planet_textures = self.planet_textures(planet)
        main_shader, composite_shader, low_shader, tile_shader = self.shaders_for(motif)
        # The tile pass first (tileMain()), then the sphere at full resolution, or at a lower one and scaled up
        tiles = self.offscreen((tiles_width, tiles_height, 'RGBA32F'))
        low = self.offscreen((low_width, low_height, 'RGBA16F')) if resolution < 1.0 else None
        with tiles.bind():
            gpu.state.active_framebuffer_get().clear(color=(0.0, 0.0, 0.0, 0.0))
            gpu.state.depth_test_set('NONE')
            self.draw_pass(tile_shader, levels)
        if low is None:
            with target():
                gpu.state.depth_test_set('LESS_EQUAL')
                gpu.state.depth_mask_set(True)
                self.draw_pass(main_shader, levels, tileMask=tiles.texture_color, **planet_textures)
        else:
            with low.bind():
                gpu.state.active_framebuffer_get().clear(color=(0.0, 0.0, 0.0, 0.0))
                gpu.state.depth_test_set('NONE')
                self.draw_pass(low_shader, levels, tileMask=tiles.texture_color, **planet_textures)
            with target():
                gpu.state.depth_test_set('LESS_EQUAL')
                gpu.state.depth_mask_set(True)
                self.draw_pass(composite_shader, levels, self.composite_batch, lowColor=low.texture_color)
        gpu.state.depth_mask_set(False)
        gpu.state.depth_test_set('NONE')

    def draw_tiles(self, obj, tileset, world, lift, camera, view, window, view_projection, world_from_map, width,
                   height, fog, target, drone_view=True):
        """The 3D tiles in view, into the view's framebuffer (target()), depth tested and written: the tileset's loads
        taken in (within its budget), its selection for this view made or reused (tiles3d.Tileset.selection()) and
        drawn, from the camera's place in meters (camera: map units from the map's origin). Returns the selection.
        The drone's view (drone_view) also selects the ground around the drone, in view or not (tiles3d.AROUND), and
        its selection is the one the ground is measured in (ground_height(), surface_distance()): with any view's, a
        second viewport's tiles, for another camera somewhere else, were the drone's ground every other draw."""
        import gpu
        s = map_scale(world)
        camera_m = origin_meters(world, lift) + camera / s
        forward = np.linalg.solve(world_from_map, -view[2, :3])
        forward /= np.linalg.norm(forward)
        planes = tiles3d.view_planes(view_projection, world_from_map, s)
        # Pixels per unit of the view's tangent, up the view: a tile's screen-space error is its error times this
        # over its distance
        K = 0.5 * window[1][1] * height
        tileset.take_results()
        name = (obj.name, width, height)
        last = tileset.selection_cache.get(name)
        around = 0.0
        if drone_view:
            # The drone's distance to the ground: the flight's own last measure (the floor's, each tick), else made
            measured = _nearest.get(obj.name)
            if measured is not None and time.perf_counter() - measured[1] < NEAREST_FRESH:
                around = measured[3]
            else:
                d = surface_distance(obj, world, camera, tileset)
                around = max(d, 0.0) / s if d is not None else 0.0
        # How far the view may move before it's selected for again: a share of the drone's distance to the ground, or
        # for another view, of its distance to its nearest tile
        nearest = around if around > 0.0 else tileset.nearest(last, camera_m) if last is not None else 1.0
        selection = tileset.selection(name, camera_m, forward, planes, K, float(setting(obj, "tiles_error")), nearest,
                                      around, bool(setting(obj, "tiles_motion")))
        if drone_view:
            tileset.last_draw = selection.ground
            tileset.last_fallbacks = selection.ground_fallbacks
            tileset.last_holes = selection.ground_holes
        tileset.last_shown = selection.shown
        values = tiles3d.frame_values(view_projection, world_from_map, fog)
        with target():
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.depth_mask_set(True)
            gpu.state.face_culling_set('NONE')
            tileset.draw(selection.draw, camera_m, s, values, selection.fallbacks)
        tileset.start_selections()
        return selection

    def planet_tiles(self, obj, planet, plan, camera, ray_matrix, pixel_size, bias):
        """The planet's tiles in view (planet.view_tiles(), about a millisecond), worked out again once the camera has
        moved VIEW_MOVE of its height or turned about VIEW_TURN radians, or the world changed"""
        if not plan.on_planet:
            return []
        world = plan.world
        key = (world.text(), planet.pyramid.folder, bias, tuple(pixel_size))
        last = self.planet_views.get(obj.name)
        if (last is not None and last[0] == key
                and np.max(np.abs(camera - last[1])) < VIEW_MOVE * max(plan.height, 1e-300)
                and np.max(np.abs(ray_matrix - last[2])) < VIEW_TURN * np.max(np.abs(ray_matrix))):
            return last[3]
        needed = planets.view_tiles(planet.pyramid, camera, plan.normal, world.kappa, world.rho, world.n,
                                    ray_matrix, pixel_size, bias)
        self.planet_views[obj.name] = (key, np.array(camera), np.array(ray_matrix), needed)
        return needed

    def planet_textures(self, planet):
        """The painting passes' planet samplers: the planet's atlas and page table, or empty ones without a planet"""
        if planet is not None and planet.atlas is not None:
            return {"planetAtlas": planet.atlas.texture_color, "planetPages": planet.pages}
        if self.empty_textures is None:
            import gpu
            self.empty_textures = {
                "planetAtlas": gpu.types.GPUTexture((1, 1), format='RGBA8',
                                                    data=gpu.types.Buffer('FLOAT', 4, [0.0, 0.0, 0.0, 1.0])),
                "planetPages": gpu.types.GPUTexture((1, 1), format='RGBA32F',
                                                    data=gpu.types.Buffer('FLOAT', 4, [0.0, 0.0, 0.0, 0.0]))}
        return self.empty_textures

    def offscreen(self, key):
        """An offscreen of (width, height, format), kept for each size auto_resolution moves between, a handful. Made
        inside the keeper's bind (keeper()): making one binds the window's framebuffer, and every bind's exit then went
        back to that, not the view's, so the view's overlays drew into the window that frame and Metal refused the
        composite, which writes depth. An offscreen's bind, unlike a framebuffer's, gives back what was bound before."""
        import gpu
        offscreen = self.offscreens.get(key)
        if offscreen is None:
            if len(self.offscreens) > 12:
                self.offscreens = {}
            with keeper().bind():
                offscreen = self.offscreens[key] = gpu.types.GPUOffScreen(key[0], key[1], format=key[2])
        return offscreen

    def draw_pass(self, shader, levels, batch=None, **samplers):
        shader.bind()
        shader.uniform_block("sphereValues", self.values)
        shader.uniform_block("levelValues", levels)
        for name, texture in samplers.items():
            shader.uniform_sampler(name, texture)
        (batch or self.batch).draw(shader)

    def draw_info(self, obj, ui_scale=1.0):
        import blf
        tileset = tiles_of(obj)
        if tileset is not None:
            # The attribution the tiles' terms require, always (under the info line)
            tiles3d.draw_attribution(tileset, tileset.last_shown, ui_scale)
        if not setting(obj, "show_info"):
            return
        plan = self.plans.get(obj.name)
        if plan is None and tileset is None:
            return
        if plan is not None:
            plan = plan[1]
            world = plan.world
            text = (f"Sphere  N 2^{world.n}  levels {plan.levels[0] if plan.levels else '-'}..{plan.levels[-1] if plan.levels else '-'}"
                    f"  height {plan.height:.4g}  plan {self.plan_ms:.1f} ms")
        else:
            text = f"Sphere  N 2^{world_of(obj).n}"
        p = planet_of(obj)
        if p is not None and plan is not None:
            text += f"  planet {'under' if plan.on_planet else 'covered'}, {p.loads} tiles loaded, {p.missing} to load"
        if tileset is not None:
            text += f"  3D {len(tileset.last_shown)} drawn, {tileset.info()}, lift {lift_of(obj) * tiles3d.RADIUS:.0f} m"
        blf.size(0, 13)
        blf.color(0, 1.0, 1.0, 1.0, 0.85)
        blf.position(0, 12, 32 if tileset is None else 34.0 * ui_scale, 0)
        blf.draw(0, text)


_keeper = None       # a 1 x 1 offscreen, made outside any drawing, that the others are made inside (Floor.offscreen())


def keeper():
    global _keeper
    if _keeper is None:
        import gpu
        _keeper = gpu.types.GPUOffScreen(1, 1)
    return _keeper


def make_keeper():
    """From a timer once registered, so the keeper isn't made in a draw (where making it would bind the window's
    framebuffer for the rest of that frame)"""
    try:
        keeper()
    except Exception:          # no GPU (background mode): made, if ever, at the first draw
        pass


def floor():
    global _floor
    if _floor is None:
        _floor = Floor()
    return _floor


@background.foreground
def draw_spheres():
    import bpy
    context = bpy.context
    if context.region_data is None:
        return
    for obj in sphere_objects(context.scene):
        if obj.visible_get():
            floor().draw(context, context.scene, obj)


@background.foreground
def draw_info():
    import bpy
    if _floor is None:
        return
    for obj in sphere_objects(bpy.context.scene):
        _floor.draw_info(obj, bpy.context.preferences.system.ui_scale)


def forget(_dummy=None):
    """After loading a file: another file's worlds, plans and planets don't apply. 3D tilesets are kept, by their
    source: a new one would start a new session (with Google, a billed request)."""
    global _floor
    _worlds.clear()
    _to_map.clear()
    _grounds.clear()
    _ground_last.clear()
    _nearest.clear()
    _bases.clear()
    _floor = None
    planets.forget()


_handles = []


def register():
    import bpy
    _handles.append(bpy.types.SpaceView3D.draw_handler_add(draw_spheres, (), 'WINDOW', 'POST_VIEW'))
    _handles.append(bpy.types.SpaceView3D.draw_handler_add(draw_info, (), 'WINDOW', 'POST_PIXEL'))
    if not bpy.app.background:
        bpy.app.timers.register(make_keeper, first_interval=0.0)
    handler = bpy.app.handlers.persistent(forget)
    if not any(getattr(h, "__name__", "") == "forget" and getattr(h, "__module__", "") == __name__
               for h in bpy.app.handlers.load_post):
        bpy.app.handlers.load_post.append(handler)


def unregister():
    import bpy
    for handle in _handles:
        bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
    _handles.clear()
    global _keeper
    _keeper = None
    for h in list(bpy.app.handlers.load_post):
        if getattr(h, "__name__", "") == "forget" and getattr(h, "__module__", "") == __name__:
            bpy.app.handlers.load_post.remove(h)
    forget()
    tiles3d.forget()
