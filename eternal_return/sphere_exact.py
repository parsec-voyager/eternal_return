"""Points of the cube placed on the sphere with as many digits as a depth needs, for what has to be exact far past
float64: the pieces of a giant level next to the drone (parts.py).

sphere.surface_offsets() places a point from the anchor in float64 without cancellation, so it's exact to about 1e-15
of the point's distance from the anchor. That's plenty for anything near the anchor, but a giant level's pillar has its
wall right by the drone and its vertex, from which it's placed, far away: diving at the foot of a pillar a trillion
times the drone's height, float64 puts its wall a thousandth of the drone's height off, and every loop deeper eight
times more. Here the same chart is worked out with Python's decimal module at a precision that grows with the depth:
sphere_chart.py's series with its float64 constants taken exactly (as the tests' oracle does,
~/blender_tests/sphere/stage2/oracle.py), and surface_offsets()'s folds, each measured from the anchor's distance from
the wall's foot (the series misses the midline wall by about 3e-16, which deep down is far more than a drone, and
measured that way the folds join exactly). So offsets() is surface_offsets() with more digits, and agrees with it to
float64's rounding.

Slow next to float64 (a point takes about 0.1 ms at 50 digits and grows with them), so it's only for a few points: the
chart's values are kept by point, and a world's own (its anchor and the walls' feet) by world.
"""

import decimal
import math
from decimal import Decimal
from fractions import Fraction

try:
    from . import sphere_chart as chart
except ImportError:          # tests import it by itself
    import sphere_chart as chart

# The digits a world needs: enough for N times a difference of directions to be exact to well under the drone's
# height, the chart's corner (where the cube root's differences lose digits) included, as the oracle has it
DIGITS_BASE = 30
DIGITS_PER_BIT = 0.302 * 1.4
# Digits come in steps of this many, so a world's values stay usable over a few loops
DIGITS_STEP = 16


def kept(cache, key, most, make):
    """cache[key], made (make()) the first time it's asked for; a cache past most entries is emptied first"""
    value = cache.get(key)
    if value is None:
        if len(cache) > most:
            cache.clear()
        value = cache[key] = make()
    return value


def digits_for(n, h_units):
    """The precision for a world N = 2^n with the drone h_units units above the sphere"""
    digits = DIGITS_BASE + DIGITS_PER_BIT * n + max(0.0, -math.log10(max(h_units, 1e-300)))
    return int(math.ceil(digits / DIGITS_STEP) * DIGITS_STEP)


def number(x):
    """An int, Fraction or float as a Decimal, at the context's precision (exact for ints and floats)"""
    if isinstance(x, Fraction):
        return Decimal(x.numerator) / Decimal(x.denominator)
    return Decimal(x)


class C:
    """A complex number of two Decimals"""
    __slots__ = ("re", "im")

    def __init__(self, re, im=Decimal(0)):
        self.re, self.im = re, im

    @staticmethod
    def of(z):
        return C(Decimal(z.real), Decimal(z.imag))

    def __add__(self, o):
        return C(self.re + o.re, self.im + o.im)

    def __sub__(self, o):
        return C(self.re - o.re, self.im - o.im)

    def __mul__(self, o):
        return C(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)

    def __truediv__(self, o):
        d = o.re * o.re + o.im * o.im
        return C((self.re * o.re + self.im * o.im) / d, (self.im * o.re - self.re * o.im) / d)

    def abs2(self):
        return self.re * self.re + self.im * self.im

    def to_complex(self):
        return complex(float(self.re), float(self.im))


def root(x, n, first):
    """x's n-th root (n = 3 or 5: n - 1 a power of 2), by Newton's method from first(z), the float64 root of x as a
    float64 z, which picks the branch; a value below a float's range is scaled up by a power of 2^n first"""
    if x.re == 0 and x.im == 0:
        return C(Decimal(0), Decimal(0))
    shift = 0
    while x.to_complex() == 0:
        x = x * C(Decimal(2) ** (100 * n))
        shift += 100
    v = C.of(first(x.to_complex()))
    order = C(Decimal(n))
    limit = Decimal(10) ** (-2 * decimal.getcontext().prec + 4)
    for _ in range(64):
        power = v * v                      # v^(n - 1), by squaring
        for _ in range((n - 1).bit_length() - 2):
            power = power * power
        step = (power * v - x) / (power * order)
        v = v - step
        if step.abs2() <= v.abs2() * limit:
            break
    return v * C(Decimal(2) ** -shift) if shift else v


def cube_root(x):
    """The principal cube root (float64's start picks it)"""
    return root(x, 3, lambda z: z ** (1.0 / 3.0))


_constants = {}      # precision -> the chart's constants as Decimals


def constants():
    prec = decimal.getcontext().prec
    values = _constants.get(prec)
    if values is None:
        half = Decimal("0.5").sqrt()
        values = _constants[prec] = {
            "A": [Decimal(a) for a in chart.A_RANCIC], "I13": C.of(chart.I13), "RA": C(Decimal(chart.RA)),
            "CB": C.of(chart.CB), "CC": C.of(chart.CC), "i": C(Decimal(0), Decimal(1)), "half": half,
            # The domain's walls' unit normals (sphere_chart.WALLS), exactly
            "walls": [(Decimal(0), Decimal(0), Decimal(1)), (Decimal(0), half, -half), (half, -half, Decimal(0))]}
    return values


def direction(N, a, b):
    """n(P) for the domain's point P = (N, a, b), a and b exact (ints or Fractions): (x, y, z), Decimals at the
    context's precision (sphere_chart.Reference's n0, with every digit)"""
    k = constants()
    z = C(number(Fraction(N - a, 2 * N)), number(Fraction(N - b, 2 * N)))
    Z = z * z
    Z = Z * Z
    W = C(Decimal(0), Decimal(0))
    for coefficient in reversed(k["A"][1:]):
        W = (W + C(coefficient)) * Z
    u = k["I13"] * cube_root(W * k["i"])
    w = (u - k["RA"]) / (k["CB"] + k["CC"] * u)
    s = 1 + w.abs2()
    return ((2 - s) / s, 2 * w.re / s, 2 * w.im / s)


_directions = {}     # (a / N, b / N, precision) -> direction()


def direction_kept(N, a, b):
    """direction(), kept by the point (the same point at every depth: after a loop it's a, b and N times 2^k) and the
    precision"""
    key = (Fraction(a, N), Fraction(b, N), decimal.getcontext().prec)
    return kept(_directions, key, 4096, lambda: direction(N, a, b))


def add(u, v):
    return (u[0] + v[0], u[1] + v[1], u[2] + v[2])


def scaled(u, s):
    return (u[0] * s, u[1] * s, u[2] * s)


def dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


class Placer:
    """A world's points placed with digits digits: offsets() and placed() are sphere.surface_offsets() and
    sphere.placed() for exact offsets, with Decimals out"""

    def __init__(self, world, digits):
        self.world, self.digits = world, int(digits)
        self.N, self.a, self.b = world.N, world.A_f[1], world.A_f[2]
        self.G = [[int(v) for v in row] for row in world.G]
        with self.context():
            k = constants()
            self.n0 = direction_kept(self.N, self.a, self.b)
            # N times the anchor's distance from each wall, measured from the wall's foot nearest it along the face
            # (sphere_chart.Reference.wall_distances()), so the folds below join exactly
            N, a, b = self.N, self.a, self.b
            feet = ((a, 0), (Fraction(a + b, 2), Fraction(a + b, 2)), (N, b))
            self.walls = []
            for nu, (fa, fb) in zip(k["walls"], feet):
                if fa == a and fb == b:
                    self.walls.append(Decimal(0))
                    continue
                foot = direction_kept(N, fa, fb)
                m = tuple((p - q) * N for p, q in zip(foot, self.n0))
                self.walls.append(max(Decimal(0), -dot(m, nu)))
            self.nA = self.unfold(self.n0)
            self.rho = Decimal(world.rho)

    def context(self):
        return decimal.localcontext(prec=self.digits)

    def unfold(self, v):
        """A vector of the folded anchor's frame in the map's cube axes: G^T v"""
        G = self.G
        return tuple(G[0][i] * v[0] + G[1][i] * v[1] + G[2][i] * v[2] for i in range(3))

    def offsets(self, deltas):
        """N (n(A + d) - n(A)) (cube axes, units) for points A + d of the cube's surface, d exact (three ints or
        Fractions each, the cube's axes): a tuple of three Decimals for each, as surface_offsets() works it out"""
        N, a, b = self.N, self.a, self.b
        G = self.G
        w0, w1, w2 = b, a - b, N - a
        k = constants()
        out = []
        with self.context():
            for delta in deltas:
                delta = [Fraction(c) for c in delta]
                d = [sum(G[r][c] * delta[c] for c in range(3)) for r in range(3)]
                # Folded into the domain, exactly (sphere.surface_offsets()), noting the walls in turn
                walls = []
                for _ in range(chart.FOLD_STEPS):
                    if d[2] < -w0:
                        d[2] = -2 * w0 - d[2]
                        walls.append(0)
                    elif d[1] - d[2] < -w1:
                        d[1], d[2] = d[2] - w1, d[1] + w1
                        walls.append(1)
                    elif d[0] - d[1] < -w2:
                        d[0], d[1] = d[1] - w2, d[0] + w2
                        walls.append(2)
                    else:
                        break
                n1 = direction_kept(N, a + d[1], b + d[2])
                m = tuple((p - q) * N for p, q in zip(n1, self.n0))
                # Back through the reflections, first ones last:
                # m' = s_1 ... s_k m + sum_j s_1 ... s_(j-1) (-2 s_j nu_j)
                for wall in reversed(walls):
                    m = reflect(m, wall)
                    m = add(m, scaled(k["walls"][wall], -2 * self.walls[wall]))
                out.append(self.unfold(m))
        return out

    def placed(self, deltas):
        """The points A + d on the sphere in map coordinates from the map's origin, and the sphere's unit normals there
        (sphere.placed(), no lift): two lists of Decimal triples"""
        ms = self.offsets(deltas)
        with self.context():
            points = [scaled(m, self.rho) for m in ms]
            normals = []
            for m in ms:
                n = add(self.nA, scaled(m, Decimal(1) / self.N))
                normals.append(scaled(n, 1 / dot(n, n).sqrt()))
        return points, normals


def reflect(m, wall):
    """A domain wall's reflection of a vector (sphere_chart.WALLS: z = 0, y = z, x = y), exactly: a sign change or a
    swap"""
    if wall == 0:
        return (m[0], m[1], -m[2])
    if wall == 1:
        return (m[0], m[2], m[1])
    return (m[1], m[0], m[2])


_placers = {}        # (world text, digits) -> Placer


def placer(world, digits):
    """The world's Placer at digits digits, kept"""
    return kept(_placers, (world.text(), int(digits)), 16, lambda: Placer(world, digits))


def forget():
    _directions.clear()
    _placers.clear()
    _constants.clear()
