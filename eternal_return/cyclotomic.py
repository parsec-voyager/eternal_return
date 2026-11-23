"""Exact points of N-fold tilings, for N = 8 (Ammann-Beenker) and N = 10 (Penrose, girih), as one engine.

A point is a cyclotomic integer of Z[zeta], zeta = e^(2 pi i / N): a tuple (a0, a1, a2, a3) of Python ints meaning
a0 + a1 zeta + a2 zeta^2 + a3 zeta^3. Both rings have degree 4, so four ints do: zeta^4 is -1 for N = 8 and
-1 + zeta - zeta^2 + zeta^3 for N = 10 (zeta^4 - zeta^3 + zeta^2 - zeta + 1 = 0). Every vertex of a tiling with unit
edges in those directions is one, and so is where every tile is at every level, in that level's own units, as long as
the loop factor is a unit of the ring: 1 + sqrt 2 for N = 8, the golden ratio phi for N = 10. Then scaling by it either
way is exact, and so are turns by 360 / N degrees and mirroring. A factor that's in the ring but isn't a unit (darb_imam's
2 phi^3) scales up exactly, but down it puts powers of 2 under the coefficients: they may be Fractions, which xy_float()
and mul() take (tidy() writes whole ones back as ints).

Coordinates: x is a + b sqrt d (d = 2 or 5), and y is that times STRETCH (1 for N = 8, sin 36 degrees for N = 10, which
isn't in Q(sqrt 5)). xy_float() turns them into floats accurate to the last bit however big a and b are: deep in a dive
a tile's place a few units from the map's origin is written with thousand-bit coefficients that cancel, and plain float
arithmetic would make it garbage. The Ammann-Beenker prototype's exact.py (~/blender_tests/substitution) is the N = 8
original; the girih prototype's test_penrose.py (~/blender_tests/girih) checks this agrees with it. Used by
substitution.py's endless girih floors (penrose.py, darb_imam.py), and with the Q(sqrt 5) numbers below by the Penrose
surface maps (surface_substitution.py, parts_substitution.py).
"""

import functools
import math
from fractions import Fraction


class Ring:
    def __init__(self, n):
        assert n in (8, 10), "Z[zeta] of degree 4: N = 8 or 10"
        self.n = n
        # zeta^4 in the basis
        self.fourth = (-1, 0, 0, 0) if n == 8 else (-1, 1, -1, 1)
        self.zero, self.one = (0, 0, 0, 0), (1, 0, 0, 0)
        self.zeta = (0, 1, 0, 0)
        self.inverse_zeta = self.power(n - 1)
        self.d = 2 if n == 8 else 5
        # Each basis power's x as (a, b): a + b sqrt d, and y as (a, b) times STRETCH
        if n == 8:
            half = Fraction(1, 2)
            self.re = [(1, 0), (0, half), (0, 0), (0, -half)]
            self.im = [(0, 0), (0, half), (1, 0), (0, half)]
            self.stretch = 1.0
        else:
            quarter = Fraction(1, 4)
            phi = (Fraction(1, 2), Fraction(1, 2))     # 2 cos 36 = (1 + sqrt 5) / 2, and sin 72 = sin 36 phi
            self.re = [(1, 0), (quarter, quarter), (-quarter, quarter), (quarter, -quarter)]
            self.im = [(0, 0), (1, 0), phi, phi]
            self.stretch = math.sqrt((5.0 - math.sqrt(5.0)) / 8.0)     # sin 36
        self.sqrt_d = math.sqrt(self.d)
        # The same as whole numbers over 4: x = (P + Q sqrt d) / 4, P = sum a_i xp_i, Q = sum a_i xq_i, and y likewise
        self.xp = [int(4 * Fraction(r[0])) for r in self.re]
        self.xq = [int(4 * Fraction(r[1])) for r in self.re]
        self.yp = [int(4 * Fraction(r[0])) for r in self.im]
        self.yq = [int(4 * Fraction(r[1])) for r in self.im]
        assert all(4 * Fraction(c) == int(4 * Fraction(c)) for r in self.re + self.im for c in r)

    # --- Arithmetic ---------------------------------------------------------------------------------------------

    def add(self, a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])

    def sub(self, a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3])

    def neg(self, a):
        return (-a[0], -a[1], -a[2], -a[3])

    def scale(self, a, k):
        return (a[0] * k, a[1] * k, a[2] * k, a[3] * k)

    def times_zeta(self, a):
        f = self.fourth
        return (a[3] * f[0], a[0] + a[3] * f[1], a[1] + a[3] * f[2], a[2] + a[3] * f[3])

    def rot(self, a, k):
        """a times zeta^k: turned k * 360 / N degrees counterclockwise"""
        k %= self.n
        if k >= self.n // 2:            # zeta^(N/2) = -1
            a, k = self.neg(a), k - self.n // 2
        for _ in range(k):
            a = self.times_zeta(a)
        return a

    def power(self, k):
        return self.rot(self.one, k)

    def mul(self, a, b):
        if not (type(a[0]) is int and type(a[1]) is int and type(a[2]) is int and type(a[3]) is int):
            # Fractions (darb_imam.py's places): in whole numbers over their common denominator, then back, many times
            # faster than sixteen products of Fractions
            common = math.lcm(*(Fraction(c).denominator for c in a))
            if all(type(c) is int for c in b):
                whole = self.mul(tuple(int(c * common) for c in a), b)
                return tuple(c // common if c % common == 0 else Fraction(c, common) for c in whole)
        product = self.zero
        for i in range(4):
            if a[i]:
                product = self.add(product, self.scale(self.rot(b, i), a[i]))
        return product

    def conj(self, a):
        """The mirror image across the x axis (complex conjugate): zeta -> zeta^-1"""
        result, power = self.zero, self.one
        for i in range(4):
            if a[i]:
                result = self.add(result, self.scale(power, a[i]))
            power = self.mul(power, self.inverse_zeta)
        return result

    # --- Coordinates --------------------------------------------------------------------------------------------

    def exact_xy(self, a):
        """a's coordinates as ((a, b), (a, b)) Fractions: x = a + b sqrt d, y = (a + b sqrt d) * stretch"""
        x = [Fraction(0), Fraction(0)]
        y = [Fraction(0), Fraction(0)]
        for i in range(4):
            if a[i]:
                for part in (0, 1):
                    x[part] += a[i] * self.re[i][part]
                    y[part] += a[i] * self.im[i][part]
        return tuple(x), tuple(y)

    def xy_float(self, a):
        """a's coordinates as accurate floats, whatever its coefficients: in whole numbers (4 x = P + Q sqrt d, and the
        same for y over its stretch), many times faster than with Fractions, which took most of the add-on's time
        working out where giant tiles are. Coefficients may be Fractions (darb_imam.py's places, whose loop factor
        isn't a unit): then over their common denominator."""
        den = 4
        if not (type(a[0]) is int and type(a[1]) is int and type(a[2]) is int and type(a[3]) is int):
            common = math.lcm(*(Fraction(c).denominator for c in a))
            a = tuple(int(c * common) for c in a)
            den *= common
        px = a[0] * self.xp[0] + a[1] * self.xp[1] + a[2] * self.xp[2] + a[3] * self.xp[3]
        qx = a[0] * self.xq[0] + a[1] * self.xq[1] + a[2] * self.xq[2] + a[3] * self.xq[3]
        py = a[0] * self.yp[0] + a[1] * self.yp[1] + a[2] * self.yp[2] + a[3] * self.yp[3]
        qy = a[0] * self.yq[0] + a[1] * self.yq[1] + a[2] * self.yq[2] + a[3] * self.yq[3]
        return float_sqrt_int(px, qx, self.d, den), float_sqrt_int(py, qy, self.d, den) * self.stretch

    def xy_fixed(self, a, bits):
        """a's coordinates times 2^bits as whole numbers, to within a unit or two, whatever its coefficients: for
        differences of places that cancel far past a float's 53 bits (substitution.py's giant levels, whose tiles are
        1e12 and more of the drone across, drawn exactly round it)"""
        den = 4
        if not (type(a[0]) is int and type(a[1]) is int and type(a[2]) is int and type(a[3]) is int):
            common = math.lcm(*(Fraction(c).denominator for c in a))
            a = tuple(int(c * common) for c in a)
            den *= common
        px = a[0] * self.xp[0] + a[1] * self.xp[1] + a[2] * self.xp[2] + a[3] * self.xp[3]
        qx = a[0] * self.xq[0] + a[1] * self.xq[1] + a[2] * self.xq[2] + a[3] * self.xq[3]
        py = a[0] * self.yp[0] + a[1] * self.yp[1] + a[2] * self.yp[2] + a[3] * self.yp[3]
        qy = a[0] * self.yq[0] + a[1] * self.yq[1] + a[2] * self.yq[2] + a[3] * self.yq[3]
        y = fixed_sqrt_int(py, qy, self.d, den, bits)
        return fixed_sqrt_int(px, qx, self.d, den, bits), (y * self.stretch_fixed(bits)) >> bits

    def stretch_fixed(self, bits):
        """STRETCH times 2^bits, rounded down: 1, or sin 36 degrees = sqrt((5 - sqrt 5) / 8)"""
        if self.n == 8:
            return 1 << bits
        return math.isqrt((((5 << bits) - math.isqrt(5 << (2 * bits))) << bits) >> 3)

    def xy_small(self, a):
        """a's coordinates as plain floats, for coefficients under about 2^50"""
        (xa, xb), (ya, yb) = self.exact_xy(a)
        return float(xa) + float(xb) * self.sqrt_d, (float(ya) + float(yb) * self.sqrt_d) * self.stretch

    def tidy(self, a):
        """a with whole Fractions written as ints, which are many times faster to work with"""
        return tuple(c.numerator if type(c) is Fraction and c.denominator == 1 else c for c in a)

    def unit_index(self, a):
        """k if a is exactly zeta^k, else None"""
        x, y = self.xy_small(a)
        k = round(math.atan2(y, x) / (2 * math.pi / self.n)) % self.n
        return k if self.power(k) == a else None


def float_sqrt_int(p, q, d, den, bits=64):
    """(p + q sqrt d) / den, whole numbers, as the nearest float, to within 2^-bits of its own size, however big p and
    q are (see float_sqrt()). +-inf past float range."""
    shift = bits + 2 * max(p.bit_length(), q.bit_length()) + 6
    root = math.isqrt(d * q * q << (2 * shift))
    num = (p << shift) + (root if q >= 0 else -root)
    try:
        return num / (den << shift)
    except OverflowError:
        return math.inf if num > 0 else -math.inf


def fixed_sqrt_int(p, q, d, den, bits):
    """(p + q sqrt d) / den, whole numbers, times 2^bits, rounded down (to within a unit or two)"""
    root = math.isqrt(d * q * q << (2 * bits))
    return ((p << bits) + (root if q >= 0 else -root)) // den


def float_sqrt(a, b, d, bits=64):
    """a + b sqrt d (ints or Fractions) as the nearest float, to within 2^-bits of its own size, however big a and b
    are: |a + b sqrt d| is at least about 1 / (2 sqrt d max(|a|, |b|)) when not 0, so that many more bits do it.
    +-inf past float range."""
    a, b = Fraction(a), Fraction(b)
    den = a.denominator * b.denominator
    big_a = a.numerator * b.denominator
    big_b = b.numerator * a.denominator
    shift = bits + 2 * max(big_a.bit_length(), big_b.bit_length()) + 6
    root = math.isqrt(d * big_b * big_b << (2 * shift))
    num = (big_a << shift) + (root if big_b >= 0 else -root)
    try:
        return num / (den << shift)
    except OverflowError:
        return math.inf if num > 0 else -math.inf


# ----- Q(sqrt 5), N = 10's coordinates: exact numbers (a, b), ints or Fractions, meaning a + b sqrt 5 -----

def q_add(x, y):
    return x[0] + y[0], x[1] + y[1]


def q_sub(x, y):
    return x[0] - y[0], x[1] - y[1]


def q_mul(x, y):
    return x[0] * y[0] + 5 * x[1] * y[1], x[0] * y[1] + x[1] * y[0]


def q_scale(x, k):
    return x[0] * k, x[1] * k


def q_div(x, y):
    """x / y (y not 0): times y's conjugate over its norm"""
    norm = Fraction(y[0] * y[0] - 5 * y[1] * y[1])
    return (Fraction(x[0] * y[0] - 5 * x[1] * y[1]) / norm, Fraction(x[1] * y[0] - x[0] * y[1]) / norm)


def q_float(x):
    return float_sqrt(x[0], x[1], 5)


def q_fraction(x, bits):
    """a + b sqrt 5 as a Fraction, sqrt 5 to bits bits"""
    root = Fraction(math.isqrt(5 << (2 * bits)), 1 << bits)
    return x[0] + x[1] * root


Q_PHI = (Fraction(1, 2), Fraction(1, 2))               # phi = (1 + sqrt 5) / 2
Q_PHI_INV = (Fraction(-1, 2), Fraction(1, 2))          # 1 / phi = (sqrt 5 - 1) / 2
_phi_powers = {0: (Fraction(1), Fraction(0))}


def q_phi(d):
    """phi^d exactly, d of either sign. Every 16th power is kept, and the asked ones: a level asks for its own every
    build."""
    found = _phi_powers.get(d)
    if found is None:
        step, factor = (1, Q_PHI) if d > 0 else (-1, Q_PHI_INV)
        nearest = max((k for k in list(_phi_powers) if 0 <= k * step <= d * step), key=abs)
        found = _phi_powers[nearest]
        for k in range(nearest + step, d + step, step):
            found = q_mul(found, factor)
            if k % 16 == 0 or k == d:
                _phi_powers[k] = found
    return found


@functools.lru_cache(maxsize=4096)
def phi_power_float(d, shift=0):
    """2^shift phi^-d as a float, accurate to the last bit however big d and shift are (each alone overflows past
    1000 or so): phi^d = (L + F sqrt 5) / 2, Lucas and Fibonacci numbers, in whole numbers. Kept: a settle asks for
    the same few dozen some 200 times (1 ms of every tick deep down, 2026-11-14)."""
    a, b = 2, 0                          # 2 phi^k = a + b sqrt 5
    x, y, k = 1, 1, d                    # 2 phi = 1 + sqrt 5, squared as we go
    while k:
        if k & 1:
            a, b = (a * x + 5 * b * y) // 2, (a * y + b * x) // 2
        x, y = (x * x + 5 * y * y) // 2, (x * y + y * x) // 2
        k >>= 1
    # 2 phi^d = a + b sqrt 5 > 0; its float as mantissa and exponent
    bits = 64
    shift_bits = bits + 2 * max(a.bit_length(), b.bit_length())
    big = (a << shift_bits) + math.isqrt(5 * b * b << (2 * shift_bits))          # (2 phi^d) 2^shift_bits
    e = big.bit_length()
    mantissa = (big >> max(e - 64, 0)) / 2.0 ** min(e, 64)                     # in [0.5, 1)
    exponent = e - shift_bits - 1                                              # phi^d = mantissa 2^exponent
    return math.ldexp(1.0 / mantissa, shift - exponent)


# Z[zeta_10] coefficients (a0, a1, a2, a3) in Z[phi]: x = Re(z) and y = Im(z) / sin 36 as p + q phi, whole numbers
# (phi = 1 + zeta^2 - zeta^3, so no halves: 2 Re(z) for x)

def im_pair(c):
    """Im(z) / sin 36 for ring coefficients c (..., 4), numpy arrays, as p + q phi: (p, q)"""
    return c[..., 1], c[..., 2] + c[..., 3]


def re2_pair(c):
    """2 Re(z) for ring coefficients c (..., 4), numpy arrays, as p + q phi: (p, q)"""
    return 2 * c[..., 0] - c[..., 2] + c[..., 3], c[..., 1] + c[..., 2] - c[..., 3]


def ring_im(z):
    """im_pair() of one ring element (four Python ints)"""
    return z[1], z[2] + z[3]


def ring_re2(z):
    """re2_pair() of one ring element"""
    return 2 * z[0] - z[2] + z[3], z[1] + z[2] - z[3]


RING8, RING10 = Ring(8), Ring(10)
