"""Conformal charts of the sphere for endless sphere maps (sphere.py), exact at any depth: Rančić, Purser and Mesinger
(1996)'s conformal cube, whose lattice runs straight across the cube's edges and pinches at its 8 corners.

The sphere is symmetric under the cube's 48 rotations and reflections, and so is its lattice (and the pattern drawn on
it), so everything is worked out in one fundamental triangle of them, the domain: on face +x, from the face's center to
its edge's midpoint to its corner, the directions with x >= y >= z >= 0. Any point of the sphere is folded into it by
reflecting it across the domain's three walls (the planes z = 0, y = z and x = y: a face's midline, its diagonal and
its edge), and folding never moves two points further apart, so a pixel folded into the domain is no further from the
anchor folded there than it was from the anchor.

In the domain, one series does: Rančić's, around the face's corner. A point P = (N, a, b) on the cube (N units from its
center to a face, a and b whole for the lattice's vertices, N >= a >= b >= 0) has

    z = ((N - a) + i (N - b)) / 2N,  Z = z^4,  W = sum A_k Z^k,  u = e^(i pi/6) (i W)^(1/3),
    w = (u - RA) / (CB + CC u)  (stereographic, from the face's center),  n = (1 - |w|^2, 2w) / (1 + |w|^2)

(n's x first, then the face's y + i z). Its real coefficients put the face's edge and diagonal exactly on their walls;
the midline is only as good as the coefficients, off by about 3e-16, which deep down would be a seam of 3e-16 N units.
So the anchor's distance from each wall is measured from the wall's nearest lattice point, its foot (wall_distances()):
the chart is then, in effect, Rančić's moved by those 3e-16 onto the walls, and folds join exactly.

Deep down every difference from the anchor is far smaller than the numbers it's a difference of. So the chart is worked
out as differences, each step of it by an identity with no cancellation (Q4's for z^4, divided differences for the
series, the cube root's (a - b) / (a^2 + ab + b^2), the Mobius map's and the stereographic projection's), and every value
is scaled by a power of 2 chosen for the anchor and the view (Reference): its absolute values by the anchor's distance
from the corner, which deep in a corner dive is far below 2^-1000, and its differences by the view's size. So the same
arithmetic holds from the globe down, in float64 here and float32 on the GPU (the shader in sphere.py does
inverse_difference() per pixel), with nothing smaller than about 2^-30 of what's kept.

Lattice points are exact Python ints. Points are given in units (a, b), offsets D as complex numbers D_a + i D_b.

The conformal icosahedron's chart (IcoReference, for sphere_ico.py) is at the end: the same arithmetic on its own
domain, 1/120 of the sphere.
"""

import cmath
import math
from fractions import Fraction

import numpy as np

# Rančić et al. (1996), Table B1: W(Z) = sum A[k] Z^k
A_RANCIC = (
    +0.00000000000000, +1.47713062600964, -0.38183510510174, -0.05573058001191, -0.00895883606818,
    -0.00791315785221, -0.00486625437708, -0.00329251751279, -0.00235481488325, -0.00175870527475,
    -0.00135681133278, -0.00107459847699, -0.00086944475948, -0.00071607115121, -0.00059867100093,
    -0.00050699063239, -0.00043415191279, -0.00037541003286, -0.00032741060100, -0.00028773091482,
    -0.00025458777519, -0.00022664642371, -0.00020289261022, -0.00018254510830, -0.00016499474461,
    -0.00014976117168, -0.00013646173946, -0.00012478875823, -0.00011449267279, -0.00010536946150,
    -0.00009725109376)
RA = math.sqrt(3.0) - 1.0
CB = complex(-1.0, 1.0)
CC = RA * CB / 2.0
I13 = cmath.exp(1j * math.pi / 6.0)
# The Mobius map's determinant, for its differences (and its inverse's): (u - RA) / (CB + CC u) has ad - bc = CB + CC RA
MOBIUS_DET = CB + CC * RA

# The domain's walls, inward normals (cube axes): the face's midline, its diagonal, its edge. Every mirror of the cube's
# symmetry is one of these turned by a symmetry, and a point of the domain is on the positive side of all three.
WALLS = np.array([(0.0, 0.0, 1.0), (0.0, math.sqrt(0.5), -math.sqrt(0.5)), (math.sqrt(0.5), -math.sqrt(0.5), 0.0)])
MIDLINE, DIAGONAL, EDGE = 0, 1, 2
# A wall further than FAR_WALL views from the anchor is never near anything in view: its distance is FAR_WALL_DISTANCE,
# which fits a float32
FAR_WALL = 2.0 ** 24
FAR_WALL_DISTANCE = 1.0e30
# Folding takes at most 9 reflections (the cube group's longest element), but rounding near a wall can ask for one more
FOLD_STEPS = 12


def fraction_float(numerator, denominator, exponent=0):
    """numerator / denominator * 2^exponent as a float, for ints of any size: exact to rounding, 0 below a float's
    range (never an overflow on the way)"""
    value = Fraction(numerator, denominator)
    value = value * (1 << exponent) if exponent >= 0 else value / (1 << -exponent)
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def is_far(dz):
    """Whether dz~ (views) is past FAR_WALL, for any size of dz~: abs() overflows on parts past half a float's range"""
    return max(abs(dz.real), abs(dz.imag)) > FAR_WALL or abs(dz) > FAR_WALL


def length(v):
    """|v| over the last axis (3), with no overflow or underflow on the way"""
    return np.hypot(np.hypot(v[..., 0], v[..., 1]), v[..., 2])


def log2_int(n):
    """floor(log2 n) for a positive int, and its fraction, as a float: log2 of any size of int"""
    shift = max(0, n.bit_length() - 60)
    return shift + math.log2(n >> shift)


def fold_point(P):
    """The cube point P (three ints, on the cube's surface) folded into the domain, exactly, and the signed permutation
    G (a 3 x 3 int matrix) that does it: G @ P is the folded point, x >= y >= z >= 0. Points on the domain's walls fold
    either way; the first sorted one is taken."""
    absolute = [abs(c) for c in P]
    order = sorted(range(3), key=lambda i: -absolute[i])
    G = np.zeros((3, 3), dtype=np.int64)
    for row, i in enumerate(order):
        G[row, i] = 1 if P[i] >= 0 else -1
    return [absolute[i] for i in order], G


def series_ratio(Z, terms=len(A_RANCIC)):
    """W(Z) / Z = sum A_k Z^(k-1), for absolute Z: the scaled series is then W~ = Z~ series_ratio(t Z~) (W = t W~ for
    Z = t Z~), which holds however small t, and never overflows"""
    q = np.zeros_like(np.asarray(Z, dtype=complex))
    for k in range(terms - 1, 0, -1):
        q = q * Z + A_RANCIC[k]
    return q


def series_slope(Z, terms=len(A_RANCIC)):
    """dW/dZ = sum k A_k Z^(k-1), for absolute Z (the same for the scaled series)"""
    dw = np.zeros_like(np.asarray(Z, dtype=complex))
    for k in range(terms - 1, 0, -1):
        dw = dw * Z + k * A_RANCIC[k]
    return dw


def series_divided(Z0, Z1, terms=len(A_RANCIC)):
    """(W(Z1) - W(Z0)) / (Z1 - Z0) for absolute Z0, Z1 (the same for the scaled series), with no cancellation however
    close they are: sum A_k times the divided difference of Z^k, sum over j of Z1^j Z0^(k-1-j)"""
    total = np.zeros(np.broadcast(Z0, Z1).shape, dtype=complex)
    divided = np.ones_like(total)          # Z^k's divided difference, k = 1
    power0 = np.ones_like(total)           # Z0^(k-1)
    for k in range(1, terms):
        if k > 1:
            power0 = power0 * Z0
            divided = divided * Z1 + power0
        total = total + A_RANCIC[k] * divided
    return total


def quartic_divided(z0, z1):
    """(z1^4 - z0^4) / (z1 - z0)"""
    return (z1 + z0) * (z1 * z1 + z0 * z0)


def fourth_root(Z):
    """z with z^4 = Z in the domain's sector (arg z from 45 to 90 degrees, continued to 22.5 and 112.5 either side, so a
    point rounded just past a wall stays continuous)"""
    angle = np.angle(Z)
    angle = np.where(angle > math.pi / 2, angle, angle + 2.0 * math.pi)
    return np.abs(Z) ** 0.25 * np.exp(0.25j * angle)


class Reference:
    """The chart at an exact point of the domain, the anchor A = (N, a, b), N = 2^n, with every value scaled for
    differences of up to about view_units (units) from it (see the module's docstring):

    - absolute values at A: z, Z, W and v scaled by eps = 2^e (z = eps z~, Z = eps^4 Z~, W = eps^4 W~, v = eps^(4/3) v~),
      e the larger of log2 |z| and the view's; w and n as they are
    - differences: dz = delta dz~ (delta = 2^e_delta, the view's size in z), dZ and dW = delta eps^3 d~, dv, du, dw and
      dn = sigma d~ (sigma = delta eps^(1/3)), m = N dn = scale d~n
    """

    def __init__(self, n, a, b, view_units):
        self.n, self.a, self.b = int(n), int(a), int(b)
        N = 1 << self.n
        self.N = N
        assert N >= self.a >= self.b >= 0, (N, a, b)
        # e_delta: the view's size in z, a power of 2
        self.e_delta = math.floor(math.log2(max(view_units, 1e-300))) - 1 - self.n
        # e: A's distance from the corner, |z| = |(N - a) + i (N - b)| / 2N, or the view's, whichever is bigger
        far = N - self.b
        e_anchor = (log2_int(far) - 1 - self.n) if far > 0 else -math.inf
        self.e = math.floor(max(e_anchor, self.e_delta))
        e, e_delta = self.e, self.e_delta
        self.z0 = complex(fraction_float(N - self.a, 2 * N, -e), fraction_float(N - self.b, 2 * N, -e))
        self.t = math.ldexp(1.0, 4 * e)                     # eps^4, 0 deep in a corner
        self.r = 2.0 ** (4.0 * e / 3.0)                     # eps^(4/3)
        self.rho1 = math.ldexp(1.0, e_delta - e)            # delta / eps, at most 1
        self.sigma = 2.0 ** (e_delta + e / 3.0)
        self.scale = 2.0 ** (self.n + e_delta + e / 3.0)    # N sigma: m = scale * dn~
        # D in units = unit_scale * dz~ (dz = -D / 2N)
        self.unit_scale = -math.ldexp(1.0, e_delta + 1 + self.n)
        self.Z0 = self.z0 ** 4
        self.W0 = self.Z0 * complex(series_ratio(self.t * self.Z0))
        self.v0 = (1j * self.W0) ** (1.0 / 3.0)
        self.u0 = I13 * self.v0
        self.w0 = (self.r * self.u0 - RA) / (CB + CC * self.r * self.u0)
        s0 = 1.0 + abs(self.w0) ** 2
        self.s0 = s0
        self.n0 = np.array([(2.0 - s0) / s0, 2.0 * self.w0.real / s0, 2.0 * self.w0.imag / s0])

    # ----- forward: an offset D (units) from A to m = N (n(A + D) - n(A)) -----

    def forward_difference(self, D):
        """m = N (n(A + D) - n(A)) (cube axes, (..., 3)) for offsets D (complex, units, along a + i b), with no
        cancellation, for points A + D in the domain (or just past its walls)"""
        return self.forward_scaled(np.asarray(D, dtype=complex) / self.unit_scale)

    def dz_of(self, da, db):
        """dz~ for an exact offset (da, db) in units, ints or Fractions of any size"""
        exponent = -(self.e_delta + 1 + self.n)
        da, db = Fraction(da), Fraction(db)
        return complex(fraction_float(-da.numerator, da.denominator, exponent),
                       fraction_float(-db.numerator, db.denominator, exponent))

    def forward_scaled(self, dz):
        """forward_difference() for offsets given as dz~ (D / unit_scale)"""
        dz = np.asarray(dz, dtype=complex)
        # Each stage's value at the point straight from the last's (z1^4, not Z0 + dZ): near the corner, where they're
        # small, a sum of the anchor's and the difference would keep only the digits of the anchor's
        z1 = self.z0 + self.rho1 * dz
        dZ = dz * quartic_divided(self.z0, z1)
        Z1 = z1 ** 4
        dW = dZ * series_divided(self.t * self.Z0, self.t * Z1)
        W1 = Z1 * series_ratio(self.t * Z1)
        v1 = (1j * W1) ** (1.0 / 3.0)
        # 0 / 0 only for no offset at the corner itself
        squares = v1 * v1 + v1 * self.v0 + self.v0 * self.v0
        dv = np.where(squares != 0, 1j * dW / np.where(squares != 0, squares, 1.0), 0.0)
        du = I13 * dv
        u1 = I13 * v1
        dw = MOBIUS_DET * du / ((CB + CC * self.r * u1) * (CB + CC * self.r * self.u0))
        return self.scale * self.stereo_difference(dw)

    def stereo_difference(self, dw):
        """dn~ (cube axes) for dw~"""
        w0, s0 = self.w0, self.s0
        ds = 2.0 * (np.conj(w0) * dw).real + self.sigma * np.abs(dw) ** 2
        s1 = s0 + self.sigma * ds
        dxy = 2.0 * (dw * s0 - w0 * ds) / (s0 * s1)
        dx = -2.0 * ds / (s0 * s1)
        return np.stack([dx, dxy.real, dxy.imag], axis=-1)

    # ----- inverse: m to D, as the GPU does it per pixel -----

    def inverse_difference(self, m, newton=4, jacobian=False, terms=len(A_RANCIC)):
        """The offsets D (complex, units) from A of the domain's points whose directions are n(A) + m / N, m (..., 3) in
        cube axes. With jacobian, also dD/dm~ as (..., 3) complex rows (D's change per unit of m / scale). Fewer terms
        of the series and Newton steps for estimates (a view's extent): 14 and 2 are good to float32's rounding."""
        m = np.asarray(m, dtype=float)
        dn = m / self.scale
        dX, dY, dZn = dn[..., 1], dn[..., 2], dn[..., 0]
        X0, Y0, Z0n = self.n0[1], self.n0[2], self.n0[0]
        dw = ((dX + 1j * dY) * (1.0 + Z0n) - (X0 + 1j * Y0) * dZn) / ((1.0 + Z0n + self.sigma * dZn) * (1.0 + Z0n))
        w1 = self.w0 + self.sigma * dw
        du = MOBIUS_DET * dw / ((1.0 - CC * w1) * (1.0 - CC * self.w0))
        dv = du / I13
        v1 = self.v0 + self.rho1 * dv
        dW = -1j * dv * (v1 * v1 + v1 * self.v0 + self.v0 * self.v0)
        # The series backward: Newton on dZ dW/dZ(Z0, Z0 + dZ) = dW
        dZ = dW / series_slope(self.t * self.Z0, terms)
        for _ in range(newton):
            Z1 = self.Z0 + self.rho1 * dZ
            dZ = dZ - (dZ * series_divided(self.t * self.Z0, self.t * Z1, terms) - dW) / series_slope(self.t * Z1, terms)
        Z1 = self.Z0 + self.rho1 * dZ
        z1 = fourth_root(Z1)
        q = quartic_divided(self.z0, z1)
        dz = np.where(np.abs(q) > 0, dZ / np.where(np.abs(q) > 0, q, 1.0), (z1 - self.z0) / self.rho1)
        D = self.unit_scale * dz
        if not jacobian:
            return D
        # dD/dn~ at the point: the chain's complex derivatives, then the stereographic projection's
        # At the cube's corner itself (z = 0) the stretch is infinite: inf, not a warning
        z1 = np.where(z1 == 0, 1e-300, z1)
        n1 = self.n0 + self.sigma * dn
        Xp, Yp, Zp = n1[..., 1], n1[..., 2], n1[..., 0]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            chain = (self.unit_scale / (4.0 * z1 ** 3 * series_slope(self.t * Z1, terms)) * (-3j * v1 * v1) / I13
                     * MOBIUS_DET / (1.0 - CC * w1) ** 2)
        # dw = ((dX + i dY)(1 + Z) - (X + iY) dZ) / (1 + Z)^2, n's components (x, y, z) = (Z, X, Y)
        rows = np.stack([-(Xp + 1j * Yp), 1.0 + Zp, 1j * (1.0 + Zp)], axis=-1) / ((1.0 + Zp) ** 2)[..., None]
        return D, chain[..., None] * rows

    # ----- the walls -----

    def wall_distances(self):
        """N times the distance of n(A) from each of the domain's walls (WALLS), each measured from the wall's foot
        nearest A along the face, so the chart meets the walls exactly (see the module's docstring): never negative.
        A wall more than FAR_WALL views away is FAR_WALL_DISTANCE: nothing in view is near it."""
        N, a, b = self.N, self.a, self.b
        feet = [(MIDLINE, 0, -b), (DIAGONAL, Fraction(b - a, 2), Fraction(a - b, 2)), (EDGE, N - a, 0)]
        out = np.zeros(3)
        for wall, da, db in feet:
            if da == 0 and db == 0:
                continue
            dz = self.dz_of(da, db)
            if is_far(dz):
                out[wall] = FAR_WALL_DISTANCE
                continue
            out[wall] = max(0.0, -float(self.forward_scaled(dz) @ WALLS[wall]))
        return out


class Charts:
    """The two references a view needs: the anchor's, and the domain's corner's (the cube's corner, N, N, N), where the
    chart is singular: next to it z^4 and the cube root squeeze offsets, and a point there measured from an anchor
    further away loses most of its digits on the way back (taking z = Z^(1/4) of a Z that is a difference). Measured from
    the corner instead, it's exact (z0 = 0). So every point closer to the corner than half its distance from the anchor
    uses the corner's reference (choose()); m_corner is the corner's offset from the anchor."""

    def __init__(self, n, a, b, view_units, corner_view_units=None):
        self.anchor = Reference(n, a, b, view_units)
        N = 1 << n
        self.corner = Reference(n, N, N, corner_view_units or view_units)
        if a == N and b == N:
            self.m_corner = np.zeros(3)
        else:
            dz = self.anchor.dz_of(N - a, N - b)
            self.m_corner = np.full(3, FAR_WALL_DISTANCE) if is_far(dz) else self.anchor.forward_scaled(dz)

    def choose(self, mf):
        """Which reference each folded offset uses: True for the corner. Lengths by hypot, not squared: views past 2^512
        units square past a float's range, and inf < inf would never choose the corner."""
        mf = np.asarray(mf, dtype=float)
        return length(mf - self.m_corner) < 0.5 * length(mf)

    def inverse(self, mf, jacobian=False, estimate=False):
        """inverse_difference() through whichever reference each point uses: D from that reference, (J,) and the mask
        of points measured from the corner. estimate: with fewer terms and steps (see inverse_difference())."""
        mf = np.asarray(mf, dtype=float)
        use = self.choose(mf)
        options = {"newton": 2, "terms": 14} if estimate else {}
        D = np.zeros(use.shape, dtype=complex)
        J = np.zeros(use.shape + (3,), dtype=complex)
        for mask, ref, offset in ((~use, self.anchor, 0.0), (use, self.corner, self.m_corner)):
            if not mask.any():
                continue
            out = ref.inverse_difference(mf[mask] - offset, jacobian=jacobian, **options)
            if jacobian:
                # A corner's reference scaled for a far smaller view than its points' (a first guess, see
                # sphere.inverse_from_guess()) can have underflowed to a scale of 0: its points' D are inf too
                D[mask], J[mask] = out[0], out[1] * (self.anchor.scale / ref.scale if ref.scale else math.inf)
            else:
                D[mask] = out
        return (D, J, use) if jacobian else (D, use)


def fold_offsets(m, walls_s, steps=FOLD_STEPS, normals=WALLS):
    """Folds points, given by their offsets m (..., 3, cube axes, N n - N n(A)) from the anchor A in the domain, into
    the domain, by reflecting each across the wall it's furthest on the wrong side of until it's on none's (walls_s:
    A's distances from them, N times; normals: the walls', the cube's or the icosahedron's), as the GPU does it. Returns
    the folded offsets and each point's reflection (a 3 x 3 matrix: folded = R @ point, about the sphere's center).

    Any wall a point is past is a step of its fold, but a point on a wall's plane can be rounded a fraction of a float's
    step past it, where its reflection rounds back to itself: taking the walls in order, it would never get to the ones
    it's really past. So the furthest first, and a point its reflection doesn't move is on its walls, to rounding."""
    m = np.array(m, dtype=float)
    flat = m.reshape(-1, 3)
    R = np.broadcast_to(np.eye(3), (len(flat), 3, 3)).copy()
    normals = np.asarray(normals, dtype=float)
    rows = np.arange(len(flat))
    for _ in range(steps):
        d = walls_s[None, :] + flat @ normals.T           # (n, 3)
        k = np.argmin(d, axis=1)                          # the wall each is furthest past
        dk = d[rows, k]
        nu = normals[k]
        folded = flat - np.minimum(2.0 * dk, 0.0)[:, None] * nu
        has = (dk < 0.0) & np.any(folded != flat, axis=1)
        if not has.any():
            break
        flat = np.where(has[:, None], folded, flat)
        reflect = np.eye(3)[None] - 2.0 * nu[:, :, None] * nu[:, None, :]
        R = np.where(has[:, None, None], reflect @ R, R)
    return flat.reshape(m.shape), R.reshape(m.shape[:-1] + (3, 3))


# ----- the icosahedron's chart -----
#
# The conformal icosahedron (prototyped in ~/blender_tests/sphere/icosa.py): 20 equilateral faces, each mapped onto its
# spherical triangle keeping every angle. The sphere and its lattice are symmetric under the icosahedron's 120 rotations
# and reflections, so the domain is 1/120 of it: on one face, from a corner V (an icosahedron vertex) to the middle of
# its edge to the next corner V1, to the face's center. In the face's plane (side 1, V at 0, V1 at 1, the third corner V2
# at e^(i pi/3)) that's 0 <= arg xi <= 30 degrees, Re xi <= 1/2; a lattice point (i + j e^(i pi/3)) / N, i and j whole,
# is in it when N - i - j >= i >= j >= 0: its weights on V, V1 and V2, sorted. Around V, one series:
#
#     T = xi^6,  G = sum g_k T^k = T H(T)^5,  w = G^(1/5)  (stereographic, from V),  n = (1 - |w|^2, 2w) / (1 + |w|^2)
#
# (n's x along V, then the face's y + i z, y toward V1), H the prototype's per-corner series (w = xi^(6/5) H(xi^6)). Its
# real coefficients put the edge (arg xi = 0) and the altitude through V (30 degrees, arg w = 36) exactly on their walls;
# the third, the edge's perpendicular bisector (Re xi = 1/2, where V1's series takes over), only as well as the series
# was fitted (neighboring corners' series agree to 1e-12), so its distance is measured from its foot too. The same
# cancellation-free differences as the cube's: sextic and quintic divided differences for xi^6 and the fifth root, the
# series' divided differences, the stereographic projection's; no Mobius map, the projection is centered on V already.

# H, fitted so neighboring corners' series agree where their regions meet (icosa.fit_consistent()'s, refit to 1.4e-14 with
# a better Jacobian: the prototype's agreed to 1.2e-12). Its last terms are below float64's reach at |T| <= 1/27.
H_ICO = (0.6530242508183581, -0.024372523536082654, -0.004759875203873396, -0.001913890601164128,
         -0.0010145893538036496, -0.0006179704191403955, -0.0003530049290494886, -0.0001814755761356314,
         -0.00010859168717732942, -6.210438351560339e-05, -3.327736134211878e-05, -1.6367576441270294e-05,
         -7.143253484342688e-06, -2.627109594875171e-06)


def series_fifth_power(h, terms):
    """G = T H(T)^5's coefficients, g[0] = 0, up to T^(terms - 1)"""
    power = [1.0] + [0.0] * (terms - 1)
    for _ in range(5):
        power = [sum(power[k - m] * h[m] for m in range(min(k, len(h) - 1) + 1)) for k in range(terms)]
    return (0.0,) + tuple(power[:terms - 1])


G_ICO = series_fifth_power(H_ICO, 16)
ICO_SIDE = math.atan(2.0)                       # an edge, as an angle at the center
ICO_ALTITUDE = math.radians(36.0)               # the altitude through V, in the w plane
ROOT3_HALF = math.sqrt(3.0) / 2.0
# The domain's walls, inward normals (map axes, the domain's frame): the edge V V1 (z = 0), the altitude through V, the
# edge's perpendicular bisector (the plane between V and V1)
ICO_WALLS = np.array([(0.0, 0.0, 1.0), (0.0, math.sin(ICO_ALTITUDE), -math.cos(ICO_ALTITUDE)),
                      (math.sin(ICO_SIDE / 2.0), -math.cos(ICO_SIDE / 2.0), 0.0)])
ICO_EDGE, ICO_ALTITUDE_WALL, ICO_BISECTOR = 0, 1, 2
# The icosahedral group's longest element takes 15 reflections
ICO_FOLD_STEPS = 18


def ico_series_ratio(T, terms=len(G_ICO)):
    """G(T) / T"""
    q = np.zeros_like(np.asarray(T, dtype=complex))
    for k in range(terms - 1, 0, -1):
        q = q * T + G_ICO[k]
    return q


def ico_series_slope(T, terms=len(G_ICO)):
    dg = np.zeros_like(np.asarray(T, dtype=complex))
    for k in range(terms - 1, 0, -1):
        dg = dg * T + k * G_ICO[k]
    return dg


def ico_series_divided(T0, T1, terms=len(G_ICO)):
    """(G(T1) - G(T0)) / (T1 - T0), with no cancellation however close they are"""
    total = np.zeros(np.broadcast(T0, T1).shape, dtype=complex)
    divided = np.ones_like(total)
    power0 = np.ones_like(total)
    for k in range(1, terms):
        if k > 1:
            power0 = power0 * T0
            divided = divided * T1 + power0
        total = total + G_ICO[k] * divided
    return total


def power_divided(x0, x1, p):
    """(x1^p - x0^p) / (x1 - x0) = sum x1^j x0^(p-1-j): no cancellation while x0 and x1 are within 360 / p degrees"""
    total = np.zeros(np.broadcast(x0, x1).shape, dtype=complex)
    for j in range(p):
        total = total * x1 + x0 ** j
    return total


def sector_root(Z, p):
    """x with x^p = Z, arg Z taken in (-90, 270] degrees (the domain's Z lie between 0 and 180), so a point rounded just
    past a wall stays continuous"""
    angle = np.angle(Z)
    angle = np.where(angle > -math.pi / 2, angle, angle + 2.0 * math.pi)
    return np.abs(Z) ** (1.0 / p) * np.exp(1j * angle / p)


class IcoReference:
    """The icosahedron's chart at an exact point of the domain, the anchor (i, j) on the lattice of N = 2^n (its
    weights on V1 and V2), scaled as Reference is: absolute values by eps = 2^e, the larger of the anchor's distance
    from V and the view's (xi = eps xi~, T and G = eps^6 ~, w = eps^(6/5) w~), differences by delta = 2^e_delta, the
    view's size in xi (dxi = delta dxi~, dT and dG = delta eps^5 ~, dw and dn = sigma ~ with sigma = delta eps^(1/5)),
    m = N dn = scale dn~. Offsets D in units: N dxi, along the face's x + i y (a lattice step along i is 1, along j
    e^(i pi/3))."""

    def __init__(self, n, i, j, view_units):
        self.n, self.a, self.b = int(n), int(i), int(j)
        N = 1 << self.n
        self.N = N
        assert N - self.a - self.b >= self.a >= self.b >= 0, (N, i, j)
        self.e_delta = math.floor(math.log2(max(view_units, 1e-300))) - self.n
        # |xi| >= i / N (the weights are sorted, so i is the bigger), and at most 2 i / N
        e_anchor = (log2_int(self.a) - self.n) if self.a > 0 else -math.inf
        self.e = math.floor(max(e_anchor, self.e_delta))
        e, e_delta = self.e, self.e_delta
        self.xi0 = self.xi_of(self.a, self.b, -e - self.n)
        self.t = 2.0 ** (6 * e) if 6 * e > -1074 else 0.0         # eps^6, 0 deep at V
        self.r = 2.0 ** (6.0 * e / 5.0)                             # eps^(6/5)
        self.rho1 = math.ldexp(1.0, e_delta - e)                    # delta / eps, at most 1
        self.sigma = 2.0 ** (e_delta + e / 5.0)
        self.scale = 2.0 ** (self.n + e_delta + e / 5.0)
        self.unit_scale = math.ldexp(1.0, e_delta + self.n)
        self.T0 = self.xi0 ** 6
        self.G0 = self.T0 * complex(ico_series_ratio(self.t * self.T0))
        self.w0t = complex(sector_root(self.G0, 5))                # w~ at the anchor
        self.w0 = self.r * self.w0t
        s0 = 1.0 + abs(self.w0) ** 2
        self.s0 = s0
        self.n0 = np.array([(2.0 - s0) / s0, 2.0 * self.w0.real / s0, 2.0 * self.w0.imag / s0])

    @staticmethod
    def xi_of(di, dj, exponent):
        """(di + dj e^(i pi/3)) * 2^exponent as a complex float, for ints or Fractions of any size"""
        di, dj = Fraction(di), Fraction(dj)
        x = di + dj / 2
        return complex(fraction_float(x.numerator, x.denominator, exponent),
                       ROOT3_HALF * fraction_float(dj.numerator, dj.denominator, exponent))

    def dz_of(self, di, dj):
        """dxi~ for an exact lattice offset (di, dj), ints or Fractions of any size"""
        return self.xi_of(di, dj, -(self.e_delta + self.n))

    @staticmethod
    def lattice_offset(D):
        """D (units, the face's x + i y) as lattice steps (di, dj): floats"""
        D = np.asarray(D, dtype=complex)
        dj = D.imag / ROOT3_HALF
        return D.real - dj / 2.0, dj

    def forward_difference(self, D):
        return self.forward_scaled(np.asarray(D, dtype=complex) / self.unit_scale)

    def forward_scaled(self, dxi):
        dxi = np.asarray(dxi, dtype=complex)
        xi1 = self.xi0 + self.rho1 * dxi
        dT = dxi * power_divided(self.xi0, xi1, 6)
        T1 = xi1 ** 6
        dG = dT * ico_series_divided(self.t * self.T0, self.t * T1)
        G1 = T1 * ico_series_ratio(self.t * T1)
        w1 = sector_root(G1, 5)
        sums = power_divided(self.w0t, w1, 5)
        dw = np.where(sums != 0, dG / np.where(sums != 0, sums, 1.0), 0.0)
        return self.scale * stereo_difference(self.w0, self.s0, self.sigma, dw)

    def inverse_difference(self, m, newton=4, jacobian=False, terms=len(G_ICO)):
        m = np.asarray(m, dtype=float)
        dn = m / self.scale
        dX, dY, dZn = dn[..., 1], dn[..., 2], dn[..., 0]
        X0, Y0, Z0n = self.n0[1], self.n0[2], self.n0[0]
        dw = ((dX + 1j * dY) * (1.0 + Z0n) - (X0 + 1j * Y0) * dZn) / ((1.0 + Z0n + self.sigma * dZn) * (1.0 + Z0n))
        w1 = self.w0t + self.rho1 * dw
        dG = dw * power_divided(self.w0t, w1, 5)
        dT = dG / ico_series_slope(self.t * self.T0, terms)
        for _ in range(newton):
            T1 = self.T0 + self.rho1 * dT
            dT = dT - (dT * ico_series_divided(self.t * self.T0, self.t * T1, terms) - dG) / ico_series_slope(
                self.t * T1, terms)
        T1 = self.T0 + self.rho1 * dT
        xi1 = sector_root(T1, 6)
        q = power_divided(self.xi0, xi1, 6)
        dxi = np.where(np.abs(q) > 0, dT / np.where(np.abs(q) > 0, q, 1.0), (xi1 - self.xi0) / self.rho1)
        D = self.unit_scale * dxi
        if not jacobian:
            return D
        xi1 = np.where(xi1 == 0, 1e-300, xi1)
        n1 = self.n0 + self.sigma * dn
        Xp, Yp, Zp = n1[..., 1], n1[..., 2], n1[..., 0]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            chain = self.unit_scale * 5.0 * w1 ** 4 / (6.0 * xi1 ** 5 * ico_series_slope(self.t * T1, terms))
        rows = np.stack([-(Xp + 1j * Yp), 1.0 + Zp, 1j * (1.0 + Zp)], axis=-1) / ((1.0 + Zp) ** 2)[..., None]
        return D, chain[..., None] * rows

    def wall_distances(self):
        """As Reference.wall_distances(): N times n(A)'s distance from each wall, from the wall's foot nearest A"""
        N, i, j = self.N, self.a, self.b
        feet = [(ICO_EDGE, 0, -j), (ICO_ALTITUDE_WALL, Fraction(j - i, 2), Fraction(i - j, 2)),
                (ICO_BISECTOR, Fraction(N - j, 2) - i, 0)]
        out = np.zeros(3)
        for wall, di, dj in feet:
            if di == 0 and dj == 0:
                continue
            dz = self.dz_of(di, dj)
            if is_far(dz):
                out[wall] = FAR_WALL_DISTANCE
                continue
            out[wall] = max(0.0, -float(self.forward_scaled(dz) @ ICO_WALLS[wall]))
        return out


def stereo_difference(w0, s0, sigma, dw):
    """dn~ (the domain's axes) for dw~, the stereographic projection's difference at w0 (absolute, s0 = 1 + |w0|^2)"""
    ds = 2.0 * (np.conj(w0) * dw).real + sigma * np.abs(dw) ** 2
    s1 = s0 + sigma * ds
    dxy = 2.0 * (dw * s0 - w0 * ds) / (s0 * s1)
    dx = -2.0 * ds / (s0 * s1)
    return np.stack([dx, dxy.real, dxy.imag], axis=-1)


class IcoCharts(Charts):
    """Charts for the icosahedron: the anchor's reference and V's (the domain's corner, (0, 0)), chosen as Charts does"""

    def __init__(self, n, i, j, view_units, corner_view_units=None):
        self.anchor = IcoReference(n, i, j, view_units)
        self.corner = IcoReference(n, 0, 0, corner_view_units or view_units)
        if i == 0 and j == 0:
            self.m_corner = np.zeros(3)
        else:
            dz = self.anchor.dz_of(-i, -j)
            self.m_corner = np.full(3, FAR_WALL_DISTANCE) if is_far(dz) else self.anchor.forward_scaled(dz)


def ico_geometry():
    """The icosahedron in the domain's frame: its 12 vertices (V = (1, 0, 0) first, V1 in the xy plane second, V2
    third: the domain's face), its 20 faces (vertex triples, counterclockwise seen from outside), every vertex reached
    from V by the walls' reflections"""
    def reflect(v, nu):
        return v - 2.0 * (v @ nu) * nu
    V = np.array([1.0, 0.0, 0.0])
    V1 = np.array([math.cos(ICO_SIDE), math.sin(ICO_SIDE), 0.0])
    found = [V, V1]
    frontier = [V, V1]
    while frontier:
        new = []
        for v in frontier:
            for nu in ICO_WALLS:
                u = reflect(v, nu)
                if all(np.linalg.norm(u - f) > 1e-9 for f in found):
                    found.append(u)
                    new.append(u)
        frontier = new
    assert len(found) == 12, len(found)
    neighbors = [k for k in range(12) if abs(found[k] @ V - 1.0 / math.sqrt(5.0)) < 1e-9]
    third = next(k for k in neighbors if abs(found[k] @ V1 - 1.0 / math.sqrt(5.0)) < 1e-9 and found[k][2] > 0)
    order = [0, 1, third] + [k for k in range(2, 12) if k != third]
    vertices = np.array([found[k] for k in order])
    faces = []
    for a in range(12):
        for b in range(a + 1, 12):
            for c in range(b + 1, 12):
                if min(vertices[a] @ vertices[b], vertices[b] @ vertices[c], vertices[a] @ vertices[c]) > 0.4:
                    normal = np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
                    faces.append((a, b, c) if normal @ vertices[a] > 0 else (a, c, b))
    assert len(faces) == 20, len(faces)
    return vertices, faces


ICO_VERTICES, ICO_FACES = ico_geometry()


def ico_vertex(direction):
    """The icosahedron vertex nearest a direction"""
    return int(np.argmax(ICO_VERTICES @ np.asarray(direction, dtype=float)))


def ico_fold_point(vertices, weights):
    """A lattice point of a face (its three vertices and their weights, ints summing to N) folded into the domain: the
    weights sorted (N - i - j >= i >= j), and the rotation or reflection G (3 x 3, floats) taking the point's corners,
    in that order, to the domain's V, V1, V2 (G @ point = folded point). Ties keep the lower vertex first."""
    order = sorted(range(3), key=lambda k: (-weights[k], vertices[k]))
    corners = [vertices[k] for k in order]
    source = np.stack([ICO_VERTICES[c] for c in corners], axis=1)
    target = np.stack([ICO_VERTICES[0], ICO_VERTICES[1], ICO_VERTICES[2]], axis=1)
    G = target @ np.linalg.inv(source)
    # Orthogonal to rounding: polish it (the nearest orthogonal matrix)
    u, _s, vt = np.linalg.svd(G)
    return [weights[k] for k in order], corners, u @ vt
