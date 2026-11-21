"""Endless sphere maps on the icosahedron, sphere.py's second layout (prototyped in ~/blender_tests/sphere/icosa.py,
icoworld.py and pattern.paint_ico()): each of the 20 faces carries a triangular lattice that splits 2 x 2 at every
level, mapped onto the sphere by the conformal icosahedron chart (sphere_chart.IcoReference), so its lines run straight
across the faces' edges. Its star motif (IcoStars, the "stars" motif; sphere_sierpinski.Gasket is the default): a
12-point rosette's outline at every lattice corner, where six triangles meet, which at the icosahedron's 12 corners,
where five do, closes up into a 10-point girih sun; and gold bars along the triangles' edges from one rosette's tip to
the next. Finer levels lie on top, as on the cube.

Everything else is the cube's (sphere.py): the anchor, a lattice point here (a face's three corners and its whole
weights on them, summing to N), dive and climb loops by powers of 2, recentering, the plan, the shader's ray, fold
and level walk. The fold is into 1/120 of the sphere (sphere_chart's ICO_WALLS), where the anchor's weights are sorted;
G, the fold's rotation or reflection, is in floats (the icosahedron's coordinates are irrational), which is only ever
applied to offsets, so it costs nothing exact.

A pixel's place is measured from its nearest lattice corner, exactly, in skew coordinates (I along the face's first
edge, J along its second, at 60 degrees), and so are the bars' sides and ends (their widths and ends are dyadic in skew
units, so Sterbenz's lemma places them exactly at any level, as the cube's lines). The rosette's outline is worked out
from that offset in float32, good to 2^-25 of a triangle: only a dive exactly at a giant rosette's outline would show.
The colors are the prototype's: each corner's from its weights on its face's corners' numbers (the same from every
face around it), its rosette's 30-degree sectors alternating, so they alternate around the 10-point suns too.
"""

import math
from fractions import Fraction

import numpy as np

try:
    from . import sphere_chart as chart
except ImportError:          # tests import it by itself
    import sphere_chart as chart

LAYOUT = "ico"
ROOT3_HALF = chart.ROOT3_HALF

# The motif, in units of a level's triangles' side (the prototype's ROSETTE_* and BAR_*)
ROSETTE_POINTS = 12
ROSETTE_OUTER = 0.34
ROSETTE_INNER = 0.34 * 0.62
# The gold bars' half-width across their lattice line in skew units (the coordinate that's constant along it, of which
# a triangle's height is 1): the prototype's 0.011 / (sqrt 3 / 2), to 0.1%, dyadic
BAR_SKEW = 13 / 1024
# Where a bar starts, along its edge from a rosette's middle: the prototype's 0.36, dyadic
BAR_FROM = 23 / 64
# The level buffer's rows per level (sphere.level_buffer())
LEVEL_ROWS = 6


def sphere_module():
    """sphere.py, for what the layouts share (the outline's width, the colors, the surface's geometry), imported when
    needed: sphere imports this module"""
    try:
        from . import sphere
    except ImportError:
        import sphere
    return sphere


def canonical(vertices, weights):
    """A lattice point as one face and its weights, the same for every way of naming it (a point on an edge or a corner
    is on two or five faces): the face of lowest vertex numbers holding its corners of weight, in the face's own
    counterclockwise order from its lowest vertex"""
    weight = {int(v): int(w) for v, w in zip(vertices, weights) if w > 0}
    held = set(weight)
    face = min((f for f in chart.ICO_FACES if held <= set(f)), key=lambda f: sorted(f))
    k = face.index(min(face))
    face = face[k:] + face[:k]
    return tuple(face), tuple(weight.get(v, 0) for v in face)


def rosette_distance(qx, qy):
    """The distance of a point (from a lattice corner, in triangle sides) from the 12-point rosette's outline, and which
    of its 30-degree sectors it's in (0 or 1, alternating, the first counterclockwise from the face's first edge),
    as the shader's rosetteDistance(): folded by the rosette's symmetry into 0 to 15 degrees, counting the reflections
    that change the sector's parity. Tips at multiples of 30 degrees (along the edges and the altitudes)."""
    flips = 0
    if qx < 0.0:
        qx, flips = -qx, flips + 1
    if qy < 0.0:
        qy, flips = -qy, flips + 1
    if qy > qx:
        qx, qy = qy, qx
    s30 = -0.5 * qx + ROOT3_HALF * qy
    if s30 > 0.0:
        qx, qy, flips = qx + s30, qy - 2.0 * ROOT3_HALF * s30, flips + 1
    n15 = (-math.sin(math.pi / 12), math.cos(math.pi / 12))
    s15 = n15[0] * qx + n15[1] * qy
    if s15 > 0.0:
        qx, qy = qx - 2.0 * s15 * n15[0], qy - 2.0 * s15 * n15[1]
    ax, ay = ROSETTE_OUTER, 0.0
    bx, by = ROSETTE_INNER * math.cos(math.pi / 12), ROSETTE_INNER * math.sin(math.pi / 12)
    ex, ey = bx - ax, by - ay
    t = min(1.0, max(0.0, ((qx - ax) * ex + (qy - ay) * ey) / (ex * ex + ey * ey)))
    return math.hypot(qx - ax - t * ex, qy - ay - t * ey), flips % 2


def nearest_corner(I, J):
    """The lattice corner nearest a point (I, J) in skew coordinates (Fractions or floats), by the hexagonal grid's
    rounding: round all three of I, J and -I - J, and fix the one rounded furthest"""
    K = -I - J
    rI, rJ, rK = round(I), round(J), round(K)
    eI, eJ, eK = abs(rI - I), abs(rJ - J), abs(rK - K)
    if eI > eJ and eI > eK:
        rI = -rJ - rK
    elif eJ > eK:
        rJ = -rI - rK
    return int(rI), int(rJ)


class IcoWorld:
    """N = 2^n, the anchor A = (face, weights): three icosahedron vertices of a face (sphere_chart.ICO_FACES) and the
    anchor's whole weights on them, summing to N (canonical()), and rho, as sphere.World. Folded: A_f = (N - i - j, i,
    j), its weights sorted, on the domain's V, V1, V2, the face's corners taken there (corners) by G."""

    layout = LAYOUT
    level_shift = 0                  # a level's cells are N / 2^level units (the cube's 2N / 2^level)
    chart_sign = 1                   # the chart's dxi~ runs along the face's x, y
    unit_angle = math.atan(2.0)      # radians N units span on the sphere: an edge, 63.4 degrees
    walls = chart.ICO_WALLS
    fold_steps = chart.ICO_FOLD_STEPS

    def __init__(self, n, A, rho):
        self.n, self.rho = int(n), float(rho)
        self.A = canonical(*A)
        assert self.n >= 0 and sum(self.A[1]) == self.N and min(self.A[1]) >= 0, (self.n, self.A)
        weights, corners, self.G = chart.ico_fold_point(*self.A)
        self.A_f = tuple(weights)
        self.corners = tuple(corners)
        self._references = {}
        self._levels = {}
        self._uniform = {}

    @property
    def N(self):
        return 1 << self.n

    @property
    def kappa(self):
        return math.ldexp(1.0 / self.rho, -self.n) if self.n < 1070 else 0.0

    def charts(self, view_units=1.0, corner_view_units=None):
        key = (math.floor(math.log2(max(view_units, 1e-300))),
               math.floor(math.log2(max(corner_view_units or view_units, 1e-300))))
        charts = self._references.get(key)
        if charts is None:
            if len(self._references) > 8:
                self._references.clear()
            charts = self._references[key] = chart.IcoCharts(self.n, self.A_f[1], self.A_f[2], 2.0 ** key[0],
                                                              2.0 ** key[1])
        return charts

    def normal(self):
        return self.G.T @ self.charts().anchor.n0

    def text(self):
        return " ".join([LAYOUT, str(self.n)] + [str(v) for v in self.A[0]] + [format(w, "x") for w in self.A[1]])

    @staticmethod
    def from_text(text, rho):
        _layout, n, v0, v1, v2, *weights = text.split()
        return IcoWorld(int(n), ((int(v0), int(v1), int(v2)), [int(w, 16) for w in weights]), rho)

    def cell(self, level):
        return Fraction(self.N, 1 << level)

    def to_corner(self):
        """From the folded anchor to the domain's corner V, where the corner's reference is, in units (the face's
        x + i y)"""
        i, j = self.A_f[1], self.A_f[2]
        return complex(-(float(i) + float(j) / 2.0), -ROOT3_HALF * float(j))

    def dived_anchor(self, k):
        """The anchor after a dive loop by 2^k"""
        return self.A[0], [w << k for w in self.A[1]]

    @staticmethod
    def climbed_anchor(A, k):
        """An anchor A on the 2^k times coarser lattice after a climb loop by 2^k"""
        return A[0], [w >> k for w in A[1]]

    def nearest_vertex(self, p, step=1):
        """As sphere.nearest_vertex(): the lattice point (its weights multiples of step) nearest the point of the sphere
        below map point p, and its offset from the anchor's point (map units), exactly"""
        sp = sphere_module()
        nA = self.normal()
        m = sp.surface_offset(self, p, nA)
        charts, mf, R, (D, corner) = sp.inverse_from_guess(self, m, max(1.0, float(np.linalg.norm(m)) * 4.0))
        ref = charts.anchor
        di, dj = chart.IcoReference.lattice_offset(complex(D))
        N, i, j = self.N, self.A_f[1], self.A_f[2]
        base_i, base_j = (0, 0) if corner else (i, j)
        # Nearest on the coarser lattice: in its own steps
        Pi, Pj = nearest_corner((Fraction(base_i) + Fraction(float(di))) / step,
                                (Fraction(base_j) + Fraction(float(dj))) / step)
        Pi, Pj = Pi * step, Pj * step
        # In the domain, N - i - j >= i >= j >= 0 (a lattice point on a wall is on both sides)
        Pj = max(0, Pj)
        Pi = max(Pi, Pj)
        if 2 * Pi + Pj > N:
            Pi = step * ((N - Pj) // (2 * step))
            Pj = min(Pj, Pi)
        m_new_f = ref.forward_scaled(ref.dz_of(Pi - i, Pj - j))
        unfold = (R @ self.G).T
        vertices = [chart.ico_vertex(unfold @ chart.ICO_VERTICES[k]) for k in range(3)]
        A_new = canonical(vertices, (N - Pi - Pj, Pi, Pj))
        offset = m + unfold @ (m_new_f - mf)
        return A_new, self.rho * offset

    # ----- the levels -----

    def level_cell(self, level):
        """The folded anchor's place at a level, in its triangles (C = N / 2^level units) along the skew axes: each
        axis's cell index and place in it, 0 <= r < 1"""
        C = self.cell(level)
        out = []
        for coord in (self.A_f[1], self.A_f[2]):
            q = Fraction(coord) / C
            k0 = math.floor(q)
            out.append((k0, q - k0))
        return C, out


# ----- the star motif -----

class IcoStars:
    """The icosahedron's star motif (the module's docstring), as a motif for sphere.Plan and the shader (see
    sphere.CubeStars)"""

    name = "stars"
    layout = LAYOUT
    first_level = 0

    @property
    def base(self):
        return sphere_module().FLOOR

    @property
    def finest_footprint(self):
        sp = sphere_module()
        return sp.RING / sp.FINEST_PIXELS

    def level_data_exact(self, world, level):
        """What the shader needs of a level: the anchor's place as three float32 words each for I, J, I + J, I + J / 2,
        J + I / 2 and (J - I) / 2 (the bars' sides and ends), its cell's indices mod 4 and 2^level's (the colors), and
        how many cells it is from V"""
        sp = sphere_module()
        C, ((kI, rI), (kJ, rJ)) = world.level_cell(level)
        values = []
        for r in (rI, rJ, rI + rJ, rI + rJ / 2, rJ + rI / 2, (rJ - rI) / 2):
            values += sp.words(r)
        to_corner = float(kI + kJ + 2) if kI + kJ < 2 ** 20 else 1e9
        return {"words": values, "mod": (kI % 4, kJ % 4, (1 << level) % 4 if level < 2 else 0), "to_corner": to_corner,
                "cell": C}

    def corner_level_data(self, world, level):
        """level_data() measured from V, a corner of every level"""
        return {"words": [0.0] * 18, "mod": (0, 0, (1 << level) % 4 if level < 2 else 0), "to_corner": 0.0,
                "cell": None}

    def level_rows(self, world, level, data, kappa_anchor, kappa_corner, fine):
        """A level's LEVEL_ROWS rows of 4 for the shader's levelAt()"""
        w = data["words"]
        kI, kJ, top = data["mod"]
        return [(kappa_anchor, kappa_corner, float(top), data["to_corner"]), w[0:4], w[4:8], w[8:12], w[12:16],
                (w[16], w[17], float(kI), float(kJ))]

    def uniform_exact(self, world, level, extent):
        """Whether a level is one color over the view, every offset up to extent units from the folded anchor: None if
        an edge of a piece may pass through it, else the color (sRGB) covering it, or False for none. The bars from the
        exact offsets from the nearest corner; the rosette in float64, with a margin far above its rounding. A rosette
        only colors the view when the view stays clear of the domain's walls, which turn the colors."""
        sp = sphere_module()
        C, ((kI, rI), (kJ, rJ)) = world.level_cell(level)
        e = Fraction(extent) / C
        if e >= Fraction(1, 8):
            return None
        pI, pJ = nearest_corner(rI, rJ)
        dI, dJ = rI - pI, rJ - pJ
        es = 2 * e                     # the view's reach in skew units
        pieces = []
        for across, along in ((dJ, dI + dJ / 2), (dI, dJ + dI / 2), (dI + dJ, (dJ - dI) / 2)):
            band = sp.interval_side(abs(across) - Fraction(BAR_SKEW) - es, abs(across) - Fraction(BAR_SKEW) + es)
            span = sp.interval_side(Fraction(BAR_FROM) - abs(along) - es, Fraction(BAR_FROM) - abs(along) + es)
            if band == sp.OUTSIDE or span == sp.OUTSIDE:
                where = sp.OUTSIDE
            elif band == sp.INSIDE and span == sp.INSIDE:
                where = sp.INSIDE
            else:
                where = sp.PARTLY
            pieces.append((where, sp.GOLD))
        qx, qy = float(dI + dJ / 2), ROOT3_HALF * float(dJ)
        dist, sector = rosette_distance(qx, qy)
        margin = float(e) + 1e-12
        h = sp.RING / 2
        if dist - h > margin:
            rosette = sp.OUTSIDE
        elif dist + margin < h:
            rosette = sp.INSIDE
        else:
            rosette = sp.PARTLY
        color = None
        if rosette == sp.INSIDE:
            # One sector throughout, and clear of the domain's walls
            angle = math.atan2(qy, qx)
            to_ray = abs(math.remainder(angle, math.pi / 6)) * math.hypot(qx, qy)
            N, i, j = world.N, world.A_f[1], world.A_f[2]
            # In Fractions: deep down, the anchor's ints have no floats
            to_wall = min(Fraction(ROOT3_HALF) * j, Fraction(i - j, 2), Fraction(N - 2 * i - j, 2)) / C
            if to_ray <= margin or to_wall <= float(e) * 1.01 + 1e-12:
                return None
            labels = world.corners
            m1, m2 = kI + pI, kJ + pJ
            m0 = (1 << level) - m1 - m2
            parity = (labels[0] * m0 + labels[1] * m1 + labels[2] * m2) % 4
            flip = 1 if np.linalg.det(world.G) < 0 else 0
            color = sp.SQUARE[parity] if (sector ^ flip) == 0 else sp.TIP[parity]
        pieces.append((rosette, color))
        if any(where == sp.PARTLY for where, _ in pieces):
            return None
        for where, c in pieces:
            if where == sp.INSIDE:
                return c
        return False

    def cuts(self, world, level):
        return 0

    def paint_source(self):
        return sphere_module().STAR_HELPERS + paint_source()

    def defines(self, info):
        sphere_module().star_defines(info)
        star_defines(info)


STARS = IcoStars()


# ----- the shader's icosahedron parts (sphere.fragment_source()) -----

def vertex_lookup_source():
    """GLSL: icoVertex(v), the number of the icosahedron vertex nearest a direction (in the domain's frame)"""
    lines = ["int icoVertex(vec3 v)", "{", "    int best = 0;",
             "    float most = dot(v, vec3(%r, %r, %r));" % tuple(float(c) for c in chart.ICO_VERTICES[0]),
             "    float d;"]
    for k in range(1, 12):
        lines.append("    d = dot(v, vec3(%r, %r, %r)); if (d > most) { most = d; best = %d; }"
                     % (*(float(c) for c in chart.ICO_VERTICES[k]), k))
    lines += ["    return best;", "}"]
    return "\n".join(lines)


def chart_defines(info):
    """The icosahedron's chart's and fold's defines for sphere.make_shader()"""
    table = " ".join(f"if (k == {k}) return {chart.G_ICO[k]!r};" for k in range(1, len(chart.G_ICO))) + " return 0.0;"
    info.define("SERIES_TABLE", table)
    info.define("TERMS", str(len(chart.G_ICO)))
    for k, wall in enumerate(chart.ICO_WALLS):
        info.define(f"WALL{k}", "vec3(%r, %r, %r)" % tuple(float(c) for c in wall))
    info.define("FOLD_STEPS", str(chart.ICO_FOLD_STEPS))
    info.define("ROOT3_HALF", repr(ROOT3_HALF))
    info.define("INV_ROOT3", repr(1.0 / math.sqrt(3.0)))


def star_defines(info):
    """The star motif's own defines (sphere.star_defines() are both layouts' stars')"""
    info.define("ROSETTE_OUTER", repr(ROSETTE_OUTER))
    info.define("ROSETTE_INNER", repr(ROSETTE_INNER))
    info.define("BAR_SKEW", repr(BAR_SKEW))
    info.define("BAR_FROM", repr(BAR_FROM))
    info.define("N15", "vec2(%r, %r)" % (-math.sin(math.pi / 12), math.cos(math.pi / 12)))
    info.define("INNER15", "vec2(%r, %r)" % (ROSETTE_INNER * math.cos(math.pi / 12),
                                             ROSETTE_INNER * math.sin(math.pi / 12)))
    info.define("V_DOMAIN0", "vec3(%r, %r, %r)" % tuple(float(c) for c in chart.ICO_VERTICES[0]))
    info.define("V_DOMAIN1", "vec3(%r, %r, %r)" % tuple(float(c) for c in chart.ICO_VERTICES[1]))
    info.define("V_DOMAIN2", "vec3(%r, %r, %r)" % tuple(float(c) for c in chart.ICO_VERTICES[2]))
    # Within a corner's own hexagon, only its own lines and rosette show while a tile reaches less than this
    info.define("TILE_REACH", repr(0.1))


CHART_SOURCE = """
// x with x^p = Z, arg Z taken in (-90, 270] degrees (sphere_chart.sector_root()). |Z| from Z scaled by its larger
// part: length() squares Z's parts, and T = xi^6 is below float32's squares within 6e-4 of a view's size from V, where
// every pixel came out at V (a dark hole, a deep dive at an icosahedron corner)
vec2 sectorRoot(vec2 Z, float p)
{
    float a = atan(Z.y, Z.x);
    if (a <= -HALF_PI) {
        a += 2.0 * PI;
    }
    float big = max(abs(Z.x), abs(Z.y));
    float r = big > 0.0 ? pow(big, 1.0 / p) * pow(length(Z / big), 1.0 / p) : 0.0;
    return r * vec2(cos(a / p), sin(a / p));
}

// (x1^p - x0^p) / (x1 - x0)
vec2 powerDivided(vec2 x0, vec2 x1, int p)
{
    vec2 total = vec2(0.0);
    vec2 power0 = vec2(1.0, 0.0);
    for (int j = 0; j < p; j++) {
        total = cmul(total, x1) + power0;
        power0 = cmul(power0, x0);
    }
    return total;
}

// The icosahedron's chart backward (sphere_chart.IcoReference.inverse_difference()): dxi~ for a folded offset m from
// the reference, and its steps along two pixel steps dm1, dm2. The Ref's z0 is xi~, Z0 T~, v0 w~, w0 the absolute w.
vec2 inverseDifference(Ref f, vec3 m, vec3 dm1, vec3 dm2, out vec2 g1, out vec2 g2)
{
    vec3 dn = m / f.scale;
    float X0 = f.n0.y, Y0 = f.n0.z, Z0n = f.n0.x;
    vec2 dw = (vec2(dn.y, dn.z) * (1.0 + Z0n) - vec2(X0, Y0) * dn.x) / ((1.0 + Z0n + f.sigma * dn.x) * (1.0 + Z0n));
    vec2 w1 = f.v0 + f.rho1 * dw;
    vec2 dG = cmul(dw, powerDivided(f.v0, w1, 5));
    vec2 T0abs = f.t * f.Z0;
    vec2 dT = cdiv(dG, seriesSlope(T0abs));
    for (int i = 0; i < f.newton; i++) {
        vec2 T1 = f.Z0 + f.rho1 * dT;
        dT -= cdiv(cmul(dT, seriesDivided(T0abs, f.t * T1)) - dG, seriesSlope(f.t * T1));
    }
    vec2 T1 = f.Z0 + f.rho1 * dT;
    vec2 xi1 = sectorRoot(T1, 6.0);
    vec2 q = powerDivided(f.z0, xi1, 6);
    vec2 dxi = dot(q, q) > 0.0 ? cdiv(dT, q) : (xi1 - f.z0) / max(f.rho1, 1.0e-30);
    // The pixel's steps: 5 w^4 / (6 xi^5 G'(T)), then the stereographic projection's
    vec3 n1 = f.n0 + f.sigma * dn;
    vec2 xx = dot(xi1, xi1) > 0.0 ? xi1 : vec2(1.0e-30, 0.0);
    vec2 w2 = csq(w1);
    vec2 x2 = csq(xx);
    vec2 chain = cdiv(5.0 * csq(w2), cmul(6.0 * cmul(csq(x2), xx), seriesSlope(f.t * T1)));
    float iz = 1.0 / ((1.0 + n1.x) * (1.0 + n1.x));
    vec2 a1 = (vec2(dm1.y, dm1.z) * (1.0 + n1.x) - vec2(n1.y, n1.z) * dm1.x) * iz / f.scale;
    vec2 a2 = (vec2(dm2.y, dm2.z) * (1.0 + n1.x) - vec2(n1.y, n1.z) * dm2.x) * iz / f.scale;
    g1 = cmul(chain, a1);
    g2 = cmul(chain, a2);
    return dxi;
}
"""

PAINT_SOURCE = """
VERTEX_LOOKUP

// The 12-point rosette's outline distance from q (from its middle, in triangle sides) and which of its 30-degree
// sectors q is in (sphere_ico.rosette_distance())
float rosetteDistance(vec2 q, out int sector)
{
    int flips = 0;
    if (q.x < 0.0) { q.x = -q.x; flips++; }
    if (q.y < 0.0) { q.y = -q.y; flips++; }
    if (q.y > q.x) { q = q.yx; }
    float s30 = dot(q, vec2(-0.5, ROOT3_HALF));
    if (s30 > 0.0) { q -= 2.0 * s30 * vec2(-0.5, ROOT3_HALF); flips++; }
    float s15 = dot(q, N15);
    if (s15 > 0.0) { q -= 2.0 * s15 * N15; }
    sector = flips & 1;
    vec2 a = vec2(ROSETTE_OUTER, 0.0);
    vec2 e = INNER15 - a;
    float t = clamp(dot(q - a, e) / dot(e, e), 0.0, 1.0);
    return length(q - a - t * e);
}

// A level's rows (sphere_ico.IcoStars.level_rows()): the anchor's place (words) in skew coordinates and the
// combinations the bars need, its cell's indices mod 4, 2^level mod 4, and how many cells it is from V
struct Level {
    vec4 t0;
    vec3 wI, wJ, wS, wX, wY, wM;
    vec2 cellMod;
    float toCorner;
};

Level levelAt(int i, bool useCorner)
{
    Level L;
    L.t0 = levelValues.rows[LEVEL_ROWS * i];
    L.toCorner = L.t0.w;
    if (useCorner) {
        L.wI = vec3(0.0); L.wJ = L.wI; L.wS = L.wI; L.wX = L.wI; L.wY = L.wI; L.wM = L.wI;
        L.cellMod = vec2(0.0);
    }
    else {
        vec4 t1 = levelValues.rows[LEVEL_ROWS * i + 1];
        vec4 t2 = levelValues.rows[LEVEL_ROWS * i + 2];
        vec4 t3 = levelValues.rows[LEVEL_ROWS * i + 3];
        vec4 t4 = levelValues.rows[LEVEL_ROWS * i + 4];
        vec4 t5 = levelValues.rows[LEVEL_ROWS * i + 5];
        L.wI = t1.xyz; L.wJ = vec3(t1.w, t2.xy); L.wS = vec3(t2.zw, t3.x); L.wX = t3.yzw; L.wY = t4.xyz;
        L.wM = vec3(t4.w, t5.xy); L.cellMod = t5.zw;
    }
    return L;
}

// The lattice corner nearest a rough place (I, J), as the hexagonal grid rounds (sphere_ico.nearest_corner())
vec2 nearestCorner(float I, float J)
{
    float K = -I - J;
    float rI = floor(I + 0.5), rJ = floor(J + 0.5), rK = floor(K + 0.5);
    float eI = abs(rI - I), eJ = abs(rJ - J), eK = abs(rK - K);
    if (eI > eJ && eI > eK) {
        rI = -rJ - rK;
    }
    else if (eJ > eK) {
        rJ = -rI - rK;
    }
    return vec2(rI, rJ);
}

// Whether a place dI, dJ from its nearest corner (skew), its footprint F (triangle sides), is so far from the corner's
// rosette and the bars along its three lines that it covers none of them, off as in clearOfLines() (the rosette is
// between ROSETTE_INNER and ROSETTE_OUTER from the corner, a bar's sides BAR_SKEW from its line, its end BAR_FROM from
// the corner along it)
bool icoClear(float dI, float dJ, float F, float off)
{
    vec2 q = vec2(dI + 0.5 * dJ, ROOT3_HALF * dJ);
    float rho = length(q);
    float m = HALF_RING + 0.5 * F + 2.0 * off;
    if (rho >= ROSETTE_INNER - m && rho <= ROSETTE_OUTER + m) {
        return false;
    }
    // In skew units a footprint F is at most 2 F across, and F along
    float across = BAR_SKEW + F + 2.0 * off;
    float along = BAR_FROM - F - 2.0 * off;
    return (abs(dJ) >= across || abs(dI + 0.5 * dJ) <= along)
        && (abs(dI) >= across || abs(dJ + 0.5 * dI) <= along)
        && (abs(dI + dJ) >= across || abs(0.5 * (dJ - dI)) <= along);
}

// One bar: across, its line's side distances lo (from -BAR_SKEW) and hi (to BAR_SKEW), exact; along, its end's
// distance (past BAR_FROM from the corner), exact; F the footprints across and along
float barCover(float lo, float hi, float alongPast, float Facross, float Falong)
{
    return band(lo, hi, Facross, BAR_SKEW) * spanCover(alongPast, 1.0e30, Falong);
}

// One level painted over color: d the pixel's offset from the reference in this level's triangles (the face's x, y),
// ga, gb its steps per pixel; info: foldInfo()'s (the face's corners' numbers, and whether the fold mirrors); cuts:
// the Sierpinski motifs' count, not the stars'
vec3 paintAt(vec3 color, Level L, vec2 d, vec2 ga, vec2 gb, vec4 info, inout float cuts)
{
    // Skew offsets and their steps
    float dI = d.x - d.y * INV_ROOT3, dJ = 2.0 * INV_ROOT3 * d.y;
    vec2 gI = ga - gb * INV_ROOT3, gJ = 2.0 * INV_ROOT3 * gb;
    float F = max(abs(ga.x) + abs(ga.y), abs(gb.x) + abs(gb.y));
    float xI = (L.wI.x + dI) + (L.wI.y + L.wI.z), xJ = (L.wJ.x + dJ) + (L.wJ.y + L.wJ.z);
    vec2 k = nearestCorner(xI, xJ);
    // The rough offset tells most pixels they're clear of everything
    if (icoClear(xI - k.x, xJ - k.y, F, ROUGH * (1.0 + abs(dI) + abs(dJ)))) {
        return color;
    }
    // The exact offsets from the corner, and every side and end from them
    float eI = lineAt(L.wI, k.x, 0.0, dI), eJ = lineAt(L.wJ, k.y, 0.0, dJ);
    float FI = abs(gI.x) + abs(gI.y), FJ = abs(gJ.x) + abs(gJ.y);
    float FS = abs(gI.x + gJ.x) + abs(gI.y + gJ.y);
    float FX = abs(ga.x) + abs(ga.y);
    vec2 gY = gJ + 0.5 * gI, gM = 0.5 * (gJ - gI);
    float FY = abs(gY.x) + abs(gY.y), FM = abs(gM.x) + abs(gM.y);
    float dS = dI + dJ, dX = d.x, dY = dJ + 0.5 * dI, dM = 0.5 * (dJ - dI);
    float kS = k.x + k.y, kX = k.x + 0.5 * k.y, kY = k.y + 0.5 * k.x, kM = 0.5 * (k.y - k.x);
    // Gold bars along the corner's three lines: J = k.y (the first edge's direction), I = k.x, I + J = kS
    float aX = lineAt(L.wX, kX, 0.0, dX), aY = lineAt(L.wY, kY, 0.0, dY), aM = lineAt(L.wM, kM, 0.0, dM);
    float pastX = aX >= 0.0 ? lineAt(L.wX, kX, BAR_FROM, dX) : -lineAt(L.wX, kX, -BAR_FROM, dX);
    float pastY = aY >= 0.0 ? lineAt(L.wY, kY, BAR_FROM, dY) : -lineAt(L.wY, kY, -BAR_FROM, dY);
    float pastM = aM >= 0.0 ? lineAt(L.wM, kM, BAR_FROM, dM) : -lineAt(L.wM, kM, -BAR_FROM, dM);
    float gold = barCover(lineAt(L.wJ, k.y, -BAR_SKEW, dJ), -lineAt(L.wJ, k.y, BAR_SKEW, dJ), pastX, FJ, FX);
    gold = max(gold, barCover(lineAt(L.wI, k.x, -BAR_SKEW, dI), -lineAt(L.wI, k.x, BAR_SKEW, dI), pastY, FI, FY));
    gold = max(gold, barCover(lineAt(L.wS, kS, -BAR_SKEW, dS), -lineAt(L.wS, kS, BAR_SKEW, dS), pastM, FS, FM));
    color = mix(color, GOLD, gold);
    // The rosette, colored by the corner's weights on its face's corners' numbers, its sectors alternating (a mirror
    // turns their order round)
    int sector;
    float dist = rosetteDistance(vec2(eI + 0.5 * eJ, ROOT3_HALF * eJ), sector);
    float cover = band(dist + HALF_RING, HALF_RING - dist, F, HALF_RING);
    if (cover > 0.0) {
        int m1 = int(mod(L.cellMod.x + k.x, 4.0) + 0.5), m2 = int(mod(L.cellMod.y + k.y, 4.0) + 0.5);
        int m0 = (int(L.t0.z + 0.5) + 8 - m1 - m2) % 4;
        int p = (int(info.x) * m0 + int(info.y) * m1 + int(info.z) * m2) % 4;
        color = mix(color, (sector ^ int(info.w)) == 0 ? squareColor(p) : tipColor(p), cover);
    }
    return color;
}

// tileMain()'s test of a level (see the cube's tileClear()): within a corner's own hexagon only its own rosette and
// lines show, while the tile reaches less than TILE_REACH
bool tileClear(Level L, vec2 d, float F, float r, bool useCorner)
{
    if (r > TILE_REACH) {
        return false;
    }
    float far = abs(d.x) + abs(d.y) + 2.0 * r + (useCorner ? L.toCorner : 0.0);
    float off = 2.0 * ROUGH * (2.0 + far) + 2.0 * r;
    float dI = d.x - d.y * INV_ROOT3, dJ = 2.0 * INV_ROOT3 * d.y;
    float xI = (L.wI.x + dI) + (L.wI.y + L.wI.z), xJ = (L.wJ.x + dJ) + (L.wJ.y + L.wJ.z);
    vec2 k = nearestCorner(xI, xJ);
    return icoClear(xI - k.x, xJ - k.y, F, off);
}

// What the colors need of a pixel's fold F (folded = F unfolded): the numbers of the face's corners it took to V, V1,
// V2, and whether it mirrors
vec4 foldInfo(mat3 F)
{
    mat3 back = transpose(F);
    return vec4(float(icoVertex(back * V_DOMAIN0)), float(icoVertex(back * V_DOMAIN1)), float(icoVertex(back * V_DOMAIN2)),
                determinant(F) < 0.0 ? 1.0 : 0.0);
}
"""


def paint_source():
    return PAINT_SOURCE.replace("VERTEX_LOOKUP", vertex_lookup_source())
