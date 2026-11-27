"""Endless 3D parts on the icosahedron sphere's lattice ("do the 3d parts for penrose and icosahedron tilings"):
parts.py's pieces, a vertex piece on each lattice point and an edge piece along each edge, at every level, on
sphere_ico.py's lattice, each of the 20 faces split 2 x 2 at every level. Six edges meet at a lattice point, and five
at the icosahedron's 12 corners, where the rosettes close up into suns.

The icosahedron's chart for parts_sphere.SphereLattice (IcoChart; the cube's is parts_sphere.CubeChart), which lays
the pieces out and finds the giant levels' for both layouts the same way:
- Which points are near which, and their edges: the 20 faces as a surface.Topology of triangles, whose keys name a
  point the same from every face it's on (surface.LocalLattice finds a level's points near a point, by the flat
  icosahedron's distances). A key is a lattice point: a face's corners and whole weights on them.
- Where a lattice point is on the sphere (offsets()): from the anchor, without cancellation, as the cube's
  sphere.surface_offsets() does it. The point is folded into the chart's domain (1/120 of the sphere) wall by wall, each
  wall's reflection a relabeling of the icosahedron's 12 corners, and which side of a wall a point is on decided from
  its whole weights exactly: a wall that cuts a face is the face's altitude through a corner, and the face's other two
  corners are equally far either side of it. The folded point's offset from the folded anchor goes through the chart
  (sphere_chart.IcoReference, from the anchor or from the domain's corner V), and back through the same walls on the
  sphere, each one as sigma m - 2 s nu (s the anchor's distance from the wall), never from N n(A) itself.
- With more digits than float64's (IcoPlacer: the giant levels' pieces deep down, as sphere_exact.Placer is the cube's):
  the chart's series in decimals and the walls' normals to as many digits. The anchor's fold G is irrational, so the
  placer works in the folded anchor's frame and leaves G (its "turn") to parts.exact_piece(), which turns only the
  camera, near the anchor, and the window's faces, a window across.
- Where the drone is in a level's triangles, from the anchor's weights (IcoChart.cell_place()), for the giant levels.
"""

import decimal
import math
from decimal import Decimal
from fractions import Fraction

import numpy as np

try:
    from . import parts_sphere
    from . import sphere
    from . import sphere_chart as chart
    from . import sphere_exact
    from . import sphere_ico
    from . import surface
except ImportError:          # tests import it by itself
    import parts_sphere
    import sphere
    import sphere_chart as chart
    import sphere_exact
    import sphere_ico
    import surface

ROOT3_HALF = chart.ROOT3_HALF
WALLS = chart.ICO_WALLS


# ----- the icosahedron's corners, faces and walls, combinatorially -----

def _reflect(v, nu):
    return v - 2.0 * (v @ nu) * nu


# Each wall's reflection as a relabeling of the 12 corners, and which side of it each corner is on (+1 the domain's)
WALL_PERM = [[chart.ico_vertex(_reflect(chart.ICO_VERTICES[v], nu)) for v in range(12)] for nu in WALLS]
_dots = chart.ICO_VERTICES @ WALLS.T
WALL_SIDE = [[0 if abs(_dots[v, w]) < 1e-9 else (1 if _dots[v, w] > 0.0 else -1) for v in range(12)]
             for w in range(len(WALLS))]
FOLD_LIMIT = 4 * chart.ICO_FOLD_STEPS
# Whole offsets under this (in units) are exact as floats, x + y / 2 too
SMALL = 1 << 51

_topology = None


def topology():
    """The icosahedron's 20 faces as a surface.Topology (its keys and neighbors; its flat places only pick points)"""
    global _topology
    if _topology is None:
        _topology = surface.Topology(None, positions=chart.ICO_VERTICES, faces=chart.ICO_FACES)
    return _topology


def side(point, wall):
    """Which side of a wall a point is on (+1 the domain's, -1 the other, 0 on it), exactly: point a {corner: weight}
    of its face's corners. A wall cutting a face is its altitude through one corner, the other two mirror images, so
    the side is which of those two weighs more; otherwise every corner with weight is on the same side (or on it)."""
    plus = minus = None
    signs = WALL_SIDE[wall]
    for v, weight in point.items():
        s = signs[v]
        if s > 0:
            plus = weight
        elif s < 0:
            minus = weight
    if plus is not None and minus is not None:
        return (plus > minus) - (plus < minus)
    return 1 if plus is not None else -1 if minus is not None else 0


def fold(point):
    """A point ({corner: weight}, the anchor's fold's corner numbers) folded into the domain, exactly: the walls it's
    reflected across in turn, and its weights on V1 and V2 (i, j) there"""
    walls = []
    point = {v: w for v, w in point.items() if w}
    for _ in range(FOLD_LIMIT):
        wall = next((w for w in range(len(WALLS)) if side(point, w) < 0), None)
        if wall is None:
            break
        perm = WALL_PERM[wall]
        point = {perm[v]: w for v, w in point.items()}
        walls.append(wall)
    else:
        raise ValueError("parts_ico: a point didn't fold into the domain")
    assert set(point) <= {0, 1, 2}, point
    return walls, point.get(1, 0), point.get(2, 0)


def anchor_perm(world):
    """The anchor's fold G as a relabeling of the corners: corner v goes to G v's"""
    G = np.asarray(world.G, dtype=float)
    return [chart.ico_vertex(G @ chart.ICO_VERTICES[v]) for v in range(12)]


_perms = {}          # world text -> anchor_perm()


def perm_of(world):
    return sphere_exact.kept(_perms, world.text(), 64, lambda: anchor_perm(world))


# ----- lattice points: a face's corners and weights on them (any whole multiples of a level's cells) -----

def key_point(key, M):
    """A Topology key of a level of M cells across as (corners, weights in cells: whole, summing to M)"""
    f, I, J = topology().params(key, M)[0]
    return tuple(topology()._corners[f]), (M - I - J, I, J)


def point_key(corners, weights, C, M):
    """A lattice point (corners, weights: multiples of C units) as a key of the level of M cells across"""
    held = {int(v): Fraction(w) / C for v, w in zip(corners, weights) if w}
    top = topology()
    f = next(f for f in range(len(top._corners)) if set(held) <= set(top._corners[f]))
    c = top._corners[f]
    I, J = held.get(c[1], Fraction(0)), held.get(c[2], Fraction(0))
    assert I.denominator == 1 and J.denominator == 1, "not a point of the level's lattice"
    return top.key(f, int(I), int(J), M)


def scaled(point, factor):
    corners, weights = point
    return corners, tuple(w * factor for w in weights)


# ----- placing lattice points on the sphere from the anchor -----

def offsets(world, points, view_units=None):
    """N (n(P) - n(A)) in the map's axes (the icosahedron's own, ICO_VERTICES') for lattice points P (corners, weights
    summing to N: ints or Fractions), as floats: the module's docstring. view_units: about how far the points reach
    from the anchor, in units (what the chart is scaled for), worked out if not given."""
    perm = perm_of(world)
    i, j = world.A_f[1], world.A_f[2]
    folds = []
    for corners, weights in points:
        point = {}
        for v, w in zip(corners, weights):
            if w:
                point[perm[int(v)]] = point.get(perm[int(v)], 0) + w
        walls, ip, jp = fold(point)
        folds.append((walls, ip, jp))
    if view_units is None:
        # The skew offsets' Euclidean sizes, in units
        size = lambda x, y: math.hypot(sphere.wall_float(x) + sphere.wall_float(y) / 2.0, ROOT3_HALF * sphere.wall_float(y))
        sizes = [size(ip - i, jp - j) for _w, ip, jp in folds]
        view_units = max([1.0] + [2.0 * s for s in sizes if math.isfinite(s)])
    charts = world.charts(view_units)
    s = charts.anchor.wall_distances()
    m = np.zeros((len(points), 3))
    anchor_rows, anchor_dz, corner_rows, corner_dz = [], [], [], []
    small_rows, small_d = [], []
    for row, (walls, ip, jp) in enumerate(folds):
        di, dj = ip - i, jp - j
        # From the domain's corner V if it's nearer that than the anchor (as the GPU and sphere.surface_offsets() do):
        # |x + y e^(i pi/3)|^2 = x^2 + x y + y^2, exactly
        if 4 * (ip * ip + ip * jp + jp * jp) < di * di + di * dj + dj * dj:
            corner_rows.append(row)
            corner_dz.append(charts.corner.dz_of(ip, jp))
        elif type(di) is int and type(dj) is int and abs(di) < SMALL and abs(dj) < SMALL:
            small_rows.append(row)
            small_d.append((di, dj))
        else:
            anchor_rows.append(row)
            anchor_dz.append(charts.anchor.dz_of(di, dj))
    if small_rows:
        # Whole offsets a float holds exactly (a level's points near its own world's anchor): dz_of() all at once, the
        # same to the bit (di + dj / 2 is exact, and the scale a power of 2)
        d = np.array(small_d, dtype=np.float64)
        e = -(charts.anchor.e_delta + world.n)
        anchor_rows += small_rows
        anchor_dz += (np.ldexp(d[:, 0] + d[:, 1] / 2.0, e) + 1j * (ROOT3_HALF * np.ldexp(d[:, 1], e))).tolist()
    with np.errstate(invalid="ignore", over="ignore"):
        if anchor_rows:
            m[anchor_rows] = charts.anchor.forward_scaled(np.array(anchor_dz))
        if corner_rows:
            m[corner_rows] = charts.m_corner + charts.corner.forward_scaled(np.array(corner_dz))
    # Back through each point's walls, first ones last: m' = sigma_1 ... sigma_k m + sum_j sigma_1 ... sigma_(j-1)
    # (-2 s_j nu_j), the same walls together
    by_walls = {}
    for row, (walls, _ip, _jp) in enumerate(folds):
        if walls:
            by_walls.setdefault(tuple(walls), []).append(row)
    for walls, rows in by_walls.items():
        S, shift = np.eye(3), np.zeros(3)
        for wall in walls:
            nu = WALLS[wall]
            if s[wall] == chart.FAR_WALL_DISTANCE:
                raise ValueError("parts_ico: a point folded across a wall far past the view")
            shift += S @ (-2.0 * s[wall] * nu)
            S = S @ (np.eye(3) - 2.0 * np.outer(nu, nu))
        m[rows] = m[rows] @ S.T + shift
    return m @ np.asarray(world.G, dtype=float)


def placed(world, points, view_units=None):
    """Lattice points on the sphere in map coordinates from the map's origin (no lift), and the sphere's unit normals
    there: (points, normals), as sphere.placed()"""
    m = offsets(world, points, view_units)
    nA = world.normal()
    normals = nA + np.ldexp(m, -world.n) if world.n < 1070 else np.broadcast_to(nA, m.shape).copy()
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    return world.rho * m, normals


class Below:
    """The point of the sphere below map point p, folded into the chart's domain (as sphere.Below is the cube's): where
    it is there, from the folded anchor or the corner V (exact base, float skew offset), and the fold's way back.
    lattice_point() rounds it to any level's lattice."""

    def __init__(self, world, p):
        m = sphere.surface_offset(world, p, world.normal())
        _charts, _mf, R, (D, corner) = sphere.inverse_from_guess(world, m, max(1.0, float(np.linalg.norm(m)) * 4.0))
        di, dj = chart.IcoReference.lattice_offset(complex(D))
        self.di, self.dj = float(di), float(dj)
        self.N, i, j = world.N, world.A_f[1], world.A_f[2]
        self.base = (0, 0) if bool(corner) else (i, j)
        unfold = (np.asarray(R, dtype=float) @ np.asarray(world.G, dtype=float)).T
        self.corners = tuple(chart.ico_vertex(unfold @ chart.ICO_VERTICES[k]) for k in range(3))

    def lattice_point(self, step):
        """The point of the lattice of step units (a power of 2: an int, or a Fraction under 1) nearest it, exactly,
        as (corners, weights): as sphere_ico.IcoWorld.nearest_vertex() rounds it"""
        step = Fraction(step)
        N = self.N
        Pi, Pj = sphere_ico.nearest_corner((Fraction(self.base[0]) + Fraction(self.di)) / step,
                                           (Fraction(self.base[1]) + Fraction(self.dj)) / step)
        Pi, Pj = Pi * step, Pj * step
        # In the domain, N - i - j >= i >= j >= 0
        Pj = max(Fraction(0), Pj)
        Pi = max(Pi, Pj)
        if 2 * Pi + Pj > N:
            Pi = step * ((N - Pj) // (2 * step))
            Pj = min(Pj, Pi)
        return self.corners, (N - Pi - Pj, Pi, Pj)


# ----- with more digits: the giant levels' pieces deep down -----

def _fifth_root(x):
    """A complex of Decimals' fifth root, its argument taken in (-90, 270] degrees as sphere_chart.sector_root() takes
    it (the domain's values are between 0 and 180, and one on the altitude wall, at 180, mustn't go to -180 by a
    rounding), by sphere_exact.root()"""

    def first(z):
        angle = math.atan2(z.imag, z.real)
        if angle <= -math.pi / 2.0:
            angle += 2.0 * math.pi
        return abs(z) ** 0.2 * complex(math.cos(angle / 5.0), math.sin(angle / 5.0))

    return sphere_exact.root(x, 5, first)


_constants = {}      # precision -> the icosahedron chart's constants as Decimals


def constants():
    prec = decimal.getcontext().prec
    values = _constants.get(prec)
    if values is None:
        five = Decimal(5).sqrt()
        cos_side = 1 / five                                        # cos atan 2
        walls = [(Decimal(0), Decimal(0), Decimal(1)),
                 (Decimal(0), (10 - 2 * five).sqrt() / 4, -(1 + five) / 4),              # sin 36, -cos 36
                 (((1 - cos_side) / 2).sqrt(), -((1 + cos_side) / 2).sqrt(), Decimal(0))]
        values = _constants[prec] = {"G": [Decimal(g) for g in chart.G_ICO], "root3_half": Decimal(3).sqrt() / 2,
                                     "walls": walls}
    return values


def direction(N, i, j):
    """n(P) for the domain's point P (weights N - i - j, i, j on V, V1, V2; i, j ints or Fractions) in the domain's
    frame: (x, y, z), Decimals at the context's precision (sphere_chart.IcoReference's n0, with every digit)"""
    C = sphere_exact.C
    k = constants()
    number = sphere_exact.number
    xi = C(number(Fraction(i, N) + Fraction(j, 2 * N)), k["root3_half"] * number(Fraction(j, N)))
    x2 = xi * xi
    x3 = x2 * xi
    T = x3 * x3
    W = C(Decimal(0), Decimal(0))
    for g in reversed(k["G"][1:]):
        W = (W + C(g)) * T
    w = _fifth_root(W)
    s = 1 + w.abs2()
    return ((2 - s) / s, 2 * w.re / s, 2 * w.im / s)


_directions = {}


def direction_kept(N, i, j):
    key = (Fraction(i, N), Fraction(j, N), decimal.getcontext().prec)
    return sphere_exact.kept(_directions, key, 4096, lambda: direction(N, i, j))


def _reflect_exact(m, nu):
    d = 2 * sphere_exact.dot(m, nu)
    return (m[0] - d * nu[0], m[1] - d * nu[1], m[2] - d * nu[2])


class IcoPlacer:
    """A world's lattice points placed with digits digits (offsets()' with Decimals), in the folded anchor's frame:
    the map's axes turned by turn (G, floats), which parts.exact_piece() applies to the camera and the window's faces"""

    def __init__(self, world, digits):
        self.world, self.digits = world, int(digits)
        self.N, self.i, self.j = world.N, world.A_f[1], world.A_f[2]
        self.turn = np.asarray(world.G, dtype=float)
        self.perm = perm_of(world)
        with self.context():
            k = constants()
            N, i, j = self.N, self.i, self.j
            self.n0 = direction_kept(N, i, j)
            # N times the anchor's distance from each wall, measured from the wall's foot nearest it along the face
            # (sphere_chart.IcoReference.wall_distances()), so the folds join exactly
            feet = ((i, 0), (Fraction(i + j, 2), Fraction(i + j, 2)), (Fraction(N - j, 2), j))
            self.walls = []
            for nu, (fi, fj) in zip(k["walls"], feet):
                if fi == i and fj == j:
                    self.walls.append(Decimal(0))
                    continue
                foot = direction_kept(N, fi, fj)
                m = tuple((p - q) * N for p, q in zip(foot, self.n0))
                self.walls.append(max(Decimal(0), -sphere_exact.dot(m, nu)))
            self.rho = Decimal(world.rho)

    def context(self):
        return decimal.localcontext(prec=self.digits)

    def offsets(self, points):
        """N (n(P) - n(A)) in the folded anchor's frame for lattice points P (corners, weights): Decimal triples"""
        k = constants()
        out = []
        with self.context():
            for corners, weights in points:
                point = {}
                for v, w in zip(corners, weights):
                    if w:
                        point[self.perm[int(v)]] = point.get(self.perm[int(v)], 0) + w
                walls, ip, jp = fold(point)
                n1 = direction_kept(self.N, ip, jp)
                m = tuple((p - q) * self.N for p, q in zip(n1, self.n0))
                for wall in reversed(walls):
                    m = _reflect_exact(m, k["walls"][wall])
                    m = sphere_exact.add(m, sphere_exact.scaled(k["walls"][wall], -2 * self.walls[wall]))
                out.append(m)
        return out

    def placed_points(self, points):
        """The points from the map's origin and the sphere's unit normals there, in the folded anchor's frame: two
        lists of Decimal triples"""
        ms = self.offsets(points)
        with self.context():
            positions = [sphere_exact.scaled(m, self.rho) for m in ms]
            normals = []
            for m in ms:
                n = sphere_exact.add(self.n0, sphere_exact.scaled(m, Decimal(1) / self.N))
                normals.append(sphere_exact.scaled(n, 1 / sphere_exact.dot(n, n).sqrt()))
        return positions, normals

    def unfold(self, v):
        """A Decimal vector of the folded frame in the map's axes: G^T v, G's floats taken as they are (only for what
        doesn't need every digit: a level object's place)"""
        G = self.turn.tolist()
        return tuple(sum(Decimal(G[r][c]) * v[r] for r in range(3)) for c in range(3))


_placers = {}


def placer(world, digits):
    return sphere_exact.kept(_placers, (world.text(), int(digits)), 16, lambda: IcoPlacer(world, digits))


# ----- parts.py's view of the icosahedron's lattice -----

_sizes = {}


def cell_sizes(world, levels):
    """Each level's cells at the anchor (map units), measured, as {level: size}, as parts.cell_sizes(): from the
    level's lattice point nearest the folded anchor on its face to the next one along the face, placed exactly"""
    sizes = sphere_exact.kept(_sizes, world.text(), 64, dict)
    levels = [k for k in levels if k not in sizes]
    if levels:
        N, i, j = world.N, world.A_f[1], world.A_f[2]
        corners = world.corners
        points = []
        for level in levels:
            C = Fraction(N, 1 << level)
            I, J = C * round(i / C), C * round(j / C)
            J = min(J, N - I)
            Y = (I + C, J) if I + J + C <= N else (I - C, J)
            for a, b in ((I, J), Y):
                points.append((corners, (N - a - b, a, b)))
        p, _normals = placed(world, points)
        sizes.update(zip(levels, np.linalg.norm(p[1::2] - p[0::2], axis=1).tolist()))
    return sizes


class IcoChart:
    """The icosahedron sphere's chart for parts_sphere.SphereLattice (parts_sphere.CubeChart is the cube's): its
    polyhedron the icosahedron's 20 triangles, its world points (corners, weights)"""

    name = "ico"
    shift = 0                    # a level's cells are N / 2^level units

    @staticmethod
    def topology():
        return topology()

    @staticmethod
    def world(n, A, rho):
        return sphere_ico.IcoWorld(n, A, rho)

    @staticmethod
    def cell(world, level):
        return Fraction(world.N, 1 << level)

    key_point = staticmethod(key_point)
    scaled = staticmethod(scaled)
    placed = staticmethod(placed)
    placer = staticmethod(placer)

    @staticmethod
    def key_of(point, C, M):
        return point_key(*point, C, M)

    @staticmethod
    def local_placed(level, center, lattice):
        """A LocalLattice's points round center on the sphere, in the level's own world (n = level + REDUCED, anchored
        at center, its cells 2^REDUCED units): (places, normals), each point placed from its whole weights (the flat
        icosahedron's places aren't the chart's)"""
        M, grow = 1 << level, 1 << parts_sphere.REDUCED
        local = sphere_ico.IcoWorld(level + parts_sphere.REDUCED, scaled(key_point(center, M), grow), 1.0)
        return placed(local, [scaled(key_point(key, M), grow) for key in lattice.keys])

    @staticmethod
    def on_level(world, point, C):
        """Whether a world point is a lattice point of the level of cells C units: its weights whole multiples of C"""
        return all((Fraction(w) / C).denominator == 1 for w in point[1])

    @staticmethod
    def near(world, point):
        """Whether a world point is near enough the map's anchor to be placed from it in float64: within N >>
        NEAR_SHIFT units of it, deep enough to need no digits past float64's"""
        if world.n >= parts_sphere.NEAR_N:
            return False
        m = offsets(world, [point])
        return bool(np.isfinite(m).all()) and float(np.linalg.norm(m)) <= math.ldexp(1.0, world.n
                                                                                     - parts_sphere.NEAR_SHIFT)

    @staticmethod
    def exact_placed(exact, world, points):
        """World points and their normals with a placer's digits, in its folded frame"""
        return exact.placed_points(points)

    @staticmethod
    def below(world, q):
        return Below(world, q)

    @staticmethod
    def center_point(below, C):
        return below.lattice_point(C)

    @staticmethod
    def to_corner(world, below):
        """The point below's distance from the domain's corner V, in units (the skew place as x + i y)"""
        vi = sphere.wall_float(below.base[0]) + below.di
        vj = sphere.wall_float(below.base[1]) + below.dj
        return math.hypot(vi + vj / 2.0, ROOT3_HALF * vj)

    @staticmethod
    def cell_place(world, below, C, level):
        """Where the point below is in its triangle of the level of cells C units: its distances from the lattice
        corners round it (sorted, triangle sides) and from the nearest lattice line (the three families: J, I and
        I + J whole, each a triangle's height, sqrt 3 / 2, apart)"""
        n = world.n
        fI = float((Fraction(below.base[0]) % C) / C) + math.ldexp(below.di, level - n)
        fJ = float((Fraction(below.base[1]) % C) / C) + math.ldexp(below.dj, level - n)
        fI, fJ = fI - math.floor(fI), fJ - math.floor(fJ)
        corners = sorted(math.hypot(fI - a + (fJ - b) / 2.0, ROOT3_HALF * (fJ - b))
                         for a in (-1, 0, 1, 2) for b in (-1, 0, 1, 2))
        return corners, min(abs(x - round(x)) for x in (fI, fJ, fI + fJ)) * ROOT3_HALF

    @staticmethod
    def distortion(n, level, to_corner):
        """parts_sphere.CubeChart.distortion() for the icosahedron's cells (N / 2^level units): the pinch at its
        corners (the chart's conformal factor changes by about 1 / 5 of the distance from V, taken as the cube's 1 / 3)
        and the face's own"""
        return parts_sphere._distortion(n - level, to_corner, level)

    @staticmethod
    def cell_sizes(world, levels, below=None):
        return cell_sizes(world, levels)


ICO = IcoChart()
LATTICE = parts_sphere.SphereLattice(ICO)


def forget():
    _perms.clear()
    _sizes.clear()
    _placers.clear()
    _directions.clear()
    _constants.clear()
