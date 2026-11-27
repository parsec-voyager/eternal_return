"""The sphere maps' Sierpinski motifs, filled color rather than lines on dark (sphere.py's "sierpinski" motif, the
default; previewed in ~/blender_tests/sphere/tests/folds/cube_sphere_explorer.html, its "Layered" levels):
- the cube's binary carpet (Carpet): Pascal's triangle mod 2 from every face's corners. Measured from a face's edges
  toward its middle, a square is cut out at a level where both coordinates' binary digits there are 1: each cell of a
  level has its quarter toward the face's middle cut out, outlined along its two inner sides. The face's middle is the
  first level's cut-out, and three faces' right-angled gaskets meet at each cube corner.
- the icosahedron's Sierpinski globe (Gasket): every triangle's middle quarter is cut out and outlined, at every level,
  so all 20 faces are nested Sierpinski gaskets, five of them meeting at each of the icosahedron's 12 corners.
A cut-out is colored by its level (its hue turns HUE_STEP a level), bright where its points have been cut out an odd
number of times and dim where even, so nested cut-outs alternate. Every cut-out holds the pattern again, smaller, so a
dive never ends in a flat patch.

The levels are layered as the star motifs' are: each is drawn over the coarser ones, coming in as its cells grow on
screen from SHOW_FROM to SHOW_FULL pixels across, so zooming in brings smaller shapes in on top of bigger ones. The
outlines stay about a pixel wide at every level (OUTLINE of a cell, held between OUTLINE_MIN and OUTLINE_MAX pixels),
so a coarse level's don't swell into bands.

Every edge is at 0, 1/2 or 1 of a level's cells, dyadic, so the GPU places it exactly at any level, as the stars' lines
(sphere.py's lineAt()).
- The carpet's cells are the lattice's, measured from the face's middle. The carpet is mirrored across the face's
  middle lines and the cube's edges, which are walls of the chart's domain, so a cell past one (only a pixel rounded
  across it gets there) is read mirrored, and the plan's tests of the view fold it there too. A cut-out's count needs
  every coarser level: the shader counts as it walks the levels, from the count above the levels it draws
  (Carpet.cuts()).
- The gasket's count is the orientation of the triangle that's cut out: each cut turns a point's triangle round, so
  a cut-out pointing as its face does was cut an even number of times. Each level needs only its own place.
"""

import math
from fractions import Fraction

MOTIF = "sierpinski"

# A level's shapes come in as its cells grow from SHOW_FROM to SHOW_FULL pixels across (the finest level drawn is the
# first whose cells are SHOW_FROM pixels where the floor is finest)
SHOW_FROM, SHOW_FULL = 4.0, 10.0
# An outline's half-width: this share of a cell (the carpet's cell side, the gasket's triangle side), held between these
# many pixels of footprint
OUTLINE = 0.025
OUTLINE_MIN, OUTLINE_MAX = 0.4, 1.0
# The cut-outs' colors: hue by level, from HUE0 at the first; dim ones DIM of the bright ones' value
HUE0, HUE_STEP = 0.58, 0.11
SATURATION = 0.72
DIM = 0.72
# The outlines' color, and the color of what's never cut out (sRGB)
INK = (14.0, 11.5, 33.2)
UNCUT = (33.2, 25.5, 71.4)

CUT, FAR = "cut", "far"
# The cells (from the anchor's) a level's data says are out of reach
OUT_OF_REACH = 2 ** 20


def sphere_module():
    """sphere.py, for what the motifs share with it (words(), the colors), imported when needed: sphere imports this
    module"""
    try:
        from . import sphere
    except ImportError:
        import sphere
    return sphere


def reach(cells):
    """A count of cells for the shader, or a big number of the same sign when it's out of reach (the sign compared, not
    copied: a count past 2^1024, a level's cells from a dive that deep, has no float)"""
    if abs(cells) < OUT_OF_REACH:
        return float(cells)
    return 1.0e9 if cells > 0 else -1.0e9


def hole_srgb(index, odd):
    """A cut-out's color (sRGB 0..255) at a pattern's level index (the carpet's level - 1, the gasket's level), bright
    if its points were cut out an odd number of times"""
    return sphere_module().hsv(HUE0 + HUE_STEP * index, SATURATION, 1.0 if odd else DIM)


def hole_colors(index):
    """hole_srgb()'s bright and dim colors, linear, for the shader"""
    to_linear = sphere_module().srgb_to_linear
    return to_linear(hole_srgb(index, True)), to_linear(hole_srgb(index, False))


# ----- the cube's binary carpet -----

def carpet_zone(c, extent, C, N, margin):
    """Where a view is along one axis of the carpet at a level: c the folded anchor's coordinate (units from the face's
    middle), extent the view's reach either side of it (units), C the level's cells (units) and margin how far (in
    cells) the level's edges must stay for no pixel to cover them. CUT if the view stays in the cut-outs' half of a
    cell ([0, 1/2] from the cell's side toward the face's middle), FAR if in the other half, else None. The view is
    folded across the face's middle line (0) and the cube's edge (N) first, where the carpet is mirrored."""
    lo, hi = Fraction(c) - extent, Fraction(c) + extent
    if lo < 0:
        lo, hi = Fraction(0), max(-lo, hi)
    if hi > N:
        lo, hi = min(lo, 2 * N - hi), Fraction(N)
    M = int(N / C)
    x_lo, x_hi = lo / C, hi / C
    K = min(math.floor(x_lo), M - 1)
    if x_hi > K + 1:
        return None
    near = 0 if K == 0 else margin           # across the middle line, the cut-out carries on mirrored
    far = 0 if K == M - 1 else margin        # past the cube's edge, the cell is this one mirrored
    if x_lo >= K + near and x_hi <= K + Fraction(1, 2) - margin:
        return CUT
    if x_lo >= K + Fraction(1, 2) + margin and x_hi <= K + 1 - far:
        return FAR
    return None


class Carpet:
    """The cube's binary carpet (see the module's docstring), for sphere.Plan and the shader: level ell's cells are
    2N / 2^ell units, and its pattern's index (the explorer's level) is ell - 1"""

    name = MOTIF
    layout = "cube"
    first_level = 1                  # level 1's cells are a face's quarters, which meet at its middle
    base = UNCUT
    finest_footprint = 1.0 / SHOW_FROM

    def level_data_exact(self, world, level):
        """The folded anchor's place in its cell of a level along a and b (three float32 words each), the cells from its
        own below which the carpet is mirrored (the face's middle line) and from which on (the cube's edge), along
        each, and the level's colors"""
        sp = sphere_module()
        C = world.cell(level)
        M = 1 << (level - 1)
        values, low, high = [], [], []
        for c in (int(world.A_f[1]), int(world.A_f[2])):
            q = Fraction(c) / C
            K = math.floor(q)
            values += sp.words(q - K)
            low.append(reach(-K))
            high.append(reach(M - K))
        return {"words": values, "low": low, "high": high, "corner_low": reach(-M), "colors": hole_colors(level - 1),
                "cell": C}

    def corner_level_data(self, world, level):
        """level_data_exact() measured from the cube's corner, (N, N), the start of the cells past the cube's edge"""
        M = 1 << (level - 1)
        return {"words": [0.0] * 6, "low": [reach(-M)] * 2, "high": [0.0, 0.0], "corner_low": reach(-M),
                "colors": hole_colors(level - 1), "cell": None}

    def level_rows(self, world, level, data, kappa_anchor, kappa_corner, fine):
        """A level's rows of 4 for the shader's levelAt()"""
        w, (low_a, low_b), (high_a, high_b) = data["words"], data["low"], data["high"]
        bright, dim = data["colors"]
        return [(kappa_anchor, kappa_corner, data["corner_low"], 0.0), tuple(w[0:4]), (w[4], w[5], low_a, high_a),
                (low_b, high_b, 0.0, 0.0), (*bright, 0.0), (*dim, 0.0)]

    def uniform_exact(self, world, level, extent, cuts_offset=0):
        """Whether a level is one color over the view (offsets up to extent units along a and b from the folded
        anchor): None if an edge or outline may pass through it, else the color of its cut-out covering the view, or
        False if none (sphere.uniform_exact()'s). Exact, in Fractions; the edges kept 2 views' reach away, past a
        pixel's outline and its own footprint. cuts_offset: cut-outs above a planet's handoff, not counted."""
        C = world.cell(level)
        e = Fraction(extent) / C
        if e >= Fraction(1, 8):
            return None
        zones = [carpet_zone(int(c), Fraction(extent), C, world.N, 2 * e) for c in world.A_f[1:3]]
        if FAR in zones:
            return False
        if zones == [CUT, CUT]:
            return hole_srgb(level - 1, (self.cuts(world, level) - cuts_offset) % 2)
        return None

    def cuts(self, world, level):
        """How many times (mod 2) the folded anchor is cut out at the levels from the first to level: at a level ell <=
        n, a coordinate is in a cell's cut-out half where its bit n - ell is 0 (the cells are 2^(n + 1 - ell) units),
        and below the unit lattice always (it's on the cells' sides toward the middle). On the cube's edge, never. Only
        asked of a level whose cut-out covers the view, so the anchor is on no coarser level's edge (but the middle
        line's, where the carpet is mirrored)."""
        n, N = world.n, world.N
        a, b = int(world.A_f[1]), int(world.A_f[2])
        if a == N or b == N:
            return 0
        above = min(level, n)
        bits = ((a | b) >> (n - above)) & ((1 << above) - 1)
        count = above - bin(bits).count("1") + max(0, level - n)
        return count % 2

    def paint_source(self):
        return CARPET_SOURCE

    def defines(self, info):
        defines(info)


CARPET_SOURCE = """
// How much a level shows, its cells 1 / F pixels across (F the footprint in cells), and whether it's too fine for any
float showLevel(float F) { return smoothstep(SHOW_FROM, SHOW_FULL, 1.0 / max(F, 1.0e-30)); }
bool tooFine(float F) { return F * SHOW_FROM >= 1.0; }
float outlineWidth(float F) { return clamp(OUTLINE, OUTLINE_MIN * F, OUTLINE_MAX * F); }

// A level's rows (sphere_sierpinski.Carpet.level_rows()): the anchor's place in its cell along a and b, the cells (from
// the reference's) below which the carpet is mirrored across the face's middle line, and from which on across the
// cube's edge, and the cut-outs' colors
struct Level {
    vec4 t0;
    vec3 wa, wb;
    vec2 lowCell, highCell;
    vec3 bright, dim;
    float toCorner;     // how far the anchor's cell is from the cube corner's, in cells along both axes
};

Level levelAt(int i, bool useCorner)
{
    Level L;
    L.t0 = levelValues.rows[LEVEL_ROWS * i];
    vec4 t2 = levelValues.rows[LEVEL_ROWS * i + 2];
    vec4 t3 = levelValues.rows[LEVEL_ROWS * i + 3];
    L.bright = levelValues.rows[LEVEL_ROWS * i + 4].rgb;
    L.dim = levelValues.rows[LEVEL_ROWS * i + 5].rgb;
    L.toCorner = abs(t2.w) + abs(t3.y);
    if (useCorner) {
        L.wa = vec3(0.0); L.wb = L.wa;
        L.lowCell = vec2(L.t0.z); L.highCell = vec2(0.0);
    }
    else {
        vec4 t1 = levelValues.rows[LEVEL_ROWS * i + 1];
        L.wa = t1.xyz; L.wb = vec3(t1.w, t2.xy);
        L.lowCell = vec2(t2.z, t3.x); L.highCell = vec2(t2.w, t3.y);
    }
    return L;
}

vec3 holeColor(Level L, float cuts)
{
    return mod(cuts, 2.0) > 0.5 ? L.bright : L.dim;
}

// The rough place along an axis (x, in cells from the reference's): in its cell, from the side toward the face's middle
float roughPlace(float x, float low, float high)
{
    float k = floor(x);
    return k < low || k >= high ? 1.0 - (x - k) : x - k;
}

// One axis of the carpet at a pixel, exactly (lineAt()): its place in its cell from the cell's side toward the face's
// middle line, and that minus 1/2 and minus 1, each exact near 0; whether that side is the middle line (across which the
// cut-out carries on, mirrored), and whether the other is the cube's edge (past which the next cell is this one
// mirrored, so it has no cut-out there). Cells below low are mirrored across the middle line, from high on across the
// edge.
struct CarpetAxis {
    float place, pastHalf, pastOne;
    bool middle, edge;
};

CarpetAxis carpetAxis(vec3 w, float d, float low, float high)
{
    float k = cellOffset(w, d);
    float x = lineAt(w, k, 0.0, d), h = lineAt(w, k, 0.5, d), y = lineAt(w, k, 1.0, d);
    bool mirrored = k < low || k >= high;
    CarpetAxis A;
    A.place = mirrored ? -y : x;
    A.pastHalf = mirrored ? -h : h;
    A.pastOne = mirrored ? -x : y;
    A.middle = k == low || k == low - 1.0;
    A.edge = k == high - 1.0 || k == high;
    return A;
}

// How much of a pixel F wide along an axis is in its cell's span from 0 (open on the middle line) to toEnd past it,
// or in the next cell's from its start
float carpetSpans(CarpetAxis A, float toEnd, float F, float invF)
{
    float own = spanScaled(A.middle ? 1.0e30 : A.place, toEnd, F, invF);
    return own + (A.edge ? 0.0 : spanScaled(A.pastOne, 1.0e30, F, invF));
}

// One level painted over color (sphere.paint_level's interface): d the pixel's offset from the reference in this
// level's cells, ga, gb its steps per pixel; cuts counts the times the pixel has been cut out, coarse to fine
vec3 paintAt(vec3 color, Level L, vec2 d, vec2 ga, vec2 gb, vec4 info, inout float cuts)
{
    float Fx = max(abs(ga.x) + abs(ga.y), 1.0e-30), Fy = max(abs(gb.x) + abs(gb.y), 1.0e-30);
    float show = showLevel(max(Fx, Fy));
    // Most pixels are far from the level's edges: a rough place tells which part of its cell a pixel is in (good to
    // float32's rounding of its offset). An outline reaches at most a pixel past its line, and a pixel half its own.
    float off = ROUGH * (1.0 + abs(d.x) + abs(d.y));
    float ua = roughPlace((L.wa.x + d.x) + (L.wa.y + L.wa.z), L.lowCell.x, L.highCell.x);
    float ub = roughPlace((L.wb.x + d.y) + (L.wb.y + L.wb.z), L.lowCell.y, L.highCell.y);
    float ma = 1.5 * Fx + off, mb = 1.5 * Fy + off;
    if ((ua > 0.5 + ma && ua < 1.0 - ma) || (ub > 0.5 + mb && ub < 1.0 - mb)) {
        return color;
    }
    if (ua > ma && ua < 0.5 - ma && ub > mb && ub < 0.5 - mb) {
        cuts += 1.0;
        return mix(color, holeColor(L, cuts), show);
    }
    // Near an edge: every distance exactly
    CarpetAxis A = carpetAxis(L.wa, d.x, L.lowCell.x, L.highCell.x);
    CarpetAxis B = carpetAxis(L.wb, d.y, L.lowCell.y, L.highCell.y);
    float ix = 1.0 / Fx, iy = 1.0 / Fy;
    // The cut-out: the quarter of the cell toward the face's middle, and of the next cells'
    float cover = carpetSpans(A, -A.pastHalf, Fx, ix) * carpetSpans(B, -B.pastHalf, Fy, iy);
    color = mix(color, holeColor(L, cuts + 1.0), show * cover);
    if (A.pastHalf < 0.0 && B.pastHalf < 0.0) {
        cuts += 1.0;
    }
    // Its outline along its two inner sides, each on past the corner by the other's half-width, so they meet square
    float ha = outlineWidth(Fx), hb = outlineWidth(Fy);
    float inkA = spanScaled(A.pastHalf + ha, ha - A.pastHalf, Fx, ix) * carpetSpans(B, hb - B.pastHalf, Fy, iy);
    float inkB = spanScaled(B.pastHalf + hb, hb - B.pastHalf, Fy, iy) * carpetSpans(A, ha - A.pastHalf, Fx, ix);
    return mix(color, INK, show * max(inkA, inkB));
}

// tileMain()'s test of a level (see the stars' tileClear()): every pixel of the tile is in the far half of its cell
// along one axis, clear of the outlines, so none is cut out
bool tileClear(Level L, vec2 d, float F, float r, bool useCorner)
{
    float far = abs(d.x) + abs(d.y) + 2.0 * r + (useCorner ? L.toCorner : 0.0);
    float m = 2.0 * ROUGH * (2.0 + far) + r + 1.5 * F;
    float ua = roughPlace((L.wa.x + d.x) + (L.wa.y + L.wa.z), L.lowCell.x, L.highCell.x);
    float ub = roughPlace((L.wb.x + d.y) + (L.wb.y + L.wb.z), L.lowCell.y, L.highCell.y);
    return (ua > 0.5 + m && ua < 1.0 - m) || (ub > 0.5 + m && ub < 1.0 - m);
}

vec4 foldInfo(mat3 F)
{
    return vec4(0.0);
}
"""


# ----- the icosahedron's Sierpinski globe -----

class Gasket:
    """The icosahedron's Sierpinski globe (see the module's docstring), for sphere.Plan and the shader: level ell's
    triangles are N / 2^ell units on a side, and its pattern's index is ell. Places are in skew coordinates (I, J, and
    I + J, of which a triangle's height is 1), as sphere_ico's rosettes': the middle lines of a level's triangles are
    where any of them is k + 1/2, and a triangle points as its face does where I + J's cell is I's plus J's."""

    name = MOTIF
    layout = "ico"
    first_level = 0                  # level 0's triangle is the face
    base = UNCUT
    finest_footprint = 1.0 / SHOW_FROM

    def level_data_exact(self, world, level):
        """The folded anchor's place in its triangle of a level, I, J and I + J (three float32 words each), how many
        cells it is from V (the domain's corner, for the shader's error bounds) and the level's colors"""
        sp = sphere_module()
        C, ((kI, rI), (kJ, rJ)) = world.level_cell(level)
        values = []
        for r in (rI, rJ, rI + rJ):
            values += sp.words(r)
        return {"words": values, "to_corner": reach(kI + kJ + 2), "colors": hole_colors(level), "cell": C}

    def corner_level_data(self, world, level):
        """level_data_exact() measured from V, a corner of every level"""
        return {"words": [0.0] * 9, "to_corner": 0.0, "colors": hole_colors(level), "cell": None}

    def level_rows(self, world, level, data, kappa_anchor, kappa_corner, fine):
        w = data["words"]
        bright, dim = data["colors"]
        return [(kappa_anchor, kappa_corner, data["to_corner"], 0.0), tuple(w[0:4]), tuple(w[4:8]), (w[8], 0.0, 0.0, 0.0),
                (*bright, 0.0), (*dim, 0.0)]

    def uniform_exact(self, world, level, extent):
        """Carpet.uniform_exact() for the gasket: the view reaches 2 e in skew units (e its extent in triangle sides),
        and its middle lines are kept as far again away. Past the domain's walls the lattice carries on mirrored, and
        so do the middle lines and cut-outs (only their brightness would turn, across a face's edge, which no cut-out
        crosses)."""
        C, ((kI, rI), (kJ, rJ)) = world.level_cell(level)
        e = Fraction(extent) / C
        if e >= Fraction(1, 8):
            return None
        half = Fraction(1, 2)
        rS = rI + rJ
        fS = rS - math.floor(rS)
        if min(abs(rI - half), abs(rJ - half), abs(fS - half)) <= 4 * e:
            return None
        up = rS < 1
        cut = (rI < half and rJ < half and fS > half) if up else (rI > half and rJ > half and fS < half)
        return hole_srgb(level, up) if cut else False

    def cuts(self, world, level):
        return 0

    def paint_source(self):
        return GASKET_SOURCE

    def defines(self, info):
        defines(info)


GASKET_SOURCE = """
float showLevel(float F) { return smoothstep(SHOW_FROM, SHOW_FULL, 1.0 / max(F, 1.0e-30)); }
bool tooFine(float F) { return F * SHOW_FROM >= 1.0; }

// A level's rows (sphere_sierpinski.Gasket.level_rows()): the anchor's place in skew coordinates I, J, I + J, how many
// cells it is from V, the cut-outs' colors
struct Level {
    vec4 t0;
    vec3 wI, wJ, wS;
    vec3 bright, dim;
    float toCorner;
};

Level levelAt(int i, bool useCorner)
{
    Level L;
    L.t0 = levelValues.rows[LEVEL_ROWS * i];
    L.toCorner = L.t0.z;
    L.bright = levelValues.rows[LEVEL_ROWS * i + 4].rgb;
    L.dim = levelValues.rows[LEVEL_ROWS * i + 5].rgb;
    if (useCorner) {
        L.wI = vec3(0.0); L.wJ = L.wI; L.wS = L.wI;
    }
    else {
        vec4 t1 = levelValues.rows[LEVEL_ROWS * i + 1];
        vec4 t2 = levelValues.rows[LEVEL_ROWS * i + 2];
        vec4 t3 = levelValues.rows[LEVEL_ROWS * i + 3];
        L.wI = t1.xyz; L.wJ = vec3(t1.w, t2.xy); L.wS = vec3(t2.zw, t3.x);
    }
    return L;
}

// Whether a place in a level's triangle, by its skew coordinates' distances from their middle lines (h) and whether
// the triangle points as its face does, is in the triangle's middle, cut out
bool inMiddle(vec3 h, bool up)
{
    return up ? h.x < 0.0 && h.y < 0.0 && h.z > 0.0 : h.x > 0.0 && h.y > 0.0 && h.z < 0.0;
}

// One level painted over color (see the carpet's paintAt()): d the pixel's offset from the reference in this level's
// triangles (the face's x, y), ga, gb its steps per pixel
vec3 paintAt(vec3 color, Level L, vec2 d, vec2 ga, vec2 gb, vec4 info, inout float cuts)
{
    float dI = d.x - d.y * INV_ROOT3, dJ = 2.0 * INV_ROOT3 * d.y, dS = dI + dJ;
    vec2 gI = ga - gb * INV_ROOT3, gJ = 2.0 * INV_ROOT3 * gb, gS = gI + gJ;
    vec3 Fk = max(vec3(abs(gI.x) + abs(gI.y), abs(gJ.x) + abs(gJ.y), abs(gS.x) + abs(gS.y)), vec3(1.0e-30));
    float show = showLevel(max(abs(ga.x) + abs(ga.y), abs(gb.x) + abs(gb.y)));
    // Most pixels are far from the middle lines, the only edges: the rough place tells whether they're cut out
    float off = ROUGH * (1.0 + abs(dI) + abs(dJ));
    vec3 x = vec3((L.wI.x + dI) + (L.wI.y + L.wI.z), (L.wJ.x + dJ) + (L.wJ.y + L.wJ.z),
                  (L.wS.x + dS) + (L.wS.y + L.wS.z));
    vec3 k = floor(x);
    vec3 h = x - k - 0.5;
    if (all(greaterThan(abs(h), 1.5 * Fk + off))) {
        bool up = k.z == k.x + k.y;
        return inMiddle(h, up) ? mix(color, up ? L.bright : L.dim, show) : color;
    }
    // Near one: exactly
    k = vec3(cellOffset(L.wI, dI), cellOffset(L.wJ, dJ), cellOffset(L.wS, dS));
    h = vec3(lineAt(L.wI, k.x, 0.5, dI), lineAt(L.wJ, k.y, 0.5, dJ), lineAt(L.wS, k.z, 0.5, dS));
    bool up = k.z == k.x + k.y;
    // The middle triangle, inside its three sides: how many pixels outside each the pixel is
    vec3 outside = (up ? vec3(h.x, h.y, -h.z) : vec3(-h.x, -h.y, h.z)) / Fk;
    float cover = clip01(0.5 - max(max(outside.x, outside.y), outside.z));
    color = mix(color, up ? L.bright : L.dim, show * cover);
    // Its outline: the middle lines, about a pixel wide
    vec3 w = clamp(vec3(OUTLINE_SKEW), OUTLINE_MIN * Fk, OUTLINE_MAX * Fk);
    float ink = max(max(spanCover(h.x + w.x, w.x - h.x, Fk.x), spanCover(h.y + w.y, w.y - h.y, Fk.y)),
                    spanCover(h.z + w.z, w.z - h.z, Fk.z));
    return mix(color, INK, show * ink);
}

// tileMain()'s test of a level: every pixel of the tile is clear of the middle lines (whose outlines reach 1.5
// footprints, at most 2 F in skew units, as the tile's reach is at most 2 r) and outside the middle triangle
bool tileClear(Level L, vec2 d, float F, float r, bool useCorner)
{
    float far = abs(d.x) + abs(d.y) + 2.0 * r + (useCorner ? L.toCorner : 0.0);
    float m = 2.0 * ROUGH * (2.0 + far) + 2.0 * r + 3.0 * F;
    float dI = d.x - d.y * INV_ROOT3, dJ = 2.0 * INV_ROOT3 * d.y, dS = dI + dJ;
    vec3 x = vec3((L.wI.x + dI) + (L.wI.y + L.wI.z), (L.wJ.x + dJ) + (L.wJ.y + L.wJ.z),
                  (L.wS.x + dS) + (L.wS.y + L.wS.z));
    vec3 k = floor(x);
    vec3 h = x - k - 0.5;
    return all(greaterThan(abs(h), vec3(m))) && !inMiddle(h, k.z == k.x + k.y);
}

vec4 foldInfo(mat3 F)
{
    return vec4(0.0);
}
"""


def defines(info):
    """Both motifs' shader defines"""
    sp = sphere_module()
    info.define("SHOW_FROM", repr(SHOW_FROM))
    info.define("SHOW_FULL", repr(SHOW_FULL))
    info.define("OUTLINE", repr(OUTLINE))
    info.define("OUTLINE_SKEW", repr(OUTLINE * 2.0 / math.sqrt(3.0)))
    info.define("OUTLINE_MIN", repr(OUTLINE_MIN))
    info.define("OUTLINE_MAX", repr(OUTLINE_MAX))
    info.define("INK", sp.glsl_vec3(INK))


CARPET = Carpet()
GASKET = Gasket()
