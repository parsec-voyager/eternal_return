"""The looks a surface map is given (phase 7 of the engines' plan, 2026-11-25: one place for them, the add-on's, which
the Tile Selected Object button and the map builders both take them from): a floor motif on every level's cells
(surface_substitution.py's settings: strapwork, the Inferno's stars, or a plain floor), the 3D frames standing on the
lattice (parts.py's settings: open frames), the mesh's own faces' tones (the floor's colors between motif bars), and
the scaling config that makes a map loop and scale with the drone.

The motifs are flat pieces in a cell's units, [0, 1] x [0, 1], each a polygon with a paint (motif.Pieces lays them out):
bands as a core between two outline strips of the floor's color, so diving into any band reaches the dark floor again
through its finer levels' outlines (every endless floor's look since 2026-11-15). The frames: an octagonal post on
every vertex, and along every edge a beam with a horseshoe ring standing under its middle, thin and open between, so
the coarser levels' frames round the drone don't wall it in; four colors by level in turn, the posts ivory.
"""

import colorsys
import json
import math

import numpy as np


# ----- colors -----

def srgb(hex_color, alpha=1.0):
    """A #rrggbb color as linear RGBA"""
    def channel(c):
        c = int(hex_color[c:c + 2], 16) / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return [channel(1), channel(3), channel(5), alpha]


def hsv(h, s, v, a=1.0):
    """An sRGB HSV color as linear RGBA"""
    rgb = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb] + [a]


# The mesh's faces' tones: dark, so the motif's bars and the frames read over them
FACE_TONES = ("#1c1747", "#0f3140", "#2a1438")


def face_tones(mesh, tones=FACE_TONES):
    """A face "Col" attribute (the floor's colors, surface.Topology reads it): each face one of tones, none the same as
    a face it shares an edge with (greedily, in face order)"""
    colors = [srgb(t) for t in tones]
    by_edge = {}
    for p in mesh.polygons:
        for key in p.edge_keys:
            by_edge.setdefault(key, []).append(p.index)
    chosen = []
    for p in mesh.polygons:
        taken = {chosen[g] for key in p.edge_keys for g in by_edge[key] if g < p.index}
        chosen.append(next((k for k in range(len(colors)) if k not in taken), p.index % len(colors)))
    attribute = mesh.attributes.get("Col") or mesh.attributes.new("Col", 'FLOAT_COLOR', 'FACE')
    attribute.data.foreach_set("color", [c for k in chosen for c in colors[k]])


# ----- flat pieces in a cell -----

def counterclockwise(polygons):
    """Polygons (n, k, 2), each reversed if it winds clockwise, so faces built from them point up"""
    polygons = np.asarray(polygons, dtype=float)
    x, y = polygons[..., 0], polygons[..., 1]
    area = (x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(axis=1)
    polygons = polygons.copy()
    polygons[area < 0] = polygons[area < 0, ::-1]
    return polygons


def star_polygon(points, outer, inner, center=(0.0, 0.0), angle=0.0):
    """A star's outline as a closed polygon, counterclockwise: points tips at radius outer, starting at angle, and
    between them corners at radius inner. Two squares, one turned 45 degrees, are star_polygon(8, r, r cos(pi / 4) /
    cos(pi / 8))."""
    k = np.arange(2 * points)
    radius = np.where(k % 2 == 0, outer, inner)
    theta = angle + k * math.pi / points
    return np.stack([center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)], axis=1)


def outline(polygon, width):
    """A closed polygon's outline, width wide and centered on it, as one quad per edge (n, 4, 2), each
    counterclockwise, meeting at mitered corners (no gaps, no overlaps). Quad k runs along the edge from corner k to
    corner k + 1, its corners 0 to 1 and 3 to 2 along it."""
    polygon = np.asarray(polygon, dtype=float)
    edges = np.roll(polygon, -1, axis=0) - polygon
    normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    # A corner's miter: along the two edges' normals' bisector, as far as keeps both sides width / 2 off the edges
    before, after = np.roll(normals, 1, axis=0), normals
    bisector = before + after
    miter = bisector / (bisector * after).sum(axis=1, keepdims=True) * (width / 2)
    out, inside = polygon + miter, polygon - miter
    quads = np.stack([inside, np.roll(inside, -1, axis=0), np.roll(out, -1, axis=0), out], axis=1)
    return counterclockwise(quads)


def bar(start, end, half_width):
    """A straight bar from start to end, half_width either side of the line, as a counterclockwise quad (4, 2)"""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    along = (end - start) / np.linalg.norm(end - start)
    across = np.array([-along[1], along[0]]) * half_width
    return counterclockwise(np.array([start - across, end - across, end + across, start + across])[None])[0]


def strap(start, end, offset, half_width, start_edge, end_edge, start_cut=0.0, end_cut=None):
    """A strip along the line from start to end, offset across it (to its left) and half_width either side of that, its
    ends cut along the lines through start and end square to start_edge and end_edge (unit normals pointing in, toward
    the other end), start_cut and end_cut (the same unless given) in from them: a band ending where it passes under
    another, the way strapwork shows a crossing. A counterclockwise quad (4, 2), its corners 0 to 1 and 3 to 2 along it."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    along = (end - start) / np.linalg.norm(end - start)
    left = np.array([-along[1], along[0]])

    def corner(point, edge, cut, side):
        # point + left side + along t, cut in from the line through point square to edge
        edge = np.asarray(edge, dtype=float) / np.linalg.norm(edge)
        return point + left * side + along * (cut - side * (left @ edge)) / (along @ edge)
    end_cut = start_cut if end_cut is None else end_cut
    low, high = offset - half_width, offset + half_width
    return counterclockwise(np.array([corner(start, start_edge, start_cut, low), corner(end, end_edge, end_cut, low),
                                      corner(end, end_edge, end_cut, high),
                                      corner(start, start_edge, start_cut, high)])[None])[0]


def square(half, center=(0.0, 0.0)):
    """A square half on each side of center, counterclockwise (4, 2)"""
    (u, v), h = center, half
    return np.array([(u - h, v - h), (u + h, v - h), (u + h, v + h), (u - h, v + h)])


def strapwork_pieces(inset=0.07, frame=0.012, frame_core=0.008, diamond=0.011, diamond_core=0.007, cut=0.008):
    """The strapwork motif: a gold frame inset in from the cell's edges (half width frame, core frame_core), the four
    sides of the diamond through the edges' midpoints within it (half width diamond, core diamond_core), cut short of
    the frame by cut, and the interlaced crossings of the diamonds' lines in the gutters at the cell's bottom and left
    midpoints (its neighbors draw the other two). Every band is its core between two borders. Paints: "square" (the
    frame), "diamond", and "diamond_below" and "diamond_left" (the neighbors' lines where they pass under this cell's),
    each with a "_border"."""
    gap = cut
    inner = inset + frame + gap         # where the diamond's sides start, in from the cell's edges
    reach = inset - frame - gap         # how far the crossings in the gutters reach, from the cell's edges
    under = diamond + gap               # where the line below starts, from the middle of the one on top
    # A level's pieces lie in one plane, so they mustn't overlap: two diamond bands crossing at right angles overlap
    # within diamond sqrt(2) of the crossing along the gutter's middle, so the crossing reaches further than that
    assert reach > diamond * math.sqrt(2) and inner < 0.25

    def band(strip, paint, half, core):
        """A band as its core and a border to each side: strip(offset, half_width) draws one strip, offset across"""
        border = (half - core) / 2
        return [(strip(0.0, core), paint), (strip(-(core + border), border), paint + "_border"),
                (strip(core + border, border), paint + "_border")]

    def frame_strip(offset, half):
        # The frame's outline, offset out from its middle, as one mitered quad per side
        return outline(square(0.5 - inset + offset, (0.5, 0.5)), 2 * half)

    pieces = [(quad, paint) for strips, paint in band(frame_strip, "square", frame, frame_core) for quad in strips]

    def strap_band(start, end, start_edge, end_edge, start_cut, end_cut, paint):
        return band(lambda offset, half: strap(start, end, offset, half, start_edge, end_edge, start_cut, end_cut),
                    paint, diamond, diamond_core)

    # The diamond's sides, from the edges' midpoints (the edges' normals there pointing into the cell), cut at the frame
    midpoints = [(0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5)]
    inward = [(0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (1.0, 0.0)]
    for k in range(4):
        n = (k + 1) % 4
        pieces += strap_band(midpoints[k], midpoints[n], inward[k], inward[n], inner, inner, "diamond")
    # The crossings in the gutters, at the bottom midpoint (along x, "/" on top) and the left one (along y, "\" on
    # top), out to reach either side of the cell's edge: the line on top, then the one below in two pieces
    r = reach
    pieces += strap_band((0.5 - r, -r), (0.5 + r, r), (0.0, 1.0), (0.0, -1.0), 0.0, 0.0, "diamond")
    pieces += strap_band((0.5, 0.0), (0.5 - r, r), (-1.0, 1.0), (0.0, -1.0), under, 0.0, "diamond")
    pieces += strap_band((0.5, 0.0), (0.5 + r, -r), (1.0, -1.0), (0.0, 1.0), under, 0.0, "diamond_below")
    pieces += strap_band((-r, 0.5 + r), (r, 0.5 - r), (1.0, 0.0), (-1.0, 0.0), 0.0, 0.0, "diamond")
    pieces += strap_band((0.0, 0.5), (r, 0.5 + r), (1.0, 1.0), (-1.0, 0.0), under, 0.0, "diamond")
    pieces += strap_band((0.0, 0.5), (-r, 0.5 - r), (-1.0, -1.0), (1.0, 0.0), under, 0.0, "diamond_left")
    return pieces


# The shear that makes the cell [0, 1] x [0, 1] a rhomb of two equilateral triangles of side 1, (0, 0), (1, 0),
# (0, 1) and (1, 0), (1, 1), (0, 1): a triangle face's cells are halves of its square cells
EQUILATERAL = ((2.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)), (0.0, 1.0))


def triangle_pieces(shear=EQUILATERAL, inset=0.07, frame=0.012, frame_core=0.008, medial=0.011,
                    medial_core=0.007, cut=0.008):
    """The strapwork motif on triangles (the triangle maps' floor, and Strapwork on a triangle mesh's cells) as flat
    pieces in the cell [0, 1] x [0, 1], which shear makes a rhomb of two equilateral triangles, up (0, 0),
    (1, 0), (0, 1) and down (1, 0), (1, 1), (0, 1): drawn in the world's shape and taken back through shear's inverse,
    so its bands are as wide every way. Each triangle has a gold frame inset in from its sides (half width frame, its
    core frame_core), and inside it its medial triangle (through its sides' middles, half width medial, core
    medial_core), cut short of the frame by cut: the medial triangles' sides are the next level's cells' sides, so they
    run on straight from triangle to triangle, through the gutters between frames, where two cross at each side's
    middle, interlaced (the line parallel to the triangle's next side, counterclockwise, on top). Paints: "frame",
    "up" and "down" (each medial triangle's, by its triangle; in a gutter the line on top the up one's), each with
    a "_border" either side."""
    gap = cut
    inner = inset + frame + gap         # where the medial sides start, in from the triangle's sides
    reach = inset - frame - gap         # how far the crossings reach either side of a side, in the gutter
    under = medial + gap                # where the line below starts, from the middle of the one on top
    # Two bands crossing at 60 degrees overlap within medial / sin(60) of the crossing across the gutter (and the line
    # below starts under / sin(60) along itself from it): the gutter must be wider than that
    assert reach > under / math.sin(math.pi / 3) + medial and inner < 1.0 / 6.0
    shear = np.array(shear, dtype=float)[:2, :2]
    to_cell = np.linalg.inv(shear)
    corners = shear @ np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])     # the cell's corners in the world
    p00, p10, p11, p01 = corners.T
    triangles = {"up": (p00, p10, p01), "down": (p10, p11, p01)}

    def band(strip, paint, half, core):
        """A band as its core and a border to each side: strip(offset, half_width) draws one strip, offset across"""
        border = (half - core) / 2
        return [(strip(0.0, core), paint), (strip(-(core + border), border), paint + "_border"),
                (strip(core + border, border), paint + "_border")]

    def inward(a, b, c):
        """The unit normal of side a-b pointing into the triangle (toward c)"""
        along = (b - a) / np.linalg.norm(b - a)
        n = np.array([-along[1], along[0]])
        return n if n @ (c - a) > 0 else -n

    pieces = []
    for name, (a, b, c) in triangles.items():
        middle = (a + b + c) / 3.0
        # The frame: the triangle shrunk about its middle until its sides are inset in from the triangle's (its
        # inradius is a third of its height, 1)
        shrink = (1.0 / 3.0 - inset) / (1.0 / 3.0)

        def frame_strip(offset, half, a=a, b=b, c=c, middle=middle, shrink=shrink):
            k = shrink + offset / (1.0 / 3.0)
            return outline(np.array([middle + k * (p - middle) for p in (a, b, c)]), 2 * half)
        pieces += [(quad, "frame" + paint[len("frame"):]) for strips, paint in band(frame_strip, "frame", frame,
                                                                                    frame_core)
                   for quad in strips]
        # The medial triangle's sides: from each side's middle to the next side's, cut at the frame
        sides = [(a, b, c), (b, c, a), (c, a, b)]
        for k in range(3):
            (p, q, r), (p2, q2, r2) = sides[k], sides[(k + 1) % 3]
            start, end = (p + q) / 2.0, (p2 + q2) / 2.0
            pieces += band(lambda offset, half, start=start, end=end, p=p, q=q, r=r, p2=p2, q2=q2, r2=r2:
                           strap(start, end, offset, half, inward(p, q, r), inward(p2, q2, r2), inner,
                                              inner), name, medial, medial_core)
    # The crossings in the gutters at the up triangle's sides' middles: each side's two lines, parallel to the
    # triangle's other two sides, out to reach either side of it
    a, b, c = triangles["up"]
    for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
        m = (p + q) / 2.0
        n = inward(p, q, r)
        over, below = (r - q) / np.linalg.norm(r - q), (r - p) / np.linalg.norm(r - p)

        def gutter_strap(start, end, start_edge, end_edge, start_cut, paint):
            return band(lambda offset, half: strap(start, end, offset, half, start_edge, end_edge,
                                                                start_cut, 0.0), paint, medial, medial_core)
        t = reach / abs(over @ n)
        s0, s1 = m - t * over, m + t * over
        toward = n if n @ (s1 - s0) > 0 else -n
        pieces += gutter_strap(s0, s1, toward, -toward, 0.0, "up")
        # The line below in two, from the one on top (its normal toward each end) out to the gutter's sides
        across = np.array([-over[1], over[0]])
        t = reach / abs(below @ n)
        for sign in (1.0, -1.0):
            end = m + sign * t * below
            away = across if across @ (end - m) > 0 else -across
            out = n if n @ (end - m) > 0 else -n
            pieces += gutter_strap(m, end, away, -out, under, "down")
    return [(counterclockwise((to_cell @ np.asarray(polygon, dtype=float).T).T[None])[0], paint)
              for polygon, paint in pieces]


# The substitution rules' floors (a map's "rule"), and what their open frames change: Penrose levels are phi times
# finer, not 2: some 29 levels between cells a 32nd of the drone and the giant ones' threshold, against the square
# maps' 19, so more level objects, and with phi times as many levels each built out to 10 of its cells, not 14 (the
# next coarser goes on to 16 of them): 41 draws a second on the sphere against 29, the look the same (2026-11-16)
RULES = {"penrose": {"count": 36, "build": 10.0}}

# The floor motifs a map can have (floor_settings()), for the Tile Selected Object button's choice
FLOORS = (("strapwork", "Strapwork", "Gold square frames and interlaced diamonds, outlined, on every level's cells"),
          ("inferno", "Inferno", "The Inferno's eight-point stars and gold lines on every level's cells"),
          ("penrose", "Penrose", "Penrose rhombs' strapwork, never repeating at any level: the shape's tiles cut in "
           "rhombs, one at each tile's corner"),
          ("plain", "Plain", "The mesh's own faces' tones alone, no motif"))


def floor_settings(name, triangles=False):
    """The floor motif's settings (an object's "surface_substitution": surface_substitution.py draws them on a square
    lattice's cells, surface_cells.py), or None for a plain floor. On a mesh of triangles (triangles) Strapwork is
    the triangle maps' (triangle_pieces()), drawn on the triangles' cells"""
    from . import surface_substitution as SS
    settings = dict(SS.DEFAULTS)
    if name == "plain":
        return None
    if name in RULES:
        # A substitution rule's own strapwork (Hankin lines at 54 degrees), its defaults
        return {k: v for k, v in settings.items() if v is not None}
    if name == "strapwork" and triangles:
        pieces = triangle_pieces()
        hues = [0.55 + 0.25 * p for p in range(4)]
        settings.update({"paints": {"frame": [hsv(0.13, 0.8, 1.0)], "frame_border": [hsv(0.08, 0.9, 0.3)],
                                    "up": [hsv(h, 0.85, 1.0) for h in hues],
                                    "up_border": [hsv(h, 0.9, 0.3) for h in hues],
                                    "down": [hsv(h + 0.125, 0.85, 1.0) for h in hues],
                                    "down_border": [hsv(h + 0.125, 0.9, 0.3) for h in hues]},
                         # Its bands' borders are their outline strips (the outline color, left out far off): a
                         # border is (half width - core) / 2 of a band's 2 x half width each side
                         "half_width": 0.011, "outline": (0.011 - 0.007) / 2.0 / 0.022,
                         "strips": ["frame_border", "up_border", "down_border"]})
    elif name == "strapwork":
        pieces = strapwork_pieces()
        hues = [0.55 + 0.25 * p for p in range(4)]
        diamond = [hsv(h, 0.85, 1.0) for h in hues]
        settings.update({"paints": {"square": [hsv(0.13, 0.8, 1.0)], "diamond": diamond, "diamond_below": diamond,
                                    "diamond_left": diamond},
                         "half_width": 0.011, "outline": (0.011 - 0.007) / 2.0 / 0.022,
                         "strips": ["square_border", "diamond_border", "diamond_below_border",
                                    "diamond_left_border"]})
    elif name == "inferno":
        star_r, ring, line, gap = 0.36, 0.025, 0.01, 0.04
        star = star_polygon(8, star_r, star_r * math.cos(math.pi / 4) / math.cos(math.pi / 8), (0.5, 0.5))
        pieces = [(quad, "tip" if k % 4 in (0, 3) else "square") for k, quad in enumerate(outline(star, ring))]
        pieces += [(bar((gap, 0.0), (1 - gap, 0.0), line), "gold"), (bar((0.0, gap), (0.0, 1 - gap), line), "gold")]
        settings.update({"paints": {"square": [hsv(0.63 + 0.25 * p, 0.9, 1.0) for p in range(4)],
                                    "tip": [hsv(0.55 + 0.25 * p, 0.9, 0.75) for p in range(4)],
                                    "gold": [hsv(0.13, 0.8, 1.0)]},
                         "half_width": ring / 2.0, "outline": 0.25})
    else:
        raise ValueError(f"designs: no floor {name!r} (one of {', '.join(f[0] for f in FLOORS)})")
    settings["pieces"] = [{"uv": [float(x) for x in np.asarray(quad).ravel()], "paint": paint}
                          for quad, paint in pieces]
    return {k: v for k, v in settings.items() if v is not None}


# ----- the 3D frames -----

def horseshoe_ring(top, band, low, segments=12):
    """A horseshoe ring standing on the floor, as a polygon in x and z (the edge from x = 0 to 1): an arc of radius r
    round (1/2, c), wider than it is at the floor, its crown at top, band thick, its two legs down to low; r is 0.6 of
    top (the ring 1.2 times as wide as it is high, a horseshoe's proportions), c = top - r"""
    r = 0.6 * top
    c = top - r
    rings = []
    for radius in (r, r - band):
        foot = math.atan2(low - c, math.sqrt(max(radius * radius - (low - c) ** 2, 0.0)))
        rings.append([[0.5 + radius * math.cos(t), c + radius * math.sin(t)]
                      for t in [foot + (math.pi - 2 * foot) * k / segments for k in range(segments + 1)]])
    outer, inner = rings
    return outer + inner[::-1]


def frames_settings(arch_top=0.35, low=-0.025, first_level=0, rule="square"):
    """The open frames' settings (an object's "parts", parts.py): beams arch_top of a cell high (the posts stand 0.03
    higher), reaching low below the floor, from first_level, on a map of the rule's (a square lattice's, or a
    substitution rule's, RULES). A post's base is flat, on the plane between the faces that meet at its vertex, and
    they fall away under it (up to 33 degrees round a torus's tube: tan(16.3 degrees) of its radius), so it reaches
    low under the floor."""
    post_radius = 0.05
    octagon = [[post_radius * math.cos(math.radians(22.5 + 45 * k)),
                post_radius * math.sin(math.radians(22.5 + 45 * k))] for k in range(8)]
    reach, thickness, depth = 0.02, 0.04, 0.025
    beam = [[-reach, arch_top - thickness], [1.0 + reach, arch_top - thickness], [1.0 + reach, arch_top],
            [-reach, arch_top]]
    # The ring's crown a little into the beam, so the two join
    ring = horseshoe_ring(arch_top - thickness + 0.01, thickness, low)
    arch_colors = [srgb(c) for c in ("#2747a8", "#1b9e93", "#e0a526", "#c4466b")]
    post_colors = [srgb(c) for c in ("#f2e6c9", "#e8dcc0", "#f2e6c9", "#e8dcc0")]
    return {
        "pieces": [
            {"on": "vertex", "polygon": octagon, "axes": "xy", "low": low, "high": arch_top + 0.03, "open": ["-z"],
             "colors": post_colors},
            {"on": "edge", "polygon": beam, "axes": "xz", "low": -depth, "high": depth, "colors": arch_colors},
            {"on": "edge", "polygon": ring, "axes": "xz", "low": -depth, "high": depth, "open": [],
             "colors": arch_colors},
        ],
        "first_level": first_level, "build": 14.0, "margin": 3.0, "fade": 0, "fine": 32.0, "exact_from": 8192.0,
        "window": 1.25, "count": 28, **RULES.get(rule, {}),
    }


# ----- scaling -----

def scaling_config(factor=2):
    """A surface map's scaling config (a Text an object's "scaling_config" names): auto scale to the surface, loops by
    the factor's power nearest 8 (8 = 2^3; a map loops only by whole powers of its factor) between scales 1 / loop and
    1, and the horizon. Tidal lock off: on a mesh's many faces approximating a curve, the direction to its nearest point
    swings at every edge, a bumpy ride (on only for smooth surfaces like the Earth's, or sharp shapes like a cube)."""
    loop = float(factor ** max(1, round(math.log(8.0, factor))))
    return json.dumps({
        "loop": True,
        "loop_factor": loop,
        "auto_scale": True,
        "auto_scale_config": {"auto_scale_mesh": True, "auto_scale_split_axes": False},
        "tidal_lock": False,
        "tidal_lock_config": {"tidal_lock_mesh": True, "tidal_lock_split_axes": False},
        "horizon": True,
        "horizon_config": {"horizon_mesh": True},
        "wrap": False,
        "min_scale": 1.0 / loop,
        "max_scale": 1.0,
    }, indent=4) + "\n"
