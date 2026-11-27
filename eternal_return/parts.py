"""Endless 3D parts on a sphere map's lattice, or a surface map's: pieces on its vertices and along its edges, like
pillars and the arches between them, at every level, from the globe (or the whole mesh) down forever, standing on the
surface: a sphere map's (see ~/blender_tests/sphere/parts), any quad mesh's (surface.py, see ~/blender_tests/surface),
and triangles and odd corners, where a vertex piece stands among 3 to 24 edge pieces, and where the surface folds,
stretched down to reach the floor.

Every level of a sphere map's lattice (sphere.py: each cube face split 2 x 2 at every level) has the same parts, at the
level's own size: a vertex piece on each of its vertices and an edge piece along each of its edges. The lattice is on
the sphere through a conformal chart, and a conformal map is a similarity wherever it's looked at closely: a cell much
smaller than the sphere, away from a cube corner, is a square, only scaled and turned. So each piece is one solid,
placed by a frame worked out from the lattice points it stands on, exactly:
- a vertex piece: on the vertex, standing straight up the surface's normal there, its size the mean distance to the
  vertex's neighbors (4, or 3 at a cube corner, where three faces meet)
- an edge piece: from the edge's first vertex to its second, so both its ends are exactly on them, whatever the chart
  does in between (near a cube corner it bends the lattice's lines, and a coarse cell is a fair share of the globe),
  its size the edge's length, and up its ends' normals: at the height it meets the vertex pieces, each end is on its
  vertex piece's axis (edge_frames(), an affine frame: a rigid one stood on the normals' mean, and where they turn,
  over a surface map's folds, a beam's ends leant off its posts)
Pieces meeting at a vertex meet exactly, and an edge piece that reaches a little into the vertex pieces at its ends (an
arch's ends into the pillars) hides whatever the chart bends between them. A piece's flat base is on the tangent plane
at its vertex, and the sphere curves away under it, so pieces standing on the sphere reach a little below it (the
map's "low").

The levels are drawn two ways, by their cells' size next to the drone's height:

The drone's own levels and those a little bigger (cells up to exact_from times the drone's height) are instanced. Each
is built into one of the map's level objects (made here, children of the map's object): its pieces as points, which a
geometry nodes modifier instances the pieces' meshes on. A level is built around the lattice point below the drone, out
to "build" of its cells, and stands until the point below the drone has moved "margin" cells from there. The coarsest
level (first_level, while it's instanced) has nothing coarser going on past its edge, so it reaches to the window
instead, if the lattice can afford it (lattice.first_radius(): a surface map's whole mesh, once). Loops and recenters
change only the level objects' transforms (a dive loop scales map units, a recenter moves the map's origin), worked out
exactly from the anchor each level was built for. Every tick, each piece sinks into the sphere by how far it is from
the drone (fade()): within fade[0] of its cells it stands in full, and by fade[1] it has sunk out of sight. The drone is
at least its height from every piece, so that one rule also raises each finer level out of the sphere as the drone
comes down to fade[1], then fade[0], times its cells. So nothing pops: levels come and go, and so do pieces at the edge
of a level's window, by rising and sinking, and where they've sunk, the sphere's motif paints the lattice.

Or, with no fade (the "fade" setting 0), nothing sinks: every piece a level object holds stands in full, like the
endless maps' patches (2d_time_endless's cube frames), and a level is drawn from when its cells are a fine-th of the
drone's height, where its pieces are a pixel or two, so finer levels come in too small to see and take shape only
through the view's own perspective as the drone comes down. Low pieces read as the surface's texture from afar. Each
level reaches "build" of its cells around the drone, and where the camera looks, "ahead" of them (the view's reach), and
the coarser levels go on past it. Built "build" (10) all round, a level with cells a 16th of the drone's height reached
only 0.7 heights ahead, its pieces still 100 px wide at its edge, so whatever the drone flew toward got its finer pieces
only once it was nearly over it; built 32 all round, a level had ten times the pieces, and the frame rate halved. So a
level is built "ahead" all round, and shows past "build" only what a view VIEW_ROOM wider than the camera's sees (aim():
its points' "shown", which the node group instances only where set). Turning, that's worked out again in the tick, with
no build (built again on every turn, the worker thread's queue kept the floor's builds waiting 0.7 s); and what's shown
all round is round the point below the drone, which flies on through what's built, so a level is built again only after
"margin" as much a share of "ahead" as of "build". The drone camera's lens is wide (122 degrees across), so a level
still shows a third to half of its disc: in flights over the Penrose sphere's frames, 3,300 to 6,300 pieces a level
against 11,700 built, at the frame rate and worker thread load of 20 all round. Flown fast and low (auto scale shrinking
the drone 0.005x a second on the HUD, gliding in at 20 to 30 degrees), the levels between the biggest and the finest
went unbuilt where the drone looked: their builds waited up to 2.4 s on the worker thread, behind the finest levels',
and were then built round where the drone had
been. So a build still waiting is handed in again round the drone's newest point, in its place in line (background.py),
the coarser levels' builds go first, and a level is built round a point far enough ahead that a build lasts 2.5 times
the flight (settle()): in such a glide, a level between was missing where the view looked in none of the ticks, where
builds round the drone itself missed one in a third of them, and the finest level, 40 of its cells off, in about a
fifth.

The bigger levels, every one up to the globe however deep a dive goes, are the giant levels (Exact), built the way the
endless maps build their big pillars (tiling.py): only their pieces within reach of the camera (the window, "window"
times the drone camera's clip end), into one mesh, each face cut to the window. Instances can't do this: a piece's
place, turn and size are float32, which near the drone is a share of the piece's size off, and a giant pillar's wall
by the drone is a hundred millionth of its size. So in the mesh every corner is from the camera, within the window, and
everything from which it's worked out is exact: the drone's place in a level's cells comes from the anchor's digits, and
a piece's frame from its lattice points, placed in float64 while that's exact to well under the drone's height, and
past that with as many digits as the depth needs (sphere_exact.py). Diving at a pillar's foot, the wall stands, as
steady as the floor, a thousand loops down. Finding the pieces in reach goes up every giant level in turn: the drone's
place in the level's cells, from the digits, rules out most levels at once (the chart is near a similarity over a
cell, except at the cube's corners), the rest are placed in float64, and only the few pieces near the window are worked
out in full. A level the drone is deep inside a vertex piece of has the same vertex's piece around it at every coarser
level the vertex is on, bigger, so those aren't looked at again.

A level goes from instanced to exact once its cells are exact_from times the drone's height, and back below half that,
so a drone hovering at the boundary doesn't build them again and again; both draw the same solids, so the hand-off
doesn't show. A level coming back to instances stays exact until its instances are built.

A piece stays drawn however near the camera comes, and with the camera inside it, like any mesh: the near plane cuts
it open at clip start, and from inside, its walls are seen from within. Only clip end (and the window past it), and
sinking with distance, take pieces away. (Pieces the camera was inside of or within clip start of used to be left out,
so the drone saw out of a pillar it flew into, but the pillar vanished as it went in.)

Settings, the map object's "parts" property (written by the map's build script):
- pieces: a list, each {"on": "vertex" or "edge", "polygon": corners in two axes, "axes": those axes ("xy", "xz" or
  "yz"), "low", "high": how far it's extruded along the third, "open": caps left out ("-z" and so on), "colors": a list
  of linear RGBA, one for each level in turn (level mod its length)}, in the piece's own units: a vertex piece's
  around its vertex at (0, 0, 0), z up and x toward a neighbor, a cell across; an edge piece's from its first vertex
  at (0, 0, 0) to its second at (1, 0, 0), z up, an edge long. See tiling.solid_faces().
- first_level: the coarsest level with parts (level 0's cells are the cube's faces, level 3's an eighth of an edge)
- build, margin, fade: see above (cells); fade 0 for none
- ahead: with no fade, how far a level reaches where the camera looks (cells): built that far all round, once its
  builds are on the worker thread (a flight's first builds, in the tick, reach "build"), and shown that far only in
  view; at most "build" for no further than that
- fine: with no fade, the finest level's cells are this many times smaller than the drone's height
- exact_from: from what size of cells, in drone heights, a level is giant
- window: how far the giant levels' pieces are built, in drone camera clip ends
- count: how many level objects the map may use, at most

What's asked of the lattice the parts stand on goes through an object of its own (lattice_of()):
parts_sphere.SphereLattice for a sphere map, everything above, with its layout's chart (the cube's, or the
icosahedron's: parts_ico.IcoChart); surface.MeshLattice for a surface map, whose faces are flat, so its pieces are
placed from the anchor without a chart, and its giant levels' pieces worked out with Fractions (surface.add_levels()).

Other lattices: the icosahedron sphere's (parts_ico.py: triangles, 5 edges at its 12 corners) and any substitution
rule's (parts_substitution.SubstitutionLattice, one per rule). The Penrose surface maps' is each level's rhombs' corners
and edges, levels phi times finer, so a lattice says how much finer each level is, its ratio; and their frames are on
every other level only (phi^2 apart, less dense): a lattice's level_step, 1 for the rest, says which levels have parts
(drawn()); the levels between are left out of the instanced and giant levels alike. Darb-i Imam's is its girih tiles'
corners and edges, on every level (LAMBDA = 8.5 apart).

Not yet: maps with 3D tiles (the lift: parts stand on the sphere, not the tiles' ground).
"""

import json
import math
import time
import traceback

import numpy as np

try:
    from . import background
    from . import schedule
    from . import sphere
    from . import sphere_exact
    from . import tiling
except ImportError:          # tests import it by itself
    import background
    import schedule
    import sphere
    import sphere_exact
    import tiling

PROPERTY = "parts"
LEVEL_KEY = "parts_level"            # on a level object: the level it holds, absent while free
OWNER_KEY = "parts_of"               # on a level object: its map's object's name
EXACT_KEY = "parts_exact_of"         # on the giant levels' object: its map's object's name
NODE_GROUP = "_ER_Parts"
VERSION_KEY = "er_version"
VERSION = 2                          # 2: instanced by a "frame" matrix (was "rotation" and "scale")
MODIFIER = "Parts"
MATERIAL = "_ER_Parts"
KINDS = ("vertex", "edge")

DEFAULTS = {"first_level": 3, "build": 11.0, "ahead": 32.0, "margin": 3.0, "fade": [4.0, 8.0], "fine": 32.0,
            "exact_from": 8192.0, "window": 1.25,
            "count": 24}
# How long a tick may spend building levels, seconds; a level not built in time waits for the next tick, and until it's
# built, its pieces aren't there (it's built before it has risen, unless the drone outruns it)
BUILD_BUDGET = 0.004
# A level built this many dive or climb loops ago is built again: its pieces were placed in map units 2^LOOPS_STALE
# times bigger or smaller, and float32 keeps a fixed share of their size
LOOPS_STALE = 9
# Fade steps: a piece's sinking is written only when it changes by one of these, so a still drone writes nothing
FADE_STEPS = 64
# A level's sinking is worked out again once the camera has moved this share of its cells: a step of the fade is 1/16
# of a cell (fade[1] - fade[0] = 4 cells over 64 steps)
FADE_MOVE = 1.0 / 32.0
# A sunk piece goes this far below its height, so its top isn't drawn in the same place as the sphere
SUNK = 1.05
# Giant levels: a piece is worked out with more digits than float64's once float64 would put it this share of the
# drone's height off (float64 places a point to about FLOAT_ERROR of its distance from the anchor)
EXACT_SHARE = 1e-7
FLOAT_ERROR = 1e-14
# Builds on the worker thread land a while after they're asked for, and the drone flies on meanwhile; and it looks
# the way it flies. So a level is measured from a point ahead of where the drone will be when a build asked for now
# lands (the latency: from asking to writing, smoothed, the map's schedule.Motion's, as is the velocity), ahead along
# its flight by a lead: "margin" cells, or as far as it flies in LEAD_MOST s if that's less. It's built
# again once that point is "margin" cells from its center, and built round a point a lead further on. Flying
# straight, a build then lands with "build" plus twice the lead cells of it ahead of the drone and less the lead
# behind, and is built again with the drone at its center: twice the flight a build, and twice the reach ahead that
# the view sees, of the same pieces. Hovering, or with the view looking another way than the flight (drifting back,
# or straight down), there's no lead, and a level is built round the drone. Built round
# where the drone was when asked, the finest levels' cells passed by 15 to 40 a second at a cruise, and the drone was
# past the edge of a level built 10 cells out (in 55% of the ticks past the margin).
LEAD_MOST = 1.0
# A rising level (see settle()) is handed over before its instances are built once its cells are this many times
# smaller than exact_from: its pieces in the window grow as the square of that, each one worked out on the main thread
# at every window build (climbing at 0.005x/s, a level kept giant for 2 s had 360 pieces, ticks of 3 s)
RISING_LATE = 4.0
# Diving, a level wanted within EARLY_TIME s (and at most EARLY_MOST of them, the next finer ones) is built before it's
# wanted, round where the drone will be then, and kept hidden until it is (settle()). In a fast dive (7.6 halvings a
# second) a new level came every 0.18 s, went from the finest wanted to the fourth finest in 0.55 s, and
# its first build, asked for once it was wanted, landed 0.1 to 0.3 s later: the finest three or four levels were
# missing in nine ticks of ten
EARLY_TIME = 0.4
EARLY_MOST = 2
# The view's reach ("ahead"): what a level shows past "build" is what a view this much wider than the camera's sees,
# on every side (at most VIEW_WIDEST off its axis), worked out again once the view has turned VIEW_TURN from the one it
# was worked out for (its axis, or its up, as in a roll), or the camera has moved as far as turns the nearest piece
# shown only for being in view that much (aim())
VIEW_ROOM = math.radians(10.0)
VIEW_TURN = math.radians(5.0)
VIEW_WIDEST = math.radians(85.0)


def is_parts(obj):
    return (obj is not None and hasattr(obj, "get") and obj.get(PROPERTY) is not None
            and (sphere.is_sphere(obj) or _surface().is_surface(obj)))


def _surface():
    # Imported when first needed: surface.py builds on this module's classes
    try:
        from . import surface
    except ImportError:
        import surface
    return surface


def lattice_of(obj):
    """The lattice a map's parts stand on: a sphere map's (parts_sphere.SphereLattice, with the cube's chart or the
    icosahedron's, parts_ico.IcoChart), a surface map's (surface.MeshLattice) or a substitution surface map's
    (parts_substitution.SubstitutionLattice, one per rule)"""
    if sphere.is_sphere(obj):
        if sphere.world_of(obj).layout == "ico":
            return _module("parts_ico").LATTICE
        return _module("parts_sphere").SPHERE
    surface = _surface()
    if surface.is_substitution(obj):
        return _module("parts_substitution").lattice_of(obj)
    return surface.lattice_of(obj)


def _module(name):
    """One of the lattices' modules, imported when first needed (they build on this module's classes)"""
    import importlib
    package = __name__.rpartition(".")[0]
    return importlib.import_module(f"{package}.{name}" if package else name)


def setting(settings, name):
    value = settings.get(name)
    return DEFAULTS[name] if value is None else value


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ----- the lattice near a point: its vertices and edges, from the cube's -----

# ----- the pieces' meshes, and the node group that instances them -----

def references(obj, settings):
    """The collection of the pieces' meshes, one object for each kind of piece (vertex, edge) and level color in turn
    (index: phase * 2 + kind), made again when the pieces change: (collection, phases, their Shape)"""
    import bpy
    pieces = list(settings["pieces"])
    phases = max([len(p.get("colors", [])) for p in pieces] + [1])
    stamp = json.dumps([{k: v for k, v in dict(p).items()} for p in pieces], sort_keys=True, default=_plain)
    name = obj.name + ".parts"
    collection = bpy.data.collections.get(name)
    shape = _shapes.get(stamp)
    if shape is None:
        if len(_shapes) > 16:
            _shapes.clear()
        shape = _shapes[stamp] = Shape(pieces)
    if collection is not None and collection.get("stamp") == stamp and len(collection.objects) == 2 * phases:
        for ref in collection.objects:
            no_shadow(ref)
        return collection, phases, shape
    if collection is None:
        collection = bpy.data.collections.new(name)
    for child in list(collection.objects):
        mesh = child.data
        bpy.data.objects.remove(child, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    material = parts_material()
    for phase in range(phases):
        for k, kind in enumerate(KINDS):
            corners, sizes, colors = [np.zeros((0, 3))], [np.zeros(0, dtype=np.int64)], [np.zeros((0, 4))]
            for piece in pieces:
                if piece.get("on", "vertex") != kind:
                    continue
                c, s = tiling.solid_faces([[float(u) for u in corner] for corner in piece["polygon"]],
                                          piece.get("axes", "xy"), float(piece.get("low", 0.0)),
                                          float(piece.get("high", 1.0)), set(piece.get("open", [])))
                corners.append(c)
                sizes.append(s)
                colors.append(np.tile(piece_color(piece, phase), (len(s), 1)))
            # Named so the collection's children come in index order (Collection Info sorts them by name)
            ref_name = f"{obj.name}.parts.ref.{phase * 2 + k:03d}"
            ref = bpy.data.objects.new(ref_name, bpy.data.meshes.new(ref_name))
            collection.objects.link(ref)
            tiling.write_mesh(ref, np.concatenate(corners), np.concatenate(sizes), np.concatenate(colors))
            ref.data.materials.append(material)
            no_shadow(ref)
    collection["stamp"] = stamp
    return collection, phases, shape


def no_shadow(obj):
    """Keep obj (a piece's reference mesh, whose instances are the parts, or a mesh rebuilt as the drone flies) out of
    Rendered mode's shadow maps. EEVEE draws its sun's shadow maps again wherever a caster's geometry changes, and the
    parts change all the time: each level is built again as the drone moves on, and a draw after one took 1.3 s (an M1
    Air at 2752 x 1582, 1,800 instances; 55 ms casting none). They're still lit, and Solid mode has no shadows."""
    if obj.visible_shadow:
        obj.visible_shadow = False


def piece_color(piece, phase):
    """A piece's linear RGBA at a level of phase phase"""
    colors = list(piece.get("colors", [[0.8, 0.8, 0.8, 1.0]]))
    return [float(v) for v in colors[phase % len(colors)]]


def face_colors(settings, faces, phase):
    """Each face's linear RGBA at a level of phase phase (faces: window_faces()'s, each with its piece's index), a
    piece's color read once (a post is dozens of faces of one piece, and read for every face, a climb's giant windows
    read them a million times)"""
    by_piece = {}
    colors = []
    for _, index in faces:
        color = by_piece.get(index)
        if color is None:
            color = by_piece[index] = piece_color(settings["pieces"][index], phase)
        colors.append(color)
    return colors


def _plain(value):
    """IDProperty values (lists, groups) as plain JSON"""
    if hasattr(value, "to_list"):
        return value.to_list()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def square_root(x):
    return x.sqrt() if hasattr(x, "sqrt") else math.sqrt(x)


class Prism:
    """A piece's solid in its own units (see tiling.solid_faces()): its polygon in two axes, extruded along the third
    from low to high"""

    def __init__(self, piece):
        axes = piece.get("axes", "xy")
        self.a, self.b = "xyz".index(axes[0]), "xyz".index(axes[1])
        self.k = 3 - self.a - self.b
        self.polygon = np.array([[float(u) for u in corner] for corner in piece["polygon"]])
        self.low, self.high = float(piece.get("low", 0.0)), float(piece.get("high", 1.0))
        self.open = set(piece.get("open", []))
        # Its outline's edges, from each corner to the next
        self.edges = np.roll(self.polygon, -1, axis=0) - self.polygon
        self.edge_squares = np.sum(self.edges * self.edges, axis=1)
        self._typed = {}

    def corners(self):
        out = np.zeros((2 * len(self.polygon), 3))
        out[:, self.a], out[:, self.b] = np.tile(self.polygon[:, 0], 2), np.tile(self.polygon[:, 1], 2)
        out[:, self.k] = np.repeat([self.low, self.high], len(self.polygon))
        return out

    def distance(self, points):
        """How far points (m, 3, its own units) are outside it, negative inside: their distance from its outline in
        the polygon's plane and from its extent along the third axis, the bigger of the two. Never more than their true
        distance, and like it, it changes no faster than they move."""
        across = self.across(points[:, [self.a, self.b]])
        w = points[:, self.k]
        return np.maximum(across, np.maximum(self.low - w, w - self.high))

    def across(self, points):
        """The distance of 2D points (m, 2) from its outline, negative inside it"""
        a, ab = self.polygon, self.edges
        ap = points[:, None, :] - a[None, :, :]
        t = np.clip(np.sum(ap * ab, axis=2) / self.edge_squares, 0.0, 1.0)
        distance = np.min(np.linalg.norm(ap - t[..., None] * ab, axis=2), axis=1)
        # Inside: an odd number of edges crossing the line through the point, to its right
        y = points[:, 1:2]
        crosses = (a[:, 1] > y) != (a[:, 1] + ab[:, 1] > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = a[:, 0] + (y - a[:, 1]) * ab[:, 0] / ab[:, 1]
        inside = np.sum(crosses & (points[:, 0:1] < x), axis=1) % 2 == 1
        return np.where(inside, -distance, distance)

    def typed(self, kind):
        """Its polygon, low and high as numbers of a type (float, or Decimal at the context's precision: exact)"""
        values = self._typed.get(kind)
        if values is None:
            values = self._typed[kind] = ([(kind(x), kind(y)) for x, y in self.polygon.tolist()], kind(self.low),
                                          kind(self.high))
        return values

    def distance_one(self, p):
        """distance() of one point (three floats, or Decimals), worked out in its own type: the same as distance(), and
        with Decimals, as exact as their precision"""
        polygon, low, high = self.typed(type(p[0]))
        x, y, w = p[self.a], p[self.b], p[self.k]
        best = None
        inside = False
        for (ax, ay), (bx, by) in zip(polygon, polygon[1:] + polygon[:1]):
            ex, ey = bx - ax, by - ay
            px, py = x - ax, y - ay
            t = min(max((px * ex + py * ey) / (ex * ex + ey * ey), 0), 1)
            dx, dy = px - t * ex, py - t * ey
            square = dx * dx + dy * dy
            if best is None or square < best:
                best = square
            if (ay > y) != (by > y) and x < ax + (y - ay) * ex / ey:
                inside = not inside
        across = square_root(best)
        return max(-across if inside else across, low - w, w - high)

    def faces(self):
        """Its faces, for a mesh cut to a window (Exact): tiling.solid_faces()'s, each as a (m, 3) array of its
        corners, its sides as they are and its caps in triangles (convex, so cutting one keeps it in one piece)"""
        polygon = self.polygon.tolist()
        corners, sizes = tiling.solid_faces(polygon, "xyz"[self.a] + "xyz"[self.b], self.low, self.high, self.open)
        starts = np.concatenate([[0], np.cumsum(sizes)])
        out = []
        # Its sides first, one for each of the polygon's edges, then its caps
        for number, (size, start) in enumerate(zip(sizes, starts)):
            face = corners[start:start + size]
            if number < len(polygon):
                out.append(face)
            else:
                out += [face[list(t)] for t in ear_triangles(face[:, [self.a, self.b]])]
        return out


def ear_triangles(polygon):
    """A simple polygon (m, 2) in triangles, each as three of its corners' indices, wound as it is"""
    points = [tuple(p) for p in np.asarray(polygon, dtype=float).tolist()]
    area = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]))
    sign = 1.0 if area > 0 else -1.0
    left = list(range(len(points)))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    triangles = []
    while len(left) > 3:
        for k in range(len(left)):
            i0, i1, i2 = left[k - 1], left[k], left[(k + 1) % len(left)]
            p0, p1, p2 = points[i0], points[i1], points[i2]
            if cross(p0, p1, p2) * sign <= 0.0:
                continue
            if any(cross(p0, p1, points[j]) * sign > 0.0 and cross(p1, p2, points[j]) * sign > 0.0
                   and cross(p2, p0, points[j]) * sign > 0.0 for j in left if j not in (i0, i1, i2)):
                continue
            triangles.append((i0, i1, i2))
            del left[k]
            break
        else:
            break                    # degenerate: what's left goes as a fan
    triangles += [(left[0], left[k], left[k + 1]) for k in range(1, len(left) - 1)]
    return triangles


class Shape:
    """The pieces' solids by kind (vertex, edge), in their own units: each kind's Prisms, its height (the highest any
    of its pieces reach, for sinking) and its reach (how far any of its corners is from where its distance from the
    camera is measured: a vertex piece's vertex, an edge piece's middle). For the giant levels, also each kind's faces
    (with the index of the piece each is of, for its color), how far it reaches in plan (a vertex piece from its vertex,
    an edge piece across its edge), its lowest point, and a vertex piece's core: the biggest upright cylinder around
    its vertex inside it (radius, bottom, top), or None; and how far past its edge's ends an edge piece reaches. And
    meet: the height at which edge pieces meet the vertex pieces at their ends (the mean height of their corners
    within a vertex piece's plan reach of an end: a beam's, not an arch's middle), where edge_frames() stands their
    ends on the vertex pieces' axes."""

    def __init__(self, pieces):
        self.prisms, self.heights, self.reach = [], [], []
        self.faces, self.plan, self.floor, self.core = [], [], [], []
        self.past = 0.0
        self.meet = 0.0
        for kind, measured_from in zip(KINDS, ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0))):
            indices = [i for i, p in enumerate(pieces) if p.get("on", "vertex") == kind]
            prisms = [Prism(pieces[i]) for i in indices]
            corners = np.concatenate([p.corners() for p in prisms]) if prisms else np.zeros((1, 3))
            self.prisms.append(prisms)
            self.heights.append(float(corners[:, 2].max()) if prisms else 0.0)
            self.reach.append(float(np.linalg.norm(corners - measured_from, axis=1).max()))
            self.faces.append([(face, i) for i, prism in zip(indices, prisms) for face in prism.faces()])
            across = np.hypot(corners[:, 0], corners[:, 1]) if kind == "vertex" else np.abs(corners[:, 1])
            self.plan.append(float(across.max()))
            self.floor.append(float(corners[:, 2].min()))
            if kind == "edge" and prisms:
                self.past = float(np.maximum(-corners[:, 0], corners[:, 0] - 1.0).max())
                reach_v = self.plan[0] if self.prisms[0] else 0.0
                ends = corners[(corners[:, 0] <= reach_v) | (corners[:, 0] >= 1.0 - reach_v)]
                self.meet = float(ends[:, 2].mean()) if len(ends) else 0.0
            core = None
            for prism in prisms if kind == "vertex" else ():
                if prism.k != 2:
                    continue
                inside = -float(prism.across(np.zeros((1, 2)))[0])
                if inside > 0.0 and (core is None or inside > core[0]):
                    core = (inside, prism.low, prism.high)
            self.core.append(core)
        self.reach = np.array(self.reach)


_shapes = {}         # pieces' stamp -> Shape


def parts_material():
    """The pieces' material: their "Col" colors (which Solid mode shows by attribute), lit, in Rendered mode"""
    import bpy
    material = bpy.data.materials.get(MATERIAL)
    if material is not None:
        return material
    material = bpy.data.materials.new(MATERIAL)
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    shader = nodes.get("Principled BSDF")
    attribute = nodes.new("ShaderNodeAttribute")
    attribute.attribute_name = "Col"
    attribute.location = (-300, 300)
    if shader is not None:
        links.new(attribute.outputs["Color"], shader.inputs["Base Color"])
        shader.inputs["Roughness"].default_value = 0.6
    return material


def node_group():
    """The level objects' node group: the pieces' meshes (the collection input's children) instanced on the object's
    points, by their "ref" (which) and "shown" attributes, each moved to its point and taken there by its "frame" (a
    4 x 4 matrix attribute, its linear part: a piece's axes times its size, which for an edge piece needn't be a
    rotation, see edge_frames())"""
    import bpy
    group = bpy.data.node_groups.get(NODE_GROUP)
    if group is not None and group.get(VERSION_KEY) != VERSION:
        bpy.data.node_groups.remove(group)
        group = None
    if group is not None:
        return group
    group = bpy.data.node_groups.new(NODE_GROUP, 'GeometryNodeTree')
    group.is_modifier = True
    group[VERSION_KEY] = VERSION
    interface = group.interface
    interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    interface.new_socket("Pieces", in_out='INPUT', socket_type='NodeSocketCollection').description = \
        "The pieces' meshes, in order: each level color's vertex piece, then its edge piece"
    nodes, links = group.nodes, group.links
    group_input = nodes.new('NodeGroupInput')
    group_output = nodes.new('NodeGroupOutput')
    collection = nodes.new('GeometryNodeCollectionInfo')
    collection.transform_space = 'ORIGINAL'
    collection.inputs["Separate Children"].default_value = True
    collection.inputs["Reset Children"].default_value = True
    links.new(group_input.outputs["Pieces"], collection.inputs["Collection"])

    def attribute(name, data_type):
        node = nodes.new('GeometryNodeInputNamedAttribute')
        node.data_type = data_type
        node.inputs["Name"].default_value = name
        return node.outputs["Attribute"]

    instance = nodes.new('GeometryNodeInstanceOnPoints')
    links.new(group_input.outputs["Geometry"], instance.inputs["Points"])
    links.new(attribute("shown", 'BOOLEAN'), instance.inputs["Selection"])
    links.new(collection.outputs["Instances"], instance.inputs["Instance"])
    instance.inputs["Pick Instance"].default_value = True
    links.new(attribute("ref", 'INT'), instance.inputs["Instance Index"])
    # Each instance's transform: to its point (the instance's position, which fade() sinks), then its frame
    position = nodes.new('GeometryNodeInputPosition')
    to_point = nodes.new('FunctionNodeCombineTransform')
    links.new(position.outputs["Position"], to_point.inputs["Translation"])
    multiply = nodes.new('FunctionNodeMatrixMultiply')
    links.new(to_point.outputs["Transform"], multiply.inputs[0])
    links.new(attribute("frame", 'FLOAT4X4'), multiply.inputs[1])
    transform = nodes.new('GeometryNodeSetInstanceTransform')
    links.new(instance.outputs["Instances"], transform.inputs["Instances"])
    links.new(multiply.outputs["Matrix"], transform.inputs["Transform"])
    links.new(transform.outputs["Instances"], group_output.inputs["Geometry"])
    for x, node in enumerate((group_input, collection, instance, transform, group_output)):
        node.location = (x * 250, 0)
    for x, node in enumerate((position, to_point, multiply)):
        node.location = (500 + x * 200, -250)
    return group


# ----- the level objects -----

def map_child(obj, name, key):
    """A new mesh object of the map's (a level's, the floor's): a child of the map's object, in its collections,
    marked as the map's (key), and clicking it selects the map (scaling_target)"""
    import bpy
    child = bpy.data.objects.new(name, bpy.data.meshes.new(name))
    for c in obj.users_collection:
        c.objects.link(child)
    child.parent = obj
    child[key] = obj.name
    child["scaling_target"] = obj
    return child


def pool(obj):
    """The map's level objects, made the first time: children of the map's object (map_child()), points instanced by
    the node group, unselectable"""
    settings = obj[PROPERTY]
    wanted = int(setting(settings, "count"))
    objects = sorted((o for o in obj.children if o.get(OWNER_KEY) == obj.name), key=lambda o: o.name)
    group = node_group()
    # A file saved with an older node group (made again in node_group(), which leaves its modifiers without one)
    for level_obj in objects:
        modifier = level_obj.modifiers.get(MODIFIER)
        if modifier is not None and modifier.node_group is not group:
            collection, _phases, _shape = references(obj, settings)
            modifier.node_group = group
            socket = group.interface.items_tree["Pieces"]
            getattr(modifier.properties.inputs, socket.identifier).value = collection
    if len(objects) >= wanted:
        return objects
    collection, _phases, _shape = references(obj, settings)
    for k in range(len(objects), wanted):
        level_obj = map_child(obj, f"{obj.name}.parts.{k:02d}", OWNER_KEY)
        level_obj.hide_select = True
        modifier = level_obj.modifiers.new(MODIFIER, 'NODES')
        modifier.node_group = group
        socket = group.interface.items_tree["Pieces"]
        getattr(modifier.properties.inputs, socket.identifier).value = collection
        objects.append(level_obj)
    return objects


def exact_object(obj):
    """The map's giant levels' object, made the first time: a plain mesh, a child of the map's object (map_child()),
    unselectable"""
    found = next((o for o in obj.children if o.get(EXACT_KEY) == obj.name), None)
    if found is not None:
        no_shadow(found)
        return found
    exact_obj = map_child(obj, f"{obj.name}.parts.exact", EXACT_KEY)
    exact_obj.hide_select = True
    exact_obj.data.materials.append(parts_material())
    no_shadow(exact_obj)
    return exact_obj


class Built:
    """A level as built into its object (kept in memory, by object name): the world it was built for (n, anchor), its
    center (the lattice point below the drone then: its place, up and cells), and each piece's place, up, size and
    height, all in map units then (the level's own units: the level object's transform takes them to today's map),
    with what was last written of its sinking"""

    def __init__(self, level, world_text, n, center, points, ups, sizes, tops, distance_points, frames, kinds):
        self.level, self.world_text, self.n = level, world_text, n
        self.center_point, self.center_up, self.cell = center
        self.points, self.ups, self.sizes, self.tops, self.distance_points = (points, ups, sizes, tops,
                                                                              distance_points)
        self.frames, self.kinds = frames, kinds        # each piece's axes (columns) and kind (index into KINDS)
        self.current = points                          # each piece's place as last written (sunk)
        self.written = None
        self.transform = None        # (world text, offset, scale) of the last transform worked out
        self.placed = world_text     # the world text the level object's transform is for (built at the identity)
        self.faded = None            # the camera the sinking was last worked out for, in its own units
        self.radius = None           # the cells it was built out to (infinite: the whole level, which never moves)
        self.aimed = None            # the camera and its axes (View) what's shown was last worked out for, if aimed

    def away(self, point):
        """How far a point (the level's own units: the camera, or the point of the surface below it) is from the level's
        center along the surface, in its cells"""
        v = point - self.center_point
        return float(np.linalg.norm(v - (v @ self.center_up) * self.center_up)) / self.cell


def piece_distances(shape, frames, places, sizes, kinds, camera):
    """How far a camera is outside each of some pieces (frames (n, 3, 3), axes as columns, places, sizes and kinds),
    negative inside, in their own units: the camera in each one's own units, against its kind's solids (an edge
    piece's frame needn't be a rotation: through its inverse)"""
    # A frame that's still singular (a degenerate mesh's) is never near: left at the identity, its distance infinite
    singular = ~(np.abs(np.linalg.det(frames)) > 1e-12)
    if singular.any():
        frames = frames.copy()
        frames[singular] = np.eye(3)
    local = np.einsum("nij,nj->ni", np.linalg.inv(frames), camera - places) / sizes[:, None]
    distance = np.full(len(sizes), math.inf)
    for kind, prisms in enumerate(shape.prisms):
        of = kinds == kind
        for prism in prisms if of.any() else ():
            distance[of] = np.minimum(distance[of], prism.distance(local[of]))
    distance[singular] = math.inf
    return distance


_built = {}          # level object name -> Built


def window_rows(shape, pieces, local, rounding, scale, window):
    """Which of a level's pieces laid out round a vertex (in their own units, scale map units each) are within window
    (map units) of the camera, there local, float64's rounding of their place rounding: (rows, each piece's distance
    outside it in its own units, negative inside, and how far float64's rounding could take that). Every lattice's
    giant levels check the pieces near the drone this way before working the ones in reach out exactly."""
    distances = piece_distances(shape, pieces.frames, pieces.places, pieces.sizes, pieces.kinds, local) * pieces.sizes
    tolerance = (FLOAT_ERROR * (np.linalg.norm(pieces.places, axis=1) + pieces.sizes + np.linalg.norm(local))
                 + rounding)
    return np.flatnonzero(np.abs(distances) <= window / scale + tolerance), distances, tolerance


def upright_frames(toward, up):
    """Vertex pieces' frames: z up their vertex's normal (up, unit), just as it is, and x toward toward (a neighbor)
    made perpendicular to it, their axes as columns, (n, 3, 3). (Keeping x along toward instead, with z made
    perpendicular to it, leant each piece toward its first neighbor by half the surface's turn to it: on the torus's
    level 0, 16 degrees.)"""
    x = perpendicular(toward, up)
    return np.stack([x, np.cross(up, x), up], axis=2)


def perpendicular(v, axis):
    """v (n, 3) made perpendicular to unit axis (n, 3) and unit; where v is along axis, the coordinate axis furthest
    from it, made so (a vertex piece whose neighbor is straight up its normal, an edge piece along its up)"""
    out = v - np.sum(v * axis, axis=1, keepdims=True) * axis
    length = np.linalg.norm(out, axis=1)
    along = ~(length > 1e-9 * np.linalg.norm(v, axis=1))
    if along.any():
        other = np.eye(3)[np.argmin(np.abs(axis[along]), axis=1)]
        out[along] = other - np.sum(other * axis[along], axis=1, keepdims=True) * axis[along]
        length[along] = np.linalg.norm(out[along], axis=1)
    return out / length[:, None]


def edge_frames(start, end, up_start, up_end, meet, scales):
    """Edge pieces' frames (axes as columns, (n, 3, 3)), places and lengths, from their ends' places and normals
    (each (n, 3), the normals unit) and their ends' vertex pieces' sizes over the edge's length (scales, (n, 2)): the
    affine map from an edge piece's own units (x from 0 at its first vertex to 1 at its second, in edge lengths; y
    across and z up, in its ends' vertex pieces' sizes, s) that, at the height meet (Shape.meet), stands each of its
    ends on its vertex piece's axis at that piece's own scale, and puts its foot's middle on the edge's middle. With
    each end's normal times its scale, a = n s / L:
        x = (end - start) / L + meet (a_end - a_start), z = (a_start + a_end) / 2, y across, (s_start + s_end) / 2L long,
        place = start - meet L (a_end - a_start) / 2
    On a flat face with even cells it's the rigid frame along the edge, z up. Where the normals turn (over a fold, or
    on a sphere's lattice), a rigid frame stood on their mean, and its ends leant off the vertex pieces by meet times
    half the turn: around the torus's tube, by up to 0.1 of a cell at its beams, which are thinner than that. And an
    edge piece scaled by its own length stood up to twice as high as its vertex pieces where a corner's edges differ
    in length (Suzanne's, by up to 3 times): its beam passed over the posts. This one leans and scales its ends with the
    vertex pieces (x's extra stretch: they spread over a bulge, close in a saddle), at the price of a shear, too small
    to see (its foot's ends slide along the floor by meet times the turn over 4)."""
    along = end - start
    lengths = np.linalg.norm(along, axis=1)
    folded = ~(np.linalg.norm(up_start + up_end, axis=1) > 2.0 * FOLDED_BACK)
    a_start, a_end = up_start * scales[:, 0:1], up_end * scales[:, 1:2]
    turn = a_end - a_start
    z = 0.5 * (a_start + a_end)
    # Where the surface folds back on itself between the ends (their normals nearly opposite), rigid on the first one's
    if folded.any():
        turn[folded] = 0.0
        z[folded] = perpendicular(up_start[folded], along[folded] / lengths[folded, None]) * scales[folded, 0:1]
    x = along / lengths[:, None] + meet * turn
    y = np.cross(z, x)
    length = np.linalg.norm(y, axis=1)
    flat = ~(length > 1e-9 * np.linalg.norm(z, axis=1) * np.linalg.norm(x, axis=1))
    if flat.any():
        # Up along the edge: any way across it
        y[flat] = perpendicular(z[flat], x[flat] / np.linalg.norm(x[flat], axis=1, keepdims=True))
        length[flat] = 1.0
    y *= (0.5 * (scales[:, 0] + scales[:, 1]) / length)[:, None]
    places = start - (0.5 * meet * lengths)[:, None] * turn
    return np.stack([x, y, z], axis=2), places, lengths


# An edge piece's ends' normals are folded back (edge_frames()) when their mean is shorter than this (174 degrees apart)
FOLDED_BACK = 0.05
# A vertex piece's foot reaches this much further down than the floor falls away under its edge (foot_stretch())
FOOT_ROOM = 1.1


def foot_stretch(shape, slope):
    """How much a vertex piece is stretched along its up, about its top, for its foot to reach the floor under it: a
    piece stands on its vertex's tangent plane, and where the surface folds at the vertex (a surface map's mesh edges
    and corners: Suzanne's fold by up to 150 degrees) the floor falls away under its edge by its plan reach times slope
    (the tangent of the steepest face's tilt from the vertex's normal), past the piece's own foot (Shape.floor), and it
    stood in the air. 1 where its foot reaches already (any flat face). A float, or for an array of slopes, an array."""
    top, low = shape.heights[0], shape.floor[0]
    foot = -shape.plan[0] * np.asarray(slope, dtype=float) * FOOT_ROOM
    stretch = np.where((foot < low) & (top > low), (top - foot) / max(top - low, 1e-300), 1.0)
    return stretch if stretch.ndim else float(stretch)


class LaidOut:
    """A level's pieces as laid out on its lattice points near a vertex (lay_out()): each a place (its vertex, or by its
    edge's first vertex: edge_frames()), up, size, frame (axes as columns), kind, height and the point its distance is
    measured from, and the lattice rows it stands on (piece_rows())"""

    def lay_out(self, inner_mask, neighbors, edges, middle, points, normals, settings, shape, slopes=None):
        """The pieces on lattice points (places points, normals normals, the vertex they're around at row middle):
        a vertex piece on each inner point (inner_mask), its size the mean distance to its neighbors (neighbors: rows,
        -1 for none, the first one its x axis), stretched down to the floor where that falls away (slopes: each
        point's, see foot_stretch(); none on a sphere), an edge piece along each of edges (pairs of rows)"""
        inner = np.flatnonzero(inner_mask)
        # A vertex's size: the mean distance to its neighbors
        nb = neighbors[inner]
        valid = nb >= 0
        gaps = np.linalg.norm(points[np.where(valid, nb, 0)] - points[inner][:, None, :], axis=2)
        sizes_v = np.sum(np.where(valid, gaps, 0.0), axis=1) / np.maximum(valid.sum(axis=1), 1)
        toward = points[nb[:, 0]] - points[inner]
        frames_v = upright_frames(toward, normals[inner])
        places_v = points[inner]
        if slopes is not None and _has(settings, "vertex"):
            stretch = foot_stretch(shape, slopes[inner])
            if np.any(stretch != 1.0):
                # Along z about the top: z' = top + (z - top) stretch
                frames_v = frames_v.copy()
                frames_v[:, :, 2] *= stretch[:, None]
                places_v = places_v + ((1.0 - stretch) * shape.heights[0] * sizes_v)[:, None] * normals[inner]
        at = int(np.flatnonzero(inner == middle)[0])
        self.center_frame = (points[inner][at], normals[inner][at], float(sizes_v[at]))
        i, j = edges[:, 0], edges[:, 1]
        # Each edge's ends' vertex sizes over its length (edges join inner points, which have sizes)
        size_of = np.zeros(len(points))
        size_of[inner] = sizes_v
        lengths = np.linalg.norm(points[j] - points[i], axis=1)
        self.edge_scales = np.stack([size_of[i], size_of[j]], axis=1) / lengths[:, None]
        frames_e, places_e, lengths = edge_frames(points[i], points[j], normals[i], normals[j], shape.meet,
                                                  self.edge_scales)
        ups_e = normals[i] + normals[j]
        ups_e /= np.linalg.norm(ups_e, axis=1, keepdims=True)
        blocks = []
        if _has(settings, "vertex"):
            blocks.append((places_v, normals[inner], sizes_v, frames_v, 0, shape.heights[0], points[inner]))
        if _has(settings, "edge"):
            # Its height in its sizes: z is in its ends' vertex sizes, the size in edge lengths
            tops_e = shape.heights[1] * 0.5 * (self.edge_scales[:, 0] + self.edge_scales[:, 1])
            blocks.append((places_e, ups_e, lengths, frames_e, 1, tops_e, 0.5 * (points[i] + points[j])))
        # The lattice rows each piece stands on, worked out when asked for (piece_rows())
        self.vertex_count = len(inner) if _has(settings, "vertex") else 0
        self.inner_rows, self.inner_neighbors, self.edge_rows = inner, nb, edges
        self.places = np.concatenate([b[0] for b in blocks])
        self.ups = np.concatenate([b[1] for b in blocks])
        self.sizes = np.concatenate([b[2] for b in blocks])
        self.frames = np.concatenate([b[3] for b in blocks])
        self.kinds = np.concatenate([np.full(len(b[0]), b[4], dtype=np.int32) for b in blocks])
        self.tops = np.concatenate([np.broadcast_to(b[5], len(b[0])) for b in blocks])
        self.distance_points = np.concatenate([b[6] for b in blocks])
        self.points, self.normals = points, normals
        self.middle = middle
        # The vertex piece on the middle (the vertex they're around), if there are vertex pieces
        self.middle_piece = at if self.vertex_count else None

    def piece_rows(self, row):
        """The lattice rows piece row stands on: a vertex piece's vertex then its neighbors, an edge piece's two
        vertices"""
        if row < self.vertex_count:
            return [int(self.inner_rows[row])] + [int(u) for u in self.inner_neighbors[row] if u >= 0]
        return [int(v) for v in self.edge_rows[row - self.vertex_count]]


def build_level(lattice, obj, level_obj, world, level, center, settings, phases, shape, placer, radius=None):
    """Build a level into its object, around center (a lattice point of the level): its vertices and edges out to
    radius cells ("build" unless given; infinite for the whole level), placed exactly (lattice.pieces(): Pieces on a
    sphere), each a point with its piece's frame"""
    if radius is None:
        radius = float(setting(settings, "build"))
    pieces = lattice.pieces(world, level, center, radius, settings, shape)
    write_level(level_obj, world, level, pieces, pieces.origin(placer), level_phase(level, phases, level_step(lattice)),
                radius)


def write_level(level_obj, world, level, pieces, origin, phase, radius):
    """A level's pieces (built for world, out to radius, the vertex they're around at origin in its map) written into
    its object as points with their pieces' frames: build_level()'s last step, or a worker thread's build taken in a
    later tick (settle()), its pieces in the colors of phase (level_phase())"""
    scale = pieces.scale(world.rho)
    origin = np.array([float(v) for v in origin])
    P = origin + scale * pieces.places
    refs = phase * 2 + pieces.kinds
    mesh = level_obj.data
    mesh.clear_geometry()
    mesh.vertices.add(len(P))
    mesh.attributes["position"].data.foreach_set("vector", P.astype(np.float32).ravel())
    for name, kind in (("frame", 'FLOAT4X4'), ("ref", 'INT'), ("shown", 'BOOLEAN')):
        if mesh.attributes.get(name) is None:
            mesh.attributes.new(name, kind, 'POINT')
    sizes = scale * pieces.sizes
    # The frames times their sizes, as 4 x 4 matrices, stored column by column
    matrices = np.zeros((len(P), 4, 4))
    matrices[:, :3, :3] = pieces.frames * sizes[:, None, None]
    matrices[:, 3, 3] = 1.0
    mesh.attributes["frame"].data.foreach_set("value", matrices.transpose(0, 2, 1).astype(np.float32).ravel())
    mesh.attributes["ref"].data.foreach_set("value", refs.astype(np.int32))
    mesh.attributes["shown"].data.foreach_set("value", np.zeros(len(P), dtype=bool))
    mesh.update()
    level_obj[LEVEL_KEY] = level
    level_obj.matrix_basis.identity()
    point, up, cell = pieces.center_frame
    built = _built[level_obj.name] = Built(level, world.text(), world.n, (origin + scale * point, up, scale * cell), P,
                                           pieces.ups, sizes, pieces.tops, origin + scale * pieces.distance_points,
                                           pieces.frames, pieces.kinds)
    built.radius = radius


# How much of the view round them a level's pieces fill, against the floor's strapwork's (surface_substitution.INK): posts,
# beams and rings standing up, where the strapwork is thin bars on the ground
INK = 1.0
# What a build is guessed to take of the workers' time until one's measured (schedule.Costs): s a cell squared (a
# Penrose level's 32 cells round, ~11,700 pieces, took ~110 ms on a worker), or s for a whole level
COST_FIRST = 1e-4


def cost_kind(radius):
    """A 3D parts build's kind and units, for schedule.Costs: its cells squared, or the whole level"""
    return ("parts whole", 1.0) if math.isinf(radius) else ("parts", radius * radius)


def _has(settings, kind):
    return any(p.get("on", "vertex") == kind for p in settings["pieces"])


def wanted_levels(lattice, world, settings, h, high, below=None):
    """The levels with cells from 1 / fade[1] (or with no fade, 1 / fine) of the drone's height h (map units) to high,
    where the drone is (below: the point below it, lattice.cell_sizes()), one more
    either way, no coarser than first_level. Their cells are measured (cell_sizes()): only where the chart is about
    linear over a level's cells are they 2^level of the anchor's units, and near a cube corner, which the chart pinches,
    a level's cells are up to N^(1/3) times bigger than that, the coarse ones and the fine ones at the corner alike."""
    fade = settings.get("fade", DEFAULTS["fade"])
    first = int(setting(settings, "first_level"))
    low = h / float(fade[1] if fade else setting(settings, "fine"))
    unit = lattice.unit_level(world)
    sizes = lattice.cell_sizes(world, [unit], below)
    # From the coarsest the anchor's units would want, finer, until a level's cells are below low: never coarser than
    # the levels wanted, since the chart only ever makes cells bigger than that. A level's cells are lattice.ratio
    # times smaller than the next coarser's (2, or a Penrose map's phi).
    ratio = lattice.ratio
    steps = math.log2(high / sizes[unit]) if ratio == 2.0 else math.log(high / sizes[unit]) / math.log(ratio)
    level = max(first, unit - math.floor(steps))
    while True:
        batch = [k for k in range(level, level + LEVEL_BATCH) if k not in sizes]
        sizes.update(lattice.cell_sizes(world, batch, below))
        level += LEVEL_BATCH
        if sizes[level - 1] < low:
            break
    measured = sorted(k for k in sizes if k >= first)
    coarse = [k for k in measured if sizes[k] <= high]
    fine = [k for k in measured if sizes[k] >= low]
    if not coarse or not fine:
        return []
    # With parts on every level_step-th level only (drawn()), the one more either way is the next of those
    step = level_step(lattice)
    return list(range(max(first, coarse[0] - step), fine[-1] + 1 + step))


def level_step(lattice):
    return getattr(lattice, "level_step", 1)


def level_phase(level, phases, step=1):
    """A level's color phase (piece_color()): with parts on every step-th level only (drawn()), each of those takes
    the next phase, so all of a piece's colors still come round in turn"""
    return (level // step) % phases


def drawn(lattice, level, first):
    """Whether level has parts: every lattice.level_step-th level, counted from first (first_level, which reaches to
    the window), so the same levels have them wherever the drone is and however it has looped"""
    return (level - first) % level_step(lattice) == 0


# Levels measured at a time (cell_sizes()), going finer
LEVEL_BATCH = 8

# A level's pieces laid out round a lattice point, in the level's own units (a lattice's LocalPieces: what doesn't depend
# on the map's anchor), kept for every point looked at, so a giant level's pieces near the drone are placed once, not
# every loop: (lattice, level, point, radius, kinds, heights) -> its LocalPieces
_local = {}
LOCAL_KEPT = 256


class MapState:
    """What a map keeps between ticks besides its levels: the finest giant level plus one (boundary: the coarsest
    instanced level), the giant levels as built (Exact), the levels whose last build on the worker thread failed
    (built here next time, so the error shows), and for building ahead: the point below's motion and how long the
    worker thread's builds take to be written (the map's schedule.Motion, shared with its floor's)"""

    def __init__(self, name):
        self.boundary = None
        self.exact = None
        self.failed = set()
        self.motion = schedule.motion_of(name)
        self.asked = {}              # level -> when its build on the worker thread was asked for
        self.queued = {}             # level -> (world text, the point its build was asked round), map units
        self.on_worker = False       # whether its lattice's builds go to the worker thread (lattice.pieces_job())
        self.early = {}              # level -> how long until it's wanted (s), for those built early (settle())
        self.exact_failures = 0      # how many of the giant levels' builds on the worker thread failed in a row


def jobs_owner(obj):
    """Whose jobs a map's parts builds are on the worker thread (background.py): not the map's own name, which its
    strapwork's builds go by (each takes what's finished by its owner)"""
    return "parts:" + obj.name


def exact_jobs_owner(obj):
    """Whose jobs a map's giant levels' window builds are (settle_exact()): apart from its levels' (jobs_owner()), whose
    finished jobs settle() takes by level"""
    return "parts-exact:" + obj.name


_maps = {}           # map object name -> MapState


def exact_ratio(settings, shape, window):
    """exact_from, or more if the window (in drone heights) is so wide next to it that a giant level's pieces in it
    wouldn't all be standing in full: instanced, those pieces sink by how far they are (fade[0]), and in the window
    every piece stands, so where they're handed over every piece in the window has to be within fade[0] cells (with no
    fade, within the cells it was built out to, less how far the drone may have moved since)"""
    fade = settings.get("fade", DEFAULTS["fade"])
    standing = float(fade[0]) if fade else float(setting(settings, "build")) - float(setting(settings, "margin"))
    room = max(standing - float(shape.reach.max()) - 0.5, 0.5)
    return max(float(setting(settings, "exact_from")), window / room)


def settle(obj, camera_world, budget=BUILD_BUDGET, view=None):
    """At the end of a tick: give the instanced levels the drone's height wants level objects, build those never
    built or which the drone has moved "margin" of their cells from (as many as budget seconds allow; measured ahead
    of it as it flies the way the view looks, view: the camera's rotation in the world; see LEAD_MOST), put every
    level object where its level is in today's map, and sink each piece by how far it is (fade()), or with the view's
    reach, show what the view sees (aim()); then build the giant levels' pieces in the window, if the drone has left
    it or looped (settle_exact()). Only a build works exactly from the anchor; the rest, every tick, in floats from
    what was built."""
    settings = obj.get(PROPERTY)
    if settings is None:
        return
    start = time.perf_counter()
    lattice = lattice_of(obj)
    world = lattice.world(obj)
    text = world.text()
    camera, q, h = lattice.camera(obj, world, camera_world)
    # The drone camera's clip end in map units: nothing past it is drawn
    unit = float(np.linalg.norm(lattice.world_vector(obj, (1.0, 0.0, 0.0))))
    clip_end = sphere.drone_clip_end() / unit
    window = float(setting(settings, "window")) * clip_end
    placer = lattice.placer(world, h)
    objects = pool(obj)
    collection, phases, shape = references(obj, settings)
    state = _maps.get(obj.name)
    if state is None:
        state = _maps[obj.name] = MapState(obj.name)
    # The giant levels' window builds finished on the worker thread, taken in first: a level they hand over to the
    # giant levels is hidden below in this same tick (sinking)
    exact_landed = land_exact(lattice, obj, state)
    exact_from = exact_ratio(settings, shape, window / h)
    # Each level's cells measured where the drone is, as its floor's are
    below = lattice.below(obj, world, q)
    wanted = wanted_levels(lattice, world, settings, h, 2.0 * exact_from * h, below)
    sizes = lattice.cell_sizes(world, wanted, below)
    first = int(setting(settings, "first_level"))
    # The boundary between giant levels and instanced ones: a level becomes giant once its cells are 2 exact_from
    # drone heights, and instanced again under exact_from, once its instances are built (rising: every such level at
    # once, the finest first, handed over in turn from the boundary down; climbing, one at a time, each waiting its build
    # on the worker thread, fell behind, and the giant levels it left kept ever more pieces in the window)
    boundary = state.boundary
    if wanted and (boundary is None or not wanted[0] <= boundary <= wanted[-1] + 1):
        boundary = next((k for k in wanted if sizes[k] < exact_from * h), wanted[-1] + 1)
    while wanted and boundary <= wanted[-1] and sizes[boundary] >= 2.0 * exact_from * h:
        boundary += 1
    low = max(first, wanted[0]) if wanted else boundary
    rising = []
    while wanted and boundary - 1 - len(rising) >= low and sizes[boundary - 1 - len(rising)] < exact_from * h:
        rising.append(boundary - 1 - len(rising))
    # A level with no parts (drawn()) has nothing to build before it's handed over
    rising = [k for k in rising if drawn(lattice, k, first)]
    # Diving, a level past the boundary whose pieces the giant levels' window doesn't have yet (built on the worker
    # thread, settle_exact()) stays instanced until they're written: sinking, the way a rising level waits for its
    # instances (built here, the window had it in the same tick)
    sinking = []
    if exact_async(lattice, state) and wanted:
        have = set(state.exact.levels)
        sinking = [k for k in range(first, boundary) if k not in have and drawn(lattice, k, first)
                   and any(o.get(LEVEL_KEY) == k and _built.get(o.name) is not None for o in objects)]
    instanced = rising + sinking + [k for k in wanted if k >= boundary and drawn(lattice, k, first)]
    # Diving, the next finer levels with parts, built before they're wanted (EARLY_TIME): each with how long until it
    # is, from the dive's rate, d(ln h)/dt. A level is wanted (wanted_levels()) once the one level_step coarser has
    # cells of h / fade[1] (or "fine").
    early = {}
    rate = state.motion.descent
    if wanted and background.enabled and state.on_worker and rate < 0.0:
        fades = settings.get("fade", DEFAULTS["fade"])
        ratio = float(fades[1] if fades else setting(settings, "fine"))
        step = level_step(lattice)
        soon_h = h * math.exp(rate * EARLY_TIME)
        for k in wanted_levels(lattice, world, settings, soon_h, 2.0 * exact_from * soon_h, below):
            if k > wanted[-1] and drawn(lattice, k, first) and len(early) < EARLY_MOST:
                cell = lattice.cell_sizes(world, [k - step], below)[k - step]
                early[k] = max(0.0, math.log(ratio * cell / h) / rate) if cell > 0.0 else EARLY_TIME
    state.early = early
    if early:
        sizes = {**sizes, **lattice.cell_sizes(world, sorted(early), below)}
    keep = set(instanced) | set(early)
    owner = jobs_owner(obj)
    by_level = {}
    for level_obj in objects:
        level = level_obj.get(LEVEL_KEY)
        if level is None:
            continue
        if level not in keep or level in by_level:
            if level not in keep:
                background.cancel((owner, level))
            del level_obj[LEVEL_KEY]
            _clear(level_obj)
        else:
            by_level[level] = level_obj
    # The worker thread's finished builds, written into their levels' objects (a level no longer instanced is dropped)
    for key, result in background.finished(owner):
        level = key[1]
        asked = state.asked.pop(level, None)
        if asked is not None:
            state.motion.took("parts", start - asked)
        if result is background.FAILED:
            state.failed.add(level)
        else:
            job_world, pieces, origin, radius = result
            kind, units = cost_kind(radius)
            state.motion.costs.took(kind, background.ran(key), units)
        if result is not background.FAILED and level in by_level:
            write_level(by_level[level], job_world, level, pieces, origin,
                        level_phase(level, phases, level_step(lattice)), radius)
    free = [o for o in objects if o.get(LEVEL_KEY) is None]
    margin = float(setting(settings, "margin"))
    build = float(setting(settings, "build"))
    # Where the drone is over the surface, which a level's center is measured from, and where it will be when a build
    # asked for now is written (soon)
    ground = lattice.ground(obj, world, camera, q)
    velocity = state.motion.velocity(world, text, ground, h, start, lattice.transform)
    # Once something is drawn, a lattice that can (lattice.pieces_job()) has its builds done on the worker thread: the
    # level stays as it was built until the new build is taken in a later tick (above). A flight's start, or a file just
    # loaded, builds here, budget seconds a tick, as does a level whose last build there failed.
    worker = background.enabled and any(_built.get(o.name) is not None for o in by_level.values())
    wait = state.motion.wait("parts") if worker and state.on_worker else 0.0
    soon = ground + velocity * wait
    speed = float(np.linalg.norm(velocity))
    # The camera's right, up and back in map units (unit, as columns)
    axes = None
    if view is not None:
        axes = np.linalg.solve(np.array(obj.matrix_world.to_3x3()), np.array(view, dtype=float))
        axes /= np.linalg.norm(axes, axis=0)
    # Each level's lead along the flight (map units of this world: see LEAD_MOST), as far as the view looks that way:
    # drifting backward or sideways, or looking straight down, the drone sees round itself
    facing = 0.0
    if worker and speed > 0.0 and axes is not None:
        facing = max(0.0, float(-axes[:, 2] @ velocity) / speed)
    lead = lambda level, cells: (velocity * (facing * min(cells * sizes[level], speed * LEAD_MOST) / speed)
                                 if facing > 0.0 else np.zeros(3))
    # The view's reach: with no fade, once the builds are on the worker thread, a level (but the coarsest, which has
    # its own reach) is built out to "ahead" of its cells, and shows that far only what the camera sees (aim())
    tangents = sphere.view_tangents() if axes is not None else None
    sight = View(axes, tangents, build) if tangents is not None and not settings.get("fade", DEFAULTS["fade"]) else None
    ahead_cells = float(setting(settings, "ahead"))
    in_view = worker and state.on_worker and sight is not None and ahead_cells > build

    def share(level, radius):
        """The share of the view a build of level out to radius cells puts its pieces in (schedule.view_share(),
        times INK)"""
        return INK * (1.0 if math.isinf(radius) else schedule.view_share(radius * sizes[level], h))

    def cells_away(level, cells):
        """How long until the drone has flown that many of level's cells (0 if it has)"""
        if cells <= 0.0:
            return 0.0
        return cells * sizes[level] / speed if speed > 0.0 else 10.0

    # What's due: (stake, wanted in s, level, its object, radius, lead, the point to build round or None)
    due = []
    for level in instanced + sorted(early):
        if level in sinking:
            # Only kept drawn as it was until the giant levels' window has it: not built again
            continue
        level_obj = by_level.get(level)
        if level_obj is None:
            if not free:
                continue
            level_obj = by_level[level] = free.pop()
            level_obj[LEVEL_KEY] = level
            _built.pop(level_obj.name, None)
        if level in early:
            # Built once, out to "build", round where the drone will be when it's wanted (the point below's velocity,
            # in heights a second, falls with the height: e^(rate t)); built again only after it's wanted
            built = _built.get(level_obj.name)
            if built is None or built.level != level or abs(world.n - built.n) > LOOPS_STALE:
                t = early[level]
                at = q + velocity * ((1.0 - math.exp(rate * t)) / -rate)
                due.append((share(level, build), t, level, level_obj, build, margin, at))
            continue
        # The coarsest level has none coarser going on past its edge: it reaches as far as the window, if it can
        viewed = in_view and level != first
        radius = (lattice.first_radius(world, level, build, window / sizes[level] + margin) if level == first
                  else ahead_cells if viewed else build)
        # How far the drone may move from where it was built round before it's built again: with the view's reach,
        # what's shown all round moves with the drone (aim()), so "margin" as much a share of "ahead" as of "build"
        step = margin * ahead_cells / build if viewed else margin
        # Its lead, cells: with the view's reach, as far as keeps the drone's "build" all round inside what's built
        # when the build is asked for (two leads behind its center), and it's built again once the point a lead ahead
        # of the drone is that lead and "step" past its center: a build lasts 2.5 times the flight it did
        led = (ahead_cells - build) / 2.0 if viewed else margin
        if viewed:
            step += led
        built = _built.get(level_obj.name)
        behind = False
        if built is not None and built.level == level and math.isfinite(built.radius) and math.isfinite(radius):
            # One the drone has left the build of shows nothing round it: built again as if never built (its quick
            # build first). Asked for its full reach, a coarse level's was ten times the quick build's cost for the
            # same share of the view, and it waited a second and a half behind the floor's
            offset, scale = lattice.transform(world, text, built)
            behind = built.away((soon - offset) / scale) > built.radius
        if built is None or built.level != level or abs(world.n - built.n) > LOOPS_STALE or behind:
            # A level with nothing of its own drawn is built out to "build" first, a tenth of the pieces of "ahead", and
            # out to its full reach after (below): in a fast dive some six new levels came a second, each built in full
            # (about 110 ms on a worker thread), and most were passed before they were drawn, the view only floor
            quick = viewed and math.isfinite(radius)
            reach = build if quick else radius
            due.append((share(level, reach), 0.0, level, level_obj, reach, margin if quick else led, None))
            continue
        if math.isinf(radius):
            continue
        offset, scale = lattice.transform(world, text, built)
        if built.radius != radius:
            # Its quick build (above), or a reach that's changed: built to radius, worth the ring it adds, wanted once
            # the drone nears the edge of what's there
            moved = built.away((soon - offset) / scale)
            if built.radius >= radius:
                stake, needed_in = schedule.REFRESH * share(level, radius), 0.0
            else:
                stake = max(share(level, radius) - share(level, max(built.radius - moved, 0.0)), 0.0)
                needed_in = cells_away(level, built.radius - margin - moved)
            due.append((stake, needed_in, level, level_obj, radius, led, None))
            continue
        moved = built.away((soon + lead(level, led) - offset) / scale)
        if moved > step:
            # Worth the share of the view it adds to what's still in reach of the last build round the drone, wanted
            # once the drone will be near its edge when it lands
            stake = max(share(level, radius) - share(level, max(radius - moved, 0.0)), schedule.REFRESH * share(
                level, radius))
            due.append((stake, cells_away(level, radius - step - moved), level, level_obj, radius, led, None))
    plain = None
    # What each is worth (schedule.value(), by the same rule as the floor's), the best asked for first (a tick asks for
    # budget s at most)
    parts_wait = state.motion.wait("parts")
    rated = []
    for stake, needed_in, level, level_obj, radius, led, at in due:
        kind, units = cost_kind(radius)
        cost = state.motion.costs.guess(kind, units, COST_FIRST)
        rated.append((schedule.priority(stake, needed_in, parts_wait, cost), level, level_obj, radius, led, at))
    for priority, level, level_obj, radius, led, at in sorted(rated, key=lambda r: r[0]):
        if time.perf_counter() - start > budget:
            break
        key = (owner, level)
        # Round where the drone will be, two leads on along its flight (LEAD_MOST); an early level's round where it
        # will be when it's wanted
        if at is not None:
            ahead = at - q
        else:
            ahead = velocity * wait + 2.0 * lead(level, led) if wait > 0.0 or facing > 0.0 else np.zeros(3)
        # A build still waiting on the worker thread is handed in again round the new point once that's "margin" of the
        # level's cells from where it was asked for (submit(replace=True): in its place in line), not built round where
        # the drone was when it was asked, which in a fast glide was up to 2 s and hundreds of cells back
        queued = state.queued.get(level)
        if background.pending(key) and (background.running(key) or queued is not None and queued[0] == text
                                        and np.linalg.norm(q + ahead - queued[1]) < margin * sizes[level]):
            background.rate(key, priority)
            continue
        if ahead.any():
            center = lattice.center(world, lattice.below(obj, world, q + ahead), level)
        else:
            center = lattice.center(world, below, level)
        job = None
        if worker and level not in state.failed:
            if plain is None:
                plain = settings.to_dict() if hasattr(settings, "to_dict") else settings
            job = lattice.pieces_job(world, level, center, radius, plain, shape)
        if job is not None:
            if background.submit(key, *job, priority=priority, replace=True):
                state.asked[level] = start
                state.queued[level] = (text, q + ahead)
                state.on_worker = True
            continue
        state.failed.discard(level)
        background.cancel(key)
        try:
            build_level(lattice, obj, level_obj, world, level, center, settings, phases, shape, placer, radius)
        except ValueError:
            # One that fails here too stays as it was, the error printed, and goes to the worker thread again next
            # time: raised, it ended the flight (a Penrose level cut with no tiles near the drone)
            traceback.print_exc()
    # Handed over from the boundary down: each rising level once its instances are built, or it has no parts, or it's
    # RISING_LATE behind
    while (wanted and boundary - 1 >= low and sizes[boundary - 1] < exact_from * h
           and (boundary - 1 not in rising or sizes[boundary - 1] * RISING_LATE < exact_from * h
                or boundary - 1 in by_level and _built.get(by_level[boundary - 1].name) is not None)):
        boundary -= 1
    state.boundary = boundary if wanted else None
    for level, level_obj in by_level.items():
        built = _built.get(level_obj.name)
        if built is None:
            continue
        if level < boundary and level not in sinking or level not in instanced:
            # Rising, not handed over yet: built, but the giant levels still draw it; or early, not wanted yet
            _hide_level(level_obj, built)
            continue
        offset, scale = lattice.transform(world, text, built)
        place(level_obj, built, text, offset, scale)
        if sight is not None and level != first and built.radius > build:
            aim(level_obj, built, (camera - offset) / scale, window / scale, sight)
        else:
            fade(level_obj, built, (camera - offset) / scale, settings)
    giant = [k for k in range(first, boundary) if drawn(lattice, k, first)] if wanted else []
    settle_exact(lattice, obj, state, world, text, camera, q, h, clip_end, window, giant, settings, shape, phases,
                 placer, exact_landed)


def _hide_level(level_obj, built):
    if built.written is not None and not built.written.any():
        return
    built.written = np.zeros(len(built.points), dtype=np.int16)
    # Worked out again once it's shown, however little the camera has moved by then
    built.faded = built.aimed = None
    level_obj.data.attributes["shown"].data.foreach_set("value", np.zeros(len(built.points), dtype=bool))
    level_obj.data.update()


def place(level_obj, built, text, offset, scale):
    """Put a level object where its level is in today's map (world text), if it isn't there yet: only its transform
    changes (its points and their instances stay as they were)"""
    if built.placed == text:
        return
    from mathutils import Matrix, Vector
    built.placed = text
    level_obj.matrix_basis = Matrix.Translation(Vector(offset)) @ Matrix.Scale(scale, 4)


def fade(level_obj, built, camera, settings):
    """Sink each of a level's pieces by how far it is from the camera (in its own cells: standing within fade[0], gone
    by fade[1]), down its up by its height: the camera in the level's own units. Worked out again only once the camera
    has moved FADE_MOVE of the level's cells (less can't move a piece by a step), and written only where it changed.
    A piece the camera is in stays drawn (see the module's docstring)."""
    last = built.faded
    if last is not None and float(np.linalg.norm(camera - last)) < FADE_MOVE * built.cell:
        return
    built.faded = camera
    fades = settings.get("fade", DEFAULTS["fade"])
    if not fades:
        # Nothing sinks: every piece stands, written once
        if built.written is None or not np.all(built.written == FADE_STEPS):
            built.written = np.full(len(built.points), FADE_STEPS, dtype=np.int16)
            built.current = built.points
            mesh = level_obj.data
            mesh.attributes["position"].data.foreach_set("vector", built.points.astype(np.float32).ravel())
            mesh.attributes["shown"].data.foreach_set("value", np.ones(len(built.points), dtype=bool))
            mesh.update()
        return
    near, far = fades
    distance = np.linalg.norm(built.distance_points - camera, axis=1) / built.sizes
    shown = smoothstep(float(far), float(near), distance)
    steps = np.round(shown * FADE_STEPS).astype(np.int16)
    if built.written is not None and np.array_equal(steps, built.written):
        return
    built.written = steps
    sunk = (1.0 - steps / FADE_STEPS) * built.tops * SUNK * built.sizes
    built.current = built.points - sunk[:, None] * built.ups
    mesh = level_obj.data
    mesh.attributes["position"].data.foreach_set("vector", built.current.astype(np.float32).ravel())
    mesh.attributes["shown"].data.foreach_set("value", steps > 0)
    mesh.update()


class View:
    """The camera's view this tick, for the view's reach: its right, up and back (unit, as columns, in map units), how
    wide it sees (VIEW_ROOM wider than the camera) and how far a level reaches all round ("build" cells)"""

    def __init__(self, axes, tangents, build):
        self.axes, self.build = axes, build
        self.tangents = [math.tan(min(math.atan(t) + VIEW_ROOM, VIEW_WIDEST)) for t in tangents]

    def sees(self, points, eye, window):
        """Which of points ((n, 3)) the view sees from eye, as far as window (past it, past the clip end, nothing's
        drawn, and a coarse level's "ahead" of its cells reaches far past it)"""
        local = (points - eye) @ self.axes
        depth = -local[:, 2]
        return ((depth > 0.0) & (np.abs(local[:, 0]) <= depth * self.tangents[0])
                & (np.abs(local[:, 1]) <= depth * self.tangents[1])
                & (np.einsum("ij,ij->i", local, local) <= window * window))

    def turned(self, axes):
        """How far this view is turned from a camera turned as axes (like self.axes): its axis's or its up's turn,
        radians"""
        dots = np.sum(self.axes[:, 1:] * axes[:, 1:], axis=0)
        return float(np.arccos(np.clip(dots, -1.0, 1.0)).max())


def aim(level_obj, built, camera, window, view):
    """Show a level's pieces (with no fade, built out past view.build) within view.build of its cells round the point
    below the camera (so the drone flies on through what's built: settle()), and those the view sees from the camera
    out to window (the level's own units): the view's reach. Worked out again once the view has turned VIEW_TURN from
    the one it was worked out for, counting the camera's move as the turn it makes of the nearest piece shown only for
    being in view (as far as the camera is over the level, and view.build cells along it), and written only where it
    changed."""
    up = built.center_up
    height = float((camera - built.center_point) @ up)
    if built.aimed is not None:
        eye, axes = built.aimed
        nearest = math.hypot(height, view.build * built.cell)
        moved = float(np.linalg.norm(camera - eye))
        if view.turned(axes) + moved / max(nearest, 1e-300) <= VIEW_TURN:
            return
    built.aimed = (camera, view.axes)
    # Along the level from the point below the camera
    along = built.distance_points - (camera - height * up)
    along -= np.outer(along @ up, up)
    near = np.einsum("ij,ij->i", along, along) <= (view.build * built.cell) ** 2
    steps = np.where(near | view.sees(built.distance_points, camera, window), FADE_STEPS, 0).astype(np.int16)
    if built.written is not None and np.array_equal(steps, built.written):
        return
    built.written = steps
    built.faded = None
    mesh = level_obj.data
    mesh.attributes["shown"].data.foreach_set("value", steps > 0)
    mesh.update()


# ----- the giant levels: their pieces in the window, exactly, as one mesh cut to it -----

class ExactPiece:
    """A giant level's piece in the window: its frame (axes as columns, float64, and its inverse) and size (map
    units: a float, or with digits a Decimal, which can be past a float's range), the camera it was built for in its
    own units (u, the same type), its faces cut to the window (corners from that camera, map units) with their colors,
    and how far the camera is outside it (map units). at: that camera from the window's middle (Exact.center), map
    units then (a level added to the window later was worked out for the camera then, Exact.add())"""

    def __init__(self, level, kind, frame, size, u, digits, faces, colors, distance):
        self.level, self.kind, self.frame, self.size, self.u, self.digits = level, kind, frame, size, u, digits
        self.inverse = np.linalg.inv(frame)
        self.faces, self.colors, self.distance = faces, colors, distance
        self.at = np.zeros(3)

    def rescale(self, scale):
        """In map units scale times bigger (a power of 2: a loop's), so exactly"""
        if scale == 1.0:
            return
        if self.digits is None:
            self.size = self.size / scale
        else:
            import decimal
            with decimal.localcontext(prec=self.digits):
                self.size = self.size / decimal.Decimal(scale)
        self.distance /= scale
        self.faces = [face / scale for face in self.faces]

    def distance_at(self, move, shape):
        """How far a camera moved by move (map units, from the one it was built for) is outside it, in map units"""
        step = (self.inverse @ move).tolist()
        if self.digits is None:
            u = [a + b / self.size for a, b in zip(self.u, step)]
            return min(prism.distance_one(u) for prism in shape.prisms[self.kind]) * self.size
        import decimal
        from decimal import Decimal
        with decimal.localcontext(prec=self.digits):
            u = [a + Decimal(b) / self.size for a, b in zip(self.u, step)]
            return float(min(prism.distance_one(u) for prism in shape.prisms[self.kind]) * self.size)


class Exact(schedule.Window):
    """The giant levels as built into the map's exact object (kept in memory, by map): its window (schedule.Window: the
    object's mesh is from the camera then), the levels, and the pieces in the window. placed: the world text the
    object's transform is for."""

    def __init__(self, world_text, n, levels, center, window):
        super().__init__(world_text, n, center, window)
        self.levels = levels
        self.pieces = []


# The giant levels' window builds go before the levels' (background.submit()'s priority, lower first): a level handed
# over waits for one, instanced (sinking), holding a level object (after the levels', a dive at a plane kept 13 levels
# sinking, and the finest ran out of objects)
EXACT_PRIORITY = -1e29


# The giant levels' window is built in the tick once this many of its builds on the worker thread have failed in a row
# (exact_async()). After the first, one build that failed (its tiles pickled while the tick was changing them) left
# the map building its windows in the tick for the rest of the flight, 0.6 s frozen at a time in a climb.
EXACT_FAILURES = 3


def exact_async(lattice, state):
    """Whether the map's giant levels' window is built on the worker thread (settle_exact()): once there is one, if its
    lattice can (lattice.add_levels_job()) and its builds there haven't kept failing (EXACT_FAILURES)"""
    return (background.enabled and state.exact is not None and state.exact_failures < EXACT_FAILURES
            and getattr(lattice, "add_levels_job", None) is not None)


def exact_job(function, args, world, camera, window, levels, fresh, window_for):
    """A giant levels' window build on the worker thread: lattice.add_levels_job()'s (function, args), function(*args)
    -> (pieces, then: what the main thread does with them first, or None), its world args[0]; with what it was built
    for (land_exact())"""
    # (In a build process if function builds_apart: background.call())
    pieces, then = background.call(function, *args)
    return world, camera, window, levels, fresh, window_for, pieces, then


def land_exact(lattice, obj, state):
    """The giant levels' window builds finished on the worker thread, taken into the map's window (a fresh one, or
    the levels added to the one it was asked for, if that's still the map's): whether any was"""
    landed = False
    for _key, result in background.finished(exact_jobs_owner(obj)):
        if result is background.FAILED:
            state.exact_failures += 1
            continue
        state.exact_failures = 0
        world, camera, window, levels, fresh, window_for, pieces, then = result
        if then is not None:
            then()
        if fresh:
            built = state.exact = Exact(world.text(), world.n, (), camera, window)
        else:
            built = state.exact
            if built is None or built is not window_for:
                continue
        add_pieces(lattice, built, world, camera, pieces)
        built.levels = tuple(sorted(set(built.levels) | set(levels)))
        landed = True
    return landed


def add_pieces(lattice, built, world, camera, pieces):
    """Giant levels' pieces (lattice.add_levels(): from camera, in world's map units) into the window, in its units"""
    offset, scale = lattice.transform(world, world.text(), built)
    at = (camera - offset) / scale - built.center
    for piece in pieces:
        piece.rescale(scale)
        piece.at = at
        piece.faces = [face + at for face in piece.faces]
        built.pieces.append(piece)


def settle_exact(lattice, obj, state, world, text, camera, q, h, clip_end, window, levels, settings, shape, phases,
                 placer, landed=False):
    """The giant levels' pieces in the window (add_levels()), built again when the camera's view (clip_end) reaches
    past it or it's grown far wider than wanted (the drone has shrunk: its corners are float32 from its middle, so
    they'd be a growing share of the drone's height off), and a level's own added or taken away when it crosses the
    boundary; and put where they are in today's map. Loops don't build it: they only scale it, by a power of 2.

    Once there's a window, a lattice that can (lattice.add_levels_job()) has those builds done on the worker thread
    (exact_async()): the window stays as it was until they're taken in (land_exact(), at the start of the next
    settle()), a level handed over meanwhile still instanced (sinking). Built here, the plane maps' took 50 to 110 ms
    two or three times a loop, which a dive at a plane comes to once a second."""
    exact_obj = exact_object(obj)
    built = state.exact
    # A loop changes only the window's transform (by a power of 2, exactly): it's the same window, the same size
    rebuild = built is None
    if not rebuild:
        offset, scale = lattice.transform(world, text, built)
        local = (camera - offset) / scale
        rebuild = (not built.covers(float(np.linalg.norm(local - built.center)), clip_end / scale)
                   or built.overbuilt(window / scale))
    write = landed
    if exact_async(lattice, state):
        key = (exact_jobs_owner(obj), 0)
        # A level no longer giant goes at once (its instances are drawn: it rose); those to add, and a fresh window,
        # one job at a time. (Sets: a plane map has a thousand giant levels, and lists compared item by item cost the
        # tick 5 ms.)
        added = []
        if built.levels != tuple(levels):
            wanted, have = set(levels), set(built.levels)
            if not have <= wanted:
                built.pieces = [piece for piece in built.pieces if piece.level in wanted]
                built.levels = tuple(level for level in built.levels if level in wanted)
                have &= wanted
                write = True
            added = [level for level in levels if level not in have]
        if (rebuild or added) and not background.pending(key):
            asked = levels if rebuild else added
            plain = settings.to_dict() if hasattr(settings, "to_dict") else settings
            job = lattice.add_levels_job(world, camera, q, h, window, asked, plain, shape, phases, placer)
            if job is not None:
                function, args = job
                background.submit(key, exact_job, function, args, args[0], np.array(camera, dtype=float), window,
                                  tuple(asked), rebuild, built, priority=EXACT_PRIORITY)
    else:
        if rebuild:
            built = state.exact = Exact(world.text(), world.n, (), np.array(camera, dtype=float), window)
            offset, scale = lattice.transform(world, text, built)
        # A level crossing the boundary (three times a loop) only adds or takes away its own pieces: the window's other
        # levels stay as they were, and an added level's are taken to the window's units (map units when it was built)
        if built.levels != tuple(levels):
            built.pieces = [piece for piece in built.pieces if piece.level in levels]
            added = [level for level in levels if level not in built.levels]
            if added:
                at = (camera - offset) / scale - built.center
                for piece in lattice.add_levels(world, camera, q, h, window, added, settings, shape, phases, placer):
                    piece.rescale(scale)
                    piece.at = at
                    piece.faces = [face + at for face in piece.faces]
                    built.pieces.append(piece)
            built.levels = tuple(levels)
            rebuild = True
        write |= rebuild
    built = state.exact
    offset, scale = lattice.transform(world, text, built)
    if built.placed != text:
        from mathutils import Matrix, Vector
        built.placed = text
        exact_obj.matrix_basis = Matrix.Translation(Vector(offset + scale * built.center)) @ Matrix.Scale(scale, 4)
    if write:
        write_exact(exact_obj, built)


def write_exact(exact_obj, built):
    """The exact object's mesh: the faces of every piece in the window"""
    corners, sizes, colors = [np.zeros((0, 3))], [np.zeros(0, dtype=np.int64)], [np.zeros((0, 4))]
    for piece in built.pieces:
        for face, color in zip(piece.faces, piece.colors):
            corners.append(face)
            sizes.append([len(face)])
            colors.append([color])
    tiling.write_mesh(exact_obj, np.concatenate(corners), np.concatenate(sizes).astype(np.int64),
                      np.concatenate(colors))


def exact_piece(pieces, row, level, camera, local, window, shape, settings, phases, placer):
    """A giant level's piece (row of pieces) worked out for the window: its frame and size from the lattice points it
    stands on, the camera in its own units, how far it's outside the piece, and its faces cut to the window (a cube
    2 window / size of its own units across, around the camera), each corner then from the camera in map units. In
    float64 from Pieces (the camera there: local) without placer, or with placer()'s digits, exactly, from the map's
    anchor. None if no face reaches the window."""
    kind = int(pieces.kinds[row])
    rows = pieces.piece_rows(row)
    digits = None
    # A placer may place in a frame of its own, turn (3 x 3, floats) from the map's axes: the icosahedron's folded
    # anchor's (parts_ico), whose turn is irrational, so applied only to the camera, near the anchor, and the faces,
    # a window across: never to the pieces' places, which are far
    turn = None
    if placer is None:
        points = [tuple(float(v) for v in pieces.points[r]) for r in rows]
        normals = [tuple(float(v) for v in pieces.normals[r]) for r in rows]
        camera_typed = tuple(float(v) for v in local)
        scale = pieces.scale(pieces.world.rho)
        reach = window / scale
        context = None
    else:
        import decimal
        from decimal import Decimal
        exact = placer()
        digits = exact.digits
        points, normals = pieces.exact_placed(rows, exact)
        turn = getattr(exact, "turn", None)
        context = decimal.localcontext(prec=digits)
        camera_typed = tuple(Decimal(float(v)) for v in (camera if turn is None else turn @ np.asarray(camera, float)))
        scale = None
        reach = Decimal(window)
    with context if context is not None else _no_context():
        number = type(points[0][0])
        scales = None if kind == 0 else tuple(number(float(s)) for s in pieces.edge_scales[row - pieces.vertex_count])
        place, size, axes = piece_axes(kind, points, normals, number(shape.meet), scales=scales)
        if turn is not None and np.linalg.det(turn) < 0.0:
            # A turn that mirrors (the icosahedron's fold at most anchors) mirrors y = z x x too: turned back to the
            # map's axes, the frame worked out here would be left-handed, the piece its mirror image
            axes = (axes[0], _scale(axes[1], -1), axes[2])
        inverse = inverse_3x3([[axes[c][r] for c in range(3)] for r in range(3)])
        relative = _sub(camera_typed, place)
        u = tuple(_dot(inverse[c], relative) / size for c in range(3))
        distance = min(prism.distance_one(u) for prism in shape.prisms[kind])
        half = reach / size
        if scale is not None:
            # The pieces' units to the map's
            size = size * scale
        corners = window_faces(shape, kind, u, half, size, number)
        frame = np.array([[float(axes[c][i]) for c in range(3)] for i in range(3)])
        distance = float(distance * size)
    if not corners:
        return None
    faces = [face @ frame.T for face, _ in corners]
    if turn is not None:
        faces = [face @ turn for face in faces]
        frame = turn.T @ frame
    colors = face_colors(settings, corners, level % phases)
    return ExactPiece(level, kind, frame, size, u, digits, faces, colors, distance)


def piece_axes(kind, points, normals, meet, stretch=None, top=None, scales=None):
    """A piece's place, size and axes (its frame's columns) from the lattice points it stands on (a vertex piece's
    vertex then its neighbors, an edge piece's two vertices) and their normals (unit), as LaidOut lays them out
    (upright_frames(), edge_frames(), a vertex piece stretched about its height top by stretch: foot_stretch(), an edge
    piece's ends at their vertex pieces' scales), in the numbers' own type: floats, or Decimals (exactly, at their
    precision) with meet, stretch, top and scales ones too"""
    P = points[0]
    if kind == 0:
        gaps = [_sub(p, P) for p in points[1:]]
        size = sum(square_root(_dot(g, g)) for g in gaps) / len(gaps)
        z = normals[0]
        x = _perpendicular(gaps[0], z)
        if stretch is not None and stretch != 1:
            return _add(P, _scale(z, (1 - stretch) * top * size)), size, (x, _cross(z, x), _scale(z, stretch))
        return P, size, (x, _cross(z, x), z)
    along = _sub(points[1], P)
    size = square_root(_dot(along, along))
    s0, s1 = scales
    sum_n = _add(normals[0], normals[1])
    a0, a1 = _scale(normals[0], s0), _scale(normals[1], s1)
    turn = _sub(a1, a0)
    z = _scale(_add(a0, a1), 1 / type(size)(2))
    if not square_root(_dot(sum_n, sum_n)) > type(size)(2 * FOLDED_BACK):
        turn = _scale(turn, 0)
        z = _scale(_perpendicular(normals[0], _scale(along, 1 / size)), s0)
    x = _add(_scale(along, 1 / size), _scale(turn, meet))
    y = _cross(z, x)
    if not square_root(_dot(y, y)) > type(size)(1e-9) * square_root(_dot(z, z)) * square_root(_dot(x, x)):
        y = _perpendicular(z, _unit(x))
    else:
        y = _unit(y)
    y = _scale(y, (s0 + s1) / 2)
    return _sub(P, _scale(turn, meet * size / 2)), size, (x, y, z)


def inverse_3x3(m):
    """A 3 x 3 matrix's inverse (rows of numbers of one type: with Fractions, exactly), by its adjugate"""
    (a, b, c), (d, e, f), (g, h, i) = m
    A, B, C = e * i - f * h, f * g - d * i, d * h - e * g
    det = a * A + b * B + c * C
    return [[A / det, (c * h - b * i) / det, (b * f - c * e) / det],
            [B / det, (a * i - c * g) / det, (c * d - a * f) / det],
            [C / det, (b * g - a * h) / det, (a * e - b * d) / det]]


class _no_context:
    def __enter__(self):
        return None

    def __exit__(self, *_):
        return False


def window_faces(shape, kind, u, half, size, number):
    """A piece's faces cut to the window, a cube 2 half of its own units across round the camera (u, in its units, as
    numbers of type number: floats, Fractions or Decimals, which the cut is worked out in): [(corners from the camera in
    map units (m, 3, floats: size map units a piece unit), the piece's index)]. Every face against the window at once
    (face_stack()) in floats: those wholly outside it left out, and with floats, those wholly inside it (most of them:
    near the boundary a piece is a few hundredths of the window across) taken as they are; only the rest cut (one at a
    time, face by face, the float path's pieces took 1.4 ms each). The corners are scaled to map units before they're
    floats: a giant piece's own units can be far past a float's range, the window's corners never."""
    u_float = np.array([float(c) for c in u])
    half_float = float(half)
    slack = 4e-16 * (1.0 + float(np.abs(u_float).max()))
    stacked, starts, ends = face_stack(shape, kind)
    if not len(starts):
        return []
    near_all = stacked - u_float
    # The camera's place less its float, so the corners from the camera are as good as floats get relative to their own
    # size (the face's float corner less the camera's float is exact where they're within twice each other)
    try:
        u_rest = np.array([float(c - number(f)) for c, f in zip(u, u_float.tolist())]) if number is not float else None
    except OverflowError:
        u_rest = None
    lo = np.minimum.reduceat(near_all, starts, axis=0)
    hi = np.maximum.reduceat(near_all, starts, axis=0)
    reaching = ~(np.any(lo > half_float + slack, axis=1) | np.any(hi < -half_float - slack, axis=1))
    inside = np.maximum(hi, -lo).max(axis=1) < half_float - slack
    corners = []
    for f in np.flatnonzero(reaching):
        face, index = shape.faces[kind][f]
        if number is float and inside[f]:
            corners.append((near_all[starts[f]:ends[f]] * float(size), index))
            continue
        # (On a worker thread, out of the main thread's way between faces: the cut is pure Python, exactly, and without
        # these the tick's own Python ran at half speed beside it)
        background.give_way()
        if u_rest is not None and 0.0 < half_float and float(size) < math.inf:
            # Its corners from the camera in floats (less the camera's rest): within FLOAT_CUT_REACH of the window's
            # size, cut in floats (rounding moves a cut by under 2^-33 of the window); only a face reaching far past it,
            # a giant piece's, cut exactly (every face exactly, a laid-out map's giant frames' window took a second a
            # build)
            near_float = near_all[starts[f]:ends[f]] - u_rest
            if float(np.abs(near_float).max()) <= FLOAT_CUT_REACH * half_float:
                cut = cut_to_window(near_float.tolist(), half_float)
                if len(cut) >= 3:
                    corners.append((np.array(cut) * float(size), index))
                continue
        cut = cut_to_window([[number(float(c)) - u[i] for i, c in enumerate(corner)] for corner in face.tolist()],
                            half)
        if len(cut) >= 3:
            corners.append((np.array([[float(c * size) for c in corner] for corner in cut]), index))
    return corners


# window_faces() cuts a face in floats if its corners are no further from the camera than this many times the window's
# half side (else exactly)
FLOAT_CUT_REACH = 2.0 ** 20


def face_stack(shape, kind):
    """A kind's faces (shape.faces) as one array of their corners, with where each face's start and end, made once"""
    stacks = shape.__dict__.setdefault("_face_stacks", {})
    if kind not in stacks:
        faces = [face for face, _index in shape.faces[kind]]
        lengths = np.array([len(face) for face in faces], dtype=np.int64)
        ends = np.cumsum(lengths)
        stacks[kind] = (np.concatenate(faces).astype(float) if faces else np.zeros((0, 3)), ends - lengths, ends)
    return stacks[kind]


def cut_to_window(polygon, half):
    """A convex polygon (corners, lists of three numbers of one type) cut to the cube from -half to half on every
    axis (Sutherland and Hodgman's), in the corners' own type, so with Decimals as exact as their precision"""
    for axis in range(3):
        for sign in (1, -1):
            if not polygon:
                return polygon
            out = []
            previous = polygon[-1]
            previous_in = sign * previous[axis] <= half
            for corner in polygon:
                corner_in = sign * corner[axis] <= half
                if corner_in != previous_in:
                    t = (sign * half - previous[axis]) / (corner[axis] - previous[axis])
                    out.append([p + (c - p) * t for p, c in zip(previous, corner)])
                if corner_in:
                    out.append(corner)
                previous, previous_in = corner, corner_in
            polygon = out
    return polygon


def _sub(u, v):
    return (u[0] - v[0], u[1] - v[1], u[2] - v[2])


def _add(u, v):
    return (u[0] + v[0], u[1] + v[1], u[2] + v[2])


def _scale(u, s):
    return (u[0] * s, u[1] * s, u[2] * s)


def _dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _cross(u, v):
    return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])


def _unit(u):
    return _scale(u, 1 / square_root(_dot(u, u)))


def _perpendicular(v, axis):
    """perpendicular() of one vector, in its numbers' own type"""
    out = _sub(v, _scale(axis, _dot(v, axis)))
    if not square_root(_dot(out, out)) > type(v[0])(1e-9) * square_root(_dot(v, v)):
        k = min(range(3), key=lambda i: abs(axis[i]))
        other = tuple(type(axis[0])(1 if i == k else 0) for i in range(3))
        out = _sub(other, _scale(axis, _dot(other, axis)))
    return _unit(out)


def _clear(level_obj):
    level_obj.data.clear_geometry()
    level_obj.data.update()
    _built.pop(level_obj.name, None)


def forget(_dummy=None):
    """After loading a file: another file's levels don't apply; each is built again at its first settle (and builds
    under way on the worker thread are thrown away)"""
    _built.clear()
    for name in _maps:
        background.forget("parts:" + name)
        background.forget("parts-exact:" + name)
    _maps.clear()
    _local.clear()
    schedule.forget()
    sphere_exact.forget()
    import sys
    package = __name__.rpartition(".")[0]
    for name in ("parts_sphere", "parts_ico", "parts_substitution"):
        module = sys.modules.get(f"{package}.{name}" if package else name)
        if module is not None:
            module.forget()


def register():
    import bpy
    handler = bpy.app.handlers.persistent(forget)
    if not any(getattr(h, "__name__", "") == "forget" and getattr(h, "__module__", "") == __name__
               for h in bpy.app.handlers.load_post):
        bpy.app.handlers.load_post.append(handler)


def unregister():
    import bpy
    for h in list(bpy.app.handlers.load_post):
        if getattr(h, "__name__", "") == "forget" and getattr(h, "__module__", "") == __name__:
            bpy.app.handlers.load_post.remove(h)
