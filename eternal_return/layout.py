"""A shape laid out as a map's mesh, once, when the map is built (phase 7 of the engines' plan, 2026-11-24). The flight
code only splits a mesh's faces and evaluates its surface (surface.py; subdivision.py for a smooth one), so what makes
a map's cells square on its shape (its faces' layout) and its smooth surface the shape's (its points) is worked out
here, before flying, never per point in flight.

laid_out(): any object's shape as a new mesh of about so many quads with corners of other than 4 faces where the shape
needs them (Blender's QuadriFlow, which lines its quads up with the shape's curvature), or of about so many triangles
with corners of other than 6 where it needs them (triangulated(), from QuadriFlow's quads), regularized() on the
shape, and for a smooth chart, fit().

A shape with one hole (a torus, a knotted tube, a mug) is laid out conformally instead (ringed(), ring_layout.py): a
regular lattice of squares or equilateral triangles, every corner regular, only the tiles' size changing.

triangulated(): a quad mesh's quads each cut in two triangles, the diagonals leaning alike, so a patch of quads with
4 at each corner makes triangles with 6 at each (QuadriFlow lays out quads only); then in turn regularized() and
edges flipped where two triangles have gone far from Delaunay's (the angles facing their shared edge over 180
degrees): only where the cut grid's shear can't close up round the shape (a torus's tube) does it take a pair of
corners of 5 and 7. A corner of other than 6 is a corner of Loop's surface's that isn't regular (slower to work out,
and a star in the floor's pattern).

regularized(): a mesh's corners slid over the shape until its faces are as near regular (square, equilateral) as they
get, each free to be its own size (conformal). QuadriFlow makes its quads all one size, so round a bend (a tube's inside
against its outside) they stretch, 15% at the median on a knotted tube; and a cell's shape where the drone dives is its
face's at that point at every depth below (its splits are alike), so stretched there is stretched all the way down.

fit(): a mesh's points moved so its smooth surface passes through the places wanted at its vertices. A subdivision
surface's points are blends of its mesh's (positive weights), so it lies inside its mesh: a mesh laid on a shape has a
smooth surface shrunk inside the shape, most where it bends most (a face's side times the curvature, squared).
"""

import math

import numpy as np

# How many faces laid_out() lays a shape out in by default: the map's biggest tiles (each splits forever below them;
# fewer faces make them bigger, and the smooth surface follows the shape less closely)
LAYOUT_FACES = 1000
# A shape QuadriFlow can't lay out as it is (an edge of three faces or more, or overlapping parts to be one, merge) is
# first made whole by voxels this share of its size across
VOXEL_SHARE = 1.0 / 256.0
# ... and its vertices this share of its size apart or nearer welded first (welded())
WELD_SHARE = 1e-6
# regularized()'s rounds at most, and when it stops sooner: its corners moving less than this share of the shape's size;
# each face's size stays within this range of what it started at (toward a corner of 3 or 5 faces conformal cells
# shrink or grow without end: unbounded, Suzanne's shrank to nothing in 400 rounds)
REGULAR_ROUNDS = 400
REGULAR_STILL = 1e-6
REGULAR_SIZES = (0.5, 2.0)
# The regular polygon of each number of sides, its corners counterclockwise round its middle, as complex numbers
REGULAR = {n: np.exp(2j * np.pi * np.arange(n) / n) for n in (3, 4)}

# triangulated()'s turns of regularized() (so many rounds each) and flips, and how far past 180 degrees the angles
# facing an edge go before it's flipped. Measured 2026-11-26 (corners of other than 6; the angles' median off 60):
# Botsch and Kobbelt's isotropic remeshing (long edges split, short ones collapsed, flips toward 6, relaxing) made a
# torus 98 odd corners, 4.0 degrees (a quad's diagonal is 1.32 times the side of the equilateral triangle of its
# half's area, at their split threshold, 4/3); the aligned cuts with flips toward 6, 9 and 19 degrees (the sheared
# grid couldn't close round the tube); these flips 12 and 6.6 (cube 10 and 1.4, Suzanne 106 and 5.4: most of its odd
# corners its quads' own)
TRIANGLE_TURNS = 6
TRIANGLE_ROUNDS = 60
FLIP_PAST = math.radians(10.0)

# fit() stops when every limit point is this close to where it's wanted (times the mesh's size), or after so many
# rounds (each closes the gap by a ninth at least: the limit masks' weights are positive and sum to 1, and a regular
# vertex's damp the finest wiggle, a point up and its neighbors down, to a ninth)
FIT_TOLERANCE = 1e-9
FIT_ROUNDS = 1000


def limit_points(masks, points):
    """The limit points of points by masks (chart.limit_masks()'s), each vertex on no face where it is"""
    rows, columns, weights = masks
    out = np.array(points, dtype=float)
    if len(rows):
        starts = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])
        out[rows[starts]] = np.add.reduceat(weights[:, None] * out[columns], starts, axis=0)
    return out


def fit(chart, wanted):
    """Points (vertices, 3) whose limit points by chart (a subdivision chart of the mesh: its limit_masks()) are wanted
    (vertices, 3), by rounds that move each point by how far its limit point misses: (points, the worst miss over the
    mesh's size, rounds taken). A flat-faced (bilinear) chart's surface passes through its points: they're wanted."""
    wanted = np.asarray(wanted, dtype=float)
    if not hasattr(chart, "limit_masks"):
        return wanted.copy(), 0.0, 0
    masks = chart.limit_masks()
    size = float(np.ptp(wanted, axis=0).max()) or 1.0
    points = wanted.copy()
    worst = np.inf
    for rounds in range(FIT_ROUNDS + 1):
        miss = wanted - limit_points(masks, points)
        last, worst = worst, float(np.abs(miss).max()) / size
        if worst <= FIT_TOLERANCE:
            break
        if worst > 2.0 * last:
            raise ArithmeticError(f"layout.fit: its rounds grew the miss ({last:.3g} to {worst:.3g} of the size)")
        points += miss
    return points, worst, rounds


def regularized(points, faces, onto, fixed=None, rounds=REGULAR_ROUNDS):
    """points (vertices, 3) of faces (tuples of 3 or 4 vertices), moved until each face is near the regular polygon
    nearest it (any size, turned any way in its plane): each round, every corner goes to the mean of its faces' regular
    polygons' corners there, then onto the shape (onto: points to the shape's nearest points); fixed corners (an open
    edge's) stay. Returns the points."""
    points = np.array(points, dtype=float)
    groups = {n: np.array([f for f in faces if len(f) == n], dtype=np.int64).reshape(-1, n) for n in REGULAR}
    groups = {n: rows for n, rows in groups.items() if len(rows)}
    if not groups:
        return points
    shares = np.zeros(len(points))
    for rows in groups.values():
        shares += np.bincount(rows.ravel(), minlength=len(points))
    free = shares > 0
    if fixed is not None:
        free &= ~np.asarray(fixed, dtype=bool)
    size = float(np.ptp(points, axis=0).max()) or 1.0
    first = {}
    for _round in range(rounds):
        sums = np.zeros_like(points)
        for n, rows in groups.items():
            corners = points[rows]
            middle = corners.mean(axis=1)
            # Each face's plane: its normal (Newell's: the sum of its sides' crosses round its middle) and an axis
            # along its first side
            offset = corners - middle[:, None]
            normal = np.cross(offset, np.roll(offset, -1, axis=1)).sum(axis=1)
            normal /= np.linalg.norm(normal, axis=1, keepdims=True)
            axis = corners[:, 1] - corners[:, 0]
            axis -= np.einsum("fk,fk->f", axis, normal)[:, None] * normal
            axis /= np.linalg.norm(axis, axis=1, keepdims=True)
            other = np.cross(normal, axis)
            flat = np.einsum("fck,fk->fc", offset, axis) + 1j * np.einsum("fck,fk->fc", offset, other)
            # The nearest regular polygon: the unit one scaled and turned by z, least squares over the corners
            shape = REGULAR[n]
            z = flat @ np.conj(shape) / float(np.sum(np.abs(shape) ** 2))
            if n not in first:
                first[n] = np.abs(z)
            z *= np.clip(np.abs(z), REGULAR_SIZES[0] * first[n], REGULAR_SIZES[1] * first[n]) / np.abs(z)
            polygon = z[:, None] * shape
            wanted = middle[:, None] + polygon.real[..., None] * axis[:, None] + polygon.imag[..., None] * other[:, None]
            np.add.at(sums, rows.ravel(), wanted.reshape(-1, 3))
        moved = points.copy()
        moved[free] = onto(sums[free] / shares[free, None])
        step = float(np.abs(moved - points).max())
        points = moved
        if step <= REGULAR_STILL * size:
            break
    return points


def triangulated(mesh, onto):
    """mesh's quads cut in two triangles each, the diagonals leaning alike (_cut_quads()), then in turn regularized()
    and Delaunay's flips (_flip_far()), as the module's docstring says. Its corners are kept on the shape (onto:
    points to the shape's nearest points), an open edge's where they are."""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(mesh)
    _cut_quads(bm)
    for _turn in range(TRIANGLE_TURNS):
        bm.verts.ensure_lookup_table()
        bm.verts.index_update()
        points = np.array([v.co for v in bm.verts], dtype=np.float64)
        faces = [tuple(v.index for v in f.verts) for f in bm.faces]
        fixed = np.array([v.is_boundary for v in bm.verts])
        points = regularized(points, faces, onto, fixed=fixed, rounds=TRIANGLE_ROUNDS)
        for v, p in zip(bm.verts, points.tolist()):
            v.co = p
        bm.normal_update()
        _flip_far(bm)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def _cut_quads(bm):
    """Every quad cut in two triangles along a diagonal leaning the way its neighbors' do (across a shared side, the
    two quads' diagonals meet its opposite ends): a patch of quads with 4 at each corner makes a patch of triangles
    with 6 at each corner (cut each along its own best diagonal, Blender's beauty, a torus came out with 86 corners
    of 5 or 7, none needed). Where the leaning can't agree all round (round a corner of 3 or 5 quads), a quad takes
    the first it's reached by. Other faces by Blender's beauty."""
    import bmesh
    from collections import deque

    def across(face, x):
        verts = list(face.verts)
        return verts[(verts.index(x) + 2) % 4]
    choice = {}
    for start in [f for f in bm.faces if len(f.verts) == 4]:
        if start in choice:
            continue
        choice[start] = start.verts[0]
        queue = deque([start])
        while queue:
            face = queue.popleft()
            ends = {choice[face], across(face, choice[face])}
            for edge in face.edges:
                a, b = edge.verts
                for other in edge.link_faces:
                    if other is not face and len(other.verts) == 4 and other not in choice:
                        choice[other] = b if a in ends else a
                        queue.append(other)
    for face, x in choice.items():
        bmesh.utils.face_split(face, x, across(face, x))
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3], quad_method='BEAUTY',
                          ngon_method='BEAUTY')


def _flip_far(bm):
    """Edges flipped where the two angles facing them come to more than 180 degrees and FLIP_PAST (the new edge inside
    the two triangles' quad: their far corners on either side of it, seen along their normals)"""
    import bmesh
    for e in list(bm.edges):
        if not e.is_valid or len(e.link_faces) != 2 or e.is_boundary:
            continue
        f, g = e.link_faces
        if len(f.verts) != 3 or len(g.verts) != 3:
            continue
        a, b = e.verts
        c = next(v for v in f.verts if v is not a and v is not b)
        d = next(v for v in g.verts if v is not a and v is not b)
        if c is d or any(x.other_vert(c) is d for x in c.link_edges):
            continue
        if (a.co - c.co).angle(b.co - c.co, 0.0) + (a.co - d.co).angle(b.co - d.co, 0.0) <= math.pi + FLIP_PAST:
            continue
        up = f.normal + g.normal
        if up.length < 1e-9:
            continue
        across = d.co - c.co
        heights = [across.cross(v.co - c.co).dot(up) for v in (a, b)]
        if heights[0] * heights[1] >= 0.0:
            continue
        bmesh.utils.edge_rotate(e, True)
    bm.normal_update()


def corner_rhombs(mesh, smooth=True):
    """mesh's faces each cut into a quad at each corner (the corner, its two sides' middles, the face's middle), each
    labeled a thick Penrose rhomb: a Penrose map's mesh (surface_substitution.KIND, its loops (B, A, C, A'): B the
    corner, C the middle, counterclockwise as the face). Penrose's matching rules then agree across every edge,
    whatever the mesh (~/blender_tests/penrose_surface/rhomb_mesh.py, 2026-11-13), with special points where the
    rhombs' angles don't come to 360 degrees: a corner of v faces 72 v, a face's middle 72 times its sides, a side's
    middle 432 (an open side's, 216). On a smooth surface its points are Catmull-Clark's first split's (an open edge's
    by its rim's B-spline, as subdivision.LocalMesh splits), so its surface is the mesh's own (a quad mesh's exactly);
    else the faces' and sides' own middles. Returns (the new mesh, its vertices that aren't special points)."""
    import bpy
    try:
        from . import surface_substitution as SS
        from . import penrose
    except ImportError:
        import surface_substitution as SS
        import penrose
    P = points_of(mesh)
    faces = [tuple(p.vertices) for p in mesh.polygons]
    middles = np.array([P[list(f)].mean(axis=0) for f in faces])
    sides = {}
    for t, f in enumerate(faces):
        for s in range(len(f)):
            a, b = f[s], f[(s + 1) % len(f)]
            sides.setdefault((min(a, b), max(a, b)), []).append(t)
    side_keys = list(sides)
    side_id = {key: len(P) + k for k, key in enumerate(side_keys)}
    faces_at = faces_of_vertex(faces, len(P))
    opened = open_vertices(sides)
    if smooth:
        side_points = np.array([(P[a] + P[b] + middles[ts[0]] + middles[ts[1]]) / 4.0 if len(ts) == 2
                                else (P[a] + P[b]) / 2.0 for (a, b), ts in sides.items()])
        corners = P.copy()
        rim = {}
        for (a, b), ts in sides.items():
            if len(ts) == 1:
                rim.setdefault(a, []).append(b)
                rim.setdefault(b, []).append(a)
        for x, ts in enumerate(faces_at):
            if not ts:
                continue
            if x in rim:
                if len(rim[x]) == 2:
                    p, q = rim[x]
                    corners[x] = (P[p] + 6.0 * P[x] + P[q]) / 8.0
                continue
            # (Q + 2 R + (n - 3) P) / n: Q the faces' middles' mean, R its sides' middles' (each side once)
            n = len(ts)
            near = {y for t in ts for y in faces[t] if (min(x, y), max(x, y)) in sides and y != x}
            Q = middles[ts].mean(axis=0)
            R = np.mean([(P[x] + P[y]) / 2.0 for y in near], axis=0)
            corners[x] = (Q + 2.0 * R + (n - 3.0) * P[x]) / n
    else:
        side_points = np.array([(P[a] + P[b]) / 2.0 for a, b in side_keys])
        corners = P
    middle_id = len(P) + len(side_keys)
    points = np.concatenate([corners, side_points, middles])
    quads = []
    for t, f in enumerate(faces):
        n = len(f)
        for s in range(n):
            x, after, before = f[s], f[(s + 1) % n], f[(s - 1) % n]
            quads.append((x, side_id[(min(x, after), max(x, after))], middle_id + t,
                          side_id[(min(before, x), max(before, x))]))
    rhombs = bpy.data.meshes.new(mesh.name)
    rhombs.from_pydata(points.tolist(), [], quads)
    rhombs.update()
    if [tuple(p.vertices) for p in rhombs.polygons] != quads:
        raise ValueError("layout: the rhombs' corners came out in another order")
    kinds = rhombs.attributes.new(SS.KIND, 'INT', 'FACE')
    kinds.data.foreach_set("value", [penrose.THICK] * len(quads))
    # The flat points: a corner of 5 faces (5 x 72), a pentagon's middle
    flat = [x for x in range(len(P)) if len(faces_at[x]) == 5 and x not in opened]
    flat += [middle_id + t for t, f in enumerate(faces) if len(f) == 5]
    return rhombs, flat


def faces_of_vertex(faces, count):
    """Each vertex's faces (lists of face indices), for count vertices"""
    out = [[] for _ in range(count)]
    for t, f in enumerate(faces):
        for x in f:
            out[x].append(t)
    return out


def open_vertices(sides):
    """The vertices on an open side (sides: (least, greatest) vertex -> its faces)"""
    return {x for key, ts in sides.items() if len(ts) == 1 for x in key}


def ringed(mesh, places, count, onto, triangles=False):
    """mesh (QuadriFlow's layout, its corners at places on the shape) laid out again as ring_layout()'s regular
    lattice of about count corners if the shape has one hole (squares, or equilateral triangles: triangles); True if
    it was. The conformal map is worked out on QuadriFlow's mesh with each quad cut in four (the new corners put on
    the shape: onto), finer than the lattice: at the lattice's own size, the faces' chords across the bends bent it."""
    import bmesh
    try:
        from . import ring_layout
    except ImportError:
        import ring_layout
    bm = bmesh.new()
    bm.from_mesh(mesh)
    for v, p in zip(bm.verts, places.tolist()):
        v.co = p
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=1, use_grid_fill=True)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.verts.index_update()
    points = onto(np.array([v.co for v in bm.verts], dtype=np.float64))
    triangles_in = np.array([[v.index for v in f.verts] for f in bm.faces], dtype=np.int64)
    bm.free()
    found = ring_layout.ring_layout(points, triangles_in, count, "triangles" if triangles else "quads")
    if found is None:
        return False
    corners, faces = found
    mesh.clear_geometry()
    mesh.from_pydata(onto(corners).tolist(), [], faces.tolist())
    mesh.update()
    return True


def world_mesh(obj, depsgraph):
    """A new mesh of obj as it's drawn (its modifiers applied), in world coordinates"""
    import bpy
    mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph), depsgraph=depsgraph)
    mesh.transform(obj.matrix_world)
    return mesh


def manifold(mesh):
    """Whether every edge of mesh has one face or two, and every vertex a face (what QuadriFlow lays out as is)"""
    counts = np.zeros(len(mesh.edges), dtype=np.int64)
    edges = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("edge_index", edges)
    np.add.at(counts, edges, 1)
    used = np.zeros(len(mesh.vertices), dtype=bool)
    corners = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", corners)
    used[corners] = True
    return bool(len(mesh.polygons)) and bool(((counts >= 1) & (counts <= 2)).all()) and bool(used.all())


def open_corners(mesh):
    """Whether each vertex of mesh is on an edge of one face"""
    counts = np.zeros(len(mesh.edges), dtype=np.int64)
    edges = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("edge_index", edges)
    np.add.at(counts, edges, 1)
    ends = np.empty(2 * len(mesh.edges), dtype=np.int64)
    mesh.edges.foreach_get("vertices", ends)
    out = np.zeros(len(mesh.vertices), dtype=bool)
    out[ends.reshape(-1, 2)[counts == 1].ravel()] = True
    return out


def welded(mesh, distance):
    """mesh with its vertices closer than distance merged: a shape's pieces that only touch (a text's caps and sides,
    a mesh's unwelded seams) made one surface, not open edges QuadriFlow keeps as boundaries"""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=distance)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def points_of(mesh):
    co = np.empty(3 * len(mesh.vertices), dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    return co.reshape(-1, 3).astype(np.float64)


def laid_out(obj, faces=LAYOUT_FACES, chart="catmull_clark", merge=False, seed=0, triangles=False):
    """obj's shape (as drawn, in world coordinates) laid out as a new mesh of about faces quads, or with triangles,
    triangles (see the module's docstring): voxels first if QuadriFlow can't take it as it is, or merge (overlapping
    parts made one); its corners moved onto the shape (QuadriFlow's are near it), then for a smooth chart
    ("catmull_clark", or "loop" for triangles) fit() so its surface passes through them. Returns (the mesh, a dict of
    what was done: faces, corners by how many faces meet there other than a regular corner's, the fit's miss and
    rounds, whether voxels were used)."""
    import bpy
    from mathutils.bvhtree import BVHTree
    try:
        from . import surface
    except ImportError:
        import surface
    depsgraph = bpy.context.evaluated_depsgraph_get()
    shape = world_mesh(obj, depsgraph)
    points = points_of(shape)
    size = float(np.ptp(points, axis=0).max()) if len(points) else 0.0
    if not size:
        bpy.data.meshes.remove(shape)
        raise ValueError(f"layout: {obj.name} has no shape to lay out")
    welded(shape, WELD_SHARE * size)
    points = points_of(shape)
    work = bpy.data.objects.new("_er_layout", shape.copy())
    bpy.context.scene.collection.objects.link(work)
    try:
        voxels = merge or not manifold(shape)
        if voxels:
            remesh = work.modifiers.new("voxels", 'REMESH')
            remesh.mode, remesh.voxel_size = 'VOXEL', VOXEL_SHARE * size
            whole = bpy.data.meshes.new_from_object(work.evaluated_get(bpy.context.evaluated_depsgraph_get()))
            work.modifiers.remove(remesh)
            old, work.data = work.data, whole
            bpy.data.meshes.remove(old)
        with bpy.context.temp_override(object=work, active_object=work, selected_objects=[work],
                                       selected_editable_objects=[work]):
            # (Triangles: half as many quads, each about two of them)
            result = bpy.ops.object.quadriflow_remesh(target_faces=max(int(faces) // (2 if triangles else 1), 1),
                                                      mode='FACES', seed=seed,
                                                      use_mesh_symmetry=False, use_preserve_sharp=False,
                                                      use_preserve_boundary=True, smooth_normals=False)
        laid = work.data
        if 'FINISHED' not in result or not len(laid.polygons):
            raise ValueError(f"layout: QuadriFlow couldn't lay {obj.name} out ({result})")
        # Its corners onto the shape (the object's own, not the voxels'), then regularized there, an open edge's
        # corners kept on it
        tree = BVHTree.FromPolygons([tuple(p) for p in points.tolist()], [tuple(p.vertices) for p in shape.polygons])
        onto = lambda places: np.array([tree.find_nearest(p)[0] for p in places.tolist()], dtype=np.float64)
        wanted = onto(points_of(laid))
        # (A lattice of triangles has half as many corners as faces)
        ring = ringed(laid, wanted, int(faces) // (2 if triangles else 1), onto, triangles=triangles)
        if ring:
            wanted = points_of(laid)
        elif triangles:
            laid.vertices.foreach_set("co", wanted.astype(np.float32).ravel())
            triangulated(laid, onto)
            wanted = points_of(laid)
        faces = [tuple(p.vertices) for p in laid.polygons]
        wanted = regularized(wanted, faces, onto, fixed=open_corners(laid))
        top = surface.Topology(None, positions=wanted, faces=faces, chart=chart)
        placed, miss, rounds = fit(top.chart, wanted)
        laid.vertices.foreach_set("co", placed.astype(np.float32).ravel())
        laid.update()
        valences = np.bincount([len(vf) for vf in top.vertex_faces])
        regular = 6 if triangles else 4
        done = {"faces": len(laid.polygons),
                "corners": {k: int(c) for k, c in enumerate(valences) if c and k != regular},
                "fit_miss": miss, "fit_rounds": rounds, "voxels": voxels, "ring": ring}
        work.data = bpy.data.meshes.new("_er_layout_empty")
        return laid, done
    finally:
        mesh = work.data
        bpy.data.objects.remove(work)
        bpy.data.meshes.remove(mesh)
        bpy.data.meshes.remove(shape)
