"""A shape laid out as a map's mesh, once, when the map is built (phase 7 of the engines' plan, 2026-11-24). The flight
code only splits a mesh's faces and evaluates its surface (surface.py; subdivision.py for a smooth one), so what makes
a map's cells square on its shape (its faces' layout) and its smooth surface the shape's (its points) is worked out
here, before flying, never per point in flight.

laid_out(): any object's shape as a new mesh of about so many quads with corners of other than 4 faces where the shape
needs them (Blender's QuadriFlow, which lines its quads up with the shape's curvature), squared() on the shape, and for
a smooth chart, fit().

squared(): a quad mesh's corners slid over the shape until its quads are as near square as they get, each free to be
its own size (conformal). QuadriFlow makes its quads all one size, so round a bend (a tube's inside against its outside)
they stretch, 15% at the median on a knotted tube; and a cell's shape where the drone dives is its face's at that
point at every depth below (its splits are alike), so stretched there is stretched all the way down.

fit(): a mesh's points moved so its smooth surface passes through the places wanted at its vertices. A subdivision
surface's points are blends of its mesh's (positive weights), so it lies inside its mesh: a mesh laid on a shape has a
smooth surface shrunk inside the shape, most where it bends most (a face's side times the curvature, squared).
"""

import numpy as np

# How many faces laid_out() lays a shape out in by default: the map's biggest tiles (each splits forever below them;
# fewer faces make them bigger, and the smooth surface follows the shape less closely)
LAYOUT_FACES = 1000
# A shape QuadriFlow can't lay out as it is (an edge of three faces or more, or overlapping parts to be one, merge) is
# first made whole by voxels this share of its size across
VOXEL_SHARE = 1.0 / 256.0
# ... and its vertices this share of its size apart or nearer welded first (welded())
WELD_SHARE = 1e-6
# squared()'s rounds at most, and when it stops sooner: its corners moving less than this share of the shape's size; each
# quad's size stays within this range of what it started at (toward a corner of 3 or 5 faces conformal cells shrink or
# grow without end: unbounded, Suzanne's shrank to nothing in 400 rounds)
SQUARE_ROUNDS = 400
SQUARE_STILL = 1e-6
SQUARE_SIZES = (0.5, 2.0)
# The unit square's corners, counterclockwise from its middle, as complex numbers
SQUARE = np.array([-0.5 - 0.5j, 0.5 - 0.5j, 0.5 + 0.5j, -0.5 + 0.5j])

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


def squared(points, quads, onto, fixed=None):
    """points (vertices, 3) of quads (faces, 4), moved until each quad is near the square nearest it (any size, turned
    any way in its plane): each round, every corner goes to the mean of its quads' squares' corners there, then onto the
    shape (onto: points to the shape's nearest points); fixed corners (an open edge's) stay. Returns the points."""
    points = np.array(points, dtype=float)
    quads = np.asarray(quads, dtype=np.int64)
    if not len(quads):
        return points
    shares = np.bincount(quads.ravel(), minlength=len(points)).astype(float)
    free = shares > 0
    if fixed is not None:
        free &= ~np.asarray(fixed, dtype=bool)
    size = float(np.ptp(points, axis=0).max()) or 1.0
    first = None
    for _round in range(SQUARE_ROUNDS):
        corners = points[quads]
        middle = corners.mean(axis=1)
        # Each quad's plane: its normal (across its diagonals) and an axis along its first and third sides
        normal = np.cross(corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 1])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        axis = corners[:, 1] - corners[:, 0] + corners[:, 2] - corners[:, 3]
        axis -= np.einsum("fk,fk->f", axis, normal)[:, None] * normal
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        other = np.cross(normal, axis)
        offset = corners - middle[:, None]
        flat = np.einsum("fck,fk->fc", offset, axis) + 1j * np.einsum("fck,fk->fc", offset, other)
        # The nearest square: the unit square scaled and turned by z, least squares over the corners
        z = flat @ np.conj(SQUARE) / float(np.sum(np.abs(SQUARE) ** 2))
        if first is None:
            first = np.abs(z)
        z *= np.clip(np.abs(z), SQUARE_SIZES[0] * first, SQUARE_SIZES[1] * first) / np.abs(z)
        square = z[:, None] * SQUARE
        wanted = middle[:, None] + square.real[..., None] * axis[:, None] + square.imag[..., None] * other[:, None]
        sums = np.zeros_like(points)
        np.add.at(sums, quads.ravel(), wanted.reshape(-1, 3))
        moved = points.copy()
        moved[free] = onto(sums[free] / shares[free, None])
        step = float(np.abs(moved - points).max())
        points = moved
        if step <= SQUARE_STILL * size:
            break
    return points


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


def laid_out(obj, faces=LAYOUT_FACES, chart="catmull_clark", merge=False, seed=0):
    """obj's shape (as drawn, in world coordinates) laid out as a new mesh of about faces quads (see the module's
    docstring): voxels first if QuadriFlow can't take it as it is, or merge (overlapping parts made one); its corners
    moved onto the shape (QuadriFlow's are near it), then for a smooth chart ("catmull_clark") fit() so its surface
    passes through them. Returns (the mesh, a dict of what was done: faces, corners by how many faces meet there,
    the fit's miss and rounds, whether voxels were used)."""
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
            result = bpy.ops.object.quadriflow_remesh(target_faces=int(faces), mode='FACES', seed=seed,
                                                      use_mesh_symmetry=False, use_preserve_sharp=False,
                                                      use_preserve_boundary=True, smooth_normals=False)
        laid = work.data
        if 'FINISHED' not in result or not len(laid.polygons):
            raise ValueError(f"layout: QuadriFlow couldn't lay {obj.name} out ({result})")
        # Its corners onto the shape (the object's own, not the voxels'), then squared there, an open edge's corners
        # kept on it
        tree = BVHTree.FromPolygons([tuple(p) for p in points.tolist()], [tuple(p.vertices) for p in shape.polygons])
        onto = lambda places: np.array([tree.find_nearest(p)[0] for p in places.tolist()], dtype=np.float64)
        faces = [tuple(p.vertices) for p in laid.polygons]
        wanted = onto(points_of(laid))
        quads = np.array([f for f in faces if len(f) == 4], dtype=np.int64).reshape(-1, 4)
        wanted = squared(wanted, quads, onto, fixed=open_corners(laid))
        top = surface.Topology(None, positions=wanted, faces=faces, chart=chart)
        placed, miss, rounds = fit(top.chart, wanted)
        laid.vertices.foreach_set("co", placed.astype(np.float32).ravel())
        laid.update()
        valences = np.bincount([len(vf) for vf in top.vertex_faces])
        done = {"faces": len(laid.polygons), "corners": {k: int(c) for k, c in enumerate(valences) if c and k != 4},
                "fit_miss": miss, "fit_rounds": rounds, "voxels": voxels}
        work.data = bpy.data.meshes.new("_er_layout_empty")
        return laid, done
    finally:
        mesh = work.data
        bpy.data.objects.remove(work)
        bpy.data.meshes.remove(mesh)
        bpy.data.meshes.remove(shape)
