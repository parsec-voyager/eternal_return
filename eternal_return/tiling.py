"""Polygon and mesh helpers the surface engine's floors and levels (surface.py, surface_substitution.py, motif.py) and
the 3D parts (parts.py) share: convex polygons cut by half planes (clip_half_planes()), polygons kept as padded corner
arrays (polygons_of(), corners_of(), take_corners()), huge polygons cut near the camera for depth precision
(split_near()), a piece extruded into a solid (solid_faces()), and a mesh written straight into Blender's attribute
arrays (write_mesh()).
"""

import numpy as np

# The engine in Rust (the eternal_return_core wheel's), for what's been moved there. A core that doesn't have a
# function yet (an older wheel, or none) leaves the Python here to do it: the same arrays either way.
try:
    from eternal_return_core import _native
except ImportError:
    _native = None


def clip_half_planes(points, counts, normals, offsets):
    """Convex polygons (n, corners, d: points of any dimension) each cut to normal . p <= offset (its own normal and
    offset), keeping order"""
    n, width = points.shape[:2]
    if n == 0:
        return points, counts
    k = np.arange(width)
    valid = k < counts[:, None]
    previous = take_corners(points, (k - 1) % np.maximum(counts, 1)[:, None])
    # Padding corners past a polygon's count hold anything (NaN, inf): computed and thrown away, quietly
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        f = (points * normals[:, None, :]).sum(axis=2) - offsets[:, None]
        g = (previous * normals[:, None, :]).sum(axis=2) - offsets[:, None]
        inside, was_inside = f <= 0, g <= 0
        t = g / (g - f)
        crossing = previous + t[..., None] * (points - previous)
    stacked = np.stack([crossing, points], axis=2).reshape(n, 2 * width, points.shape[2])
    flags = np.stack([valid & (inside != was_inside), valid & inside], axis=2).reshape(n, 2 * width)
    return compact_corners(stacked, flags)


def polygons_of(corners, sizes):
    """Polygons given as their corners in a row (m, d) and how many each has, as split_near() takes them: (points (n,
    widest, d), counts)"""
    sizes = np.asarray(sizes, dtype=np.int64)
    width = int(sizes.max()) if len(sizes) else 3
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    index = starts[:, None] + np.minimum(np.arange(width), np.maximum(sizes - 1, 0)[:, None])
    return np.asarray(corners)[index], sizes


def corners_of(points, counts):
    """The other way round: (corners in a row, counts)"""
    valid = np.arange(points.shape[1]) < counts[:, None]
    return points[valid], counts


# split_near(): a polygon is cut once it's more than this many times bigger than its distance from the middle (or the
# tile's size near it)
SPLIT_RATIO = 4.0


def split_near(points, counts, middle, tile, ratio=SPLIT_RATIO):
    """split_near_here() by the core (Rust: some twenty times faster, and without Python's lock), when it has it and
    the arrays are what it takes"""
    cut = getattr(_native, "split_near", None)
    if cut is None or points.dtype != np.float64 or counts.dtype != np.int64 or points.ndim != 3 or points.shape[2] < 3:
        return split_near_here(points, counts, middle, tile, ratio)
    width, d = points.shape[1:]
    pieces, width, piece_counts, rows = cut(np.ascontiguousarray(points), np.ascontiguousarray(counts), width, d,
                                            tuple(float(m) for m in middle), float(tile), float(ratio))
    return (np.frombuffer(pieces, dtype=np.float64).reshape(-1, width, d),
            np.frombuffer(piece_counts, dtype=np.int64), np.frombuffer(rows, dtype=np.int64))


def split_near_here(points, counts, middle, tile, ratio=SPLIT_RATIO):
    """Convex polygons (n, corners, d: their first three coordinates a place, any after carried along) that are more
    than ratio times bigger than their distance from middle (or tile, near it), cut on the planes at middle +- tile *
    2^k along each axis: tiles() in three dimensions, for surfaces that fold (a mesh's levels and floor). A polygon's
    depth near the camera is interpolated from its corners, in float32 from its object's middle, so one reaching tens of
    thousands of units off (a coarse level's band holding the whole view, a floor cell a quarter of the window across)
    had it wrong near the camera by about as much as the layers of finer levels on top are apart, and they z-fought: a
    giant band flickered against the floor as the camera moved. Cut so, each piece is no more than about its distance
    from the camera across. Returns (points, counts, rows: the polygon each piece is of)."""
    n, width = points.shape[:2]
    rows = np.arange(n)
    if n == 0:
        return points, counts, rows
    middle = np.asarray(middle, dtype=float)
    valid = (np.arange(width) < counts[:, None])[..., None]
    with np.errstate(invalid="ignore"):
        place = points[..., :3] - middle
        low = np.where(valid, place, np.inf).min(axis=1)
        high = np.where(valid, place, -np.inf).max(axis=1)
    extent = (high - low).max(axis=1)
    bound = np.maximum(tile, np.maximum(low, -high).max(axis=1))
    big = np.isfinite(extent) & (extent > ratio * bound)
    if not big.any():
        return points, counts, rows
    reach = float(np.abs(np.concatenate([low[big], high[big]])).max())
    steps = [tile]
    while steps[-1] < reach:
        steps.append(2.0 * steps[-1])
    cuts = sorted([-s for s in steps] + [0.0] + steps)
    work, work_counts, work_rows = points[big], counts[big], rows[big]
    for axis in range(3):
        for c in cuts:
            at = middle[axis] + c
            w_valid = np.arange(work.shape[1]) < work_counts[:, None]
            with np.errstate(invalid="ignore"):
                w_low = np.where(w_valid, work[..., axis], np.inf).min(axis=1)
                w_high = np.where(w_valid, work[..., axis], -np.inf).max(axis=1)
            across = np.flatnonzero((w_low < at) & (w_high > at))
            if not len(across):
                continue
            normal = np.zeros((len(across), work.shape[2]))
            normal[:, axis] = 1.0
            below, below_counts = clip_half_planes(work[across], work_counts[across], normal, np.full(len(across), at))
            above, above_counts = clip_half_planes(work[across], work_counts[across], -normal,
                                                   np.full(len(across), -at))
            keep = np.setdiff1d(np.arange(len(work)), across)
            size = max(work.shape[1], below.shape[1], above.shape[1])
            pad = lambda a: np.pad(a, ((0, 0), (0, size - a.shape[1]), (0, 0)))
            work = np.concatenate([pad(work[keep]), pad(below), pad(above)])
            work_counts = np.concatenate([work_counts[keep], below_counts, above_counts])
            work_rows = np.concatenate([work_rows[keep], work_rows[across], work_rows[across]])
            good = work_counts >= 3
            work, work_counts, work_rows = work[good], work_counts[good], work_rows[good]
    size = max(width, work.shape[1])
    pad = lambda a: np.pad(a, ((0, 0), (0, size - a.shape[1]), (0, 0)))
    small = ~big
    return (np.concatenate([pad(points[small]), pad(work)]), np.concatenate([counts[small], work_counts]),
            np.concatenate([rows[small], work_rows]))


def take_corners(points, indices):
    """points (n, corners, 2) rearranged along each row by indices (n, m)"""
    return points[np.arange(len(points))[:, None], indices]


def compact_corners(points, flags):
    """The corners of each row of points (n, corners, 2 or 3) that flags (n, corners) keeps, moved to its start in
    order, as (points, counts); past each row's count, zeros. Each kept corner goes straight to its place (how many kept
    ones come before it): a stable argsort of every row and take_along_axis took twice as long, and most of a girih
    level's split (tiles())."""
    counts = flags.sum(axis=1)
    out = np.zeros((len(points), max(1, int(counts.max(initial=0))), points.shape[2]))
    rows, columns = np.nonzero(flags)
    out[rows, (np.cumsum(flags, axis=1) - 1)[rows, columns]] = points[rows, columns]
    return out, counts


def solid_faces(polygon, axes="xy", low=0.0, high=1.0, open_faces=()):
    """A piece's faces in its own units, for a mesh of its own (parts.py instances them): a simple polygon of any shape
    (an arch's outline too), its corners in the two axes named (e.g. "xz" for an outline standing up), extruded along
    the third from low to high: its sides, and its caps unless open ("-y", "+y" and so on), wound to face out whichever
    way the polygon was given. As (corners (n, 3), corners per face (m,))."""
    a, b = ("xyz".index(axes[0]), "xyz".index(axes[1]))
    k = 3 - a - b
    polygon = np.asarray(polygon, dtype=float)
    area = np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) - np.roll(polygon[:, 0], -1) * polygon[:, 1])
    # Counterclockwise looking down the extrusion axis, so its sides face out and its high cap up it
    if (area < 0) == ((a, b, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1))):
        polygon = polygon[::-1]
    count = len(polygon)
    rings = np.zeros((2, count, 3))
    rings[:, :, a], rings[:, :, b] = polygon[:, 0], polygon[:, 1]
    rings[0, :, k], rings[1, :, k] = low, high
    following = np.roll(np.arange(count), -1)
    sides = np.stack([rings[0], rings[0][following], rings[1][following], rings[1]], axis=1).reshape(-1, 3)
    corners, sizes = [sides], [np.full(count, 4)]
    axis = "xyz"[k]
    if "+" + axis not in open_faces:
        corners.append(rings[1])
        sizes.append([count])
    if "-" + axis not in open_faces:
        corners.append(rings[0][::-1])
        sizes.append([count])
    return np.concatenate(corners), np.concatenate(sizes).astype(np.int64)


def write_mesh(obj, corners, sizes, colors, normals=None, convex=False):
    """obj's mesh as faces, each with its own corners (a face's in a row), colored per face by "Col", which Solid mode
    shows, with normals as Blender's free custom normals if given.

    Corners and the corner list go straight into their attribute arrays, and the faces get only their starts (each
    face's corner count follows from the next one's start): through vertices.co, loops.vertex_index and loop_total,
    a girih level of 10k faces took 6.9 ms, which a dive spent some 30 times a second, and 1.4 ms this way, the same
    mesh (~/blender_tests/girih/perf/write_mesh_equal.py).

    What Blender does with a written mesh before it's drawn is kept short too (a tenth of the main thread
    in a dive, with some 20 levels written a second):
    - Without normals given, the faces are flat ("sharp_face"): each has its own corners, so it looks the same, but
      for a smooth mesh Blender works out every vertex's normal (and which faces meet at it, for that).
    - With convex, faces of five corners and more are written as quadrilaterals (and a triangle, for an odd one)
      fanned from their first corner, sharing its corners: Blender cuts a polygon of five corners or more into
      triangles by a general method (for any outline), ten times the work of a quadrilateral's. Only for polygons
      known to be convex, and flat.
    - "Col" is a color per vertex (a face's corners are its own vertices), so the colors don't depend on how the
      faces are cut: a caller's array of one value per corner still fits ("Col", or its own attributes on points)."""
    if obj is None or obj.type != 'MESH':
        return
    mesh = obj.data
    count = len(corners)
    sizes = np.asarray(sizes, dtype=np.int64)
    starts = np.cumsum(sizes) - sizes
    corner_vert = np.arange(count, dtype=np.int32)
    face_starts = starts
    if convex and normals is None and len(sizes) and int(sizes.max()) > 4:
        # Polygon f in (sizes - 1) // 2 pieces: piece j has its corners 0, 2j + 1, 2j + 2 and, unless that's past its
        # last, 2j + 3
        pieces = (sizes - 1) // 2
        polygon = np.repeat(np.arange(len(sizes)), pieces)
        j = np.arange(len(polygon)) - np.repeat(np.cumsum(pieces) - pieces, pieces)
        first = starts[polygon]
        four = 2 * j + 3 < sizes[polygon]
        piece_sizes = np.where(four, 4, 3)
        face_starts = np.cumsum(piece_sizes) - piece_sizes
        corner_vert = np.empty(int(piece_sizes.sum()), dtype=np.int32)
        corner_vert[face_starts] = first
        corner_vert[face_starts + 1] = first + 2 * j + 1
        corner_vert[face_starts + 2] = first + 2 * j + 2
        corner_vert[face_starts[four] + 3] = (first + 2 * j + 3)[four]
    mesh.clear_geometry()
    mesh.vertices.add(count)
    mesh.loops.add(len(corner_vert))
    mesh.polygons.add(len(face_starts))
    if count:
        mesh.attributes["position"].data.foreach_set("vector", np.asarray(corners, dtype=np.float32).ravel())
        mesh.attributes[".corner_vert"].data.foreach_set("value", corner_vert)
    mesh.polygons.foreach_set("loop_start", face_starts.astype(np.int32))
    attribute = mesh.color_attributes.get("Col") or mesh.color_attributes.new("Col", 'FLOAT_COLOR', 'POINT')
    if count:
        colors = np.repeat(np.asarray(colors, dtype=np.float32).reshape(-1, 4), sizes, axis=0)
        attribute.data.foreach_set("color", colors.ravel())
    mesh.color_attributes.active_color = attribute
    mesh.color_attributes.render_color_index = 0
    if normals is not None:
        custom = mesh.attributes.get("custom_normal") or mesh.attributes.new("custom_normal", 'FLOAT_VECTOR', 'CORNER')
        if count:
            custom.data.foreach_set("vector", np.asarray(normals, dtype=np.float32).ravel())
    elif len(face_starts):
        sharp = mesh.attributes.get("sharp_face") or mesh.attributes.new("sharp_face", 'BOOLEAN', 'FACE')
        sharp.data.foreach_set("value", np.ones(len(face_starts), dtype=bool))
    mesh.update()
