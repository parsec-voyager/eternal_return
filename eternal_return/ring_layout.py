"""A shape with one hole (a torus, a twisted torus, a knotted tube, a mug) laid out as a perfectly regular lattice of
squares or equilateral triangles, conformally (phase 7, 2026-11-26). Such a surface is conformally a flat torus: the
plane modulo a lattice of two periods. Mapped there (its holomorphic 1-form, integrated), any lattice of the plane
that holds the periods comes back to it as a mesh with every corner regular (4 quads, 6 triangles), its tiles square
or equilateral wherever they are, only their size changing (smaller round a torus's inside, as the conformal factor
goes). QuadriFlow keeps its tiles one size, so round a torus it fits fewer of them on the inside, and every such step
is a corner of 3 or 5 quads (18 on a torus of r/R 0.3); cut into triangles, its square grid's shear can't close round
the tube, and the flips that ease it scatter corners of 5 and 7 (the user, 2026-11-26: "the torus topology seems a bit
off here, is there not a cleaner way").

The steps, on a triangle mesh of the shape (closed, one piece, one hole):
- tree-cotree (Eppstein): a spanning tree of the edges, one of the faces across the rest, and the two edges left over,
  one for each of the torus's two independent loops
- a closed 1-form for each (1 on its edge, 0 on the tree, the cotree's edges set so every face sums to 0), made
  harmonic (plus the gradient that minimizes its cotan energy: a Laplace solve)
- the conjugate of one (its face vectors turned a quarter about the normals), as a combination of the two: with it, a
  holomorphic 1-form, integrated along the tree into a complex coordinate z per corner, and its two periods
- a lattice of the shape wanted (squares, or equilateral triangles) holding the two periods as lattice vectors, about
  so many points in the torus: the periods' ratio is matched by a ratio of the ideal lattice's points, the
  rest is a slight linear stretch (as a torus builder's cell counts round its ring and tube are rounded)
- each lattice point placed where it falls in the mesh (in z), the lattice's own connections its faces

Pure numpy (no Blender): layout.laid_out() hands it QuadriFlow's mesh of the shape, cut into triangles.
"""

import math
from collections import deque

import numpy as np

# The Laplace solve (conjugate gradients): its relative residual, and its iterations at most
SOLVE_TOLERANCE = 1e-10
SOLVE_ROUNDS = 20000
# How far a lattice's point count may be from the count asked for (a share either way), to find the lattice nearest
# the shape wanted
COUNT_RANGE = (0.8, 1.25)
# A lattice with rows round the tube (the shorter period a row of its cells) is taken over a turned one fitting the
# periods better while its stretch is under this: a torus of r/R 0.3 fit by a turned lattice 0.1% off came out with
# lines spiralling round its tube, where rows round it were 0.6% off (rings round the tube, as a torus builder's)
ALIGNED_STRETCH = 0.03
# A lattice point on a face's side or corner is found in each face it touches: taken in the first, with this slack (in
# the face's barycentric coordinates)
INSIDE_SLACK = 1e-9


def genus_one(points, triangles):
    """Whether the triangle mesh is one closed piece with one hole: every edge has two faces, and V - E + F = 0"""
    triangles = np.asarray(triangles)
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    if not (counts == 2).all():
        return False
    used = np.unique(triangles)
    if len(used) != len(points):
        return False
    # One piece: a walk over the edges reaches every vertex
    neighbors = [[] for _ in range(len(points))]
    for a, b in unique.tolist():
        neighbors[a].append(b)
        neighbors[b].append(a)
    seen = {0}
    queue = deque([0])
    while queue:
        for y in neighbors[queue.popleft()]:
            if y not in seen:
                seen.add(y)
                queue.append(y)
    return len(seen) == len(points) and len(points) - len(unique) + len(triangles) == 0


def ring_layout(points, triangles, count, shape="triangles"):
    """The shape (points (V, 3), triangles (F, 3), counterclockwise from outside: one closed piece with one hole,
    genus_one()) laid out as about count corners of a regular lattice (shape "triangles": equilateral, "quads":
    square), conformally (see the module's docstring). Returns (corners (N, 3), faces: (2N, 3) triangles or (N, 4)
    quads, counterclockwise from outside), or None where it can't be done (not one hole, a degenerate mesh)."""
    P = np.asarray(points, dtype=np.float64)
    T = np.asarray(triangles, dtype=np.int64)
    if not genus_one(P, T):
        return None
    mesh = _Mesh(P, T)
    forms = [mesh.harmonic(mesh.closed_form(g)) for g in mesh.generators]
    if len(forms) != 2:
        return None
    phi = mesh.holomorphic(*forms)
    z, faces_z, periods = mesh.integrated(phi)
    if periods is None:
        return None
    lattice = _Lattice(periods, count, shape)
    return lattice.placed(P, T, faces_z)


class _Mesh:
    """A closed triangle mesh's edges (each once, from its lesser corner to its greater), each face's sides (edge
    indices and signs: +1 where the face runs along the edge's way), cotan weights, face normals and areas"""

    def __init__(self, P, T):
        self.P, self.T = P, T
        sides = np.stack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=1)          # (F, 3, 2)
        flat = np.sort(sides.reshape(-1, 2), axis=1)
        self.edges, index = np.unique(flat, axis=0, return_inverse=True)
        self.side_edge = index.reshape(-1, 3)
        self.side_sign = np.where(sides[:, :, 0] < sides[:, :, 1], 1.0, -1.0)
        a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
        normal = np.cross(b - a, c - a)
        double_area = np.linalg.norm(normal, axis=1)
        self.area = double_area / 2.0
        self.normal = normal / np.maximum(double_area, 1e-300)[:, None]
        # Each side's cotangent of the angle across it, half to the edge's weight
        weights = np.zeros(len(self.edges))
        corners = (a, b, c)
        for k in range(3):
            o, p, q = corners[(k + 2) % 3], corners[k], corners[(k + 1) % 3]      # side k runs p -> q, o opposite
            u, v = p - o, q - o
            cot = np.einsum("fk,fk->f", u, v) / np.maximum(np.linalg.norm(np.cross(u, v), axis=1), 1e-300)
            np.add.at(weights, self.side_edge[:, k], 0.5 * cot)
        self.weights = weights
        self._trees()

    def _trees(self):
        """A spanning tree of the edges (from corner 0: each corner's parent edge), one of the faces across the edges
        off it (each face's parent side), and the edges on neither: a torus's two loops' generators"""
        V, E = len(self.P), len(self.edges)
        on_tree = np.zeros(E, dtype=bool)
        at = [[] for _ in range(V)]
        for e, (a, b) in enumerate(self.edges.tolist()):
            at[a].append((e, b))
            at[b].append((e, a))
        self.order = [0]
        self.parent = {0: None}
        queue = deque([0])
        while queue:
            x = queue.popleft()
            for e, y in at[x]:
                if y not in self.parent:
                    self.parent[y] = (e, x)
                    on_tree[e] = True
                    self.order.append(y)
                    queue.append(y)
        faces_of = [[] for _ in range(E)]
        for f, row in enumerate(self.side_edge.tolist()):
            for e in row:
                faces_of[e].append(f)
        on_cotree = np.zeros(E, dtype=bool)
        self.face_order = [0]
        self.face_parent = {0: None}
        queue = deque([0])
        while queue:
            f = queue.popleft()
            for e in self.side_edge[f].tolist():
                if on_tree[e]:
                    continue
                g = faces_of[e][0] if faces_of[e][1] == f else faces_of[e][1]
                if g not in self.face_parent:
                    self.face_parent[g] = e
                    on_cotree[e] = True
                    self.face_order.append(g)
                    queue.append(g)
        self.on_tree = on_tree
        self.generators = np.flatnonzero(~on_tree & ~on_cotree).tolist()

    def closed_form(self, generator):
        """A closed 1-form (a number per edge, along its way) that's 1 on generator, 0 on the tree and the other
        generators, each cotree edge set (the face tree peeled from its leaves) so its face sums to 0"""
        form = np.zeros(len(self.edges))
        form[generator] = 1.0
        for f in reversed(self.face_order[1:]):
            e = self.face_parent[f]
            row, signs = self.side_edge[f], self.side_sign[f]
            k = int(np.flatnonzero(row == e)[0])
            rest = sum(signs[j] * form[row[j]] for j in range(3) if j != k)
            form[e] = -rest / signs[k]
        return form

    def _laplacian(self, x):
        a, b = self.edges[:, 0], self.edges[:, 1]
        flow = self.weights * (x[a] - x[b])
        out = np.zeros_like(x)
        np.add.at(out, a, flow)
        np.add.at(out, b, -flow)
        return out

    def harmonic(self, form):
        """form plus the gradient that makes it harmonic: f minimizing the cotan energy sum w (form + f_b - f_a)^2,
        L f = div form, by conjugate gradients (L is singular by a constant: kept to mean 0)"""
        a, b = self.edges[:, 0], self.edges[:, 1]
        rhs = np.zeros(len(self.P))
        np.add.at(rhs, a, self.weights * form)
        np.add.at(rhs, b, -self.weights * form)
        rhs -= rhs.mean()
        diagonal = np.zeros(len(self.P))
        np.add.at(diagonal, a, self.weights)
        np.add.at(diagonal, b, self.weights)
        diagonal = np.where(np.abs(diagonal) > 1e-300, diagonal, 1.0)
        f = np.zeros(len(self.P))
        r = rhs.copy()
        z = r / diagonal
        p = z.copy()
        rz = r @ z
        norm = np.linalg.norm(rhs) or 1.0
        for _ in range(SOLVE_ROUNDS):
            q = self._laplacian(p)
            step = rz / (p @ q)
            f += step * p
            r -= step * q
            r -= r.mean()
            if np.linalg.norm(r) <= SOLVE_TOLERANCE * norm:
                break
            z = r / diagonal
            rz, rz_old = r @ z, rz
            p = z + (rz / rz_old) * p
        return form + f[b] - f[a]

    def face_vectors(self, form):
        """A closed form's vector in each face (its plane): X with X . side = the form along the side, for its first
        two sides"""
        T, P = self.T, self.P
        u, v = P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]
        along_u = self.side_sign[:, 0] * form[self.side_edge[:, 0]]
        # (corner 0 to 2 against the third side, 2 -> 0)
        along_v = -self.side_sign[:, 2] * form[self.side_edge[:, 2]]
        uu, uv, vv = (np.einsum("fk,fk->f", x, y) for x, y in ((u, u), (u, v), (v, v)))
        det = np.maximum(uu * vv - uv * uv, 1e-300)
        alpha = (along_u * vv - along_v * uv) / det
        beta = (along_v * uu - along_u * uv) / det
        return alpha[:, None] * u + beta[:, None] * v

    def holomorphic(self, h1, h2):
        """h1 + i * h1 (* the conjugate: its face vectors turned a quarter about the normals, which a harmonic form's
        on a smooth surface is, as the combination of h1 and h2 nearest it in the cotan inner product)"""
        X = [self.face_vectors(h) for h in (h1, h2)]
        turned = np.cross(self.normal, X[0])
        inner = np.array([[np.sum(self.area * np.einsum("fk,fk->f", X[i], X[j])) for j in range(2)] for i in range(2)])
        with_turned = np.array([np.sum(self.area * np.einsum("fk,fk->f", X[i], turned)) for i in range(2)])
        s = np.linalg.solve(inner, with_turned)
        return h1 + 1j * (s[0] * h1 + s[1] * h2)

    def integrated(self, phi):
        """phi (complex, per edge) integrated along the tree into a coordinate z per corner; each face's corners' z
        from its corner 0's along its sides (a face across the cut is whole there); the two loops' periods (each
        generator edge's loop: along it, then back along the tree). The orientation made counterclockwise (faces'
        z counterclockwise as they are from outside: else the conjugate)."""
        z = np.zeros(len(self.P), dtype=complex)
        for y in self.order[1:]:
            e, x = self.parent[y]
            z[y] = z[x] + (phi[e] if self.edges[e][0] == x else -phi[e])
        periods = []
        for g in self.generators:
            a, b = self.edges[g]
            periods.append(z[a] + phi[g] - z[b])
        T = self.T
        z0 = z[T[:, 0]]
        z1 = z0 + self.side_sign[:, 0] * phi[self.side_edge[:, 0]]
        z2 = z0 - self.side_sign[:, 2] * phi[self.side_edge[:, 2]]
        faces_z = np.stack([z0, z1, z2], axis=1)
        signed = ((z1 - z0).conjugate() * (z2 - z0)).imag
        if np.median(signed) < 0.0:
            z, faces_z, periods = z.conjugate(), faces_z.conjugate(), [p.conjugate() for p in periods]
        p0, p1 = periods
        if abs((p0.conjugate() * p1).imag) < 1e-9 * abs(p0) * abs(p1):
            return z, faces_z, None
        if (p0.conjugate() * p1).imag < 0.0:
            p0, p1 = p1, p0
        return z, faces_z, (p0, p1)


def _reduced(p0, p1):
    """The period lattice's shortest basis (Gauss's reduction), the shorter first, counterclockwise from it to the
    other: a torus's tube and ring loops, not the diagonal combinations the tree-cotree's loops can be"""
    if abs(p1) < abs(p0):
        p0, p1 = p1, p0
    while True:
        p1 = p1 - round((p1 / p0).real) * p0
        if abs(p1) >= abs(p0):
            break
        p0, p1 = p1, p0
    if (p0.conjugate() * p1).imag < 0.0:
        p1 = -p1
    return p0, p1


class _Lattice:
    """The regular lattice in z holding the periods: the ideal one (1 and omega: a quarter turn for squares, a sixth
    for triangles) through a linear map L sending its points A and B to the periods, A and B chosen so L is as near a
    similarity as can be with about count points in the torus (det [A B] in the ideal's own coordinates)"""

    def __init__(self, periods, count, shape):
        self.shape = shape
        self.omega = 1j if shape == "quads" else complex(0.5, math.sqrt(3.0) / 2.0)
        p0, p1 = _reduced(*periods)
        tau = p1 / p0
        w = self.omega
        best = None
        # A over the ideal lattice's points out to where count of them fill the torus; B the nearest to tau A
        reach = int(math.ceil(2.0 * math.sqrt(count * COUNT_RANGE[1] / max(abs(tau.imag), 1e-3)))) + 2
        x, y = np.meshgrid(np.arange(-reach, reach + 1), np.arange(-reach, reach + 1), indexing="ij")
        A = (x + y * w).ravel()
        ax, ay = x.ravel(), y.ravel()
        target = tau * A
        # target's ideal coordinates: target = bx + by w
        by = np.round(target.imag / w.imag)
        bx = np.round(target.real - by * w.real)
        det = ax * by - ay * bx
        keep = (det >= COUNT_RANGE[0] * count) & (det <= COUNT_RANGE[1] * count) & (det > 0)
        if not keep.any():
            raise ValueError("ring_layout: no lattice of about that many points")
        B = bx + by * w
        error = np.abs(B[keep] / A[keep] - tau) / abs(tau)
        k = int(np.argmin(error))
        # A along the ideal lattice's first axis: the tube's period one row of cells (the lattice's turns are its
        # symmetries, so that's every row direction)
        aligned = np.flatnonzero(ay[keep] == 0)
        if len(aligned) and error[aligned].min() <= ALIGNED_STRETCH:
            k = int(aligned[np.argmin(error[aligned])])
        self.K = np.array([[ax[keep][k], ay[keep][k]], [bx[keep][k], by[keep][k]]], dtype=np.int64)
        self.count = int(det[keep][k])
        # L: ideal coordinates (as 2D) -> z (as 2D), with L(A) = p0, L(B) = p1
        ideal = np.array([[self.K[0, 0] + self.K[0, 1] * w.real, self.K[1, 0] + self.K[1, 1] * w.real],
                          [self.K[0, 1] * w.imag, self.K[1, 1] * w.imag]])
        target2 = np.array([[p0.real, p1.real], [p0.imag, p1.imag]])
        self.L = target2 @ np.linalg.inv(ideal)
        self.L_inv = np.linalg.inv(self.L)
        self.stretch = float(error[k])
        self._hermite()

    def _hermite(self):
        """K (the periods in the ideal lattice's coordinates, rows) in Hermite's form: rows (a, b) and (0, d), a d =
        count, 0 <= b < d, so each point mod the periods has one representative (x mod a, then y mod d)"""
        (p, q), (r, s) = self.K.tolist()
        # Column 0 to gcd by row operations
        while r != 0:
            t = p // r
            p, q, r, s = r, s, p - t * r, q - t * s
        if p < 0:
            p, q = -p, -q
        if s < 0:
            s = -s
        self.a, self.b, self.d = p, q % s, s
        assert self.a * self.d == self.count

    def key(self, x, y):
        """The representative of ideal point (x, y) mod the periods (arrays), as one index"""
        k = np.floor_divide(x, self.a)
        x = x - k * self.a
        y = (y - k * self.b) % self.d
        return x * self.d + y

    def ideal(self, zs):
        """z (complex array) -> the ideal lattice's coordinates (x, y) as floats"""
        v = self.L_inv @ np.array([zs.real.ravel(), zs.imag.ravel()])
        px, py = v[0], v[1]
        yy = py / self.omega.imag
        return (px - yy * self.omega.real).reshape(zs.shape), yy.reshape(zs.shape)

    def placed(self, P, T, faces_z):
        """Each lattice point placed where it falls in a face (its corners' z), barycentrically between the face's
        corners; the lattice's faces between them. None if a point falls in no face."""
        fx, fy = self.ideal(faces_z)
        corners = np.full((self.count, 3), np.nan)
        for f in range(len(T)):
            xs, ys = fx[f], fy[f]
            x0, x1 = int(math.floor(xs.min())), int(math.ceil(xs.max()))
            y0, y1 = int(math.floor(ys.min())), int(math.ceil(ys.max()))
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1), indexing="ij")
            gx, gy = gx.ravel(), gy.ravel()
            # Barycentric coordinates in the face's ideal triangle
            ux, uy = xs[1] - xs[0], ys[1] - ys[0]
            vx, vy = xs[2] - xs[0], ys[2] - ys[0]
            det = ux * vy - uy * vx
            if abs(det) < 1e-300:
                continue
            dx, dy = gx - xs[0], gy - ys[0]
            l1 = (dx * vy - dy * vx) / det
            l2 = (ux * dy - uy * dx) / det
            l0 = 1.0 - l1 - l2
            inside = (l0 >= -INSIDE_SLACK) & (l1 >= -INSIDE_SLACK) & (l2 >= -INSIDE_SLACK)
            if not inside.any():
                continue
            keys = self.key(gx[inside], gy[inside])
            fresh = np.isnan(corners[keys, 0])
            if not fresh.any():
                continue
            l = np.stack([l0[inside][fresh], l1[inside][fresh], l2[inside][fresh]], axis=1)
            corners[keys[fresh]] = l @ P[T[f]]
        if np.isnan(corners).any():
            return None
        x, y = np.divmod(np.arange(self.count), self.d)
        k = self.key
        if self.shape == "quads":
            faces = np.stack([k(x, y), k(x + 1, y), k(x + 1, y + 1), k(x, y + 1)], axis=1)
        else:
            faces = np.concatenate([np.stack([k(x, y), k(x + 1, y), k(x, y + 1)], axis=1),
                                    np.stack([k(x + 1, y), k(x + 1, y + 1), k(x, y + 1)], axis=1)])
        return corners, faces
