"""Planets on sphere maps (sphere.py's "planet" setting): a whole planet's image as the sphere's surface where it has
detail, the map's motif taking over below that. The image is a tile pyramid made by tools/make_planet.py (NASA's Blue
Marble for Earth, or any equirectangular image of a whole planet).

Where it is: the map's axes are the planet's, as Earth's are in ECEF coordinates: +z the north pole, +x longitude 0 on
the equator, +y 90 degrees east. A point's latitude and longitude are its sphere normal's (the planet is a sphere, so
its latitude is the image's geodetic one along the normal, the way gravity points).

What's drawn (the shader's planetColor()): per pixel, the pyramid's level whose texels are about a pixel across (by
the pixel's footprint in longitude and latitude, the bigger, so nothing shimmers), blended with the next finer one by
how far between them it is (trilinear), each sampled bilinearly. A tile that isn't loaded yet falls back to a coarser
one; level 0 is always loaded. Past the finest level the texels just grow, and the motif comes in over them: the plan
draws only the motif's levels finer than handoff texels of the finest level (sphere.Plan), so from far away the
planet is all there is, and diving in it gives way to the motif level by level.

Loading (Planet.update(), from the draw, on the main thread, within LOAD_BUDGET seconds a draw): the view's tiles are
found by casting a grid of rays (view_tiles(), the shader's math in numpy), coarse levels first. A tile's JPEG is read
from its level's pack, decoded by ImBuf and drawn into its slot of the atlas, an offscreen of SLOTS x SLOTS tiles; the
least recently used tile not in view gives up its slot. A page table (a texture, PAGE x PAGE entries a level, a tile
at (column, row) mod PAGE, checked against the tile's own place) tells the shader each tile's slot.
"""

import json
import math
import os
import time

import numpy as np

# The page table's entries per level each way: a level's tiles in view are never PAGE apart (a view spans a few)
PAGE = 64
# The most time a draw spends loading tiles, and the most slots the atlas has each way (each a tile and its border)
LOAD_BUDGET = 0.002
ATLAS_SIZE = 4096
# The grid of rays view_tiles() casts across the view, each way: a tile in view is a few of them across at the least
# (texels are about a pixel), but one of the next finer level, blended in, near the horizon may be fewer
GRID = (65, 37)
# Tests: the shader shows which levels it used instead of the planet (planetColor())
DEBUG = False
# Below this share of a pixel's footprint the poles' longitudes are measured as if this far from the pole (cos(lat)),
# so the poles' crowded texels don't pull the level to the coarsest
POLE_GUARD = 0.05

PLANET_SOURCE = """
vec3 srgbToLinear(vec3 c)
{
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}

// uPlanet: x = 1 where the sphere's surface is the planet, y = n (N = 2^n), z = log2 of level 0's texels per radian
// plus the level bias, w = the finest level. uPlanetGrid: level 0's tile columns and rows, the tile size T, the
// atlas's slots each way.

// The planet's color at level-0 texel coordinates uv0 (from longitude -180 and the south pole) from level k's tile
// there, if it's loaded (Planet's page table): bilinear, the tile's border pixel meeting its neighbor's
bool planetLevel(int k, vec2 uv0, out vec3 color)
{
    color = vec3(0.0);
    float T = uPlanetGrid.z;
    vec2 uv = uv0 * exp2(float(k));
    vec2 tile = floor(uv / T);
    vec2 local = uv - tile * T;
    int columns = int(uPlanetGrid.x + 0.5) << k;
    int rows = int(uPlanetGrid.y + 0.5) << k;
    int c = int(tile.x) % columns;
    if (c < 0) {
        c += columns;
    }
    int r = int(tile.y);
    // At the north pole itself uv is the top edge, the last row's top
    if (r >= rows) {
        local.y += T * float(r - (rows - 1));
        r = rows - 1;
    }
    if (r < 0) {
        local.y = 0.0;
        r = 0;
    }
    vec4 page = texelFetch(planetPages, ivec2(c % PAGE, r % PAGE + PAGE * k), 0);
    if (page.w < 0.5 || int(page.y + 0.5) != c || int(page.z + 0.5) != r) {
        return false;
    }
    int slot = int(page.x + 0.5);
    int across = int(uPlanetGrid.w + 0.5);
    int size = int(T + 0.5) + 2;
    ivec2 origin = ivec2(slot % across, slot / across) * size;
    // The tile's own pixel p is its stored pixel p + 1 (the border), and texel centers are at + 0.5
    vec2 q = clamp(local + 0.5, vec2(0.0), vec2(T + 1.0));
    vec2 q0 = floor(q);
    vec2 f = q - q0;
    ivec2 i0 = origin + ivec2(q0);
    ivec2 i1 = min(i0 + 1, origin + size - 1);
    vec3 c00 = srgbToLinear(texelFetch(planetAtlas, i0, 0).rgb);
    vec3 c10 = srgbToLinear(texelFetch(planetAtlas, ivec2(i1.x, i0.y), 0).rgb);
    vec3 c01 = srgbToLinear(texelFetch(planetAtlas, ivec2(i0.x, i1.y), 0).rgb);
    vec3 c11 = srgbToLinear(texelFetch(planetAtlas, i1, 0).rgb);
    color = mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
    return true;
}

// The level whose texels are a pixel across (minus the bias), fractional: from the pixel's steps dm1, dm2 (units, so
// radians times N), their longitude and latitude, the bigger
float planetLevelAt(vec3 nn, vec3 dm1, vec3 dm2)
{
    float rxy = max(length(nn.xy), POLE_GUARD);
    float lon1 = (nn.x * dm1.y - nn.y * dm1.x) / (rxy * rxy), lon2 = (nn.x * dm2.y - nn.y * dm2.x) / (rxy * rxy);
    // Along the meridian, the step's part across the equator's plane over cos(lat)
    float lat1 = (dm1.z - nn.z * dot(nn, dm1)) / rxy, lat2 = (dm2.z - nn.z * dot(nn, dm2)) / rxy;
    float F = max(max(abs(lon1), abs(lon2)), max(abs(lat1), abs(lat2)));
    return uPlanet.y - log2(max(F, 1.0e-30)) - uPlanet.z;
}

// The planet under the pixel at the sphere's normal nn
vec3 planetColor(vec3 nn, vec3 dm1, vec3 dm2)
{
    float lon = atan(nn.y, nn.x);
    float lat = atan(nn.z, length(nn.xy));
    float tau = uPlanetGrid.x * uPlanetGrid.z / (2.0 * PI);
    vec2 uv0 = vec2(lon + PI, lat + HALF_PI) * tau;
    float finest = uPlanet.w;
    float level = clamp(planetLevelAt(nn, dm1, dm2), 0.0, finest);
    int k0 = int(floor(level));
    float f = level - float(k0);
    vec3 color;
    int k = k0;
    while (k > 0 && !planetLevel(k, uv0, color)) {
        k--;
    }
    if (k == 0) {
        planetLevel(0, uv0, color);
    }
    vec3 finer;
    bool blended = k == k0 && f > 0.0 && float(k0) < finest && planetLevel(k0 + 1, uv0, finer);
    if (blended) {
        color = mix(color, finer, f);
    }
#if PLANET_DEBUG
    // Tests (DEBUG): red, the levels fallen back from the one wanted (and a quarter if its finer one was missing);
    // green, the level wanted over 8
    bool finerMissing = k == k0 && f > 0.0 && float(k0) < finest && !blended;
    color = vec3(0.25 * float(k0 - k) + (finerMissing ? 0.25 : 0.0), level / 8.0, 0.0);
#endif
    return color;
}
"""


class Pyramid:
    """A planet's tiles on disk (tools/make_planet.py): its geometry, and a tile's pixels"""

    def __init__(self, folder):
        self.folder = folder
        with open(os.path.join(folder, "planet.json")) as f:
            self.meta = json.load(f)
        self.T = int(self.meta["tile"])
        self.columns = int(self.meta["columns"])
        self.rows = int(self.meta["rows"])
        self.levels = int(self.meta["levels"])
        self.size = self.T + 2
        self.finest = self.levels - 1
        # Level 0's texels per radian, and the finest level's texel in radians
        self.tau = self.columns * self.T / (2.0 * math.pi)
        self.finest_texel = 1.0 / (self.tau * 2.0 ** self.finest)
        self.indexes = [np.load(os.path.join(folder, f"level{k}.index.npy"), mmap_mode="r")
                        for k in range(self.levels)]
        self.packs = [None] * self.levels

    def tile_bytes(self, key):
        k, r, c = key
        if self.packs[k] is None:
            self.packs[k] = open(os.path.join(self.folder, f"level{k}.pack"), "rb")
        offset, length = (int(v) for v in self.indexes[k][r, c])
        pack = self.packs[k]
        pack.seek(offset)
        return pack.read(length)

    def tile_pixels(self, key):
        """A tile's pixels, RGBA float32 in 0..1 (sRGB), bottom-up, (T + 2) x (T + 2)"""
        import imbuf
        ib = imbuf.load_from_buffer(self.tile_bytes(key))
        with ib.with_buffer() as buffer:
            pixels = np.asarray(buffer, dtype=np.float32) * (1.0 / 255.0)
        ib.free()
        return pixels

    def close(self):
        for pack in self.packs:
            if pack is not None:
                pack.close()
        self.packs = [None] * self.levels


def view_tiles(pyramid, camera, normal, kappa, rho, n, ray_matrix, pixel_size, bias, grid=GRID):
    """The tiles the view needs, coarse levels first, each level's from the view's middle out: (level, row, column).
    The shader's rayHit(), surfaceOffset() and planetLevelAt() for a grid of rays across the view (clip -1..1), each
    sampling its level and the next finer one where it blends them."""
    xs = np.linspace(-1.0, 1.0, grid[0])
    ys = np.linspace(-1.0, 1.0, grid[1])
    X, Y = np.meshgrid(xs, ys)
    at = np.stack([X.ravel(), Y.ravel(), np.ones(X.size)], axis=1)
    M = np.asarray(ray_matrix, dtype=np.float64)[:3, :3]
    dirs = at @ M.T
    c = np.asarray(camera, dtype=np.float64)
    nA = np.asarray(normal, dtype=np.float64)
    k = float(kappa)
    a = k * np.einsum("ij,ij->i", dirs, dirs)
    b = k * (dirs @ c) + dirs @ nA
    q = k * float(c @ c) + 2.0 * float(c @ nA)
    disc = b * b - a * q
    hit = (disc >= 0.0) & (b < 0.0) & (q > 0.0)
    if not hit.any():
        return []
    dirs, b, a, disc, at = dirs[hit], b[hit], a[hit], disc[hit], at[hit]
    t = q / (-b + np.sqrt(disc))
    p = c + t[:, None] * dirs
    pn = p @ nA
    L = np.sqrt(np.maximum(0.0, 1.0 + k * (2.0 * pn + k * np.einsum("ij,ij->i", p, p))))
    nu = k * p + nA
    nn = nu / np.linalg.norm(nu, axis=1, keepdims=True)
    steps = []
    for column, size in ((0, pixel_size[0]), (1, pixel_size[1])):
        dr = M[:, column] * size
        nu_dr = nu @ dr
        nu_dir = np.einsum("ij,ij->i", nu, dirs)
        dp = t[:, None] * (dr[None] - dirs * (nu_dr / nu_dir)[:, None])
        dm = (dp - nn * np.einsum("ij,ij->i", nn, dp)[:, None]) / (rho * L[:, None])
        steps.append(dm)
    rxy = np.maximum(np.linalg.norm(nn[:, :2], axis=1), POLE_GUARD)
    F = np.zeros(len(nn))
    for dm in steps:
        lon = (nn[:, 0] * dm[:, 1] - nn[:, 1] * dm[:, 0]) / (rxy * rxy)
        lat = (dm[:, 2] - nn[:, 2] * np.einsum("ij,ij->i", nn, dm)) / rxy
        F = np.maximum(F, np.maximum(np.abs(lon), np.abs(lat)))
    level = n - np.log2(np.maximum(F, 1e-300)) - (math.log2(pyramid.tau) + bias)
    level = np.clip(level, 0.0, pyramid.finest)
    lon = np.arctan2(nn[:, 1], nn[:, 0])
    lat = np.arctan2(nn[:, 2], np.linalg.norm(nn[:, :2], axis=1))
    u0, v0 = (lon + math.pi) * pyramid.tau, (lat + math.pi / 2) * pyramid.tau
    k0 = np.floor(level).astype(np.int64)
    finer = (level > k0) & (k0 < pyramid.finest)
    middle = np.hypot(at[:, 0], at[:, 1])
    kk = np.concatenate([k0, k0[finer] + 1])
    u, v, d = (np.concatenate([x, x[finer]]) for x in (u0, v0, middle))
    s = np.exp2(kk) / pyramid.T
    columns, rows = pyramid.columns << kk, pyramid.rows << kk
    r = np.clip(np.floor(v * s).astype(np.int64), 0, rows - 1)
    c = np.floor(u * s).astype(np.int64) % columns
    # Each tile once, nearest the middle first within its level, coarse levels first
    order = np.lexsort((d, c, r, kk))
    code = (kk[order] << 42) | (r[order] << 21) | c[order]
    first = np.ones(len(code), bool)
    first[1:] = code[1:] != code[:-1]
    keep = order[first]
    keep = keep[np.lexsort((d[keep], kk[keep]))]
    return [(int(kk[i]), int(r[i]), int(c[i])) for i in keep]


class Planet:
    """A pyramid's tiles on the GPU: the atlas, the page table and which tile is in which slot"""

    def __init__(self, folder):
        self.pyramid = Pyramid(folder)
        self.across = max(1, ATLAS_SIZE // self.pyramid.size)
        self.slots = self.across * self.across
        self.atlas = None
        self.pages = None
        self.pages_changed = True
        self.page_data = np.zeros((PAGE * self.pyramid.levels, PAGE, 4), np.float32)
        self.slot_of = {}                       # (level, row, column) -> slot
        self.key_of = [None] * self.slots
        self.last_used = np.zeros(self.slots, np.int64)
        self.frame = 0
        self.copy_shader = None
        self.copy_batch = None
        self.loads = 0
        self.missing = 0
        self.pinned = [(0, r, c) for r in range(self.pyramid.rows) for c in range(self.pyramid.columns)]
        assert len(self.pinned) < self.slots, "the atlas can't hold level 0"

    def make_gpu(self, keeper):
        """The atlas, made inside the keeper's bind (sphere.Floor.offscreen()), and the copy shader"""
        import gpu
        from gpu_extras.batch import batch_for_shader
        with keeper.bind():
            side = self.across * self.pyramid.size
            self.atlas = gpu.types.GPUOffScreen(side, side, format='RGBA8')
        info = gpu.types.GPUShaderCreateInfo()
        info.push_constant('VEC4', "rect")
        info.sampler(0, 'FLOAT_2D', "tile")
        info.vertex_in(0, 'VEC2', "pos")
        interface = gpu.types.GPUStageInterfaceInfo("planet_copy_interface")
        interface.smooth('VEC2', "uv")
        info.vertex_out(interface)
        info.fragment_out(0, 'VEC4', "fragColor")
        info.vertex_source("void main() { uv = pos; "
                           "gl_Position = vec4(mix(rect.xy, rect.zw, pos) * 2.0 - 1.0, 0.0, 1.0); }")
        info.fragment_source("void main() { ivec2 s = textureSize(tile, 0); "
                             "fragColor = texelFetch(tile, clamp(ivec2(uv * vec2(s)), ivec2(0), s - 1), 0); }")
        self.copy_shader = gpu.shader.create_from_info(info)
        self.copy_batch = batch_for_shader(self.copy_shader, 'TRI_STRIP', {"pos": [(0, 0), (1, 0), (0, 1), (1, 1)]})

    def free_slot(self, needed):
        """A slot for a new tile: an empty one, else the least recently used tile's that isn't needed or pinned"""
        for slot, key in enumerate(self.key_of):
            if key is None:
                return slot
        order = np.argsort(self.last_used)
        for slot in order:
            key = self.key_of[slot]
            if key[0] == 0 or key in needed:
                continue
            self.evict(int(slot))
            return int(slot)
        return None

    def evict(self, slot):
        key = self.key_of[slot]
        if key is None:
            return
        k, r, c = key
        entry = self.page_data[r % PAGE + PAGE * k, c % PAGE]
        if entry[3] > 0.5 and int(entry[1]) == c and int(entry[2]) == r:
            entry[:] = 0.0
        del self.slot_of[key]
        self.key_of[slot] = None
        self.pages_changed = True

    def upload(self, keys):
        """Loads tiles into slots (their pixels drawn into the atlas) and writes their pages"""
        import gpu
        tiles = []
        for key, slot in keys:
            pixels = self.pyramid.tile_pixels(key)
            size = self.pyramid.size
            buffer = gpu.types.Buffer('FLOAT', size * size * 4, pixels.ravel())
            tiles.append((key, slot, gpu.types.GPUTexture((size, size), format='RGBA8', data=buffer)))
        side = self.across * self.pyramid.size
        blend = gpu.state.blend_get()
        with self.atlas.bind():
            gpu.state.depth_test_set('NONE')
            gpu.state.blend_set('NONE')
            self.copy_shader.bind()
            for key, slot, texture in tiles:
                x0 = (slot % self.across) * self.pyramid.size
                y0 = (slot // self.across) * self.pyramid.size
                self.copy_shader.uniform_float("rect", (x0 / side, y0 / side, (x0 + self.pyramid.size) / side,
                                                        (y0 + self.pyramid.size) / side))
                self.copy_shader.uniform_sampler("tile", texture)
                self.copy_batch.draw(self.copy_shader)
        gpu.state.blend_set(blend)
        for key, slot, _texture in tiles:
            k, r, c = key
            # A tile that shares its page entry with another (PAGE apart) takes it over
            self.page_data[r % PAGE + PAGE * k, c % PAGE] = (slot, c, r, 1.0)
            self.slot_of[key] = slot
            self.key_of[slot] = key
            self.last_used[slot] = self.frame
            self.loads += 1
        self.pages_changed = True

    def update(self, keeper, needed, budget=None):
        """From the draw: level 0 in (all of it, the first time), then as much of what the view needs as budget
        seconds allow (LOAD_BUDGET's), in the order given; the page table texture remade if anything changed"""
        import gpu
        budget = LOAD_BUDGET if budget is None else budget
        self.frame += 1
        if self.atlas is None:
            self.make_gpu(keeper)
            self.upload([(key, i) for i, key in enumerate(self.pinned)])
        for key in needed:
            slot = self.slot_of.get(key)
            if slot is not None:
                self.last_used[slot] = self.frame
        start = time.perf_counter()
        missing = [key for key in needed if key not in self.slot_of]
        needed_set = set(needed)
        loaded = 0
        while missing and time.perf_counter() - start < budget:
            batch = []
            for key in missing[:4]:
                slot = self.free_slot(needed_set)
                if slot is None:
                    break
                # Taken now, so the next free_slot() doesn't hand it out again
                self.key_of[slot] = key
                batch.append((key, slot))
            if not batch:
                break
            self.upload(batch)
            loaded += len(batch)
            missing = missing[len(batch):]
        self.missing = len(missing)
        if self.pages_changed or self.pages is None:
            data = gpu.types.Buffer('FLOAT', self.page_data.size, self.page_data.ravel())
            self.pages = gpu.types.GPUTexture((PAGE, PAGE * self.pyramid.levels), format='RGBA32F', data=data)
            self.pages_changed = False
        return loaded

    def uniforms(self, plan_uses_planet, n, bias):
        """The shader's uPlanet and uPlanetGrid rows"""
        p = self.pyramid
        return ((1.0 if plan_uses_planet else 0.0, float(n), math.log2(p.tau) + float(bias), float(p.finest)),
                (float(p.columns), float(p.rows), float(p.T), float(self.across)))

    def close(self):
        self.pyramid.close()


def handoff_level(world, motif, pyramid, handoff):
    """The coarsest motif level drawn over a planet: the first whose cells are at most handoff of the finest level's
    texels across. A level's cells are 2^(level_shift - level) of N units, unit_angle radians N units where a cell is
    biggest (the middle of the layout's face); never N itself, which has no float past 2^1023."""
    limit = handoff * pyramid.finest_texel
    return max(motif.first_level, math.ceil(world.level_shift + math.log2(world.unit_angle / limit)))


def defines(info):
    info.define("PAGE", str(PAGE))
    info.define("PLANET_DEBUG", "1" if DEBUG else "0")
    info.define("POLE_GUARD", repr(POLE_GUARD))


_planets = {}


def planet(folder):
    """The Planet for a pyramid folder, made the first time (None if it can't be read)"""
    folder = os.path.abspath(os.path.expanduser(folder))
    p = _planets.get(folder)
    if p is None and folder not in _planets:
        try:
            p = Planet(folder)
        except (OSError, ValueError, KeyError, AssertionError) as error:
            print(f"Sphere planet {folder}: {error}")
            p = None
        _planets[folder] = p
    return p


def forget():
    for p in _planets.values():
        if p is not None:
            p.close()
    _planets.clear()
