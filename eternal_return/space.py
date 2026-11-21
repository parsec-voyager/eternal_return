"""The sky over the Earth (sphere maps that are the Earth, sphere.meters_log2()), from the ground up into space with
stars (the user, 2026-11-16: "i want the atmosphere to fade to space with stars when you fly upward").

Drawn by the sphere's draw handler before anything of the planet, at the far plane, behind everything the view has
drawn: wherever there's nothing, the sky. Its brightness along a view ray is how much clear air the ray passes through
to space (the clear air of sphere.py's two layers, which is what makes the sky blue), counted in the sea-level zenith's
columns (air mass, m): the sky is the fog color times 1 - e^(-K m), so on the ground, where every direction looks
through at least a column of air, it's the fog color everywhere, as the plain sky (tools/sky.py), and climbing out of
the air, the zenith darkens first, through deep blue to black, while the horizon, looked at through hundreds of
kilometers of air, stays bright: from space, a thin glowing limb around the planet. The stars come in as the sky goes
dark, fixed to the map's axes (the Earth's), far away like the real ones: flying, scaling and looping never move them.

Air mass along a ray from height h (in scale heights, r = R + h from the center): looking up or level, e^-h Ch(x, mu),
x = r / H, mu the cosine from straight up, Ch the Chapman function, sqrt(pi x / 2) erfcx(mu sqrt(x / 2)) (exact for a
planet much bigger than its air), erfcx(y) ~ 2 / (sqrt(pi) (y + sqrt(y^2 + 4 / pi))); looking down past the planet,
the ray dips to its lowest point, hMin, and climbs out again: 2 e^-hMin Ch(xMin, 0) - e^-h Ch(x, -mu). A ray that would
meet the ground sees the planet, not the sky (where it isn't drawn yet, the fog color).

The stars: a cube's six faces each cut into cells, in three layers (STAR_LAYERS: cells across a face, the chance a
cell has a star, its brightest), a star at a random place in its cell, drawn as a small round spot of a pixel or two.
"""

import math

import numpy as np

# The sky's air mass to brightness: 1 - e^(-K m). On the ground the zenith's m is 1: 0.98 of the fog color.
SKY_K = 4.0
# (cells across a cube face, the chance a cell has a star, the brightest's brightness): rare bright stars, many dim
STAR_LAYERS = ((96, 0.012, 1.0), (192, 0.006, 0.55), (320, 0.004, 0.3))
# A star's spot: its radius, in pixels (Gaussian sigma)
STAR_SIZE = 0.65

VERTEX_SOURCE = """
void main()
{
    ndc = pos;
    // At the far plane: behind everything drawn, and drawn only where nothing is
    gl_Position = vec4(pos, 1.0, 1.0);
}
"""

FRAGMENT_SOURCE = """
float erfcx(float y)
{
    return 2.0 / (SQRT_PI * (y + sqrt(y * y + 4.0 / PI)));
}

float chapman(float x, float mu)
{
    return sqrt(0.5 * PI * x) * erfcx(mu * sqrt(0.5 * x));
}

// The clear air's columns along a ray from height h (scale heights), x = r / H, mu its cosine from straight up
float airMass(float h, float x, float mu)
{
    if (mu >= 0.0) {
        return exp(-h) * chapman(x, mu);
    }
    // x - x sqrt(1 - mu^2), without cancellation: how far below the camera the ray's lowest point is
    float drop = x * mu * mu / (1.0 + sqrt(max(0.0, 1.0 - mu * mu)));
    float hMin = h - drop;
    if (hMin < 0.0) {
        return 1.0e4;
    }
    return 2.0 * exp(-hMin) * chapman(x - drop, 0.0) - exp(-h) * chapman(x, -mu);
}

float hash(vec3 p)
{
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

// One layer of stars along a direction d (the stars' axes): its cube face, the cell, and the star in it, if any
vec3 starLayer(vec3 d, float cells, float chance, float brightest, float layer)
{
    vec3 a = abs(d);
    float face;
    vec2 uv;
    if (a.x >= a.y && a.x >= a.z) {
        face = d.x > 0.0 ? 0.0 : 1.0;
        uv = d.yz / a.x;
    } else if (a.y >= a.z) {
        face = d.y > 0.0 ? 2.0 : 3.0;
        uv = d.xz / a.y;
    } else {
        face = d.z > 0.0 ? 4.0 : 5.0;
        uv = d.xy / a.z;
    }
    vec2 g = (uv * 0.5 + 0.5) * cells;
    vec2 cell = floor(g);
    vec3 seed = vec3(cell, face * 7.0 + layer * 61.0);
    if (hash(seed) > chance) {
        return vec3(0.0);
    }
    vec2 at = cell + 0.2 + 0.6 * vec2(hash(seed + 11.0), hash(seed + 23.0));
    // The star's direction, and the angle to it in pixels: round on the sky, wherever on the cube's face it is
    vec2 st = at / cells * 2.0 - 1.0;
    vec3 star = face < 2.0 ? vec3(face < 1.0 ? 1.0 : -1.0, st) :
                face < 4.0 ? vec3(st.x, face < 3.0 ? 1.0 : -1.0, st.y) : vec3(st, face < 5.0 ? 1.0 : -1.0);
    float distance = length(cross(d, normalize(star))) / uSky2.w;
    float spot = exp(-0.5 * distance * distance / (STAR_SIZE * STAR_SIZE));
    float tint = hash(seed + 37.0);
    vec3 color = mix(vec3(1.0, 0.82, 0.66), vec3(0.72, 0.84, 1.0), tint);
    return color * brightest * pow(hash(seed + 51.0), 3.0) * spot;
}

void main()
{
    vec3 dir = normalize(mat3(uRay) * vec3(ndc, 1.0));
    float mu = dot(dir, uUp.xyz);
    float m = airMass(uSky2.x, uSky2.y, mu);
    float sky = clamp((1.0 - exp(-SKY_K * m)) / (1.0 - exp(-SKY_K)), 0.0, 1.0);
    vec3 color = uSky.rgb * sky;
    if (sky < 0.999) {
        vec3 d = normalize(mat3(uStarAxes) * dir);
        vec3 stars = vec3(0.0);
        STAR_LAYERS_SOURCE
        color += stars * (1.0 - sky);
    }
    fragColor = vec4(color, 1.0);
}
"""

ROWS = ("uSky", "uSky2", "uUp")
MATRICES = ("uRay", "uStarAxes")


def make_shader():
    import gpu
    info = gpu.types.GPUShaderCreateInfo()
    rows = (" ".join(f"mat4 m{i};" for i in range(len(MATRICES))) + " "
            + " ".join(f"vec4 r{i};" for i in range(len(ROWS))))
    info.typedef_source(f"struct SpaceValues {{ {rows} }};")
    info.uniform_buf(0, "SpaceValues", "spaceValues")
    # Struct members numbered, names as defines (as tiles3d's)
    for i, name in enumerate(MATRICES):
        info.define(name, f"spaceValues.m{i}")
    for i, name in enumerate(ROWS):
        info.define(name, f"spaceValues.r{i}")
    info.define("PI", repr(math.pi))
    info.define("SQRT_PI", repr(math.sqrt(math.pi)))
    info.define("SKY_K", repr(SKY_K))
    info.define("STAR_SIZE", repr(STAR_SIZE))
    layers = " ".join(f"stars += starLayer(d, {float(cells)!r}, {chance!r}, {brightest!r}, {float(i)!r});"
                      for i, (cells, chance, brightest) in enumerate(STAR_LAYERS))
    info.vertex_in(0, 'VEC2', "pos")
    interface = gpu.types.GPUStageInterfaceInfo("space_interface")
    interface.smooth('VEC2', "ndc")
    info.vertex_out(interface)
    info.fragment_out(0, 'VEC4', "fragColor")
    info.vertex_source(VERTEX_SOURCE)
    info.fragment_source(FRAGMENT_SOURCE.replace("STAR_LAYERS_SOURCE", layers))
    return gpu.shader.create_from_info(info)


class Sky:
    """The sky's shader and full-screen triangle"""

    def __init__(self):
        from gpu_extras.batch import batch_for_shader
        self.shader = make_shader()
        self.batch = batch_for_shader(self.shader, 'TRIS', {"pos": [(-1.0, -1.0), (3.0, -1.0), (-1.0, 3.0)]})
        self.values = None         # kept until the next draw, past this one's use of it

    def draw(self, ray_world, star_axes, fog_color, height, center, up, pixel_angle):
        """Draw the sky into the bound framebuffer, behind what's there: ray_world takes a view's clip coordinates
        (x, y, 1) to a ray's direction (world space), star_axes world directions to the stars' (the map's axes);
        height and center: the camera's height and distance from the planet's center, in the clear air's scale
        heights; up: the camera's way up (world space); pixel_angle: radians a pixel spans."""
        import gpu
        data = np.zeros(16 * len(MATRICES) + 4 * len(ROWS), np.float32)
        for i, matrix in enumerate((ray_world, star_axes)):
            m4 = np.eye(4)
            m4[:3, :3] = np.asarray(matrix, dtype=np.float64)
            data[16 * i:16 * i + 16] = m4.T.ravel()          # GLSL keeps matrices by columns
        start = 16 * len(MATRICES)
        data[start:start + 12] = (*fog_color, 0.0, max(-1e30, min(1e30, height)), min(center, 1e30), 0.0, pixel_angle,
                                  *up, 0.0)
        self.values = gpu.types.GPUUniformBuf(data)
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(False)
        self.shader.bind()
        self.shader.uniform_block("spaceValues", self.values)
        self.batch.draw(self.shader)


_sky = None


def sky():
    global _sky
    if _sky is None:
        _sky = Sky()
    return _sky
