"""Mandelbrot floors: a flat floor painted with the Mandelbrot set, per pixel on the GPU, that the drone can dive into
for as long as the numbers hold out.

The floor is the plane through the map's object's origin (the scene's "mandelbrot_object"), the real axis along world X
and the imaginary along Y. A point p of it, in the map's frame (world location + scene.references.world_offset, see
Keep Drone at Origin), is

    c = center + units * (p - anchor)

Where the floor is in the complex plane is kept exactly, like an endless map's anchor (see endless.py): center in
fixed point Python integers, with as many bits as the zoom needs, anchor a point of the floor near the drone, and units
(complex units per meter) as a mantissa and an exponent. Every loop the drone makes, about whatever point, scales the
floor's mapping by the loop factor about the same point (loop(), from ER_OT_Fly.loop_scale()): the set isn't
self-similar, so only that keeps the view the same through a loop. The anchor moves to the loop's pivot, so the drone
is never far from it and its distance from it always fits a float64.

Drawing (a viewport draw handler, so in Solid and Rendered mode alike): each pixel's c is a reference point's plus a
small offset. The GPU iterates, in float32, only how far the pixel's orbit is from the reference orbit (perturbation,
with Zhuoran's rebasing), and the reference orbit is iterated here exactly, in fixed point with enough bits for the
zoom, a chunk a frame. A search keeps looking for a longer-lived reference near the drone, iterating its candidates
as offsets from the current reference.

Deep down, a pixel's offset from the reference is far smaller than float32's smallest number, about 1e-38: the GPU gets
the offsets in units of about the view's size instead (2^scaleExponent), and keeps each pixel's offset from the
reference orbit scaled, with an exponent of its own, until it's grown big enough to be a float32 by itself. And the
deeper, the more iterations a pixel needs: while a pixel's offset is tiny next to the reference orbit, advancing it is
all but linear, so many iterations combine into one jump, which the GPU takes from tables (skip-ahead, bilinear
approximation, see bla_tables()). The CPU's float64 offsets (the search, the tables, the reference's distance) run out
at a zoom of about 1e300.

The map's object is a plane mesh at the floor, displayed as its bounds (drawn, it fought the floor for depth): what auto
scale measures to, in mesh mode, so loops happen about the point under the drone. A click that hits nothing while the
drone camera looks down at the floor selects it (select_floor_if_looking_at_it(), from ER_OT_Fly.click()).

The object's custom properties:
- mandelbrot_c, mandelbrot_anchor, mandelbrot_scale: the mapping (see Mapping), written by every loop
- mandelbrot_version: counts the mapping's writes, so an undo, which puts an older mapping back, is noticed
- iterations, iterations_per_decade: the iteration limit under the drone, raised by iterations_per_decade for every
  factor of 10 of zoom; a pixel covering 10 times more of the set gets iterations_per_decade fewer
- palette: "earth" (the set is land, the rest ocean, shallow near the coast and deep far from it) or "rainbow"
  (bands of color by iteration count; color_density, color_speed and filament_glow are for it)
- color_density (color cycles per iteration), color_speed (palette cycles per second)
- filament_glow: the color of filaments and color bands finer than a pixel, from the palette's gray (0) to white (1)
- smooth_filaments: tell those filaments by the distance estimate (smooth, about a quarter slower) rather than by how
  fast the iteration count climbs across a pixel (in 2 x 2 pixel blocks); down to zooms of about 1e28, past which
  the distance estimate's derivative overflows a float32
- skip_ahead: jump many iterations at once while a pixel's orbit stays close to the reference's (see bla_tables())
- resolution: the share of the view's pixels, across and down, the set is worked out at, scaled up to the view
- auto_resolution: lower the resolution, down to MIN_RESOLUTION, while frames miss the display's refresh, and raise it
  back up to resolution once they've kept up for a while
- show_info: the zoom, iterations, resolution and frame time in the corner of each view
"""

import collections
import math
import time
from fractions import Fraction

import blf
import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

CENTER = "mandelbrot_c"
ANCHOR = "mandelbrot_anchor"
SCALE = "mandelbrot_scale"
VERSION = "mandelbrot_version"
# On the scene: the name of the map's object, so drawing finds it without looking through every object
SCENE_KEY = "mandelbrot_object"
CAMERA_NAME = "Drone_Camera"
# The drone rig, which a click may select without having hit the map
RIG_NAMES = {"Drone", "Drone_Camera", "clip_start", "clip_end"}

# Fixed point bits beyond what the zoom needs (a pixel is about 1/1000 of the view under the drone): the center has
# CENTER_MARGIN, a reference orbit ORBIT_MARGIN, which also covers its rounding errors growing along the orbit
CENTER_MARGIN = 80
ORBIT_MARGIN = 64
# A reference whose orbit's bits fall within this many of what the zoom needs is replaced at once
ORBIT_MIN_MARGIN = 16
# Reference orbit iterations per frame, in fixed point Python integers (about 0.6 ms per 1000 at 160 bits)
ORBIT_STEPS = 2000

# The reference orbit is a texture this many texels wide, one complex number per texel
ORBIT_WIDTH = 1024
# Escaping beyond this radius (squared) rather than 2 makes the smooth iteration count smooth
BAILOUT = 1.0e6
# How far toward the camera the floor is drawn, as a share of its distance, so nothing at the floor's own depth shows
# through. The floor looks the same, since every point moves along its own view ray.
PULL = 0.99
# The fog's per-channel extinction distances, in fog distances: Earth's air, as tools/fog_look.py (EARTH_EXTINCTION)
EXTINCTION = (0.349, 0.256, 0.179)
# Fully fogged beyond this many fog distances (red is down to 1e-4 there), so those pixels skip the set
FOG_CUTOFF = 9.0 * max(EXTINCTION)
# The reference search: candidates in a disk this many drone heights across around the point below the drone, and
# iterations per frame it spends on them
SEARCH_CANDIDATES = 64
SEARCH_RADIUS = 8.0
SEARCH_STEPS = 250
# A new search for a better reference starts once the drone has moved SEARCH_MOVE heights or zoomed SEARCH_ZOOM times
# since the last (or at once, if the reference has to go): a dive changes the zoom by 10% every few frames, and a
# search every frame cost 3.6 ms of each
SEARCH_MOVE = 4.0
SEARCH_ZOOM = 2.0
# A reference further than REFERENCE_REACH drone heights from the point below the camera is searched away from, and one
# further than REFERENCE_LIMIT is replaced at once: a pixel's offset from it, in float32, would blur the view
REFERENCE_REACH = 30.0
REFERENCE_LIMIT = 200.0
# auto_resolution: a frame slower than MISSED_FRAME missed the display's refresh (it took two, 33 ms at 60 Hz; flight
# jitters a few ms around 16.7 without missing any). MISSED_TO_LOWER of the last RECENT_FRAMES lower the resolution a
# step (a lone hitch doesn't; missed frames come between ones that make it, so not in a row), and FAST_FRAMES_TO_RAISE
# in a row without a miss raise it a step. A step up that turns out too slow doubles the wait before trying it again
# (up to MAX_RAISE_WAIT), so a resolution on the edge doesn't pulse.
MISSED_FRAME = 0.025
RECENT_FRAMES = 12
MISSED_TO_LOWER = 2
FAST_FRAMES_TO_RAISE = 60
MAX_RAISE_WAIT = 480
RESOLUTION_STEP = 0.05
MIN_RESOLUTION = 0.2
# Skip-ahead (bilinear approximation): a jump is taken while the orbit's offset from the reference is small enough that
# its square, the part a jump leaves out, is under BLA_EPSILON of the rest (2^-22: as close as float32 rounding
# leaves plain iterations). The tables hold for offsets of the pixel's c from the reference's up to BLA_NEAR drone
# heights (the floor near the camera, where iteration counts are highest), with room for the view to zoom out twice as
# far before they're rebuilt.
BLA_EPSILON = 2.0 ** -22
BLA_NEAR = 40.0
BLA_WIDTH = 1024
# Offsets are scaled on the GPU while 2^scaleExponent is under 2^SCALED_BELOW (float32 holds down to about 2^-126)
SCALED_BELOW = -96
# The set is about this wide, for the zoom shown and the iteration limit
SET_WIDTH = 3.0
# Past this zoom the CPU's float64 offsets (the search, the skip-ahead tables) underflow
FLOAT64_ZOOM = 1.0e300
MAX_ITERATIONS = 50000

DEFAULTS = {"iterations": 200, "iterations_per_decade": 250, "palette": "earth", "color_density": 0.03,
            "color_speed": 0.0, "filament_glow": 1.0, "smooth_filaments": False, "skip_ahead": True, "resolution": 0.5,
            "auto_resolution": True, "show_info": True}

PALETTES = ("rainbow", "earth")
# The earth palette, in sRGB (as a color picker shows them): ocean from deep to the surf at the coast, shoals (coast
# finer than a pixel), beach, land
EARTH_COLORS = {"DEEP_OCEAN": (6, 24, 64), "OCEAN": (16, 62, 132), "SHALLOW": (32, 156, 190), "SURF": (128, 214, 214),
                "SHOALS": (150, 212, 200), "BEACH": (222, 204, 160), "LAND": (58, 104, 44)}


def srgb_to_linear(rgb):
    """An sRGB color, 0 to 255, as linear floats: the floor's output is color managed, so it writes linear colors"""
    def channel(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return tuple(channel(c) for c in rgb)

VERTEX_SOURCE = """
void main()
{
    offset = pos;
    gl_Position = viewProjection * vec4(pos * (height * PULL), -height * PULL, 1.0);
}
"""

# Scales the floor, worked out at a lower resolution, up to the view, drawing the same rings so the depth is the floor's
COMPOSITE_VERTEX_SOURCE = """
void main()
{
    gl_Position = viewProjection * vec4(pos * (height * PULL), -height * PULL, 1.0);
    clip = gl_Position;
}
"""

COMPOSITE_FRAGMENT_SOURCE = """
void main()
{
    // Bilinear by hand: the offscreen texture is sampled nearest, which scaled up in blocks that crawled in flight
    ivec2 size = textureSize(floorColor, 0);
    vec2 p = (clip.xy / clip.w * 0.5 + 0.5) * vec2(size) - 0.5;
    ivec2 i = ivec2(floor(p));
    vec2 f = p - floor(p);
    ivec2 top = size - 1;
    vec3 a = texelFetch(floorColor, clamp(i, ivec2(0), top), 0).rgb;
    vec3 b = texelFetch(floorColor, clamp(i + ivec2(1, 0), ivec2(0), top), 0).rgb;
    vec3 c = texelFetch(floorColor, clamp(i + ivec2(0, 1), ivec2(0), top), 0).rgb;
    vec3 d = texelFetch(floorColor, clamp(i + ivec2(1, 1), ivec2(0), top), 0).rgb;
    fragColor = vec4(mix(mix(a, b, f.x), mix(c, d, f.x), f.y), 1.0);
}
"""

FRAGMENT_SOURCE = """
vec2 cmul(vec2 a, vec2 b)
{
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

vec2 orbit_at(int m)
{
    return texelFetch(orbit, ivec2(m % ORBIT_WIDTH, m / ORBIT_WIDTH), 0).xy;
}

// An offset's size, measured without squaring it: the GPU flushes numbers under about 1e-38 to 0, so the squares of
// an offset under about 1e-19 vanish. length() of one was 0, its log2 -infinity, under every skip-ahead radius: every
// jump was taken, wrong, in a square of floor around the reference (a diamond, with the view turned) from about 1e18,
// and the whole view went flat from 1e21.
float max_part(vec2 v)
{
    return max(abs(v.x), abs(v.y));
}

// log2 |v|, from its bigger part
float log2_size(vec2 v)
{
    float big = max_part(v);
    if (big == 0.0) {
        return -1.0e30;
    }
    vec2 unit = v / big;
    return log2(big) + 0.5 * log2(dot(unit, unit));
}

void main()
{
    // Where this pixel's floor point is, from the camera, in camera heights; only its distance decides the fog
    float distance = height * sqrt(1.0 + dot(offset, offset));
    vec3 transmission = exp(-distance / (fogDistance * EXTINCTION));

    // This pixel's c, from the reference point's, in units of 2^scaleExponent (about the view's size under the drone),
    // so it's an ordinary number at any depth
    vec2 dc = cameraFromReference + offset * unitsPerHeight;
    // A pixel covers this much of the complex plane, in the same units. Derivatives before any branching: they're
    // undefined where neighboring pixels take different branches.
    float pixel = length(fwidth(offset)) * unitsPerHeight;
    // The iterations this pixel needs: the limit under the drone, where a pixel covers the least of the set (pixelRef),
    // less iterationsPerDecade for every factor of 10 more it covers. Far away, a pixel covers a stretch of the set
    // whose finer detail it can't show anyway.
    int cap = clamp(maxIterations - int(iterationsPerDecade * log(max(pixel / pixelRef, 1.0)) / log(10.0)),
                    min(minIterations, maxIterations), maxIterations);

    int n = 0;
    float r2 = 0.0;
    bool escaped = false;
    float edge = 1.0e9;        // the distance from the set, in pixels (smooth filaments)
    if (distance < fogDistance * FOG_CUTOFF) {
        // The pixel's z minus the reference orbit's, d, in one of two forms. Deep down it starts far below float32's
        // smallest numbers, so it's kept scaled: d = dS 2^(scaleExponent + k), dS renormalized as it grows. Once d
        // fits a float32 by itself it's kept as d. (Squared, a scaled d's square is far smaller still and, under
        // float32's smallest, rightly nothing.)
        bool scaled = scaleExponent < SCALED_BELOW;
        vec2 dS = vec2(0.0);
        int k = 0;
        vec2 d = vec2(0.0);
        vec2 dcU = ldexp(dc, ivec2(scaleExponent));
        vec2 Z = vec2(0.0);    // the reference orbit's z at m
        vec2 z = vec2(0.0);
        vec2 dz = vec2(0.0);   // dz/dc, for the distance estimate (smooth filaments, not scaled)
        int m = 0;
        // Skip-ahead's tables hold for offsets from the reference up to blaCMax (near the camera). A pixel stops
        // trying once its offset is past every jump's radius, until it's rebased.
        bool skip = blaLevels > 0 && dot(dc, dc) <= blaCMax2;
        while (n < cap) {
            int steps = 0;
            if (skip && m > 0) {
                // log2 of the offset's size (unscaled), what the tables' radii are kept as: only worked out here, as
                // the plain iterations at an orbit's end are most of the work
                float logSize = scaled ? log2_size(dS) + float(scaleExponent + k) : log2_size(d);
                // Skip-ahead (see bla_tables()): the biggest jump of 2, 4, 8, ... iterations from m whose table entry
                // holds for an offset this size, within the iterations left. Jumps of 2^j start at m = 1 + i 2^j,
                // so only those up to the trailing zeros of m - 1 start here. A jump is
                // d -> A d + B dc, on whichever form the offset is in.
                int first = m - 1;
                int jump = blaLevels - 1;
                if (first > 0) {
                    int zeros = 0;
                    while (((first >> zeros) & 1) == 0) {
                        zeros++;
                    }
                    jump = min(jump, zeros);
                }
                for (; jump >= 1 && steps == 0; jump--) {
                    int entry = first >> jump;
                    if (entry >= (blaCount >> jump) || (1 << jump) > cap - n) {
                        continue;
                    }
                    int i = blaSpan - (blaSpan >> jump) + entry;
                    ivec2 at = ivec2(i % BLA_WIDTH, i / BLA_WIDTH);
                    if (logSize < texelFetch(blaLogR, at, 0).x) {
                        vec4 ab = texelFetch(blaAB, at, 0);
                        if (scaled) {
                            // A jump's A can be up to 1e30: dS brought down to about 1 first, so the product fits a
                            // float32, and again after. Measured by its bigger part, not length(), whose squares
                            // overflow a float32 past 1e19.
                            int down = max(0, int(floor(log2(max(max(abs(dS.x), abs(dS.y)), 1.0e-30)))));
                            dS = ldexp(dS, ivec2(-down));
                            k += down;
                            dS = cmul(ab.xy, dS) + cmul(ab.zw, ldexp(dc, ivec2(-k)));
                            down = max(0, int(floor(log2(max(max(abs(dS.x), abs(dS.y)), 1.0e-30)))));
                            dS = ldexp(dS, ivec2(-down));
                            k += down;
                        }
                        else {
                            // B dc in dc's units first: dc itself is under float32's smallest numbers deep down (the
                            // GPU makes those 0), but B can be big enough to make B dc count
                            d = cmul(ab.xy, d) + ldexp(cmul(ab.zw, dc), ivec2(scaleExponent));
                        }
                        dz = cmul(ab.xy, dz) + ab.zw;
                        steps = 1 << jump;
                    }
                }
                skip = steps > 0 || logSize < blaMaxLogR;
            }
            if (steps == 0) {
#if SMOOTH_FILAMENTS
                dz = 2.0 * cmul(z, dz) + vec2(1.0, 0.0);
#endif
                if (scaled) {
                    dS = 2.0 * cmul(Z, dS) + ldexp(cmul(dS, dS), ivec2(scaleExponent + k)) + ldexp(dc, ivec2(-k));
                }
                else {
                    d = 2.0 * cmul(Z, d) + cmul(d, d) + dcU;
                }
                steps = 1;
            }
            if (scaled && dot(dS, dS) > 1.8e19) {
                dS = ldexp(dS, ivec2(-32));
                k += 32;
            }
            // Big enough to be a float32 by itself
            if (scaled && scaleExponent + k >= SCALED_BELOW) {
                d = ldexp(dS, ivec2(scaleExponent + k));
                scaled = false;
            }
            m += steps;
            n += steps;
            Z = orbit_at(m);
            vec2 dNow = scaled ? ldexp(dS, ivec2(scaleExponent + k)) : d;
            z = Z + dNow;
            r2 = dot(z, z);
            if (r2 > BAILOUT) {
                escaped = true;
                break;
            }
            // Rebasing (Zhuoran): once the pixel's orbit is closer to 0 than to the reference's, follow it from the
            // start of the reference orbit instead, where Z is 0, so d stays small next to Z. Also at the reference
            // orbit's end, if it escaped before this pixel. Compared by their bigger parts, which tiny offsets keep
            // (both squares would be 0, and the pixel never rebased).
            if (max_part(z) < max_part(dNow) || m >= orbitLength - 1) {
                d = z;
                scaled = false;
                m = 0;
                Z = vec2(0.0);
                skip = blaLevels > 0 && dot(dc, dc) <= blaCMax2;
            }
        }
#if SMOOTH_FILAMENTS
        if (escaped && scaleExponent >= SCALED_BELOW) {
            float pixelUnscaled = pixel * exp2(float(scaleExponent));
            edge = 0.5 * sqrt(r2) * log(r2) / max(length(dz), 1.0e-30) / max(pixelUnscaled, 1.0e-37);
        }
#endif
    }

    // The smooth iteration count, continuous from band to band, and how many color bands a pixel spans, worked out for
    // every pixel (derivatives again). n counts the iteration that escaped, so it's n - log2(log2 |z|).
    float nu = escaped ? float(n) - log2(0.5 * log(r2) / log(2.0)) : 0.0;
    float t = nu * colorDensity + colorPhase;
    float bands = fwidth(t);
    // Near the set the iteration count climbs steeply: many iterations across a pixel means filaments finer than it.
    // The distance estimate (smooth filaments) needs dz/dc, which overflows a float32 deep down.
#if SMOOTH_FILAMENTS
    float filaments = scaleExponent >= SCALED_BELOW ? exp(-edge) : smoothstep(1.0, 6.0, fwidth(nu));
#else
    float filaments = smoothstep(1.0, 6.0, fwidth(nu));
#endif
    // An inside pixel next to escaped ones is most likely a sample that happened to land on a long-lived point among
    // filaments finer than a pixel: far away, these were black specks twinkling over the frost as the drone moved
    bool inside = !escaped;
    float insideEdge = min(1.0, fwidth(inside ? 1.0 : 0.0));
    vec3 fine = vec3(0.5 + 0.5 * filamentGlow);

    vec3 color;
    if (palette == PALETTE_EARTH) {
        // The set is land, the rest ocean: deep far from the coast, shallow near it. How far from the coast, in camera
        // heights: about 1 / (ln 2 times how fast the smooth iteration count climbs across the floor), the distance
        // estimate, so the shallows keep their size next to the drone at every depth.
        float coast = (pixel / unitsPerHeight) / max(0.6931472 * fwidth(nu), 1.0e-30);
        float depth = clamp((log2(coast) + 9.0) / 9.0, 0.0, 1.0);
        color = mix(SURF, SHALLOW, smoothstep(0.0, 0.25, depth));
        color = mix(color, OCEAN, smoothstep(0.2, 0.65, depth));
        color = mix(color, DEEP_OCEAN, smoothstep(0.55, 1.0, depth));
        // Coast finer than a pixel: the set's filaments, as thin as can be, in water as shallow as it gets, so shoals
        color = mix(color, SHOALS, filaments);
        if (inside) {
            color = mix(LAND, BEACH, insideEdge);
        }
    }
    else if (inside) {
        color = mix(INSIDE, fine, insideEdge);
    }
    else {
        color = 0.5 + 0.5 * cos(6.2831853 * (t + vec3(0.0, 0.33, 0.67)));
        // Color bands and filaments finer than a pixel would flicker as noise: they fade to one color instead, the
        // palette's gray average or white (filament_glow)
        color = mix(color, fine, max(smoothstep(0.3, 1.0, bands), filaments));
    }
    fragColor = vec4(color * transmission + fogColor * (1.0 - transmission), 1.0);
}
"""


# ----- exact numbers: complex fixed point, (re, im, bits) for (re + i im) / 2^bits -----

def to_float(value, bits):
    """value / 2^bits as a float, for an integer of any size"""
    shift = max(0, value.bit_length() - 60)
    return math.ldexp(float(value >> shift), shift - bits)


def with_bits(value, bits, new_bits):
    """value / 2^bits in new_bits fixed point, rounded to nearest"""
    if new_bits >= bits:
        return value << (new_bits - bits)
    shift = bits - new_bits
    return (value + (1 << (shift - 1))) >> shift


def exact_difference(a, b):
    """a - b, both (re, im, bits), as a complex float: small numbers, like an offset from a reference"""
    bits = max(a[2], b[2])
    re = with_bits(a[0], a[2], bits) - with_bits(b[0], b[2], bits)
    im = with_bits(a[1], a[2], bits) - with_bits(b[1], b[2], bits)
    return complex(to_float(re, bits), to_float(im, bits))


def exact_difference_scaled(a, b, exponent):
    """a - b, both (re, im, bits), in units of 2^exponent, as a complex float: an offset at any depth"""
    bits = max(a[2], b[2])
    re = with_bits(a[0], a[2], bits) - with_bits(b[0], b[2], bits)
    im = with_bits(a[1], a[2], bits) - with_bits(b[1], b[2], bits)
    return complex(to_float(re, bits + exponent), to_float(im, bits + exponent))


def exact_plus(a, offset):
    """a, (re, im, bits), plus a complex float offset, exactly (an offset is a binary fraction), rounded to a's bits"""
    scale = 1 << a[2]
    return (a[0] + round(Fraction(offset.real) * scale), a[1] + round(Fraction(offset.imag) * scale), a[2])


def needed_bits(units_per_height):
    """The bits that tell apart points 1/1000 of a view apart, under the drone, at units_per_height"""
    return max(0, math.ceil(-math.log2(units_per_height))) + 10


class Mapping:
    """Where the floor is in the complex plane, exactly (see the module's docstring): c at map point p is
    center + units * (p - anchor), center exact, anchor a float64 point near the drone, units = mantissa 2^exponent"""

    def __init__(self, obj):
        re, im, bits = obj[CENTER].split()
        self.center = (int(re, 16), int(im, 16), int(bits))
        self.anchor = tuple(obj[ANCHOR])
        mantissa, exponent = obj[SCALE].split()
        self.mantissa, self.exponent = float(mantissa), int(exponent)
        self.version = obj.get(VERSION, 0)

    @staticmethod
    def write_new(obj, center, units_per_meter, anchor=(0.0, 0.0)):
        """Give obj a mapping: center a complex float (exact as a binary fraction), units_per_meter a float"""
        mantissa, exponent = math.frexp(units_per_meter)
        bits = CENTER_MARGIN + max(0, math.ceil(-math.log2(units_per_meter)))
        exact = exact_plus((0, 0, bits), center)
        obj[CENTER] = f"{exact[0]:#x} {exact[1]:#x} {bits}"
        obj[ANCHOR] = anchor
        obj[SCALE] = f"{mantissa!r} {exponent}"
        obj[VERSION] = obj.get(VERSION, 0) + 1

    def units(self):
        """Complex units per meter, as a float (0 past float64's range)"""
        return math.ldexp(self.mantissa, self.exponent)

    def log2_units(self):
        return math.log2(self.mantissa) + self.exponent

    def c_at(self, map_point):
        """The exact c at a map point, (re, im, bits). The offset from the anchor is units times a float64 distance:
        exact as a fraction, rounded to the center's bits."""
        dx, dy = map_point[0] - self.anchor[0], map_point[1] - self.anchor[1]
        scale = Fraction(self.mantissa) * (Fraction(2) ** (self.exponent + self.center[2]))
        re, im, bits = self.center
        return re + round(scale * Fraction(dx)), im + round(scale * Fraction(dy)), bits

    def loop(self, k, pivot):
        """The drone was scaled by k about pivot (a map point): scale the mapping by k about it too"""
        center = self.c_at(pivot)
        mantissa, exponent = math.frexp(self.mantissa / k)
        self.mantissa, self.exponent = mantissa, self.exponent + exponent
        # Bits for the new scale, growing as the drone dives and shrinking as it climbs
        bits = CENTER_MARGIN + max(0, math.ceil(-self.log2_units()))
        self.center = (with_bits(center[0], center[2], bits), with_bits(center[1], center[2], bits), bits)
        self.anchor = (float(pivot[0]), float(pivot[1]))

    def save(self, obj):
        re, im, bits = self.center
        obj[CENTER] = f"{re:#x} {im:#x} {bits}"
        obj[ANCHOR] = self.anchor
        obj[SCALE] = f"{self.mantissa!r} {self.exponent}"
        self.version = obj.get(VERSION, 0) + 1
        obj[VERSION] = self.version


# ----- the map's object and its mapping -----

_mappings = {}       # object name -> Mapping, the one loops change and drawing reads
_floor = None        # the Floor drawing (its shaders and reference orbit), for one object


def floor_object(scene):
    """The scene's Mandelbrot floor object, or None"""
    name = scene.get(SCENE_KEY) if scene else None
    obj = scene.objects.get(name) if name else None
    return obj if obj is not None and CENTER in obj else None


def mapping_of(obj):
    """obj's Mapping, read again if its properties changed under it (an undo)"""
    mapping = _mappings.get(obj.name)
    if mapping is None or mapping.version != obj.get(VERSION, 0):
        mapping = _mappings[obj.name] = Mapping(obj)
    return mapping


def world_offset(scene):
    """Where the world origin is in the map's frame (see Keep Drone at Origin), as Python floats"""
    references = getattr(scene, "references", None)
    return tuple(references.world_offset) if references else (0.0, 0.0, 0.0)


def setting(obj, name):
    return obj.get(name, DEFAULTS[name])


def loop(scene, k, pivot):
    """From ER_OT_Fly.loop_scale(): the drone was scaled by k about pivot (world space). The scene's floor
    follows, whatever was selected: a floor that didn't would look k times smaller or bigger after the loop."""
    obj = floor_object(scene)
    if obj is None or k == 1.0:
        return
    offset = world_offset(scene)
    mapping = mapping_of(obj)
    mapping.loop(k, (pivot[0] + offset[0], pivot[1] + offset[1]))
    mapping.save(obj)


def select_floor_if_looking_at_it(scene, camera):
    """From ER_OT_Fly.click(), after its click at the view's center: if it selected nothing (but the drone rig)
    and the drone camera looks down at the floor, select the floor's object and make it active, as a click would have if
    its plane were drawn"""
    obj = floor_object(scene)
    if obj is None or camera is None:
        return
    if any(o.name not in RIG_NAMES for o in bpy.context.selected_objects):
        return
    forward = camera.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))
    if forward.z >= 0.0 or camera.matrix_world.translation.z <= obj.matrix_world.translation.z:
        return
    for o in bpy.context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


# ----- reference orbits -----

class OrbitJob:
    """The orbit of 0 under z -> z^2 + c for an exact c, in fixed point of `precision` bits, a chunk at a time, until it
    escapes (|z| > 2) or reaches limit iterations. The values, as complex floats, are all the GPU needs: offsets from
    them are small next to them."""

    def __init__(self, c, precision, limit):
        self.c = (with_bits(c[0], c[2], precision), with_bits(c[1], c[2], precision), precision)
        self.precision = precision
        self.limit = limit
        self.values = [0j]
        self.zr = self.zi = 0
        self.escaped = False
        self.done = False

    def step(self, count):
        """Iterate up to count more; True once done"""
        if self.done:
            return True
        p = self.precision
        cr, ci, zr, zi = self.c[0], self.c[1], self.zr, self.zi
        four = 4 << p
        values = self.values
        for _ in range(count):
            if len(values) > self.limit:
                self.done = True
                break
            zr2, zi2 = (zr * zr) >> p, (zi * zi) >> p
            # The last value escaped: it stays in the orbit, where a pixel that outlasts it rebases
            if len(values) > 1 and zr2 + zi2 > four:
                self.escaped = self.done = True
                break
            zr, zi = zr2 - zi2 + cr, ((zr * zi) >> (p - 1)) + ci
            values.append(complex(to_float(zr, p), to_float(zi, p)))
        self.zr, self.zi = zr, zi
        return self.done


class Search:
    """Looks for a better reference point, a few iterations a frame: the candidate in a disk around the point below the
    drone whose orbit lasts longest. Every pixel whose orbit outlasts the reference's has to go on from its start in
    plain float32, which loses the view's detail at depth, so the reference wants to be as close to the set as can be
    found. The candidates are iterated as offsets from the current reference's orbit, in float64, with rebasing, so the
    search works at any depth."""

    def __init__(self, orbit, center, radius, iterations):
        # A sunflower spiral fills the disk evenly, the center first
        k = np.arange(SEARCH_CANDIDATES)
        spiral = np.sqrt(k / SEARCH_CANDIDATES) * np.exp(1j * k * math.pi * (3.0 - math.sqrt(5.0)))
        self.offsets = center + radius * spiral
        self.orbit = orbit
        self.d = np.zeros(SEARCH_CANDIDATES, complex)
        self.m = np.zeros(SEARCH_CANDIDATES, int)
        self.alive = np.ones(SEARCH_CANDIDATES, bool)
        self.count = np.full(SEARCH_CANDIDATES, iterations)
        self.n = 0
        self.iterations = iterations

    def step(self, steps):
        """Iterate up to steps more. Returns the best candidate's offset from the reference once done, else None."""
        orbit, last = self.orbit, len(self.orbit) - 1
        for _ in range(steps):
            if self.n >= self.iterations or not self.alive.any():
                return self.offsets[int(np.argmax(self.count))]
            z_ref = orbit[np.minimum(self.m, last)]
            self.d = np.where(self.alive, 2.0 * z_ref * self.d + self.d * self.d + self.offsets, self.d)
            self.m += self.alive
            self.n += 1
            z = orbit[np.minimum(self.m, last)] + self.d
            r2 = z.real ** 2 + z.imag ** 2
            escaped = self.alive & (r2 > 4.0)
            self.count[escaped] = self.n
            self.alive &= ~escaped
            rebase = self.alive & ((r2 < self.d.real ** 2 + self.d.imag ** 2) | (self.m >= last))
            self.d[rebase] = z[rebase]
            self.m[rebase] = 0
        return None


def bla_tables(orbit, c_max):
    """Skip-ahead tables for a reference orbit (bilinear approximation, after Zhuoran and Claude Heiland-Allen). From
    reference index m, an offset d steps to 2 Z_m d + d^2 + dc; leaving out d^2 it's A d + B dc, A = 2 Z_m and B = 1,
    close enough while |d| < R = BLA_EPSILON |A|. Two jumps in a row merge into one, A = A2 A1, B = A2 B1 + B2, which
    holds while d holds for the first and its result, at most |A1| |d| + |B1| |dc|, for the second:
    R = min(R1, (R2 - |B1| c_max) / |A1|), for every dc up to c_max. Level k's jumps are 2^k iterations, starting at
    m = 1 + j 2^k, made from level k - 1's in pairs.

    The radii span every depth, so they're kept as log2. Returns the levels packed for the shader, level k at
    2N - 2N / 2^k (N the power of 2 at or above the single steps' count), as an RGBA32F array (A, B) and an R32F one
    (log2 R), BLA_WIDTH wide; the counts; and the biggest radius of a jump (level 1), past which no jump holds. None
    if the orbit is too short."""
    Z = np.asarray(orbit, complex)
    count = len(Z) - 2
    if count < 2:
        return None
    A = 2.0 * Z[1:count + 1]
    levels = [(A, np.ones(count, complex), BLA_EPSILON * np.abs(A))]
    while len(levels[-1][0]) >= 2:
        A, B, R = levels[-1]
        half = len(A) // 2
        A1, B1, R1 = A[0:2 * half:2], B[0:2 * half:2], R[0:2 * half:2]
        A2, B2, R2 = A[1:2 * half:2], B[1:2 * half:2], R[1:2 * half:2]
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            R_ = np.minimum(R1, np.maximum(0.0, (R2 - np.abs(B1) * c_max) / np.abs(A1)))
            A_ = A2 * A1
            B_ = A2 * B1 + B2
        # Past float32 in the texture, or nonsense: no jump
        bad = ~(np.isfinite(R_) & np.isfinite(A_) & np.isfinite(B_)) | (np.abs(A_) > 1e30) | (np.abs(B_) > 1e30)
        R_[bad], A_[bad], B_[bad] = 0.0, 0.0, 0.0
        levels.append((A_, B_, R_))
    span = 2 * (1 << (count - 1).bit_length())
    rows = (span + BLA_WIDTH - 1) // BLA_WIDTH
    ab = np.zeros((rows * BLA_WIDTH, 4), np.float32)
    log_r = np.full(rows * BLA_WIDTH, -1.0e30, np.float32)
    for k, (A, B, R) in enumerate(levels):
        start = span - (span >> k)
        ab[start:start + len(A)] = np.stack([A.real, A.imag, B.real, B.imag], 1)
        with np.errstate(divide='ignore'):
            log_r[start:start + len(A)] = np.maximum(np.log2(R), -1.0e30)
    first = levels[1][2] if len(levels) > 1 else np.zeros(1)
    max_log_r = float(np.log2(first.max())) if first.max() > 0.0 else -1.0e30
    return {"ab": ab, "log_r": log_r, "rows": rows, "count": count, "span": span, "levels": len(levels),
            "max_log_r": max_log_r}


class Reference:
    """A finished reference orbit: its exact c, its values, and their texture once drawn"""

    def __init__(self, job, units_per_height):
        self.c = job.c
        self.precision = job.precision
        self.orbit = np.array(job.values, complex)
        self.length = len(self.orbit)
        self.escaped = job.escaped
        self.units_per_height = units_per_height
        self.bla = None        # skip-ahead tables (see bla_tables()), for c_max at a units per height
        self.rows = (self.length + ORBIT_WIDTH - 1) // ORBIT_WIDTH
        self.data = np.zeros((self.rows * ORBIT_WIDTH, 2), np.float32)
        self.data[:self.length, 0], self.data[:self.length, 1] = self.orbit.real, self.orbit.imag
        self._texture = None

    def texture(self):
        # Made where there's a GPU context, when first drawn
        if self._texture is None:
            buffer = gpu.types.Buffer('FLOAT', self.data.size, self.data.ravel())
            self._texture = gpu.types.GPUTexture((ORBIT_WIDTH, self.rows), format='RG32F', data=buffer)
        return self._texture

    def update_bla(self, units_per_height):
        """Skip-ahead tables for offsets up to BLA_NEAR heights at twice this view's scale, rebuilt once the view has
        zoomed a factor of 2 either way (zoomed in, they'd hold, but jump less than they could)"""
        bla = self.bla
        if bla is not None and 0.5 < units_per_height / bla["units_per_height"] < 2.0:
            return
        c_max = 2.0 * BLA_NEAR * units_per_height
        self.bla = bla_tables(self.orbit, c_max) or {"levels": 0}
        self.bla.update({"units_per_height": units_per_height, "c_max": c_max, "textures": None})

    def bla_textures(self):
        bla = self.bla
        if bla["textures"] is None:
            size = (BLA_WIDTH, bla["rows"])
            ab = gpu.types.Buffer('FLOAT', bla["ab"].size, bla["ab"].ravel())
            log_r = gpu.types.Buffer('FLOAT', bla["log_r"].size, bla["log_r"])
            bla["textures"] = (gpu.types.GPUTexture(size, format='RGBA32F', data=ab),
                               gpu.types.GPUTexture(size, format='R32F', data=log_r))
        return bla["textures"]


# ----- drawing -----

def ring_mesh():
    """The floor around the camera, in camera heights: a fan right below it, then rings sqrt(2) apart out to 2^24
    heights, past any clip end. Each triangle spans less than a factor of sqrt(2) in distance from the camera, so
    float32 interpolates the floor point closely everywhere; one big quad, its corners kilometers away, would blur the
    pattern under the drone."""
    segments = 96
    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles)], 1)
    radii = [2.0 ** (k / 2) for k in range(-12, 49)]
    positions = [(0.0, 0.0)] + [tuple(r * v) for r in radii for v in ring]
    indices = [(0, 1 + i, 1 + (i + 1) % segments) for i in range(segments)]
    for k in range(len(radii) - 1):
        inner, outer = 1 + k * segments, 1 + (k + 1) * segments
        for i in range(segments):
            j = (i + 1) % segments
            indices += [(inner + i, outer + i, outer + j), (inner + i, outer + j, inner + j)]
    return positions, indices


# The floor shader's values for a draw, in a uniform buffer: as push constants they came to more than the 128 bytes
# every GPU is sure to take (Blender warned). After the matrix, rows of 4 (std140), each value in a place the shader
# reads it from by its name, through a define; the rows in FLOOR_INT_ROWS hold ints.
FLOOR_ROWS = ("view", "look", "fog", "detail", "counts", "bla")
FLOOR_INT_ROWS = ("counts", "bla")
FLOOR_VALUES = {
    "cameraFromReference": ("view", "xy"), "height": ("view", "z"), "unitsPerHeight": ("view", "w"),
    "fogDistance": ("look", "x"), "colorDensity": ("look", "y"), "colorPhase": ("look", "z"),
    "filamentGlow": ("look", "w"),
    "fogColor": ("fog", "xyz"), "iterationsPerDecade": ("fog", "w"),
    "pixelRef": ("detail", "x"), "blaCMax2": ("detail", "y"), "blaMaxLogR": ("detail", "z"),
    "maxIterations": ("counts", "x"), "minIterations": ("counts", "y"), "orbitLength": ("counts", "z"),
    "palette": ("counts", "w"),
    "scaleExponent": ("bla", "x"), "blaLevels": ("bla", "y"), "blaCount": ("bla", "z"), "blaSpan": ("bla", "w"),
}


def floor_values_buffer(view_projection, values):
    """The floor shader's uniform buffer for a draw (see FLOOR_VALUES): the matrix by columns, as GLSL keeps it, then
    each value in its place"""
    data = np.zeros(16 + 4 * len(FLOOR_ROWS), np.float32)
    ints = data.view(np.int32)
    data[:16] = np.array(view_projection, np.float32).T.ravel()
    for name, value in values.items():
        row, parts = FLOOR_VALUES[name]
        start = 16 + 4 * FLOOR_ROWS.index(row)
        for part, v in zip(parts, value if len(parts) > 1 else (value,)):
            if row in FLOOR_INT_ROWS:
                ints[start + "xyzw".index(part)] = int(v)
            else:
                data[start + "xyzw".index(part)] = float(v)
    return gpu.types.GPUUniformBuf(data)


def make_shader(smooth_filaments=False):
    info = gpu.types.GPUShaderCreateInfo()
    info.define("SMOOTH_FILAMENTS", "1" if smooth_filaments else "0")
    info.define("PULL", repr(PULL))
    info.define("ORBIT_WIDTH", str(ORBIT_WIDTH))
    info.define("BAILOUT", repr(BAILOUT))
    info.define("FOG_CUTOFF", repr(FOG_CUTOFF))
    info.define("EXTINCTION", "vec3(%r, %r, %r)" % EXTINCTION)
    info.define("INSIDE", "vec3(0.0, 0.0, 0.0)")
    info.define("PALETTE_EARTH", str(PALETTES.index("earth")))
    for name, rgb in EARTH_COLORS.items():
        info.define(name, "vec3(%r, %r, %r)" % srgb_to_linear(rgb))
    info.define("SCALED_BELOW", str(SCALED_BELOW))
    info.define("BLA_WIDTH", str(BLA_WIDTH))
    rows = " ".join(f"{'ivec4' if row in FLOOR_INT_ROWS else 'vec4'} {row};" for row in FLOOR_ROWS)
    info.typedef_source(f"struct FloorValues {{ mat4 view_projection; {rows} }};")
    info.uniform_buf(0, "FloorValues", "floorValues")
    info.define("viewProjection", "floorValues.view_projection")
    for name, (row, parts) in FLOOR_VALUES.items():
        info.define(name, f"floorValues.{row}.{parts}")
    info.sampler(0, 'FLOAT_2D', "orbit")
    info.sampler(1, 'FLOAT_2D', "blaAB")
    info.sampler(2, 'FLOAT_2D', "blaLogR")
    info.vertex_in(0, 'VEC2', "pos")
    interface = gpu.types.GPUStageInterfaceInfo("mandelbrot_floor_interface")
    interface.smooth('VEC2', "offset")
    info.vertex_out(interface)
    info.fragment_out(0, 'VEC4', "fragColor")
    info.vertex_source(VERTEX_SOURCE)
    info.fragment_source(FRAGMENT_SOURCE)
    return gpu.shader.create_from_info(info)


def make_composite_shader():
    info = gpu.types.GPUShaderCreateInfo()
    info.define("PULL", repr(PULL))
    info.push_constant('MAT4', "viewProjection")
    info.push_constant('FLOAT', "height")
    info.sampler(0, 'FLOAT_2D', "floorColor")
    info.vertex_in(0, 'VEC2', "pos")
    interface = gpu.types.GPUStageInterfaceInfo("mandelbrot_composite_interface")
    interface.smooth('VEC4', "clip")
    info.vertex_out(interface)
    info.fragment_out(0, 'VEC4', "fragColor")
    info.vertex_source(COMPOSITE_VERTEX_SOURCE)
    info.fragment_source(COMPOSITE_FRAGMENT_SOURCE)
    return gpu.shader.create_from_info(info)


class Floor:
    """Draws one object's floor: shaders, the reference orbit and its search, the lower resolution pass"""

    def __init__(self, obj):
        self.object_name = obj.name
        self.shaders = {False: make_shader(False), True: make_shader(True)}
        self.composite_shader = make_composite_shader()
        positions, indices = ring_mesh()
        self.batch = batch_for_shader(self.shaders[False], 'TRIS', {"pos": positions}, indices=indices)
        self.composite_batch = batch_for_shader(self.composite_shader, 'TRIS', {"pos": positions}, indices=indices)
        self.offscreens = {}       # by size, for the lower resolution pass
        self.values_buffer = None  # the shader's values for the last draw (see FLOOR_VALUES)
        self.resolution = None     # auto_resolution's, a multiple of RESOLUTION_STEP
        self.fast_frames = 0
        self.recent_missed = collections.deque(maxlen=RECENT_FRAMES)
        self.raise_wait = FAST_FRAMES_TO_RAISE
        self.raised = False        # the last step was up, so slowing down now means it was one too far
        self.reference = None
        self.job = None            # the next reference's orbit, being iterated
        self.job_needed = False    # the current reference can't stay (too far, short or imprecise), so take the next
        self.search = None
        self.searched = None       # where the last finished search was, so a still drone doesn't keep searching
        self.last_update = None
        self.last_draw = None
        self.frame_time = None
        self.max_iterations = DEFAULTS["iterations"]
        self.zoom_log10 = 0.0

    # ----- the reference orbit -----

    def replace_now(self, below, units_per_height):
        """A reference at the point below the drone, iterated all at once"""
        job = OrbitJob(below, needed_bits(units_per_height) + ORBIT_MARGIN, self.max_iterations)
        job.step(MAX_ITERATIONS + 1)
        self.reference = Reference(job, units_per_height)
        self.job = self.search = None
        self.searched = None

    def update_reference(self, below, units_per_height):
        """Keep a reference near below (the exact c under the drone), searching for a better one all along"""
        reference = self.reference
        if reference is None:
            self.replace_now(below, units_per_height)
            return
        from_reference = exact_difference(below, reference.c)
        reach = abs(from_reference) / units_per_height
        precision_left = reference.precision - needed_bits(units_per_height)
        if reach > REFERENCE_LIMIT or precision_left < ORBIT_MIN_MARGIN:
            self.replace_now(below, units_per_height)
            return
        # Too far, too short for the iterations the zoom needs now, or running low on bits: replace it as soon as the
        # next one is ready
        self.job_needed = (reach > REFERENCE_REACH or precision_left < ORBIT_MARGIN // 2
                           or (not reference.escaped and reference.length - 1 < self.max_iterations))
        if self.job is not None:
            if self.job.step(ORBIT_STEPS):
                candidate = Reference(self.job, units_per_height)
                if self.job_needed or candidate.length >= reference.length:
                    self.reference = candidate
                self.job = None
            return
        if self.search is None:
            if self.searched is not None and not self.job_needed:
                c, units = self.searched
                if (abs(exact_difference(below, c)) < SEARCH_MOVE * units
                        and 1.0 / SEARCH_ZOOM < units_per_height / units < SEARCH_ZOOM):
                    return
            self.search = Search(reference.orbit, from_reference, SEARCH_RADIUS * units_per_height,
                                 self.max_iterations)
            self.searched = (below, units_per_height)
        found = self.search.step(SEARCH_STEPS)
        if found is not None:
            self.search = None
            # The reference's c in the new orbit's bits first, so the offset isn't rounded to the old ones
            precision = needed_bits(units_per_height) + ORBIT_MARGIN
            c = reference.c
            base = (with_bits(c[0], c[2], precision), with_bits(c[1], c[2], precision), precision)
            self.job = OrbitJob(exact_plus(base, complex(found)), precision, self.max_iterations)

    def update(self, scene, obj, mapping):
        """Once a frame, whichever view draws first"""
        now = time.perf_counter()
        if self.last_update is not None and now - self.last_update < 0.004:
            return
        self.last_update = now
        camera = bpy.data.objects.get(CAMERA_NAME)
        if camera is None:
            return
        offset = world_offset(scene)
        location = camera.matrix_world.translation
        height = location.z - obj.matrix_world.translation.z
        if height <= 0.0:
            return
        units_per_height = mapping.units() * height
        if units_per_height <= 0.0:
            return
        self.zoom_log10 = math.log10(SET_WIDTH / 2.0) - math.log10(units_per_height)
        self.max_iterations = int(min(MAX_ITERATIONS, setting(obj, "iterations")
                                      + setting(obj, "iterations_per_decade") * max(self.zoom_log10, 0.0)))
        below = mapping.c_at((location.x + offset[0], location.y + offset[1]))
        self.update_reference(below, units_per_height)
        if self.reference is not None and setting(obj, "skip_ahead"):
            self.reference.update_bla(units_per_height)

    # ----- resolution -----

    def current_resolution(self, obj):
        top = min(1.0, max(MIN_RESOLUTION, float(setting(obj, "resolution"))))
        if not setting(obj, "auto_resolution"):
            return top
        if self.resolution is None or self.resolution > top:
            self.resolution = top
        return self.resolution

    def adjust_resolution(self, obj, interval):
        """auto_resolution, after each frame of interval seconds (see MISSED_FRAME)"""
        if not setting(obj, "auto_resolution"):
            return
        top = min(1.0, max(MIN_RESOLUTION, float(setting(obj, "resolution"))))
        current = self.current_resolution(obj)
        missed = interval > MISSED_FRAME
        self.recent_missed.append(missed)
        self.fast_frames = 0 if missed else self.fast_frames + 1
        if sum(self.recent_missed) >= MISSED_TO_LOWER and current > MIN_RESOLUTION:
            lower = round((current - RESOLUTION_STEP) / RESOLUTION_STEP) * RESOLUTION_STEP
            self.resolution = max(MIN_RESOLUTION, lower)
            self.raise_wait = min(MAX_RAISE_WAIT, self.raise_wait * 2) if self.raised else FAST_FRAMES_TO_RAISE
            self.raised = False
            self.recent_missed.clear()
            self.fast_frames = 0
        elif self.fast_frames >= self.raise_wait and current < top:
            self.resolution = min(top, current + RESOLUTION_STEP)
            self.raised = True
            self.fast_frames = 0

    # ----- per view -----

    def draw(self, context, scene, obj, mapping):
        region_data = context.region_data
        self.update(scene, obj, mapping)
        if self.reference is None:
            return
        # Frame time, between draws of the view looking through the drone camera (or any view, without one)
        if region_data.view_perspective == 'CAMERA' or not self.frame_time:
            now = time.perf_counter()
            if self.last_draw is not None:
                interval = now - self.last_draw
                self.frame_time = interval if self.frame_time is None or interval > 0.5 else (
                    0.9 * self.frame_time + 0.1 * interval)
                # Only frames drawn one after another, as in flight: not a redraw after a pause
                if interval < 0.1:
                    self.adjust_resolution(obj, interval)
            self.last_draw = now

        view = region_data.view_matrix
        camera_location = view.inverted().translation
        height = camera_location.z - obj.matrix_world.translation.z
        if height <= 0.0:
            return
        units_per_height = mapping.units() * height
        if units_per_height <= 0.0:
            return
        # The view and projection without the camera's translation: the floor is drawn from the camera, in camera
        # heights, so its numbers are small wherever the drone is
        view_projection = region_data.window_matrix @ view.to_3x3().to_4x4()

        # Offsets go to the GPU in units of 2^exponent, about the view's size under the drone (see the module's
        # docstring)
        exponent = math.floor(mapping.log2_units() + math.log2(height))
        units_scaled = math.ldexp(mapping.mantissa * height, mapping.exponent - exponent)
        offset = world_offset(scene)
        here = mapping.c_at((camera_location.x + offset[0], camera_location.y + offset[1]))
        from_reference = exact_difference_scaled(here, self.reference.c, exponent)
        camera = bpy.data.objects.get(CAMERA_NAME)
        drone_scale = camera.matrix_world.to_scale()[0] if camera else 1.0
        fog_color = tuple(scene.get("fog_color", (0.114, 0.352, 0.913)))[:3]

        resolution = self.current_resolution(obj)
        region = context.region
        width, height_pixels = max(1, round(region.width * resolution)), max(1, round(region.height * resolution))
        offscreen = None
        if resolution < 1.0:
            offscreen = self.offscreens.get((width, height_pixels))
            if offscreen is None:
                # Kept for each size auto_resolution moves between, a handful
                if len(self.offscreens) > 8:
                    self.offscreens = {}
                offscreen = self.offscreens[(width, height_pixels)] = gpu.types.GPUOffScreen(
                    width, height_pixels, format='RGBA16F')

        # What a pixel right under the camera covers (the least any does): the camera height times a pixel's angle
        # (from the projection), along both axes as fwidth() measures it
        pixel_angle = 2.0 / (region_data.window_matrix[1][1] * height_pixels)
        bla = self.reference.bla if setting(obj, "skip_ahead") else None
        if bla and bla["levels"]:
            ab_texture, log_r_texture = self.reference.bla_textures()
            c_max_scaled = math.ldexp(bla["c_max"], -exponent)
        else:
            # The shader still binds something there
            bla = None
            ab_texture = log_r_texture = self.reference.texture()
        palette = setting(obj, "palette")
        # Kept until the next draw, past this one's use of it
        self.values_buffer = floor_values_buffer(view_projection, {
            "cameraFromReference": (from_reference.real, from_reference.imag),
            "height": height,
            "unitsPerHeight": units_scaled,
            "fogDistance": float(scene.get("fog_distance", 23000.0)) * drone_scale,
            "colorDensity": float(setting(obj, "color_density")),
            "colorPhase": (time.time() * float(setting(obj, "color_speed"))) % 1.0,
            "filamentGlow": float(setting(obj, "filament_glow")),
            "palette": PALETTES.index(palette) if palette in PALETTES else 0,
            "maxIterations": self.max_iterations,
            "minIterations": int(setting(obj, "iterations")),
            "iterationsPerDecade": float(setting(obj, "iterations_per_decade")),
            "pixelRef": math.sqrt(2.0) * pixel_angle * units_scaled,
            "scaleExponent": exponent,
            "orbitLength": self.reference.length,
            "fogColor": fog_color,
            "blaLevels": bla["levels"] if bla else 0,
            "blaCount": bla["count"] if bla else 0,
            "blaSpan": bla["span"] if bla else 0,
            "blaCMax2": c_max_scaled ** 2 if bla else 0.0,
            "blaMaxLogR": bla["max_log_r"] if bla else -1.0e30,
        })
        shader = self.shaders[bool(setting(obj, "smooth_filaments"))]
        shader.bind()
        shader.uniform_block("floorValues", self.values_buffer)
        shader.uniform_sampler("orbit", self.reference.texture())
        shader.uniform_sampler("blaAB", ab_texture)
        shader.uniform_sampler("blaLogR", log_r_texture)
        if offscreen is None:
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.depth_mask_set(True)
            self.batch.draw(shader)
        else:
            # The set at the lower resolution, on the fog color (all the floor's horizon is), then scaled up to the
            # view by drawing the floor again, which tests and writes the view's depth
            with offscreen.bind():
                gpu.state.active_framebuffer_get().clear(color=(*fog_color, 1.0))
                gpu.state.depth_test_set('NONE')
                self.batch.draw(shader)
            composite = self.composite_shader
            composite.bind()
            composite.uniform_float("viewProjection", view_projection)
            composite.uniform_float("height", height)
            composite.uniform_sampler("floorColor", offscreen.texture_color)
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.depth_mask_set(True)
            self.composite_batch.draw(composite)
        gpu.state.depth_mask_set(False)
        gpu.state.depth_test_set('NONE')

    def draw_info(self, obj):
        if not setting(obj, "show_info") or self.reference is None:
            return
        frame = f"{self.frame_time * 1000:.1f} ms/frame" if self.frame_time else ""
        if self.zoom_log10 > math.log10(FLOAT64_ZOOM):
            frame += "  past float64's range: climb back up"
        text = (f"Mandelbrot  zoom 1e{self.zoom_log10:.1f}  iterations {self.max_iterations}  "
                f"resolution {self.current_resolution(obj):.2f}{' auto' if setting(obj, 'auto_resolution') else ''}  "
                f"reference {self.reference.length - 1}  {frame}")
        blf.size(0, 13)
        blf.color(0, 1.0, 1.0, 1.0, 0.85)
        blf.position(0, 12, 12, 0)
        blf.draw(0, text)


def floor_for(obj):
    """The Floor drawing obj, made on the first draw (it needs a GPU context)"""
    global _floor
    if _floor is None or _floor.object_name != obj.name:
        _floor = Floor(obj)
    return _floor


def draw_floor():
    context = bpy.context
    obj = floor_object(context.scene)
    if obj is None or context.region_data is None:
        return
    floor_for(obj).draw(context, context.scene, obj, mapping_of(obj))


def draw_info():
    obj = floor_object(bpy.context.scene)
    if obj is not None and _floor is not None and _floor.object_name == obj.name:
        _floor.draw_info(obj)


def redraw_for_color_cycling():
    """Keeps color cycling moving when nothing else redraws the views"""
    obj = floor_object(bpy.context.scene)
    if obj is None or not setting(obj, "color_speed"):
        return 0.5
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
    return 1.0 / 60.0


@bpy.app.handlers.persistent
def forget(_dummy=None):
    """After loading a file: another file's mappings and floor don't apply"""
    global _floor
    _mappings.clear()
    _floor = None


_handles = []


def register():
    _handles.append(bpy.types.SpaceView3D.draw_handler_add(draw_floor, (), 'WINDOW', 'POST_VIEW'))
    _handles.append(bpy.types.SpaceView3D.draw_handler_add(draw_info, (), 'WINDOW', 'POST_PIXEL'))
    bpy.app.timers.register(redraw_for_color_cycling, persistent=True)
    if forget not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(forget)


def unregister():
    for handle in _handles:
        bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
    _handles.clear()
    if bpy.app.timers.is_registered(redraw_for_color_cycling):
        bpy.app.timers.unregister(redraw_for_color_cycling)
    if forget in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(forget)
    forget()
