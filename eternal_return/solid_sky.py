"""Solid mode's sky (tools/sky.py) shown only in Solid mode, and Solid mode's fog.

Workbench can't draw a sky, so a map's Solid sky is six image empties around the drone camera, the one thing Solid
mode draws unlit. But Blender draws image empties as overlays in every shading mode, and in the Rendered viewport
they'd cover the world's own sky (which the fog fades into), and EEVEE's see-through objects too, since those write no
depth for the empties to hide behind. So the Solid_sky collection is turned off in viewports whenever the view you
fly in is in Material Preview or Rendered, and back on in Solid or Wireframe. Blender says when any 3D view's shading
changes (the message bus), and files are checked as they load, since subscriptions don't outlive a file.

Workbench has no fog either (only grey volume smoke, tools/fog_look.py), so the fog is drawn here, over the meshes, in
every perspective 3D view in Solid mode (the user, 2026-11-16: "where fog grows and shrinks with the drones scale. i
want this for both solid and render mode if possible"). Add-on drawing goes to Blender's overlay layer, which is laid
over the meshes by its alpha, one for all three channels, and is depth-tested against them. So the fog is see-through
layers of fog at growing distances from the eye, each drawn only where the meshes are farther: a pixel is covered by
every layer nearer than its mesh, and they add up to the fog that far. The layers are spaced so each covers an equal
share of what's still seen, and each pixel's layer is moved a share of the way to the next by a fixed noise pattern
(dithering), so their steps don't show as rings. Distances are the scene's fog_distance times the drone camera's scale,
as Rendered's are (fog_look.py), so the fog grows and shrinks with the drone and loops look the same.

Rendered's fog fades each of a surface's channels at its own distance while the sky's light builds up at another, so
far things go dark blue before they're gone; with one alpha, Solid can't fade a channel alone. So the layers fade the
surface at the channels' middle distance (their geometric mean), and each layer's color puts back what Rendered has
there besides that fading surface: the sky's light, and the slower channels of a mid-grey surface (MEAN_SURFACE).

Every layer covers the background too, so behind everything the sky is drawn again here, the way the Solid sky's
pictures show it (kept the same as tools/sky.py's default sky: the sun's disc and glow on the fog color, which is
the same everywhere else),
from the scene's sun lamp and fog_color. Maps with a plain sky (no Solid_sky) end in the fog color, which their background is. The Earth's maps
are left alone: their fog is the Earth's air in meters (sphere.py), drawn with the sphere.

Measured (2026-11-16, M1 Air, 2560 x 1506 view): 16 layers add about 1 ms of GPU time to a draw; 24 added 4.7.
"""
import functools
import math

import bpy
from bpy.app.handlers import persistent

from . import background

COLLECTION = "Solid_sky"
_owner = object()

FOG_GROUP = "Fog"                   # fog_look.py's shader group: a map has Rendered fog
FOG_LAYERS = 16
# Rendered's fog, in fog distances (fog_look.py): each channel's surface light fades to 1 / e (red, green, blue), and
# the sky's light builds up to 1 - 1 / e: the stylized look and Earth's air
STYLIZED = ((0.18, 0.35, 1.8), (2.6, 2.6, 2.6))
EARTH = ((0.349, 0.256, 0.179), (0.349, 0.256, 0.179))
MEAN_SURFACE = 0.5                  # the surface brightness whose slower channels the layers put back
# The sky, as tools/sky.py makes it: the sun at its lamp's compass direction this many degrees up, its disc's radius
# in degrees and its edge's half width, its brightness over the sky, and its glow's (strength, width) terms, each
# strength exp((cos angle - 1) / width)
SUN_HEIGHT = 40.0
SUN_RADIUS = 1.25
SUN_EDGE = 0.06
SUN_BRIGHTNESS = 12.0
GLOW = ((0.55, 0.003), (0.3, 0.03), (0.12, 0.25))


def flying_shading():
    """The shading of the view you fly in: the biggest 3D view looking through the scene's camera, else the biggest
    3D view. None without a 3D view."""
    views = []
    window_manager = bpy.context.window_manager
    for window in (window_manager.windows if window_manager else []):
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                through_camera = space.region_3d is not None and space.region_3d.view_perspective == 'CAMERA'
                views.append((through_camera, area.width * area.height, space.shading.type))
    return max(views)[2] if views else None


def update(*_args):
    """Turn the Solid sky off in viewports while the flying view is in Material Preview or Rendered, else on. Only on
    a change: switching a collection off rebuilds Blender's dependency graph."""
    collection = bpy.data.collections.get(COLLECTION)
    shading = flying_shading()
    if collection is None or shading is None:
        return
    hide = shading in {'MATERIAL', 'RENDERED'}
    if collection.hide_viewport != hide:
        collection.hide_viewport = hide


def fog_look(scene):
    """The scene's Rendered fog as (extinction, airlight) per channel in fog distances (STYLIZED or EARTH), or None
    for a map without fog, an Earth map's (its fog is drawn with the sphere), or a Mandelbrot map's (its floor, all
    there is, has its own fog, and every layer would be drawn under it for nothing: 4 ms a draw)"""
    group = bpy.data.node_groups.get(FOG_GROUP)
    if group is None or "fog_distance" not in scene or "fog_color" not in scene:
        return None
    from . import mandelbrot, sphere
    if mandelbrot.floor_object(scene) is not None:
        return None
    if any(sphere.setting(obj, "planet") or sphere.setting(obj, "tiles") for obj in sphere.sphere_objects(scene)):
        return None
    # As fog_look.is_earth(): the group says, or, built before it said, by the Invert node only Earth's air has
    earth = group.get("earth", any(node.bl_idname == 'ShaderNodeInvert' for node in group.nodes))
    return EARTH if earth else STYLIZED


def mean(values):
    """Geometric mean"""
    return math.prod(values) ** (1.0 / len(values))


@functools.lru_cache(maxsize=8)
def layer_ends(look):
    """Each layer's end (see fog_layers()) in fog distances: they're the same share of any fog distance, so they're
    found once a look, not every draw (40 bisection steps a layer were half a millisecond of every frame, 2026-11-17)"""
    extinction, airlight = look
    fade, build = mean(extinction), mean(airlight)

    def along(d):
        return 1.0 - 0.5 * (math.exp(-d / fade) + math.exp(-d / build))

    def at(share):
        """The distance that far along (bisection: along() only grows)"""
        low, high = 0.0, 50.0 * max(fade, build)
        for _ in range(40):
            middle = 0.5 * (low + high)
            low, high = (middle, high) if along(middle) < share else (low, middle)
        return 0.5 * (low + high)

    return tuple(at(1.0 - 0.5 / (FOG_LAYERS + 1) if i == FOG_LAYERS else (i + 1) / (FOG_LAYERS + 1))
                 for i in range(FOG_LAYERS + 1))


def fog_layers(distance, near, look, color):
    """The fog's layers for fog distance distance (world units), starting near the eye: (start, end, alpha, color),
    each drawn at a distance between its start and end (dithered), alpha its cover and color its own (not
    premultiplied). A surface fades at the channels' middle distance and the sky's light builds up at its own, often
    much slower (STYLIZED), so the layers are spaced by how far along both are, half each: an even share of the way
    each, so the far silhouettes Rendered keeps until the sky's light is whole are there too. The last, half a share
    farther, covers what's left with the fog color."""
    extinction, airlight = look
    fade = mean(extinction) * distance
    ends = layer_ends(look)

    def covered(d):
        """What Rendered adds to a surface seen through d of fog, besides the surface fading as the layers fade it"""
        seen = math.exp(-d / fade)
        return [c * (1.0 - math.exp(-d / (a * distance))) + MEAN_SURFACE * max(0.0, math.exp(-d / (e * distance)) - seen)
                for c, e, a in zip(color, extinction, airlight)]

    layers = []
    drawn, start, kept = [0.0, 0.0, 0.0], near, 1.0
    for i in range(FOG_LAYERS + 1):
        last = i == FOG_LAYERS
        end = max(start, ends[i] * distance)
        # The surface keeps exp(-d / fade): this layer covers its share of what the layers before it left
        alpha = 1.0 if last else 1.0 - math.exp(-end / fade) / kept
        if alpha <= 0.0:
            continue
        kept = math.exp(-end / fade)
        target = list(color) if last else covered(end)
        # The overlay so far is drawn (premultiplied); this layer's color makes it the target
        own = [max(0.0, min(1.0, (t - d * (1.0 - alpha)) / alpha)) for t, d in zip(target, drawn)]
        layers.append((start, end, alpha, own))
        start = end
        drawn = [d * (1.0 - alpha) + o * alpha for d, o in zip(drawn, own)]
    return layers


def sun_object(scene):
    """The scene's sun lamp, as tools/solid_sun.py finds it: one named Sun, else the first of type SUN, else None"""
    suns = [o for o in scene.objects if o.type == 'LIGHT' and o.data.type == 'SUN']
    return next((o for o in suns if o.name == "Sun"), suns[0] if suns else None)


# The view's ray through a pixel, from its projection P (GL convention, as Blender's window_matrix): (P00, P11, P02,
# P12) in uRay. The eye looks along -z.
RAY_SOURCE = """
vec3 viewRay(vec2 ndc)
{
    return normalize(vec3((ndc.x - uRay.z) / uRay.x, (ndc.y - uRay.w) / uRay.y, -1.0));
}
"""

FOG_FRAGMENT = """
void main()
{
    // Interleaved gradient noise: a fixed pattern on the screen, how far along toward its end this pixel's layer is
    float n = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    float z = viewRay(ndc).z * mix(uLayer.x, uLayer.y, n);
    // Window depth of view depth z, from P22 and P23
    gl_FragDepth = clamp(0.5 - 0.5 * (uDepth.x * z + uDepth.y) / z, 0.0, 1.0);
    fragColor = vec4(uColor.rgb * uLayer.z, uLayer.z);
}
"""

SKY_FRAGMENT = """
void main()
{
    vec3 d = normalize(mat3(uWorldFromView) * viewRay(ndc));
    float f = dot(d, uSun.xyz);
    float disc = smoothstep(uSun.w, uHorizon.w, f);
    fragColor = vec4(uHorizon.rgb + (GLOW_SUM + SUN_BRIGHTNESS * disc) * uSunColor.rgb, 1.0);
}
"""


class Fog:
    """Solid mode's fog and sky shaders (see the module's docstring), drawn as full-screen triangles"""

    def __init__(self):
        import gpu
        from gpu_extras.batch import batch_for_shader
        interface = gpu.types.GPUStageInterfaceInfo("solid_fog_interface")
        interface.smooth('VEC2', "ndc")

        fog = gpu.types.GPUShaderCreateInfo()
        fog.push_constant('VEC4', "uRay")
        fog.push_constant('VEC4', "uDepth")         # P22, P23, the layer plane's window depth
        fog.push_constant('VEC4', "uLayer")         # start, end, alpha
        fog.push_constant('VEC4', "uColor")
        fog.vertex_in(0, 'VEC2', "pos")
        fog.vertex_out(interface)
        fog.fragment_out(0, 'VEC4', "fragColor")
        # Every pixel's depth is at least its triangle's (the layer's start, at the view's widest corner), so the GPU
        # can still drop the pixels a mesh is in front of before shading them
        fog.depth_write('GREATER')
        fog.vertex_source("void main() { ndc = pos; gl_Position = vec4(pos, uDepth.z, 1.0); }")
        fog.fragment_source(RAY_SOURCE + FOG_FRAGMENT)
        self.fog = gpu.shader.create_from_info(fog)

        sky = gpu.types.GPUShaderCreateInfo()
        sky.push_constant('MAT4', "uWorldFromView")
        sky.push_constant('VEC4', "uRay")
        sky.push_constant('VEC4', "uHorizon")       # fog color, cosine where the sun's disc is whole
        sky.push_constant('VEC4', "uSun")           # toward the sun, cosine where its disc starts
        sky.push_constant('VEC4', "uSunColor")
        sky.define("SUN_BRIGHTNESS", repr(SUN_BRIGHTNESS))
        sky.define("GLOW_SUM", "(" + " + ".join(f"{s!r} * exp((f - 1.0) / {w!r})" for s, w in GLOW) + ")")
        sky.vertex_in(0, 'VEC2', "pos")
        sky.vertex_out(interface)
        sky.fragment_out(0, 'VEC4', "fragColor")
        # At the far plane, so it's only drawn where nothing else is
        sky.vertex_source("void main() { ndc = pos; gl_Position = vec4(pos, 1.0, 1.0); }")
        sky.fragment_source(RAY_SOURCE + SKY_FRAGMENT)
        self.sky = gpu.shader.create_from_info(sky)

        triangle = {"pos": [(-1.0, -1.0), (3.0, -1.0), (-1.0, 3.0)]}
        self.fog_batch = batch_for_shader(self.fog, 'TRIS', triangle)
        self.sky_batch = batch_for_shader(self.sky, 'TRIS', triangle)

    def draw(self, context, look):
        import gpu
        scene = context.scene
        camera = bpy.data.objects.get("Drone_Camera")
        P = context.region_data.window_matrix
        if camera is None or P[3][3] != 0.0:        # orthographic: no eye to measure from
            return
        ray = (P[0][0], P[1][1], P[0][2], P[1][2])
        near = P[2][3] / (P[2][2] - 1.0)
        # The view's widest ray, at a corner: a layer's plane goes where its start is along it, nearest the eye
        corner = max(math.hypot((sx - ray[2]) / ray[0], (sy - ray[3]) / ray[1])
                     for sx in (-1.0, 1.0) for sy in (-1.0, 1.0))
        cosine = 1.0 / math.sqrt(1.0 + corner * corner)
        distance = float(scene["fog_distance"]) * camera.matrix_world.to_scale().x
        if not distance > 0.0:
            return
        color = tuple(scene["fog_color"])[:3]

        gpu.state.depth_test_set('LESS')
        gpu.state.depth_mask_set(False)
        gpu.state.blend_set('ALPHA_PREMULT')
        self.fog.bind()
        self.fog.uniform_float("uRay", ray)
        for start, end, alpha, own in fog_layers(distance, 2.0 * near, look, color):
            z = -start * cosine
            plane = max(-1.0, min(1.0, (P[2][2] * z + P[2][3]) / -z))
            self.fog.uniform_float("uDepth", (P[2][2], P[2][3], plane, 0.0))
            self.fog.uniform_float("uLayer", (start, end, alpha, 0.0))
            self.fog.uniform_float("uColor", (*own, 1.0))
            self.fog_batch.draw(self.fog)
        gpu.state.blend_set('NONE')

        # The sky the layers covered, on a map with the Solid sky
        sun = sun_object(scene)
        collection = bpy.data.collections.get(COLLECTION)
        if sun is not None and collection is not None and not collection.hide_viewport:
            toward = sun.matrix_world.to_3x3().col[2]
            azimuth = math.atan2(toward.y, toward.x)
            height = math.radians(SUN_HEIGHT)
            radius, edge = math.radians(SUN_RADIUS), math.radians(SUN_EDGE)
            gpu.state.depth_test_set('LESS_EQUAL')
            self.sky.bind()
            self.sky.uniform_float("uWorldFromView", context.region_data.view_matrix.inverted().to_3x3().to_4x4())
            self.sky.uniform_float("uRay", ray)
            self.sky.uniform_float("uHorizon", (*color, math.cos(radius - edge)))
            self.sky.uniform_float("uSun", (math.cos(height) * math.cos(azimuth), math.cos(height) * math.sin(azimuth),
                                            math.sin(height), math.cos(radius + edge)))
            self.sky.uniform_float("uSunColor", (*sun.data.color, 1.0))
            self.sky_batch.draw(self.sky)
        gpu.state.depth_test_set('NONE')


_fog = None


@background.foreground
def draw_fog():
    """POST_VIEW: the fog (see the module's docstring) in a Solid 3D view of a map with fog, with Solid Fog on"""
    global _fog
    context = bpy.context
    space = context.space_data
    scene = context.scene
    if (context.region_data is None or space is None or space.shading.type != 'SOLID'
            or not scene.controls_parameters.solid_fog):
        return
    look = fog_look(scene)
    if look is None:
        return
    if _fog is None:
        _fog = Fog()
    _fog.draw(context, look)


def subscribe():
    bpy.msgbus.clear_by_owner(_owner)
    bpy.msgbus.subscribe_rna(key=(bpy.types.View3DShading, "type"), owner=_owner, args=(), notify=update)


@persistent
def on_load(_filepath=None):
    """load_post: listen again, and match the loaded file's views once its windows exist"""
    subscribe()
    bpy.app.timers.register(lambda: update() or None, first_interval=0.1)


_handles = []


def register():
    if on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_load)
    subscribe()
    # Before the Mandelbrot and sphere floors' handlers (__init__ registers this first), so their floors, fogged by
    # their own shaders, are drawn over the layers
    _handles.append(bpy.types.SpaceView3D.draw_handler_add(draw_fog, (), 'WINDOW', 'POST_VIEW'))


def unregister():
    if on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_load)
    bpy.msgbus.clear_by_owner(_owner)
    while _handles:
        bpy.types.SpaceView3D.draw_handler_remove(_handles.pop(), 'WINDOW')
