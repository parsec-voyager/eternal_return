"""The sky seen while flying: the sun's disc and glow on the fog color, which is the same everywhere else, in both Solid
mode and the Rendered viewport (the user, 2026-11-16: no clouds, "i only want the blue sky and fog"; then a sky
"uniform and match the fog"; then the sun back, but "no color change, just uniform blue that matches the fog"; then
"give the sun the glow it had before rather than just a white circle"). apply(gradient=True) puts back the deeper blue
overhead the sky had before that, apply(glow=False) leaves the disc alone, and apply(solid_clouds=True,
rendered_clouds=True) puts clouds in it.

apply(plain=True) makes a plain sky instead, the fog color everywhere in both modes: the Sky group gives the fog color
alone, to the world and to the fog's sky light, and Solid mode has no sky objects, only its background in the fog
color. The Earth's maps have it, under their own sky (space.py); it was every map's for an hour on 2026-11-16, until
the user missed the sun on the others. plain_sky() makes a map's sky plain in place.

The sky is a function of view direction only, so it sits at infinity: moving, scaling, looping and wrapping never
change it, the way the real sky doesn't change as you walk.

Rendered (EEVEE): the Sky shader node group turns a view direction into the sky's color: the sun's disc (and glow),
worked out per pixel, on the fog color, so fog_color and the lamp's color still change it live. The world uses it, and
so does the fog's sky light (fog_look's Fog group), looked up along the view ray (-Incoming), without the disc:
whatever the fog hides fades into the sky right behind it (glow included). No clouds by default: a cloud behind
something fading in through the fog shows through it, which looked wrong (the user, 2026-10-30). apply(rendered_clouds=
True) puts them back, from Sky_pattern, an 8-bit equirectangular image of their cover and light (Blender's convention:
column u looks along azimuth pi - 2 pi u, row v up at elevation (v - 1/2) pi, checked in EEVEE); a float picture this
size packs into 100 MB.

Solid (Workbench) can't draw a sky: its background is one color, and it lights every face, so a mesh sun facing the
camera would be lit from behind (Workbench flips back faces to face the viewer too). Image empties are the exception:
they're drawn unlit, from their picture. So six image empties make a cube around the drone camera, each showing its
face of the sky, worked out here in numpy, drawn behind everything (Depth: Back). They're children of Solid_sky, a
mesh with no geometry (it draws nothing), which follows the camera's position and is scaled from its clip end, so the
cube always lies just beyond the far clip plane, which hides the frames Blender draws around image empties (see
solid_sky()). Its rotation stays put, so the sky stays fixed while the camera turns. They can't be selected, so a drone
click on the sky selects nothing. Image empties are overlays: turning overlays off hides the Solid sky, leaving Solid
mode's background, which is set to the fog color too. Solid mode had clouds until 2026-11-16, since it has no fog.

The sun is where the maps' light comes from: the Sun lamp's compass direction at fog_look.SUN_HEIGHT degrees up, as in
both modes' shading (solid_sun.py, fog_look.sunlight()). The pictures are made for that direction and the fog color,
so a Sun that turns or a new fog_color later needs apply() run again for Solid mode.

Usage: Blender -b map.blend --python tools/sky.py -- output.blend [--plain | --gradient | --no-glow] [--seed N] [--cover C] [--solid-clouds]
       [--rendered-clouds]
(saves a copy; to change a map you have open, run apply() in its Blender). From another script: apply(scene).
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

# Run as a script (Blender --python tools/sky.py), its folder isn't on the path for its neighbors
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fog_look
from solid_sun import sun_object

PATTERN = "Sky_pattern"
PATTERN_WIDTH = 4096                # equirectangular, so 0.088 degrees a pixel
FACE_SIZE = 1024                    # Solid mode's cube faces: about the same 0.088 degrees a pixel in the middle
GROUP = "Sky"
SOLID_SKY = "Solid_sky"
FACE_OVERLAP = 3                    # texels each cube face reaches past its edges: see solid_sky()
FACE_DISTANCE = 2.0                 # clip ends from the camera to each cube face's middle: see solid_sky()
SUN_RADIUS = 1.25                   # degrees: about 5 times the real sun, which would be 12 pixels across the
                                    # drone camera's 122 degree view
SUN_BRIGHTNESS = 12.0               # the disc, over the sky's brightness: clips to white
# The glow around the sun: (strength, width) pairs, exp((cos angle - 1) / width), so width 0.003 is about 4 degrees
GLOW = ((0.55, 0.003), (0.3, 0.03), (0.12, 0.25))
ZENITH = (0.30, 0.50, 0.85)         # straight up, times the horizon's color: deeper blue
# Clouds: a flat layer overhead, seen in perspective. Its fractal noise has CLOUD_OCTAVES octaves from CLOUD_SCALE
# blobs per cloud height; it's cloud where the noise passes CLOUD_COVER, solid CLOUD_SOFT above that.
CLOUD_SCALE = 1.4
CLOUD_OCTAVES = 7
CLOUD_COVER = 0.06
CLOUD_SOFT = 0.22
CLOUD_HORIZON = (0.015, 0.12)       # clouds fade in between these heights of the view direction (its z)
CLOUD_SHADOW = (0.24, 0.27, 0.34)   # thick cloud's underside (linear)
CLOUD_LIT = (0.95, 0.95, 0.95)      # thin, lit cloud, times the lamp's color
CLOUD_SILVER = 1.2                  # how much the sun's glow lights the clouds' edges
SEED = 7


def toward_sun(sun, height=fog_look.SUN_HEIGHT):
    """Where the sun is in the sky: the lamp's compass direction (it shines along its -Z) at height degrees up"""
    toward = sun.matrix_world.to_3x3().col[2]
    azimuth = math.atan2(toward.y, toward.x)
    h = math.radians(height)
    return np.array([math.cos(h) * math.cos(azimuth), math.cos(h) * math.sin(azimuth), math.sin(h)])


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class CloudNoise:
    """Fractal value noise on the cloud layer. Each octave is turned and shifted, so their lattices don't line up."""

    def __init__(self, seed, cover=CLOUD_COVER):
        self.cover = cover
        rng = np.random.default_rng(seed)
        self.table = rng.uniform(-1.0, 1.0, (512, 512))
        self.turns = [(math.cos(a), math.sin(a)) for a in rng.uniform(0.0, 2.0 * math.pi, CLOUD_OCTAVES)]
        self.shifts = rng.uniform(0.0, 512.0, (CLOUD_OCTAVES, 2))

    def value(self, x, y):
        n = self.table.shape[0]
        ix, iy = np.floor(x), np.floor(y)
        fx, fy = x - ix, y - iy
        ix, iy = ix.astype(np.int64) % n, iy.astype(np.int64) % n
        jx, jy = (ix + 1) % n, (iy + 1) % n
        sx, sy = fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy)
        t = self.table
        bottom = t[iy, ix] + (t[iy, jx] - t[iy, ix]) * sx
        top = t[jy, ix] + (t[jy, jx] - t[jy, ix]) * sx
        return bottom + (top - bottom) * sy

    def fractal(self, x, y, footprint):
        """The noise at (x, y) on the layer. footprint: how much of the layer a pixel covers there. Octaves finer than
        two pixels fade out to their average, 0, instead of aliasing into speckle near the horizon."""
        total = np.zeros_like(x)
        norm = 0.0
        for k in range(CLOUD_OCTAVES):
            frequency, amplitude = CLOUD_SCALE * 2.0 ** k, 0.5 ** k
            c, s = self.turns[k]
            u = (c * x - s * y) * frequency + self.shifts[k, 0]
            v = (s * x + c * y) * frequency + self.shifts[k, 1]
            keep = np.clip((0.5 - frequency * footprint) / 0.25, 0.0, 1.0)
            total += amplitude * keep * self.value(u, v)
            norm += amplitude
        return total / norm


def sun_glow(facing):
    """The glow around the sun, by the cosine of the angle from it, as the Sky node group works it out"""
    return sum(strength * np.exp((facing - 1.0) / width) for strength, width in GLOW)


def sun_disc_edges():
    """The cosines of the angles where the sun's disc starts fading in and is whole: its edge is a smoothstep about a
    tenth of a degree wide, anti-aliased over a couple of pixels"""
    edge = math.radians(0.06)
    return math.cos(math.radians(SUN_RADIUS) + edge), math.cos(math.radians(SUN_RADIUS) - edge)


def sky_pattern(directions, pixel_angle, sun_direction, noise):
    """The sky's pattern along unit directions (..., 3): the sun's glow and disc (0 to 1), cloud cover and cloud light,
    each (...). pixel_angle: how many radians a pixel spans, for the clouds' anti-aliasing. noise None: no clouds."""
    facing = directions @ sun_direction
    glow = sun_glow(facing)
    disc = smoothstep(*sun_disc_edges(), facing)

    cover = np.zeros(facing.shape)
    light = np.ones(facing.shape)
    up = directions[..., 2]
    sky = up > CLOUD_HORIZON[0]
    if noise is not None and sky.any():
        d = directions[sky]
        z = d[:, 2]
        # The layer at height 1: where the view ray meets it, and how much of it a pixel covers (the ray's slant
        # stretches a pixel by 1 / z^2)
        x, y = d[:, 0] / z, d[:, 1] / z
        footprint = pixel_angle / (z * z)
        density = noise.fractal(x, y, footprint)
        # Thick cloud is dark underneath; cloud with more cloud on its sun side is shaded by it
        sun_x, sun_y = sun_direction[0] / sun_direction[2], sun_direction[1] / sun_direction[2]
        to_sun = np.stack([sun_x - x, sun_y - y])
        to_sun /= np.maximum(np.linalg.norm(to_sun, axis=0), 1e-6)
        step = 0.04
        sunward = noise.fractal(x + step * to_sun[0], y + step * to_sun[1], footprint)
        thickness = smoothstep(noise.cover, noise.cover + 0.45, density)
        lit = 1.0 - 0.65 * thickness + 2.5 * (density - sunward)
        cover[sky] = smoothstep(noise.cover, noise.cover + CLOUD_SOFT, density) * smoothstep(*CLOUD_HORIZON, z)
        light[sky] = np.clip(lit, 0.0, 1.0)
    return glow, disc, cover, light


def sky_colors(directions, glow, disc, cover, light, horizon, sun_color, gradient=True):
    """The sky's linear color from its pattern, as the Sky node group works it out with Disc 1 (keep the two the
    same); gradient: deeper blue overhead, else the horizon's color everywhere"""
    horizon, sun_color = np.asarray(horizon), np.asarray(sun_color)
    height = np.sqrt(np.clip(directions[..., 2], 0.0, 1.0))[..., None]
    base = horizon + (horizon * np.asarray(ZENITH) - horizon) * height if gradient else horizon
    sky = base + (glow + SUN_BRIGHTNESS * disc)[..., None] * sun_color
    shadow, lit = np.asarray(CLOUD_SHADOW), np.asarray(CLOUD_LIT) * sun_color
    cloud = shadow + (lit - shadow) * light[..., None] + CLOUD_SILVER * glow[..., None] * sun_color
    return sky + (cloud - sky) * cover[..., None]


def equirect_directions(width):
    """Unit directions for every pixel of a width x width / 2 equirectangular image, rows from the bottom"""
    height = width // 2
    azimuth = math.pi - 2.0 * math.pi * (np.arange(width) + 0.5) / width
    elevation = ((np.arange(height) + 0.5) / height - 0.5) * math.pi
    az, el = np.meshgrid(azimuth, elevation)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1)


def store_image(name, pixels, colorspace, alpha=None):
    """pixels (rows from the bottom, columns, RGB, 0 to 1) as a packed 8-bit image (a PNG in the .blend) in colorspace.
    alpha: (rows, columns), else opaque."""
    rows, columns = pixels.shape[:2]
    image = bpy.data.images.get(name)
    if image is not None and (tuple(image.size) != (columns, rows) or image.is_float):
        bpy.data.images.remove(image)
        image = None
    if image is None:
        image = bpy.data.images.new(name, columns, rows, alpha=alpha is not None)
    image.colorspace_settings.name = colorspace
    image.alpha_mode = 'STRAIGHT'
    alpha = np.ones((rows, columns)) if alpha is None else alpha
    rgba = np.concatenate([pixels, alpha[..., None]], -1).astype(np.float32)
    image.pixels.foreach_set(rgba.ravel())
    image.pack()
    return image


def linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def add_driver(socket, index, expression, variables):
    fog_look.add_driver(socket, index, expression, variables)


def color_driven_by(nodes, label, variables_by_channel):
    rgb = nodes.new('ShaderNodeRGB')
    rgb.label = label
    for channel, (expression, variables) in enumerate(variables_by_channel):
        add_driver(rgb.outputs[0], channel, expression, variables)
    return rgb.outputs[0]


def sky_group(scene, sun, sun_direction, clouds, plain=False, glow=True, gradient=False):
    """The Sky shader node group: Vector (a view direction in the world) and Disc (1 to draw the sun's disc, 0 for
    the fog's sky light) in, the sky's color out, as sky_colors() works it out, with clouds from Sky_pattern or none.
    The sun stays at sun_direction, where Solid mode's sky and the clouds' light were worked out for, rather than
    following the lamp. plain: the fog color alone, whatever the direction; glow: the sun's glow around its disc;
    gradient: a deeper blue overhead, else the fog color everywhere."""
    group = bpy.data.node_groups.get(GROUP) or bpy.data.node_groups.new(GROUP, 'ShaderNodeTree')
    if group.animation_data:
        group.animation_data_clear()
    # Kept when they're right already: new sockets would unlink the Sky nodes in the world and the Fog group
    sockets = [(s.in_out, s.socket_type, s.name) for s in group.interface.items_tree if s.item_type == 'SOCKET']
    wanted = [('INPUT', 'NodeSocketVector', 'Vector'), ('INPUT', 'NodeSocketFloat', 'Disc'),
              ('OUTPUT', 'NodeSocketColor', 'Color')]
    if sorted(sockets) != sorted(wanted):
        group.interface.clear()
        for in_out, socket_type, name in wanted:
            group.interface.new_socket(name, in_out=in_out, socket_type=socket_type)
    nodes, links = group.nodes, group.links
    nodes.clear()
    group_in, group_out = nodes.new('NodeGroupInput'), nodes.new('NodeGroupOutput')

    def math_node(operation, a, b=None, clamp=False):
        node = nodes.new('ShaderNodeMath')
        node.operation, node.use_clamp = operation, clamp
        for socket, value in zip(node.inputs, (a, b)):
            if value is None:
                continue
            if isinstance(value, (int, float)):
                socket.default_value = value
            else:
                links.new(value, socket)
        return node.outputs[0]

    def mix(blend, a, b, factor=1.0):
        node = nodes.new('ShaderNodeMix')
        node.data_type, node.blend_type = 'RGBA', blend
        for socket, value in ((node.inputs['Factor'], factor), (fog_look.color_socket(node.inputs, 'A'), a),
                              (fog_look.color_socket(node.inputs, 'B'), b)):
            if isinstance(value, (int, float)):
                socket.default_value = value
            elif isinstance(value, tuple):
                socket.default_value = (*value, 1.0)
            else:
                links.new(value, socket)
        return fog_look.color_socket(node.outputs, 'Result')

    def gray(value):
        node = nodes.new('ShaderNodeCombineColor')
        for socket in node.inputs[:3]:
            links.new(value, socket)
        return node.outputs[0]

    # The sun's glow and disc (sun_glow(), sun_disc_edges()), from the cosine of the angle to it
    toward = nodes.new('ShaderNodeCombineXYZ')
    toward.label = "Toward the sun"
    for socket, value in zip(toward.inputs, sun_direction):
        socket.default_value = float(value)
    unit = nodes.new('ShaderNodeVectorMath')
    unit.operation = 'NORMALIZE'
    links.new(group_in.outputs['Vector'], unit.inputs[0])
    dot = nodes.new('ShaderNodeVectorMath')
    dot.operation = 'DOT_PRODUCT'
    links.new(unit.outputs['Vector'], dot.inputs[0])
    links.new(toward.outputs[0], dot.inputs[1])
    facing = dot.outputs['Value']
    glow_sum = None
    for strength, width in GLOW:
        term = math_node('MULTIPLY', math_node('EXPONENT', math_node('MULTIPLY', math_node('SUBTRACT', facing, 1.0),
                                                                     1.0 / width)), strength)
        glow_sum = term if glow_sum is None else math_node('ADD', glow_sum, term)
    disc = nodes.new('ShaderNodeMapRange')
    disc.interpolation_type = 'SMOOTHSTEP'
    disc.clamp = True
    disc.inputs['From Min'].default_value, disc.inputs['From Max'].default_value = sun_disc_edges()
    links.new(facing, disc.inputs['Value'])
    # Drawn only when asked
    disc = math_node('MULTIPLY', disc.outputs['Result'], SUN_BRIGHTNESS)
    disc = math_node('MULTIPLY', disc, group_in.outputs['Disc'])

    direction = nodes.new('ShaderNodeSeparateXYZ')
    links.new(group_in.outputs['Vector'], direction.inputs['Vector'])
    height = math_node('POWER', math_node('MAXIMUM', direction.outputs['Z'], 0.0), 0.5)

    fog_color = [("c", {"c": ('SCENE', scene, f'["fog_color"][{i}]')}) for i in range(3)]
    horizon = color_driven_by(nodes, "Fog color (driven)", fog_color)
    if plain:
        links.new(horizon, group_out.inputs['Color'])
        return group
    sun_color = color_driven_by(nodes, "Sun color (driven)",
                                [("c", {"c": ('LIGHT', sun.data, f"color[{i}]")}) for i in range(3)])
    base = mix('MIX', horizon, mix('MULTIPLY', horizon, tuple(ZENITH)), height) if gradient else horizon
    sky = mix('ADD', base, mix('MULTIPLY', sun_color, gray(math_node('ADD', glow_sum, disc) if glow else disc)))
    if clouds:
        pattern = nodes.new('ShaderNodeTexEnvironment')
        pattern.image = bpy.data.images[PATTERN]
        pattern.interpolation = 'Linear'
        links.new(group_in.outputs['Vector'], pattern.inputs['Vector'])
        channels = nodes.new('ShaderNodeSeparateColor')
        links.new(pattern.outputs['Color'], channels.inputs['Color'])
        cover, light = channels.outputs[0], channels.outputs[1]
        lit = mix('MULTIPLY', tuple(CLOUD_LIT), sun_color)
        cloud = mix('ADD', mix('MIX', tuple(CLOUD_SHADOW), lit, light),
                    mix('MULTIPLY', sun_color, gray(math_node('MULTIPLY', glow_sum, CLOUD_SILVER))))
        sky = mix('MIX', sky, cloud, cover)
    links.new(sky, group_out.inputs['Color'])
    return group


def rendered_sky(scene, group):
    """The world shows the Sky, and the fog's sky light (the Fog group's fog color) comes from it along the view ray"""
    world = fog_look.fog_world(scene)
    nodes, links = world.node_tree.nodes, world.node_tree.links
    background = next(n for n in nodes if n.type == 'BACKGROUND')
    # The background's fog color drivers live on the world's node tree
    if world.node_tree.animation_data:
        for fcurve in list(world.node_tree.animation_data.drivers):
            if fcurve.data_path.startswith(f'nodes["{background.name}"]'):
                world.node_tree.animation_data.drivers.remove(fcurve)
    for node in [n for n in nodes if n.type in {'TEX_COORD', 'GROUP'}]:
        nodes.remove(node)
    coordinates = nodes.new('ShaderNodeTexCoord')
    sky = nodes.new('ShaderNodeGroup')
    sky.node_tree = group
    sky.inputs['Disc'].default_value = 1.0
    # In a world shader, Generated is the view direction (checked in EEVEE)
    links.new(coordinates.outputs['Generated'], sky.inputs['Vector'])
    links.new(sky.outputs['Color'], background.inputs['Color'])

    fog = bpy.data.node_groups.get(fog_look.GROUP)
    if fog is None:
        return "no fog to take its sky light from the Sky"
    fog_nodes, fog_links = fog.nodes, fog.links
    old = next((n for n in fog_nodes if n.bl_idname == 'ShaderNodeRGB' and n.label == "Fog color (driven)"), None)
    sky = next((n for n in fog_nodes if n.bl_idname == 'ShaderNodeGroup' and n.node_tree == group), None)
    if old is None and sky is None:
        return "the Fog group has no fog color node to replace"
    if sky is None:
        geometry = fog_nodes.new('ShaderNodeNewGeometry')
        along_ray = fog_nodes.new('ShaderNodeVectorMath')
        along_ray.operation = 'SCALE'
        along_ray.inputs['Scale'].default_value = -1.0
        fog_links.new(geometry.outputs['Incoming'], along_ray.inputs[0])
        sky = fog_nodes.new('ShaderNodeGroup')
        sky.node_tree = group
        sky.inputs['Disc'].default_value = 0.0
        fog_links.new(along_ray.outputs[0], sky.inputs['Vector'])
    if old is not None:
        for link in list(old.outputs[0].links):
            fog_links.new(sky.outputs['Color'], link.to_socket)
        if fog.animation_data:
            for fcurve in list(fog.animation_data.drivers):
                if fcurve.data_path.startswith(f'nodes["{old.name}"]'):
                    fog.animation_data.drivers.remove(fcurve)
        fog_nodes.remove(old)
    return "the fog fades into the sky behind it"


def face_frames():
    """The cube's six faces: (name, middle, right, up), unit vectors in the world. The image runs right and up."""
    frames = []
    for name, middle in (("+X", (1, 0, 0)), ("-X", (-1, 0, 0)), ("+Y", (0, 1, 0)), ("-Y", (0, -1, 0)),
                         ("+Z", (0, 0, 1)), ("-Z", (0, 0, -1))):
        middle = Vector(middle)
        up = Vector((0, 1, 0)) if abs(middle.z) > 0.5 else Vector((0, 0, 1))
        right = up.cross(middle)
        frames.append((name, middle, right, up))
    return frames


def solid_sky(scene, sun_direction, noise, horizon, sun_color, glow=True, gradient=False):
    """Solid mode's sky: six image empties in a cube around the drone camera (see the module's docstring). Blender
    draws a thin frame around every image empty, in the wire color, which would show the cube's edges as black lines.
    Depth Back draws the pictures at the far clip plane, wherever they are (an infinite projection), but the frames
    where they are, so the cube is put beyond the clip end, FACE_DISTANCE clip ends away: the frames are clipped and
    the pictures stay. noise None: no clouds; glow, gradient: as sky_group()."""
    camera = bpy.data.objects["Drone_Camera"]
    collection = bpy.data.collections.get(SOLID_SKY)
    if collection is None:
        collection = bpy.data.collections.new(SOLID_SKY)
        scene.collection.children.link(collection)
    holder = bpy.data.objects.get(SOLID_SKY)
    if holder is None:
        holder = bpy.data.objects.new(SOLID_SKY, bpy.data.meshes.new(SOLID_SKY))
        collection.objects.link(holder)
    holder.hide_select = True
    holder.hide_render = True
    holder.constraints.clear()
    follow = holder.constraints.new('COPY_LOCATION')
    follow.target = camera
    if holder.animation_data:
        holder.animation_data_clear()
    for axis in range(3):
        fcurve = holder.driver_add("scale", axis)
        driver = fcurve.driver
        driver.type = 'SCRIPTED'
        variable = driver.variables.new()
        variable.name = "c"
        variable.targets[0].id_type = 'CAMERA'
        variable.targets[0].id = camera.data
        variable.targets[0].data_path = "clip_end"
        driver.expression = f"{FACE_DISTANCE} * c"

    # Texel middles on each face, at distance 1, rows from the bottom. Each face reaches FACE_OVERLAP texels past the
    # cube's edges, over its neighbors, where they show the same sky, and its outermost ring of texels is
    # transparent. Whichever face is drawn on top shows its edge there, and the GPU wraps the picture around, so its
    # last half texel is blended with the opposite edge's sky: drawn opaque, that was a faint dotted line along every
    # edge of the cube. Transparent, the neighbor's sky shows through, whichever way round they're drawn.
    reach = 1.0 + 2.0 * FACE_OVERLAP / FACE_SIZE
    ring = np.ones((FACE_SIZE, FACE_SIZE))
    ring[[0, -1], :] = ring[:, [0, -1]] = 0.0
    t = ((np.arange(FACE_SIZE) + 0.5) / FACE_SIZE * 2.0 - 1.0) * reach
    across, upward = np.meshgrid(t, t)
    pixel_angle = (math.pi / 2.0) / FACE_SIZE
    made = []
    for name, middle, right, up in face_frames():
        points = (np.asarray(middle)[None, None] + across[..., None] * np.asarray(right)[None, None]
                  + upward[..., None] * np.asarray(up)[None, None])
        directions = points / np.linalg.norm(points, axis=-1, keepdims=True)
        sun_glow_, disc, cover, light = sky_pattern(directions, pixel_angle, sun_direction, noise)
        color = sky_colors(directions, sun_glow_ if glow else 0.0 * sun_glow_, disc, cover, light, horizon, sun_color,
                           gradient=gradient)
        image = store_image(f"Sky_face{name}", linear_to_srgb(color), 'sRGB', alpha=ring)
        face = bpy.data.objects.get(f"Solid_sky{name}")
        if face is None:
            face = bpy.data.objects.new(f"Solid_sky{name}", None)
            collection.objects.link(face)
        face.parent = holder
        face.empty_display_type = 'IMAGE'
        face.data = image
        # The picture spans display size, from its corner at the offset: a 2 x 2 face around its middle, and the overlap
        face.empty_display_size = 2.0 * reach
        face.empty_image_offset = (-0.5, -0.5)
        face.empty_image_depth = 'BACK'
        face.empty_image_side = 'DOUBLE_SIDED'
        face.show_empty_image_orthographic = True
        face.show_empty_image_perspective = True
        face.show_empty_image_only_axis_aligned = False
        face.use_empty_image_alpha = True
        face.color = (1.0, 1.0, 1.0, 1.0)
        face.matrix_parent_inverse = Matrix.Identity(4)
        face.matrix_basis = Matrix.Translation(middle) @ Matrix((right, up, right.cross(up))).transposed().to_4x4()
        face.hide_select = True
        face.hide_render = True
        made.append(face)
    return made


def remove_solid_sky():
    """Remove Solid mode's sky (solid_sky()): its image empties, their holder and pictures, and its collection.
    Returns how many objects went."""
    objects = [o for o in bpy.data.objects if o.name == SOLID_SKY or o.name.startswith(SOLID_SKY + "+")
               or o.name.startswith(SOLID_SKY + "-")]
    meshes = [o.data for o in objects if o.type == 'MESH']
    for obj in objects:
        bpy.data.objects.remove(obj)
    for mesh in meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for image in [i for i in bpy.data.images if i.name.startswith("Sky_face")]:
        if image.users == 0:
            bpy.data.images.remove(image)
    collection = bpy.data.collections.get(SOLID_SKY)
    if collection is not None:
        bpy.data.collections.remove(collection)
    return len(objects)


def solid_background(color):
    """Solid mode's background, in the 3D views that use their own (Viewport) color, set to color (linear, as Blender
    keeps it): it's what shows with overlays off, where the Solid sky's image empties go"""
    for screen in bpy.data.screens:
        for area in screen.areas:
            for space in area.spaces:
                if space.type == 'VIEW_3D' and space.shading.background_type == 'VIEWPORT':
                    space.shading.background_color = color


def hide_relationship_lines():
    """No relationship lines in any 3D view: they'd draw a dashed line from each sky face to Solid_sky (and from a
    map's parts to the empty they're parented to)"""
    for screen in bpy.data.screens:
        for area in screen.areas:
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.overlay.show_relationship_lines = False


def apply(scene, seed=SEED, cover=CLOUD_COVER, rendered_clouds=False, solid_clouds=False, plain=False, glow=True,
          gradient=False):
    """Put the sky (see the module's docstring) in scene, made afresh: the sun's disc on the fog color, glow its glow
    around it, gradient a deeper blue overhead, solid_clouds and rendered_clouds whether each mode's sky has clouds, seed
    picks them, cover how much sky they leave (higher, less cloud); plain, the fog color everywhere, no sun. Returns a
    line saying what it did."""
    sun = sun_object(scene)
    if sun is None:
        raise ValueError(f"{scene.name} has no sun lamp to put in the sky")
    fog_look.ensure_fog_properties(scene)
    sun_direction = toward_sun(sun)
    if plain:
        group = sky_group(scene, sun, sun_direction, False, plain=True)
        if PATTERN in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[PATTERN])
        fog = rendered_sky(scene, group)
        remove_solid_sky()
        solid_background(list(scene["fog_color"]))
        hide_relationship_lines()
        return (f"Sky: plain, the fog color everywhere; Rendered: the world shows it, and {fog}; Solid: the background "
                f"in the fog color; relationship lines off in every 3D view")
    noise = CloudNoise(seed, cover)

    if rendered_clouds:
        # The clouds' pattern, in bands of rows to keep numpy's arrays small
        directions = equirect_directions(PATTERN_WIDTH)
        pattern = np.zeros(directions.shape)
        pixel_angle = 2.0 * math.pi / PATTERN_WIDTH
        for start in range(0, directions.shape[0], 128):
            band = sky_pattern(directions[start:start + 128], pixel_angle, sun_direction, noise)
            pattern[start:start + 128, :, 0], pattern[start:start + 128, :, 1] = band[2], band[3]
        store_image(PATTERN, pattern, 'Non-Color')

    group = sky_group(scene, sun, sun_direction, rendered_clouds, glow=glow, gradient=gradient)
    if not rendered_clouds and PATTERN in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[PATTERN])
    fog = rendered_sky(scene, group)
    # Solid mode in the Rendered sky's colors: the horizon is the fog color, which Blender keeps linear, as the sky
    # and Solid mode's background both take it
    horizon = list(scene["fog_color"])
    faces = solid_sky(scene, sun_direction, noise if solid_clouds else None, horizon, list(sun.data.color), glow,
                      gradient)
    solid_background(horizon)
    hide_relationship_lines()
    elevation = math.degrees(math.asin(sun_direction[2]))
    azimuth = math.degrees(math.atan2(sun_direction[1], sun_direction[0]))
    return (f"Sky: sun at azimuth {azimuth:.0f}, {elevation:.0f} degrees up, "
            f"{'with its glow' if glow else 'its disc alone'}, "
            f"{'deeper blue overhead, the fog color at the horizon' if gradient else 'on the fog color everywhere'}; Rendered: "
            f"the world shows it{' with clouds' if rendered_clouds else ', no clouds'}, and {fog}; Solid: "
            f"{len(faces)} image empties around Drone_Camera, "
            f"{f'with clouds (seed {seed}, cover {cover})' if solid_clouds else 'no clouds'}, and the background in "
            f"the fog color; relationship lines off in every 3D view")


def plain_sky(scene):
    """Make a map's sky plain (see the module's docstring) in place: the Sky group the fog color alone, and Solid mode's
    sky objects gone, its background the fog color. Nothing else is touched. Returns a line saying what it did."""
    done = []
    group = bpy.data.node_groups.get(GROUP)
    sun = sun_object(scene)
    if group is not None and sun is not None:
        sky_group(scene, sun, toward_sun(sun), False, plain=True)
        done.append("the Sky group (world and fog) is the fog color alone")
    removed = remove_solid_sky()
    if removed:
        done.append(f"Solid sky removed ({removed} objects)")
    if "fog_color" in scene:
        solid_background(list(scene["fog_color"]))
        done.append("Solid background the fog color")
    if PATTERN in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[PATTERN])
        done.append("Rendered clouds removed")
    return "; ".join(done) or "no sky"

if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    outputs = [a for a in args if a.endswith(".blend")]
    if not outputs:
        raise SystemExit("sky.py needs an output .blend: it saves a copy, never over the map")
    seed = int(args[args.index("--seed") + 1]) if "--seed" in args else SEED
    cover = float(args[args.index("--cover") + 1]) if "--cover" in args else CLOUD_COVER
    print(apply(bpy.context.scene, seed, cover, "--rendered-clouds" in args, "--solid-clouds" in args,
                plain="--plain" in args, glow="--no-glow" not in args, gradient="--gradient" in args))
    bpy.ops.wm.save_as_mainfile(filepath=outputs[0], copy=True)
    print("SAVED", outputs[0])
