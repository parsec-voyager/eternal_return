"""The aerial-perspective fog for flying in the Rendered viewport (EEVEE), shared by eternal_return.blend (see
update_template.py) and the maps built from it, so they all have one fog.

What ensure_fog_look() adds to a scene:
- Scene custom properties, to change while flying (Scene Properties -> Custom Properties): fog_distance, how far you
  see at scale 1, and fog_color, the sky that far things fade into.
- The Fog_look and Fog_look_plain materials, sharing the Fog node group: Solid mode's look (the "Col" color attribute,
  or for Fog_look_plain the object's viewport color, times the basic_bright MatCap) with aerial perspective, the way
  far mountains go blue. Distances follow the drone camera's scale, so loops (which scale the drone and the view
  together) look the same. They go on every mesh without a material, except the drone rig.
- The Fog_sky world, plain fog color, so the far distance melts into the background. The previous world is kept.
- The Standard view transform: AgX turns the blue grey.

Two looks: the stylized one, where a surface's red, green and blue light fade at different distances, red first, so it
goes blue before it's gone, while the sky's light builds up more slowly; and Earth's air (earth=True), clear up close
with no tint until mountain distances, fog_distance then being the visibility.

Solid mode can't show it: its only fog is a grey volume, and it draws attribute-colored faces opaque. So the add-on
draws Solid mode's fog over the view itself, made to look like this one (eternal_return/solid_sky.py).

Sunlight (ensure_fog_look() in a scene with a sun lamp, or light_by_sun() for a map that has the fog): surfaces also
take the sun lamp's light, in its color, the shade
times 1 - k + 4 k f, where f is how squarely the surface faces the sun (0 facing away) and k the scene's sun_light (0
to 1, 0 the MatCap alone). f averages 1 / 4 over every direction, so surfaces facing every way equally often keep the
plain look's brightness on average. It's worked out here because the materials are emission, which no lamp lights
(only the drone's HUD showed the Sun), and taking the lamp's real light (Shader to RGB) would tie the map's brightness
to a strength set for the HUD: measured, a white surface facing a strength-1 sun comes out 0.318, so the user's Sun
(45) would make it 14 times white. Drivers follow the lamp's rotation and color, so turning it turns the light. It
comes from the lamp's compass direction at sun_height degrees up, SUN_HEIGHT by default, the height of Solid mode's
studio key light (solid_sun.py), so both modes agree; None follows the lamp's own height, but the user's Sun grazes
the horizon, which leaves floors unlit. No shadows, which cost a lot (see solid_sun.py).
"""

import bpy
import math
import os

from solid_sun import sun_object

FOG_DISTANCE = 60.0
FOG_COLOR = (95 / 255, 160 / 255, 245 / 255)        # cornflower blue (sRGB)
# In fog distances: a surface's red, green and blue light fade to 1 / e this far away, and the sky's light builds up to
# 1 - 1 / e this far
FOG_EXTINCTION = (0.18, 0.35, 1.8)
FOG_AIRLIGHT = 2.6
# Sunlight (see the module's docstring): how much it lights, the scene's sun_light (0 to 1) unless it has one, and
# how high it comes from, in degrees
SUN_LIGHT = 0.5
SUN_HEIGHT = 40.0
# Earth's atmosphere (ensure_fog_look(earth=True)): fog_distance is then the visibility at scale 1, how far you can see
# (contrast drops to 2%), and each channel fades the way air does on a clear day: air molecules (Rayleigh, 5.8, 13.5,
# 33.1 x 10^-6 per meter for red, green, blue) plus haze (Angstrom exponent 1.3), adding up to 1 / e at these fractions
# of a 23 km visibility (8.0, 5.9 and 4.1 km). Both the surface's light and the sky's use the same per-channel fading, as
# in real air, so far things go blue (blue goes first) and end in the sky color.
EARTH_VISIBILITY = 23000.0
PARTS_MATERIAL = "_ER_Parts"      # eternal_return/parts.py MATERIAL
EARTH_EXTINCTION = (0.349, 0.256, 0.179)
MATCAP = "studiolights/matcap/basic_bright.exr"     # Solid mode's MatCap, in Blender's datafiles
GROUP = "Fog"
MATERIAL = "Fog_look"                  # objects with a "Col" color attribute
PLAIN_MATERIAL = "Fog_look_plain"      # objects without one: their viewport color (Object Properties > Viewport Display)
DISSOLVE_MATERIAL = "Fog_look_dissolve"     # Fog_look that turns transparent by the object's "dissolve" property
WORLD = "Fog_sky"
RIG = {"Drone", "Drone_Camera", "clip_end"}


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def color_socket(sockets, name):
    """The Mix node's float, vector and color sockets share names: the color one."""
    return next(s for s in sockets if s.name == name and s.type == 'RGBA')


def add_driver(socket, index, expression, variables):
    """Drive socket.default_value (one channel with index, else the value) by an expression of variables: name ->
    (id_type, id, data_path) for single properties, or (object, transform_type) for world-space transforms."""
    fcurve = socket.driver_add("default_value") if index is None else socket.driver_add("default_value", index)
    driver = fcurve.driver
    driver.type = 'SCRIPTED'
    for name, target in variables.items():
        variable = driver.variables.new()
        variable.name = name
        if len(target) == 3:
            variable.type = 'SINGLE_PROP'
            variable.targets[0].id_type, variable.targets[0].id = target[0], target[1]
            variable.targets[0].data_path = target[2]
        else:
            variable.type = 'TRANSFORMS'
            variable.targets[0].id = target[0]
            variable.targets[0].transform_type, variable.targets[0].transform_space = target[1], 'WORLD_SPACE'
            if target[1].startswith('ROT'):
                # Sunlight's direction formulas are for XYZ Euler angles, whatever the lamp's own rotation mode
                variable.targets[0].rotation_mode = 'XYZ'
    driver.expression = expression


def drive_fog_color(socket, scene):
    for index in range(3):
        add_driver(socket, index, "c", {"c": ('SCENE', scene, f'["fog_color"][{index}]')})


def fog_shading(scene, earth=False, sun=None, sun_height=None):
    """The Fog shader node group, built afresh: Color in, the shaded and fogged surface out. Both materials use it, so
    the fog lives in one place. earth: Earth's atmosphere (see EARTH_EXTINCTION) instead of the stylized blue. sun,
    sun_height: see the module's docstring."""
    group = bpy.data.node_groups.get(GROUP) or bpy.data.node_groups.new(GROUP, 'ShaderNodeTree')
    if group.animation_data:
        group.animation_data_clear()
    # Kept when they're right already: new sockets would unlink every material using the group (Fog_look_dissolve too)
    sockets = sorted((s.in_out, s.socket_type, s.name) for s in group.interface.items_tree if s.item_type == 'SOCKET')
    if sockets != [('INPUT', 'NodeSocketColor', 'Color'), ('OUTPUT', 'NodeSocketShader', 'Shader')]:
        group.interface.clear()
        group.interface.new_socket("Color", in_out='INPUT', socket_type='NodeSocketColor')
        group.interface.new_socket("Shader", in_out='OUTPUT', socket_type='NodeSocketShader')
    group["earth"] = earth
    nodes, links = group.nodes, group.links
    nodes.clear()
    group_in, group_out = nodes.new('NodeGroupInput'), nodes.new('NodeGroupOutput')
    emission = nodes.new('ShaderNodeEmission')

    # The color times the MatCap, looked up by the camera-space normal, as Solid mode does
    geometry = nodes.new('ShaderNodeNewGeometry')
    to_camera = nodes.new('ShaderNodeVectorTransform')
    to_camera.vector_type, to_camera.convert_from, to_camera.convert_to = 'NORMAL', 'WORLD', 'CAMERA'
    to_uv = nodes.new('ShaderNodeVectorMath')
    to_uv.operation = 'MULTIPLY_ADD'
    to_uv.inputs[1].default_value = to_uv.inputs[2].default_value = (0.5, 0.5, 0.0)
    matcap = nodes.new('ShaderNodeTexImage')
    path = os.path.join(bpy.utils.system_resource('DATAFILES'), MATCAP)
    matcap.image = next((i for i in bpy.data.images if i.name == os.path.basename(MATCAP)), None) or bpy.data.images.load(path)
    if not matcap.image.packed_file:
        matcap.image.pack()
    matcap.extension = 'EXTEND'
    shade = nodes.new('ShaderNodeMix')
    shade.data_type, shade.blend_type = 'RGBA', 'MULTIPLY'
    shade.inputs['Factor'].default_value = 1.0
    links.new(geometry.outputs['Normal'], to_camera.inputs['Vector'])
    links.new(to_camera.outputs['Vector'], to_uv.inputs[0])
    links.new(to_uv.outputs['Vector'], matcap.inputs['Vector'])
    links.new(group_in.outputs['Color'], color_socket(shade.inputs, 'A'))
    links.new(matcap.outputs['Color'], color_socket(shade.inputs, 'B'))
    if sun is not None:
        shade = sunlight(scene, nodes, links, geometry, shade, sun, sun_height)

    # How many fog distances away the surface is. The fog distance follows the drone camera's scale.
    camera = nodes.new('ShaderNodeCameraData')
    distance = nodes.new('ShaderNodeValue')
    distance.label = "Fog distance (driven)"
    away = nodes.new('ShaderNodeMath')
    away.operation = 'DIVIDE'
    links.new(camera.outputs['View Distance'], away.inputs[0])
    links.new(distance.outputs[0], away.inputs[1])

    def transmission(rate):
        """exp(-away / rate): how much of something's light is left"""
        scaled = nodes.new('ShaderNodeMath')
        scaled.operation = 'MULTIPLY'
        scaled.inputs[1].default_value = -1.0 / rate
        left = nodes.new('ShaderNodeMath')
        left.operation = 'EXPONENT'
        links.new(away.outputs[0], scaled.inputs[0])
        links.new(scaled.outputs[0], left.inputs[0])
        return left.outputs[0]

    # The surface's light, fading per channel, plus the sky's light, building up to the fog color
    surface_left = nodes.new('ShaderNodeCombineColor')
    for channel, rate in enumerate(EARTH_EXTINCTION if earth else FOG_EXTINCTION):
        links.new(transmission(rate), surface_left.inputs[channel])
    surface = nodes.new('ShaderNodeMix')
    surface.data_type, surface.blend_type = 'RGBA', 'MULTIPLY'
    surface.inputs['Factor'].default_value = 1.0
    links.new(color_socket(shade.outputs, 'Result'), color_socket(surface.inputs, 'A'))
    links.new(surface_left.outputs[0], color_socket(surface.inputs, 'B'))
    sky = nodes.new('ShaderNodeRGB')
    sky.label = "Fog color (driven)"
    airlight = nodes.new('ShaderNodeMix')
    if earth:
        # The sky's light builds up per channel as the surface's fades: sky x (1 - what's left)
        airlight.data_type, airlight.blend_type = 'RGBA', 'MULTIPLY'
        airlight.inputs['Factor'].default_value = 1.0
        built_up = nodes.new('ShaderNodeInvert')
        links.new(surface_left.outputs[0], built_up.inputs['Color'])
        links.new(sky.outputs[0], color_socket(airlight.inputs, 'A'))
        links.new(built_up.outputs['Color'], color_socket(airlight.inputs, 'B'))
    else:
        airlight.data_type = 'RGBA'
        color_socket(airlight.inputs, 'A').default_value = (0.0, 0.0, 0.0, 1.0)
        built_up = nodes.new('ShaderNodeMath')
        built_up.operation = 'SUBTRACT'
        built_up.inputs[0].default_value = 1.0
        links.new(transmission(FOG_AIRLIGHT), built_up.inputs[1])
        links.new(built_up.outputs[0], airlight.inputs['Factor'])
        links.new(sky.outputs[0], color_socket(airlight.inputs, 'B'))
    total = nodes.new('ShaderNodeMix')
    total.data_type, total.blend_type = 'RGBA', 'ADD'
    total.inputs['Factor'].default_value = 1.0
    links.new(color_socket(surface.outputs, 'Result'), color_socket(total.inputs, 'A'))
    links.new(color_socket(airlight.outputs, 'Result'), color_socket(total.inputs, 'B'))
    links.new(color_socket(total.outputs, 'Result'), emission.inputs['Color'])
    links.new(emission.outputs[0], group_out.inputs['Shader'])

    # Never 0, which would divide by 0 and switch the driver off for the session
    add_driver(distance.outputs[0], None, "max(0.000001, f * s)",
               {"f": ('SCENE', scene, '["fog_distance"]'), "s": (bpy.data.objects["Drone_Camera"], 'SCALE_AVG')})
    drive_fog_color(sky.outputs[0], scene)
    return group


def sunlight(scene, nodes, links, geometry, shade, sun, sun_height):
    """shade times the sun's light (see the module's docstring), as a Mix node whose Result replaces shade's"""
    rotation = {name: (sun, transform) for name, transform in (("rx", 'ROT_X'), ("ry", 'ROT_Y'), ("rz", 'ROT_Z'))}
    toward = nodes.new('ShaderNodeCombineXYZ')
    toward.label = "Toward the sun (driven)"
    # The lamp shines along its -Z: its +Z in the world, from its XYZ rotation
    add_driver(toward.inputs[0], None, "cos(rz)*sin(ry)*cos(rx)+sin(rz)*sin(rx)", rotation)
    add_driver(toward.inputs[1], None, "sin(rz)*sin(ry)*cos(rx)-cos(rz)*sin(rx)", rotation)
    direction = toward.outputs[0]
    if sun_height is None:
        add_driver(toward.inputs[2], None, "cos(ry)*cos(rx)", rotation)
    else:
        # Its compass direction, at sun_height
        flat = nodes.new('ShaderNodeVectorMath')
        flat.operation = 'NORMALIZE'
        links.new(direction, flat.inputs[0])
        lowered = nodes.new('ShaderNodeVectorMath')
        lowered.operation = 'MULTIPLY_ADD'
        lowered.inputs[1].default_value = [math.cos(math.radians(sun_height))] * 3
        lowered.inputs[2].default_value = (0.0, 0.0, math.sin(math.radians(sun_height)))
        links.new(flat.outputs[0], lowered.inputs[0])
        direction = lowered.outputs[0]
    facing = nodes.new('ShaderNodeVectorMath')
    facing.operation = 'DOT_PRODUCT'
    links.new(geometry.outputs['Normal'], facing.inputs[0])
    links.new(direction, facing.inputs[1])
    lit = nodes.new('ShaderNodeMath')
    lit.operation, lit.use_clamp = 'MAXIMUM', False
    lit.inputs[1].default_value = 0.0
    links.new(facing.outputs['Value'], lit.inputs[0])
    tint = nodes.new('ShaderNodeRGB')
    tint.label = "Sun color (driven)"
    for channel in range(3):
        add_driver(tint.outputs[0], channel, "c", {"c": ('LIGHT', sun.data, f"color[{channel}]")})
    # 1 - k + 4 k * facing * the lamp's color
    sunlit = nodes.new('ShaderNodeMix')
    sunlit.data_type, sunlit.blend_type = 'RGBA', 'MULTIPLY'
    sunlit.inputs['Factor'].default_value = 1.0
    links.new(tint.outputs[0], color_socket(sunlit.inputs, 'A'))
    links.new(lit.outputs[0], color_socket(sunlit.inputs, 'B'))
    light = nodes.new('ShaderNodeVectorMath')
    light.operation = 'MULTIPLY_ADD'
    light.label = "Sun light (driven)"
    strength = {"k": ('SCENE', scene, '["sun_light"]')}
    for channel in range(3):
        add_driver(light.inputs[1], channel, "4 * k", strength)
        add_driver(light.inputs[2], channel, "1 - k", strength)
    links.new(color_socket(sunlit.outputs, 'Result'), light.inputs[0])
    lit_shade = nodes.new('ShaderNodeMix')
    lit_shade.data_type, lit_shade.blend_type = 'RGBA', 'MULTIPLY'
    lit_shade.inputs['Factor'].default_value = 1.0
    links.new(color_socket(shade.outputs, 'Result'), color_socket(lit_shade.inputs, 'A'))
    links.new(light.outputs[0], color_socket(lit_shade.inputs, 'B'))
    return lit_shade


def fog_material(name, group, vertex_colors):
    """A material that feeds the Fog group the "Col" color attribute, or else the object's viewport color (a missing
    attribute reads black, so one material can't fall back to it)."""
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    fog = nodes.new('ShaderNodeGroup')
    fog.node_tree = group
    if vertex_colors:
        source = nodes.new('ShaderNodeAttribute')
        source.attribute_name = "Col"
    else:
        source = nodes.new('ShaderNodeObjectInfo')
    links.new(source.outputs['Color'], fog.inputs['Color'])
    links.new(fog.outputs['Shader'], output.inputs['Surface'])
    return material


def dissolve_material(group):
    """Fog_look_dissolve: Fog_look, see-through by the object's "dissolve" custom property (0 solid, 1 gone; a missing
    one reads 0). Blended, so it fades smoothly rather than in dither noise, and only its front surface draws, so its
    own back faces don't show through. Costlier than Fog_look, so only for the few objects that need it."""
    material = fog_material(DISSOLVE_MATERIAL, group, True)
    nodes, links = material.node_tree.nodes, material.node_tree.links
    output = next(n for n in nodes if n.type == 'OUTPUT_MATERIAL')
    fog = next(n for n in nodes if n.type == 'GROUP')
    dissolve = nodes.new('ShaderNodeAttribute')
    dissolve.attribute_type, dissolve.attribute_name = 'OBJECT', "dissolve"
    mix = nodes.new('ShaderNodeMixShader')
    links.new(dissolve.outputs['Fac'], mix.inputs['Fac'])
    links.new(fog.outputs['Shader'], mix.inputs[1])
    links.new(nodes.new('ShaderNodeBsdfTransparent').outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs['Surface'])
    material.surface_render_method = 'BLENDED'
    material.use_transparency_overlap = False
    return material


def fog_world(scene):
    """The Fog_sky world: plain fog color. The scene's previous world keeps a fake user, so it isn't lost."""
    world = bpy.data.worlds.get(WORLD)
    if world is None:
        world = bpy.data.worlds.new(WORLD)
        world.use_nodes = True
        background = next(n for n in world.node_tree.nodes if n.type == 'BACKGROUND')
        drive_fog_color(background.inputs['Color'], scene)
    if scene.world and scene.world != world:
        scene.world.use_fake_user = True
    scene.world = world
    return world


def ensure_fog_properties(scene, sun=False):
    """The scene's fog_distance and fog_color, and with sun its sun_light, unless it has them already. Drivers that
    read them need them first: a driver reading a missing property errors, and Blender switches it off for the
    session."""
    if "fog_distance" not in scene:
        scene["fog_distance"] = FOG_DISTANCE
    if "fog_color" not in scene:
        scene["fog_color"] = [srgb_to_linear(c) for c in FOG_COLOR]
        scene.id_properties_ui("fog_color").update(subtype='COLOR', min=0.0, max=1.0, soft_min=0.0, soft_max=1.0)
    if sun and "sun_light" not in scene:
        scene["sun_light"] = SUN_LIGHT
        scene.id_properties_ui("sun_light").update(min=0.0, max=1.0, soft_min=0.0, soft_max=1.0)


def ensure_fog_look(scene, objects=None, earth=False, sun=True, sun_height=SUN_HEIGHT):
    """Add the fog to scene and put Fog_look (or Fog_look_plain, for meshes without a "Col" attribute) on objects
    (default: every mesh without a material, except the drone rig). Keeps fog_distance and fog_color if the scene
    already has them, unless fog_distance is the other look's default: earth (Earth's atmosphere, see
    EARTH_EXTINCTION) has it as the visibility, EARTH_VISIBILITY by default, and the stylized look FOG_DISTANCE. Lit by
    sun (see the module's docstring): by default the scene's sun lamp (sun_object()), if it has one; a lamp; or False
    for none."""
    if sun is True:
        sun = sun_object(scene)
    sun = sun or None
    ensure_fog_properties(scene, sun is not None)
    if earth and scene["fog_distance"] == FOG_DISTANCE:
        scene["fog_distance"] = EARTH_VISIBILITY
    elif not earth and scene["fog_distance"] == EARTH_VISIBILITY:
        scene["fog_distance"] = FOG_DISTANCE
    group = fog_shading(scene, earth, sun, sun_height)
    materials = {True: fog_material(MATERIAL, group, True), False: fog_material(PLAIN_MATERIAL, group, False)}
    fog_world(scene)
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    if objects is None:
        objects = [o for o in scene.objects if o.type == 'MESH' and o.name not in RIG and not o.data.materials]
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(materials["Col" in obj.data.color_attributes])


def fog_parts(scene, distance=EARTH_VISIBILITY):
    """Fog on a sphere or surface map (call after ensure_fog_look()): the add-on's parts and floor material
    (_ER_Parts, parts.py, which takes it by name when it builds them) made a fog material, kept with a fake user
    until the add-on builds them, so the Fog group is kept too (with no users, saving drops both, and these maps had no
    fog: the user, 2026-11-16, "i just reflew torus_square and there is no fog"); and fog_distance at distance drone
    sizes, as on every map, so it grows and shrinks with the drone (the sphere's own shader takes it too). The fog is
    there to hide the clip end's sharp edge (the user, same day): at the template's 23000, with the clip end at 50000
    (100000 clip starts of 0.5, which scale with the drone too), even red is down to 0.2% there."""
    material = fog_material(PARTS_MATERIAL, bpy.data.node_groups[GROUP], True)
    material.use_fake_user = True
    scene["fog_distance"] = distance
    return material


def is_earth(group):
    """Whether the Fog group has Earth's air: as it says, or, built before it said, by the Invert node only that look
    has"""
    return bool(group.get("earth", any(n.bl_idname == 'ShaderNodeInvert' for n in group.nodes)))


def light_by_sun(scene, sun=None, sun_height=SUN_HEIGHT):
    """Add sunlight (see the module's docstring) to a scene that has the fog, keeping its look, materials, fog_distance
    and fog_color. sun: the scene's sun lamp (sun_object()) by default. Returns a line saying what it did."""
    group = bpy.data.node_groups.get(GROUP)
    if group is None:
        raise ValueError(f"{scene.name} has no fog ({GROUP} node group) to light")
    sun = sun or sun_object(scene)
    if sun is None:
        raise ValueError(f"{scene.name} has no sun lamp to light it")
    earth = is_earth(group)
    ensure_fog_properties(scene, sun=True)
    fog_shading(scene, earth, sun, sun_height)
    toward = sun.matrix_world.to_3x3().col[2].normalized()
    height = math.degrees(math.asin(max(-1.0, min(1.0, toward.z))))
    return (f"Rendered mode lit by {sun.name}, keeping the {'Earth air' if earth else 'stylized'} fog: from azimuth "
            f"{math.degrees(math.atan2(toward.y, toward.x)):.0f}, "
            f"{f'{sun_height:.0f} degrees up (the lamp is {height:.1f})' if sun_height is not None else f'{height:.1f} degrees up'}, "
            f"sun_light {scene['sun_light']:.2f} (Scene Properties > Custom Properties, 0 to 1)")
