"""Make eternal_return.blend, the template new maps start from, from scratch: from an empty Blender, the drone rig
(made by this repo's add-on, as Add Drone does), the user's add-on settings and controller mapping, the sample
twisted torus with its loop config and the sun, the fog (fog_look.py), the sky with the sun in it in both viewport
modes (sky.py), and the viewport settings maps want, Relationship Lines off among them. The sun lights both viewport modes: the fog's materials take its light in Rendered mode (fog_look.py's
sunlight, how much being the scene's sun_light), and Solid mode's studio lighting faces it (solid_sun.py), without
Solid mode's shadow, which cost too much.

Usage:
    Blender -b --factory-startup --python tools/make_template.py -- <out.blend> [--fog earth|stylized] [--overwrite]

It won't replace an existing file without --overwrite (Blender keeps the old one as <out>.blend1).

- The add-on is registered from this repo. Its SDL wheel comes from the installed extension's packages, which
  --factory-startup leaves off the path.
- The scaling configs are the files in the Proton Drive blender folder, loaded as external Texts, so editing a file
  updates the scenes that use it: loop_config (the torus's), plane_config, and default_config, which the add-on applies
  when the drone deselects an object (Auto Deselect), so releasing the click undoes a clicked object's loop or wrap.
- floor_config, for maps built like the Inferno (endless floors of nested levels), made here from the add-on's
  default_config.json: auto scale by height above the floor only, and loop by 3 whenever the drone's scale leaves 0.98 / 3
  to 1. That keeps the drone between the same two heights, so a map only has to be built for that band of scales, and
  the 2% past 1 / 3 keeps a loop from landing on the band's other end and bouncing straight back. Wrap is off: an
  endless map moves the drone back itself; another map turns it on for its own range.
- The fog's materials, Fog_look (the "Col" color attribute) and Fog_look_plain (the object's viewport color), both
  kept for new objects, and the EEVEE settings the fog maps use: no shadows or ray tracing, which the fog's
  emission-only materials don't use (the sun's shadows then don't render either), and 4 viewport samples.
- --fog earth (the default) is Earth's air, clear up close and blue only at mountain distances, 23 km visibility at
  scale 1. --fog stylized is the thicker blue fog, 60 m. Either way the drone camera's clip is the add-on's: a 0.5 m
  clip start (Default Clip Start, the user's since 2026-10-29) and 100,000 times that, 50 km, past where Earth's air has
  hidden everything, so the fog ends the view rather than a hard edge. The fog shows in the Rendered viewport
  (Z > Rendered).
"""

import glob
import math
import os
import sys

import bpy
from mathutils import Quaternion, Vector

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The Proton Drive blender folder, found by pattern so the account name isn't written here
BLENDER_FOLDER = next(iter(sorted(glob.glob(os.path.expanduser("~/Library/CloudStorage/ProtonDrive-*-folder/blender")))),
                      "")

sys.path.insert(0, os.path.join(REPO, "tools"))
import fog_look
import sky
import solid_sun

# The scene: 60 fps, and a frame range long enough to fly for hours, since the drone's physics run as frames play
FPS = 60
FRAMES = (1, 1048574)
RESOLUTION = (1920, 1080)

# The user's add-on settings that differ from its defaults. The axes are the RadioMaster Pocket's (EdgeTX, AETR):
# throttle on 5 and scale on 2. The wrap settings are what the last config clicked left, which the next click resets.
CONTROLS = {
    "throttle_invert": True, "scale_invert": False, "loop": True, "loop_factor": 8.0, "keep_drone_at_origin": True,
    "auto_deselect": True, "bounded": True, "auto_brake": True, "default_clip_start": 0.5,
    "wrap_split_axes_y": False, "wrap_split_axes_z": False, "wrap_range_x": (-1.8886380195617676, 2.1113619804382324),
}
AXIS_MAPPING = {"throttle": "5", "scale": "2"}
OTHER = {"volume": 0.0}
# The HUD on clip_start (its geometry nodes' inputs): no gimbal values, altitude or speed readouts; scale speed on
HUD = {"Socket_10": False, "Socket_11": False, "Socket_12": False, "Socket_20": True}

DRONE_LOCATION = (0.0, -28.0, 0.0)
# A sample object to fly around, with loop_config: a twisted torus (Extra Objects' Twisted Torus, 48 x 12 segments,
# one twist, made 4 times its 1 m size) made into a lattice of beams by a Wireframe modifier. A sun on it, for
# materials that aren't the fog's.
TORUS_LOCATION = (-7.50634765625, 12.936676025390625, -11.833893775939941)
TORUS_ROTATION = (-0.41221681237220764, -2.347379446029663, 0.032775308936834335)
SUN_LOCATION = (-7.506345748901367, 12.936676025390625, -11.833893775939941)
SUN_ROTATION = (-1.5556696653366089, 1.2218681573867798, -1.1466861963272095)
SUN_ENERGY, SUN_ANGLE = 45.0, 0.009180432185530663
CONFIGS = ("loop_config", "plane_config", "default_config")
# floor_config: the add-on's default_config.json with these values (see the module's docstring)
FLOOR_LOOP = 3.0
FLOOR_CONFIG = {'"loop": false': '"loop": true', '"loop_factor": 10.0': f'"loop_factor": {FLOOR_LOOP}',
                '"auto_scale": false': '"auto_scale": true',
                '"auto_scale_split_axes": false': '"auto_scale_split_axes": true',
                '"auto_scale_x": true': '"auto_scale_x": false', '"auto_scale_y": true': '"auto_scale_y": false',
                '"min_scale": 0.000001': f'"min_scale": {0.98 / FLOOR_LOOP!r}', '"max_scale": 33000000.0': '"max_scale": 1.0'}

# The Layout workspace's 3D view, looking through the drone camera: Solid mode with the fullmetal MatCap, random object
# colors, screen-space cavity, and the blue sky
SKY = (0.0, 0.31855452060699463, 1.0)
FOG_LOOKS = ("earth", "stylized")


def register_addon():
    """This repo's add-on, as Blender would load it as an extension"""
    sys.path[:0] = glob.glob(os.path.join(bpy.utils.user_resource('EXTENSIONS'), ".local", "lib", "python*",
                                          "site-packages"))
    sys.path.insert(0, REPO)
    import eternal_return
    eternal_return.register()


def empty_scene():
    """The factory startup scene without its cube, camera, light and collection"""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)
    bpy.data.orphans_purge(do_recursive=True)


def scene_settings(scene):
    scene.render.fps, scene.render.fps_base = FPS, 1.0
    scene.frame_start, scene.frame_end = FRAMES
    scene.frame_current = FRAMES[0]
    scene.render.resolution_x, scene.render.resolution_y = RESOLUTION
    scene.render.engine = 'BLENDER_EEVEE'
    # The fog maps' settings: their materials are emission only
    scene.eevee.use_shadows = False
    scene.eevee.use_raytracing = False
    scene.eevee.taa_samples = 4


def add_drone(scene):
    """The rig, as the add-on's Add Drone makes it, with the user's settings"""
    for name, value in CONTROLS.items():
        setattr(scene.controls_parameters, name, value)
    for name, value in AXIS_MAPPING.items():
        setattr(scene.controls_axis_mapping, name, value)
    for name, value in OTHER.items():
        setattr(scene.other_parameters, name, value)
    bpy.ops.er.add_to_scene()
    drone, camera = bpy.data.objects["Drone"], bpy.data.objects["Drone_Camera"]
    drone.location = DRONE_LOCATION
    # Hidden, as in the user's template: it's only the three vertices the camera hangs from
    drone.hide_set(True)
    hud = bpy.data.objects["clip_start"].modifiers[0]
    for identifier, value in HUD.items():
        getattr(hud.properties.inputs, identifier).value = value
    scene.camera = camera


def twisted_torus(major_radius, minor_radius, major_segments, minor_segments, twists):
    """Extra Objects' Twisted Torus: a ring of circles around Z, each turned by its share of the twists"""
    verts, faces = [], []
    for i in range(major_segments):
        around = Quaternion((0.0, 0.0, 1.0), 2 * math.pi * i / major_segments)
        twist = 2 * math.pi * i / major_segments * twists
        for j in range(minor_segments):
            angle = 2 * math.pi * j / minor_segments + twist
            verts.append(around @ Vector((major_radius + math.cos(angle) * minor_radius, 0.0,
                                          math.sin(angle) * minor_radius)))
    for i in range(major_segments):
        n = (i + 1) % major_segments
        for j in range(minor_segments):
            k = (j + 1) % minor_segments
            faces.append((i * minor_segments + j, n * minor_segments + j, n * minor_segments + k,
                          i * minor_segments + k))
    return verts, faces


def add_sample(scene):
    collection = bpy.data.collections.new("the sun")
    scene.collection.children.link(collection)

    verts, faces = twisted_torus(4.0, 1.0, 48, 12, 1)
    mesh = bpy.data.meshes.new("TwistedTorus.002")
    mesh.from_pydata([tuple(v) for v in verts], [], faces)
    torus = bpy.data.objects.new("TwistedTorus.002", mesh)
    collection.objects.link(torus)
    wireframe = torus.modifiers.new("Wireframe", 'WIREFRAME')
    wireframe.thickness, wireframe.offset = 0.24, 0.0
    # Applied, as the template's was: the beams are the mesh
    beams = bpy.data.meshes.new_from_object(torus.evaluated_get(bpy.context.evaluated_depsgraph_get()))
    torus.modifiers.remove(wireframe)
    torus.data = beams
    bpy.data.meshes.remove(mesh)
    beams.name = "TwistedTorus.002"
    torus.location, torus.rotation_euler = TORUS_LOCATION, TORUS_ROTATION

    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", 'SUN'))
    sun.data.energy, sun.data.angle = SUN_ENERGY, SUN_ANGLE
    sun.location, sun.rotation_euler = SUN_LOCATION, SUN_ROTATION
    collection.objects.link(sun)
    return torus


def load_configs(torus):
    texts = {}
    for name in CONFIGS:
        path = os.path.join(BLENDER_FOLDER, name)
        if os.path.exists(path):
            texts[name] = bpy.data.texts.load(path)
        else:
            print(f"No {path}: {name} left out")
    if "loop_config" in texts:
        torus["scaling_config"] = texts["loop_config"]

    # floor_config, from the add-on's own config file so it has every key, laid out the same
    with open(os.path.join(REPO, "eternal_return", "default_config.json")) as f:
        content = f.read().replace("\r\n", "\n")
    for old, new in FLOOR_CONFIG.items():
        assert content.count(old) == 1, old
        content = content.replace(old, new)
    floor = bpy.data.texts.new("floor_config")
    floor.write(content)
    texts["floor_config"] = floor
    return list(texts)


def set_viewport(scene):
    for screen in bpy.data.screens:
        for area in screen.areas:
            for space in area.spaces:
                if space.type != 'VIEW_3D':
                    continue
                # Outline Selected would outline whatever the drone clicks
                space.overlay.show_outline_selected = False
                if screen.name != "Layout":
                    continue
                shading = space.shading
                shading.type = 'SOLID'
                shading.color_type = 'RANDOM'
                shading.show_cavity, shading.cavity_type = True, 'SCREEN'
                shading.background_type, shading.background_color = 'VIEWPORT', SKY
                space.overlay.show_floor = False
                space.clip_start, space.clip_end = 0.1, 10000.0
                space.region_3d.view_perspective = 'CAMERA'
    # Every view's lighting, and Workbench renders', from the sun
    solid_sun.apply(scene)
    # The sky with that sun in it, in both modes, and no Relationship Lines (see sky.py)
    sky.apply(scene)


def main():
    if not ARGS or ARGS[0].startswith("--"):
        raise SystemExit("Usage: Blender -b --factory-startup --python tools/make_template.py -- <out.blend> "
                         "[--fog earth|stylized] [--overwrite]")
    output = os.path.abspath(os.path.expanduser(ARGS[0]))
    look = ARGS[ARGS.index("--fog") + 1] if "--fog" in ARGS else "earth"
    if look not in FOG_LOOKS:
        raise SystemExit(f"--fog is earth or stylized, not {look}")
    if os.path.exists(output) and "--overwrite" not in ARGS:
        raise SystemExit(f"{output} exists: pass --overwrite to replace it")

    register_addon()
    empty_scene()
    scene = bpy.context.scene
    scene_settings(scene)
    add_drone(scene)
    torus = add_sample(scene)
    configs = load_configs(torus)
    fog_look.ensure_fog_look(scene, [torus], earth=look == "earth")
    # Fog_look, for meshes with vertex colors, has no user yet: kept, for new objects
    bpy.data.materials[fog_look.MATERIAL].use_fake_user = True
    # The factory world, which ensure_fog_look() kept with a fake user
    for world in list(bpy.data.worlds):
        if world.name != fog_look.WORLD:
            bpy.data.worlds.remove(world)
    set_viewport(scene)
    bpy.data.orphans_purge(do_recursive=True)

    # Paths kept absolute, so the configs still load in maps saved anywhere from it
    bpy.ops.wm.save_as_mainfile(filepath=output, relative_remap=False)
    print("MADE", output)
    print(f"  fog: {look}, fog_distance {scene['fog_distance']:g}; drone camera clip "
          f"{scene.camera.data.clip_start:g} to {scene.camera.data.clip_end:g}; configs: {', '.join(configs)}")


if __name__ == "__main__":
    main()
