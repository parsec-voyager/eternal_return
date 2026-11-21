"""The drone's rig, built in code: the Drone, a flat triangle the core moves (physics.py), and the Drone_Camera,
parented to the triangle's three vertices and tilted up by the scene's Camera Tilt (a driver). The HUD is drawn
(hud.py, hud_draw.py).

Until stage B (2026-11-20), the Drone was appended from DroneCam.blend, with DroneCam's geometry nodes simulation as its
modifier (DRONE_MODIFIER, with the _DroneCam_ node group and the groups in it), whose inputs held the physics settings.
The rig was the same in every scene (surveyed 2026-11-20: 112 drones in 110 files, all the same three vertices but one,
the same inputs). migrate() takes it out of a scene from before, keeping all it held:
- the drone where the simulation showed it, as the next flight goes on from there
- the modifier's inputs, as the scene's physics settings (physics.DronePhysics)
- the camera tilt driver, reading the scene's Camera Tilt instead of the modifier's input, at the tilt the camera shows
- a recording of inputs keyed on the rig's _DroneCam_Inputs group, as a recording on the core (physics.Recording),
  replayed as the rig replayed it: from rest at the Drone object's origin, by the rig's own gravity
- and the carriers that took the sticks and gravity into the simulation mid-flight, before the core, go.
It runs at the start of the first flight or bake on the scene, and like any change, is only in the file once it's saved.
"""
import math
import re

import bpy
from bpy_extras import anim_utils

from . import gravity
from . import physics

DRONE_MODIFIER = "_DroneCam_GeometryNodes"
MESH_NAME = "_ER_Drone"
# The rig's triangle, about its center, in the Drone object's space: 20 cm wide, 20 cm deep
TRIANGLE = ((-0.1, -1.0 / 15.0, 0.0), (0.1, -1.0 / 15.0, 0.0), (0.0, 2.0 / 15.0, 0.0))

# The rig's modifier inputs, by the physics setting they were
SOCKETS = {"mass": "Socket_28", "gravity": "Socket_3", "ground_height": "Socket_9", "reference_area": "Socket_27",
           "drag_coefficient": "Socket_25", "air_density": "Socket_26", "camera_tilt": "Socket_15"}
# The camera's tilt driver: its expression, and what its variable reads
TILT_EXPRESSION = "90*pi/180+var"
TILT_PATH = "drone_physics.camera_tilt"
# The rig's group the recordings of inputs were keyed on, and what carried the sticks into the simulation mid-flight
INPUTS_GROUP = "_DroneCam_Inputs"
STICKS_OBJECT = "Drone_Sticks"
STICK_PATH = re.compile(r'nodes\["Group Output"\]\.inputs\[(\d+)\]\.default_value')


def build_drone(scene):
    """A new Drone, at rest at the world origin, turned as DroneCam's was added: its triangle's normal (Z) along world
    -Y"""
    mesh = bpy.data.meshes.new(MESH_NAME)
    mesh.from_pydata(TRIANGLE, [], [(0, 1, 2)])
    physics.keep_rest(mesh)
    drone_obj = bpy.data.objects.new("Drone", mesh)
    drone_obj.rotation_euler = (-math.pi / 2, 0.0, 0.0)
    scene.collection.objects.link(drone_obj)
    scene.drone_physics.rig_read = True
    if physics.available():
        physics.write(drone_obj, physics.at_rest())
    return drone_obj


def build_camera(scene, drone_obj, clip_start):
    """A new Drone_Camera on drone_obj's triangle, a 10 mm lens with the add-on's clip range from clip_start"""
    camera_data = bpy.data.cameras.new(name="Drone_Camera")
    camera_data.lens = 10
    camera_data.clip_start = clip_start
    camera_data.clip_end = clip_start * 100_000
    camera_obj = bpy.data.objects.new("Drone_Camera", camera_data)
    scene.collection.objects.link(camera_obj)
    camera_obj.rotation_euler[0] = math.pi / 2
    camera_obj.parent = drone_obj
    camera_obj.parent_type = 'VERTEX_3'
    camera_obj.parent_vertices = [0, 1, 2]
    drive_camera_tilt(camera_obj, scene)
    return camera_obj


def drive_camera_tilt(camera_obj, scene):
    """Have camera_obj's X rotation follow the scene's Camera Tilt, in place of any driver it had there"""
    animation = camera_obj.animation_data or camera_obj.animation_data_create()
    old = animation.drivers.find("rotation_euler", index=0)
    if old is not None:
        animation.drivers.remove(old)
    driver = animation.drivers.new("rotation_euler", index=0).driver
    driver.type = 'SCRIPTED'
    var = driver.variables.new()
    var.name = "var"
    var.type = 'SINGLE_PROP'
    var.targets[0].id_type = 'SCENE'
    var.targets[0].id = scene
    var.targets[0].data_path = TILT_PATH
    driver.expression = TILT_EXPRESSION


def socket_value(modifier, identifier, default):
    socket = getattr(modifier.properties.inputs, identifier, None)
    return socket.value if socket is not None else default


def rig_groups(modifier):
    """The rig's node groups: the modifier's and every group inside it"""
    found = []
    stack = [modifier.node_group] if modifier.node_group else []
    while stack:
        group = stack.pop()
        if group not in found:
            found.append(group)
            stack.extend(node.node_tree for node in group.nodes if node.type == 'GROUP' and node.node_tree)
    return found


def has_rig(drone_obj):
    return drone_obj is not None and DRONE_MODIFIER in drone_obj.modifiers


def migrate(scene, drone_obj, camera_obj, depsgraph):
    """Take DroneCam's rig out of drone_obj (see the module's docstring). Returns whether there was one."""
    if not has_rig(drone_obj) or not physics.available():
        return False
    modifier = drone_obj.modifiers[DRONE_MODIFIER]

    # Where the simulation shows the drone, while it still can
    pose = physics.read_state(drone_obj, depsgraph)
    physics.keep_rest(drone_obj.data)

    drone_physics = scene.drone_physics
    if not drone_physics.rig_read:
        for name, identifier in SOCKETS.items():
            if name != "ground_height":
                setattr(drone_physics, name, socket_value(modifier, identifier, getattr(drone_physics, name)))
        ground = socket_value(modifier, SOCKETS["ground_height"], physics.NO_GROUND)
        drone_physics.ground = ground > physics.NO_GROUND
        if drone_physics.ground:
            drone_physics.ground_height = ground
        if camera_obj is not None:
            # The tilt the camera shows, should its driver not have followed the rig's input (some scenes' drivers
            # were set to a constant, or read the input by a path Blender no longer has)
            shown = camera_obj.evaluated_get(depsgraph).rotation_euler.x
            drone_physics.camera_tilt += math.remainder(shown - camera_tilt_angle(drone_physics), math.tau)
        drone_physics.rig_read = True

    groups = rig_groups(modifier)
    convert_recording(scene, drone_obj, modifier, groups)
    if camera_obj is not None:
        drive_camera_tilt(camera_obj, scene)

    drone_obj.modifiers.remove(modifier)
    remove_groups(groups)
    remove_carriers()

    physics.write(drone_obj, pose)
    return True


def camera_tilt_angle(drone_physics):
    """The camera's X rotation its tilt driver gives (TILT_EXPRESSION)"""
    return math.pi / 2 + drone_physics.camera_tilt


def remove_groups(groups):
    """The rig's groups, but those something outside the rig still uses, and the groups inside those. By what uses
    them rather than their user counts: a node editor left showing one counts as a user too."""
    user_map = bpy.data.user_map(subset=groups)
    keep = {group for group in groups if any(user not in groups for user in user_map[group])}
    grown = True
    while grown:
        grown = False
        for group in groups:
            if group not in keep and any(user in keep for user in user_map[group]):
                keep.add(group)
                grown = True
    bpy.data.batch_remove([group for group in groups if group not in keep])


def remove_carriers():
    """The objects that carried the sticks (Drone_Sticks) and gravity (gravity.CARRIER) into the simulation in scenes
    saved mid-flight before the core"""
    for obj in [obj for obj in bpy.data.objects if obj.name == STICKS_OBJECT or gravity.is_carrier(obj)]:
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def convert_recording(scene, drone_obj, modifier, groups):
    """A recording of inputs on the rig (the sticks keyed on its _DroneCam_Inputs group's outputs, the drone's commands)
    as a recording on the core, assigned to the Drone. The rig replayed it from rest at the Drone object's origin,
    simulating from the frame before the first key, or the scene's start frame (its simulation starts there at the
    earliest), by its own physics settings (its gravity down its Z axis) and the scene's flight mode: so does the
    core."""
    group = next((group for group in groups if group.name.startswith(INPUTS_GROUP)), None)
    animation = group.animation_data if group else None
    action = animation.action if animation else None
    if action is None or animation.action_slot is None:
        return
    channelbag = anim_utils.action_get_channelbag_for_slot(action, animation.action_slot)
    outputs = group.nodes.get("Group Output")
    sticks = {}
    for fcurve in channelbag.fcurves if channelbag and outputs else ():
        match = STICK_PATH.fullmatch(fcurve.data_path)
        if match and int(match.group(1)) < len(outputs.inputs) and len(fcurve.keyframe_points):
            sticks[outputs.inputs[int(match.group(1))].name] = fcurve
    if not sticks:
        return
    first = min(int(math.ceil(fcurve.keyframe_points[0].co.x)) for fcurve in sticks.values())
    last = max(int(math.floor(fcurve.keyframe_points[-1].co.x)) for fcurve in sticks.values())

    ground = socket_value(modifier, SOCKETS["ground_height"], physics.NO_GROUND)
    level, limit = physics.core.flight_mode_params(physics.stick_settings(scene))
    constant = {
        "gravity": (0.0, 0.0, socket_value(modifier, SOCKETS["gravity"], 0.0)),
        "mass": socket_value(modifier, SOCKETS["mass"], 0.5),
        "drag_coefficient": socket_value(modifier, SOCKETS["drag_coefficient"], 0.8),
        "air_density": socket_value(modifier, SOCKETS["air_density"], 1.225),
        "reference_area": socket_value(modifier, SOCKETS["reference_area"], 0.04),
        "ground": ground - drone_obj.matrix_world.translation.z if ground > physics.NO_GROUND else physics.NO_GROUND,
        "auto_level": level, "angle_limit": physics.NO_ANGLE_LIMIT if limit is None else limit,
    }
    start = max(first - 1, scene.frame_start)
    take = physics.Recording.start(drone_obj, scene, physics.at_rest(), start_frame=start)
    for at in range(start + 1, last + 1):
        inputs = dict(constant)
        for name, key in (("thrust", "throttle"), ("pitch_rate", "pitch"), ("roll_rate", "roll"),
                          ("yaw_rate", "yaw")):
            inputs[name] = sticks[key].evaluate(at) if key in sticks else 0.0
        take.key_inputs(at, inputs)
    take.finish()
    take.action.name = f"{physics.RECORDING_NAME} (DroneCam)"
