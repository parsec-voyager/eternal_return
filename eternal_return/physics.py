"""The drone's flight on the Eternal Return core (the eternal_return_core wheel).

The core is one pure function, step(state, commands, params, dt), in float64: gravity, the turn for the step, thrust,
drag, the move, and the ground, in the Drone object's space (see eternal_return_core's spec/physics.md). It flew every
reference flight recorded from DroneCam's geometry nodes simulation to float32 precision, so it flies as that rig did,
minus the rig's stray Clamp that limited roll one way only in Angle mode. The rig itself is gone since stage B
(2026-11-20): rig.py builds the drone without it and takes it out of older scenes.

The Drone's mesh is one flat triangle, the rest triangle (REST_KEY), and the Drone_Camera is parented to its three
vertices. Once a frame, before the frame is evaluated (flight.fly_frame()), the drone's state is stepped, and the mesh's
vertices are set to the rest triangle under the new orientation and position, with the "vel" attribute
(velocity.VELOCITY_ATTRIBUTE) the HUD's velocity marker reads. The camera then follows. The mesh stays where a flight
left it: the next flight starts from the same position, orientation and velocity (take_over() reads them back from the
mesh), and the view doesn't move in between.

The physics settings are the scene's (DronePhysics, scene.drone_physics, in the Drone panel), read every frame, so the
panel changes them mid-flight. Brake raises the air density for as long as it holds (brake_density). The flight mode is
the scene's controls, and the gravity the add-on's pull toward the selected object (gravity.py).

Recording inputs keys everything the core is given each frame onto the Drone object (DroneRecording,
obj.drone_recording: the commands and the physics settings), along with the Drone object's own transform, which
scaling, orbit, recenters and Keep Drone at Origin move, in an action of its own (RECORDING_KEY holds where it started).
Playing the timeline replays it (replay()): the same steps from the same start, so a replay is the flight, exactly: the
recorded flight itself flies on the float32 values the keyframes hold. The keyed sticks can be edited in the Graph
Editor, and the replay flies them. Bake Animation bakes the replay into an editable camera.
"""
import math

import bpy
import mathutils
import numpy as np
from bpy_extras import anim_utils

try:
    import eternal_return_core as core
except ImportError:
    # Not shipped for this platform (the wheel is macOS arm64 only so far): no flights
    core = None

from . import velocity

# How far (in the drone's own units) the drone can drift from the Drone object's origin before the object is moved onto
# it (advance()). Far from the origin, float32 vertices make the camera shake.
RECENTER_DISTANCE = 500.0

# The rig's Ground Height when there's no ground, as recordings key it: anything this low is no ground
NO_GROUND = -1.0e30
# The Angle Limit recordings key for none
NO_ANGLE_LIMIT = -1.0

# The custom property on the Drone's mesh keeping its rest vertices while the core moves them, as a flat list
REST_KEY = "er_rest"
# The custom property marking an action as a recording of inputs, holding its start: a dict of start_frame, position,
# velocity, orientation (w, x, y, z), dt (the step) and recenter_distance, all float64
RECORDING_KEY = "er_recording"
RECORDING_NAME = "Drone Recording"

# The drone's state while flying (a core.DroneState), else None
state = None
# The frame state was last stepped into or read for. Blender runs the frame change handlers again for a frame it
# renders: only a step to the next frame is flight.
frame = None
# The stick settings of the last tick (a core.StickSettings), for the flight mode's physics
settings = None
# The frame the last recenter happened on, and by how much in the Drone object's space (for the flight harness)
recentered = None
# The air density brake holds while it brakes (flight.FlightFeatures.brake()), else None for the scene's own
brake_density = None
# The recording being keyed while flying with Record on (a Recording), else None
recording = None
# What replay() last showed: (the recording's action pointer, start frame), the frame, and the state there
_replayed = None


def tag_scene(physics_settings, context):
    """Have the drivers reading a setting (the camera's tilt driver, rig.drive_camera_tilt()) see its change: set from
    Python, nothing else tags the scene"""
    physics_settings.id_data.update_tag()


class DronePhysics(bpy.types.PropertyGroup):
    """The drone's physics settings, the scene's: what DroneCam's rig had as its geometry nodes modifier's inputs.
    rig.py reads them from that rig, in scenes from before stage B."""
    mass: bpy.props.FloatProperty(
        name="Mass", description="The drone's mass. Heavier, the same thrust speeds it up less", default=0.5,
        min=0.001, soft_max=10.0, unit='MASS')
    gravity: bpy.props.FloatProperty(
        name="Gravity",
        description="Pull along the Drone's Z axis (negative pulls down), or with an object selected, toward it "
                    "(negative pulls in). In the drone's own units, so it handles the same at any scale",
        default=0.0, soft_min=-20.0, soft_max=20.0, unit='ACCELERATION')
    ground: bpy.props.BoolProperty(
        name="Ground", description="Stop the drone falling below Ground Height (world Z)", default=False)
    ground_height: bpy.props.FloatProperty(
        name="Ground Height", description="The world height the drone can't go below", default=0.0, unit='LENGTH')
    drag_coefficient: bpy.props.FloatProperty(
        name="Drag Coefficient", description="How much the drone's shape drags in the air", default=0.8, min=0.0,
        soft_max=2.0)
    reference_area: bpy.props.FloatProperty(
        name="Reference Area", description="The drone's area facing the air, for drag", default=0.04, min=0.0,
        soft_max=1.0, unit='AREA')
    air_density: bpy.props.FloatProperty(
        name="Air Density", description="The air's density (kg/m³): denser, the drone slows down sooner. Brake raises "
                                        "it while braking", default=1.225, min=0.0, soft_max=10.0)
    camera_tilt: bpy.props.FloatProperty(
        name="Camera Tilt", description="The camera's tilt up from the drone's frame, as on an FPV drone",
        default=math.pi / 2, soft_min=0.0, soft_max=math.pi / 2, subtype='ANGLE', update=tag_scene)
    rig_read: bpy.props.BoolProperty(
        name="Read from Rig", description="The settings were read from DroneCam's rig once (rig.py)", default=False)


class DroneRecording(bpy.types.PropertyGroup):
    """What the core was given on a recorded frame, keyed on the Drone object (see Recording): the four commands, and
    the physics as they were then. Editable like any keyframes."""
    thrust: bpy.props.FloatProperty(name="Thrust", description="Thrust, in newtons")
    pitch_rate: bpy.props.FloatProperty(name="Pitch Rate", description="Pitch rate, in radians a second")
    roll_rate: bpy.props.FloatProperty(name="Roll Rate", description="Roll rate, in radians a second")
    yaw_rate: bpy.props.FloatProperty(name="Yaw Rate", description="Yaw rate, in radians a second")
    gravity: bpy.props.FloatVectorProperty(name="Gravity", size=3,
                                           description="Gravity, in the Drone object's space and units")
    mass: bpy.props.FloatProperty(name="Mass")
    drag_coefficient: bpy.props.FloatProperty(name="Drag Coefficient")
    air_density: bpy.props.FloatProperty(name="Air Density")
    reference_area: bpy.props.FloatProperty(name="Reference Area")
    ground: bpy.props.FloatProperty(name="Ground", description=f"The ground's height in the Drone object's space "
                                                               f"({NO_GROUND:g} for none)")
    auto_level: bpy.props.FloatProperty(name="Auto-Level", description="Auto-level strength (0 to 1)")
    angle_limit: bpy.props.FloatProperty(name="Angle Limit", description=f"In radians ({NO_ANGLE_LIMIT:g} for none)")
    # What the HUD showed (hud.py: recorded()), so a replay shows it too
    hud_axes: bpy.props.FloatVectorProperty(name="HUD Sticks", size=4, description="The sticks' positions (-1 to 1)")
    hud_altitude: bpy.props.FloatVectorProperty(name="HUD Altitude", size=2, description="log10 meters, sign")
    hud_speed: bpy.props.FloatProperty(name="HUD Speed", description="log10 meters a second")
    hud_scale: bpy.props.FloatProperty(name="HUD Scale", description="log10 of the scale")
    hud_scale_speed: bpy.props.FloatProperty(name="HUD Scale Speed", description="log10 of the scale's change a second")


# The inputs a frame's step is given, as keyed: (DroneRecording property, size)
CHANNELS = (("thrust", 1), ("pitch_rate", 1), ("roll_rate", 1), ("yaw_rate", 1), ("gravity", 3), ("mass", 1),
            ("drag_coefficient", 1), ("air_density", 1), ("reference_area", 1), ("ground", 1), ("auto_level", 1),
            ("angle_limit", 1))
# What the HUD showed on a recorded frame, as keyed: held from frame to frame
HUD_CHANNELS = (("hud_axes", 4), ("hud_altitude", 2), ("hud_speed", 1), ("hud_scale", 1), ("hud_scale_speed", 1))


def available():
    return core is not None


def is_flying():
    return state is not None


def at_rest():
    """Upright at the Drone object's origin, not moving"""
    return core.DroneState((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


def rest_positions(mesh):
    """The rest triangle as an (n, 3) float64 array: kept on the mesh once the core has moved it, else the mesh's own
    vertices"""
    rest = mesh.get(REST_KEY)
    if rest is not None and len(rest) == 3 * len(mesh.vertices):
        return np.array(rest, dtype=np.float64).reshape(-1, 3)
    positions = np.empty(3 * len(mesh.vertices), dtype=np.float64)
    mesh.vertices.foreach_get("co", positions)
    return positions.reshape(-1, 3)


def keep_rest(mesh):
    """Keep the mesh's vertices as its rest triangle, before the core first moves them"""
    if REST_KEY not in mesh:
        mesh[REST_KEY] = rest_positions(mesh).ravel().tolist()


def fit_pose(rest, moved):
    """The rotation matrix and position moving rest onto moved (both (n, 3)) as one rigid piece, by least squares
    (Kabsch): exact up to the float32 rounding of the moved vertices. A flat triangle pins the rotation down too: its
    normal's sign comes from keeping the rotation proper."""
    rest_center, moved_center = rest.mean(axis=0), moved.mean(axis=0)
    u, _, vt = np.linalg.svd((rest - rest_center).T @ (moved - moved_center))
    flip = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    rotation = vt.T @ np.diag((1.0, 1.0, flip)) @ u.T
    return rotation, moved_center - rotation @ rest_center


def read_state(drone_obj, depsgraph):
    """The drone's state as the evaluated mesh shows it: the core's last state written to the mesh, or in a scene from
    before stage B, the rig's simulation's (its evaluated mesh can have more vertices than the rest mesh, the original
    add-on's in-camera UI: the drone's own come first)"""
    rest = rest_positions(drone_obj.data)
    drone_eval = drone_obj.evaluated_get(depsgraph)
    mesh = drone_eval.data
    n = len(rest)
    if len(mesh.vertices) < n:
        return at_rest()
    moved = np.empty(3 * len(mesh.vertices), dtype=np.float64)
    mesh.vertices.foreach_get("co", moved)
    rotation, position = fit_pose(rest, moved.reshape(-1, 3)[:n])
    speed = velocity.drone_velocity(drone_eval)
    orientation = mathutils.Matrix(rotation.tolist()).to_quaternion()
    return core.DroneState(tuple(position.tolist()), tuple(speed) if speed is not None else (0.0, 0.0, 0.0),
                           (orientation.w, orientation.x, orientation.y, orientation.z))


def take_over(drone_obj, scene, depsgraph, record):
    """Start flying from where the drone is: its position, orientation and velocity as shown now. With record, the
    flight is recorded from here (Recording); without, a recording the Drone was replaying is put aside, kept in the
    file (its keyed transform would hold the Drone in place)."""
    global state, frame, recentered, brake_density, recording, _replayed
    state = read_state(drone_obj, depsgraph)
    keep_rest(drone_obj.data)
    frame = scene.frame_current
    recentered = brake_density = None
    _replayed = None
    put_aside_recording(drone_obj)
    recording = Recording.start(drone_obj, scene, state) if record else None
    write(drone_obj, state)


def reset(drone_obj):
    """Put the drone at rest: upright at the Drone object's origin, not moving (the flight harness starts its flights
    from there)"""
    global state
    state = at_rest()
    if recording is not None:
        recording.restart(state)
    write(drone_obj, state)


def stop():
    """End a flight. The mesh stays as the flight left it, for the next one."""
    global state, frame, settings, brake_density, recording
    if recording is not None:
        recording.finish()
    state = frame = settings = brake_density = recording = None


def stick_settings(scene):
    """The scene's controls as the core's stick settings (the panel's percentages as fractions)"""
    controls = scene.controls_parameters
    rates = scene.rotation_controls_actual_rates
    axes = ("throttle", "yaw", "pitch", "roll")
    rotation_axes = ("yaw", "pitch", "roll")
    return core.StickSettings(
        mode=controls.flying_mode,
        invert={axis: getattr(controls, f"{axis}_invert") for axis in axes},
        # A deadzone of the whole stick is no valid setting for the core (the old conversion divided by zero)
        deadzone={axis: min(getattr(controls, f"{axis}_deadzone") / 100.0, 0.999) for axis in axes},
        throttle_centered=controls.throttle_centered,
        max_thrust=scene.throttle_control_power.power,
        center_rate={axis: getattr(rates, f"{axis}_center_sensitivity") for axis in rotation_axes},
        max_rate={axis: getattr(rates, f"{axis}_max_rate") for axis in rotation_axes},
        expo={axis: getattr(rates, f"{axis}_expo") for axis in rotation_axes},
        auto_level_strength=controls.autoleveling_strength / 100.0,
        angle_limit=controls.angle_limit_value,
    )


def sticks_to_commands(dict_controls_raw, scene):
    """The four flight axes' raw sticks as the core's commands: thrust in newtons and rates in radians a second, keyed
    as flight.dict_controls keys them. Keeps the settings for the flight mode's physics (step_frame())."""
    global settings
    settings = stick_settings(scene)
    sticks = core.Sticks(**{axis: float(dict_controls_raw.get(axis, 0.0))
                            for axis in ("throttle", "yaw", "pitch", "roll")})
    commands = core.sticks_to_commands(sticks, settings)
    return {"throttle": commands.thrust, "yaw": commands.yaw_rate, "pitch": commands.pitch_rate,
            "roll": commands.roll_rate}


def air_density(scene):
    """The air density the drone flies in: brake's while it brakes, else the scene's"""
    return brake_density if brake_density is not None else scene.drone_physics.air_density


def frame_inputs(scene, drone_obj, commands, gravity):
    """Everything this frame's step is given, keyed as DroneRecording keys it: commands (the four flight axes, keyed
    as sticks_to_commands() returns them), the scene's physics, the flight mode's leveling and angle limit, and gravity
    (a vector in the Drone object's space). The ground is at Ground Height in the world: in the drone's space, minus
    the Drone object's world height, as the rig had it."""
    drone_physics = scene.drone_physics
    level, limit = core.flight_mode_params(settings) if settings is not None else (0.0, None)
    ground = drone_physics.ground_height - drone_obj.matrix_world.translation.z if drone_physics.ground else NO_GROUND
    return {
        "thrust": commands["throttle"], "pitch_rate": commands["pitch"], "roll_rate": commands["roll"],
        "yaw_rate": commands["yaw"], "gravity": tuple(float(c) for c in gravity), "mass": drone_physics.mass,
        "drag_coefficient": drone_physics.drag_coefficient, "air_density": air_density(scene),
        "reference_area": drone_physics.reference_area, "ground": ground,
        "auto_level": level, "angle_limit": NO_ANGLE_LIMIT if limit is None else limit,
    }


def advance(drone_state, inputs, dt, recenter_distance):
    """One step of dt from drone_state with inputs (frame_inputs()), recentered: once the drone has drifted
    recenter_distance from the Drone object's origin, the state is moved back onto it, by the shift returned (else
    None), which the Drone object has to move by so the camera doesn't: far from the origin, float32 vertices would make
    it shake"""
    commands = core.Commands(thrust=inputs["thrust"], pitch_rate=inputs["pitch_rate"], roll_rate=inputs["roll_rate"],
                             yaw_rate=inputs["yaw_rate"])
    params = core.DroneParams(
        mass=inputs["mass"], gravity=tuple(inputs["gravity"]), drag_coefficient=inputs["drag_coefficient"],
        air_density=inputs["air_density"], reference_area=inputs["reference_area"],
        ground_z=None if inputs["ground"] <= NO_GROUND else inputs["ground"],
        auto_level_strength=inputs["auto_level"],
        angle_limit=None if inputs["angle_limit"] < 0.0 else inputs["angle_limit"],
    )
    drone_state = core.step(drone_state, commands, params, dt)
    position = drone_state.position
    if math.hypot(*position) > recenter_distance:
        return core.DroneState((0.0, 0.0, 0.0), drone_state.velocity, drone_state.orientation), position
    return drone_state, None


def step_frame(scene, drone_obj, commands, gravity, hud_values=None):
    """Fly the drone into the scene's current frame, before it's evaluated: one step of 1 / fps with commands (the
    four flight axes, keyed as sticks_to_commands() returns them), then the mesh moved there. Recording, what the HUD
    shows (hud_values, Recording.key_hud()) is keyed too. Only a step from the
    frame before: a frame rendered again isn't flight, and a jump (scrubbing, playback starting over) isn't either.
    A recenter (advance(), at RECENTER_DISTANCE, or the recording's) moves the Drone object by the shift."""
    global state, frame, recentered
    if state is None or scene.frame_current == frame:
        return
    stepped = scene.frame_current == frame + 1
    frame = scene.frame_current
    if not stepped:
        return

    inputs = frame_inputs(scene, drone_obj, commands, gravity)
    if recording is not None:
        inputs = recording.key_inputs(frame, inputs)
        dt, recenter_distance = recording.dt, recording.recenter_distance
    else:
        dt, recenter_distance = scene.render.fps_base / scene.render.fps, RECENTER_DISTANCE
    state, shift = advance(state, inputs, dt, recenter_distance)
    if shift is not None:
        drone_obj.matrix_world = drone_obj.matrix_world @ mathutils.Matrix.Translation(shift)
        recentered = (frame, shift)
    if recording is not None:
        recording.key_transform(frame)
        if hud_values is not None:
            recording.key_hud(frame, hud_values)
    write(drone_obj, state)


def rotation_matrix(orientation):
    """The rotation matrix of a unit quaternion (w, x, y, z), in float64 (mathutils' would be float32)"""
    w, x, y, z = orientation
    return np.array((
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
        (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
        (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)),
    ))


def write(drone_obj, drone_state):
    """Put the Drone's mesh where drone_state has the drone: the rest triangle turned and moved, every vertex carrying
    the velocity"""
    mesh = drone_obj.data
    rest = rest_positions(mesh)
    moved = rest @ rotation_matrix(drone_state.orientation).T + np.array(drone_state.position, dtype=np.float64)
    mesh.vertices.foreach_set("co", moved.ravel())

    attribute = mesh.attributes.get(velocity.VELOCITY_ATTRIBUTE)
    if attribute is None or attribute.domain != 'POINT' or attribute.data_type != 'FLOAT_VECTOR':
        if attribute is not None:
            mesh.attributes.remove(attribute)
        attribute = mesh.attributes.new(velocity.VELOCITY_ATTRIBUTE, 'FLOAT_VECTOR', 'POINT')
    attribute.data.foreach_set("vector", np.tile(np.array(drone_state.velocity, dtype=np.float64),
                                                 len(mesh.vertices)))
    mesh.update_tag()


def float32(value):
    """value as the float32 a keyframe holds, as a Python float"""
    return float(np.float32(value))


def transform_path(obj):
    """The data path of obj's rotation in its rotation mode"""
    return {'QUATERNION': "rotation_quaternion", 'AXIS_ANGLE': "rotation_axis_angle"}.get(obj.rotation_mode,
                                                                                          "rotation_euler")


def is_recording(action):
    return action is not None and RECORDING_KEY in action


def assigned_recording(drone_obj):
    """The recording the Drone replays (its action, if that's a recording), else None"""
    animation = drone_obj.animation_data if drone_obj else None
    action = animation.action if animation else None
    return action if is_recording(action) else None


def put_aside_recording(drone_obj):
    """Take the Drone's recording off it, so a flight moves it freely, but keep it in the file (a fake user): assigning
    it again replays it"""
    action = assigned_recording(drone_obj)
    if action is not None:
        action.use_fake_user = True
        drone_obj.animation_data.action = None


class Recording:
    """A flight being recorded: every frame's inputs keyed onto the Drone object's drone_recording (DroneRecording),
    with its transform, in a new action of its own. Keyframes hold float32, so the flight flies on the float32 values
    (key_inputs()), and its replay is the same flight."""

    def __init__(self, drone_obj, action, start_frame, dt, recenter_distance):
        self.drone_obj = drone_obj
        self.action = action
        self.start_frame = start_frame
        self.dt = dt
        self.recenter_distance = recenter_distance
        self.channelbag = anim_utils.action_ensure_channelbag_for_slot(action, drone_obj.animation_data.action_slot)
        self.fcurves = {}
        # The rotation last keyed, which the next keeps continuous with: an orbit past half a turn would otherwise key
        # an Euler angle jumping by a full turn, which plays back as a spin between the frames
        self.euler = None

    @classmethod
    def start(cls, drone_obj, scene, start_state, start_frame=None):
        """A new recording from start_state at start_frame (default: the scene's current frame), assigned to the Drone
        (any action it had before is kept if it holds anything)"""
        animation = drone_obj.animation_data or drone_obj.animation_data_create()
        old = animation.action
        if old is not None and (is_recording(old) or old.layers):
            old.use_fake_user = True
        action = bpy.data.actions.new(RECORDING_NAME)
        animation.action = action
        if animation.action_slot is None:
            animation.action_slot = action.slots.new('OBJECT', drone_obj.name)
        take = cls(drone_obj, action, scene.frame_current if start_frame is None else start_frame,
                   scene.render.fps_base / scene.render.fps, RECENTER_DISTANCE)
        take.restart(start_state)
        return take

    @classmethod
    def of(cls, drone_obj):
        """The Drone's assigned recording as a Recording, else None"""
        action = assigned_recording(drone_obj)
        if action is None or drone_obj.animation_data.action_slot is None:
            return None
        start = action[RECORDING_KEY]
        take = cls(drone_obj, action, int(start["start_frame"]), float(start["dt"]), float(start["recenter_distance"]))
        take.fcurves = {(fcurve.data_path, fcurve.array_index): fcurve for fcurve in take.channelbag.fcurves}
        return take

    def restart(self, start_state):
        """Record from start_state at the start frame"""
        self.action[RECORDING_KEY] = {
            "start_frame": self.start_frame, "position": list(start_state.position),
            "velocity": list(start_state.velocity), "orientation": list(start_state.orientation), "dt": self.dt,
            "recenter_distance": self.recenter_distance}
        self.key_transform(self.start_frame)

    def start_state(self):
        start = self.action[RECORDING_KEY]
        return core.DroneState(tuple(start["position"]), tuple(start["velocity"]), tuple(start["orientation"]))

    def fcurve(self, data_path, index, group):
        fcurve = self.fcurves.get((data_path, index))
        if fcurve is None:
            fcurve = self.channelbag.fcurves.new(data_path, index=index, group_name=group)
            self.fcurves[data_path, index] = fcurve
        return fcurve

    def key(self, data_path, index, at, value, group, interpolation='LINEAR'):
        point = self.fcurve(data_path, index, group).keyframe_points.insert(at, value, options={'FAST'})
        point.interpolation = interpolation

    def key_inputs(self, at, inputs):
        """Key inputs (frame_inputs()) at frame at, and return them as keyed, in float32"""
        keyed = {}
        for name, size in CHANNELS:
            values = inputs[name] if size > 1 else (inputs[name],)
            values = tuple(float32(value) for value in values)
            for index, value in enumerate(values):
                self.key(f"drone_recording.{name}", index, at, value, "Inputs")
            keyed[name] = values if size > 1 else values[0]
        return keyed

    def key_hud(self, at, values):
        """Key what the HUD shows (HUD_CHANNELS names to tuples) at frame at"""
        for name, size in HUD_CHANNELS:
            for index, value in enumerate(values[name][:size]):
                self.key(f"drone_recording.{name}", index, at, value, "HUD", 'CONSTANT')

    def hud_at(self, at):
        """What the HUD showed at frame at (HUD_CHANNELS names to tuples), or None for a recording without it"""
        if ("drone_recording.hud_axes", 0) not in self.fcurves:
            return None
        values = {}
        for name, size in HUD_CHANNELS:
            fcurves = [self.fcurves.get((f"drone_recording.{name}", index)) for index in range(size)]
            values[name] = tuple(fcurve.evaluate(at) if fcurve is not None else 0.0 for fcurve in fcurves)
        return values

    def key_transform(self, at):
        """Key the Drone object's transform at frame at"""
        obj = self.drone_obj
        rotation_path = transform_path(obj)
        rotation = getattr(obj, rotation_path)
        if rotation_path == "rotation_euler":
            rotation = rotation.copy()
            if self.euler is not None:
                rotation.make_compatible(self.euler)
            self.euler = rotation
        for data_path, values in (("location", obj.location), (rotation_path, rotation), ("scale", obj.scale)):
            for index, value in enumerate(values):
                self.key(data_path, index, at, value, "Object Transforms", 'LINEAR')

    def finish(self):
        """Recalculate the keyframes' handles, which inserting FAST skipped"""
        for fcurve in self.fcurves.values():
            fcurve.update()

    def end_frame(self):
        """The last frame with recorded inputs"""
        fcurve = self.fcurves.get(("drone_recording.thrust", 0))
        if fcurve is None or not len(fcurve.keyframe_points):
            return self.start_frame
        return int(round(fcurve.keyframe_points[-1].co.x))

    def inputs_at(self, at):
        """The inputs keyed at frame at, as frame_inputs() has them"""
        inputs = {}
        for name, size in CHANNELS:
            values = []
            for index in range(size):
                fcurve = self.fcurves.get((f"drone_recording.{name}", index))
                values.append(fcurve.evaluate(at) if fcurve is not None else 0.0)
            inputs[name] = tuple(values) if size > 1 else values[0]
        return inputs


def replay(scene):
    """Show the Drone where its recording has it at the scene's current frame, while not flying: from the recording's
    start, every recorded frame stepped as the flight stepped it (advance()), held before its start and after its end.
    Playing forward, one step a frame; any jump replays from the start (a minute's recording in about 50 ms), so the
    Drone shows edits to the keyframes from the next jump on. The Drone object's transform plays from its keyframes."""
    global _replayed
    if core is None or state is not None:
        return
    references = getattr(scene, "references", None)
    drone_obj = references.drone_obj if references else None
    take = Recording.of(drone_obj) if drone_obj else None
    if take is None:
        return
    at = min(max(scene.frame_current, take.start_frame), take.end_frame())
    key = (take.action.as_pointer(), take.start_frame)
    if _replayed is not None and _replayed[0] == key and _replayed[1] == at:
        drone_state = _replayed[2]
    elif _replayed is not None and _replayed[0] == key and _replayed[1] == at - 1:
        drone_state, _ = advance(_replayed[2], take.inputs_at(at), take.dt, take.recenter_distance)
    else:
        drone_state = take.start_state()
        for each in range(take.start_frame + 1, at + 1):
            drone_state, _ = advance(drone_state, take.inputs_at(each), take.dt, take.recenter_distance)
    _replayed = (key, at, drone_state)
    write(drone_obj, drone_state)


@bpy.app.handlers.persistent
def replay_handler(scene, depsgraph=None):
    replay(scene)


def register():
    bpy.utils.register_class(DronePhysics)
    bpy.utils.register_class(DroneRecording)
    bpy.types.Scene.drone_physics = bpy.props.PointerProperty(type=DronePhysics)
    bpy.types.Object.drone_recording = bpy.props.PointerProperty(type=DroneRecording)
    if replay_handler not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(replay_handler)


def unregister():
    if replay_handler in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(replay_handler)
    del bpy.types.Object.drone_recording
    del bpy.types.Scene.drone_physics
    bpy.utils.unregister_class(DroneRecording)
    bpy.utils.unregister_class(DronePhysics)
