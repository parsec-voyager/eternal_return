"""Flying the drone: everything the Fly operator (operators.ER_OT_Fly) does each tick and at the start and end of a
flight, as the FlightFeatures mixin the operator is made from, and the flight's state, as module globals (the Drone, its
camera, the selected object, the controls as the last tick read them). The operator itself only runs the modal timer,
reads the controller and plays the sound; the flight harness (tools/flight_sim.py) runs these methods without it."""
import bpy
import sdl2
from . import scaling_settings
from . import background
from . import endless
from . import fractal
from . import gravity
from . import horizon
from . import hud
from . import hud_render
from . import mandelbrot
from . import odometer
from . import parts
from . import physics
from . import sphere
from . import surface
from . import velocity
from . import controls
from . import gamepad
import mathutils
from mathutils import Matrix
from mathutils.bvhtree import BVHTree
import bmesh
import contextlib
import time
import math
import json
import traceback
import numpy as np

# Longest tick, in seconds, the modal applies. The timer runs every 0.01 s.
MAX_MODAL_INTERVAL = 0.1

# How far the Drone_Camera can get from the world origin, in multiples of its clip_start, before Keep Drone at Origin
# moves the scene back under it. Like physics.RECENTER_DISTANCE, but for float32 rounding in world space. In multiples
# of clip_start, not of the drone's scale, since nothing closer than clip_start is drawn, and the HUD sits right
# there. At 500 times the scale, with a default clip start of 0.01 (50,000 times clip_start), float32 put the HUD
# up to 0.8 px off on a 1920 px wide frame, and it shook. At 5,000 times clip_start it's at most 0.1 px.
ORIGIN_DISTANCE = 5000.0

# Auto scale keeps the selected object (or its surface) at least a default clip start away, shrinking the drone if need
# be, and with Bounded, the scale control grows it no further: this much more than that (see growth_room())
GROWTH_BUFFER = 0.01

# Tidal lock skips a tick that would turn the drone further than this, in radians. Flight bends the direction to the
# selected object (or the point on it auto scale measures to) only a little each tick, but passing through it, or across
# the plane Split Axes measures to on one axis, flips it, and turning the drone around for that would be a jump, not an
# orbit.
TIDAL_LOCK_MAX_ANGLE = math.pi / 2

# Scaling config keys, as paths into the config JSON, for the control_settings property each one sets
SCALING_CONFIG_PATHS = {
    "loop": ("loop",),
    "loop_factor": ("loop_factor",),
    "wrap": ("wrap",),
    "wrap_split_axes": ("wrap_config", "wrap_split_axes"),
    "wrap_range": ("wrap_config", "wrap_range"),
    "wrap_split_axes_x": ("wrap_config", "wrap_split_axes_x"),
    "wrap_range_x": ("wrap_config", "wrap_range_x"),
    "wrap_split_axes_y": ("wrap_config", "wrap_split_axes_y"),
    "wrap_range_y": ("wrap_config", "wrap_range_y"),
    "wrap_split_axes_z": ("wrap_config", "wrap_split_axes_z"),
    "wrap_range_z": ("wrap_config", "wrap_range_z"),
    "wrap_rotation": ("wrap_config", "wrap_rotation"),
    "wrap_pivot": ("wrap_config", "wrap_pivot"),
    "auto_scale": ("auto_scale",),
    "auto_scale_mesh": ("auto_scale_config", "auto_scale_mesh"),
    "auto_scale_split_axes": ("auto_scale_config", "auto_scale_split_axes"),
    "auto_scale_x": ("auto_scale_config", "auto_scale_x"),
    "auto_scale_y": ("auto_scale_config", "auto_scale_y"),
    "auto_scale_z": ("auto_scale_config", "auto_scale_z"),
    "auto_scale_rotation": ("auto_scale_config", "auto_scale_rotation"),
    "tidal_lock": ("tidal_lock",),
    "tidal_lock_mesh": ("tidal_lock_config", "tidal_lock_mesh"),
    "tidal_lock_split_axes": ("tidal_lock_config", "tidal_lock_split_axes"),
    "tidal_lock_x": ("tidal_lock_config", "tidal_lock_x"),
    "tidal_lock_y": ("tidal_lock_config", "tidal_lock_y"),
    "tidal_lock_z": ("tidal_lock_config", "tidal_lock_z"),
    "tidal_lock_rotation": ("tidal_lock_config", "tidal_lock_rotation"),
    "horizon": ("horizon",),
    "horizon_mesh": ("horizon_config", "horizon_mesh"),
    "gravity_mesh": ("gravity_config", "gravity_mesh"),
    "min_scale": ("min_scale",),
    "max_scale": ("max_scale",),
}

# Scaling config keys written in degrees in the JSON, like the UI shows them, but stored in radians
SCALING_CONFIG_DEGREES = {"wrap_rotation", "auto_scale_rotation", "tidal_lock_rotation"}

# Scaling config keys that can be null for no value, mapped to the property that says whether they have one
SCALING_CONFIG_NULLABLE = {"wrap_pivot": "wrap_custom_pivot"}

dict_controls = {}
record_start_frame = 0
record_end_frame = 250

drone_obj = None
drone_camera = None
clip_end_obj = None
clip_end_ratio = None

selected_obj = None
selected_obj_bool = False

# world = bpy.context.scene.world

min_scale = scaling_settings.DEFAULT_MIN_SCALE

# The sticks the drone flies by
STICKS = ("throttle", "yaw", "pitch", "roll")

def rig_gravity():
    """The drone's own Gravity (the scene's physics, physics.DronePhysics), which pulls along the Drone object's Z
    axis, wherever tidal lock and orbit have turned it"""
    return bpy.context.scene.drone_physics.gravity

def anchored_map(obj):
    """sphere.py or surface.py if obj is one of their maps, else None: maps anchored at a point of a lattice drawn on a
    surface (a sphere's, or any mesh's faces), which loop about the anchor's point by powers of 2, keep the drone near
    it, and are measured to their surface, mesh mode or not"""
    if not isinstance(obj, bpy.types.Object):
        return None
    if sphere.is_sphere(obj):
        return sphere
    if surface.is_surface(obj):
        return surface
    return None


def loops_go_deeper(obj, scene):
    """Whether a loop about obj goes deeper into the same world, which only a self-similar one does: an endless map
    (fractals too), a sphere or surface map (substitution floors too), or the scene's Mandelbrot floor, which loops
    with the drone whatever is selected. Their loops relabel the world's meters (see odometer.py), so the HUD's Scale
    keeps going down as the drone dives. A loop about any other object (a plain mesh, vr_drone_simple's, 2026-11-23)
    only makes the drone the loop factor bigger next to the same mesh: there the drone's true scale is its Blender
    scale, and flying back out it reads its normal size again, not the loops' 64^-n."""
    if mandelbrot.floor_object(scene) is not None:
        return True
    if not isinstance(obj, bpy.types.Object):
        return False
    return anchored_map(obj) is not None or endless.is_endless(obj)


def scaling_config_text(text):
    """A scaling config Text's JSON, from its file when it's external and the file changed since Blender last read or
    wrote it (Text.is_modified): a .blend keeps its own copy of an external Text and doesn't reread the file on load,
    so without this an edited config file (like a map folder's surface_config) would stay unused until its Text was
    reloaded by hand in the Text Editor, map by map. Changes made to the Text inside Blender are kept as long as the
    file doesn't change after them (else the file wins, the conflict the Text Editor shows). Blender tells by the file's
    time, which a Text made in Blender and then given a filepath never recorded: its file wins until the Text is saved
    or reloaded in the Text Editor. The Text itself is left
    as it is: bpy.ops.text.reload crashes Blender without a Text Editor to run in. Whether it has a file is its
    filepath, not is_in_memory, which stays True for a Text made in Blender and then given a filepath (the map builds')."""
    if text.filepath and text.is_modified:
        try:
            with open(bpy.path.abspath(text.filepath, library=text.library), encoding="utf-8") as file:
                return file.read()
        except OSError:
            pass
    return text.as_string()


def get_camera_matrix():
    """drone_camera's world matrix as of now. Its own matrix_world is stale until the depsgraph updates after the drone
    moves, and updating it in the tick (bpy.context.view_layer.update()) cost 1 to 2 ms each time, about 3 of the
    tick's 4 ms: the Drone's and the HUD's geometry nodes read their own transforms, so every update evaluated them
    again, on top of the frame's own evaluation. The camera is parented to the vertices the drone's physics move, which
    only move when a frame is simulated, so in between it moves with drone_obj as one piece: by the drone's move since
    the depsgraph last evaluated them, and by its own location, rotation and scale since.

    Worked out in float64 from the drone's location, rotation and scale (matrix_basis), which the depsgraph combines,
    rather than from the matrix_world last set, which they were rounded from. It then agrees with the depsgraph to a
    few float32 steps of the camera's position, as close as the depsgraph's own rounding (from matrix_world, it was up
    to 32 steps off). The matrices are compared exactly: mathutils takes matrices within 1e-7 of each other for equal,
    and at a small scale that's a real move."""
    depsgraph = bpy.context.view_layer.depsgraph
    drone_eval, camera_eval = drone_obj.evaluated_get(depsgraph), drone_camera.evaluated_get(depsgraph)
    drone_matrix = np.array(drone_obj.matrix_basis if drone_obj.parent is None else drone_obj.matrix_world, dtype=np.float64)
    drone_matrix_eval = np.array(drone_eval.matrix_world, dtype=np.float64)
    camera_basis = np.array(drone_camera.matrix_basis, dtype=np.float64)
    camera_basis_eval = np.array(camera_eval.matrix_basis, dtype=np.float64)
    if np.array_equal(drone_matrix, drone_matrix_eval) and np.array_equal(camera_basis, camera_basis_eval):
        return camera_eval.matrix_world.copy()

    # A tick asks a dozen times, most of them between two moves of the drone: the same matrices, the same answer
    global camera_matrix_kept
    camera_world_eval = np.array(camera_eval.matrix_world, dtype=np.float64)
    given = (drone_matrix, drone_matrix_eval, camera_basis, camera_basis_eval, camera_world_eval)
    if camera_matrix_kept is not None and all(np.array_equal(a, b) for a, b in zip(camera_matrix_kept[0], given)):
        return camera_matrix_kept[1].copy()
    drone_move = drone_matrix @ np.linalg.inv(drone_matrix_eval)
    camera_move = np.linalg.inv(camera_basis_eval) @ camera_basis
    matrix = Matrix((drone_move @ camera_world_eval @ camera_move).tolist())
    camera_matrix_kept = (given, matrix)
    return matrix.copy()


# get_camera_matrix()'s last answer: (the matrices it was worked out from, the camera's matrix)
camera_matrix_kept = None

def get_world_objects(scene):
    """The top-level objects Keep Drone at Origin moves through delta_location: all but the Drone and the recording
    camera, which follow the Drone_Camera."""
    record_obj = scene.objects.get(scene.flight_options.record_obj)
    return [obj for obj in scene.objects if not obj.parent and obj not in (drone_obj, record_obj)]

def move_world_origin(scene, location):
    """Move the scene by -location, so location ends up at the world origin, and add location to world_offset. Objects
    move through delta_location, which Blender adds on top of location, so their own transforms and animation are left
    alone. The Drone and the recording camera move through location instead, since setting their matrix_world (as
    flying and recording do) would cancel out a delta."""
    drone_rig = scene.drone_rig
    old_offset = drone_rig.world_offset[:]
    drone_rig.world_offset = [offset + loc for offset, loc in zip(old_offset, location)]

    # world_offset is float32, so move everything by the amount it actually changed after rounding, or the Drone would
    # hop relative to the objects around it. Python floats, since mathutils vectors are float32 too.
    shift = [new - old for new, old in zip(drone_rig.world_offset, old_offset)]

    for obj in get_world_objects(scene):
        obj.delta_location = [delta - s for delta, s in zip(obj.delta_location, shift)]

    record_obj = scene.objects.get(scene.flight_options.record_obj)
    for obj in (drone_obj, record_obj):
        if obj and not obj.parent:
            obj.location = [loc - s for loc, s in zip(obj.location, shift)]

@background.foreground
def fly_frame(scene):
    """Fly the drone into the new frame, before it's evaluated: one step of the core's physics (physics.py), recorded
    with Record on. Orbit turns the drone itself, so its sticks don't turn it as well."""
    orbiting = dict_controls.get("orbit", 0.0) > 0.0 and selected_obj_bool
    if drone_obj:
        commands = {key: 0.0 if orbiting and key != "throttle" else dict_controls.get(key, 0.0) for key in STICKS}
        physics.step_frame(scene, drone_obj, commands, gravity.pull(), hud.recorded())


def start_physics(context, report=None):
    """Get the drone's physics ready for a flight, at its start (execute(), and the flight harness): the drone goes on
    from where the last flight or replay left it (physics.py), recorded from there with Record on. Returns False, after
    reporting it, when the core isn't installed for this platform."""
    if not physics.available():
        if report is not None:
            report({'ERROR'}, "Eternal Return core not installed for this platform: can't fly")
        return False
    scene = context.scene
    gravity.set_pull((0.0, 0.0, rig_gravity()))
    physics.take_over(drone_obj, scene, context.evaluated_depsgraph_get(), scene.flight_options.record)
    return True


class FlightFeatures:
    """What the Fly operator does while flying, as methods of the operator (operators.ER_OT_Fly)"""
    _gamepad = None

    _active_object = None
    _parent = None
    _parent_scale = None
    _click_bool = True
    _max_scale = scaling_settings.DEFAULT_MAX_SCALE
    _max_drag = 33_000_000
    _modal_interval = None
    _last_time = None
    _selected_obj_diff_vector = None
    _camera_moved = False
    _world_bvh = None
    _auto_scale_mesh = False
    # The selected object's diff vector when auto scale last scaled the drone, which it measures distance changes from.
    # Ticks it skips don't move it, and moves that aren't flight, like loops and zoom, scale it.
    _auto_scale_reference = None
    # Scaling auto scale still owes the drone, because min_scale or max_scale held it back. Applied with the next auto
    # scale or scaling by hand, so the drone's scale stays proportional to its distance.
    _auto_scale_remainder = 1.0
    # min_scale and max_scale as of the last tick, to notice when a scaling config or the panel changes them
    _scale_range = None
    _origin_moved_frame = None
    # Where stop_at_surface() checks the next tick's flight from, recorded at the end of each tick while auto scale is
    # on: (selected object, its matrix_world, drone_camera's location), the last two as float64 arrays
    _surface_check_start = None
    # Where tidal_lock() measures the next tick's turn from, recorded at the end of each tick while tidal lock is on:
    # (what it was measured to, see get_tidal_lock_key(), the direction from the drone camera to it)
    _tidal_lock_start = None
    # The flight's path, jumps left out, for the HUD's readouts (odometer.py), and the scene's Earths, which know their
    # own meters (sphere.meters_log2())
    _odometer = None
    _earths = ()

    @background.foreground
    def tick(self, context):
        """One step of flight, every 0.01 s on the modal timer: the controller read (the operator's read_controls()),
        the drone flown (fly_tick()), and the propellers' sound updated."""
        dict_controls_raw, controller_axes = self.read_controls()
        self.fly_tick(context, dict_controls_raw, controller_axes)
        self.update_sound(context)

    def fly_tick(self, context, dict_controls_raw, controller_axes):
        """Everything a tick does with the controls (dict_controls_raw, each control's stick position as mapped from
        the controller's axes, controller_axes): the controls converted to what they do, the scaling tools, the
        selection, and the HUD"""
        global dict_controls
        global selected_obj
        global selected_obj_bool

        # The four flight axes by the core (physics.sticks_to_commands(), which also gives the physics the flight mode)
        dict_controls = controls.convert(dict_controls_raw, context.scene)

        global min_scale
        control_settings = bpy.context.scene.control_settings

        min_scale = max(scaling_settings.DEFAULT_MIN_SCALE, control_settings.min_scale)
        min_scale = min(min_scale, scaling_settings.DEFAULT_MAX_SCALE)

        self._max_scale = min(scaling_settings.DEFAULT_MAX_SCALE, control_settings.max_scale)
        self._max_scale = max(self._max_scale, scaling_settings.DEFAULT_MIN_SCALE)

        max_loop_factor = self._max_scale / min_scale
        if control_settings.loop_factor > max_loop_factor:
            control_settings.loop_factor = max_loop_factor

        # Capped so a long gap between ticks (the computer sleeping, macOS throttling Blender in the background) isn't
        # applied as one huge step of scaling, zooming, braking or orbiting
        now = time.perf_counter()
        self._modal_interval = min(now - self._last_time, MAX_MODAL_INTERVAL)
        self._last_time = now

        self.brake()

        if drone_camera:
            # First, so the only move since the last tick it can see is flight
            self.stop_at_surface()

            # Also before anything else moves the drone: loops, zoom, wrap and orbit aren't flight, and must not turn it
            self.tidal_lock()

            self.parent()

            self.drone_ui(context, controller_axes)

            self.fit_scale_range()

            self.click(context)

            new_selected_obj = self.get_selected_object()
            selection_changed = new_selected_obj != selected_obj
            selected_obj = new_selected_obj
            selected_obj_bool = selected_obj and not self.is_drone_part(selected_obj)

            # Wrapped before measuring, too, not only after this tick's moves. The drone flew since the last tick, and a
            # large, fast one can leave the repeating part of the world in that time. Measured from out there, auto
            # scale scaled it up for being far from the selected object, so it flew further out the next time, up to
            # max_scale.
            self.wrap()
            self.recenter_endless()
            # The Earth's valley level around the drone, which auto scale measures its height over
            for obj in self.sphere_maps():
                if sphere.is_sphere(obj):
                    sphere.follow_base(obj, get_camera_matrix().translation)

            lower_bound = self.get_lower_bound()
            selected_obj_diff_vector = self.get_selected_object_diff_vector(selected_obj)

            auto_scale = bpy.context.scene.control_settings.auto_scale
            if auto_scale:
                self.auto_scale(selected_obj_diff_vector, lower_bound, selection_changed)
            else:
                # Turned back on, auto scale starts over from the drone's distance then. A kept reference went stale
                # while it was off (loops still rescaled it, flight didn't move it), and the first tick back scaled the
                # drone by the whole difference at once: by 1e-15 and less after some looping, which zeroed its scale.
                self._auto_scale_remainder = 1.0
                self._auto_scale_reference = None

            self._selected_obj_diff_vector = selected_obj_diff_vector

            bounded = bpy.context.scene.control_settings.bounded
            if dict_controls["scale"] != 1.0:
                drone_log_scale = dict_controls["scale"]**self._modal_interval
                scale_camera = bpy.context.scene.control_settings.scale_camera
                scale_lens_clip = bpy.context.scene.control_settings.scale_lens_clip
                bounded_bool = bounded and selected_obj_bool and (auto_scale or dict_controls["zoom"] != 1.0)
                if bounded_bool and auto_scale and drone_log_scale > 1.0:
                    drone_log_scale = min(drone_log_scale, self.growth_room())

                # With auto scale, scaling by hand changes the drone's scale for its distance by drone_log_scale, so
                # it applies to what auto scale still owes, too. Starting over from the drone's scale instead, held at
                # min_scale against a surface, made its scale for its distance huge, and it blew up in scale flying
                # away (even after scaling down).
                self._auto_scale_remainder = self.owed(self.scale(drone_log_scale * self._auto_scale_remainder,
                                                                  drone_camera, scale_camera, scale_lens_clip,
                                                                  bounded_bool))

            if dict_controls["zoom"] != 1.0:
                drone_log_scale = dict_controls["zoom"]**self._modal_interval
                scale_camera_zoom = bpy.context.scene.control_settings.scale_camera_zoom
                scale_lens_clip_zoom = bpy.context.scene.control_settings.scale_lens_clip_zoom

                self.scale(drone_log_scale, selected_obj, scale_camera_zoom, scale_lens_clip_zoom, bounded=False)
                self._camera_moved = True

            if self._camera_moved:
                self.remeasure_selected_obj()

            self.orbit()

            if dict_controls["camera"] != 1.0:
                camera_log_scale = dict_controls["camera"]**self._modal_interval
                scale_lens_clip_with_camera = bpy.context.scene.control_settings.scale_lens_clip_with_camera

                self.scale_camera(camera_log_scale, scale_lens_clip_with_camera)

            # Measured again, since auto scale, the scale control, zoom or the camera control may have scaled the
            # camera
            self.lens_clip(bounded, self.get_lower_bound())

            self.wrap()
            self.recenter_endless()

            self.move_and_record()

            self.record_surface_check_start()
            self.record_tidal_lock_start()
            self.horizon_ui()
            velocity.set_shown(control_settings.velocity)
            self.gravity_pull()
            self.readouts_ui()

            # Last, with the camera where the tick left it
            camera_matrix = get_camera_matrix()
            self.refresh_fractals(camera_matrix.translation, camera_matrix)
            self.settle_parts(camera_matrix)


    def fit_scale_range(self):
        """Bring the drone within min_scale and max_scale as soon as they change, like when a clicked object's scaling
        config sets a range the drone's scale is outside of: by looping if loop is on, so the view stays the same,
        otherwise by scaling in place. Otherwise it would only happen at the drone's next scaling."""
        scale_range = (min_scale, self._max_scale)
        if scale_range == self._scale_range:
            return
        self._scale_range = scale_range

        # A scale clamped into the new range is the drone's new size for its distance, not scaling auto scale still
        # owes it
        if self.scale(1.0, drone_camera, scale_camera=True, scale_lens_clip=True, bounded=False) != 1.0:
            self._auto_scale_remainder = 1.0

        # Measured now, so this tick's auto scale doesn't read a loop as flight distance
        if self._camera_moved:
            self.remeasure_selected_obj()

    def remeasure_selected_obj(self, rescale_reference=True):
        """Re-measure the distance to the selected object after loop, zoom, wrap or Keep Drone at Origin moved
        drone_camera, and scale auto scale's reference by how much the move changed it. Otherwise the next auto_scale
        would read that move as flight distance and rescale the drone by it.

        Wrap passes rescale_reference=False: in the repeating world it's for, the distance after the jump is the real
        one. Before it, a drone that flew out of the repeating part measures a meaningless distance, and scaling the
        reference by that threw auto scale off."""
        self._camera_moved = False

        # Not for the camera (see get_camera_matrix()): the moves this follows can move the selected object too, which
        # is stale until the depsgraph updates. Keep Drone at Origin moves every object, and a loop or recenter moves an
        # endless map's anchor, which its drivers read.
        bpy.context.view_layer.update()
        old_diff_vector = self._selected_obj_diff_vector
        self._selected_obj_diff_vector = self.get_selected_object_diff_vector(selected_obj)

        if not rescale_reference or self._auto_scale_reference is None or old_diff_vector is None:
            return

        old_distance = self.get_auto_scale_distance(old_diff_vector)
        if old_distance > 0.0:
            new_distance = self.get_auto_scale_distance(self._selected_obj_diff_vector)
            self._auto_scale_reference = self._auto_scale_reference * (new_distance / old_distance)
        else:
            self._auto_scale_reference = self._selected_obj_diff_vector

    def brake(self):
        """Brake by thickening the air the drone flies in (physics.brake_density), more every tick, for as long as the
        throttle is at zero with Auto Brake on, or the brake stick is held; the scene's own Air Density once it lets
        go"""
        control_settings = bpy.context.scene.control_settings
        factor = None
        if dict_controls["throttle"] == 0.0:
            if control_settings.auto_brake:
                factor = control_settings.brake_factor
            elif dict_controls["brake"] > 1.0:
                factor = dict_controls["brake"]
        if factor is None:
            physics.brake_density = None
            return
        density = physics.air_density(bpy.context.scene)
        if density < self._max_drag:
            density *= (factor * 100)**self._modal_interval
        physics.brake_density = density

    def parent(self):
        if drone_obj.parent:
            current_scale = drone_obj.parent.matrix_world.to_scale().copy()

            if self._parent == drone_obj.parent and self._parent_scale != current_scale:
                scale_change = current_scale.x / self._parent_scale.x

                drone_camera.data.clip_start *= scale_change
                drone_camera.data.clip_end *= scale_change
                self.scale_lens_clip_objects()

            self._parent = drone_obj.parent
            self._parent_scale = current_scale

    def drone_ui(self, context, controller_axes):
        """Read the sticks' positions for the HUD's gimbals (hud.py), drawn with the next redraw"""
        hud.set_axes(gamepad.read_gimbal_inputs(controller_axes, self._gamepad))

    def click(self, context):
        if dict_controls["click"] > 0.0:
            if not self._click_bool:
                self._click_bool = True

                if bpy.context.active_object:
                    bpy.ops.object.mode_set(mode='OBJECT')
                self.click_center_of_viewport(context)
                # A Mandelbrot floor's plane isn't drawn (the floor is, in its place), so the click can't hit it, and a
                # sphere map is drawn on no mesh at all
                mandelbrot.select_floor_if_looking_at_it(bpy.context.scene, drone_camera)
                sphere.select_if_looking_at_it(bpy.context.scene, drone_camera)

                selected_obj = self.get_selected_object()

                # An object can name another to select in its place with a "scaling_target" custom property. Auto
                # scale, loop and wrap measure from the selected object's origin, so parts that move, like patches of a
                # map that follow the drone, select a still object at the map's center, which holds the config.
                target = selected_obj.get("scaling_target")
                if isinstance(target, bpy.types.Object) and target.name in bpy.context.view_layer.objects:
                    target.select_set(True)
                    bpy.context.view_layer.objects.active = target
                    selected_obj = target

                scaling_config = selected_obj.get("scaling_config")
                if scaling_config:
                    self.set_scaling_config(scaling_config, selected_obj.name)

        elif self._click_bool:
            self._click_bool = False

            auto_deselect = bpy.context.scene.control_settings.auto_deselect
            if auto_deselect:
                self.select_nothing()

                default_config = bpy.data.texts.get("default_config")
                if default_config:
                    self.set_scaling_config(default_config, default_config.name)

    def select_nothing(self):
        """Deselect everything, the active object too. get_selected_object() then falls back to the drone camera, a
        drone part, which everything that follows a selected object takes as nothing selected: its origin is the
        camera's, so zoom and the lens clip measure from the same point. With no object selected, none is drawn
        selected or shown in the Properties editor."""
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        bpy.context.view_layer.objects.active = None

    def click_center_of_viewport(self, context):
        """Select what's under the middle of the view, never a part of the drone. view3d.select picks the nearest
        selectable object within a few pixels, so the camera, or a Drone left unhidden, could win the click. They're
        made unselectable just for the pick, and kept selectable otherwise, since the user may select one by hand. A
        click on nothing selects nothing (select_nothing())."""
        parts = [obj for obj in (drone_obj, drone_camera, clip_end_obj) if obj]
        selectable = [obj.hide_select for obj in parts]
        for obj in parts:
            if obj.name in bpy.context.view_layer.objects:
                obj.select_set(False)
            obj.hide_select = True
        try:
            self.select_center(context)
        finally:
            for obj, hide_select in zip(parts, selectable):
                obj.hide_select = hide_select
        if not bpy.context.selected_objects:
            self.select_nothing()

    def select_center(self, context):
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                region = next((r for r in area.regions if r.type == 'WINDOW'), None)

                center_x = int(region.width / 2)
                center_y = int(region.height / 2)

                override = context.copy()
                override['area'] = area
                override['region'] = region

                print(f"Viewport center: ({center_x}, {center_y})")

                with context.temp_override(**override):
                    bpy.ops.view3d.select(location=(center_x, center_y), deselect_all=True)

                break

    def get_selected_object(self):
        selected_obj_name = bpy.context.scene.control_settings.selected_obj
        selected_obj = bpy.context.scene.objects.get(selected_obj_name)

        if not selected_obj:
            selected_obj = bpy.context.active_object
        if not selected_obj:
            selected_obj = drone_camera

        return selected_obj

    def set_scaling_config(self, config, source_name):
        """Apply a scaling config stored as a Text datablock or a JSON string. Only the keys present are applied, so a
        config can override just a few settings. Invalid JSON or values are reported and skipped."""
        try:
            config_json = json.loads(scaling_config_text(config) if isinstance(config, bpy.types.Text) else config)
        except (TypeError, ValueError) as error:
            self.report({'WARNING'}, f"Scaling config on {source_name} isn't valid JSON: {error}")
            return

        if not isinstance(config_json, dict):
            self.report({'WARNING'}, f"Scaling config on {source_name} isn't a JSON object")
            return

        # Tells a missing key, which leaves the setting alone, apart from null
        missing = object()

        params = bpy.context.scene.control_settings
        for prop, path in SCALING_CONFIG_PATHS.items():
            value = config_json
            for key in path:
                value = value.get(key, missing) if isinstance(value, dict) else missing

            if value is missing:
                continue

            if value is None:
                if prop in SCALING_CONFIG_NULLABLE:
                    setattr(params, SCALING_CONFIG_NULLABLE[prop], False)
                continue

            try:
                if prop in SCALING_CONFIG_DEGREES:
                    value = [math.radians(angle) for angle in value]
                setattr(params, prop, value)
                if prop in SCALING_CONFIG_NULLABLE:
                    setattr(params, SCALING_CONFIG_NULLABLE[prop], True)
            except (TypeError, ValueError) as error:
                self.report({'WARNING'}, f"Scaling config on {source_name}: ignored {'.'.join(path)} ({error})")

    def is_drone_part(self, obj):
        return obj in [drone_obj, drone_camera, clip_end_obj]

    def get_selected_object_diff_vector(self, selected_obj, mesh=None):
        """From the drone camera to the selected object's origin, or with mesh, to the closest point on its surface.
        mesh defaults to auto scale's Mesh setting."""
        camera_location = get_camera_matrix().translation

        if mesh is None:
            mesh = bpy.context.scene.control_settings.auto_scale_mesh
        anchored = anchored_map(selected_obj)
        if anchored is not None:
            # A sphere map has no origin to measure to (its object's origin is the anchor's point, which moves, and its
            # center can be any distance away), mesh mode or not: the nearest point of its surface, straight below,
            # which is also the way to its center for tidal lock and the horizon. A surface map's likewise (its mesh is
            # 2^n times its size).
            obj_location = camera_location + mathutils.Vector(anchored.surface_vector(selected_obj, camera_location))
        elif mesh and isinstance(selected_obj, bpy.types.Object) and fractal.is_fractal(selected_obj):
            # An endless fractal has no mesh of its own: the closest point on the shapes it draws around the drone
            obj_location = fractal.closest_point(selected_obj, camera_location)
        elif mesh and not self.is_drone_part(selected_obj):
            obj_location = self.get_closest_point_on_mesh(selected_obj, camera_location)
        elif isinstance(selected_obj, mathutils.Vector):
            obj_location = selected_obj
        elif selected_obj != drone_obj and self.is_drone_part(selected_obj):
            # The drone's other parts ride on its camera, and their matrix_world is as stale as drone_camera's, so
            # they're measured from where the camera was then (see get_camera_matrix())
            return selected_obj.matrix_world.translation - drone_camera.matrix_world.translation
        else:
            obj_location = selected_obj.matrix_world.translation

        diff_vector = obj_location - camera_location

        return diff_vector

    def get_closest_point_on_mesh(self, selected_obj, camera_location):
        """Return the world-space point on selected_obj's surface closest to camera_location.

        The point can lie anywhere on a face or along an edge, not only on a vertex. Uses the evaluated mesh (modifiers
        applied). Non-mesh objects and meshes without vertices return their origin.
        """
        if isinstance(selected_obj, mathutils.Vector):
            return selected_obj.copy()

        matrix_world = selected_obj.matrix_world
        origin = matrix_world.translation.copy()
        if selected_obj.type != 'MESH':
            return origin

        obj_eval = selected_obj.evaluated_get(bpy.context.view_layer.depsgraph)
        mesh = obj_eval.data
        if not mesh.vertices:
            return origin

        linear = matrix_world.to_3x3()
        if not mesh.polygons:
            return self.get_closest_point_on_edges(mesh, linear, origin, camera_location)

        # Local space distances only match world space distances when the scale is uniform (and not sheared)
        gram = linear.transposed() @ linear
        uniform_scale = all(abs(gram[i][j] - (gram[0][0] if i == j else 0.0)) <= 1e-5 * gram[0][0] for i in range(3) for j in range(3))

        if uniform_scale:
            # Blender caches a BVH tree on the evaluated mesh, so this stays fast on dense meshes
            local_location = matrix_world.inverted_safe() @ camera_location
            _found, local_point, _normal, _index = obj_eval.closest_point_on_mesh(local_location)
            return matrix_world @ local_point

        closest_point = self.get_world_bvh(selected_obj, mesh, linear).find_nearest(camera_location - origin)[0]
        return origin + closest_point

    def get_closest_point_on_edges(self, mesh, linear, origin, camera_location):
        """Closest point on any edge or vertex of a mesh that has no faces."""
        coords = self.get_mesh_coords(mesh, linear) + np.array(origin)
        camera = np.array(camera_location)

        edge_verts = np.empty(len(mesh.edges) * 2, dtype=np.int32)
        mesh.edges.foreach_get("vertices", edge_verts)
        edge_start = coords[edge_verts[0::2]]
        edge_vector = coords[edge_verts[1::2]] - edge_start
        edge_length_sq = (edge_vector * edge_vector).sum(axis=1)
        edge_t = ((camera - edge_start) * edge_vector).sum(axis=1) / np.where(edge_length_sq > 0.0, edge_length_sq, 1.0)
        edge_points = edge_start + edge_vector * np.clip(edge_t, 0.0, 1.0)[:, None]

        candidates = np.vstack((coords, edge_points))
        closest = candidates[np.argmin(((candidates - camera) ** 2).sum(axis=1))]

        return mathutils.Vector(closest)

    def get_world_bvh(self, selected_obj, mesh, linear):
        """BVH tree of the mesh's faces in world rotation and scale, but not translation, so keep_drone_at_origin moving
        the object doesn't force a rebuild. Rebuilt when the object, its rotation/scale or its vertex/face count changes."""
        key = (selected_obj.name, tuple(tuple(row) for row in linear), len(mesh.vertices), len(mesh.polygons))
        if not self._world_bvh or self._world_bvh[0] != key:
            triangles = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
            mesh.loop_triangles.foreach_get("vertices", triangles)
            coords = self.get_mesh_coords(mesh, linear)
            bvh = BVHTree.FromPolygons(coords.tolist(), triangles.reshape(-1, 3).tolist(), all_triangles=True)
            self._world_bvh = (key, bvh)

        return self._world_bvh[1]

    def get_mesh_coords(self, mesh, linear):
        """Vertex positions as an (n, 3) array, transformed by the 3x3 matrix linear."""
        coords = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", coords)

        return coords.reshape(-1, 3) @ np.array(linear).T

    def auto_scale(self, selected_obj_diff_vector, lower_bound, selection_changed):
        # Toggling mesh mode switches between origin and surface distances, so skip that tick instead of zooming by
        # their ratio
        auto_scale_mesh = bpy.context.scene.control_settings.auto_scale_mesh
        mesh_mode_changed = auto_scale_mesh != self._auto_scale_mesh
        self._auto_scale_mesh = auto_scale_mesh

        if mesh_mode_changed or selection_changed or not selected_obj_bool or self._auto_scale_reference is None:
            # A new distance for the drone's scale to stay proportional to
            self._auto_scale_reference = selected_obj_diff_vector
            self._auto_scale_remainder = 1.0
            return

        # Within lower_bound of the object (or its surface, in mesh mode), the drone is shrunk until it's that far, and
        # scales from there. It used to hold still there instead, the reference where it was, and with the drone grown
        # big for its distance by the scale control, flying at a surface stopped it dead: stop_at_surface() held it
        # back, nothing scaled it down, and it stayed stuck until turned away.
        within_lower_bound = selected_obj_diff_vector.length < lower_bound
        unmoved = self._auto_scale_reference == selected_obj_diff_vector
        if dict_controls["zoom"] != 1.0:
            return
        if within_lower_bound:
            if selected_obj_diff_vector.length > 0.0:
                factor = (1.0 - GROWTH_BUFFER) * selected_obj_diff_vector.length / lower_bound
                # Or by as much as its distance shrank since, if that's more: what keeping up would have done. A
                # sphere's 3D tiles' ground can come in meters higher under the drone (a finer tile), which lifts it
                # out to its clip_start over it; shrunk only to its bound there, it flew on 30 times too big for its
                # distance.
                old_distance = self.get_auto_scale_distance(self._auto_scale_reference)
                if old_distance > 0.0:
                    factor = min(factor, self.get_auto_scale_distance(selected_obj_diff_vector) / old_distance)
                self.scale(factor, drone_camera, scale_camera=True, scale_lens_clip=True, bounded=False)
                self._auto_scale_reference = selected_obj_diff_vector
            return
        if unmoved:
            return

        old_distance = self.get_auto_scale_distance(self._auto_scale_reference)
        new_distance = self.get_auto_scale_distance(selected_obj_diff_vector)

        # No ratio to or from a distance of 0, and a remainder of 0 could never scale the drone again
        if new_distance == 0.0:
            return
        if old_distance == 0.0:
            self._auto_scale_reference = selected_obj_diff_vector
            return

        # What min_scale or max_scale holds back is kept for later. Otherwise the drone would stop scaling at the limit
        # while its distance kept changing, then scale out of proportion once it turned back: flown into a surface at
        # min_scale, it blew up in scale as soon as it moved away from it again.
        held = self.scale(new_distance / old_distance * self._auto_scale_remainder, drone_camera,
                          scale_camera=True, scale_lens_clip=True, bounded=False)
        self._auto_scale_remainder = self.owed(held)
        self._auto_scale_reference = selected_obj_diff_vector

    def get_auto_scale_distance(self, diff_vector):
        """The distance auto scale keeps the drone's scale proportional to: diff_vector's length, or with Split Axes,
        its length along just the chosen axes."""
        return self.get_split_axes_vector(diff_vector, "auto_scale").length

    def get_split_axes_settings(self, prefix):
        """The Split Axes settings of auto scale or tidal lock (prefix "auto_scale" or "tidal_lock"): None when Split
        Axes is off, otherwise (which of X, Y and Z are on, the rotation of those axes)"""
        control_settings = bpy.context.scene.control_settings
        if not getattr(control_settings, f"{prefix}_split_axes"):
            return None
        axes = tuple(getattr(control_settings, f"{prefix}_{axis}") for axis in "xyz")
        return axes, tuple(getattr(control_settings, f"{prefix}_rotation"))

    def get_split_axes_vector(self, diff_vector, prefix):
        """diff_vector, or with Split Axes (auto scale's or tidal lock's, see get_split_axes_settings()), just its part
        along the chosen axes, in world space: from the drone camera to the closest point on the line (two axes) or
        plane (one axis) through the target they measure to."""
        settings = self.get_split_axes_settings(prefix)
        if settings is None:
            return diff_vector
        axes, rotation = settings

        # The X, Y and Z axes are the world axes turned by the rotation, so express the vector along them. A rotation's
        # inverse is its transpose.
        rotation = mathutils.Euler(rotation).to_matrix()
        axes_diff_vector = rotation.transposed() @ diff_vector

        return rotation @ mathutils.Vector([value if axis else 0.0 for value, axis in zip(axes_diff_vector, axes)])

    def tidal_lock(self):
        """Turn the drone with its flight around the selected object, so the object stays at the same angle in its
        view, like a tidally locked moon always shows its planet the same face. Tidal lock is to orbit what auto scale
        is to zoom: flying a tenth of the way around a planet turns the drone a tenth of a turn, so it keeps facing its
        surface, without the orbit switch.

        It has its own Mesh and Split Axes settings, like auto scale's, but apart from them: it measures to the object's
        origin, the closest point on its mesh in mesh mode (which follows the surface's curve, and doesn't turn over a
        flat floor), or with Split Axes on two axes, the line through the origin along the axis left out (around a
        cylinder, its axis). On one axis, the direction to a plane never turns, and on all three it's the origin again.
        Apart, so auto scale can measure to a bumpy planet's surface, to shrink near the ground, while tidal lock turns
        smoothly about its center, instead of with every bump. The drone turns about the drone camera, so the
        camera stays where it is, and the simulation turns with it: its velocity follows the surface around, and auto
        leveling levels to it. The sticks still turn the drone on top of that.

        Measured from the end of the last tick (record_tidal_lock_start()) before anything else this tick moves the
        drone, so the only move it sees is flight, and the object's own moves."""
        start = self._tidal_lock_start
        if start is None or not bpy.context.scene.control_settings.tidal_lock:
            return
        if not selected_obj_bool or start[0] != self.get_tidal_lock_key():
            return

        # In float64: a far orbit turns the drone by a millionth of a radian a tick
        old_direction = np.array(start[1], dtype=np.float64)
        new_direction = np.array(self.get_tidal_lock_direction(), dtype=np.float64)
        axis = np.cross(old_direction, new_direction)
        sin_length = float(np.linalg.norm(axis))
        if sin_length == 0.0:
            return
        angle = math.atan2(sin_length, float(old_direction @ new_direction))
        if angle > TIDAL_LOCK_MAX_ANGLE:
            return

        # The turn that takes the old direction to the new one, about the camera
        pivot = Matrix.Translation(get_camera_matrix().translation)
        rotation = Matrix.Rotation(angle, 4, mathutils.Vector(axis / sin_length))
        self.set_drone_matrix(pivot @ rotation @ pivot.inverted() @ drone_obj.matrix_world)

    def get_tidal_lock_direction(self):
        """From the drone camera to the point, line or surface tidal lock keeps at the same angle (see tidal_lock())"""
        mesh = bpy.context.scene.control_settings.tidal_lock_mesh
        return self.get_split_axes_vector(self.get_selected_object_direction(selected_obj, mesh), "tidal_lock")

    def get_selected_object_direction(self, selected_obj, mesh):
        """The way from the drone camera to the selected object that the horizon and tidal lock keep level or at the
        same angle, with their own Mesh settings: get_selected_object_diff_vector(), except that a surface map with
        Mesh off is measured to its center (the mesh's origin, a torus's, as orbit and zoom pivot about,
        get_pivot_vector()). Its closest point steps from face to face across their edges, the bumpy turns Mesh off is
        for avoiding, and the diff vector measures to it with Mesh or not (as auto scale and gravity need). A unit
        vector then, since the center can be past float32's range. A sphere map's center is straight below anyway."""
        if not mesh and anchored_map(selected_obj) is surface:
            vector = self.get_pivot_vector(selected_obj)
            # Scaled down first, so its length doesn't overflow
            biggest = float(np.max(np.abs(vector)))
            if np.isfinite(biggest) and biggest > 0.0:
                vector = vector / biggest
                return mathutils.Vector(vector / np.linalg.norm(vector))
        return self.get_selected_object_diff_vector(selected_obj, mesh)

    def get_tidal_lock_key(self):
        """What tidal lock measures to. When it changes (another object is selected, mesh mode or Split Axes toggled),
        the direction jumps, and the drone isn't turned for that."""
        selection = selected_obj.copy() if isinstance(selected_obj, mathutils.Vector) else selected_obj
        return (selection, bpy.context.scene.control_settings.tidal_lock_mesh,
                self.get_split_axes_settings("tidal_lock"))

    def record_tidal_lock_start(self):
        """Record the direction to the selected object at the end of the tick, for the next tidal_lock() to turn the
        drone by how much flight changes it"""
        if not (bpy.context.scene.control_settings.tidal_lock and selected_obj_bool):
            self._tidal_lock_start = None
            return

        self._tidal_lock_start = (self.get_tidal_lock_key(), self.get_tidal_lock_direction().copy())

    def horizon_ui(self):
        """Level the HUD's horizon (horizon.py) with what's down (get_horizon_level()), or hide it while Horizon is off
        or nothing is down. At the end of the tick, so the next frame draws the newest measure."""
        level = self.get_horizon_level() if bpy.context.scene.control_settings.horizon else {}
        horizon.set_level(**level)

    def get_horizon_level(self):
        """What the horizon is level with, as horizon.set_level()'s arguments: nothing, a down direction, or a pivot
        point. With an object selected, toward it: its origin, or with Horizon's own Mesh, its closest surface point,
        along auto scale's Split Axes axes only, so over a floor split on Z it's straight down to the floor. Its own
        Mesh setting, apart from auto scale's, because on a faceted mesh the closest point's direction stays on one
        face's normal, then swings across the edge to the next one: auto scale can measure to a low-poly sphere's
        surface while the horizon stays level with its center. With nothing selected (as after Auto Deselect), the way
        the drone's gravity pulls. In zero gravity, nothing.

        The origin is sent as a point, and the HUD takes the direction to it from where the camera is in the frame it
        draws, which can be a frame past this tick. A closest surface point's direction, the face's normal, doesn't
        change as the camera slides along the face, so it's sent as a direction. A sphere or surface map's way down
        too (get_selected_object_direction(): its center, or with Mesh, its closest point), as its object's origin is
        the anchor's point, which follows the drone."""
        if selected_obj_bool:
            mesh = bpy.context.scene.control_settings.horizon_mesh
            diff_vector = self.get_selected_object_direction(selected_obj, mesh)
            down = self.get_split_axes_vector(diff_vector, "auto_scale")
            if down.length > 0.0:
                if mesh or anchored_map(selected_obj) is not None:
                    return {"down": down}
                split_axes = self.get_split_axes_settings("auto_scale")
                axes, rotation = split_axes if split_axes else ((True, True, True), (0.0, 0.0, 0.0))
                return {"pivot": get_camera_matrix().translation + diff_vector, "pivot_axes": axes,
                        "pivot_rotation": rotation}

        strength = rig_gravity()
        if strength != 0.0:
            return {"down": drone_obj.matrix_world.to_3x3() @ mathutils.Vector((0.0, 0.0, strength))}
        return {}

    def get_ground(self):
        """What the Altitude readout measures to: the selected object, or with nothing selected, a sphere or surface
        map in the scene (they recenter and build around the drone then, too), so on the Earth it's the height over the
        ground between clicks, or None"""
        if selected_obj_bool:
            return selected_obj
        return next(iter(self.sphere_maps()), None)

    def gravity_pull(self):
        """Have the Drone's Gravity pull toward the selected object (see gravity.py): toward its origin, or with
        Gravity Mesh, the closest point of its mesh, along auto scale's Split Axes axes only, like the horizon's down,
        negative pulling in, as it pulls down along the Drone's Z axis. With nothing selected, it's that again: auto
        scale and stop_at_surface() only work toward a selected object, and pulled to the Earth's ground without them,
        the drone fell through it and swung back and forth through it. At the end of the tick, with the Drone turned as
        the next frames fly it. A recording keys the pull every frame, as it replays the flight."""
        strength = rig_gravity()
        pull = mathutils.Vector((0.0, 0.0, strength))
        if selected_obj_bool:
            pull = mathutils.Vector((0.0, 0.0, 0.0))
            mesh = bpy.context.scene.control_settings.gravity_mesh
            down = self.get_split_axes_vector(self.get_selected_object_diff_vector(selected_obj, mesh), "auto_scale")
            if down.length > 0.0:
                # Into the Drone object's space, without its scale: gravity is in the drone's own units
                rotation = drone_obj.matrix_world.to_3x3().normalized()
                pull = rotation.transposed() @ down.normalized() * -strength
        gravity.set_pull(pull)

    @contextlib.contextmanager
    def jump(self):
        """A move of the drone that isn't flight (a wrap, a recenter, a loop, Keep Drone at Origin), which the HUD's
        Speed and Scale Speed leave out (see odometer.py)"""
        trip = self._odometer
        if trip is not None:
            trip.leave(get_camera_matrix().translation, self.get_true_scale_log2())
        try:
            yield
        finally:
            if trip is not None:
                trip.arrive(get_camera_matrix().translation, self.get_unit_log2(), self.get_true_scale_log2())

    def get_unit_log2(self):
        """log2 of the meters a Blender unit is in the world the drone flies through: the Earth's, if the scene has
        one, otherwise the scene's, after its loops while the drone is in a world they went deeper into
        (loops_go_deeper(); see odometer.py)"""
        for obj in self._earths:
            meters = sphere.meters_log2(obj)
            if meters is not None:
                return meters
        scene = bpy.context.scene
        return odometer.scene_unit_log2(scene, looped=loops_go_deeper(selected_obj, scene))

    def get_true_scale_log2(self):
        """The drone's true scale, as a log2: its camera's Blender scale in the world's meters"""
        camera_scale = get_camera_matrix().to_scale().z
        return (math.log2(camera_scale) if camera_scale > 0.0 else 0.0) + self.get_unit_log2()

    def readouts_ui(self):
        """Measure the HUD's Altitude, Speed, Scale and Scale Speed (hud.set_readings()) every odometer.PERIOD frames,
        those its switches show: in the world's meters, jumps left out, and the drone's true scale"""
        if self._odometer is None:
            return
        scene = bpy.context.scene
        camera_matrix = get_camera_matrix()
        unit = self.get_unit_log2()
        camera_scale = camera_matrix.to_scale().z
        scale = (math.log2(camera_scale) if camera_scale > 0.0 else 0.0) + unit
        self._odometer.size = camera_scale
        reading = self._odometer.read(camera_matrix.translation, unit, scale, scene.frame_current,
                                      scene.render.fps / scene.render.fps_base)
        if reading is None:
            return
        speed, scale_speed = reading

        shown = hud.shown_lines(scene)
        readings = {}
        if "Altitude" in shown:
            readings["Altitude"] = self.get_altitude(camera_matrix.translation, unit)
        if "Speed" in shown:
            readings["Speed"] = speed
        if "Scale" in shown:
            readings["Scale"] = scale * math.log10(2.0)
        if "Scale Speed" in shown:
            readings["Scale Speed"] = scale_speed
        hud.set_readings(readings)

    def get_altitude(self, camera_location, unit):
        """The drone's height in the world's meters, as (its log10, or None at 0, its sign): over what it falls toward
        (get_ground()), the distance auto scale measures to it (its Mesh and Split Axes settings, which each map's
        config sets for its ground: a floor's plane, a planet's surface), or with nothing there, its height over the
        world's Z = 0, Keep Drone at Origin's moves left out. On the Earth's 3D tiles, the distance to their nearest
        ground, which auto scale doesn't measure (it measures height alone, sphere.surface_vector())."""
        ground = self.get_ground()
        if ground is not None:
            if sphere.is_sphere(ground):
                diff_vector = mathutils.Vector(sphere.ground_vector(ground, camera_location))
            elif ground is selected_obj and self._selected_obj_diff_vector is not None:
                diff_vector = self._selected_obj_diff_vector
            else:
                diff_vector = self.get_selected_object_diff_vector(ground)
            height, sign = self.get_auto_scale_distance(diff_vector), 1
        else:
            z = camera_location.z + bpy.context.scene.drone_rig.world_offset[2]
            height, sign = abs(z), (1 if z >= 0.0 else -1)
        if height <= 0.0:
            return None, 1
        return math.log10(height) + unit * math.log10(2.0), sign

    def stop_at_surface(self):
        """Keep flight from carrying the drone through the surface auto scale measures to: the plane through the
        selected object's origin across the chosen axis with Split Axes on one axis, or the object's mesh in mesh mode.
        Auto scale shrinks the drone as it gets closer, so it normally never gets there. But it only scales between
        frames, and a frame of flight covers speed * frame time * scale. Scaled up enough for its distance, the drone
        jumped past the surface in one frame (diving at 2d_time_static's floor, at 13 times its height), and auto scale
        then scaled it back up on the other side. It's moved back to halfway between where it was and the surface
        instead, like auto scale keeping up would have it.

        But no closer to the surface than half the lower bound. Auto scale stops scaling within the lower bound, so in
        mesh mode, halving again every frame took the drone to within a micrometer of 2d_time_static's planet, where
        float32 couldn't tell which side it was on anymore. That close, clip_start already hides the surface.

        A sphere map's 3D tiles have a floor over their ground instead (sphere.floor_move()), which the drone glides
        along: moved out of it every tick, flown into it or not, since a finer tile's ground can come in higher under it."""
        if not selected_obj_bool or not bpy.context.scene.control_settings.auto_scale:
            return
        if isinstance(selected_obj, bpy.types.Object) and sphere.is_sphere(selected_obj) and sphere.floor_of(selected_obj) > 0.0:
            move = sphere.floor_move(selected_obj, get_camera_matrix().translation)
            if move is not None:
                drone_obj.matrix_world = Matrix.Translation(mathutils.Vector(move)) @ drone_obj.matrix_world
            return

        start = self._surface_check_start
        if start is None or start[0] != selected_obj:
            return

        # Following the selected object if it moved. In float64, since float32 is millimeters apart at a far away
        # origin, like 2d_time_static's planet's, 55 km below its surface.
        selection_matrix = self.get_selection_matrix()
        if np.array_equal(selection_matrix, start[1]):
            # Exactly where it was recorded: a sphere or surface map then has the point below it at hand
            start_location = start[2]
        else:
            object_move = selection_matrix @ np.linalg.inv(start[1])
            start_location = (object_move @ np.append(start[2], 1.0))[:3]
        end_location = np.array(get_camera_matrix().translation, dtype=np.float64)
        flight = end_location - start_location
        if not flight.any():
            return

        distances = self.get_surface_crossing(selection_matrix, start_location, end_location)
        if distances is None:
            return

        start_distance, end_distance = abs(distances[0]), abs(distances[1])
        crossing = start_distance / (start_distance + end_distance)
        standoff = (start_distance - self.get_lower_bound() / 2) / (start_distance + end_distance)
        stop_location = start_location + flight * max(0.0, min(crossing / 2, standoff))
        drone_obj.matrix_world = Matrix.Translation(stop_location - end_location) @ drone_obj.matrix_world

    def get_surface_crossing(self, selection_matrix, start_location, end_location):
        """If the move from start_location to end_location crosses the surface auto scale measures to, the signed
        distances of its start and end from it, in world units. Otherwise None. Without Split Axes or mesh mode, that's
        a point, which flight can't cross. Split Axes on two axes measures to a line, which it can't cross either."""
        control_settings = bpy.context.scene.control_settings

        # A sphere or surface map is always measured to its surface (see get_selected_object_diff_vector())
        anchored = anchored_map(selected_obj)
        if anchored is not None:
            return anchored.crossing(selected_obj, start_location, end_location)

        if control_settings.auto_scale_mesh:
            if isinstance(selected_obj, bpy.types.Object) and fractal.is_fractal(selected_obj):
                return fractal.crossing(selected_obj, mathutils.Vector(start_location), mathutils.Vector(end_location))
            if isinstance(selected_obj, mathutils.Vector) or selected_obj.type != 'MESH':
                return None
            obj_eval = selected_obj.evaluated_get(bpy.context.view_layer.depsgraph)
            mesh = obj_eval.data
            if not mesh.polygons:
                return None

            # The ray cast finds the face, in the object's local space. Its float32 start and end can be millimeters off
            # far from the object's origin, so the ray reaches a little past the move at both ends.
            to_local = np.linalg.inv(selection_matrix)
            local_start = (to_local @ np.append(start_location, 1.0))[:3]
            local_flight = (to_local @ np.append(end_location, 1.0))[:3] - local_start
            local_length = np.linalg.norm(local_flight)
            if local_length == 0.0:
                return None
            direction = local_flight / local_length
            margin = 8 * float(np.spacing(np.float32(np.abs(local_start).max() + local_length)))
            hit, _location, _normal, index = obj_eval.ray_cast(mathutils.Vector(local_start - direction * margin),
                                                               mathutils.Vector(direction),
                                                               distance=local_length + 2 * margin)
            if not hit:
                return None

            # Which side of the face's plane the move starts and ends on, in world space and float64
            polygon = mesh.polygons[index]
            normal = np.linalg.inv(selection_matrix[:3, :3]).T @ np.array(polygon.normal, dtype=np.float64)
            normal /= np.linalg.norm(normal)
            point = (selection_matrix @ np.append(np.array(mesh.vertices[polygon.vertices[0]].co, dtype=np.float64), 1.0))[:3]
            start, end = normal @ (start_location - point), normal @ (end_location - point)
            return (start, end) if start > 0.0 >= end or start < 0.0 <= end else None

        axes = (control_settings.auto_scale_x, control_settings.auto_scale_y, control_settings.auto_scale_z)
        # Only one axis measures to a plane. Two measure to the line where their planes meet, and three to the point
        # where all of them do. Stopping at each axis's plane kept the drone in one quarter of a cylinder auto scaled on
        # Y and Z, since it couldn't get past y = 0 or z = 0 to circle the axis.
        if control_settings.auto_scale_split_axes and sum(axes) == 1:
            # The X, Y and Z axes are the world axes turned by auto_scale_rotation. A rotation's inverse is its transpose.
            to_zoom_axes = np.array(mathutils.Euler(control_settings.auto_scale_rotation).to_matrix()).T
            origin = selection_matrix[:3, 3]
            axis = axes.index(True)
            start, end = ((to_zoom_axes @ (location - origin))[axis] for location in (start_location, end_location))
            return (start, end) if start > 0.0 >= end or start < 0.0 <= end else None

        return None

    def record_surface_check_start(self):
        """Record where the drone ends the tick, for the next stop_at_surface() to check the flight from"""
        if not (bpy.context.scene.control_settings.auto_scale and selected_obj_bool):
            self._surface_check_start = None
            return

        selection = selected_obj.copy() if isinstance(selected_obj, mathutils.Vector) else selected_obj
        camera_location = np.array(get_camera_matrix().translation, dtype=np.float64)
        self._surface_check_start = (selection, self.get_selection_matrix(), camera_location)

    def get_selection_matrix(self):
        """The selected object's matrix_world, or a translation to the selected point, as a float64 array"""
        if isinstance(selected_obj, mathutils.Vector):
            return np.array(Matrix.Translation(selected_obj), dtype=np.float64)
        return np.array(selected_obj.matrix_world, dtype=np.float64)

    def scale(self, scale_factor, scale_origin, scale_camera, scale_lens_clip, bounded, allow_loop=True):
        """Scale the drone by scale_factor about scale_origin, within min_scale and max_scale, looping if that's on.
        Returns the part of scale_factor the limits held back: 1.0 if it was all applied."""
        min_drone = min(drone_obj.scale)
        max_drone = max(drone_obj.scale)

        maintain_ratio = bpy.context.scene.control_settings.maintain_ratio
        default_clip_start = bpy.context.scene.control_settings.default_clip_start
        if maintain_ratio and min_scale < scaling_settings.DEFAULT_MIN_SCALE / default_clip_start:
            min_drone_scale = max(min_scale, min_scale / min(drone_camera.scale) / default_clip_start)
        else:
            min_drone_scale = min_scale

        below_min = min_drone * scale_factor < min_drone_scale
        above_max = max_drone * scale_factor > self._max_scale

        if below_min or above_max:
            # A sphere map's globe is the top: there's nothing coarser to climb into, so there the drone grows freely
            # past max_scale, to see the whole globe from as far as it likes (see sphere.py), up to the add-on's own
            # limit. Flying away, auto scale grows the drone with its distance, and its speed with it, so both grew
            # exponentially: to a scale of 7e17 in half a minute at full throttle, where the distance's float32 length
            # overflowed, auto scale scaled the drone by inf, and its matrix went NaN.
            # A surface map's top is the mesh at its own size, likewise
            anchored = anchored_map(selected_obj) if selected_obj_bool else None
            if anchored is not None and above_max and not below_min and anchored.at_top(selected_obj):
                grown = min(scale_factor, scaling_settings.DEFAULT_MAX_SCALE / max_drone)
                self.apply_scale(grown, scale_origin, scale_camera, scale_lens_clip, bounded)
                return scale_factor / grown

            loop = bpy.context.scene.control_settings.loop
            loop_factor = bpy.context.scene.control_settings.loop_factor
            if allow_loop and loop and loop_factor > 1.0 and selected_obj_bool:
                # As many loops as it takes to get within min_scale and max_scale. One isn't always enough, like
                # after clicking an object whose config sets a range far from the drone's scale, and the rest would be
                # clamped in place. It always gets there, since tick() keeps loop_factor at most max_scale / min_scale.
                if below_min:
                    loops = math.ceil(math.log(min_drone_scale / (min_drone * scale_factor), loop_factor))
                else:
                    loops = -math.ceil(math.log(max_drone * scale_factor / self._max_scale, loop_factor))
                factor = loop_factor ** loops
                # A sphere loops by powers of 2, and climbs no higher than its globe (the rest is grown freely, above),
                # nor with 3D tiles dives deeper than they have detail: there it can't loop (1.0), and is held at the
                # scale's bound like a map that doesn't loop, every tick, so without the depsgraph update (3 ms)
                if anchored is not None:
                    factor = anchored.loop_factor(selected_obj, factor)
                    # Nor dives with its surface past the clip end, out of sight at every level: a dive relabels the
                    # world 2^k times deeper about the map's origin, so the drone's numbers grow by that, and scaled
                    # down by hand while flying away from a sphere, it dived every two seconds, 5e18 units out in 40
                    # s, and the drone's scale came out 0 (2026-11-24). There it's held at min_scale instead.
                    if factor > 1.0 and self.surface_out_of_sight():
                        factor = 1.0
                if factor != 1.0:
                    self.loop_scale(factor)

                    # Not for the camera (see get_camera_matrix()): a loop changes the map too, like an endless map's
                    # anchor, which its drivers read, and that's stale until the depsgraph updates
                    bpy.context.view_layer.update()

                # Apply the scaling that was about to happen anyway, checked against the drone's new scale
                return self.scale(scale_factor, scale_origin, scale_camera, scale_lens_clip, bounded, allow_loop=False)

            requested_scale_factor = scale_factor
            if below_min:
                scale_factor = min_drone_scale / min_drone
            else:
                scale_factor = self._max_scale / max_drone

            self.apply_scale(scale_factor, scale_origin, scale_camera, scale_lens_clip, bounded)
            return requested_scale_factor / scale_factor

        self.apply_scale(scale_factor, scale_origin, scale_camera, scale_lens_clip, bounded)
        return 1.0

    def owed(self, held):
        """What auto scale still owes after a scaling the limits held back by held (scale()'s return), at most the
        drone's widest ratio of scales (max_scale / min_scale) either way: owing more is only a longer wait for
        nothing. Held at a limit no loop gets past (a sphere's globe from far out, scaling down by hand while flying
        away), it was multiplied in every tick, and a minute later it was 0, divided by (2026-11-24)."""
        most = self._max_scale / min_scale
        return min(max(held, 1.0 / most), most)

    def loop_scale(self, loop_factor):
        """Loop the drone by loop_factor about the point the selected object's world repeats around: a jump, not
        flight, and a relabeling of the world's meters (see odometer.py)"""
        with self.jump():
            self._loop_scale(loop_factor)

    def _loop_scale(self, loop_factor):
        self._camera_moved = True

        # A sphere map loops about its object's origin, the anchor's point, by a power of 2 (sphere.loop_factor()): N
        # and the anchor scale with the drone, exactly. A climb first moves the anchor to the coarser lattice under the
        # drone, and the drone back by the move of the map's origin.
        # A surface map, the same about its anchor's point on the mesh.
        anchored = anchored_map(selected_obj)
        if anchored is not None:
            if loop_factor == 1.0:
                return
            move = anchored.loop(selected_obj, loop_factor, get_camera_matrix().translation)
            if move is not None:
                drone_obj.matrix_world = Matrix.Translation(-mathutils.Vector(move)) @ drone_obj.matrix_world
            loop_origin = selected_obj.matrix_world.translation.copy()
            self.apply_scale(loop_factor, loop_origin, scale_camera=True, scale_lens_clip=True, bounded=False)
            odometer.relabel(bpy.context.scene, loop_factor)       # a sphere or surface map: loops_go_deeper()
            return

        # Loop around the point auto scale measures to: the closest surface point in mesh mode, otherwise the origin.
        # An endless map is the same after a loop about its origin wherever the drone is, since its anchor scales about
        # it too, exactly (see endless.py), so it's always that. About an endless fractal's closest point (mesh mode),
        # the anchor wouldn't have scaled with the drone.
        if endless.is_endless(selected_obj):
            loop_origin = selected_obj.matrix_world.translation.copy()
        else:
            loop_origin = get_camera_matrix().translation + self.get_selected_object_diff_vector(selected_obj)

        self.apply_scale(loop_factor, loop_origin, scale_camera=True, scale_lens_clip=True, bounded=False)
        if loops_go_deeper(selected_obj, bpy.context.scene):
            odometer.relabel(bpy.context.scene, loop_factor)

        # A Mandelbrot floor isn't self-similar, so the view stays the same through a loop only if the floor scales
        # with the drone, about the same point (see mandelbrot.py)
        mandelbrot.loop(bpy.context.scene, loop_factor, loop_origin)

        # An endless map's anchor (where its origin is in the endless lattice) scales with the loop too, exactly, and
        # the drone takes what's left over scaling it down
        if endless.is_endless(selected_obj):
            remainder = endless.scale_anchor(selected_obj, loop_factor, get_camera_matrix().translation)
            if remainder:
                self.move_drone_in(selected_obj, remainder)
            self.refresh_fractals(get_camera_matrix().translation)

    def move_drone_in(self, obj, offset):
        """Move the drone by offset along an endless map's anchored axes (see endless.along_axes()), in obj's units"""
        move = obj.matrix_world.to_3x3() @ endless.along_axes(obj, offset)
        drone_obj.matrix_world = Matrix.Translation(move) @ drone_obj.matrix_world

    def endless_maps(self):
        """The endless maps to keep the drone near the origin of: the selected object if it's one, or with nothing
        selected, every one in the scene (flying through a map without its config held)"""
        if selected_obj_bool:
            return [selected_obj] if endless.is_endless(selected_obj) else []
        return [obj for obj in bpy.context.scene.objects if endless.is_endless(obj)]

    def recenter_endless(self):
        """Keep the drone within a step of an endless map's origin, so its location never gets big enough for float32
        rounding to show: once it's further, move it back by whole steps and add them to the map's anchor, which keeps
        where it really is exactly (see endless.py). Like a wrap, but the map follows along, so it can grow without
        end."""
        for obj in self.endless_maps():
            camera_before = get_camera_matrix().translation
            steps = endless.recenter_steps(obj, camera_before)
            if any(steps):
                step = float(obj[endless.PROPERTY]["step"])
                endless.shift_anchor(obj, steps)
                with self.jump():
                    self.move_drone_in(obj, [-s * step for s in steps])
                if fractal.is_fractal(obj):
                    move = obj.matrix_world.to_3x3() @ endless.along_axes(obj, [-s * step for s in steps])
                    fractal.refresh(obj, camera_before + move)
                self.remeasure_selected_obj(rescale_reference=False)
        # A sphere or surface map moves its anchor to the lattice point under the drone, and the drone back by the move
        # of the map's origin (sphere.recenter_move(), surface.recenter_move())
        for obj in self.sphere_maps():
            move = anchored_map(obj).recenter_move(obj, get_camera_matrix().translation)
            if move is not None:
                with self.jump():
                    drone_obj.matrix_world = Matrix.Translation(-mathutils.Vector(move)) @ drone_obj.matrix_world
                self.remeasure_selected_obj(rescale_reference=False)

    def sphere_maps(self):
        """The sphere and surface maps to keep the drone near the anchor of: the selected object if it's one, or with
        nothing selected, every one in the scene, like endless_maps()"""
        if selected_obj_bool:
            return [selected_obj] if anchored_map(selected_obj) is not None else []
        return sphere.sphere_objects(bpy.context.scene) + surface.surface_objects(bpy.context.scene)

    def settle_parts(self, camera_matrix):
        """At the end of the tick, with the camera where it ended up: the surface maps' floors (surface.settle()), and
        the sphere and surface maps' 3D parts built, placed and sunk (parts.settle())"""
        for obj in self.sphere_maps():
            surface.settle(obj, camera_matrix.translation, start=False)
            if parts.is_parts(obj):
                parts.settle(obj, camera_matrix.translation, view=camera_matrix.to_3x3())

    def refresh_fractals(self, camera_location, camera_matrix=None):
        """Bring the endless fractals' level objects up to date for the drone camera at camera_location (see
        fractal.refresh()): the selected one, or every one with nothing selected, like recenter_endless(). At the end of
        the tick, with the camera's world matrix, also put what changed in place, and after a loop the eye with it
        (fractal.settle()); otherwise the eye moves once a frame (fractal.follow_eyes())."""
        for obj in self.endless_maps():
            if fractal.is_fractal(obj):
                fractal.refresh(obj, camera_location)
                if camera_matrix is not None:
                    fractal.settle(obj, camera_matrix)

    def apply_scale(self, scale_factor, scale_origin, scale_camera, scale_lens_clip, bounded):
        scale_factor_vector = mathutils.Vector((scale_factor, scale_factor, scale_factor))
        self.scale_drone(scale_origin, scale_factor_vector)

        if not scale_camera:
            self.scale_camera(1/scale_factor, scale_lens_clip=False)

        if scale_lens_clip:
            if bounded:
                drone_camera.data.clip_start = max(self.get_lower_bound(), drone_camera.data.clip_start)

            else:
                drone_camera.data.clip_start *= scale_factor
            drone_camera.data.clip_end *= scale_factor

        self.scale_lens_clip_objects()

        # selected_obj_location = selected_obj.matrix_world.translation
        # camera_location = drone_camera.matrix_world.translation
        # diff_vector =  camera_location - selected_obj_location
        # if diff_vector.length > 0.0:
        #     scale_diff_vector = diff_vector - (diff_vector * scale_factor)
        #     drone_obj.location -= scale_diff_vector

    def scale_drone(self, scale_origin, scale_factor):
        # Not Matrix.Scale along each axis: it works out 1 + (factor - 1) in float32, which is off by 1% at a factor of
        # 1e-6 and exactly 0 below about 3e-8, zeroing the drone's scale
        scale_matrix = np.diag(np.array(scale_factor, dtype=np.float64))
        drone_obj.matrix_world = self.drone_matrix_about(scale_matrix, self.get_pivot_vector(scale_origin))

    def get_pivot_vector(self, pivot):
        """From the drone camera to the point zoom and orbit scale and turn the drone about (world units, float64): the
        drone camera itself, a point, an object's origin, or a sphere or surface map's center (sphere.center_vector(),
        surface.center_vector()), since their object's origin is the anchor's point, which follows the drone. Near the
        ground, the Earth's center is millions of drone sizes away, so in float32 the pivot would be off by drone sizes.
        Past float64's range (a map 2^1024 units deep), the point straight below."""
        camera = np.array(get_camera_matrix().translation, dtype=np.float64)
        if isinstance(pivot, mathutils.Vector):
            return np.array(pivot, dtype=np.float64) - camera
        if pivot is None or pivot == drone_camera:
            return np.zeros(3)
        anchored = anchored_map(pivot)
        if anchored is not None:
            vector = anchored.center_vector(pivot, camera)
            if np.all(np.isfinite(vector)):
                return vector
            return np.array(anchored.surface_vector(pivot, camera), dtype=np.float64)
        return np.array(pivot.matrix_world.translation, dtype=np.float64) - camera

    def drone_matrix_about(self, linear, to_pivot):
        """drone_obj's matrix scaled or turned by linear (3 x 3) about the point to_pivot from the drone camera (see
        get_pivot_vector()). In float64, from the drone's own origin to the pivot, so a pivot far away, like a
        planet's center, moves the drone by exactly the arc or the share of the distance it should."""
        matrix = np.array(drone_obj.matrix_world, dtype=np.float64)
        location = matrix[:3, 3].copy()
        to_pivot = to_pivot + (np.array(get_camera_matrix().translation, dtype=np.float64) - location)
        matrix[:3, :3] = linear @ matrix[:3, :3]
        matrix[:3, 3] = location + to_pivot - linear @ to_pivot
        return Matrix(matrix.tolist())

    def orbit(self):
        if dict_controls["orbit"] > 0.0 and selected_obj_bool:
            if dict_controls["yaw"] != 0.0 or dict_controls["pitch"] != 0.0 or dict_controls["roll"] != 0.0:
                yaw_rotation = dict_controls["yaw"] * self._modal_interval
                pitch_rotation = dict_controls["pitch"] * self._modal_interval
                roll_rotation = dict_controls["roll"] * self._modal_interval

                camera_angle = bpy.context.scene.drone_physics.camera_tilt

                if camera_angle > math.pi/4:
                    rotation_euler = mathutils.Vector((pitch_rotation, -roll_rotation, -yaw_rotation))
                else:
                    rotation_euler = mathutils.Vector((pitch_rotation, yaw_rotation, -roll_rotation))

                self.rotate_drone(rotation_euler)

    def rotate_drone(self, rotation_euler):
        """Turn the drone by rotation_euler about the selected object's origin, or a sphere or surface map's center
        (get_pivot_vector())"""
        rotation_matrix = mathutils.Euler(rotation_euler).to_matrix().to_4x4()

        orbit_axes = bpy.context.scene.control_settings.orbit_axes
        if orbit_axes == "drone_axes":
            local_orientation = get_camera_matrix().to_3x3().normalized().to_4x4()
            local_orientation.translation = mathutils.Vector((0, 0, 0))
            rotation_matrix = local_orientation @ rotation_matrix @ local_orientation.inverted()

        elif orbit_axes == "selected_obj_axes":
            local_orientation = selected_obj.matrix_world.to_3x3().normalized().to_4x4()
            local_orientation.translation = mathutils.Vector((0, 0, 0))
            rotation_matrix = local_orientation @ rotation_matrix @ local_orientation.inverted()

        rotation = np.array(rotation_matrix.to_3x3(), dtype=np.float64)
        self.set_drone_matrix(self.drone_matrix_about(rotation, self.get_pivot_vector(selected_obj)))

        # min = min(drone_scale.x, drone_scale.y, drone_scale.z)
        # middle = sorted([drone_scale.x, drone_scale.y, drone_scale.z])[1]
        # max = max(drone_scale.x, drone_scale.y, drone_scale.z)
        # new_matrix = Matrix.Translation(loc) @ rot.to_matrix().to_4x4() @ Matrix.Diagonal(uniform_scale).to_4x4()

    def set_drone_matrix(self, matrix):
        """Set drone_obj's matrix_world to matrix, rebuilt from its location, rotation and average scale, so rounding
        in the many small turns of orbit and tidal lock can't build up into a skewed or stretched drone"""
        loc, rot, scale = matrix.decompose()
        uniform_scale = sum(scale) / 3.0
        drone_obj.matrix_world = Matrix.Translation(loc) @ rot.to_matrix().to_4x4() @ Matrix.Scale(uniform_scale, 4)

    def scale_camera(self, scale_factor, scale_lens_clip):
        """Scale the Drone_Camera about itself by scale_factor, relative to the drone (its parent), held within the
        min and max scale: the Camera control, and apply_scale()'s counter-scale when Scale Camera is off. The drone's
        flight is the drone object's, so it flies as before.

        On a flat screen the view doesn't change, as a pinhole camera's size doesn't show: only what reads the camera's
        world scale does (lens clip with the option on, the HUD's Scale, maps' detail and fog). Meant for VR, where the
        camera's scale is the viewer's (the eyes' separation and head motion, Godot's XROrigin3D.world_scale): the
        pilot's perceived size, changed without the drone's size or speed. That rig has to scale about the eyes, not
        the play space's floor origin, or the view rises or sinks as it scales."""
        min_camera = min(drone_camera.scale)
        max_camera = max(drone_camera.scale)

        maintain_ratio = bpy.context.scene.control_settings.maintain_ratio
        default_clip_start = bpy.context.scene.control_settings.default_clip_start
        if maintain_ratio and min_scale < scaling_settings.DEFAULT_MIN_SCALE / default_clip_start:
            drone_scale = min(drone_obj.matrix_world.to_scale())
            min_camera_scale = max(min_scale, min_scale / drone_scale / default_clip_start)
        else:
            min_camera_scale = min_scale

        if min_camera * scale_factor < min_camera_scale:
            scale_factor = min_camera_scale / min_camera

        elif max_camera * scale_factor > self._max_scale:
            scale_factor = self._max_scale / max_camera

        drone_camera.scale *= scale_factor

        if scale_lens_clip:
            drone_camera.data.clip_start *= scale_factor
            drone_camera.data.clip_end *= scale_factor

        self.scale_lens_clip_objects()

    def lens_clip(self, bounded, lower_bound):
        if dict_controls["lens_clip"] != 1.0:
            lens_log_scale = dict_controls["lens_clip"]**self._modal_interval

            if bounded:
                lens_distance = self._selected_obj_diff_vector.length - drone_camera.data.clip_start
                lens_log_distance = lens_distance - (lens_distance / lens_log_scale)
                new_length = drone_camera.data.clip_start + lens_log_distance

                bounded_buffer = 0.01
                upper_bound = self._selected_obj_diff_vector.length * (1 - bounded_buffer)

                if upper_bound > lower_bound and selected_obj != drone_obj:
                    if dict_controls["lens_clip"] > 1.0 or new_length <= upper_bound:
                        lens_log_scale = max(lower_bound, new_length) / drone_camera.data.clip_start

                else:
                    shrink_log_scale = min(lens_log_scale, 1/lens_log_scale)
                    lens_log_scale = max(lower_bound / drone_camera.data.clip_start, shrink_log_scale)

            else:
                drone_camera.data.clip_end *= lens_log_scale

            drone_camera.data.clip_start *= lens_log_scale
            self.scale_lens_clip_objects()

    def growth_room(self):
        """How much the scale control may grow the drone, with Bounded and auto scale: until the selected object (its
        surface, in mesh mode) is a default clip start away (see get_lower_bound()), and never less than 1. Grown
        further, everything near would be inside the clip start, out of sight, and auto scale would only shrink the
        drone back (see auto_scale())."""
        distance = self._selected_obj_diff_vector.length if self._selected_obj_diff_vector is not None else 0.0
        return max(1.0, (1.0 - GROWTH_BUFFER) * distance / self.get_lower_bound())

    def surface_out_of_sight(self):
        """Whether the selected object (its surface, as auto scale measures it) is farther than the camera's clip end
        (world units: lens clip scaling keeps it so, see scale_lens_clip_objects())"""
        if self._selected_obj_diff_vector is None:
            return False
        return self._selected_obj_diff_vector.length > drone_camera.data.clip_end

    def get_lower_bound(self):
        """The shortest clip_start Bounded allows: default_clip_start, in the Drone_Camera's world scale. Worked out
        from the Drone's and Drone_Camera's own scales, like scale_lens_clip_objects() does, since
        drone_camera.matrix_world is stale until the depsgraph updates. Read from it after scaling the drone in the same
        tick, the bound belonged to the drone's scale before that, and under fast scaling it ratcheted clip_start up
        past clip_end."""
        default_clip_start = bpy.context.scene.control_settings.default_clip_start
        return default_clip_start * drone_camera.scale.z * drone_obj.matrix_world.to_scale().z

    def scale_lens_clip_objects(self):
        if clip_end_obj:
            clip_end_obj_scale = drone_camera.data.clip_end * clip_end_ratio / (drone_camera.scale.z * drone_obj.matrix_world.to_scale().z)
            clip_end_obj.scale = (clip_end_obj_scale, clip_end_obj_scale, clip_end_obj_scale)

    def wrap(self):
        wrap = bpy.context.scene.control_settings.wrap
        if wrap:
            # The ranges are along the world axes turned by wrap_rotation around the pivot, so measure the drone from the
            # pivot along them. A rotation's inverse is its transpose.
            wrap_rotation = mathutils.Euler(bpy.context.scene.control_settings.wrap_rotation).to_matrix()
            drone_location = wrap_rotation.transposed() @ (get_camera_matrix().translation - self.get_wrap_pivot())
            wrap_vector = mathutils.Vector((0.0, 0.0, 0.0))

            wrap_split_axes = bpy.context.scene.control_settings.wrap_split_axes
            if not wrap_split_axes:
                wrap_range = bpy.context.scene.control_settings.wrap_range

                wrap_vector.x = self.get_wrap_offset(drone_location.x, wrap_range)
                wrap_vector.y = self.get_wrap_offset(drone_location.y, wrap_range)
                wrap_vector.z = self.get_wrap_offset(drone_location.z, wrap_range)
            else:
                wrap_split_axes_x = bpy.context.scene.control_settings.wrap_split_axes_x
                wrap_range_x = bpy.context.scene.control_settings.wrap_range_x

                wrap_split_axes_y = bpy.context.scene.control_settings.wrap_split_axes_y
                wrap_range_y = bpy.context.scene.control_settings.wrap_range_y

                wrap_split_axes_z = bpy.context.scene.control_settings.wrap_split_axes_z
                wrap_range_z = bpy.context.scene.control_settings.wrap_range_z

                if wrap_split_axes_x:
                    wrap_vector.x = self.get_wrap_offset(drone_location.x, wrap_range_x)
                if wrap_split_axes_y:
                    wrap_vector.y = self.get_wrap_offset(drone_location.y, wrap_range_y)
                if wrap_split_axes_z:
                    wrap_vector.z = self.get_wrap_offset(drone_location.z, wrap_range_z)

            if wrap_vector.length > 0.0:
                # Back from the wrap axes to world space, so a parented drone also moves the full wrap range
                wrap_vector = wrap_rotation @ wrap_vector
                with self.jump():
                    drone_obj.matrix_world = Matrix.Translation(wrap_vector) @ drone_obj.matrix_world
                self.remeasure_selected_obj(rescale_reference=False)

    def get_wrap_offset(self, location, wrap_range):
        """How far to move along one wrap axis to get from location back within wrap_range: as many range lengths as it
        takes, all at once. Moving one range length per tick, a drone that got far out of range (flying fast at a large
        scale, or clicking an object from far away) took a noticeable time to come back."""
        low, high = wrap_range
        length = high - low
        if length <= 0.0 or low <= location <= high:
            return 0.0

        return -math.floor((location - low) / length) * length

    def get_wrap_pivot(self):
        """The point the wrap ranges are measured from, in the scene's current coordinates: the Pivot location if it's
        on, otherwise the selected object's origin, or the world origin if no object is selected."""
        control_settings = bpy.context.scene.control_settings
        world_offset = mathutils.Vector(bpy.context.scene.drone_rig.world_offset)

        if control_settings.wrap_custom_pivot:
            # Pivot is in the scene's original coordinates, before Keep Drone at Origin moved it
            return mathutils.Vector(control_settings.wrap_pivot) - world_offset
        if selected_obj_bool:
            return selected_obj.matrix_world.translation.copy()
        return -world_offset

    def move_and_record(self):
        record_camera = bpy.context.scene.flight_options.record_camera
        record_obj_name = bpy.context.scene.flight_options.record_obj
        record_obj = bpy.context.scene.objects.get(record_obj_name)

        current_frame = bpy.context.scene.frame_current

        self.keep_drone_at_origin(record_camera, record_obj, current_frame)

        self.record_camera(record_camera, record_obj, current_frame)

        # Recording inputs, the Drone's transform as this frame shows it, after this tick's scaling, orbit or origin
        # move (replacing what the frame's step keyed): the world objects' moves are keyed on this frame too
        if physics.recording is not None:
            physics.recording.key_transform(current_frame)

    def keep_drone_at_origin(self, record_camera, record_obj, current_frame):
        keep_drone_at_origin = bpy.context.scene.control_settings.keep_drone_at_origin
        camera_location = get_camera_matrix().translation

        # Only once the camera is far enough from the origin, for its clip_start, that float32 rounding would show
        max_distance = ORIGIN_DISTANCE * drone_camera.data.clip_start
        if keep_drone_at_origin and camera_location.length > max_distance:
            scene = bpy.context.scene
            # Recording the camera or the inputs: either replay shows the world where it was
            record = record_camera and record_obj or physics.recording is not None

            # Key the offset the objects had until now, or the move would apply from the start of the recording. Not
            # if the origin already moved on this frame, since the previous frame then already holds that offset.
            if record and self._origin_moved_frame != current_frame:
                for obj in get_world_objects(scene):
                    obj.keyframe_insert(data_path="delta_location", frame=current_frame - 1)

            with self.jump():
                move_world_origin(scene, camera_location)

            if record:
                for obj in get_world_objects(scene):
                    obj.keyframe_insert(data_path="delta_location", frame=current_frame)
                self._origin_moved_frame = current_frame

            # The scene's objects moved through delta_location, and their matrix_world is stale until the depsgraph
            # updates, which this does
            self.remeasure_selected_obj()

    def record_camera(self, record_camera, record_obj, current_frame):
        if record_camera and record_obj:
            camera_matrix = get_camera_matrix()
            record_obj.location = camera_matrix.translation
            record_obj.rotation_euler = camera_matrix.to_euler()
            record_obj.scale = camera_matrix.to_scale()

            record_obj.keyframe_insert(data_path="location", frame=current_frame)
            record_obj.keyframe_insert(data_path="rotation_euler", frame=current_frame)
            record_obj.keyframe_insert(data_path="scale", frame=current_frame)

            if record_obj.type == 'CAMERA':
                record_obj.data.lens = drone_camera.data.lens
                record_obj.data.clip_start = drone_camera.data.clip_start
                record_obj.data.clip_end = drone_camera.data.clip_end

                record_obj.data.keyframe_insert(data_path="lens", frame=current_frame)
                record_obj.data.keyframe_insert(data_path="clip_start", frame=current_frame)
                record_obj.data.keyframe_insert(data_path="clip_end", frame=current_frame)

    def prepare_flight(self, context):
        """Everything to do before a flight opens the controller: find the drone and its camera, get the physics ready
        (start_physics()), and the maps ready for where the drone starts. False, after reporting why, if it can't fly."""
        self.get_drone_objects(context)
        self.reset_objects()

        # Report error if no drone object
        if drone_obj == None:
            self.report({'ERROR'}, "Click Add Drone first")
            return False

        # The selection as a tick leaves it, for the first tick: fit_scale_range() runs before the tick reads the
        # selection, and on the first tick it loops whenever the drone's scale is outside the scaling range and an
        # object is selected, about that object. get_drone_objects() had just reset selected_obj to None, but
        # selected_obj_bool kept the last flight's True (the user, 2026-11-16: an AttributeError in
        # get_selected_object_diff_vector() as the flight started)
        global selected_obj
        global selected_obj_bool
        if drone_camera:
            selected_obj = self.get_selected_object()
            selected_obj_bool = bool(selected_obj) and not self.is_drone_part(selected_obj)

        if not start_physics(context, self.report):
            return False

        # Endless maps make up their random digits now, rather than in a hitch on the first loop
        if drone_camera:
            for obj in context.scene.objects:
                if endless.is_endless(obj):
                    endless.prepare(obj, drone_camera.matrix_world.translation)
                if fractal.is_fractal(obj):
                    fractal.settle(obj, drone_camera.matrix_world)
                    fractal.follow(obj, drone_camera.matrix_world)
                surface.settle(obj, drone_camera.matrix_world.translation)
                if parts.is_parts(obj):
                    parts.settle(obj, drone_camera.matrix_world.translation)

        # The flight's path from here, in the world's meters (the Earth's, on an Earth)
        if drone_camera:
            self._earths = [obj for obj in sphere.sphere_objects(context.scene) if sphere.meters_log2(obj) is not None]
            camera_matrix = get_camera_matrix()
            unit = self.get_unit_log2()
            self._odometer = odometer.Odometer(camera_matrix.translation, unit,
                                               math.log2(camera_matrix.to_scale().z) + unit,
                                               context.scene.frame_current)

        # Store first frame
        global record_start_frame
        record_start_frame = bpy.context.scene.frame_current
        return True

    def abandon_flight(self, context):
        """A flight prepare_flight() got ready for stops before it started: no controller"""
        physics.stop()

    def start_flight(self, context, controller):
        """The flight starts, with the controller open: the HUD's gimbals read a gamepad's sticks as SDL maps them, and
        the drone flies a step on every frame change"""
        self._gamepad = gamepad.open_gamepad(controller)
        hud.gimbal_axes = gamepad.gimbal_axes(self._gamepad)
        background.hold_heap(True)

        if fly_frame not in bpy.app.handlers.frame_change_pre:
            bpy.app.handlers.frame_change_pre.append(fly_frame)

    def get_drone_objects(self, context):
        global drone_obj
        global drone_camera
        global clip_end_obj
        global clip_end_ratio
        global selected_obj
        global selected_obj_bool

        drone_obj = None
        drone_camera = None
        clip_end_obj = None
        clip_end_ratio = None
        selected_obj = None
        # With selected_obj, or the last flight's True outlives it into the next one's first tick
        selected_obj_bool = False

        drone_obj = context.scene.drone_rig.drone
        if drone_obj:
            for child in drone_obj.children:
                if child.type == 'CAMERA':
                    drone_camera = child

                    for child in drone_camera.children:
                        if child.name == 'clip_end':
                            clip_end_obj = child
                            clip_end_ratio = clip_end_obj.matrix_world.to_scale().z / drone_camera.data.clip_end
                    break

    def reset_objects(self):
        self._active_object = None
        self._parent = None
        self._parent_scale = None
        self._modal_interval = None
        self._last_time = time.perf_counter()
        self._origin_moved_frame = None
        self._surface_check_start = None
        self._tidal_lock_start = None
        self._odometer = None
        self._earths = ()

        if drone_obj:
            if drone_obj.parent:
                self._parent = drone_obj.parent
                self._parent_scale = drone_obj.parent.matrix_world.to_scale().copy()

    def end_flight(self, context):
        """The flight ends, before the controller closes: the HUD's flight marks go, the drone stays where the flight
        left it, and a recorded flight goes back to its first frame, ready to replay"""
        # Nothing measures the horizon once flying stops, and the drone may be moved anywhere. The velocity marker would
        # go on showing the flight's last velocity.
        horizon.set_level()
        velocity.set_shown(False)

        # Store end frame
        global record_end_frame
        record_end_frame = bpy.context.scene.frame_current

        # The flight's frame steps end, and the drone stays where the flight left it, for the next flight to go on from
        # (physics.py)
        if fly_frame in bpy.app.handlers.frame_change_pre:
            bpy.app.handlers.frame_change_pre.remove(fly_frame)
        physics.stop()
        background.hold_heap(False)
        for key in STICKS:
            dict_controls[key] = 0.0

        # A recorded flight goes back to its first frame, ready to replay
        if context.scene.flight_options.record:
            bpy.context.scene.frame_current = record_start_frame
            physics.replay(context.scene)

        if self._gamepad:
            sdl2.SDL_GameControllerClose(self._gamepad)
            self._gamepad = None

# The Reset operators that move the drone are FlightFeatures too, for the flight's own ways of scaling it (since
# 2026-11-16 the drone is scaled through drone_matrix_about(), which they didn't have)
class ER_OT_ResetScale(FlightFeatures, bpy.types.Operator):
    """Restore the scale of the Drone object to 1."""
    bl_idname = "er.reset_scale"
    bl_label = "Reset Scale"
    
    def execute(self, context):
        self.get_drone_objects(context)

        if drone_obj:
            if drone_camera:
                inverse_scale = mathutils.Vector((
                    1.0 / drone_obj.scale.x if drone_obj.scale.x != 0 else 1.0,
                    1.0 / drone_obj.scale.y if drone_obj.scale.y != 0 else 1.0,
                    1.0 / drone_obj.scale.z if drone_obj.scale.z != 0 else 1.0
                ))

                scale_camera = bpy.context.scene.control_settings.scale_camera
                if not scale_camera:
                    drone_camera.scale *= drone_obj.scale.z

                self.scale_drone(drone_camera, inverse_scale)

                scale_lens_clip = bpy.context.scene.control_settings.scale_lens_clip
                if scale_lens_clip:
                    drone_camera.data.clip_start *= inverse_scale.z
                    drone_camera.data.clip_end *= inverse_scale.z

                self.scale_lens_clip_objects()

            else:
                drone_obj.scale = (1, 1, 1)
                print("Drone_Camera not found, scale reset based on Drone origin.")

            # The HUD's Scale reads 1 again, whatever loops relabeled the world's meters by (see odometer.py)
            odometer.reset(context.scene)

            self.report({'INFO'}, "Scale reset!")

        return {'FINISHED'}

class ER_OT_ResetLocations(FlightFeatures, bpy.types.Operator):
    """Restore the original locations of the objects in your scene."""
    bl_idname = "er.reset_locations"
    bl_label = "Reset Locations"
    
    def execute(self, context):
        ER_OT_ResetScale.execute(self, context)

        world_offset = context.scene.drone_rig.world_offset
        move_world_origin(context.scene, [-offset for offset in world_offset])

        self.report({'INFO'}, "Object locations reset!")
        return {'FINISHED'}

class ER_OT_ResetCamera(FlightFeatures, bpy.types.Operator):
    """Restore the scale of the Drone_Camera object to 1."""
    bl_idname = "er.reset_camera"
    bl_label = "Reset Scale"
    
    def execute(self, context):
        self.get_drone_objects(context)

        if drone_camera:
            scale_lens_clip_with_camera = bpy.context.scene.control_settings.scale_lens_clip_with_camera
            if scale_lens_clip_with_camera:
                lens_scale = 1 / drone_camera.scale.z
                drone_camera.data.clip_start *= lens_scale
                drone_camera.data.clip_end *= lens_scale

            drone_camera.scale = (1, 1, 1)
            self.scale_lens_clip_objects()

            self.report({'INFO'}, "Drone_Camera scale reset!")

        return {'FINISHED'}

class ER_OT_ResetLensClip(FlightFeatures, bpy.types.Operator):
    """Reset the clip start length to the inputted value multiplied by the scale of the Drone object. The clip end length will be scaled proportionally."""
    bl_idname = "er.reset_lens_clip"
    bl_label = "Reset Lens Clip"
    
    def execute(self, context):
        self.get_drone_objects(context)

        if drone_camera:
            default_clip_start = bpy.context.scene.control_settings.default_clip_start

            drone_camera.data.clip_start = default_clip_start * drone_camera.matrix_world.to_scale().z
            drone_camera.data.clip_end = drone_camera.data.clip_start * 100_000

            self.scale_lens_clip_objects()

            self.report({'INFO'}, "Lens Clip reset!")

        return {'FINISHED'}

class ER_OT_ResetMinScale(bpy.types.Operator):
    """Restore Min Scale to its default value."""
    bl_idname = "er.reset_min_scale"
    bl_label = "Reset Min Scale"

    def execute(self, context):
        context.scene.control_settings.min_scale = scaling_settings.DEFAULT_MIN_SCALE

        self.report({'INFO'}, "Min Scale reset!")
        return {'FINISHED'}

class ER_OT_ResetMaxScale(bpy.types.Operator):
    """Restore Max Scale to its default value."""
    bl_idname = "er.reset_max_scale"
    bl_label = "Reset Max Scale"

    def execute(self, context):
        context.scene.control_settings.max_scale = scaling_settings.DEFAULT_MAX_SCALE

        self.report({'INFO'}, "Max Scale reset!")
        return {'FINISHED'}
