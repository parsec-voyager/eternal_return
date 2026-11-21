"""The add-on's operators: Fly (the modal loop around flight.FlightFeatures), Add Drone, Bake Animation and
Store/Load Controls, with the drone rig and flight options settings groups and the add-on preferences Store Controls
keeps the controls' settings in."""
import traceback

import bpy

from . import controls
from . import flight
from . import hud_render
from . import physics
from . import rig
from . import sound

# The modal timer's interval, in seconds: one tick of flight each
TICK_INTERVAL = 0.01
# What Bake Animation names the camera it bakes the flight into
BAKED_CAMERA = "Baked Drone Camera"

# The controls' settings groups Store and Load Controls copy between the scene and the add-on preferences
STORED_GROUPS = ("control_axes", "control_settings", "stick_rates")


class DroneRig(bpy.types.PropertyGroup):
    """The scene's drone, its camera, and how far Keep Drone at Origin has moved the world (scene.drone_rig)"""
    drone: bpy.props.PointerProperty(
        name="Drone", description="The scene's drone, which Add Drone built", type=bpy.types.Object)
    camera: bpy.props.PointerProperty(
        name="Drone Camera", description="The drone's camera", type=bpy.types.Object)
    world_offset: bpy.props.FloatVectorProperty(
        name="World Offset",
        description="How far Keep Drone at Origin has moved the scene's objects. Reset Locations moves them back.",
        size=3, subtype='TRANSLATION', default=(0.0, 0.0, 0.0))


class FlightOptions(bpy.types.PropertyGroup):
    """Recording and sound settings (scene.flight_options)"""
    record: bpy.props.BoolProperty(
        name="Record", description="Record the flight's inputs as keyframes on the Drone. Playing the timeline "
                                   "replays them", default=False)
    record_camera: bpy.props.BoolProperty(
        name="Record Camera", description="Key Drone_Camera's transforms onto another camera every tick, for "
                                          "rendering", default=False)
    record_obj: bpy.props.StringProperty(
        name="Camera Object", description="The camera Record Camera keys", default="")
    volume: bpy.props.FloatProperty(
        name="Sound Volume", description="How loud the drone sounds", default=10.0, min=0.0, max=100.0,
        subtype='PERCENTAGE', precision=0)
    drone_sound: bpy.props.EnumProperty(
        name="Drone Sound", description="The sound the drone makes while flying",
        items=[(name, text, f"Play {name} while flying") for name, text, _speed in sound.SOUNDS],
        default=sound.SOUNDS[0][0])


# The settings groups, by the scene attribute each is kept under
SCENE_GROUPS = {"drone_rig": DroneRig, "flight_options": FlightOptions}


class EternalReturnPreferences(bpy.types.AddonPreferences):
    """The controls' settings as Store Controls last kept them, for Load Controls to give any scene"""
    bl_idname = __package__
    __annotations__ = {name: bpy.props.PointerProperty(type=controls.SCENE_GROUPS[name]) for name in STORED_GROUPS}

    def draw(self, context):
        self.layout.label(text="Store and Load in the Eternal Return panel's Controls keep the controls' settings here")


class ER_OT_Fly(flight.FlightFeatures, bpy.types.Operator):
    """Fly the drone camera with the controller. Click again, or press Esc, to stop"""
    bl_idname = "er.fly"
    bl_label = "Fly"

    # What execute() set up, for cancel() to take down: on the instance once set, None until then
    _controller = None
    _timer = None
    _sound = None
    _sync_mode = None

    def read_controls(self):
        """The controller's axes, and each control's raw value from them. Through the controls module, so the flight
        harness's virtual controller (tools/flight_sim.py replaces controls.read_axes) takes effect."""
        axes = controls.read_axes(self._controller)
        return controls.map_controls(axes, bpy.context.scene), axes

    def update_sound(self, context):
        """The propellers follow the throttle: its thrust as a fraction of the Maximum Thrust"""
        if self._sound is None:
            return
        power = context.scene.control_settings.max_thrust
        throttle = abs(flight.dict_controls.get("throttle", 0.0)) / power if power > 0.0 else 0.0
        self._sound.update(throttle)

    def execute(self, context):
        window_manager = context.window_manager
        # The panel's button toggles: the running Fly operator sees the flag on its next event and stops itself
        if window_manager.er_fly_modal_running:
            window_manager.er_fly_modal_running = False
            return {'CANCELLED'}

        if not self.prepare_flight(context):
            return {'CANCELLED'}

        controller = controls.get_controller(self.report)
        if controller is None:
            controls.quit_sdl()
            self.abandon_flight(context)
            return {'CANCELLED'}
        self._controller = controller
        self.start_flight(context, controller)

        # Every frame played, and simulated: none skipped to keep up with real time
        self._sync_mode = context.scene.sync_mode
        context.scene.sync_mode = 'NONE'

        self._sound = sound.PropellerSound()

        # animation_play() toggles playback, so only while stopped
        screen = context.screen
        if screen is not None and not screen.is_animation_playing:
            bpy.ops.screen.animation_play()

        self._timer = window_manager.event_timer_add(TICK_INTERVAL, window=context.window)
        window_manager.modal_handler_add(self)
        window_manager.er_fly_modal_running = True
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if not context.window_manager.er_fly_modal_running or event.type == 'ESC':
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            try:
                self.tick(context)
            except Exception:
                # An exception escaping modal() would end the operator without cancel(), leaving playback, the frame
                # handler, SDL and the sound running
                traceback.print_exc()
                self.report({'ERROR'}, "The drone stopped after an error. See the system console for details.")
                self.cancel(context)
                return {'CANCELLED'}

        # The pilot can still use Blender while flying
        return {'PASS_THROUGH'}

    def cancel(self, context):
        """Stop flying. Blender calls this too (when a file loads, say), so anything may be only partly set up."""
        window_manager = context.window_manager
        window_manager.er_fly_modal_running = False
        if self._sync_mode is not None:
            context.scene.sync_mode = self._sync_mode
            self._sync_mode = None
        if self._timer is not None:
            window_manager.event_timer_remove(self._timer)
            self._timer = None
        screen = context.screen
        if screen is not None and screen.is_animation_playing:
            # Where playback is, not where it started: the flight goes on from there
            bpy.ops.screen.animation_cancel(restore_frame=False)

        self.end_flight(context)

        controls.close_controller(self._controller)
        self._controller = None
        controls.quit_sdl()
        if self._sound is not None:
            self._sound.stop()
            self._sound = None

        if context.scene.flight_options.record:
            self.report({'INFO'}, "Landed. The flight's inputs are keyed on the Drone: play the timeline to replay it.")
        else:
            self.report({'INFO'}, "Landed.")


def is_drone(obj):
    """A Drone this add-on built: its mesh is the drone's (rig.MESH_NAME)"""
    return obj.type == 'MESH' and obj.name.startswith("Drone") and obj.data.name.startswith(rig.MESH_NAME)


def is_drone_camera(obj):
    return obj.type == 'CAMERA' and obj.name.startswith("Drone_Camera") and obj.data.name.startswith("Drone_Camera")


def remove_drones():
    """Delete every earlier drone in the file, with its camera and their data"""
    for obj in [obj for obj in bpy.data.objects if is_drone(obj) or is_drone_camera(obj)]:
        bpy.data.objects.remove(obj, do_unlink=True)
    # Their data, and any left over without an object
    for mesh in [mesh for mesh in bpy.data.meshes if mesh.name.startswith(rig.MESH_NAME)]:
        bpy.data.meshes.remove(mesh, do_unlink=True)
    for camera in [camera for camera in bpy.data.cameras if camera.name.startswith("Drone_Camera")]:
        bpy.data.cameras.remove(camera, do_unlink=True)


class ER_OT_AddToScene(bpy.types.Operator):
    """Give the scene a new drone and drone camera, in place of any it had"""
    bl_idname = "er.add_to_scene"
    bl_label = "Add Drone to Scene"

    def execute(self, context):
        scene = context.scene
        # The HUD's render objects, remade below
        hud_render.remove(scene)

        remove_drones()

        drone_rig = scene.drone_rig
        drone = rig.build_drone(scene)
        drone_rig.drone = drone
        drone_rig.camera = rig.build_camera(scene, drone, scene.control_settings.default_clip_start)

        hud_render.update(scene)

        for obj in context.selected_objects:
            obj.select_set(False)
        drone.select_set(True)
        context.view_layer.objects.active = drone
        self.report({'INFO'}, "Drone added to the scene.")
        return {'FINISHED'}


class ER_OT_BakeAnimation(bpy.types.Operator):
    """Bake the drone camera's flight into an ordinary camera with editable location and rotation keyframes"""
    bl_idname = "er.bake_animation"
    bl_label = "Bake Animation"

    start_frame: bpy.props.IntProperty(name="Start Frame", description="The first frame to bake", default=1, min=1)
    end_frame: bpy.props.IntProperty(name="End Frame", description="The last frame to bake", default=250, min=1)

    def invoke(self, context, event):
        # The drone's recording's frames, else the last flight's
        drone_obj = context.scene.drone_rig.drone
        take = physics.Recording.of(drone_obj) if drone_obj and physics.available() else None
        if take is not None:
            self.start_frame, self.end_frame = take.start_frame, take.end_frame()
        else:
            self.start_frame, self.end_frame = flight.record_start_frame, flight.record_end_frame
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        scene = context.scene
        drone_camera = scene.drone_rig.camera
        if drone_camera is None:
            self.report({'ERROR'}, "No Drone Camera detected. Click Add Drone first")
            return {'CANCELLED'}

        # A camera following the drone camera, baked into keyframes with its constraints cleared
        camera_data = bpy.data.cameras.new(BAKED_CAMERA)
        camera_data.lens = drone_camera.data.lens
        camera_data.clip_start = drone_camera.data.clip_start
        camera_data.clip_end = drone_camera.data.clip_end
        camera = bpy.data.objects.new(BAKED_CAMERA, camera_data)
        (context.collection or scene.collection).objects.link(camera)
        for kind in ('COPY_LOCATION', 'COPY_ROTATION'):
            constraint = camera.constraints.new(kind)
            constraint.target = drone_camera

        for obj in context.selected_objects:
            obj.select_set(False)
        camera.select_set(True)
        context.view_layer.objects.active = camera
        context.view_layer.update()

        # Each frame of the range replays the recording (physics.py's frame handler) as the bake steps through it
        bpy.ops.nla.bake(frame_start=self.start_frame, frame_end=self.end_frame, only_selected=True,
                         visual_keying=True, clear_constraints=True, bake_types={'OBJECT'},
                         channel_types={'LOCATION', 'ROTATION'})
        self.report({'INFO'}, f"Baked frames {self.start_frame}-{self.end_frame} into {camera.name}: its location and "
                              f"rotation keys can be edited in the Graph Editor.")
        return {'FINISHED'}


def copy_settings(source, target):
    """Copy every add-on-defined setting of a settings group onto another of the same class: its own and its mixins'
    (control_settings' scaling settings), not Blender's own"""
    for prop in source.bl_rna.properties:
        name = prop.identifier
        if name in ("rna_type", "name") or prop.is_readonly or prop.type in ('POINTER', 'COLLECTION'):
            continue
        value = getattr(source, name)
        if getattr(prop, "array_length", 0) > 0:
            value = tuple(value)
        setattr(target, name, value)


def preferences(context):
    return context.preferences.addons[__package__].preferences


class ER_OT_StoreControls(bpy.types.Operator):
    """Keep the scene's controls settings in the add-on preferences, for Load Controls to give other scenes"""
    bl_idname = "er.store_controls"
    bl_label = "Store Controls"

    def execute(self, context):
        stored = preferences(context)
        for name in STORED_GROUPS:
            copy_settings(getattr(context.scene, name), getattr(stored, name))
        self.report({'INFO'}, "The controls' settings are kept in the preferences.")
        return {'FINISHED'}


class ER_OT_LoadControls(bpy.types.Operator):
    """Give the scene the controls settings Store Controls kept in the add-on preferences"""
    bl_idname = "er.load_controls"
    bl_label = "Load Controls"

    def execute(self, context):
        stored = preferences(context)
        for name in STORED_GROUPS:
            copy_settings(getattr(stored, name), getattr(context.scene, name))
        self.report({'INFO'}, "The scene has the controls' settings from the preferences.")
        return {'FINISHED'}


OPERATORS = (ER_OT_Fly, ER_OT_AddToScene, ER_OT_BakeAnimation, ER_OT_StoreControls, ER_OT_LoadControls)
