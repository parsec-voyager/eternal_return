"""The scaling tools' settings, and the flight settings that aren't a control's own (the HUD's Horizon and Velocity,
Solid Fog, ...): the ScalingSettings mixin of the scene's controls_parameters (controls.py),
so every one of them stays where scenes and scaling configs have them."""
import math
import bpy

# Default scale limits of the Drone object. User values cannot go beyond these.
DEFAULT_MIN_SCALE = 0.000_001
DEFAULT_MAX_SCALE = 33_000_000

DEFAULT_WRAP_RANGE = (-10.0, 10.0)

# Getter/setter factories for (min, max) FloatVectorProperties that keep min <= max.
# The value is stored in a custom property under `key`.
def range_getter(key, default=DEFAULT_WRAP_RANGE):
    def get(self):
        return self.get(key, default)
    return get

def range_setter(key, default=DEFAULT_WRAP_RANGE):
    # If an edit pushes one end past the other, the edited end is clamped to the other end
    def set(self, value):
        old_min, old_max = self.get(key, default)
        new_min, new_max = value
        if new_min > new_max:
            if new_min != old_min:
                new_min = new_max
            else:
                new_max = new_min
        self[key] = (new_min, new_max)
    return set

class ScalingSettings:
    min_scale: bpy.props.FloatProperty(
        name="Min Scale",
        description="The smallest size the Drone object can be scaled to. Cannot be less than the default.",
        default=DEFAULT_MIN_SCALE,
        min=DEFAULT_MIN_SCALE,
        max=DEFAULT_MAX_SCALE,
        precision=6
    )

    max_scale: bpy.props.FloatProperty(
        name="Max Scale",
        description="The largest size the Drone object can be scaled to. Cannot be greater than the default.",
        default=DEFAULT_MAX_SCALE,
        min=DEFAULT_MIN_SCALE,
        max=DEFAULT_MAX_SCALE,
        precision=6
    )

    loop: bpy.props.BoolProperty(
        name="Loop",
        description="Uses the Zoom feature to adjust the scale of the Drone object if it reaches the minimum or maximum size.",
        default=False
    )

    loop_factor: bpy.props.FloatProperty(
        name="Loop Factor",
        description="Must be greater than 1. The Drone object is scaled (using Zoom) by this amount when it reaches the minimum size, or 1 over this amount when it reaches the maximum size.",
        default=64.0,
        min=1.0
    )

    wrap: bpy.props.BoolProperty(
        name="Wrap",
        description="Uses the Wrap feature to adjust the position of the Drone object if it goes outside the defined range.",
        default=False
    )

    wrap_range: bpy.props.FloatVectorProperty(
        name="Wrap Range",
        description="The range of the space to wrap around, measured from the wrap pivot.",
        size=2,
        default=DEFAULT_WRAP_RANGE,
        get=range_getter("_wrap_range"),
        set=range_setter("_wrap_range")
    )

    wrap_split_axes: bpy.props.BoolProperty(
        name="Wrap Split Axes",
        description="Split the wrap range into separate ranges for each axis.",
        default=False
    )

    wrap_split_axes_x: bpy.props.BoolProperty(
        name="Wrap Split Axes X",
        description="Split the wrap range into separate ranges for each axis.",
        default=True
    )

    wrap_split_axes_y: bpy.props.BoolProperty(
        name="Wrap Split Axes Y",
        description="Split the wrap range into separate ranges for each axis.",
        default=True
    )

    wrap_split_axes_z: bpy.props.BoolProperty(
        name="Wrap Split Axes Z",
        description="Split the wrap range into separate ranges for each axis.",
        default=True
    )

    wrap_range_x: bpy.props.FloatVectorProperty(
        name="Wrap Range X",
        description="The range of the space to wrap around along the X-axis, measured from the wrap pivot.",
        size=2,
        default=DEFAULT_WRAP_RANGE,
        get=range_getter("_wrap_range_x"),
        set=range_setter("_wrap_range_x")
    )

    wrap_range_y: bpy.props.FloatVectorProperty(
        name="Wrap Range Y",
        description="The range of the space to wrap around along the Y-axis, measured from the wrap pivot.",
        size=2,
        default=DEFAULT_WRAP_RANGE,
        get=range_getter("_wrap_range_y"),
        set=range_setter("_wrap_range_y")
    )

    wrap_range_z: bpy.props.FloatVectorProperty(
        name="Wrap Range Z",
        description="The range of the space to wrap around along the Z-axis, measured from the wrap pivot.",
        size=2,
        default=DEFAULT_WRAP_RANGE,
        get=range_getter("_wrap_range_z"),
        set=range_setter("_wrap_range_z")
    )

    wrap_rotation: bpy.props.FloatVectorProperty(
        name="Wrap Rotation",
        description="Rotates the axes the wrap ranges are measured along, relative to world space, around the wrap pivot.",
        size=3,
        subtype='EULER',
        default=(0.0, 0.0, 0.0)
    )

    wrap_custom_pivot: bpy.props.BoolProperty(
        name="Wrap Custom Pivot",
        description="Measure the wrap ranges from the Pivot location. Otherwise they're measured from the selected object's origin, or the world origin if no object is selected.",
        default=False
    )

    wrap_pivot: bpy.props.FloatVectorProperty(
        name="Wrap Pivot",
        description="The location the wrap ranges are measured from when Pivot is on. In the scene's original coordinates, so it stays in place when Keep Drone at Origin moves the scene.",
        size=3,
        subtype='TRANSLATION',
        default=(0.0, 0.0, 0.0)
    )

    keep_drone_at_origin: bpy.props.BoolProperty(
        name="Keep Drone at Origin",
        description="Once the Drone_Camera gets far from the world origin for its scale, the objects in your scene are moved so the Drone_Camera is back at the origin. This allows the Drone to scale down to very small sizes without glitching. Objects are moved through their delta location, so their own location and animation are left alone, and Reset Locations moves them back exactly.",
        default=False
    )

    auto_scale: bpy.props.BoolProperty(
        name="Auto Scale",
        description="The Drone's scale will automatically adjust proportionally with its distance from the selected object's origin.",
        default=False
    )

    auto_scale_mesh: bpy.props.BoolProperty(
        name="Auto Scale Mesh",
        description="Measure the auto scale distance to the closest point on the selected object's mesh (faces, edges and vertices) instead of its origin.",
        default=False
    )

    auto_scale_split_axes: bpy.props.BoolProperty(
        name="Auto Scale Split Axes",
        description="Split the auto scale distance into separate distances for each axis.",
        default=False
    )

    auto_scale_x: bpy.props.BoolProperty(
        name="Auto Scale X",
        description="Scale along the X axis.",
        default=True
    )

    auto_scale_y: bpy.props.BoolProperty(
        name="Auto Scale Y",
        description="Scale along the Y axis.",
        default=True
    )

    auto_scale_z: bpy.props.BoolProperty(
        name="Auto Scale Z",
        description="Scale along the Z axis.",
        default=True
    )

    auto_scale_rotation: bpy.props.FloatVectorProperty(
        name="Auto Scale Rotation",
        description="Rotates the X, Y and Z axes the split auto scale distance is measured along, relative to world space.",
        size=3,
        subtype='EULER',
        default=(0.0, 0.0, 0.0)
    )

    tidal_lock: bpy.props.BoolProperty(
        name="Tidal Lock",
        description="The Drone turns with its flight around the selected object, so the object stays at the same angle in its view, like a moon that always shows its planet the same face. It is to Orbit what Auto Scale is to Zoom. The sticks still rotate the Drone on top of it.",
        default=False
    )

    tidal_lock_mesh: bpy.props.BoolProperty(
        name="Tidal Lock Mesh",
        description="Keep the closest point on the selected object's mesh at the same angle, instead of its origin. Follows the surface's curve, and doesn't turn over a flat floor, but over bumpy ground it turns with every bump.",
        default=False
    )

    tidal_lock_split_axes: bpy.props.BoolProperty(
        name="Tidal Lock Split Axes",
        description="Keep a line through the selected object's origin at the same angle, instead of the origin: the one along the axis left out of X, Y and Z. Around a cylinder, its axis. With one axis on, the direction never turns, and with all three it's the origin.",
        default=False
    )

    tidal_lock_x: bpy.props.BoolProperty(
        name="Tidal Lock X",
        description="Measure along the X axis.",
        default=True
    )

    tidal_lock_y: bpy.props.BoolProperty(
        name="Tidal Lock Y",
        description="Measure along the Y axis.",
        default=True
    )

    tidal_lock_z: bpy.props.BoolProperty(
        name="Tidal Lock Z",
        description="Measure along the Z axis.",
        default=True
    )

    tidal_lock_rotation: bpy.props.FloatVectorProperty(
        name="Tidal Lock Rotation",
        description="Rotates the X, Y and Z axes the split tidal lock direction is measured along, relative to world space.",
        size=3,
        subtype='EULER',
        default=(0.0, 0.0, 0.0)
    )

    horizon: bpy.props.BoolProperty(
        name="Horizon",
        description="Draw a horizon line on the HUD, level with what's down: the selected object (its origin, or with Mesh, its closest surface point, along Auto Scale's Split Axes), or with nothing selected, the way the Drone's gravity pulls. Hidden in zero gravity with nothing selected.",
        default=False
    )

    horizon_mesh: bpy.props.BoolProperty(
        name="Horizon Mesh",
        description="Level the horizon with the closest point on the selected object's mesh, instead of its origin. Follows the surface's curve, but on a faceted mesh (a low-poly sphere) it steps from face to face.",
        default=False
    )

    velocity: bpy.props.BoolProperty(
        name="Velocity",
        description="Mark where the drone is flying on the HUD: a ring with wings on the point its velocity points at, or flying backward, the same with an X on the point it's flying away from. Hidden while the drone is nearly still.",
        default=True
    )

    solid_fog: bpy.props.BoolProperty(
        name="Solid Fog",
        description="Draw the map's fog in Solid mode too, as in Rendered: far things fade into the fog color, at the scene's fog_distance times the drone's scale. Maps without fog, and the Earth's, are left as they are.",
        default=True
    )

    gravity_mesh: bpy.props.BoolProperty(
        name="Gravity Mesh",
        description="While flying with an object selected, the Drone's Gravity pulls toward it (negative pulls in): with Mesh, toward the closest point on its mesh, otherwise its origin, along Auto Scale's Split Axes. A sphere or surface map pulls to its ground. With nothing selected, it pulls along the Drone's Z axis as before",
        default=False
    )

    selected_obj: bpy.props.StringProperty(
        name="Selected Object",
        description="Enter an object name to serve as the selected object.",
        default=""
        # type=bpy.types.Object
    )

    auto_deselect: bpy.props.BoolProperty(
        name="Auto Deselect",
        description="Deselect everything when the click is let go, so nothing is selected between clicks, and apply the default_config text.",
        default=False
    )

    scale_lens_clip_with_camera: bpy.props.BoolProperty(
        name="Scale Lens Clip with Camera",
        description="Scale the lens clip lengths proportionally with the Drone_Camera.",
        default=True
    )

    default_clip_start: bpy.props.FloatProperty(
        name="Default Clip Start", 
        description="The default clip start length of the Drone_Camera object. This value also serves as the lower bound if Bounded is enabled. The default clip end length is 100,000 times the default clip start length.",
        default=0.5,
        min=0.000_001
    )

    maintain_ratio: bpy.props.BoolProperty(
        name="Maintain Ratio",
        description="Maintain the ratio between the lens clip lengths and the drone's scale. If scaling the drone would cause the lens clip lengths to exceed their min or max values, the drone will not be scaled.",
        default=True
    )

    orbit_axes: bpy.props.EnumProperty(
        name="Orbit Axes", 
        description="Orbit Axes", 
        items=[
            ('drone_axes', 'Drone Axes', 'Orbit the Drone along its local axes.'),
            ('selected_obj_axes', 'Object Axes', 'Orbit the Drone along the axes of the selected object.'),
            ('world_axes', 'World Axes', 'Orbit the Drone along the world axes.'),
        ],
        default='drone_axes'
    )

    bounded: bpy.props.BoolProperty(
        name="Bounded",
        description="Limit the range the clip start length can be adjusted. The lower bound is the default length, and the upper bound is the distance from the Drone_Camera object to the origin of the selected object. The clip end length will be scaled proportionally.",
        default=False
    )

    scale_lens_clip: bpy.props.BoolProperty(
        name="Scale Lens Clip",
        description="Scale the lens clip lengths proportionally with the Drone object.",
        default=True
    )

    scale_camera: bpy.props.BoolProperty(
        name="Scale Camera",
        description="Scale the Drone_Camera proportionally with the Drone object.",
        default=True
    )

    scale_lens_clip_zoom: bpy.props.BoolProperty(
        name="Scale Lens Clip",
        description="Scale the lens clip lengths proportionally with the Drone object with the Zoom feature.",
        default=True
    )

    scale_camera_zoom: bpy.props.BoolProperty(
        name="Scale Camera",
        description="Scale the Drone_Camera proportionally with the Drone object with the Zoom feature.",
        default=True
    )

    auto_brake: bpy.props.BoolProperty(
        name="Auto Brake",
        description="Keep brake on without input from the controller whenever throttle is zero.",
        default=False
    )
