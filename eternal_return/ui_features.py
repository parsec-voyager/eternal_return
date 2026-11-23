"""The side panel's parts for the add-on's own features: the UI (HUD) and Drone (physics) subpanels, the Controls panel's
Scaling Tools section, and how a feature's options are drawn under its toggle"""
import bpy
from . import flight


def feature_options(layout, enabled):
    """An indented column for a feature's options (its Mesh, Split Axes, ...), drawn under the feature's own toggle and
    greyed out while the feature is off, so it's clear which feature each option belongs to. On a row of their own,
    options next to several features' toggles read as belonging to any of them."""
    row = layout.row()
    row.separator(factor=2.0)
    column = row.column()
    column.active = enabled
    return column


class ER_PT_UI_HUD(bpy.types.Panel):
    """UI Subpanel: every display on the drone's HUD in one place: the scene's HUD settings (hud.py), the Horizon and
    Velocity add-on settings, and Solid Fog, which draws the map's fog over Solid mode's view (solid_sky.py)."""
    bl_label = "UI"
    bl_idname = "ER_PT_UI_HUD"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_context = "scene"
    bl_parent_id = "ER_PT_UI"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        control_settings = context.scene.control_settings
        layout = self.layout

        settings = context.scene.drone_hud

        def switches(names):
            """Two of the HUD's switches a row"""
            grid = layout.grid_flow(row_major=True, columns=2, even_columns=True, align=True)
            for name in names:
                grid.prop(settings, name)

        switches(("crosshairs", "ring"))
        layout.prop(settings, "gimbals")
        options = feature_options(layout, settings.gimbals)
        options.prop(settings, "gimbal_values", text="Gimbal Values")
        options.prop(settings, "deadzones", text="Deadzones")
        switches(("altitude", "speed", "scale", "scale_speed"))
        layout.prop(settings, "in_renders")

        layout.prop(control_settings, "horizon", text="Horizon")
        options = feature_options(layout, control_settings.horizon)
        options.prop(control_settings, "horizon_mesh", text="To Mesh")
        row = layout.row()
        row.prop(control_settings, "velocity", text="Velocity")
        row = layout.row()
        row.prop(control_settings, "solid_fog", text="Solid Fog")


def physics_prop(layout, context, name, text=None):
    """Draw the scene's physics setting name (physics.DronePhysics)"""
    if text is None:
        layout.prop(context.scene.drone_physics, name)
    else:
        layout.prop(context.scene.drone_physics, name, text=text)


class ER_PT_UI_Drone(bpy.types.Panel):
    """Drone Subpanel: the drone's physics (mass, gravity, drag, ...), the scene's (physics.DronePhysics)"""
    bl_label = "Drone"
    bl_idname = "ER_PT_UI_Drone"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_context = "scene"
    bl_parent_id = "ER_PT_UI"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        drone_physics = context.scene.drone_physics

        box = layout.box()
        box.label(text="Physics")
        column = box.column()
        physics_prop(column, context, "mass")
        physics_prop(column, context, "gravity")
        column.prop(drone_physics, "ground")
        options = feature_options(column, drone_physics.ground)
        physics_prop(options, context, "ground_height", text="Height")

        box = layout.box()
        box.label(text="Drag")
        column = box.column()
        physics_prop(column, context, "drag_coefficient")
        physics_prop(column, context, "reference_area")
        physics_prop(column, context, "air_density")

        box = layout.box()
        box.label(text="Camera")
        physics_prop(box, context, "camera_tilt")


def draw_scaling_tools(body, context):
    """The Controls panel's Scaling Tools section: each tool's toggle on a row of its own, with its options
    indented under it"""
    control_settings = context.scene.control_settings
    body.prop(control_settings, "loop", text="Loop")
    options = feature_options(body, control_settings.loop)
    options.prop(control_settings, "loop_factor", text="Loop Factor")

    body.prop(control_settings, "wrap", text="Wrap")
    options = feature_options(body, control_settings.wrap)
    row = options.row()
    row.prop(control_settings, "wrap_split_axes", text="Split Axes")
    if not control_settings.wrap_split_axes:
        row.prop(control_settings, "wrap_range", text="Range")
    else:
        axes = feature_options(options, True)
        for axis in "xyz":
            row = axes.row()
            row.prop(control_settings, f"wrap_split_axes_{axis}", text=axis.upper())
            row.prop(control_settings, f"wrap_range_{axis}", text="Range")
    row = options.row()
    row.prop(control_settings, "wrap_rotation", text="Rotation")
    row = options.row()
    row.prop(control_settings, "wrap_custom_pivot", text="Pivot")
    if control_settings.wrap_custom_pivot:
        row.prop(control_settings, "wrap_pivot", text="Location")

    row = body.row()
    row.prop(control_settings, "keep_drone_at_origin", text="Keep Drone at Origin")
    row.operator(
        flight.ER_OT_ResetLocations.bl_idname,
        text="Reset Locations",
        icon='LOOP_BACK'
    )

    # Not greyed out with Auto Scale off: its Mesh and Split Axes are also how the horizon, gravity and the
    # Altitude readout measure to the selected object
    body.prop(control_settings, "auto_scale", text="Auto Scale")
    options = feature_options(body, True)
    options.prop(control_settings, "auto_scale_mesh", text="To Mesh")
    options.prop(control_settings, "auto_scale_split_axes", text="Split Axes")
    if control_settings.auto_scale_split_axes:
        axes = feature_options(options, True)
        row = axes.row()
        row.prop(control_settings, "auto_scale_x", text="X")
        row.prop(control_settings, "auto_scale_y", text="Y")
        row.prop(control_settings, "auto_scale_z", text="Z")
        row = axes.row()
        row.prop(control_settings, "auto_scale_rotation", text="Rotation")

    body.prop(control_settings, "tidal_lock", text="Tidal Lock")
    options = feature_options(body, control_settings.tidal_lock)
    options.prop(control_settings, "tidal_lock_mesh", text="To Mesh")
    options.prop(control_settings, "tidal_lock_split_axes", text="Split Axes")
    if control_settings.tidal_lock_split_axes:
        axes = feature_options(options, True)
        row = axes.row()
        row.prop(control_settings, "tidal_lock_x", text="X")
        row.prop(control_settings, "tidal_lock_y", text="Y")
        row.prop(control_settings, "tidal_lock_z", text="Z")
        row = axes.row()
        row.prop(control_settings, "tidal_lock_rotation", text="Rotation")

    # The Drone's own Gravity (also in the Drone subpanel), which pulls toward the selected object while
    # flying. Off at zero.
    physics_prop(body, context, "gravity")
    options = feature_options(body, context.scene.drone_physics.gravity != 0.0)
    options.prop(control_settings, "gravity_mesh", text="To Mesh")

    row = body.row()
    sub = row.row(align=True)
    sub.prop(control_settings, "min_scale", text="Min Scale")
    sub.operator(
        flight.ER_OT_ResetMinScale.bl_idname,
        text="",
        icon='LOOP_BACK'
    )
    sub = row.row(align=True)
    sub.prop(control_settings, "max_scale", text="Max Scale")
    sub.operator(
        flight.ER_OT_ResetMaxScale.bl_idname,
        text="",
        icon='LOOP_BACK'
    )
