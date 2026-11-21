"""The side panel's parts for the add-on's own features: the UI (HUD) and Drone (physics) subpanels, the Controls panel's
Scaling Tools section, and how a feature's options are drawn under its toggle"""
import bpy
from . import flight
from . import hud
from . import rig


def feature_options(layout, enabled):
    """An indented column for a feature's options (its Mesh, Split Axes, ...), drawn under the feature's own toggle and
    greyed out while the feature is off, so it's clear which feature each option belongs to. On a row of their own,
    options next to several features' toggles read as belonging to any of them."""
    row = layout.row()
    row.separator(factor=2.0)
    column = row.column()
    column.active = enabled
    return column


def legacy_hud_modifier(context):
    """The geometry nodes modifier of DroneCam's HUD (clip_start) under the scene drone's camera, in a scene from before
    stage C that hasn't flown since (hud.migrate()), or None"""
    camera = context.scene.references.camera_obj
    legacy = hud.legacy_hud(camera)
    if legacy is None or context.scene.drone_hud.hud_read:
        return None
    return next((m for m in legacy.modifiers if m.type == 'NODES' and m.node_group), None)


class ER_PT_UI_HUD(bpy.types.Panel):
    """UI Subpanel: every display on the drone's HUD in one place: the scene's HUD settings (hud.py), the Horizon and
    Velocity add-on settings, and Solid Fog, which draws the map's fog over Solid mode's view (solid_sky.py). In a scene
    from before stage C that hasn't flown since, the switches are DroneCam's HUD's own inputs, which its first flight
    reads into the settings."""
    bl_label = "UI"
    bl_idname = "ER_PT_UI_HUD"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_context = "scene"
    bl_parent_id = "ER_PT_UI"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        controls_parameters = context.scene.controls_parameters
        layout = self.layout

        settings = context.scene.drone_hud
        modifier = legacy_hud_modifier(context)
        inputs = {} if modifier is None else {
            item.name: item.identifier for item in modifier.node_group.interface.items_tree
            if item.item_type == 'SOCKET' and item.in_out == 'INPUT'}

        def switches(names):
            """Two switches a row, the scene's settings or DroneCam's HUD's inputs"""
            grid = layout.grid_flow(row_major=True, columns=2, even_columns=True, align=True)
            for name in names:
                if modifier is None:
                    grid.prop(settings, hud.SWITCHES[name])
                else:
                    socket = getattr(modifier.properties.inputs, inputs.get(name, ""), None)
                    if socket is not None:
                        grid.prop(socket, "value", text=name)

        switches(("Crosshairs", "Ring", "Gimbals", "Gimbal Values"))
        options = feature_options(layout, settings.gimbals if modifier is None else True)
        options.prop(settings, "deadzones", text="Deadzones")
        switches(("Altitude", "Speed", "Scale", "Scale Speed"))
        layout.prop(settings, "in_renders")

        layout.prop(controls_parameters, "horizon", text="Horizon")
        options = feature_options(layout, controls_parameters.horizon)
        options.prop(controls_parameters, "horizon_mesh", text="To Mesh")
        row = layout.row()
        row.prop(controls_parameters, "velocity", text="Velocity")
        row = layout.row()
        row.prop(controls_parameters, "solid_fog", text="Solid Fog")


def physics_setting(context, name):
    """Where the panel edits a physics setting: the scene's (physics.DronePhysics), or until the scene's first flight
    takes DroneCam's rig out (rig.migrate()), the rig's modifier input the setting will be read from then, so an edit
    made before isn't lost to it. As (data, property name), or None for a setting the rig doesn't have."""
    drone_physics = context.scene.drone_physics
    drone_obj = context.scene.references.drone_obj
    if drone_physics.rig_read or not rig.has_rig(drone_obj):
        return drone_physics, name
    socket = getattr(drone_obj.modifiers[rig.DRONE_MODIFIER].properties.inputs, rig.SOCKETS.get(name, ""), None)
    return (socket, "value") if socket is not None else None


def physics_prop(layout, context, name, text=None):
    """Draw the physics setting name (physics_setting()), labelled as the scene's setting is"""
    setting = physics_setting(context, name)
    if setting is not None:
        label = text or context.scene.drone_physics.bl_rna.properties[name].name
        layout.prop(*setting, text=label)


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
        if physics_setting(context, "ground") is not None:
            column.prop(drone_physics, "ground")
            options = feature_options(column, drone_physics.ground)
            physics_prop(options, context, "ground_height", text="Height")
        else:
            # DroneCam's rig: one Ground Height, the lowest float32 for none
            physics_prop(column, context, "ground_height")

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
    controls_parameters = context.scene.controls_parameters
    body.prop(controls_parameters, "loop", text="Loop")
    options = feature_options(body, controls_parameters.loop)
    options.prop(controls_parameters, "loop_factor", text="Loop Factor")

    body.prop(controls_parameters, "wrap", text="Wrap")
    options = feature_options(body, controls_parameters.wrap)
    row = options.row()
    row.prop(controls_parameters, "wrap_split_axes", text="Split Axes")
    if not controls_parameters.wrap_split_axes:
        row.prop(controls_parameters, "wrap_range", text="Range")
    else:
        axes = feature_options(options, True)
        for axis in "xyz":
            row = axes.row()
            row.prop(controls_parameters, f"wrap_split_axes_{axis}", text=axis.upper())
            row.prop(controls_parameters, f"wrap_range_{axis}", text="Range")
    row = options.row()
    row.prop(controls_parameters, "wrap_rotation", text="Rotation")
    row = options.row()
    row.prop(controls_parameters, "wrap_custom_pivot", text="Pivot")
    if controls_parameters.wrap_custom_pivot:
        row.prop(controls_parameters, "wrap_pivot", text="Location")

    row = body.row()
    row.prop(controls_parameters, "keep_drone_at_origin", text="Keep Drone at Origin")
    row.operator(
        flight.ER_OT_ResetLocations.bl_idname,
        text="Reset Locations",
        icon='LOOP_BACK'
    )

    # Not greyed out with Auto Scale off: its Mesh and Split Axes are also how the horizon, gravity and the
    # Altitude readout measure to the selected object
    body.prop(controls_parameters, "auto_scale", text="Auto Scale")
    options = feature_options(body, True)
    options.prop(controls_parameters, "auto_scale_mesh", text="To Mesh")
    options.prop(controls_parameters, "auto_scale_split_axes", text="Split Axes")
    if controls_parameters.auto_scale_split_axes:
        axes = feature_options(options, True)
        row = axes.row()
        row.prop(controls_parameters, "auto_scale_x", text="X")
        row.prop(controls_parameters, "auto_scale_y", text="Y")
        row.prop(controls_parameters, "auto_scale_z", text="Z")
        row = axes.row()
        row.prop(controls_parameters, "auto_scale_rotation", text="Rotation")

    body.prop(controls_parameters, "tidal_lock", text="Tidal Lock")
    options = feature_options(body, controls_parameters.tidal_lock)
    options.prop(controls_parameters, "tidal_lock_mesh", text="To Mesh")
    options.prop(controls_parameters, "tidal_lock_split_axes", text="Split Axes")
    if controls_parameters.tidal_lock_split_axes:
        axes = feature_options(options, True)
        row = axes.row()
        row.prop(controls_parameters, "tidal_lock_x", text="X")
        row.prop(controls_parameters, "tidal_lock_y", text="Y")
        row.prop(controls_parameters, "tidal_lock_z", text="Z")
        row = axes.row()
        row.prop(controls_parameters, "tidal_lock_rotation", text="Rotation")

    # The Drone's own Gravity (also in the Drone subpanel), which pulls toward the selected object while
    # flying. Off at zero.
    physics_prop(body, context, "gravity")
    setting = physics_setting(context, "gravity")
    options = feature_options(body, setting is None or getattr(*setting) != 0.0)
    options.prop(controls_parameters, "gravity_mesh", text="To Mesh")

    row = body.row()
    sub = row.row(align=True)
    sub.prop(controls_parameters, "min_scale", text="Min Scale")
    sub.operator(
        flight.ER_OT_ResetMinScale.bl_idname,
        text="",
        icon='LOOP_BACK'
    )
    sub = row.row(align=True)
    sub.prop(controls_parameters, "max_scale", text="Max Scale")
    sub.operator(
        flight.ER_OT_ResetMaxScale.bl_idname,
        text="",
        icon='LOOP_BACK'
    )
