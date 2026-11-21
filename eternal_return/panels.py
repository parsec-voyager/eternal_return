"""The side panel (the 3D view's N sidebar, Eternal Return tab): Add Drone and the Fly toggle, then the UI and
Drone subpanels (ui_features.py), Controls, Record and Extras."""
import bpy

from . import controls
from . import flight
from . import ui_features

# Each control's box in the Controls panel: its icon (the title is the control's label)
CONTROL_ICONS = {"throttle": 'UV_SYNC_SELECT', "yaw": 'FORCE_MAGNETIC', "pitch": 'FORCE_MAGNETIC',
                 "roll": 'FORCE_MAGNETIC', "scale": 'UV_SYNC_SELECT', "zoom": 'UV_SYNC_SELECT', "orbit": 'FORCE_MAGNETIC',
                 "click": 'RADIOBUT_ON', "camera": 'UV_SYNC_SELECT', "lens_clip": 'UV_SYNC_SELECT', "brake": 'FORCE_DRAG'}
# The Scaling Tools section sits among the control boxes, after this one
SCALING_TOOLS_AFTER = "zoom"


class SidePanel:
    """Where every panel of the add-on lives"""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Eternal Return"


class SubPanel(SidePanel):
    bl_parent_id = "ER_PT_UI"
    bl_context = "scene"


class ER_PT_UI(SidePanel, bpy.types.Panel):
    """Main panel: add the drone, and fly it"""
    bl_label = "Eternal Return"
    bl_idname = "ER_PT_UI"

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.scale_y = 1.5
        row.operator("er.add_to_scene", text="Add Drone", icon='CON_CAMERASOLVER')
        row = layout.row()
        row.scale_y = 1.5
        if context.window_manager.er_fly_modal_running:
            row.operator("er.fly", text="STOP", icon='PAUSE')
        else:
            row.operator("er.fly", text="Start Flying", icon='PLAY')


# Each control's own rows under its Axis, Invert and Deadzone, drawn by draw_<kind>(box, scene)

def draw_throttle(box, scene):
    parameters = scene.control_settings
    row = box.row()
    row.prop(parameters, "max_thrust", text="Maximum Thrust")
    # In 3D mode the throttle's whole travel is used, reverse thrust below center, so it has no rest end to center
    if parameters.flight_mode != '3D':
        row.prop(parameters, "throttle_centered", text="Centered")


def draw_rates(control):
    def draw(box, scene):
        rates = scene.stick_rates
        row = box.row()
        row.prop(rates, f"{control}_center_sensitivity", text="Center Sensitivity")
        row.prop(rates, f"{control}_max_rate", text="Max Rate")
        row.prop(rates, f"{control}_expo", text="Expo")
    return draw


def draw_scale(box, scene):
    parameters = scene.control_settings
    box.prop(parameters, "scale_factor", text="Scale Factor")
    row = box.row()
    row.prop(parameters, "scale_lens_clip", text="Lens Clip")
    row.prop(parameters, "scale_camera", text="Camera")
    row.operator(flight.ER_OT_ResetScale.bl_idname, text="Reset", icon='LOOP_BACK')


def draw_zoom(box, scene):
    parameters = scene.control_settings
    box.prop(parameters, "zoom_factor", text="Zoom Factor")
    row = box.row()
    row.prop(parameters, "scale_lens_clip_zoom", text="Lens Clip")
    row.prop(parameters, "scale_camera_zoom", text="Camera")


def draw_orbit(box, scene):
    parameters = scene.control_settings
    row = box.row()
    row.prop(parameters, "orbit_centered", text="Centered")
    row.prop(parameters, "orbit_axes", text="Orbit Axes")


def draw_click(box, scene):
    parameters = scene.control_settings
    row = box.row()
    row.prop(parameters, "click_centered", text="Centered")
    row.prop(parameters, "auto_deselect", text="Auto Deselect")
    box.prop(parameters, "selected_obj", text="Click Override")


def draw_camera(box, scene):
    parameters = scene.control_settings
    box.prop(parameters, "camera_factor", text="Camera Factor")
    row = box.row()
    row.prop(parameters, "scale_lens_clip_with_camera", text="Lens Clip")
    row.operator(flight.ER_OT_ResetCamera.bl_idname, text="Reset", icon='LOOP_BACK')


def draw_lens_clip(box, scene):
    parameters = scene.control_settings
    box.prop(parameters, "lens_clip_factor", text="Lens Clip Factor")
    row = box.row()
    row.prop(parameters, "default_clip_start", text="Clip Start")
    row.operator(flight.ER_OT_ResetLensClip.bl_idname, text="Reset", icon='LOOP_BACK')
    row = box.row()
    row.prop(parameters, "maintain_ratio", text="Maintain Ratio")
    row.prop(parameters, "bounded", text="Bounded")


def draw_brake(box, scene):
    parameters = scene.control_settings
    box.prop(parameters, "brake_factor", text="Brake Factor")
    row = box.row()
    row.prop(parameters, "brake_centered", text="Centered")
    row.prop(parameters, "auto_brake", text="Auto Brake")


CONTROL_ROWS = {"throttle": draw_throttle, "yaw": draw_rates("yaw"), "pitch": draw_rates("pitch"),
                "roll": draw_rates("roll"), "scale": draw_scale, "zoom": draw_zoom, "orbit": draw_orbit,
                "click": draw_click, "camera": draw_camera, "lens_clip": draw_lens_clip, "brake": draw_brake}


def draw_control(column, scene, control):
    """A control's box: its title, the axis it reads with its invert and deadzone, then its own settings"""
    parameters = scene.control_settings
    box = column.box()
    box.label(text=controls.label(control), icon=CONTROL_ICONS[control])
    row = box.row()
    row.prop(scene.control_axes, control, text="Axis")
    row.prop(parameters, f"{control}_invert", text="Invert")
    row.prop(parameters, f"{control}_deadzone", text="Deadzone")
    CONTROL_ROWS[control](box, scene)


def draw_scaling_tools(column, context):
    """The Scaling Tools section, closed until opened, its body drawn by ui_features.py"""
    box = column.box()
    header, body = box.panel("ER_scaling_tools", default_closed=True)
    header.label(text="Scaling Tools", icon='LOOP_BACK')
    if body is not None:
        ui_features.draw_scaling_tools(body, context)


class ER_PT_UI_Controls(SubPanel, bpy.types.Panel):
    """Controls Subpanel: which axis drives each control and how, the scaling tools, and what the controls read"""
    bl_label = "Controls"
    bl_idname = "ER_PT_UI_Controls"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        parameters = scene.control_settings

        box = layout.box()
        box.label(text="Controls Settings in the Preferences", icon='PREFERENCES')
        row = box.row()
        row.operator("er.store_controls", text="Store")
        row.operator("er.load_controls", text="Load")

        row = layout.row()
        row.scale_y = 1.5
        split = row.split(factor=0.5)
        # No label: in half the panel's width it was cut to "Fly M...", and the mode's name says what it is
        split.prop(parameters, "flight_mode", text="")
        modes = split.row()
        if parameters.flight_mode in ('HORIZON', 'ANGLE'):
            modes.prop(parameters, "level_strength", text="Auto-Level Strength")
            if parameters.flight_mode == 'ANGLE':
                modes.prop(parameters, "angle_limit", text="Angle Limit")
        else:
            # Keeps the layout from jumping when the mode changes
            modes.label(text="")

        column = layout.column(align=True)
        for control in controls.CONTROLS:
            draw_control(column, scene, control)
            if control == SCALING_TOOLS_AFTER:
                draw_scaling_tools(column, context)

        layout.label(text="What the Controls Do", icon='GRAPH')
        grid = layout.box().grid_flow(row_major=True, columns=2, even_columns=True)
        for control in controls.CONTROLS:
            grid.prop(scene.control_outputs, control)

        layout.label(text="Controller Axes", icon='CONSOLE')
        grid = layout.box().grid_flow(row_major=True, columns=4, even_columns=True)
        for axis in range(controls.NUM_AXES):
            grid.prop(scene.controller_axes, f"axis_{axis}", text=f"Axis {axis}")


class ER_PT_UI_Record(SubPanel, bpy.types.Panel):
    """Record Subpanel: record a flight's inputs or its camera, and bake the replay into an editable camera"""
    bl_label = "Record"
    bl_idname = "ER_PT_UI_Record"

    def draw(self, context):
        layout = self.layout
        flight_options = context.scene.flight_options
        row = layout.row()
        row.prop(flight_options, "record", text="Record",
                 icon='RECORD_ON' if flight_options.record else 'RECORD_OFF')
        row.operator("er.bake_animation", text="Bake Animation", icon='AUTOMERGE_OFF')
        row = layout.row()
        row.prop(flight_options, "record_camera", text="Record Camera")
        row.prop(flight_options, "record_obj", text="Camera Object")


class ER_PT_UI_Extras(SubPanel, bpy.types.Panel):
    """Extras Subpanel: the drone's sound"""
    bl_label = "Extras"
    bl_idname = "ER_PT_UI_Extras"

    def draw(self, context):
        layout = self.layout
        flight_options = context.scene.flight_options
        layout.prop(flight_options, "drone_sound", text="Drone Sound")
        layout.prop(flight_options, "volume", text="Sound Volume", slider=True)


# In registration order, which is also their order on screen
PANELS = (ER_PT_UI, ui_features.ER_PT_UI_HUD, ui_features.ER_PT_UI_Drone, ER_PT_UI_Controls,
          ER_PT_UI_Record, ER_PT_UI_Extras)
