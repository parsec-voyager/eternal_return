"""The drone's HUD: what it shows (DroneHUD, the scene's settings), and its shapes and text, worked out here once for
every way it's shown. hud_draw.py draws them in the viewport looking through the drone camera.

They're in the HUD's space: x right, y up, z ahead of the camera, in HUD units, 0.1 of which ahead is the camera's
clip start, whatever the drone's scale, so the HUD sits just past the lens clip and is as big on screen at any scale:
- the ring round the view's center, and the crosshairs in it (off unless asked for)
- the gimbals: a square under the ring for each stick, a dot in it where the stick is, with Gimbal Values, the
  sticks' positions in percent next to them (axes 0-3 as the controller reads them, read_gimbal_inputs(): a radio's
  AETR, a gamepad's mapped the same way, so the left square is throttle up and yaw across, the right pitch and roll),
  and with Deadzones, the stick positions the deadzone of the control on each axis takes in, shaded
  (deadzone_band())
- the readouts, bottom right: Altitude, Speed, Scale and Scale Speed, measured by the flight (odometer.py), written
  out by readouts.py
- and drawn by hud_draw.py from their own modules, the horizon (horizon.py) and velocity marker (velocity.py).

A recording keeps what the HUD showed on each frame (recorded(), physics.Recording.key_hud()), so its replay shows the
same HUD, played back or rendered (replay()).

Until stage C (2026-11-20) the HUD was DroneCam's: clip_start, an object under the camera appended from DroneCam.blend,
its geometry nodes making all of this as meshes, its switches the modifier's inputs, scaled by the add-on to stay at the
lens clip. Its letters were made again every frame (about 1 ms of every frame, readouts.py's holds made it less), and
the flight's own readouts had to be laid over its own on a third object. migrate() takes it out of a scene from before,
keeping its switches as the scene's settings, at the scene's first flight like the rig (rig.migrate()); until then
it's the HUD shown, and none is drawn. Its places and sizes are kept here, measured from it.
"""
import math

import bpy
import numpy as np

from . import horizon
from . import physics
from . import readouts
from . import rig
from . import velocity

# HUD units ahead of the camera of its clip start
CLIP_START_DEPTH = 0.1
# Where the ring, crosshairs and gimbals are: a little past the clip start, which right at it cut them off now and then
# (they flickered); in front of the velocity marker and horizon (velocity.DEPTH, horizon.DEPTH). The deadzones and the
# gimbals' lines and dots on their squares, a little nearer still.
SHAPE_DEPTH = 0.1008
BAND_DEPTH = 0.1007
MARK_DEPTH = 0.1006

RING_RADIUS = 0.01
# Lines' thicknesses are given by half, as the old HUD's tubes' radii were
RING_TUBE = 0.0008
RING_SEGMENTS = 96
# The crosshairs: four spikes from the center, as wide as this at their tips
CROSSHAIR_LENGTH = 0.0065
CROSSHAIR_TIP_WIDTH = 0.00172
# The gimbals' squares: their centers (the left one's; the right one mirrors it) and half their side
GIMBAL_CENTER = (-0.0227, -0.0385)
GIMBAL_HALF = 0.0108
# The square's middle lines, and the dot's outline
GIMBAL_LINE = 0.00019
GIMBAL_DOT_HALF = 0.00125
# The values' letters, and how far from their square
VALUE_SIZE = 0.0065
VALUE_GAP = 0.0014
# The readouts: right-aligned on x, the first one's baseline at y, each next one a line further down
READOUTS_RIGHT = 0.17
READOUTS_TOP = -0.0709
READOUTS_LINE = -0.0092
READOUTS_SIZE = 0.0070854
TEXT_DEPTH = 0.101

# The settings by the old HUD's switches (its modifier inputs' names), for migrate()
SWITCHES = {"Crosshairs": "crosshairs", "Ring": "ring", "Gimbals": "gimbals", "Gimbal Values": "gimbal_values",
            "Altitude": "altitude", "Speed": "speed", "Scale": "scale", "Scale Speed": "scale_speed"}
# The readouts, top to bottom, by their settings
LINES = (("Altitude", "altitude"), ("Speed", "speed"), ("Scale", "scale"), ("Scale Speed", "scale_speed"))
LEGACY_NAME = "clip_start"
# The controls a stick axis can be mapped to (controls.py's controls_axis_mapping), and those that go one way only,
# from 0 (controls.convert())
CONTROLS = ("throttle", "yaw", "pitch", "roll", "scale", "zoom", "orbit", "click", "camera", "lens_clip", "brake")
UNIDIRECTIONAL = ("throttle", "orbit", "click", "brake")
# How a recording keys a readout it didn't have (unread), and a length or speed of 0 (log10 None)
UNREAD = -1.0e30
ZERO = -1.0e20

# What the HUD shows: the sticks' positions (axes 0-3, -1 to 1), the readouts' numbers as the flight measured them
# (set_readings()), and their text by LINES name. Kept after a flight, as the old HUD kept its inputs.
axes = (0.0, 0.0, 0.0, 0.0)
readings = {}
lines = {}
# The controller's axis each gimbal axis shows, and its sign there, as the last flight read them
# (gamepad.gimbal_axes()): a radio's own axes 0-3, a gamepad's sticks wherever SDL's mapping has them
gimbal_axes = ((0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0))


def tag_redraw(_settings=None, context=None):
    """Draw the HUD again in every 3D view: a setting or value changed with nothing else redrawing them"""
    for window in getattr(bpy.context.window_manager, "windows", ()):
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


class DroneHUD(bpy.types.PropertyGroup):
    """The HUD's switches, the scene's. Read once from DroneCam's HUD in a scene from before (migrate())."""
    crosshairs: bpy.props.BoolProperty(name="Crosshairs", description="Crosshairs in the ring", default=False,
                                       update=tag_redraw)
    ring: bpy.props.BoolProperty(name="Ring", description="A ring round the view's center", default=True,
                                 update=tag_redraw)
    gimbals: bpy.props.BoolProperty(name="Gimbals", description="Where the controller's sticks are", default=True,
                                    update=tag_redraw)
    gimbal_values: bpy.props.BoolProperty(name="Gimbal Values", description="The sticks' positions, in percent",
                                          default=True, update=tag_redraw)
    deadzones: bpy.props.BoolProperty(name="Deadzones", description="Shade each stick's deadzone on the gimbals",
                                      default=False, update=tag_redraw)
    altitude: bpy.props.BoolProperty(name="Altitude", description="The drone's height over its ground, in meters",
                                     default=True, update=tag_redraw)
    speed: bpy.props.BoolProperty(name="Speed", description="The drone's speed, in meters a second", default=True,
                                  update=tag_redraw)
    scale: bpy.props.BoolProperty(name="Scale", description="The drone's scale", default=True, update=tag_redraw)
    scale_speed: bpy.props.BoolProperty(name="Scale Speed", description="How fast the drone's scale changes",
                                        default=False, update=tag_redraw)
    in_renders: bpy.props.BoolProperty(
        name="In Renders", description="Show the HUD in renders through the drone camera too, as objects under it "
                                       "(a recorded flight's HUD frame by frame). Locks the interface while rendering",
        default=False, update=lambda settings, context: in_renders_changed(settings.id_data))
    hud_read: bpy.props.BoolProperty(
        name="Read from HUD", description="The switches were read from DroneCam's HUD once (migrate())", default=False)


def in_renders_changed(scene):
    from . import hud_render
    hud_render.in_renders_changed(scene)


def legacy_hud(camera_obj):
    """DroneCam's HUD under the camera (clip_start), or None"""
    if camera_obj is None:
        return None
    return next((child for child in camera_obj.children
                 if child.name.split(".")[0] == LEGACY_NAME and child.type == 'MESH'
                 and any(m.type == 'NODES' for m in child.modifiers)), None)


def migrate(scene, camera_obj):
    """Take DroneCam's HUD out from under the camera, its switches kept as the scene's settings (the first time).
    Returns whether there was one."""
    legacy = legacy_hud(camera_obj)
    if legacy is None:
        return False
    settings = scene.drone_hud
    modifier = next((m for m in legacy.modifiers if m.type == 'NODES' and m.node_group), None)
    if modifier is not None and not settings.hud_read:
        for item in modifier.node_group.interface.items_tree:
            if item.item_type == 'SOCKET' and item.in_out == 'INPUT' and item.name in SWITCHES:
                socket = getattr(modifier.properties.inputs, item.identifier, None)
                if socket is not None:
                    setattr(settings, SWITCHES[item.name], bool(socket.value))
        settings.hud_read = True
    delete_legacy(legacy)
    return True


def delete_legacy(legacy):
    """Delete DroneCam's HUD object, with the objects the add-on put under it (the flight's readouts, and the horizon
    and velocity marker from before they were drawn), and the node groups only they used. Anything else under it
    stays, under the camera, where it was."""
    ours = [legacy] + [obj for obj in legacy.children_recursive
                       if horizon.is_horizon(obj) or velocity.is_velocity(obj) or readouts.is_readouts(obj)]
    groups = []
    for obj in ours:
        for modifier in obj.modifiers:
            if modifier.type == 'NODES' and modifier.node_group:
                groups += [group for group in rig.rig_groups(modifier) if group not in groups]
    for child in legacy.children:
        if child not in ours:
            world = child.matrix_world.copy()
            child.parent = legacy.parent
            child.matrix_world = world
    for obj in reversed(ours):
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    rig.remove_groups(groups)


def shown_lines(scene):
    """The readouts the scene's switches show, by LINES name"""
    settings = scene.drone_hud
    return [name for name, prop in LINES if getattr(settings, prop)]


def set_readings(values):
    """Show the flight's readouts, by LINES name: Altitude as (log10 meters, or None at 0, sign), Speed as log10
    meters a second (None at 0), Scale and Scale Speed as log10 of the scale and of its change a second"""
    global readings, lines
    if values == readings:
        return
    readings = dict(values)
    texts = {}
    if "Altitude" in values:
        altitude, sign = values["Altitude"]
        texts["Altitude"] = "Altitude: " + readouts.length(altitude, sign=sign)
    if "Speed" in values:
        texts["Speed"] = "Speed: " + readouts.length(values["Speed"], "m/s")
    if "Scale" in values:
        texts["Scale"] = "Scale: " + readouts.factor(values["Scale"])
    if "Scale Speed" in values:
        texts["Scale Speed"] = "Scale Speed: " + readouts.factor(values["Scale Speed"], "x/s")
    lines = texts
    tag_redraw()


def recorded():
    """What the HUD shows, as a recording keys it (physics.HUD_CHANNELS)"""
    def number(name):
        if name not in readings:
            return UNREAD
        return ZERO if readings[name] is None else readings[name]
    altitude, sign = readings.get("Altitude", (UNREAD, 1.0))
    return {"hud_axes": axes, "hud_altitude": (ZERO if altitude is None else altitude, sign),
            "hud_speed": (number("Speed"),), "hud_scale": (number("Scale"),),
            "hud_scale_speed": (number("Scale Speed"),)}


def show_recorded(values):
    """Show what a recording keyed (recorded())"""
    def number(value):
        return None if value <= ZERO / 10.0 else value
    shown = {}
    altitude, sign = values["hud_altitude"]
    if altitude > UNREAD / 10.0:
        shown["Altitude"] = (number(altitude), 1.0 if sign >= 0.0 else -1.0)
    for name, channel in (("Speed", "hud_speed"), ("Scale", "hud_scale"), ("Scale Speed", "hud_scale_speed")):
        if values[channel][0] > UNREAD / 10.0:
            shown[name] = number(values[channel][0])
    set_axes(values["hud_axes"])
    set_readings(shown)


def replay(scene):
    """Show what the HUD showed at the scene's current frame of the Drone's recording, while not flying (as
    physics.replay() shows the drone), held before its start and after its end"""
    if physics.is_flying():
        return
    references = getattr(scene, "references", None)
    drone_obj = references.drone_obj if references else None
    take = physics.Recording.of(drone_obj) if drone_obj else None
    if take is None:
        return
    values = take.hud_at(min(max(scene.frame_current, take.start_frame + 1), take.end_frame()))
    if values is not None:
        show_recorded(values)


@bpy.app.handlers.persistent
def replay_handler(scene, depsgraph=None):
    replay(scene)


def set_axes(values):
    global axes
    values = tuple(float(value) for value in values)
    if values != axes:
        axes = values
        tag_redraw()


def gimbal_controls(scene):
    """The control mapped to the controller axis each gimbal axis (0-3) shows, with its sign there, as
    (control or None, sign)"""
    mapping = scene.controls_axis_mapping
    return tuple((next((control for control in CONTROLS if getattr(mapping, control) == str(axis)), None), sign)
                 for axis, sign in gimbal_axes)


def deadzone_band(scene, control, sign=1.0):
    """The stick positions (-1 to 1, times sign) control's deadzone takes in, as (low, high), or None without one.
    As the controls apply it (controls.convert(), the core's sticks): about the center, but
    for a one-way control on the whole stick (not centered; a throttle not in 3D), which goes from 0 at one end, so its
    deadzone is there."""
    if control is None:
        return None
    controls = scene.controls_parameters
    deadzone = min(getattr(controls, f"{control}_deadzone") / 100.0, 1.0)
    if deadzone <= 0.0:
        return None
    whole_stick = (control in UNIDIRECTIONAL and not getattr(controls, f"{control}_centered")
                   and not (control == "throttle" and controls.flying_mode == "3D"))
    if whole_stick:
        edge = 2.0 * deadzone - 1.0
        low, high = (-edge, 1.0) if getattr(controls, f"{control}_invert") else (-1.0, edge)
    else:
        low, high = -deadzone, deadzone
    return (low, high) if sign > 0.0 else (-high, -low)


def square(center, half, z):
    """A square's corners, counterclockwise from bottom left"""
    x, y = center
    return np.array([(x - half, y - half, z), (x + half, y - half, z), (x + half, y + half, z), (x - half, y + half, z)])


def shapes(settings):
    """The HUD's shapes in its space, by what they're drawn in, in this order: "fills", triangles (n, 3, 3) in the
    HUD's color; "bands", triangles, the deadzones on the gimbals' squares, in a shade between it and the marks';
    "lines", polylines (points, half their thickness) in its color; "marks", polylines over the gimbals' fill, in a
    color that stands out on it; "texts", (text, anchor, size, align: LEFT/CENTER/RIGHT, on: BASELINE/MIDDLE)"""
    z = SHAPE_DEPTH
    fills, bands, drawn, marks, texts = [], [], [], [], []
    if settings.ring:
        angle = np.linspace(0.0, 2.0 * math.pi, RING_SEGMENTS + 1)
        drawn.append((np.stack([RING_RADIUS * np.cos(angle), RING_RADIUS * np.sin(angle), np.full_like(angle, z)],
                               axis=1), RING_TUBE))
    if settings.crosshairs:
        half = CROSSHAIR_TIP_WIDTH / 2.0
        for turn in range(4):
            c, s = math.cos(turn * math.pi / 2.0), math.sin(turn * math.pi / 2.0)
            spike = [(0.0, 0.0), (-half, CROSSHAIR_LENGTH), (half, CROSSHAIR_LENGTH)]
            fills.append([(c * x - s * y, s * x + c * y, z) for x, y in spike])
    if settings.gimbals or settings.gimbal_values:
        travel = GIMBAL_HALF - GIMBAL_DOT_HALF
        controls = gimbal_controls(settings.id_data) if settings.gimbals and settings.deadzones else None
        # (side, the axis across, the axis up): the left square's sticks are axes 3 and 2, the right's 0 and 1
        for side, across, up in ((-1.0, 3, 2), (1.0, 0, 1)):
            cx, cy = side * -GIMBAL_CENTER[0], GIMBAL_CENTER[1]
            if settings.gimbals:
                corners = square((cx, cy), GIMBAL_HALF, z)
                fills += [corners[[0, 1, 2]], corners[[0, 2, 3]]]
                if controls is not None:
                    bands += deadzone_bands(settings.id_data, (cx, cy), controls[across], controls[up], travel)
                m = MARK_DEPTH
                marks.append((np.array([(cx, cy - GIMBAL_HALF, m), (cx, cy + GIMBAL_HALF, m)]), GIMBAL_LINE))
                marks.append((np.array([(cx - GIMBAL_HALF, cy, m), (cx + GIMBAL_HALF, cy, m)]), GIMBAL_LINE))
                dot = (cx + travel * clamp(axes[across]), cy + travel * clamp(axes[up]))
                marks.append((square(dot, GIMBAL_DOT_HALF, m)[[0, 1, 2, 3, 0]], GIMBAL_LINE))
            if settings.gimbal_values:
                texts.append((percent(axes[up]), (cx, cy + GIMBAL_HALF + VALUE_GAP, TEXT_DEPTH), VALUE_SIZE,
                              'CENTER', 'BASELINE'))
                texts.append((percent(axes[across]), (cx + side * (GIMBAL_HALF + VALUE_GAP), cy, TEXT_DEPTH),
                              VALUE_SIZE, 'LEFT' if side > 0 else 'RIGHT', 'MIDDLE'))
    for row, (name, prop) in enumerate(LINES):
        if getattr(settings, prop) and lines.get(name):
            texts.append((lines[name], (READOUTS_RIGHT, READOUTS_TOP + row * READOUTS_LINE, TEXT_DEPTH),
                          READOUTS_SIZE, 'RIGHT', 'BASELINE'))
    return {"fills": np.array(fills, dtype=np.float64).reshape(-1, 3, 3),
            "bands": np.array(bands, dtype=np.float64).reshape(-1, 3, 3), "lines": drawn, "marks": marks,
            "texts": texts}


def deadzone_bands(scene, center, across, up, travel):
    """A gimbal square's deadzones as triangles: a band the square's height for the control across, its width for
    the control up, where the dot is while the stick is in it (a band reaching the stick's end goes to the square's
    edge)"""
    def place(position):
        return math.copysign(GIMBAL_HALF, position) if abs(position) >= 1.0 else travel * position
    cx, cy = center
    triangles = []
    for (control, sign), sideways in ((across, True), (up, False)):
        band = deadzone_band(scene, control, sign)
        if band is None:
            continue
        low, high = place(band[0]), place(band[1])
        if sideways:
            x0, x1, y0, y1 = cx + low, cx + high, cy - GIMBAL_HALF, cy + GIMBAL_HALF
        else:
            x0, x1, y0, y1 = cx - GIMBAL_HALF, cx + GIMBAL_HALF, cy + low, cy + high
        corners = np.array([(x0, y0, BAND_DEPTH), (x1, y0, BAND_DEPTH), (x1, y1, BAND_DEPTH), (x0, y1, BAND_DEPTH)])
        triangles += [corners[[0, 1, 2]], corners[[0, 2, 3]]]
    return triangles


def clamp(value):
    return max(-1.0, min(1.0, value))


def percent(value):
    return str(int(round(100.0 * clamp(value))))


def register():
    bpy.utils.register_class(DroneHUD)
    bpy.types.Scene.drone_hud = bpy.props.PointerProperty(type=DroneHUD)
    if replay_handler not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(replay_handler)


def unregister():
    if replay_handler in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(replay_handler)
    del bpy.types.Scene.drone_hud
    bpy.utils.unregister_class(DroneHUD)
