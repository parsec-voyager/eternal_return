"""The HUD drawn by draw handlers, in a view looking through the drone camera: its ring, crosshairs, gimbals and text
(hud.py), whenever the view looks through it, and while flying, the horizon (horizon.py) and velocity marker
(velocity.py).

The horizon and velocity marker say where something is in the drone camera's view: the way down, the way the drone
flies. As geometry nodes objects under clip_start (until 2026-11-17), each read its own evaluated transform to be exact
for the frame drawn, which made the depsgraph build them again at every evaluation that moved the camera, twice a frame:
about 2.5 ms a frame together, nearly all of it the node graphs' overhead, on every map (2026-11-17, measured on the
Earth over Tokyo: 49 -> 54 fps with them hidden). A draw handler has the view matrix Blender draws that frame with, so
it's exact for every draw, and costs no evaluation at all: just a few hundred points worked out and drawn. The rest of
the HUD followed in stage C (2026-11-20), DroneCam's clip_start object, whose letters were made again every frame.

Everything is drawn in the camera's own space (the view matrix left out), so however far the camera is from the world's
origin, float32 never shakes it. The HUD's plane is a fixed depth ahead, in HUD units (hud.CLIP_START_DEPTH of them
ahead is the camera's clip start), taken from the view's own clip start, the near plane of its projection. Its shapes
are depth tested against what's drawn, and so hidden by anything right at the clip start; the text is drawn over
everything, in pixels (POST_PIXEL), at the same places.

Lines are flat, as thick on screen as the old HUD's tubes, in a color like the old HUD's in each shading mode (COLORS).
The dots are filled, with a thin line round their edge, which smooths it.
"""
import blf
import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from . import background
from . import horizon
from . import hud
from . import velocity

# Like the old HUD, which was lit grey in Solid mode, and dark in Material Preview and Rendered (no light reached it
# there); the gimbals' lines and dots in the other, to stand out on their squares, and their deadzones in between
COLORS = {'SOLID': (0.72, 0.72, 0.72, 1.0), 'WIREFRAME': (0.72, 0.72, 0.72, 1.0)}
DARK = (0.02, 0.02, 0.02, 1.0)
LIGHT_MARKS = (0.25, 0.25, 0.25, 1.0)
DARK_MARKS = (0.72, 0.72, 0.72, 1.0)
LIGHT_BANDS = (0.52, 0.52, 0.52, 1.0)
DARK_BANDS = (0.24, 0.24, 0.24, 1.0)

# A dot's edge line, px
DOT_EDGE = 1.5
DOT_RESOLUTION = 32
FONT = 0

_shaders = {}


def shader(name):
    if name not in _shaders:
        _shaders[name] = gpu.shader.from_builtin(name)
    return _shaders[name]


def colors(context):
    """The HUD's color, its marks' and its deadzones' in the view's shading mode"""
    color = COLORS.get(context.space_data.shading.type)
    return (color, LIGHT_MARKS, LIGHT_BANDS) if color else (DARK, DARK_MARKS, DARK_BANDS)


def to_hud(view_rotation, world_direction):
    """A world space direction in the HUD's space (x right, y up, z ahead): the view's, whose camera looks down -Z"""
    view = view_rotation @ world_direction
    return np.array([view[0], view[1], -view[2]])


def flight_lines(view_matrix, drone_eval):
    """The horizon's and velocity marker's lines in the HUD's space: polylines with their thickness (radius), and dots,
    centers with their radius. view_matrix is the view's (world to view), drone_eval the Drone's evaluated object (or
    None)."""
    view_rotation = view_matrix.to_3x3().normalized()
    curves, dots = [], []
    if bpy.context.scene.controls_parameters.horizon:
        down = horizon.world_down(view_matrix.inverted().translation)
        if down is not None:
            curves, dots = horizon.lines(to_hud(view_rotation, down))
    if velocity._shown and drone_eval is not None:
        local = velocity.drone_velocity(drone_eval)
        if local is not None:
            world = drone_eval.matrix_world.to_3x3().normalized() @ Vector(local)
            curves = curves + velocity.lines(to_hud(view_rotation, world), Vector(local).length)
    return curves, dots


class View:
    """How the HUD's space maps into this draw's view: to the view's space (to_view), and pixels a HUD unit is wide
    sideways at a HUD unit's depth (pixels)"""

    def __init__(self, context):
        region = context.region
        self.projection = context.region_data.window_matrix
        # The view's clip start, from its projection's near plane
        near = self.projection[2][3] / (self.projection[2][2] - 1.0)
        self.to_view = np.array([1.0, 1.0, -1.0]) * (near / hud.CLIP_START_DEPTH)
        self.pixels = self.projection[0][0] * region.width / 2.0
        self.size = (region.width, region.height)

    def to_pixels(self, point):
        """A point in the HUD's space, in the region's pixels"""
        clip = self.projection @ Vector((*(np.asarray(point) * self.to_view), 1.0))
        return ((clip.x / clip.w + 1.0) * self.size[0] / 2.0, (clip.y / clip.w + 1.0) * self.size[1] / 2.0)


def draw(context, view, fills, bands, layers, dots):
    """Draw the HUD's shapes in the view, carried from the HUD's space into the view's, lines in pixels as thick as
    they are at their depth: fills (triangles) in the HUD's color, then bands (triangles) in the deadzones', then
    layers, (polylines, color) in order, then dots in the HUD's color"""
    color, _marks, band = colors(context)
    gpu.matrix.push()
    gpu.matrix.load_identity()
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(False)
    gpu.state.blend_set('ALPHA')
    try:
        fill_shader = shader('UNIFORM_COLOR')
        line_shader = shader('POLYLINE_UNIFORM_COLOR')

        def triangles(tris, tint):
            fill_shader.bind()
            fill_shader.uniform_float("color", tint)
            batch_for_shader(fill_shader, 'TRIS', {"pos": (tris.reshape(-1, 3) * view.to_view).astype(np.float32)}
                             ).draw(fill_shader)

        def polylines(group, width, tint):
            line_shader.bind()
            line_shader.uniform_float("viewportSize", view.size)
            line_shader.uniform_float("color", tint)
            line_shader.uniform_float("lineWidth", width)
            segments = np.concatenate([np.stack([p[:-1], p[1:]], axis=1).reshape(-1, 3) for p in group]) * view.to_view
            batch_for_shader(line_shader, 'LINES', {"pos": segments.astype(np.float32)}).draw(line_shader)

        if len(fills):
            triangles(fills, color)
        if len(bands):
            triangles(bands, band)
        for curves, tint in layers:
            by_width = {}
            for points, radius in curves:
                points = np.asarray(points, dtype=np.float64)
                by_width.setdefault(round(2.0 * radius * view.pixels / float(points[0][2]), 3), []).append(points)
            for width, group in by_width.items():
                polylines(group, max(width, 1.0), tint)
        if dots:
            angle = np.linspace(0.0, 2.0 * np.pi, DOT_RESOLUTION + 1)
            ring = np.stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)], axis=1)
            rims = [center + radius * ring for center, radius in dots]
            triangles(np.concatenate([np.stack([np.repeat(center[None], DOT_RESOLUTION, axis=0), rim[:-1], rim[1:]],
                                               axis=1) for center, rim in zip([c for c, _r in dots], rims)]), color)
            polylines(rims, DOT_EDGE, color)
    finally:
        gpu.state.blend_set('NONE')
        gpu.state.depth_mask_set(True)
        gpu.state.depth_test_set('NONE')
        gpu.matrix.pop()


def draw_texts(context, view, texts):
    """The HUD's text, in pixels, at its anchors' places: each line as tall as its size (an em) at its depth"""
    color = colors(context)[0]
    blf.color(FONT, *color)
    for text, anchor, size, align, on in texts:
        x, y = view.to_pixels(anchor)
        blf.size(FONT, size * view.pixels / anchor[2])
        width = blf.dimensions(FONT, text)[0]
        x -= {'LEFT': 0.0, 'CENTER': width / 2.0, 'RIGHT': width}[align]
        if on == 'MIDDLE':
            y -= blf.dimensions(FONT, "0")[1] / 2.0
        blf.position(FONT, x, y, 0.0)
        blf.draw(FONT, text)


def hud_camera(context):
    """The drone camera, when the view looks through it and its HUD is drawn (not DroneCam's, in a scene from before
    its first flight: hud.migrate()), else None"""
    region_data = context.region_data
    if region_data is None or not region_data.is_perspective or region_data.view_perspective != 'CAMERA':
        return None
    scene = context.scene
    camera = scene.references.camera_obj if hasattr(scene, "references") else None
    if camera is None or scene.camera != camera or hud.legacy_hud(camera) is not None:
        return None
    return camera


def flying(context):
    return getattr(context.window_manager, "er_fly_modal_running", False)


@background.foreground
def draw_hud():
    """POST_VIEW: the HUD's shapes, and while flying, the horizon and velocity marker"""
    context = bpy.context
    if hud_camera(context) is None:
        return
    shapes = hud.shapes(context.scene.drone_hud)
    color, marks, _bands = colors(context)
    layers = [(shapes["lines"], color), (shapes["marks"], marks)]
    dots = []
    if flying(context) and (horizon._level is not None or velocity._shown):
        from . import flight
        drone = flight.drone_obj
        drone_eval = drone.evaluated_get(context.evaluated_depsgraph_get()) if drone is not None else None
        curves, dots = flight_lines(context.region_data.view_matrix, drone_eval)
        layers.insert(0, (curves, color))
    if len(shapes["fills"]) or any(curves for curves, _tint in layers) or dots:
        draw(context, View(context), shapes["fills"], shapes["bands"], layers, dots)


@background.foreground
def draw_hud_text():
    """POST_PIXEL: the HUD's text"""
    context = bpy.context
    if hud_camera(context) is None:
        return
    texts = hud.shapes(context.scene.drone_hud)["texts"]
    if texts:
        draw_texts(context, View(context), texts)


_handles = []


def register():
    _handles.append((bpy.types.SpaceView3D.draw_handler_add(draw_hud, (), 'WINDOW', 'POST_VIEW'), 'WINDOW'))
    _handles.append((bpy.types.SpaceView3D.draw_handler_add(draw_hud_text, (), 'WINDOW', 'POST_PIXEL'), 'WINDOW'))


def unregister():
    while _handles:
        handle, region = _handles.pop()
        bpy.types.SpaceView3D.draw_handler_remove(handle, region)
