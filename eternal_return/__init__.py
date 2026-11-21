"""Eternal Return: fly a virtual FPV drone through Blender scenes with an RC radio or gamepad, and change its scale
in flight. Registration: the settings groups, operators and panels (controls.py, operators.py, flight.py, panels.py),
the scene attributes the settings are kept under, and the add-on's own modules' handlers and draw handlers."""
import bpy

from . import controls
from . import flight
from . import fractal
from . import hud
from . import hud_draw
from . import hud_render
from . import mandelbrot
from . import operators
from . import panels
from . import parts
from . import physics
from . import solid_sky
from . import sphere
from . import surface

# Every settings group by the scene attribute it's kept under
SCENE_GROUPS = {**controls.SCENE_GROUPS, **operators.SCENE_GROUPS}

# Settings groups first, before anything that points to them (each other's pointers, the preferences), panels last,
# in their order on screen
CLASSES = (
    *SCENE_GROUPS.values(),
    *operators.OPERATORS,
    flight.ER_OT_ResetScale,
    flight.ER_OT_ResetLocations,
    flight.ER_OT_ResetCamera,
    flight.ER_OT_ResetLensClip,
    flight.ER_OT_ResetMinScale,
    flight.ER_OT_ResetMaxScale,
    operators.EternalReturnPreferences,
    *panels.PANELS,
)

# The add-on's own modules with draw handlers or load handlers, registered after the settings they read; solid_sky
# before the floors' draw handlers (mandelbrot, sphere), so they draw over its fog layers.
MODULES = (solid_sky, mandelbrot, sphere, parts, surface, hud_draw)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    physics.register()
    hud.register()
    hud_render.register()

    for name, group in SCENE_GROUPS.items():
        setattr(bpy.types.Scene, name, bpy.props.PointerProperty(type=group))
    # On the window manager, so it's never saved into a file: a file saved mid-flight doesn't open "flying"
    bpy.types.WindowManager.er_fly_modal_running = bpy.props.BoolProperty(
        name="Flying", description="The Fly operator is running", default=False)

    # Endless fractals' eyes follow the drone camera once a frame, flying or not
    if fractal.follow_eyes not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(fractal.follow_eyes)

    for module in MODULES:
        module.register()


def unregister():
    for module in reversed(MODULES):
        module.unregister()

    if fractal.follow_eyes in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(fractal.follow_eyes)

    del bpy.types.WindowManager.er_fly_modal_running
    for name in reversed(SCENE_GROUPS):
        delattr(bpy.types.Scene, name)

    hud_render.unregister()
    hud.unregister()
    physics.unregister()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
