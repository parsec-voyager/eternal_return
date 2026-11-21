"""Sunlight in Solid mode: Workbench's studio lighting fixed in the world and turned to face the map's Sun, plus Solid
mode's shadow, cast from it. For maps flown in Solid mode, where EEVEE's real sun and shadows cost too much.

Why not the MatCap the maps use: a MatCap is a picture of a lit sphere, looked up by which way a surface faces the
camera, so its light turns with the view and can never come from a sun. Studio lighting with World Space Lighting is
fixed in the world instead. Its outdoor.sl rig is a sun and a sky: a warm, fairly sharp key light about 40 degrees up,
a soft blue sky light, and a dim bounce from below. Measured in World Space Lighting (seen from above, 2026-10-30), its
key light comes from azimuth KEY_AZIMUTH at studiolight_rotate_z 0, and turning studiolight_rotate_z up turns it the
other way, so facing it toward the Sun is a rotation of KEY_AZIMUTH less the Sun's azimuth. Its height is the rig's,
KEY_ELEVATION, whatever the Sun's.

Solid mode's Shadow option casts every object's shadow along the scene's light direction (scene.display), which is set
toward the Sun's azimuth at the key light's height, so light and shadows agree. It's off unless asked for: on the
endless Inferno (1.2M triangles) it took a Workbench frame from 47 to 546 ms, about 3 fps in the user's viewport, and
it's all or nothing (Workbench ignores the objects' Shadow visibility: with the floor strips and the small arch levels
not casting, still 535 ms). The light direction is set either way, for turning it on in the viewport. At the Sun's own height they'd be
nonsense for the user's: it grazes the horizon (0.3 degrees down, a sunset in EEVEE), and its shadows would cover every
floor. Headless renders can't check it: with it on, every surface came out dark whatever the direction,
so its look and its cost on a big map are for the viewport to show. A Sun that moves later needs this run again.

Usage: Blender -b map.blend --python tools/solid_sun.py -- output.blend [--shadow 0.5]
(saves a copy; to change a map you have open, run apply() in its Blender). From another script: apply(scene).
"""
import math
import sys

import bpy
from mathutils import Vector

STUDIO_LIGHT = "outdoor.sl"
KEY_AZIMUTH = -56.0           # degrees: where outdoor.sl's key light comes from, at studiolight_rotate_z 0
KEY_ELEVATION = 39.8          # degrees: how high it is (its direction's z, 0.64)
SHADOW = 0.5                  # how dark the shadow is, 0 to 1, when it's asked for


def sun_object(scene):
    """The scene's sun lamp: one named Sun, else the first of type SUN, else None"""
    suns = [o for o in scene.objects if o.type == 'LIGHT' and o.data.type == 'SUN']
    return next((o for o in suns if o.name == "Sun"), suns[0] if suns else None)


def toward_sun(sun):
    """The direction the Sun's light comes from, in the world: a sun lamp shines along its own -Z"""
    direction = sun.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0))
    return direction.normalized()


def shadings(scene):
    """Every Solid shading the scene's views and Workbench renders use, with the 3D views kept behind another editor
    in their area (Layout's Node Editor has one), which come back when it's switched to the 3D Viewport"""
    found = [scene.display.shading]
    for screen in bpy.data.screens:
        for area in screen.areas:
            found += [space.shading for space in area.spaces if space.type == 'VIEW_3D']
    return found


def apply(scene, sun=None, shadow=None):
    """Light Solid mode from the Sun (see the module's docstring), keeping each view's colors, cavity and background.
    shadow: the shadow's darkness, or None for no shadow. Returns a line saying what it did."""
    sun = sun or sun_object(scene)
    if sun is None:
        raise ValueError(f"{scene.name} has no sun lamp to light Solid mode from")
    sun_direction = toward_sun(sun)
    azimuth = math.degrees(math.atan2(sun_direction.y, sun_direction.x))
    height = math.radians(KEY_ELEVATION)
    direction = Vector((math.cos(height) * math.cos(math.radians(azimuth)),
                        math.cos(height) * math.sin(math.radians(azimuth)), math.sin(height)))
    rotation = math.radians((KEY_AZIMUTH - azimuth + 180.0) % 360.0 - 180.0)
    views = shadings(scene)
    for shading in views:
        # Set in Solid mode: in Material Preview or Rendered, studio_light means the world's HDRI, not Solid mode's
        # light, and outdoor.sl isn't one
        mode, shading.type = shading.type, 'SOLID'
        shading.light = 'STUDIO'
        shading.studio_light = STUDIO_LIGHT
        shading.use_world_space_lighting = True
        shading.studiolight_rotate_z = rotation
        shading.show_specular_highlight = True
        shading.show_shadows = shadow is not None
        if shadow is not None:
            shading.shadow_intensity = shadow
        shading.type = mode
    scene.display.light_direction = direction
    return (f"Solid mode lit by {sun.name}: {STUDIO_LIGHT} in world space turned {math.degrees(rotation):.0f} degrees "
            f"(sun azimuth {azimuth:.0f}, lit from {KEY_ELEVATION:.0f} degrees up; the lamp is "
            f"{math.degrees(math.asin(max(-1.0, min(1.0, sun_direction.z)))):.1f}), "
            f"shadow {'off' if shadow is None else shadow}, in {len(views) - 1} views and Workbench renders")


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    shadow = (float(args[args.index("--shadow") + 1]) if args.index("--shadow") + 1 < len(args)
              and not args[args.index("--shadow") + 1].endswith(".blend") else SHADOW) if "--shadow" in args else None
    outputs = [a for a in args if a.endswith(".blend")]
    if not outputs:
        raise SystemExit("solid_sun.py needs an output .blend: it saves a copy, never over the map")
    print(apply(bpy.context.scene, shadow=shadow))
    bpy.ops.wm.save_as_mainfile(filepath=outputs[0], copy=True)
    print("SAVED", outputs[0])
