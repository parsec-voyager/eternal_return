"""The HUD's velocity marker: where the drone is flying, as opposed to where the camera looks.

It's an aircraft HUD's flight path marker: a small ring with wings and a tail, on the point of the view the drone's
velocity points at. Flying straight ahead, it sits in the middle of the HUD's ring. Turning, the drone's momentum carries
it on the old way for a while, and the marker slides off toward it, then back as the flight follows the turn. Strafing
or drifting, it shows which way. With the velocity pointing behind the camera, as when flying backward, it marks the
point straight opposite instead, the direction the drone is coming from, with an X across the ring, as space flight
games mark retrograde. Each is drawn where its direction crosses the HUD's plane, a fixed distance ahead of the
camera, so it's exact at any pitch, roll and lens. Directions too far to the side to be in view, or too slow to have a
direction, hide it.

The velocity is the drone's flight: the "vel" attribute the physics move the drone by each frame, in the Drone object's
space, written onto the Drone's mesh by the core (physics.py). Scaling, loops, wraps and Keep Drone at Origin don't
change where the drone flies among what it sees, so they don't move the marker. Orbit and Zoom do move the drone, around
or toward the selected object, but they aren't flight, so the marker leaves them out, as Tidal Lock does: orbiting, it
still shows where momentum carries the drone. Tidal Lock and Orbit turn the Drone object, and the velocity with it.

It's drawn by hud_draw.py's draw handler, which reads the velocity from the Drone's evaluated geometry and turns it into
the view by the Drone's evaluated rotation and the view's own, as Blender draws that frame, so it's exact for the frame
drawn. Python's tick only says whether it's shown. Read in the tick, it would be a frame stale whenever Blender draws a
frame before a tick has run since its frame step (see horizon.py).

Until 2026-11-17 it was a geometry nodes object under the camera, made again at every depsgraph evaluation that
moved the camera (see horizon.py).
"""
import math

import bpy
import numpy as np

# The drone's velocity attribute (physics.write()), in the Drone object's space. Every point of the drone's geometry has
# the same one.
VELOCITY_ATTRIBUTE = "vel"

# Where the marker is drawn, in HUD units (hud.py: 0.1 ahead is the camera's clip start): in front of the horizon's
# lines and dot (horizon.DEPTH, horizon.DOT_DEPTH), behind the HUD's ring and gimbals (0.1)
DEPTH = 0.1032

# The ring, and its wings and tail, beyond it: about 52 px across the ring and 104 px wing tip to wing tip on a 2752 px
# wide view. Flying straight ahead, the whole marker fits inside the HUD's ring (its outer edge 0.0108 from the middle).
RADIUS = 0.0035
TICK_LENGTH = 0.0035
# Its lines' thickness (half of it): about 7 px, between the horizon (9 px) and its latitude circles (4 px)
LINE_RADIUS = 0.00045
# Straight pieces around the ring: at 7 px thick, the gaps between them at its bends are under a fifth of a pixel
RING_RESOLUTION = 64

# Directions this close to square to the view (their Z, of 1) hide the marker: about 81 degrees off, past the corners of
# a 10 mm lens's view, and closer to 90 they'd be projected out toward infinity (like horizon.MIN_VIEW_Z)
MIN_VIEW_Z = 0.15

# Slower than this, in the drone's own meters per second, the velocity hides the marker: too little to have a direction
# worth showing (about 0.2 km/h at scale 1)
MIN_SPEED = 0.05

# Whether the next draws show it: set_shown()
_shown = False


def set_shown(shown):
    """Show or hide the velocity marker. Shown, it still hides itself while the drone is too slow or flying too far
    to the side."""
    global _shown
    _shown = bool(shown)


def drone_velocity(drone_eval):
    """The drone's velocity, in the Drone object's space, from its evaluated object (drone_eval); None without
    one"""
    mesh = getattr(drone_eval, "data", None)
    attribute = mesh.attributes.get(VELOCITY_ATTRIBUTE) if isinstance(mesh, bpy.types.Mesh) else None
    if attribute is None or attribute.domain != 'POINT' or not len(attribute.data):
        return None
    return attribute.data[0].vector


def lines(hud_velocity, speed):
    """The marker in the HUD's space (x right, y up, z ahead, in HUD units) for hud_velocity (a direction
    there) at speed (the drone's own meters per second): its lines, each a polyline of points (an array, (n, 3)) with
    its thickness (radius). Empty while hidden."""
    direction = np.asarray(hud_velocity, dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if not (_shown and speed > MIN_SPEED and length > 0.0):
        return []
    direction = direction / length
    # The camera looks along the HUD's +Z. Pointing behind it, the opposite direction is marked instead.
    behind = direction[2] < 0.0
    marked = -direction if behind else direction
    if not marked[2] > MIN_VIEW_Z:
        return []
    center = marked * (DEPTH / marked[2])

    # The marker, flat in the HUD's plane: the ring, and wings left and right of it and a tail above it. For the
    # opposite direction, an X across the ring too, from rim to rim.
    angle = np.linspace(0.0, 2.0 * math.pi, RING_RESOLUTION + 1)
    shapes = [np.stack([RADIUS * np.cos(angle), RADIUS * np.sin(angle), np.zeros_like(angle)], axis=1)]
    shapes += [np.array([[side * RADIUS, 0.0, 0.0], [side * (RADIUS + TICK_LENGTH), 0.0, 0.0]]) for side in (-1.0, 1.0)]
    shapes.append(np.array([[0.0, RADIUS, 0.0], [0.0, RADIUS + TICK_LENGTH, 0.0]]))
    if behind:
        corner = RADIUS / math.sqrt(2.0)
        shapes += [np.array([[-corner, -corner, 0.0], [corner, corner, 0.0]]),
                   np.array([[-corner, corner, 0.0], [corner, -corner, 0.0]])]
    return [(center + shape, LINE_RADIUS) for shape in shapes]
