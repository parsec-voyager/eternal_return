"""Gravity toward the selected object: the drone falls toward its origin, or with Mesh, the closest point of its surface.

The drone's Gravity (the scene's physics, physics.DronePhysics) pulls along the Drone object's Z axis,
wherever tidal lock and orbit have turned it: as Add Drone builds it, world -Y. While flying with an object selected, the same
Gravity pulls toward the object instead (negative pulls in, as it pulls down along Z; positive pushes away),
like the horizon's down: its origin, or with Gravity Mesh, the closest point of its mesh, along auto scale's Split Axes
axes only, so over a floor whose config splits auto scale on Z it pulls straight down to the floor. A sphere or surface
map is always pulled to its surface straight below (flight.FlightFeatures.get_selected_object_diff_vector()), so on the Earth the drone
falls to the ground. With nothing selected (between clicks, with Auto Deselect), it's the drone's own gravity again.

Gravity is in the drone's own units, like its thrust and drag: shrunk to a millionth of its size, it falls a millionth
as fast, so it handles the same at any scale. Diving at a surface with auto scale on, the drone shrinks as it nears it,
and falls ever slower in the world: it falls into the surface forever, never reaching it, and with loops on, into a
fractal without end.

The pull is the core's gravity parameter (physics.py), a vector in the Drone object's space, worked out every tick from
the way down and the Drone's rotation (or the drone's own (0, 0, Gravity), with nothing selected). A recording keys it
every frame, so it replays as flown.
"""

# The gravity for the next frames (set_pull())
_pull = (0.0, 0.0, 0.0)


def set_pull(pull):
    """The gravity for the next frames, in the Drone object's space and units a second squared"""
    global _pull
    _pull = tuple(float(c) for c in pull)


def pull():
    return _pull
