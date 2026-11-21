"""The controller read as a gamepad, for the HUD's gimbals (hud.py): a radio's raw axes 0-3 are already the gimbals'
axes, a gamepad's sticks are read through SDL's gamepad mapping, wherever its model and driver put them."""
import sdl2

# For a gamepad, the stick that feeds each of the HUD gimbals' axes 0-3 (hud.py), and the sign that matches it to a
# radio. The gimbals were built for a Mode 2 RC radio in AETR channel order, whose raw axes 0-3 are the right stick's X
# and Y, then the left stick's Y and X, with right and up positive. SDL gamepads report up as negative.
GIMBAL_GAMEPAD_AXES = [
    (sdl2.SDL_CONTROLLER_AXIS_RIGHTX, 1.0),
    (sdl2.SDL_CONTROLLER_AXIS_RIGHTY, -1.0),
    (sdl2.SDL_CONTROLLER_AXIS_LEFTY, -1.0),
    (sdl2.SDL_CONTROLLER_AXIS_LEFTX, 1.0),
]

def read_gimbal_inputs(joystick_axis_inputs, gamepad):
    """
    Get the stick positions for the HUD's gimbals (hud.py), in the order of its axes 0-3.

    A radio's raw axes are already in that order. A gamepad numbers its axes by model and driver (an Xbox controller
    puts its left stick on axes 0 and 1), so its sticks are read through SDL's gamepad mapping instead.

    Return:
    List of 4 normalized axis values
    """
    if gamepad:
        return [sdl2.SDL_GameControllerGetAxis(gamepad, axis) / 32768 * sign for axis, sign in GIMBAL_GAMEPAD_AXES]

    # Padded, so a controller with fewer than 4 axes leaves the missing gimbal axes centered
    return (joystick_axis_inputs + [0.0] * 4)[:4]

def gimbal_axes(gamepad):
    """The joystick axis each of read_gimbal_inputs()' four values comes from, and its sign there, as
    (axis or None, sign): a radio's are its axes 0-3 themselves, a gamepad's wherever SDL's mapping binds its sticks"""
    if not gamepad:
        return tuple((axis, 1.0) for axis in range(4))
    found = []
    for axis, sign in GIMBAL_GAMEPAD_AXES:
        bind = sdl2.SDL_GameControllerGetBindForAxis(gamepad, axis)
        found.append((bind.value.axis if bind.bindType == sdl2.SDL_CONTROLLER_BINDTYPE_AXIS else None, sign))
    return tuple(found)

def open_gamepad(controller):
    """
    Open the controller as a gamepad too, if SDL has a gamepad mapping for it. SDL's built-in mappings cover Xbox,
    PlayStation, Switch and most other gamepads, but not RC radios, which SDL treats as plain joysticks.

    Return:
    The gamepad, or None if SDL has no mapping for the controller
    """
    instance_id = sdl2.SDL_JoystickInstanceID(controller)
    for i in range(sdl2.SDL_NumJoysticks()):
        if sdl2.SDL_JoystickGetDeviceInstanceID(i) == instance_id:
            if sdl2.SDL_IsGameController(i):
                return sdl2.SDL_GameControllerOpen(i)
            break
    return None
