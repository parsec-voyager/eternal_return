"""The controller and the controls: the settings that say which of the controller's axes drives each control and how
(scene.control_axes, scene.control_settings, ...), the controller itself through SDL, and the conversion of
its stick positions into what each control does.

A tick goes axes -> raw controls -> converted controls:
- read_axes(): the controller's axes, normalized to [-1, 1), as SDL reports them
- map_controls(): each control's stick position, from the axis it's mapped to (or its rest position, unmapped). These
  are what the core (physics.sticks_to_commands()) and the HUD read, so nothing is inverted or cut by a deadzone yet.
- convert(): what each control does: the core's thrust and rates for the four flight controls, a factor a second for
  the factor controls, 0 to 1 for orbit and click (flight.dict_controls)
"""
import math

import bpy
import sdl2

from . import physics
from . import scaling_settings

# The 11 things a stick axis can drive, in the panel's order
CONTROLS = ("throttle", "yaw", "pitch", "roll", "scale", "zoom", "orbit", "click", "camera", "lens_clip", "brake")
# Converted by the core (physics.sticks_to_commands()): thrust in newtons, rates in radians a second
FLIGHT_CONTROLS = ("throttle", "yaw", "pitch", "roll")
ROTATION_CONTROLS = ("yaw", "pitch", "roll")
# On an RC radio these rest at the bottom of the stick (-1), off; Centered (gamepads) rests them at the center instead
ONE_WAY_CONTROLS = ("throttle", "orbit", "click", "brake")
# Converted to a factor a second, factor ** x: 1 at rest. Their default factors.
FACTOR_CONTROLS = {"scale": 6.0, "zoom": 2.0, "camera": 2.0, "lens_clip": 8.0, "brake": 10.0}

# Each control's default axis, the way a RadioMaster Pocket in AETR order (EdgeTX) reports its sticks and switches,
# and its default invert
DEFAULT_AXES = {"throttle": '2', "yaw": '0', "pitch": '1', "roll": '3', "scale": '5', "zoom": 'none', "orbit": '4',
                "click": '7', "camera": 'none', "lens_clip": '6', "brake": 'none'}
DEFAULT_INVERTED = {"pitch", "scale", "lens_clip"}

# How many of the controller's axes can be mapped, and shown in the panel
NUM_AXES = 8
# SDL's axis values are signed 16 bit
AXIS_RANGE = 32768

# The rest of a control's rate settings (scene.stick_rates), in degrees a second
DEFAULT_CENTER_SENSITIVITY = 145.0
DEFAULT_MAX_RATE = 533.0


def label(control):
    """A control's name as the panel shows it ("lens_clip" -> "Lens Clip")"""
    return control.replace("_", " ").title()


def add_properties(cls, properties):
    """Give a settings group the properties made from the controls table (identifier -> bpy.props definition), as if
    written as annotations in its class body. Blender reads them when the class is registered."""
    cls.__dict__["__annotations__"].update(properties)
    return cls


AXIS_ITEMS = [('none', "", "Not mapped: the control stays at rest")] + [
    (str(axis), str(axis), f"The controller's axis {axis}") for axis in range(NUM_AXES)]


class ControlAxes(bpy.types.PropertyGroup):
    """Which of the controller's axes each control reads (scene.control_axes)"""
    __annotations__ = {
        control: bpy.props.EnumProperty(
            name=label(control), description=f"The controller axis {label(control)} reads", items=AXIS_ITEMS,
            default=DEFAULT_AXES[control])
        for control in CONTROLS}


class ControllerAxes(bpy.types.PropertyGroup):
    """The controller's axes as the last tick read them, for the panel (scene.controller_axes)"""
    __annotations__ = {
        f"axis_{axis}": bpy.props.FloatProperty(
            name=f"Axis {axis}", description=f"Where the controller's axis {axis} is (-1 to 1)", default=0.0,
            precision=1)
        for axis in range(NUM_AXES)}


class ControlSettings(scaling_settings.ScalingSettings, bpy.types.PropertyGroup):
    """How the controls behave (scene.control_settings): the flight mode and the thrust at full throttle, each
    control's invert, deadzone and centering, the factor controls' factors, and from the mixin, the scaling tools'
    settings"""
    flight_mode: bpy.props.EnumProperty(
        name="Flight Mode", description="How the sticks fly the drone",
        items=[
            ('ACRO', "Acro (Hard)", "The sticks set the drone's turn rates; nothing levels it"),
            ('3D', "3D (Hard)", "Acro with reversible thrust: below center, the throttle thrusts backward"),
            ('HORIZON', "Horizon (Easy)", "Acro that levels the drone again when the sticks are let go"),
            ('ANGLE', "Angle (Easy)", "Levels the drone when the sticks are let go, and tilts it no further than "
                                      "the Angle Limit"),
        ],
        default='3D')
    level_strength: bpy.props.FloatProperty(
        name="Auto-Level Strength", description="How strongly Horizon and Angle modes level the drone",
        default=20.0, min=0.0, max=100.0, subtype='PERCENTAGE', precision=0)
    angle_limit: bpy.props.FloatProperty(
        name="Angle Limit", description="How far Angle mode lets the drone tilt from level",
        default=math.radians(60.0), soft_min=0.0, soft_max=math.pi, subtype='ANGLE')
    max_thrust: bpy.props.FloatProperty(
        name="Maximum Thrust", description="The drone's thrust at full throttle, in newtons", default=30.0,
        soft_min=0.0, soft_max=100.0, precision=1)


def per_control_parameters():
    """Every control's invert and deadzone, the one-way controls' centering, and the factor controls' factors"""
    properties = {}
    for control in CONTROLS:
        name = label(control)
        properties[f"{control}_invert"] = bpy.props.BoolProperty(
            name=f"Invert {name}", description=f"Reverse the direction of {name}'s axis",
            default=control in DEFAULT_INVERTED)
        properties[f"{control}_deadzone"] = bpy.props.FloatProperty(
            name=f"{name} Deadzone", description=f"How much of {name}'s stick travel does nothing, about its rest",
            default=10.0, min=0.0, max=100.0, subtype='PERCENTAGE', precision=1)
        if control in ONE_WAY_CONTROLS:
            properties[f"{control}_centered"] = bpy.props.BoolProperty(
                name=f"{name} Centered",
                description=f"{name}'s stick rests at its center, as on a gamepad, rather than at the bottom, as "
                            f"on an RC radio: only the upper half does anything",
                default=False)
        if control in FACTOR_CONTROLS:
            properties[f"{control}_factor"] = bpy.props.FloatProperty(
                name=f"{name} Factor", description=f"How many times over a second {name} changes things at full stick",
                default=FACTOR_CONTROLS[control], min=1.0)
    return properties


add_properties(ControlSettings, per_control_parameters())


def rate_properties(axis):
    """One rotation axis's rates: its turn rate at the stick's center and at full stick, and the expo between"""
    return {
        f"{axis}_center_sensitivity": bpy.props.FloatProperty(
            name="Center Sensitivity", description=f"How fast {axis} turns the drone near the stick's center",
            default=math.radians(DEFAULT_CENTER_SENSITIVITY), soft_min=math.radians(50.0),
            soft_max=math.radians(1000.0), subtype='ANGLE', precision=1),
        f"{axis}_max_rate": bpy.props.FloatProperty(
            name="Max Rate", description=f"How fast {axis} turns the drone at full stick",
            default=math.radians(DEFAULT_MAX_RATE), soft_min=math.radians(50.0), soft_max=math.radians(1000.0),
            subtype='ANGLE', precision=1),
        f"{axis}_expo": bpy.props.FloatProperty(
            name="Expo", description="How much the rate curve flattens near the center, for finer control there",
            default=0.5, min=0.0, max=1.0, precision=2),
    }


class StickRates(bpy.types.PropertyGroup):
    """Yaw, pitch and roll rates (scene.stick_rates), in degrees a second, stored in radians, as
    Betaflight's Actual rates: the turn rate at the stick's center, the rate at full stick, and the expo bending the
    curve between. The core turns them into the drone's turn rates (physics.stick_settings())."""
    __annotations__ = {name: prop for axis in ROTATION_CONTROLS for name, prop in rate_properties(axis).items()}


class ControlOutputs(bpy.types.PropertyGroup):
    """What each control does, as the last tick converted it, for the panel (scene.control_outputs)"""
    __annotations__ = {
        control: bpy.props.FloatProperty(
            name=label(control), description=f"{label(control)} as the last tick converted it", default=0.0,
            subtype='ANGLE' if control in ROTATION_CONTROLS else 'NONE')
        for control in CONTROLS}


# The settings groups, by the scene attribute each is kept under (also the add-on preferences' attributes)
SCENE_GROUPS = {
    "control_axes": ControlAxes,
    "controller_axes": ControllerAxes,
    "control_settings": ControlSettings,
    "stick_rates": StickRates,
    "control_outputs": ControlOutputs,
}


# SDL controller access

def get_controller(report):
    """Open the controller to fly with: the first joystick SDL lists that isn't a vJoy device (a Windows virtual
    joystick, often installed alongside a radio's own driver), or a vJoy device if there's nothing else. Every call
    opens afresh, so a handle from an earlier flight is never reused. report(type_set, message) is the operator's.

    Return:
    The SDL joystick, or None (reported) if there's none to fly with
    """
    # Since macOS 27.2, Apple's GameController framework claims USB radios like the RadioMaster Pocket and never hands
    # them to SDL. With SDL's MFi driver off, SDL finds them through IOKit. Only read by SDL_Init.
    sdl2.SDL_SetHint(sdl2.SDL_HINT_JOYSTICK_MFI, b"0")
    # The game controller subsystem loads SDL's gamepad mappings, which gamepad.open_gamepad() needs
    sdl2.SDL_Init(sdl2.SDL_INIT_JOYSTICK | sdl2.SDL_INIT_GAMECONTROLLER)

    count = sdl2.SDL_NumJoysticks()
    controller = None
    vjoy = None
    for index in range(count):
        name = (sdl2.SDL_JoystickNameForIndex(index) or b"").decode("utf-8", "replace")
        print(f"Joystick {index}: {name}")
        joystick = sdl2.SDL_JoystickOpen(index)
        if not joystick:
            continue
        print(f"  {sdl2.SDL_JoystickNumAxes(joystick)} axes")
        if "vJoy" in name:
            report({'WARNING'}, f"Skipping {name}: a vJoy virtual joystick")
            if vjoy is None:
                vjoy = joystick
            else:
                sdl2.SDL_JoystickClose(joystick)
            continue
        report({'INFO'}, f"Flying with {name}")
        controller = joystick
        break

    if controller is not None:
        if vjoy is not None:
            sdl2.SDL_JoystickClose(vjoy)
        return controller
    if vjoy is not None:
        report({'INFO'}, "Only a vJoy virtual joystick is connected: flying with it")
        return vjoy
    report({'WARNING'}, "No controller could be opened" if count > 0 else
           "No controller is connected: plug in an RC radio or gamepad and fly again")
    return None


def read_axes(controller):
    """Every axis of the controller, in axis order, as raw axes (-1 to 1), with the first NUM_AXES shown in the panel
    (scene.controller_axes)"""
    sdl2.SDL_JoystickUpdate()
    axes = [sdl2.SDL_JoystickGetAxis(controller, axis) / AXIS_RANGE
            for axis in range(sdl2.SDL_JoystickNumAxes(controller))]
    inputs = bpy.context.scene.controller_axes
    for axis, value in enumerate(axes[:NUM_AXES]):
        setattr(inputs, f"axis_{axis}", value)
    return axes


def close_controller(controller):
    if controller:
        sdl2.SDL_JoystickClose(controller)


def quit_sdl():
    sdl2.SDL_Quit()


# Raw and converted controls

def rest_value(control, scene):
    """The raw value of a control at rest (unmapped, or mapped past the controller's last axis): what convert() turns
    into off. A one-way control rests at the bottom of its stick, except the throttle in 3D mode, where the whole
    travel is thrust, forward and back."""
    parameters = scene.control_settings
    if control == "throttle" and parameters.flight_mode == '3D':
        value = 0.0
    elif control in ONE_WAY_CONTROLS:
        value = -1.0
    else:
        return 0.0
    return -value if getattr(parameters, f"{control}_invert") else value


def map_controls(axes, scene):
    """Each control's raw value: the position of the axis it's mapped to, else its rest value (rest_value())"""
    mapping = scene.control_axes
    raw = {}
    for control in CONTROLS:
        axis = getattr(mapping, control)
        if axis != 'none' and int(axis) < len(axes):
            raw[control] = axes[int(axis)]
        else:
            raw[control] = rest_value(control, scene)
    return raw


def convert_control(control, x, parameters):
    """One control that isn't a flight control, from its raw value x: inverted, a one-way control's stick mapped to
    0 to 1, the deadzone cut out (the rest of the stick stretched over the whole range, so there's no jump at its
    edge), and a factor control's factor raised to it"""
    if getattr(parameters, f"{control}_invert"):
        x = -x
    if control in ONE_WAY_CONTROLS:
        if not getattr(parameters, f"{control}_centered"):
            x = (x + 1.0) / 2.0
        # Centered, the lower half of the stick does nothing
        x = min(max(x, 0.0), 1.0)
    deadzone = getattr(parameters, f"{control}_deadzone") / 100.0
    if deadzone > 0.0:
        if deadzone >= 1.0 or abs(x) <= deadzone:
            x = 0.0
        else:
            x = math.copysign(min(max((abs(x) - deadzone) / (1.0 - deadzone), 0.0), 1.0), x)
    if control in FACTOR_CONTROLS:
        x = getattr(parameters, f"{control}_factor") ** x
    return float(x)


def convert(raw, scene):
    """What each control does, from the raw controls (map_controls()): the core's thrust (N) and rates (rad/s) for
    the four flight controls, a factor a second (1 at rest) for the factor controls, 0 to 1 for orbit and click.
    Also shown in the panel (scene.control_outputs)."""
    # Once a conversion: the core keeps the settings the physics flies the frame with
    converted = {key: float(value) for key, value in physics.sticks_to_commands(raw, scene).items()}
    parameters = scene.control_settings
    for control in CONTROLS:
        if control not in FLIGHT_CONTROLS:
            converted[control] = convert_control(control, raw[control], parameters)
    output = scene.control_outputs
    for control in CONTROLS:
        setattr(output, control, converted[control])
    return {control: converted[control] for control in CONTROLS}
