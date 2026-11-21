"""The propellers' sound while flying: the scene's chosen sound file (flight_options.drone_sound) from the add-on's
sound/ folder, looped on Blender's audio device (aud), louder and higher the harder the throttle is pushed."""
import os

import aud
import bpy

SOUND_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sound")

# The sound's pitch at idle, and how much full throttle raises it: 0.5 at idle, 2.5 at full (times the sound's speed)
IDLE_PITCH = 0.5
PITCH_RANGE = 2.0

# The Drone Sound menu's slots, in order, each its sounds as (file name, label, speed), the first whose file is in the
# sound/ folder taken. A scene keeps its choice as the slot's position, so the order stays: new slots go at the end.
# The last sound of each slot is the add-on's own (made by tools/make_sounds.py); one before it is a file of the
# pilot's, kept out of the repo (.gitignore) as it isn't the add-on's to share. Speed scales the pitch the throttle
# plays it at, so a beat isn't too fast at full throttle.
SLOTS = (
    (("propellers.wav", "Drone Propellers", 1.0),),
    (("spaceship.wav", "Spaceship", 1.0),),
    (("ufo_8bit.wav", "8-bit UFO", 1.0),),
    (("oggggm-power_176-93513.mp3", "Amen Break", 0.4), ("breakbeat.wav", "Breakbeat", 1.0)),
    (("alien.wav", "Alien Spaceship", 1.0),),
)


def sound_path(name):
    return os.path.join(SOUND_FOLDER, name)


# The menu's sounds, one (file name, label, speed) a slot: the files are those installed with the add-on
SOUNDS = tuple(next((s for s in slot if os.path.exists(sound_path(s[0]))), slot[-1]) for slot in SLOTS)
SPEEDS = {name: speed for name, _label, speed in SOUNDS}


class PropellerSound:
    """The chosen sound, playing from the start of a flight (execute()) until stop() (cancel()), silent until the first
    update()"""

    def __init__(self):
        self.device = aud.Device()
        # Put back when the flight stops: update() sets the device's volume, which Blender's own sound plays at too
        self.device_volume = self.device.volume
        self.device.volume = 0.0
        self.name = None
        self.handle = None
        self.speed = 1.0
        self.play(bpy.context.scene.flight_options.drone_sound)

    def play(self, name):
        """Loop the sound file name forever, silent until update() sets its volume"""
        self.stop_handle()
        self.name = name
        sound = aud.Sound(sound_path(name)).loop(-1)
        self.handle = self.device.play(sound)
        self.handle.volume = 0.0
        self.speed = SPEEDS.get(name, 1.0)
        self.handle.pitch = IDLE_PITCH * self.speed

    def update(self, throttle):
        """Follow the throttle (0 to 1): its volume, and its pitch, rising from idle, times the sound's speed. The
        device plays at the panel's Sound Volume, the percent used as the factor itself (the default 10 plays loud), as
        the user's settings are tuned to it."""
        throttle = min(max(throttle, 0.0), 1.0)
        flight_options = bpy.context.scene.flight_options
        if flight_options.drone_sound != self.name:
            self.play(flight_options.drone_sound)
        self.device.volume = flight_options.volume
        if self.handle is not None:
            self.handle.volume = throttle
            self.handle.pitch = (IDLE_PITCH + PITCH_RANGE * throttle) * self.speed

    def stop_handle(self):
        if self.handle is not None:
            self.handle.stop()
            self.handle = None

    def stop(self):
        """Stop playing, and give the device back its volume"""
        self.stop_handle()
        self.device.volume = self.device_volume
