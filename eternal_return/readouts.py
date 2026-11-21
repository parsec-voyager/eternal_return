"""The HUD's readouts' text (hud.py): Altitude, Speed, Scale and Scale Speed, as the flight works them out
(odometer.py), written every odometer.PERIOD frames.

Blender's coordinates and the drone's Blender scale would make poor readouts: wraps, recenters, Keep Drone at Origin
and loops all jump them, and they're in Blender units, not the Earth's meters. The add-on knows what each of those was,
so it works the four out itself. Numbers of any size are shown to three figures: lengths in
meters with SI prefixes (nm, km, Mm), past them in powers of ten, as the scales are.
"""
import math

# SI prefixes, by thousands from yocto (10^-24) to yotta (10^24). The newer ronto and quecto, ronna and quetta are
# left out: "320 rm" reads as nothing, "3.20e-25 m" does.
PREFIXES = "yzafpnµm kMGTPEZY"


def figures(log10_value):
    """A positive number given as its log10, to three significant figures, as (digits, power of ten of the first)"""
    power = math.floor(log10_value)
    mantissa = 10.0 ** (log10_value - power)
    # Rounded up to 10.0 (9.996), it's the next power's 1.00
    if round(mantissa, 2) >= 10.0:
        power += 1
        mantissa = 1.0
    return mantissa, power


def plain(mantissa, power):
    """mantissa x 10^power written out, to three significant figures"""
    value = mantissa * 10.0 ** power
    return f"{value:.{max(0, 2 - power)}f}"


def length(log10_meters, unit="m", sign=1):
    """A length (or speed, with unit "m/s") in meters with an SI prefix, to three figures: 12.3 km, 4.56 µm. Past
    the prefixes, in powers of ten: 1.23e45 m."""
    if log10_meters is None:
        return f"0 {unit}"
    mantissa, power = figures(log10_meters)
    thousands = math.floor(power / 3)
    minus = "-" if sign < 0 else ""
    middle = PREFIXES.index(" ")
    if -middle <= thousands <= middle:
        prefix = PREFIXES[thousands + middle].strip()
        return f"{minus}{plain(mantissa, power - 3 * thousands)} {prefix}{unit}"
    return f"{minus}{mantissa:.2f}e{power} {unit}"


def factor(log10_factor, unit="x"):
    """A factor (a scale, a scale's change a second), to three figures: 0.0123x, 456x, 1.23e-15x"""
    mantissa, power = figures(log10_factor)
    if -3 <= power <= 4:
        return f"{plain(mantissa, power)}{unit}"
    return f"{mantissa:.2f}e{power}{unit}"
