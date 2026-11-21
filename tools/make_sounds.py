"""Make the add-on's drone sounds (eternal_return/sound/, listed in operators.SOUNDS), synthesized from nothing but
numbers, so they're the add-on's own and can be made again:

    /Applications/Blender.app/Contents/Resources/5.2/python/bin/python3.13 tools/make_sounds.py [--out <dir>]

Blender's Python is used for its numpy. The same seed makes the same files.

Each sound is looped forever while flying (sound.PropellerSound), played from half its pitch at idle to 2.5 times at
full throttle. So:
- every loop is seamless: each oscillator makes a whole number of cycles in the loop (osc()), noise is shaped in the
  frequency domain of the whole loop (loop_noise()), so it's periodic too, and a hit's tail past the end wraps round to
  the start (add_wrapped())
- nothing is made above MAX_PARTIAL, as 2.5 times higher must stay audible and under Nyquist
- each is as loud (RMS) as the sound it replaced (DroneCam's propellers, the Pixabay ones), so the scenes' Sound Volume,
  tuned to those, still fits
"""

import argparse
import os
import wave

import numpy as np

SR = 44100
MAX_PARTIAL = 8000.0    # Hz: the highest partial or noise a sound is made with, 20 kHz at 2.5 times the pitch
SEED = 20261120


def seconds(n):
    return n / SR


def osc(freq, n, harmonics=((1, 1.0),), phase=0.0):
    """A sum of harmonics (multiple, amplitude) of a frequency that may vary (an array of n Hz, or a number), as n
    samples. The frequency is scaled a hair so the fundamental makes a whole number of cycles in the n samples: the
    loop's end then meets its start. A harmonic above MAX_PARTIAL at the frequency's highest is left out."""
    freq = np.broadcast_to(np.asarray(freq, float), (n,))
    cycles = freq.sum() / SR
    freq = freq * (max(round(cycles), 1) / cycles)
    # The phase at each sample's start, so sample 0 is at phase 0 and sample n would be at a whole number of cycles
    ph = 2 * np.pi * (np.concatenate(([0.0], np.cumsum(freq)[:-1])) / SR) + phase
    out = np.zeros(n)
    top = freq.max()
    for k, amp in harmonics:
        if k * top < MAX_PARTIAL:
            out += amp * np.sin(k * ph)
    return out


def loop_noise(rng, n, shape):
    """n samples of noise that loops seamlessly, its spectrum's magnitude shape(frequencies in Hz) (white noise
    filtered in the frequency domain of the whole loop, which is circular), cut off above MAX_PARTIAL"""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    spectrum *= shape(f) * (f < MAX_PARTIAL)
    out = np.fft.irfft(spectrum, n)
    return out / (np.sqrt((out ** 2).mean()) or 1.0)


def band(lo, hi, slope=2.0):
    """A band-pass shape for loop_noise() and filtered(): flat from lo to hi Hz, falling off slope-th order outside"""
    def shape(f):
        f = np.maximum(f, 1e-3)
        return 1.0 / np.sqrt((1 + (lo / f) ** (2 * slope)) * (1 + (f / hi) ** (2 * slope)))
    return shape


def filtered(x, shape):
    """x filtered by the magnitude shape(frequencies in Hz), circularly"""
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return np.fft.irfft(np.fft.rfft(x) * shape(f) * (f < MAX_PARTIAL), len(x))


def lfo(n, cycles, phase=0.0):
    """A sine making a whole number of cycles in n samples, from -1 to 1"""
    return np.sin(2 * np.pi * cycles * np.arange(n) / n + phase)


def add_wrapped(out, hit, at):
    """Add the samples hit into the loop out from sample at, the part past the end wrapping round to its start"""
    idx = (at + np.arange(len(hit))) % len(out)
    np.add.at(out, idx, hit)


def propellers(rng):
    """A quadcopter hovering, 4 s. Each of the four rotors' sound repeats at its shaft rate: strong at the blade pass
    (three blades, three times the shaft rate) and its multiples, weak at the rest (the blades' imbalance), with the
    motor's electrical whine (7 pole pairs) above. The rotors run at slightly different speeds, two pairs near 265 and
    315 Hz blade pass as DroneCam's had, so they beat against each other; each speed also wanders a little, as a flight
    controller holds a hover. Air rushing through the props is noise, pulsing with the blade pass."""
    n = 4 * SR
    out = np.zeros(n)
    for blade_pass, wander in ((262.0, 0.25), (268.5, 0.5), (311.75, 0.75), (318.0, 1.0)):
        shaft = blade_pass / 3
        # How fast it spins, wandering 0.6 % (whole cycles of the loop, so the loop still closes)
        freq = shaft * (1 + 0.006 * lfo(n, wander * 4, rng.uniform(0, 2 * np.pi)))
        harmonics = [(k, (1.0 if k % 3 == 0 else 0.18) / (k / 3) ** 1.1) for k in range(1, 60)]
        harmonics += [(7 * m, 0.25 / m) for m in (1, 2, 3)]
        rotor = osc(freq, n, harmonics, rng.uniform(0, 2 * np.pi))
        out += rotor * rng.uniform(0.8, 1.0)
        # The air through this rotor, louder each time a blade passes
        air = loop_noise(rng, n, band(400, 4000, 1.0))
        out += 0.9 * air * (1 + 0.6 * osc(freq * 3, n))
    return out


def spaceship(rng):
    """A big ship cruising, 8 s: a low engine drone (soft saws at 66 and 195 Hz, where the Pixabay one's peaks were),
    three voices of it a quarter-hertz apart so they slowly beat, a rumble of low noise swelling
    once a loop, and a faint shimmer of high partials fading in and out"""
    n = 8 * SR
    out = np.zeros(n)
    saw = [(k, 1.0 / k ** 1.3) for k in range(1, 80)]
    for detune in (-0.25, 0.0, 0.25):
        out += osc(66.0 + detune, n, saw, rng.uniform(0, 2 * np.pi))
        out += 0.6 * osc(2 * 97.5 + detune, n, saw[:20], rng.uniform(0, 2 * np.pi))
    out = filtered(out, band(30, 900, 1.0))
    out += 1.5 * loop_noise(rng, n, band(25, 300, 2.0)) * (1 + 0.5 * lfo(n, 1))
    shimmer = sum(osc(f, n) * (0.5 + 0.5 * lfo(n, c, rng.uniform(0, 2 * np.pi)))
                  for f, c in ((1170.0, 2), (1560.0, 3), (2340.0, 1)))
    out += 0.15 * shimmer
    return out


def ufo_8bit(rng):
    """An 8-bit game's flying saucer, 1.2 s: a square wave whose pitch warbles 10 times a second around 400 Hz, the
    pitch held for each 60th of a second as an old game's sound chip was set once a frame"""
    n = round(1.2 * SR)
    frame = SR // 60
    steps = np.arange(n) // frame
    warble = np.sin(2 * np.pi * 10 * steps / 60)
    freq = 400.0 * 2 ** (0.35 * warble)
    square = [(k, 1.0 / k) for k in range(1, 40, 2)]
    return osc(freq, n, square)


def breakbeat(rng):
    """A breakbeat, two bars at 172 BPM (2.79 s): kick, snare with ghost notes, and hats on the eighths with an open
    one closing the second bar, each hit a little late or early, as played. The throttle's pitch speeds it up."""
    beat = 60 / 172
    n = round(8 * beat * SR)
    step = n / 32
    out = np.zeros(n)

    def kick():
        t = np.arange(int(0.35 * SR)) / SR
        sweep = 50 + 110 * np.exp(-t / 0.03)
        body = np.sin(2 * np.pi * np.cumsum(sweep) / SR) * np.exp(-t / 0.18)
        click = filtered(rng.standard_normal(len(t)) * np.exp(-t / 0.003), band(1500, 6000))
        return body + 0.3 * click

    def snare(level):
        t = np.arange(int(0.25 * SR)) / SR
        tone = (np.sin(2 * np.pi * 190 * t) + 0.5 * np.sin(2 * np.pi * 330 * t)) * np.exp(-t / 0.05)
        rattle = filtered(rng.standard_normal(len(t)), band(1200, 7000)) * np.exp(-t / 0.12)
        return level * (0.6 * tone + 0.9 * rattle / np.abs(rattle).max())

    def hat(length):
        t = np.arange(int(length * 3 * SR)) / SR
        noise = filtered(rng.standard_normal(len(t)), band(6500, 8000, 3.0))
        return 0.25 * noise / np.abs(noise).max() * np.exp(-t / length)

    # (step of 32 sixteenths, hit)
    hits = [(s, kick()) for s in (0, 10, 16, 18, 26, 27)]
    hits += [(s, snare(1.0)) for s in (4, 12, 20, 28, 30)]
    hits += [(s, snare(0.25)) for s in (7, 9, 15, 23, 25)]
    hits += [(s, hat(0.03)) for s in range(0, 32, 2) if s != 30]
    hits += [(30, hat(0.15))]
    for s, hit in hits:
        late = rng.normal(0, 0.004) * SR if s else 0
        add_wrapped(out, hit, int(round(s * step + late)) % n)
    return np.tanh(1.5 * out) / 1.5


def alien(rng):
    """An alien craft, 8 s: an FM pad (a 177 Hz carrier, modulated at 1.5 times, its brightness breathing over the
    loop) under a theremin-like tone gliding about 257 Hz, a sub pulsing twice a second, and breathy noise moving
    between a low and a high band"""
    n = 8 * SR
    carrier = 177.0
    index = 2.0 + 1.5 * lfo(n, 1)
    modulator = osc(carrier * 1.5, n)
    # FM as a frequency wobble: carrier + index * modulator frequency * modulator
    pad = osc(carrier + index * carrier * 1.5 * modulator, n, [(1, 1.0), (2, 0.3)])
    pad += osc(carrier * 1.003, n, [(1, 0.5)])
    glide = 257.0 * 2 ** (0.25 * lfo(n, 2) + 0.03 * lfo(n, 40))
    theremin = osc(glide, n, [(1, 1.0), (2, 0.15), (3, 0.05)]) * (0.6 + 0.4 * lfo(n, 4, 1.0))
    sub = osc(44.25, n) * (0.5 + 0.5 * lfo(n, 16))
    fade = 0.5 + 0.5 * lfo(n, 1, 0.5)
    breath = fade * loop_noise(rng, n, band(150, 600)) + (1 - fade) * loop_noise(rng, n, band(2000, 5000))
    return 0.6 * pad + 0.5 * theremin + 0.8 * sub + 0.35 * breath


# (file, maker, the RMS of the sound it replaced, in full scale)
SOUNDS = (
    ("propellers.wav", propellers, 0.007),
    ("spaceship.wav", spaceship, 0.119),
    ("ufo_8bit.wav", ufo_8bit, 0.069),
    ("breakbeat.wav", breakbeat, 0.140),
    ("alien.wav", alien, 0.075),
)


def write_wav(path, x, rms):
    """Write x (a loop) as 16-bit mono, scaled to the RMS rms"""
    x = x - x.mean()
    x *= rms / np.sqrt((x ** 2).mean())
    if np.abs(x).max() > 0.98:
        raise ValueError(f"{path} would clip at RMS {rms}")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.round(x * 32767).astype("<i2").tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                      "eternal_return", "sound"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for name, make, rms in SOUNDS:
        x = make(rng)
        write_wav(os.path.join(args.out, name), x, rms)
        print(f"{name}: {seconds(len(x)):.2f} s")


if __name__ == "__main__":
    main()
