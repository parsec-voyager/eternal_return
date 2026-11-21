"""Auto resolution, shared by the shader floors that work a view out at a share of its pixels (the Mandelbrot floor,
mandelbrot.py, and the sphere, sphere.py): the share drops a step while frames miss the display's refresh and comes back
up once they've kept up for a while.

The frame time is measured between draws of the view looking through the drone camera (or of any view, while none
does), so a second viewport's draws don't count as frames. Only frames drawn one after another, as in flight, change the
resolution: not a redraw after a pause."""

import collections
import time

# A frame slower than MISSED_FRAME missed the display's refresh (it took two, 33 ms at 60 Hz; flight jitters a few ms
# around 16.7 without missing any). MISSED_TO_LOWER of the last RECENT_FRAMES lower the resolution a step (a lone hitch
# doesn't; missed frames come between ones that make it, so not in a row), and FAST_FRAMES_TO_RAISE in a row without a
# miss raise it a step. A step up that turns out too slow doubles the wait before trying it again (up to
# MAX_RAISE_WAIT), so a resolution on the edge doesn't pulse.
MISSED_FRAME = 0.025
RECENT_FRAMES = 12
MISSED_TO_LOWER = 2
FAST_FRAMES_TO_RAISE = 60
MAX_RAISE_WAIT = 480
RESOLUTION_STEP = 0.05
MIN_RESOLUTION = 0.2
# Draws further apart than IN_FLIGHT seconds aren't frames in a row; ones further apart than PAUSE restart the average
IN_FLIGHT = 0.1
PAUSE = 0.5


class AutoResolution:
    """One floor's resolution and frame time. top: the object's "resolution" setting (the most it may draw at), on: its
    "auto_resolution" setting"""

    def __init__(self):
        self.resolution = None     # a multiple of RESOLUTION_STEP, up to top
        self.fast_frames = 0
        self.recent_missed = collections.deque(maxlen=RECENT_FRAMES)
        self.raise_wait = FAST_FRAMES_TO_RAISE
        self.raised = False        # the last step was up, so slowing down now means it was one too far
        self.last_draw = None
        self.frame_time = None     # seconds, a running average

    def current(self, top, on):
        """The resolution to draw at now"""
        top = clamped(top)
        if not on:
            return top
        if self.resolution is None or self.resolution > top:
            self.resolution = top
        return self.resolution

    def drawn(self, timed, top, on, now=None):
        """After (or as) a view draws. timed: whether this view's draws are the frames (it looks through the drone
        camera, or no view does yet)"""
        if not (timed or not self.frame_time):
            return
        now = time.perf_counter() if now is None else now
        if self.last_draw is not None:
            interval = now - self.last_draw
            self.frame_time = interval if self.frame_time is None or interval > PAUSE else (
                0.9 * self.frame_time + 0.1 * interval)
            if interval < IN_FLIGHT and on:
                self.adjust(interval, top)
        self.last_draw = now

    def adjust(self, interval, top):
        """After each frame of interval seconds"""
        top = clamped(top)
        current = self.current(top, True)
        missed = interval > MISSED_FRAME
        self.recent_missed.append(missed)
        self.fast_frames = 0 if missed else self.fast_frames + 1
        if sum(self.recent_missed) >= MISSED_TO_LOWER and current > MIN_RESOLUTION:
            lower = round((current - RESOLUTION_STEP) / RESOLUTION_STEP) * RESOLUTION_STEP
            self.resolution = max(MIN_RESOLUTION, lower)
            self.raise_wait = min(MAX_RAISE_WAIT, self.raise_wait * 2) if self.raised else FAST_FRAMES_TO_RAISE
            self.raised = False
            self.recent_missed.clear()
            self.fast_frames = 0
        elif self.fast_frames >= self.raise_wait and current < top:
            self.resolution = min(top, current + RESOLUTION_STEP)
            self.raised = True
            self.fast_frames = 0


def clamped(top):
    return min(1.0, max(MIN_RESOLUTION, float(top)))
