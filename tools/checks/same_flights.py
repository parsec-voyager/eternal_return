"""same_flights.py <dir a> <dir b>: two runs of tools/same_flights.sh compared, flight by flight: whether the camera's
path (the log's time, frame, place, scale, clip end and selection) and the events (loops, wraps, recenters) match
exactly, and how the frames differ as decoded pixels (ffmpeg): how many are identical, and the share of pixels off by
more than 8 of 255 in a channel, mean and worst frame. Run with Blender's Python (numpy)."""
import pathlib
import subprocess
import sys

import numpy as np

OFF = 8


def pixels(flight):
    frames = sorted((flight / "frames").glob("*.png"))
    done = subprocess.run(["ffmpeg", "-v", "error", "-pattern_type", "glob", "-i", str(flight / "frames" / "*.png"),
                           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    data = np.frombuffer(done.stdout, dtype=np.uint8)
    return data.reshape(len(frames), -1, 3).astype(np.int16) if frames and data.size % len(frames) == 0 else None


def camera(flight):
    return [",".join(row.split(",")[:8]) for row in (flight / "log.csv").read_text().splitlines()]


def main():
    a, b = (pathlib.Path(p) for p in sys.argv[1:3])
    same_paths, worst = True, 0.0
    for flight in sorted(p for p in a.iterdir() if p.is_dir()):
        other = b / flight.name
        if not (other / "log.csv").exists() or not (flight / "log.csv").exists():
            print(f"{flight.name:24} not flown in both")
            continue
        path = camera(flight) == camera(other)
        events = (flight / "events.txt").read_text() == (other / "events.txt").read_text()
        same_paths = same_paths and path and events
        x, y = pixels(flight), pixels(other)
        if x is None or y is None or x.shape != y.shape:
            print(f"{flight.name:24} camera {'same' if path else 'DIFFERS'}, events {'same' if events else 'DIFFER'}, "
                  f"frames can't be compared")
            same_paths = False
            continue
        off = (np.abs(x - y).max(axis=2) > OFF).mean(axis=1) * 100.0
        identical = int((np.abs(x - y).reshape(len(x), -1).max(axis=1) == 0).sum())
        worst = max(worst, float(off.max()))
        print(f"{flight.name:24} camera {'same' if path else 'DIFFERS'}, events {'same' if events else 'DIFFER'}, "
              f"{identical}/{len(x)} frames identical, pixels off: mean {off.mean():.3f}%, worst {off.max():.2f}% "
              f"(frame {int(off.argmax()) + 1})")
    print(f"{'PATHS AND EVENTS SAME' if same_paths else 'PATHS OR EVENTS DIFFER'}; worst frame {worst:.2f}% of pixels off")


main()
