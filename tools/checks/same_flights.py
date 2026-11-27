"""same_flights.py <dir a> <dir b> [tolerance]: two runs of tools/same_flights.sh compared, flight by flight: whether
the camera's path (the log's time, frame, place, scale, clip end and selection) and the events (loops, wraps,
recenters) match, and how the frames differ as decoded pixels (ffmpeg): how many are identical, and the share of
pixels off by more than 8 of 255 in a channel, mean and worst frame. Run with Blender's Python (numpy).

The path must match exactly, unless a tolerance is given: then its numbers may differ by that share of their size
(1e-9, say), for a change that works the same numbers out another way (the engine's Rust against numpy: the last bit
of a sum can differ). A path that's off says by how much at most, and from when."""
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
    return [row.split(",")[:8] for row in (flight / "log.csv").read_text().splitlines()]


def path_off(a, b):
    """How far two camera paths are apart: (the biggest difference of a number as a share of its size, the first
    row that's off) or (inf, row) where they differ in something that isn't a number or in length; (0.0, None) the
    same."""
    worst, first = 0.0, None
    if len(a) != len(b):
        return float("inf"), min(len(a), len(b))
    for i, (x, y) in enumerate(zip(a, b)):
        if x == y:
            continue
        first = i if first is None else first
        for u, v in zip(x, y):
            if u != v:
                try:
                    u, v = float(u), float(v)
                    worst = max(worst, abs(u - v) / max(abs(u), abs(v), 1e-300))
                except ValueError:
                    return float("inf"), first
        if len(x) != len(y):
            return float("inf"), first
    return worst, first


def main():
    a, b = (pathlib.Path(p) for p in sys.argv[1:3])
    tolerance = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    same_paths, worst = True, 0.0
    for flight in sorted(p for p in a.iterdir() if p.is_dir()):
        other = b / flight.name
        if not (other / "log.csv").exists() or not (flight / "log.csv").exists():
            print(f"{flight.name:24} not flown in both")
            continue
        off_by, row = path_off(camera(flight), camera(other))
        path = off_by <= tolerance
        said = "same" if not off_by else f"{'within' if path else 'DIFFERS by'} {off_by:.1e} from row {row}"
        events = (flight / "events.txt").read_text() == (other / "events.txt").read_text()
        same_paths = same_paths and path and events
        x, y = pixels(flight), pixels(other)
        if x is None or y is None or x.shape != y.shape:
            print(f"{flight.name:24} camera {said}, events {'same' if events else 'DIFFER'}, frames can't be compared")
            same_paths = False
            continue
        off = (np.abs(x - y).max(axis=2) > OFF).mean(axis=1) * 100.0
        identical = int((np.abs(x - y).reshape(len(x), -1).max(axis=1) == 0).sum())
        worst = max(worst, float(off.max()))
        print(f"{flight.name:24} camera {said}, events {'same' if events else 'DIFFER'}, "
              f"{identical}/{len(x)} frames identical, pixels off: mean {off.mean():.3f}%, worst {off.max():.2f}% "
              f"(frame {int(off.argmax()) + 1})")
    verdict = "PATHS AND EVENTS SAME" if same_paths else "PATHS OR EVENTS DIFFER"
    print(f"{verdict}{f' (paths to within {tolerance:g})' if tolerance else ''}; worst frame {worst:.2f}% of pixels off")


main()
