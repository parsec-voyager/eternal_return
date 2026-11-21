"""3D Tiles on sphere maps (sphere.py's "tiles" setting): a planet's surface streamed as OGC 3D Tiles, for Earth
Google's Photorealistic 3D Tiles, drawn on the sphere from the whole globe down to the street.

Where they are: a tileset's coordinates are Earth-centered (ECEF: meters, +z the north pole, +x longitude 0), which are
the map's axes too (planet.py). Every vertex is moved from WGS84's ellipsoid onto the map's sphere, of RADIUS meters,
keeping its latitude (geodetic, along the normal, the way gravity points, as planet.py reads its images), longitude
and height above the ellipsoid (warp()). So the map's sphere is the planet's sea level and its normal the planet's up
everywhere, for tidal lock and the horizon, and nothing's shape changes by more than the ellipsoid's 0.6% from a sphere.
A sphere map's R = rho N map units are RADIUS meters: s = rho N / RADIUS map units a meter.

The map's origin is the anchor's point on the sphere, lifted by the map's lift (sphere.py): the ground's height under
the drone, so the drone stays near the origin, in float32, over mountains as at sea. Each tile's vertices are kept in
float32 meters from its center, and each frame the center's offset from the camera, in map units, is worked out in
float64 and given to the shader as two float32s whose sum it is (hi + lo), so tiles don't jitter at any depth they're
drawn at. That's down to map units of DEEPEST_UNIT meters, where float64's 1e-9 m at Earth's radius begins to show:
sphere maps with tiles don't dive deeper (sphere.loop_factor()).

What's drawn (Tileset.select()): the tileset's tree walked from its root, each tile's screen-space error (its geometric
error over its distance to its bounding box, in pixels) against the target ("tiles_error"), with HYSTERESIS. A tile is
replaced by its children in view once they're loaded (REPLACE; ADD draws both); where one isn't yet, the tile stands in
for it, drawn only within its footprint (a fallback), so a tile coming into view never takes the finer ones beside it
back to their parent's detail. Tiles out of the view, or behind the horizon, are left out, but for those straight below
the camera, whose ground auto scale measures to, and refined tiles' children out of view are loaded too (prefetched).
What's needed is loaded in the order the view misses it: the drone's ground, then whatever is coarsest on screen, and
of that what's nearest the middle of the view. Detail that the view's motion smears (ground streaking past) isn't
loaded until it slows (MOTION_SHARP); what's loaded is drawn at any speed. Selections are made on a thread of their
own, the draw drawing the latest; a new one is started when the draw's done, and the selector keeps out of Python
while the flight's own runs (background.give_way()).

The tree is kept in arrays (Tileset, NODE_ARRAYS), a tileset's nodes a block of rows, and walked a depth at a time, each
depth's nodes at once. Flying on, the tilesets no selection has walked for the longest are freed past NODES, their rows
used again: kept, the tree grew by a few thousand nodes a second of flight, 2 GB in five minutes.

Loading (ProcessLoader, Loader): in a background Blender of its own, threads fetch (HTTPS kept alive, a connection
each), parse the glTF, decode Draco (with the bridge library Blender's glTF add-on ships, through ctypes), warp the
vertices and decode the texture (ImBuf), and the results come back over a socket: done in Blender itself, that Python
held Python's lock, and Blender's own Python, the flight's and the draw's, waited for it. The draw only makes their GPU
batches and textures, within LOAD_BUDGET seconds. A tileset comes with its root's tile (ROOT_TILES), and no more is
asked for a second than the server allows (RATE).
Tiles that haven't been needed for a while are dropped once their data passes MEMORY bytes, the ones a view from where
the camera is would miss least first.
Nothing is kept on disk: Google's terms allow storing their content only as their responses' cache headers say, so it's
kept in memory for the Blender session at most.

Google: requests carry the key (google_key(); it's never in a URL that's printed or stored). The root's request starts a
session, whose id its children's links carry; it's the only request Google bills (1,000 a month are free), made once
per tileset per Blender session, or again if the session expires. The attribution Google's terms require is drawn in
the view's corner (draw_attribution()): Google's name, then the data providers of the tiles drawn, most frequent first.
"""

import atexit
import bisect
import ctypes
import gzip
import http.client
import json
import math
import os
import pickle
import queue
import socket
import ssl
import struct
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections import Counter, deque

import numpy as np

try:
    from . import background
    from .background import give_way
except ImportError:          # the loader's process and tests import it by itself: nothing to give way to
    background = None

    def give_way():
        pass

# WGS84's ellipsoid, which tilesets' heights are measured from
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
# The map's sphere's radius, meters: Earth's mean (IUGG)
RADIUS = 6371008.8
# The lowest ground on Earth (the Dead Sea's shore is 430 m below sea level), for the horizon
LOWEST = -500.0
# The most a point's warp moves it (the ellipsoid is 14 km inside the sphere at the poles, 7 km outside at the equator,
# and the geodetic latitude is up to 0.19 degrees from the geocentric), meters: room around tiles too big to warp
WARP_SLACK = 30000.0
# A tile whose bounding sphere is bigger than this share of RADIUS is too big to warp its center alone
WARP_LIMIT = 0.05

# The finest map unit tiles are drawn at, meters (see the docstring): 2 mm, past the finest tiles' detail
DEEPEST_UNIT = 0.002

GOOGLE = "google"
GOOGLE_ROOT = "https://tile.googleapis.com/v1/3dtiles/root.json"
GOOGLE_HOST = "tile.googleapis.com"
# Where the Google Maps Platform key is read from: this environment variable, else this file (a line, the key alone)
KEY_ENV = "GOOGLE_MAPS_API_KEY"
KEY_FILE = "~/.config/eternal_return/google_maps_key"

# Loading: worker threads, requests queued or in flight at once, seconds a draw spends making GPU objects, a request's
# timeout, how long a failed tile waits to be asked for again (doubling each time, up to RETRY_MAX). A request mostly
# waits (Google's take 100 to 250 ms), so many at once: 6 took 5 s to fill a view over Zurich from nothing, and Google
# served 63 tiles a second to 12, 125 to 24, 164 to 48. Few more in flight than workers, so what's asked for next is
# what the latest selection needs most, not what one long ago did.
WORKERS = 48
IN_FLIGHT = 56
# The most requests a second, and how many may go at once after a pause. Google allows 12,000 tile requests a minute
# (https://developers.google.com/maps/documentation/tile/usage-and-billing), 200 a second, and answers 429 past it; a
# city flown through low wants more than that, so the connections alone (as many as were in flight over each one's
# 0.15 to 0.35 s) went over it in bursts.
RATE = 185.0
RATE_BURST = 40.0
DECODERS = 2
RAMP_FIRST = 4
RAMP_STEP = 0.05
LOAD_BUDGET = 0.002
TIMEOUT = 20.0
RETRY = 2.0
RETRY_MAX = 120.0
# Whose jobs the tiles' triangles' boxes made on the worker thread are (Tileset.loaded_ground(), background.py)
BOXES_OWNER = "tiles3d boxes"
# loaded_around() keeps the loaded tiles near a point this long (seconds), while it moves (and the reach grows) less
# than this share of the reach
AROUND_KEPT, AROUND_MOVE = 0.2, 0.1
# Bytes of tiles' data (textures and vertices on the GPU, their meshes here) kept before ones not drawn lately are
# dropped, and how many selections ago a tile was last drawn for it to count as lately. A view draws 300 to 450 tiles,
# 100 to 250 MB; Blender's footprint grows by half as much again as what's counted here, and with 1 GB kept it was
# 2.8 GB on an 8 GB machine.
MEMORY = 5.0e8
KEEP_SELECTIONS = 60
# A tile that comes in when the last STALE selections didn't want it isn't made (take_results())
STALE = 4
# Detail the view's motion smears isn't loaded (select()): a tile whose place moves up to MOTION_SHARP pixels a frame
# (of FRAME seconds) across the screen (it would cross a 2752-pixel view in six seconds) is refined to the target
# error, a faster one to as many times the target as it's faster, up to MOTION_MOST times (two levels coarser, at
# four times the speed: across the view in a second and a half). The view's motion is smoothed over MOTION_SMOOTH
# seconds. Flown low through Tokyo, two thirds of the tiles asked for were of ground crossing the screen at over 8
# pixels a frame, half at over 16, gone in a second or two: at Google's 200 requests a second they kept what the
# flight was headed for waiting.
MOTION_SHARP = 8.0
MOTION_MOST = 4.0
MOTION_SMOOTH = 0.15
FRAME = 1.0 / 60.0
# The default screen-space error a tile may have before it's refined, pixels ("tiles_error")
ERROR = 16.0
# The selection is made again once the camera moves this share of its distance to the nearest tile drawn, or turns
# about this many radians, or tiles were loaded or dropped (at most every SELECT_INTERVAL seconds for that). Made
# every frame, it cost Google's 100,000-tile trees 5 ms a frame. The view's sides are widened by SELECT_MARGIN (of
# its tangents) for it, so a turn up to SELECT_TURN still finds its edges' tiles selected.
SELECT_MOVE = 0.05
SELECT_TURN = 0.03
SELECT_INTERVAL = 0.1
SELECT_MARGIN = 1.1
# The least time between selections for moves and turns: in a dive, a selection every frame kept the selector thread,
# and Python's lock, busy a tenth of the time
SELECT_MIN_INTERVAL = 0.05
# Whether a tileset is loaded with its root's tile (Loader.decode())
ROOT_TILES = True
# Tiles fetched and decoded in a background Blender of their own (ProcessLoader); off, in Blender itself (Loader)
PROCESS = True
# Selections made on a thread of their own (Tileset.selection()); off, in the draw (tests that need the selection of
# the very view they render)
ASYNC = True
# The tree's nodes kept: past it, the tilesets no selection has walked for the longest are freed (Tileset.prune()), but
# not ones walked in the last PRUNE_KEEP selections. A view walks a few thousand nodes of a hundred or so tilesets, 341
# nodes each in Google's tree (five levels of a quadtree, the last one's 256 all links to more), so this keeps what the
# last few places flown over needed, in 50 MB.
NODES = 150000
PRUNE_KEEP = 30
# Python's switch interval while tilesets are streamed: how long a thread that let go of Python's lock (numpy, the GPU)
# may wait for another to give it back. Python's 5 ms, with the loader's threads busy, kept the draw waiting 20 ms and
# more, a wait for each of numpy's calls.
SWITCH_INTERVAL = 0.001
# A tile refined stays refined until its screen-space error is this many times under the target (select())
HYSTERESIS = 1.25
# Whether tiles out of view whose siblings are in view are loaded too, and as how many levels deeper than they are
# (select())
PREFETCH = True
PREFETCH_LEVELS = 3
# The ground the drone measures to (sphere.surface_distance()) is selected within this many times its distance to it
# (select()'s around), in view or not, and straight down from anywhere within this less one times it (how far the
# drone may fly until the next selection, about): drawn tiles alone, the ground under a drone climbing away from the
# planet (the view's tiles all behind it) was only ever the last selection's tile straight below, which it flew off
# between selections, and auto scale measured to the sphere instead, kilometers below the Alps' ground
AROUND = 1.25
# The most holes a tile stands in for, clipped to their footprints (select()), and so its shader's room for them
MAX_HOLES = 8
# Root requests again for a tileset whose session has expired: tile requests failing this many times in a row, the
# session at least SESSION_MIN seconds old
SESSION_FAILURES = 24
SESSION_MIN = 1800.0

# glTF
COMPONENTS = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
NORMALIZED = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0}
# glTF is y-up, 3D Tiles z-up: a glTF's (x, y, z) is the tile's (x, -z, y) (3D Tiles' "glTF transforms")
Y_UP = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


# ----- Earth -----

def geodetic(p):
    """ECEF points (..., 3), meters: their geodetic latitude, longitude (radians) and height over WGS84's ellipsoid
    (meters). The latitude by fixed-point iteration from Bowring's guess, each step e^2 = 0.0067 times closer, so 5 are
    float64's; the height as r cos(lat) + z sin(lat) - a sqrt(1 - e^2 sin^2(lat)), which doesn't cancel."""
    p = np.asarray(p, dtype=np.float64)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    lon = np.arctan2(y, x)
    r = np.hypot(x, y)
    lat = np.arctan2(z, r * (1.0 - WGS84_E2))
    for _ in range(5):
        s = np.sin(lat)
        lat = np.arctan2(z + WGS84_E2 * WGS84_A / np.sqrt(1.0 - WGS84_E2 * s * s) * s, r)
    s, c = np.sin(lat), np.cos(lat)
    h = r * c + z * s - WGS84_A * np.sqrt(1.0 - WGS84_E2 * s * s)
    return lat, lon, h


def ecef(lat, lon, h):
    """Geodetic latitude, longitude (radians) and height (meters) to ECEF points (..., 3)"""
    lat, lon, h = (np.asarray(v, dtype=np.float64) for v in (lat, lon, h))
    s = np.sin(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * s * s)
    return np.stack([(n + h) * np.cos(lat) * np.cos(lon), (n + h) * np.cos(lat) * np.sin(lon),
                     (n * (1.0 - WGS84_E2) + h) * s], axis=-1)


def on_sphere(lat, lon, h):
    """The map's sphere's point at a latitude, longitude and height (meters, from its center)"""
    lat, lon, h = (np.asarray(v, dtype=np.float64) for v in (lat, lon, h))
    r = RADIUS + h
    c = np.cos(lat)
    return np.stack([r * c * np.cos(lon), r * c * np.sin(lon), r * np.sin(lat)], axis=-1)


def warp(p):
    """ECEF points onto the map's sphere: the same latitude, longitude and height (see the docstring)"""
    return on_sphere(*geodetic(p))


def bounding_sphere(volume, transform=None):
    """A tile's boundingVolume (box, sphere or region) as a sphere around its points once warped: center, radius
    (meters). Small ones' centers are warped and their radii grown by the warp's most stretch (the ellipsoid's radii of
    curvature are 6335 to 6400 km, so 1.01) and its bend over them; big ones are left where they are, with
    WARP_SLACK."""
    if "region" in volume:
        west, south, east, north, low, high = (float(v) for v in volume["region"])
        if east < west:
            east += 2.0 * math.pi
        if max(east - west, north - south) > 0.5:
            return np.zeros(3), RADIUS + high
        lat, lon = np.meshgrid(np.linspace(south, north, 5), np.linspace(west, east, 5))
        points = np.concatenate([on_sphere(lat, lon, np.full(lat.shape, h)).reshape(-1, 3) for h in (low, high)])
        center = 0.5 * (points.min(axis=0) + points.max(axis=0))
        radius = float(np.max(np.linalg.norm(points - center, axis=1)))
        return center, radius * 1.01 + 1.0
    if "box" in volume:
        b = np.asarray(volume["box"], dtype=np.float64)
        center, axes = b[:3], b[3:12].reshape(3, 3)
        if transform is not None:
            center = transform[:3, :3] @ center + transform[:3, 3]
            axes = axes @ transform[:3, :3].T
        radius = float(np.sqrt(np.sum(axes * axes)))
    elif "sphere" in volume:
        b = np.asarray(volume["sphere"], dtype=np.float64)
        center, radius = b[:3], float(b[3])
        if transform is not None:
            center = transform[:3, :3] @ center + transform[:3, 3]
            radius *= float(np.max(np.linalg.norm(transform[:3, :3], axis=0)))
    else:
        raise ValueError(f"bounding volume {sorted(volume)}")
    if radius > WARP_LIMIT * RADIUS or np.linalg.norm(center) < 0.5 * RADIUS:
        return center, radius + WARP_SLACK
    return warp(center), radius * 1.01 + radius * radius / RADIUS + 1.0


def box_spheres(boxes):
    """bounding_sphere() of many untransformed boxes (m, 12) at once: centers (m, 3), radii (m,); and the boxes
    themselves once warped, as their axes (m, 3, 3: unit rows) and half lengths along them (m, 3). The warp keeps a
    small box's axes (latitude, longitude and height stay, and the sphere's up at a latitude is the ellipsoid's normal
    there), stretching it by at most 1% and bending it by its size squared over RADIUS; big ones are cubes around their
    spheres. (Google's tiles' skirts reach out of their boxes sideways, by up to 1.7% of their radius: the ground
    itself doesn't.)"""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 12)
    center, axes = boxes[:, :3], boxes[:, 3:12]
    radius = np.sqrt(np.sum(axes * axes, axis=1))
    big = (radius > WARP_LIMIT * RADIUS) | (np.linalg.norm(center, axis=1) < 0.5 * RADIUS)
    centers = np.where(big[:, None], center, warp(center))
    radii = np.where(big, radius + WARP_SLACK, radius * 1.01 + radius * radius / RADIUS + 1.0)
    rows = axes.reshape(-1, 3, 3)
    lengths = np.linalg.norm(rows, axis=2)
    units = rows / np.maximum(lengths, 1e-300)[:, :, None]
    halves = lengths * 1.01 + (radius * radius / RADIUS + 1.0)[:, None]
    # A flat box's missing axis (zero length): any unit vector across the other two
    flat = lengths < 1e-9
    if flat.any():
        for i, k in zip(*np.nonzero(flat)):
            others = [units[i, j] for j in range(3) if j != k]
            normal = np.cross(*others)
            units[i, k] = normal / max(np.linalg.norm(normal), 1e-300)
    units[big] = np.eye(3)
    halves[big] = radii[big, None]
    return centers, radii, units, halves


class Records:
    """A tileset's JSON flattened (flatten()), its tiles breadth first, so each one's children are in a row: each
    one's parent (an index into these, or -1 for the root, whose parent is the node that linked to the tileset), its
    first child and how many it has, its depth under the root, content URL (or None), kind (NONE, GLB, JSON), whether
    it adds to its parent (ADD) or replaces it, geometric error, cumulative transform (transforms is None if no tile
    has one), and bounding sphere and box (warped, box_spheres())"""

    __slots__ = ("parents", "first_child", "child_count", "depths", "urls", "kinds", "adds", "errors", "transforms",
                 "centers", "radii", "axes", "halves")


def flatten(tileset, url, transform, add, error):
    """A tileset's JSON (linked from a node with this transform, refinement (add: ADD) and error) as Records. On a worker
    thread: Google's tilesets have hundreds of tiles each, and making them one by one in the draw took 42 ms a
    tileset, mostly URL joins and one warp per tile; here the warps are made at once, and Google's links are joined by
    hand (link())."""
    records = Records()
    parents, first_child, child_count, depths, urls, kinds, adds, errors, transforms, volumes = (
        [], [], [], [], [], [], [], [], [], [])
    parts = urllib.parse.urlsplit(url)
    host = f"{parts.scheme}://{parts.netloc}" if parts.netloc else None
    folder = urllib.parse.urlunsplit(parts._replace(path=parts.path.rsplit("/", 1)[0] + "/", query="", fragment=""))
    inherited = [(name, f"{name}={urllib.parse.quote_plus(value)}")
                 for name, value in urllib.parse.parse_qsl(parts.query)]

    def link(uri):
        """What resolve() makes of a link, by hand for the plain ones (Google's: paths from the host, or names in the
        tileset's own folder; in the root's tileset carrying the session, in the others not): resolve()'s URL joins
        were 75% of the tilesets' decoding, Python's lock held all along"""
        if host and uri.startswith("/") and not uri.startswith("//"):
            joined = host + uri
        elif "/" not in uri.split("?", 1)[0] and ":" not in uri and not uri.startswith((".", "#")):
            joined = folder + uri
        else:
            return resolve(url, uri)
        if "#" in joined:
            return resolve(url, uri)
        query = joined.split("?", 1)[1] if "?" in joined else ""
        own = {pair.split("=", 1)[0] for pair in query.split("&")} if query else set()
        missing = "&".join(pair for name, pair in inherited if name not in own)
        if not missing:
            return joined
        return joined + ("&" if query else ("" if joined.endswith("?") else "?")) + missing

    # Breadth first: a tile's children are given their places as it's taken, the next ones free, in a row
    waiting = deque([(tileset["root"], -1, transform, bool(add), error, 0)])
    placed = 1
    while waiting:
        tile, parent, above_transform, above_add, above_error, depth = waiting.popleft()
        tile_transform = above_transform
        if "transform" in tile:
            local = np.asarray(tile["transform"], dtype=np.float64).reshape(4, 4).T
            tile_transform = local if above_transform is None else above_transform @ local
        content = tile.get("content") or (tile.get("contents") or [None])[0]
        uri = (content or {}).get("uri") or (content or {}).get("url")
        content_url = link(uri) if uri else None
        index = len(urls)
        parents.append(parent)
        urls.append(content_url)
        kinds.append(content_kind(content_url) if content_url else NONE)
        tile_add = str(tile["refine"]).upper() == "ADD" if "refine" in tile else above_add
        tile_error = float(tile.get("geometricError", above_error))
        adds.append(tile_add)
        errors.append(tile_error)
        depths.append(depth)
        transforms.append(tile_transform)
        volumes.append(tile["boundingVolume"])
        children = tile.get("children", ())
        first_child.append(placed)
        child_count.append(len(children))
        placed += len(children)
        waiting.extend((child, index, tile_transform, tile_add, tile_error, depth + 1) for child in children)
    centers, radii = np.zeros((len(urls), 3)), np.zeros(len(urls))
    # Other volumes than plain boxes: cubes around their spheres
    axes = np.tile(np.eye(3), (len(urls), 1, 1))
    halves = np.zeros((len(urls), 3))
    plain = [i for i, (v, t) in enumerate(zip(volumes, transforms)) if "box" in v and t is None]
    if plain:
        centers[plain], radii[plain], axes[plain], halves[plain] = box_spheres([volumes[i]["box"] for i in plain])
    for i in set(range(len(urls))) - set(plain):
        centers[i], radii[i] = bounding_sphere(volumes[i], transforms[i])
        halves[i] = radii[i]
    records.parents, records.first_child = np.array(parents, np.int32), np.array(first_child, np.int32)
    records.child_count, records.depths = np.array(child_count, np.int32), np.array(depths, np.int32)
    records.urls, records.kinds, records.adds = urls, np.array(kinds, np.int8), np.array(adds, bool)
    records.errors, records.centers, records.radii = np.array(errors), centers, radii
    records.transforms = transforms if any(t is not None for t in transforms) else None
    records.axes, records.halves = axes, halves
    return records


# ----- the key, URLs and fetching -----

def google_key():
    """The Google Maps Platform key: KEY_ENV's, else KEY_FILE's first line, else None"""
    key = os.environ.get(KEY_ENV, "").strip()
    if key:
        return key
    try:
        with open(os.path.expanduser(KEY_FILE)) as f:
            return f.readline().strip() or None
    except OSError:
        return None


def resolve(base, uri):
    """A tile's content URI resolved against the URL of the tileset it's in, which may be a path from the host alone
    (Google's), with the base's query parameters it doesn't have itself (a session, as clients of 3D Tiles keep them)"""
    url = urllib.parse.urljoin(base, uri)
    base_query = urllib.parse.parse_qsl(urllib.parse.urlsplit(base).query)
    if not base_query:
        return url
    parts = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parts.query)
    names = {name for name, _value in query}
    query += [(name, value) for name, value in base_query if name not in names]
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))


def is_web(url):
    return urllib.parse.urlsplit(url).scheme in ("http", "https")


class FetchError(Exception):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class Fetcher:
    """GETs URLs (or reads files), the key added to each web request's query; a kept-alive connection per thread and
    host. Errors never show the key."""

    def __init__(self, key=None):
        self.key = key
        self.local = threading.local()
        self.context = ssl.create_default_context()
        self.requests = 0
        self.received = 0

    def redact(self, text):
        return text.replace(self.key, "<key>") if self.key else text

    def get(self, url):
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in ("http", "https"):
            path = urllib.request.url2pathname(parts.path) if parts.scheme == "file" else url
            with open(path, "rb") as f:
                return f.read()
        query = parts.query
        if self.key:
            query = (query + "&" if query else "") + urllib.parse.urlencode({"key": self.key})
        target = (parts.path or "/") + ("?" + query if query else "")
        connections = self.local.__dict__.setdefault("connections", {})
        where = (parts.scheme, parts.netloc)
        for attempt in range(2):
            connection = connections.get(where)
            if connection is None:
                if parts.scheme == "https":
                    connection = http.client.HTTPSConnection(parts.netloc, timeout=TIMEOUT, context=self.context)
                else:
                    connection = http.client.HTTPConnection(parts.netloc, timeout=TIMEOUT)
                connections[where] = connection
            try:
                connection.request("GET", target, headers={"Accept-Encoding": "gzip",
                                                           "User-Agent": "eternal_return (Blender add-on)"})
                response = connection.getresponse()
                data = response.read()
            except (OSError, http.client.HTTPException) as error:
                # A kept-alive connection the server closed: once more on a new one
                connection.close()
                del connections[where]
                if attempt:
                    raise FetchError(self.redact(f"{parts.netloc}: {error}")) from None
                continue
            self.requests += 1
            self.received += len(data)
            if response.getheader("Content-Encoding", "") == "gzip":
                data = gzip.decompress(data)
            if response.status != 200:
                raise FetchError(self.redact(f"HTTP {response.status}: {error_message(data)}"), response.status)
            return data
        raise FetchError("unreachable")

    def close(self):
        for connection in self.local.__dict__.get("connections", {}).values():
            connection.close()


def error_message(data):
    """A web API's error response's message (Google's JSON, sometimes in a list), else its first bytes"""
    try:
        body = json.loads(data)
        if isinstance(body, list) and body:
            body = body[0]
        return str(body["error"]["message"])
    except (ValueError, KeyError, TypeError, IndexError):
        return data[:200].decode("utf-8", "replace")


# ----- glTF content -----

def read_glb(data):
    """A binary glTF's JSON and binary chunk (a memoryview); b3dm (3D Tiles 1.0's batched models) unwrapped first,
    with its RTC_CENTER"""
    view = memoryview(data)
    rtc = None
    if bytes(view[:4]) == b"b3dm":
        _magic, _version, length, ft_json, ft_bin, bt_json, bt_bin = struct.unpack_from("<4s6I", view, 0)
        if ft_json:
            feature_table = json.loads(bytes(view[28:28 + ft_json]))
            rtc = feature_table.get("RTC_CENTER")
        view = view[28 + ft_json + ft_bin + bt_json + bt_bin:length]
    magic, version, length = struct.unpack_from("<4sII", view, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError("not a glTF 2 binary")
    offset, gltf, binary = 12, None, memoryview(b"")
    while offset + 8 <= length:
        chunk_length, chunk_type = struct.unpack_from("<II", view, offset)
        chunk = view[offset + 8:offset + 8 + chunk_length]
        if chunk_type == 0x4E4F534A:
            gltf = json.loads(bytes(chunk))
        elif chunk_type == 0x004E4942:
            binary = chunk
        offset += 8 + chunk_length
    if gltf is None:
        raise ValueError("glTF without JSON")
    if rtc is not None:
        gltf.setdefault("extensions", {}).setdefault("CESIUM_RTC", {"center": rtc})
    return gltf, binary


def accessor_array(gltf, binary, index):
    """An accessor's values, (count, width), as stored (normalized integers to floats)"""
    accessor = gltf["accessors"][index]
    width = WIDTHS[accessor["type"]]
    component = accessor["componentType"]
    dtype = np.dtype(COMPONENTS[component])
    count = accessor["count"]
    if "bufferView" not in accessor:
        return np.zeros((count, width), dtype)
    view = gltf["bufferViews"][accessor["bufferView"]]
    if view.get("buffer", 0) != 0:
        raise ValueError("glTF with external buffers")
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride") or dtype.itemsize * width
    values = np.ndarray((count, width), dtype, buffer=binary, offset=start, strides=(stride, dtype.itemsize)).copy()
    if accessor.get("normalized") and component in NORMALIZED:
        values = np.maximum(values / NORMALIZED[component], -1.0)
    return values


_draco = None


def draco():
    """The Draco bridge library of Blender's glTF add-on (ctypes, set up once)"""
    global _draco
    if _draco is not None:
        return _draco
    import importlib.util
    spec = importlib.util.find_spec("io_scene_gltf2")
    if spec is None or spec.origin is None:
        raise FileNotFoundError("Blender's glTF add-on (its Draco library)")
    folder = os.path.dirname(spec.origin)
    name = {"win32": "bf_intern_draco_bridge.dll", "darwin": "libbf_intern_draco_bridge.dylib"}.get(
        sys.platform, "libbf_intern_draco_bridge.so")
    candidates = [os.path.join(folder, name)]
    try:
        import bpy
        system_libs = bpy.utils.resource_path('SYSTEM_LIBS') if hasattr(bpy.utils, "resource_path") else ""
        local = bpy.utils.resource_path('LOCAL')
        if system_libs:
            candidates.append(os.path.join(system_libs, "scripts", "addons_core", "io_scene_gltf2", name))
        if local:
            candidates.append(os.path.join(os.path.dirname(local), "lib", name))
    except (ImportError, AttributeError, TypeError):
        pass
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(f"Draco library {name}")
    dll = ctypes.cdll.LoadLibrary(path)
    for function, restype, argtypes in (
            ("decoderCreate", ctypes.c_void_p, []),
            ("decoderRelease", None, [ctypes.c_void_p]),
            ("decoderDecode", ctypes.c_bool, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]),
            ("decoderReadAttribute", ctypes.c_bool, [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t,
                                                     ctypes.c_char_p]),
            ("decoderGetVertexCount", ctypes.c_uint32, [ctypes.c_void_p]),
            ("decoderGetIndexCount", ctypes.c_uint32, [ctypes.c_void_p]),
            ("decoderAttributeIsNormalized", ctypes.c_bool, [ctypes.c_void_p, ctypes.c_uint32]),
            ("decoderGetAttributeByteLength", ctypes.c_size_t, [ctypes.c_void_p, ctypes.c_uint32]),
            ("decoderCopyAttribute", None, [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]),
            ("decoderReadIndices", ctypes.c_bool, [ctypes.c_void_p, ctypes.c_size_t]),
            ("decoderGetIndicesByteLength", ctypes.c_size_t, [ctypes.c_void_p]),
            ("decoderCopyIndices", None, [ctypes.c_void_p, ctypes.c_void_p])):
        f = getattr(dll, function)
        f.restype = restype
        f.argtypes = argtypes
    _draco = dll
    return dll


def draco_decode(gltf, binary, primitive):
    """A KHR_draco_mesh_compression primitive's indices (uint32) and attributes (as its accessors say), decoded"""
    dll = draco()
    extension = primitive["extensions"]["KHR_draco_mesh_compression"]
    view = gltf["bufferViews"][extension["bufferView"]]
    start = view.get("byteOffset", 0)
    data = bytes(binary[start:start + view["byteLength"]])
    decoder = dll.decoderCreate()
    try:
        if not dll.decoderDecode(decoder, data, len(data)):
            raise ValueError("Draco can't decode the primitive")
        if not dll.decoderReadIndices(decoder, 5125):
            raise ValueError("Draco can't read the indices")
        indices = np.empty(dll.decoderGetIndicesByteLength(decoder) // 4, np.uint32)
        dll.decoderCopyIndices(decoder, indices.ctypes.data)
        attributes = {}
        for name, draco_id in extension["attributes"].items():
            if name not in primitive["attributes"]:
                continue
            accessor = gltf["accessors"][primitive["attributes"][name]]
            component, width = accessor["componentType"], WIDTHS[accessor["type"]]
            if not dll.decoderReadAttribute(decoder, draco_id, component, accessor["type"].encode()):
                raise ValueError(f"Draco can't read {name}")
            dtype = np.dtype(COMPONENTS[component])
            values = np.empty(dll.decoderGetAttributeByteLength(decoder, draco_id) // dtype.itemsize, dtype)
            dll.decoderCopyAttribute(decoder, draco_id, values.ctypes.data)
            values = values.reshape(-1, width)
            if (accessor.get("normalized") or dll.decoderAttributeIsNormalized(decoder, draco_id)) \
                    and component in NORMALIZED:
                values = np.maximum(values / NORMALIZED[component], -1.0)
            attributes[name] = values
    finally:
        dll.decoderRelease(decoder)
    return indices, attributes


def node_matrix(node):
    """A glTF node's local matrix (its matrix, or translation, rotation and scale)"""
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
    m = np.eye(4)
    x, y, z, w = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
    m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    m[:3, :3] = m[:3, :3] * np.asarray(node.get("scale", (1.0, 1.0, 1.0)))[None, :]
    m[:3, 3] = node.get("translation", (0.0, 0.0, 0.0))
    return m


def mesh_instances(gltf):
    """Each mesh the glTF's scene draws, with its node's matrix (glTF coordinates)"""
    nodes = gltf.get("nodes", [])
    scenes = gltf.get("scenes")
    if scenes:
        roots = scenes[gltf.get("scene", 0)].get("nodes", [])
    else:
        children = {c for n in nodes for c in n.get("children", [])}
        roots = [i for i in range(len(nodes)) if i not in children]
    if not nodes:
        return [(np.eye(4), mesh) for mesh in gltf.get("meshes", [])]
    out = []
    stack = [(i, np.eye(4)) for i in roots]
    while stack:
        i, parent = stack.pop()
        node = nodes[i]
        matrix = parent @ node_matrix(node)
        if "mesh" in node:
            out.append((matrix, gltf["meshes"][node["mesh"]]))
        stack.extend((c, matrix) for c in node.get("children", []))
    return out


def base_color_image(gltf, binary, primitive):
    """A primitive's base color texture: (its image's bytes, its texture coordinate set, KHR_texture_transform), or
    None. WebP's extension is used if there, ImBuf reads it; formats it doesn't (KTX2) are left out."""
    material = gltf.get("materials", [])[primitive["material"]] if "material" in primitive else None
    info = (material or {}).get("pbrMetallicRoughness", {}).get("baseColorTexture")
    if info is None:
        return None
    texture = gltf["textures"][info["index"]]
    source = texture.get("extensions", {}).get("EXT_texture_webp", {}).get("source", texture.get("source"))
    if source is None:
        return None
    image = gltf["images"][source]
    if "bufferView" in image:
        view = gltf["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        data = bytes(binary[start:start + view["byteLength"]])
    elif image.get("uri", "").startswith("data:"):
        import base64
        data = base64.b64decode(image["uri"].split(",", 1)[1])
    else:
        return None
    return data, info.get("texCoord", 0), info.get("extensions", {}).get("KHR_texture_transform")


def texture_transform(uv, transform):
    """KHR_texture_transform applied to texture coordinates"""
    if not transform:
        return uv
    sx, sy = transform.get("scale", (1.0, 1.0))
    r = transform.get("rotation", 0.0)
    ox, oy = transform.get("offset", (0.0, 0.0))
    c, s = math.cos(r), math.sin(r)
    u, v = uv[:, 0] * sx, uv[:, 1] * sy
    return np.stack([c * u + s * v + ox, -s * u + c * v + oy], axis=1).astype(np.float32)


def decode_image(data, as_bytes=False):
    """An image's pixels, RGBA float32 0..1 (or uint8, as_bytes), bottom-up rows (ImBuf's), and its (width, height)"""
    import imbuf
    ib = imbuf.load_from_buffer(data)
    try:
        width, height = ib.size
        with ib.with_buffer() as buffer:
            pixels = (np.array(buffer, dtype=np.uint8) if as_bytes
                      else np.asarray(buffer, dtype=np.float32) * (1.0 / 255.0))
    finally:
        ib.free()
    return pixels.reshape(height, width, 4), (width, height)


class Content:
    """A tile's content, decoded off the main thread: its center (warped, float64 meters), and per part (a glTF
    primitive) its vertices in float32 meters from the center, texture coordinates (v up, as ImBuf's rows), triangles
    and texture's pixels (or None); the glTF's copyright"""

    __slots__ = ("center", "parts", "copyright", "radius", "mesh")

    def __init__(self, center, parts, copyright, radius):
        self.center, self.parts, self.copyright, self.radius = center, parts, copyright, radius
        self.mesh = None        # its parts' mesh in one (content_mesh()), if made already (prepare())


def content_mesh(parts):
    """Every part's vertices and triangles in one, for Tile.ground() and Tile.nearest()"""
    vertices, triangles, first = [], [], 0
    for positions, _uv, part_triangles, _pixels in parts:
        vertices.append(positions)
        triangles.append(part_triangles.astype(np.int32) + first)
        first += len(positions)
    return (np.concatenate(vertices) if vertices else np.zeros((0, 3), np.float32),
            np.concatenate(triangles) if triangles else np.zeros((0, 3), np.int32))


def prepare(content):
    """What of a Tile can be made before Blender's main thread makes it (ProcessLoader's receive thread): its textures'
    pixels as floats, the only pixels Blender makes textures from (numpy lets go of Python's lock for it: in the draw,
    a tenth of a millisecond a tile), and its mesh (content_mesh())"""
    parts = []
    for positions, uv, triangles, pixels in content.parts:
        if pixels is not None and pixels[0].dtype == np.uint8:
            pixels = (np.multiply(pixels[0], np.float32(1.0 / 255.0), dtype=np.float32), pixels[1])
        parts.append((positions, uv, triangles, pixels))
    content.parts = parts
    content.mesh = content_mesh(parts)
    return content


def decode_content(data, transform=None, as_bytes=False):
    """A tile's glTF (glb or b3dm) as Content (its textures' pixels as uint8 with as_bytes, else float32)"""
    gltf, binary = read_glb(data)
    rtc = gltf.get("extensions", {}).get("CESIUM_RTC", {}).get("center")
    outer = np.eye(4) if transform is None else np.asarray(transform, dtype=np.float64)
    if rtc is not None:
        moved = np.eye(4)
        moved[:3, 3] = rtc
        outer = outer @ moved
    pieces = []
    for matrix, mesh in mesh_instances(gltf):
        M = outer @ Y_UP @ matrix
        for primitive in mesh.get("primitives", []):
            if primitive.get("mode", 4) != 4 or "POSITION" not in primitive.get("attributes", {}):
                continue
            image = base_color_image(gltf, binary, primitive)
            uv_name = f"TEXCOORD_{image[1]}" if image else "TEXCOORD_0"
            if "KHR_draco_mesh_compression" in primitive.get("extensions", {}):
                indices, attributes = draco_decode(gltf, binary, primitive)
                positions, uv = attributes["POSITION"], attributes.get(uv_name)
            else:
                positions = accessor_array(gltf, binary, primitive["attributes"]["POSITION"])
                uv = (accessor_array(gltf, binary, primitive["attributes"][uv_name])
                      if uv_name in primitive["attributes"] else None)
                indices = (accessor_array(gltf, binary, primitive["indices"]).ravel().astype(np.uint32)
                           if "indices" in primitive else np.arange(len(positions), dtype=np.uint32))
            P = positions.astype(np.float64) @ M[:3, :3].T + M[:3, 3]
            if uv is None:
                uv = np.zeros((len(P), 2), np.float32)
            uv = texture_transform(uv.astype(np.float32), image[2] if image else None)
            # glTF's v runs down the image, ImBuf's rows up
            uv[:, 1] = 1.0 - uv[:, 1]
            pieces.append((warp(P), uv, indices.reshape(-1, 3), image[0] if image else None))
    if not pieces:
        return Content(np.zeros(3), [], gltf.get("asset", {}).get("copyright", ""), 0.0)
    everything = np.concatenate([p[0] for p in pieces])
    center = 0.5 * (everything.min(axis=0) + everything.max(axis=0))
    radius = float(np.max(np.linalg.norm(everything - center, axis=1)))
    parts = []
    for W, uv, triangles, image in pieces:
        pixels = decode_image(image, as_bytes) if image is not None else None
        parts.append(((W - center).astype(np.float32), uv, triangles, pixels))
    return Content(center, parts, gltf.get("asset", {}).get("copyright", ""), radius)


# ----- the tree -----

# A node's state (its content's, or for a tileset's link its tileset's) and kind: no content, a model ("glb"), or a
# tileset of its own ("json"), whose root becomes its child once loaded
NEW, QUEUED, READY, FAILED = range(4)
NONE, GLB, JSON = range(3)
# The tree's arrays (Tileset), a row a node: name -> (type, a row's shape, an empty row's value). Its bounding sphere and
# box (box_spheres()) and geometric error; whether the last selection that visited it refined it (select()'s
# hysteresis); its parent (-1: the root), children (in a row) and depth; kind, whether it adds to its parent (ADD) or
# replaces it, state, whether its Tile is loaded, and the last selection that needed it; its tileset's block; and how
# many times its row has been freed (a load asked for before isn't its own any more).
NODE_ARRAYS = {
    "centers": (np.float64, (3,), 0.0), "radii": (np.float64, (), 0.0), "axes": (np.float64, (3, 3), 0.0),
    "halves": (np.float64, (3,), 0.0), "errors": (np.float64, (), 0.0), "refined": (bool, (), False),
    "parents": (np.int32, (), -1), "first_child": (np.int32, (), 0), "child_count": (np.int32, (), 0),
    "depths": (np.int32, (), 0), "kinds": (np.int8, (), NONE), "adds": (bool, (), False),
    "states": (np.int8, (), NEW), "has_tile": (bool, (), False), "drawn": (np.int64, (), -1),
    "block_of": (np.int32, (), 0), "epochs": (np.int32, (), 0),
    "hole_sums": (np.int32, (), 0), "shown_sums": (np.int32, (), 0), "marks": (np.int8, (), 0)}
# A selection's own (Tileset.walked()): its nodes' children's holes, the ones of them in view, and marks
SCRATCH = ("hole_sums", "shown_sums", "marks")
REPLACED, COVERED = 1, 2


def content_kind(url):
    path = url.split("?", 1)[0].split("#", 1)[0].lower()
    return JSON if path.endswith(".json") else GLB


class Tile:
    """A tile's content on the GPU: per part, a batch and a texture; and what the ground under the drone is found
    from (ground()): the parts' vertices and triangles, and per triangle its 2D box seen from the tile's up"""

    def __init__(self, content, shader=None):
        self.center = content.center
        self.radius = content.radius
        self.copyright = content.copyright
        self.parts = []
        self.bytes = 0
        for positions, uv, triangles, pixels in content.parts:
            if not len(triangles):
                continue
            self.bytes += positions.nbytes + uv.nbytes + triangles.nbytes
            if pixels is not None:
                self.bytes += pixels[1][0] * pixels[1][1] * 4
            # Without a shader (tests in background mode), only what ground() needs
            if shader is not None:
                self.parts.append(gpu_part(shader, positions, uv, triangles, pixels))
        self.mesh = content.mesh if content.mesh is not None else content_mesh(content.parts)
        # The mesh kept here too, beside the GPU's
        self.bytes += self.mesh[0].nbytes + self.mesh[1].nbytes
        self.boxes = None
        self.last_hit = None

    def triangle_boxes(self):
        """The triangles' boxes along the tile's up and across it, for ground() and nearest() (made the first time
        either is asked): the frame (rows: two unit vectors across the up, and the up), each triangle's least and
        most along them, and the whole tile's. Along the tile's up, not the planet's axes: the ground's triangles are
        mostly flat across it, so their boxes are thin, and few are within a distance that are not nearly as near
        themselves."""
        if self.boxes is None:
            up = self.center / np.linalg.norm(self.center)
            e1 = np.cross(up, (0.0, 0.0, 1.0) if abs(up[2]) < 0.9 else (1.0, 0.0, 0.0))
            e1 /= np.linalg.norm(e1)
            frame = np.stack([e1, np.cross(up, e1), up])
            V, T = self.mesh
            corners = (V @ frame.T)[T]
            low, high = corners.min(axis=1), corners.max(axis=1)
            whole = (low.min(axis=0), high.max(axis=0)) if len(T) else (np.zeros(3), np.zeros(3))
            self.boxes = (frame, low, high, whole)
        return self.boxes

    def nearest(self, point, limit):
        """The distance from point (meters from the tile's center) to the nearest of the tile's triangles, and that
        nearest point (from the tile's center), if it's under limit, else (inf, None): only the triangles whose boxes
        are within limit are measured"""
        V, T = self.mesh
        if not len(T):
            return math.inf, None
        frame, low, high, _whole = self.triangle_boxes()
        p = frame @ point
        outside = np.maximum(np.maximum(low - p, p - high), 0.0)
        near = np.einsum("ij,ij->i", outside, outside) <= limit * limit
        if not near.any():
            return math.inf, None
        tri = T[near]
        a, b, c = (V[tri[:, k]].astype(np.float64) for k in range(3))
        p = np.asarray(point, dtype=np.float64)
        distances = triangle_distances(p, a, b, c)
        best = int(np.argmin(distances))
        if distances[best] >= limit:
            return math.inf, None
        return float(distances[best]), closest_on_triangle(p, a[best], b[best], c[best])

    def ground(self, origin, direction):
        """Where the line origin + t direction (meters, from the tile's center; direction a unit vector) meets the
        tile's triangles: every t, nearest the triangles' boxes first narrowing them down (Moller-Trumbore). The
        triangle last hit is tried first, alone: the drone's usually still over it the next tick."""
        V, T = self.mesh
        if not len(T):
            return np.zeros(0)
        if self.last_hit is not None:
            t = triangle_hit(V, T[self.last_hit], origin, direction)
            if t is not None:
                return np.array([t])
        frame, low, high, (whole_low, whole_high) = self.triangle_boxes()
        depth = float(whole_high[2] - whole_low[2])
        o2, d2 = frame @ origin, frame @ direction
        # Along the line, within the tile's depth of the origin's height, its place across up moves this much
        t_mid = (0.0 - o2[2]) / d2[2] if abs(d2[2]) > 1e-9 else 0.0
        at = o2[:2] + t_mid * d2[:2]
        reach = abs(depth / d2[2]) * float(np.hypot(*d2[:2])) if abs(d2[2]) > 1e-9 else np.inf
        pad = reach + 1e-3
        # Not over the tile at all (a neighbor's bounding sphere often reaches the line)
        if (at[0] + pad < whole_low[0] or at[0] - pad > whole_high[0]
                or at[1] + pad < whole_low[1] or at[1] - pad > whole_high[1]):
            return np.zeros(0)
        near = ((low[:, 0] <= at[0] + pad) & (high[:, 0] >= at[0] - pad)
                & (low[:, 1] <= at[1] + pad) & (high[:, 1] >= at[1] - pad))
        if not near.any():
            return np.zeros(0)
        tri = T[near]
        v0 = V[tri[:, 0]].astype(np.float64)
        e1 = V[tri[:, 1]].astype(np.float64) - v0
        e2 = V[tri[:, 2]].astype(np.float64) - v0
        p = np.cross(direction, e2)
        det = np.einsum("ij,ij->i", e1, p)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = origin - v0
        u = np.einsum("ij,ij->i", s, p) * inv
        q = np.cross(s, e1)
        v = (q @ direction) * inv
        t = np.einsum("ij,ij->i", e2, q) * inv
        hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9)
        if hit.any():
            # The nearest ahead (else behind), as Tileset.ground() takes it
            hits = np.nonzero(hit)[0]
            ahead = hits[t[hits] >= 0.0]
            best = ahead[np.argmin(t[ahead])] if len(ahead) else hits[np.argmax(t[hits])]
            self.last_hit = int(np.nonzero(near)[0][best])
        return t[hit]


_vertex_format = None


def vertex_format():
    """The tiles' vertex format (the shaders' pos and uv), made once: gpu_extras' batch_for_shader() works one out
    from the shader for every batch"""
    global _vertex_format
    if _vertex_format is None:
        import gpu
        _vertex_format = gpu.types.GPUVertFormat()
        _vertex_format.attr_add(id="pos", comp_type='F32', len=3, fetch_mode='FLOAT')
        _vertex_format.attr_add(id="uv", comp_type='F32', len=2, fetch_mode='FLOAT')
    return _vertex_format


def gpu_part(shader, positions, uv, triangles, pixels):
    """A part's batch and texture (sRGB: sampled linear)"""
    import gpu
    vertices = gpu.types.GPUVertBuf(vertex_format(), len(positions))
    vertices.attr_fill("pos", positions)
    vertices.attr_fill("uv", uv)
    batch = gpu.types.GPUBatch(type='TRIS', buf=vertices, elem=gpu.types.GPUIndexBuf(type='TRIS', seq=triangles))
    texture = None
    if pixels is not None:
        image, (width, height) = pixels
        if image.dtype == np.uint8:
            # From the loader process (ProcessLoader); Blender's textures are only made from floats
            image = np.multiply(image, np.float32(1.0 / 255.0), dtype=np.float32)
        buffer = gpu.types.Buffer('FLOAT', image.size, image.ravel())
        texture = gpu.types.GPUTexture((width, height), format='SRGB8_A8', data=buffer)
    return batch, texture


def segment_distances(p, a, b):
    """From point p to the segments a-b (arrays of them)"""
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def triangle_distances(p, a, b, c):
    """From point p to the triangles a, b, c (arrays of their corners): to the plane where its foot is inside the
    triangle, else to the nearest edge (degenerate triangles by their edges)"""
    ab, ac = b - a, c - a
    n = np.cross(ab, ac)
    nn = np.einsum("ij,ij->i", n, n)
    ap = p - a
    t = np.einsum("ij,ij->i", ap, n) / np.maximum(nn, 1e-300)
    foot = ap - t[:, None] * n
    d00, d01, d11 = (np.einsum("ij,ij->i", x, y) for x, y in ((ab, ab), (ab, ac), (ac, ac)))
    d20, d21 = np.einsum("ij,ij->i", foot, ab), np.einsum("ij,ij->i", foot, ac)
    denominator = d00 * d11 - d01 * d01
    ok = (nn > 1e-24) & (np.abs(denominator) > 1e-300)
    safe = np.where(ok, denominator, 1.0)
    v = (d11 * d20 - d01 * d21) / safe
    w = (d00 * d21 - d01 * d20) / safe
    inside = ok & (v >= 0.0) & (w >= 0.0) & (v + w <= 1.0)
    plane = np.abs(t) * np.sqrt(nn)
    edges = np.minimum(np.minimum(segment_distances(p, a, b), segment_distances(p, b, c)), segment_distances(p, c, a))
    return np.where(inside, plane, edges)


def closest_on_triangle(p, a, b, c):
    """The point of the triangle a, b, c nearest p"""
    ab, ac = b - a, c - a
    n = np.cross(ab, ac)
    nn = float(n @ n)
    if nn > 1e-24:
        foot = (p - a) - ((p - a) @ n / nn) * n
        d00, d01, d11 = ab @ ab, ab @ ac, ac @ ac
        d20, d21 = foot @ ab, foot @ ac
        denominator = d00 * d11 - d01 * d01
        if abs(denominator) > 1e-300:
            v = (d11 * d20 - d01 * d21) / denominator
            w = (d00 * d21 - d01 * d20) / denominator
            if v >= 0.0 and w >= 0.0 and v + w <= 1.0:
                return a + foot
    best, best_d = None, math.inf
    for x, y in ((a, b), (b, c), (c, a)):
        xy = y - x
        t = min(1.0, max(0.0, float((p - x) @ xy) / max(float(xy @ xy), 1e-300)))
        q = x + t * xy
        d = float(np.linalg.norm(p - q))
        if d < best_d:
            best, best_d = q, d
    return best


def triangle_hit(V, triangle, origin, direction):
    """Where the line origin + t direction meets one triangle of vertices V (Moller-Trumbore, in plain floats: for one
    triangle, numpy's calls cost more than the arithmetic), or None"""
    a, b, c = (V[i] for i in triangle)
    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
    e1x, e1y, e1z = float(b[0]) - ax, float(b[1]) - ay, float(b[2]) - az
    e2x, e2y, e2z = float(c[0]) - ax, float(c[1]) - ay, float(c[2]) - az
    dx, dy, dz = float(direction[0]), float(direction[1]), float(direction[2])
    px, py, pz = dy * e2z - dz * e2y, dz * e2x - dx * e2z, dx * e2y - dy * e2x
    det = e1x * px + e1y * py + e1z * pz
    if abs(det) <= 1e-12:
        return None
    inv = 1.0 / det
    sx, sy, sz = float(origin[0]) - ax, float(origin[1]) - ay, float(origin[2]) - az
    u = (sx * px + sy * py + sz * pz) * inv
    if u < -1e-9 or u > 1.0 + 1e-9:
        return None
    qx, qy, qz = sy * e1z - sz * e1y, sz * e1x - sx * e1z, sx * e1y - sy * e1x
    v = (dx * qx + dy * qy + dz * qz) * inv
    if v < -1e-9 or u + v > 1.0 + 1e-9:
        return None
    return (e2x * qx + e2y * qy + e2z * qz) * inv


def line_meets_boxes(point, direction, centers, axes, halves):
    """Whether the line through point along direction (either way) meets each box (center, unit axes as rows, half
    lengths): where its stretches between each pair of the box's faces overlap"""
    start = np.einsum("ijk,ik->ij", axes, point - centers)
    along = axes @ direction
    with np.errstate(divide="ignore", invalid="ignore"):
        a, b = (-halves - start) / along, (halves - start) / along
    return np.minimum(a, b).max(axis=1) <= np.maximum(a, b).min(axis=1)


def to_background():
    """The calling thread given macOS's utility quality of service (QOS_CLASS_UTILITY), which the system runs on the
    efficiency cores while others want the performance ones: the loader process's threads (serve()), a core's work
    or more while a city streams in. Whatever else keeps the processor busy slows Blender's frame (two other
    processes each busy on a core took the scene's evaluation from 2.3 to 2.8 ms a frame, and the main thread's
    whole frame from 10.5 to 11.9), on the efficiency cores about half as much."""
    if sys.platform == "darwin":
        try:
            ctypes.CDLL(None).pthread_set_qos_class_self_np(0x11, 0)
        except (OSError, AttributeError):
            pass


class Loader:
    """Threads: fetchers (workers), each taking a job (a node) and fetching its content, mostly waiting on the network
    with Python's lock let go; and decoders, decoding what was fetched (a tileset's JSON first, then models' Content)
    and leaving it for the draw (results). Decoding holds Python's lock most of the time (JSON, the numpy around it):
    with each of 32 fetchers decoding its own, they wanted it all at once, and Blender's own Python (the draw, the
    flight's tick) waited for it, up to 110 ms. A tile takes a millisecond or two, a tileset a few: DECODERS keep up.
    Fetchers start RAMP_STEP apart after the first RAMP_FIRST, so connections open a few at a time: 32 opened at once
    took 1 to 5 s for their first requests, the root's tilesets among them, and a view's first seconds stalled."""

    def __init__(self, fetcher, workers=WORKERS, decoders=DECODERS, as_bytes=False, background=False):
        self.fetcher = fetcher
        self.as_bytes = as_bytes
        # In a process of its own (serve()): its threads out of Blender's way (to_background())
        self.background = background
        self.dead = False
        # (0 for a tileset's root's tile, which goes before the jobs waiting; the order they came in; the job)
        self.jobs = queue.PriorityQueue()
        self.submitted = 0
        self.fetched = queue.PriorityQueue()
        self.results = queue.Queue()
        self.stopped = False
        self.order = 0
        self.lock = threading.Lock()
        self.threads = ([threading.Thread(target=self.fetch, args=(i,), name=f"tiles3d-fetch-{i}", daemon=True)
                         for i in range(workers)]
                        + [threading.Thread(target=self.decode, name=f"tiles3d-decode-{i}", daemon=True)
                           for i in range(decoders)])
        self.workers, self.decoders = workers, decoders
        for thread in self.threads:
            thread.start()

    def fetch(self, i):
        if self.background:
            to_background()
        if i >= RAMP_FIRST:
            time.sleep((i - RAMP_FIRST + 1) * RAMP_STEP)
        while not self.stopped:
            job = self.jobs.get()[2]
            if job is None or self.stopped:
                break
            try:
                data = self.fetcher.get(job[1])
            except Exception as error:      # anything a request can do, kept for the draw to report
                # A tileset's root's tile that didn't come: the tileset without it
                self.results.put((job[0], job[6] if len(job) > 6 else error))
                continue
            with self.lock:
                self.order += 1
                order = self.order
            # Tilesets first: what's under them isn't known, nor asked for, till they're in
            self.fetched.put((0 if job[2] == JSON else 1, order, job, data))
        self.fetcher.close()

    def decode(self):
        if self.background:
            to_background()
        while not self.stopped:
            item = self.fetched.get()
            if item[2] is None or self.stopped:
                break
            _first, _order, job, data = item
            node, url, kind, transform, add, error = job[:6]
            # The tileset whose root's tile this is (below)
            records = job[6] if len(job) > 6 else None
            try:
                if kind == JSON:
                    result = flatten(json.loads(data), url, transform, add, error)
                    if ROOT_TILES and len(result.urls) and result.kinds[0] == GLB:
                        # Its root's tile fetched now, and the two given back as one (take_results()): the view that
                        # needs a tileset needs its root's tile, to draw or to stand in for what's under it, and asked
                        # for after the tileset's in, it waited for the next selection and its turn among the jobs,
                        # which half the time never came: a tileset fetched for nothing. Google's finest tiles are each
                        # a tileset of their own.
                        root_transform = result.transforms[0] if result.transforms is not None else None
                        with self.lock:
                            self.submitted += 1
                            self.jobs.put((0, self.submitted,
                                           (node, result.urls[0], GLB, root_transform, False, 0.0, result)))
                        continue
                else:
                    result = decode_content(data, transform, self.as_bytes)
                    if records is not None:
                        result = (records, result)
            except Exception as error:      # anything a tile's bytes can do, kept for the draw to report
                result = error if records is None else records
            self.results.put((node, result))

    @property
    def requests(self):
        return self.fetcher.requests

    @property
    def received(self):
        return self.fetcher.received

    def submit(self, node, url, kind, transform, add, error):
        with self.lock:
            self.submitted += 1
            self.jobs.put((1, self.submitted, (node, url, kind, transform, add, error)))

    def stop(self):
        self.stopped = True
        for i in range(self.workers):
            self.jobs.put((-1, i, None))
        for _i in range(self.decoders):
            self.fetched.put((-1, 0, None, None))


def records_of(values):
    """Records from their values, in their slots' order (as the loader process sends them)"""
    records = Records()
    for name, value in zip(Records.__slots__, values):
        setattr(records, name, value)
    return records


class RemoteError(Exception):
    """An error in the loader process (ProcessLoader), as it was there: its message, and an HTTP status if it had one"""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


_remote_errors = {}


def remote_error(name, message, status):
    """A RemoteError of a class named like the loader process's, so it reads the same in the view's corner"""
    cls = _remote_errors.get(name)
    if cls is None:
        cls = _remote_errors[name] = type(name, (RemoteError,), {})
    return cls(message, status)


class ProcessLoader:
    """The Loader in a background Blender of its own (serve()), its jobs sent there and its results back, over a
    socket. The Loader's threads hold Python's lock whenever they run Python (HTTP, JSON, glTF, the numpy around them),
    and Blender's own Python waited for it: two threads busy in Python took every frame of a flight from 16.7 to 27
    to 32 ms, where eight downloading without it cost next to nothing. So in Blender itself only the results are
    unpickled (their arrays as they are) and textures made from bytes (a quarter of float32's to send). Same interface
    as Loader (submit(), results, stop(), requests, received); dead once the process has ended (Tileset then loads
    in Blender again)."""

    def __init__(self, use_key):
        import multiprocessing.connection
        import subprocess
        import bpy
        here, there = socket.socketpair()
        self.connection = multiprocessing.connection.Connection(here.detach())
        child = there.detach()
        script = ("import importlib.util, sys; "
                  f"spec = importlib.util.spec_from_file_location('tiles3d_loader', {os.path.abspath(__file__)!r}); "
                  "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
                  "module.serve(int(sys.argv[-1]))")
        self.process = subprocess.Popen(
            [bpy.app.binary_path, "-b", "--factory-startup", "--python-exit-code", "1", "--python-expr", script, "--",
             str(child)], pass_fds=(child,), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        os.close(child)
        self.results = queue.Queue()
        self.outgoing = queue.Queue()
        self.dead = False
        self.stopped = False
        self.requests = 0
        self.received = 0
        self.outgoing.put(("start", use_key))
        threading.Thread(target=self.send_jobs, name="tiles3d-send", daemon=True).start()
        threading.Thread(target=self.receive, name="tiles3d-receive", daemon=True).start()

    def submit(self, node, url, kind, transform, add, error):
        self.outgoing.put(("job", node, url, kind, transform, add, error))

    def send_jobs(self):
        """On a thread of its own: sending can wait for the process to read"""
        while True:
            message = self.outgoing.get()
            try:
                self.connection.send(message)
            except (OSError, ValueError):
                self.dead = True
                return
            if message[0] == "stop":
                return

    def receive(self):
        while True:
            try:
                data = self.connection.recv_bytes()
            except (EOFError, OSError):
                self.dead = not self.stopped
                return
            # Unpickled, and its pixels made floats, between the main thread's turns at Python (give_way())
            give_way()
            message = pickle.loads(data)
            del data
            give_way()
            _kind, node, (what, values), self.requests, self.received = message
            if what == "error":
                result = remote_error(*values)
            elif what == "records":
                result = records_of(values)
            elif what == "both":
                result = (records_of(values[0]), prepare(Content(*values[1])))
            else:
                result = prepare(Content(*values))
            self.results.put((node, result))

    def stop(self):
        self.stopped = True
        self.outgoing.put(("stop",))
        process = self.process

        def reap():
            try:
                process.wait(5.0)
            except Exception:                   # still running: the socket's closed, so it will end; made to
                process.kill()
                process.wait()
        threading.Thread(target=reap, name="tiles3d-reap", daemon=True).start()


def serve(fd):
    """The loader process (ProcessLoader): a Loader's results for the jobs that come over the socket fd, sent back,
    till it's told to stop or the socket closes (Blender, the parent, ended)"""
    import multiprocessing.connection
    connection = multiprocessing.connection.Connection(fd)
    _start, use_key = connection.recv()
    fetcher = Fetcher(google_key() if use_key else None)
    loader = Loader(fetcher, as_bytes=True, background=True)

    def receive():
        to_background()
        while True:
            try:
                message = connection.recv()
            except (EOFError, OSError):
                message = ("stop",)
            if message[0] == "stop":
                loader.stop()
                loader.results.put(None)
                return
            loader.submit(*message[1:])

    threading.Thread(target=receive, name="tiles3d-serve", daemon=True).start()
    to_background()
    while True:
        item = loader.results.get()
        if item is None:
            break
        node, result = item
        # As plain values (pickle would need this module by the same name in Blender)
        if isinstance(result, Exception):
            result = ("error", (type(result).__name__, str(result), getattr(result, "status", None)))
        elif isinstance(result, Records):
            result = ("records", tuple(getattr(result, name) for name in Records.__slots__))
        elif isinstance(result, tuple):
            records, content = result
            result = ("both", (tuple(getattr(records, name) for name in Records.__slots__),
                               (content.center, content.parts, content.copyright, content.radius)))
        else:
            result = ("content", (result.center, result.parts, result.copyright, result.radius))
        try:
            connection.send(("result", node, result, fetcher.requests, fetcher.received))
        except (OSError, ValueError):
            break
    sys.stdout.flush()
    os._exit(0)


TILE_VERTEX = """
void main()
{
    // Map units from the camera: the tile's vertex scaled, plus its center's offset in two parts (uTile's two rows);
    // the second's w is 1, but the shader can't know it, so Metal's fast math can't sum the two parts first (which
    // would lose the low one)
    vec3 p = (pos * uTile[0].w + uTile[0].xyz) * uTile[1].w + uTile[1].xyz;
    worldPoint = mat3(uWorldFromMap) * p;
    groundHeight = heightAt(worldPoint);
    texCoord = uv;
#if CLIP
    tilePoint = pos;
#endif
    gl_Position = uViewProjection * vec4(worldPoint, 1.0);
}
"""

TILE_FRAGMENT = """
void main()
{
#if CLIP
    // A tile standing in for children not in yet: drawn only within their footprints (Tileset.hole_values()), so it
    // doesn't cover the finer tiles drawn beside them
    bool missing = false;
    for (int i = 0; i < MAX_HOLES; i++) {
        vec4 a = holeValues.rows[3 * i];
        vec4 b = holeValues.rows[3 * i + 1];
        vec4 c = holeValues.rows[3 * i + 2];
        vec3 q = tilePoint - a.xyz;
        missing = missing || (abs(dot(q, b.xyz)) <= a.w && abs(dot(q, c.xyz)) <= b.w);
    }
    if (!missing) {
        discard;
    }
#endif
    fragColor = vec4(foggedAt(texture(tileColor, texCoord).rgb, worldPoint, groundHeight), 1.0);
}
"""

# The fog, the tiles' and the sphere's (sphere.py): a point's color seen through the air between it and the camera, v
# (world units from the camera). Plain fog (uAir.w 0) is as thick everywhere, uFog.w its distance, EXTINCTION times as
# far for each channel. A planet's air (fog_rows(), uAir.w 1) is two layers, each thinning with height as e^(-h / H):
# haze (uHaze) and the clear air itself (uClearAir), each its extinction at the ground per world unit (rgb) and its H;
# uAir (the camera's height h0, its distance c from the planet's center, the far end's height error per unit of
# distance, 1), uUp the way up from the camera. Along a straight line, a layer's thickness is its average density
# between the ends' heights, (e^-a - e^-b) / (b - a) at a and b scale heights up, exact while the height changes evenly
# along it (from space down to the ground, nearly); a line skimming the planet dips lower in between, and is thicker
# than that. The end's height from the camera's, without cancellation however far the center is: |c up + v| - c =
# (2 c v.up + |v|^2) / (|c up + v| + c), all over c, so c^2 never overflows.
# The far end's height is only as true as the tiles drawn there, and with haze thinning every kilometer or so, a
# wrong height is a patch of wrong fog: Google's coarse tiles, drawn far away, are rough shapes whose vertices are up
# to 200 km under sea level and 40 km over it, and flat triangles hundreds of kilometers across whose middles sag tens
# of kilometers under their corners (tile-shaped patches over the globe, the user's screenshots, 2026-11-16). So a
# tile's height is its vertices' (heightAt() in the vertex shader, interpolated, not the flat triangle's), trusted as
# far as its detail goes: tiles are picked for their error on screen, so their geometric error grows with distance,
# about uAir.z (the tiles' error in pixels over the view's pixels per unit of tangent) times it. In each layer the
# height counts by 1 / (1 + (error / H)^2): fully nearby, among mountains, and not at all from space, where the ground
# is taken as sea level, smoothly. Heights below sea level count as sea level.
FOG_SOURCE = """
float heightAt(vec3 v)
{
    if (uAir.w < 0.5) {
        return 0.0;
    }
    float c = uAir.y;
    float q = min(length(v) / c, 1.0e15);
    float rise = 2.0 * dot(v, uUp.xyz) / c + q * q;
    return uAir.x + c * rise / (sqrt(max(0.0, 1.0 + rise)) + 1.0);
}

// A layer of scale height H: its average density between the camera and a point h1 high, error the point's height
// error
float layerDensity(float h1, float H, float error)
{
    float trust = error / H;
    float a = max(uAir.x, 0.0) / H;
    float b = max(h1, 0.0) / H / (1.0 + trust * trust);
    return abs(b - a) > 1.0e-3 ? (exp(-a) - exp(-b)) / (b - a) : exp(-0.5 * (a + b));
}

vec3 foggedAt(vec3 color, vec3 v, float h1)
{
    float d = length(v);
    vec3 depth;
    if (uAir.w > 0.5) {
        float error = uAir.z * d;
        depth = d * (uHaze.rgb * layerDensity(h1, uHaze.w, error)
                     + uClearAir.rgb * layerDensity(h1, uClearAir.w, error));
    } else {
        depth = d / (uFog.w * EXTINCTION);
    }
    vec3 transmission = exp(-depth);
    return color * transmission + uFog.rgb * (1.0 - transmission);
}

vec3 fogged(vec3 color, vec3 v)
{
    return foggedAt(color, v, heightAt(v));
}
"""

# How much farther each channel's light gets through plain fog than its distance (red, green, blue)
EXTINCTION = (0.349, 0.256, 0.179)

TILE_MATRICES = ("uViewProjection", "uWorldFromMap")
TILE_ROWS = ("uFog", "uAir", "uUp", "uHaze", "uClearAir")


def make_shader(clip=False):
    """The tiles' shader: each vertex placed from its tile's center offset (push constants), colored by its texture
    (sRGB, so sampled linear), in the sphere's fog; with clip, only within the footprints of a uniform buffer's holes
    (fallbacks: Selection)"""
    import gpu
    info = gpu.types.GPUShaderCreateInfo()
    rows = (" ".join(f"mat4 m{i};" for i in range(len(TILE_MATRICES))) + " "
            + " ".join(f"vec4 r{i};" for i in range(len(TILE_ROWS))))
    info.typedef_source(f"struct TileValues {{ {rows} }};\n"
                        f"struct HoleValues {{ vec4 rows[{3 * MAX_HOLES}]; }};")
    info.uniform_buf(0, "TileValues", "tileValues")
    if clip:
        info.uniform_buf(1, "HoleValues", "holeValues")
    # Struct members numbered, names as defines: names like fog clash with Blender's own Metal macros
    for i, name in enumerate(TILE_MATRICES):
        info.define(name, f"tileValues.m{i}")
    for i, name in enumerate(TILE_ROWS):
        info.define(name, f"tileValues.r{i}")
    info.define("EXTINCTION", "vec3(%r, %r, %r)" % EXTINCTION)
    info.define("CLIP", "1" if clip else "0")
    info.define("MAX_HOLES", str(MAX_HOLES))
    info.push_constant('VEC4', "uTile", 2)
    info.sampler(0, 'FLOAT_2D', "tileColor")
    info.vertex_in(0, 'VEC3', "pos")
    info.vertex_in(1, 'VEC2', "uv")
    interface = gpu.types.GPUStageInterfaceInfo("tiles3d_clip_interface" if clip else "tiles3d_interface")
    interface.smooth('VEC3', "worldPoint")
    interface.smooth('FLOAT', "groundHeight")
    interface.smooth('VEC2', "texCoord")
    if clip:
        interface.smooth('VEC3', "tilePoint")
    info.vertex_out(interface)
    info.fragment_out(0, 'VEC4', "fragColor")
    info.vertex_source(FOG_SOURCE + TILE_VERTEX)
    info.fragment_source(FOG_SOURCE + TILE_FRAGMENT)
    return gpu.shader.create_from_info(info)


class Selection:
    """What a view draws of a tileset: the tiles (node indexes, each with its Tile), the fallbacks (tiles standing in
    for children not in yet, drawn only within those children's footprints: (index, the holes' nodes)), both in one
    (shown), whether they cover everything in view (complete), and where the view was (meters), for
    Tileset.selection()'s reuse. And the
    ground selected, in view or not (select()'s around): its tiles, its fallbacks' tiles, and its fallbacks with their
    holes."""

    def __init__(self, draw, fallbacks, ground, ground_fallbacks, complete, camera, forward, key):
        self.draw, self.fallbacks, self.complete = draw, fallbacks, complete
        self.fallback_tiles = [index for index, _holes in fallbacks]
        self.shown = draw + self.fallback_tiles
        self.ground = ground
        self.ground_fallbacks = [index for index, _holes in ground_fallbacks]
        self.ground_holes = ground_fallbacks
        self.camera, self.forward, self.key = camera, forward, key


class Tileset:
    """A 3D Tiles tileset streamed onto the map's sphere: its tree, the tiles loaded, the loader, and what went wrong
    last.

    The tree is kept in arrays, a row a node (NODE_ARRAYS): a tile, or a link to a tileset of its own. A tileset's
    nodes are a block of rows in a row, breadth first, so every node's children are in a row too (first_child,
    child_count): added when its JSON comes in (add_records()), and freed, with the tilesets linked from it, once the
    tree is past NODES and no selection has walked it for the longest (prune()). Freed rows are used again, so the
    arrays stop growing."""

    def __init__(self, source):
        self.source = source
        self.google = source == GOOGLE
        self.root_url = GOOGLE_ROOT if self.google else source
        key = google_key() if self.google else None
        self.fetcher = Fetcher(key)
        self.shader = None
        self.clip_shader = None
        self.hole_buffers = {}
        self.white = None
        # Tests in background mode: tiles without GPU parts
        self.gpu = True
        self.error_message = None if key or not self.google else (
            f"no Google Maps key: put it in {KEY_FILE} (or {KEY_ENV})")
        self.loader = self.make_loader(key is not None)
        # Held by a selection while it walks the tree, and by prune() while it frees some of it
        self.walk = threading.Lock()
        self.reset()
        # The selector thread (selection()), and the views it's given to select for
        self.views = {}
        self.asked = {}
        self.views_changed = threading.Condition()
        self.closed = False
        self.selector = threading.Thread(target=self.select_views, name="tiles3d-select", daemon=True)
        self.selector.start()

    def make_loader(self, use_key):
        """The loader: in a process of its own (ProcessLoader) if PROCESS and this is Blender, else here"""
        if PROCESS:
            try:
                return ProcessLoader(use_key)
            except Exception as error:      # no process, no Blender binary: loading here instead
                self.error_message = f"loader process: {type(error).__name__}: {error}"
        return Loader(self.fetcher)

    def reset(self):
        """The tree from nothing: the root's tileset requested again (with Google, a new session)"""
        self.generation = getattr(self, "generation", -1) + 1
        self.size = 0                   # the rows used so far, freed ones among them
        self.live = 0                   # the tree's nodes
        for name, (dtype, shape, fill) in NODE_ARRAYS.items():
            setattr(self, name, np.full((0, *shape), fill, dtype))
        self.urls = []                  # per node: its content's URL, or None
        self.transforms = {}            # node -> its cumulative transform, for the few that have one
        self.retries = {}               # node -> (failures, when it may be asked for again), for the failed
        self.free = []                  # freed rows, [first, count] in order, for the next tilesets (allocate())
        # Blocks: a tileset's rows. id -> (first, count, the node that links to it), the blocks linked from its
        # nodes, and the last selection that walked it (prune()).
        self.blocks = {}
        self.block_children = {}
        self.block_seen = np.zeros(64, np.int64)
        self.block_ids = []             # freed blocks' ids
        self.tiles = {}                 # node -> Tile
        self.last_draw = []             # the drone's view's last selection's ground tiles (sphere.ground_height())
        self.last_fallbacks = []        # and its ground fallbacks' tiles (Selection)
        self.last_holes = []            # with their holes, where the ground's looked for in a hole (hole_ground())
        self.holes_kept = None          # their holes' footprints (hole_ground())
        self.last_shown = []            # the last view's tiles and fallbacks drawn, for the attribution
        self.ground_kept = {}           # their tiles and boxes, for ground() (ground_boxes())
        self.around_kept = None         # the loaded tiles near the drone, for loaded_ground() (loaded_around())
        self.batches = {}               # the drawn lists' Tiles and centers (batch())
        self.attributed = None          # the list last drawn's attribution (attribution())
        self.drops = 0                  # counts the tiles dropped: what's kept of a list is made again
        self.ground_tile = None         # the tile ground() last found the ground in
        self.bytes = 0
        self.in_flight = 0
        self.selections = 0
        self.selection_cache = {}
        self.allowance, self.allowed_at = RATE_BURST, time.time()
        self.stale = 0                  # tiles that came in no longer wanted (take_results())
        self.motions = {}               # each view's motion as of its last draw (motion())
        self.eye = None                 # the last selection's camera (trim())
        self.version = 0                # counts the tree's and the tiles' changes, for selection()
        self.failures_in_a_row = 0
        self.session_started = time.time()
        self.loaded = 0
        self.pruned = 0
        # The root: a link to the root's tileset, a block of its own
        root = self.allocate(1)
        self.radii[root] = self.halves[root] = self.errors[root] = 1.0e30
        self.axes[root] = np.eye(3)
        self.kinds[root] = JSON
        self.urls[root] = self.root_url
        self.new_block(root, 1, -1)
        self.live = 1

    # ----- the tree -----

    def make_room(self, count):
        """The per-node arrays grown (doubled) until they have room for count rows. The selection's own arrays
        (SCRATCH) start over empty: a selection walking the tree meanwhile keeps the ones it began with."""
        size = len(self.radii)
        if count <= size:
            return
        grow = max(count, 2 * size) - size
        for name, (dtype, shape, fill) in NODE_ARRAYS.items():
            if name in SCRATCH:
                setattr(self, name, np.full((size + grow, *shape), fill, dtype))
            else:
                setattr(self, name, np.concatenate([getattr(self, name), np.full((grow, *shape), fill, dtype)]))
        self.urls.extend([None] * grow)

    def allocate(self, count):
        """count rows in a row for a tileset's nodes: the first freed run they fit in (the rest of it stays free),
        else new ones"""
        for k, (first, room) in enumerate(self.free):
            if room >= count:
                if room == count:
                    del self.free[k]
                else:
                    self.free[k] = [first + count, room - count]
                return first
        first = self.size
        self.make_room(first + count)
        self.size = first + count
        return first

    def release(self, first, count):
        """Rows freed, joined with the freed runs beside them"""
        runs = self.free
        k = bisect.bisect_left(runs, [first, 0])
        runs.insert(k, [first, count])
        if k + 1 < len(runs) and first + count == runs[k + 1][0]:
            runs[k][1] += runs.pop(k + 1)[1]
        if k and runs[k - 1][0] + runs[k - 1][1] == first:
            runs[k - 1][1] += runs.pop(k)[1]

    def new_block(self, first, count, link):
        """A block for a tileset's rows, linked from node link (-1: the root's own)"""
        block = self.block_ids.pop() if self.block_ids else len(self.blocks) + len(self.block_ids)
        if block >= len(self.block_seen):
            self.block_seen = np.concatenate([self.block_seen, np.zeros(len(self.block_seen), np.int64)])
        self.blocks[block] = (first, count, link)
        self.block_children[block] = set()
        self.block_seen[block] = self.selections
        self.block_of[first:first + count] = block
        if link >= 0:
            self.block_children[int(self.block_of[link])].add(block)
        return block

    def add_records(self, link, records, root=None):
        """A tileset's tiles (Records, flatten()'s) under the node that linked to it (its root becomes the node's
        child): a block of rows. root: its root's content, if that was loaded with it."""
        count = len(records.urls)
        first = self.allocate(count)
        stop = first + count
        self.centers[first:stop] = records.centers
        self.radii[first:stop] = records.radii
        self.axes[first:stop] = records.axes
        self.halves[first:stop] = records.halves
        self.errors[first:stop] = records.errors
        self.refined[first:stop] = False
        self.kinds[first:stop] = records.kinds
        self.adds[first:stop] = records.adds
        # A node with no content has nothing to load
        self.states[first:stop] = np.where(records.kinds == NONE, READY, NEW)
        self.has_tile[first:stop] = False
        self.drawn[first:stop] = -1
        self.depths[first:stop] = records.depths + (self.depths[link] + 1)
        self.parents[first:stop] = records.parents + first
        self.parents[first] = link
        self.first_child[first:stop] = records.first_child + first
        self.child_count[first:stop] = records.child_count
        self.urls[first:stop] = records.urls
        if records.transforms is not None:
            self.transforms.update((first + i, t) for i, t in enumerate(records.transforms) if t is not None)
        self.new_block(first, count, link)
        self.live += count
        if root is not None:
            self.take_tile(first, root)
        # Linked last, its count after its first child: the selector thread walks the tree meanwhile (select()), and
        # only ever sees all of a tileset's nodes
        self.first_child[link] = first
        self.child_count[link] = 1
        self.version += 1

    def free_block(self, block):
        """A tileset's rows freed, with the tilesets linked from them and all their tiles: its link is as before its
        tileset came, and asks for it again when a view needs it"""
        for child in list(self.block_children[block]):
            self.free_block(child)
        del self.block_children[block]
        first, count, link = self.blocks.pop(block)
        stop = first + count
        self.child_count[link] = 0
        self.states[link] = NEW
        self.block_children[int(self.block_of[link])].discard(block)
        for index in (np.nonzero(self.has_tile[first:stop])[0] + first).tolist():
            self.drop(index)
        # What's still being loaded for them is thrown away when it comes (take_results())
        self.epochs[first:stop] += 1
        self.child_count[first:stop] = 0
        self.urls[first:stop] = [None] * count
        for kept in (self.transforms, self.retries):
            if kept:
                for index in [i for i in kept if first <= i < stop]:
                    del kept[index]
        self.block_ids.append(block)
        self.release(first, count)
        self.live -= count
        self.pruned += count
        self.version += 1
        # Its rows may be another tileset's soon: the loaded tiles near the drone are found again
        self.around_kept = None

    def prune(self):
        """Past NODES, the tilesets no selection has walked for the longest freed (free_block()), until the tree is a
        tenth under it: a flight streams in a few thousand nodes a second, most of them links to tilesets never
        opened, and kept, they were 2 million after five minutes, 2 GB with their URLs, and each selection's own arrays
        as long. Not the ones walked in the last PRUNE_KEEP selections: the views' latest selections, drawn until
        their next, are of those. Under the lock selections walk the tree under; if one's walking it, next time."""
        if self.live <= NODES or not self.walk.acquire(blocking=False):
            return
        try:
            recent = self.selections - PRUNE_KEEP
            ids = np.array([b for b, (_first, _count, link) in self.blocks.items() if link >= 0], dtype=np.int64)
            seen = self.block_seen[ids]
            for block in ids[np.argsort(seen, kind="stable")].tolist():
                if self.live <= 0.9 * NODES or self.block_seen[block] >= recent:
                    break
                # A tileset is walked no later than the ones linked from it, so those went with it already, or go now
                if block in self.blocks:
                    self.free_block(block)
            for name, selection in list(self.selection_cache.items()):
                if selection.number < recent:
                    del self.selection_cache[name]
        finally:
            self.walk.release()

    def coarseness(self, nodes, near, K, target):
        """How many levels coarser than the target the view would be at each of these nodes with it drawn there
        (whole numbers: 0 a tile within the target, 1 one with up to twice its error...): what's drawn there without
        it is coarser still. A tileset's link is as its root's tile, which comes with it."""
        over = self.errors[nodes] * K / (near * target)
        return np.ceil(np.log2(np.maximum(over, 1e-6))).clip(-1.0, 60.0)

    def toward(self, nodes, camera, forward):
        """How near the middle of the view each node is: the cosine of its center's angle off the view's direction"""
        C = self.centers[nodes] - camera
        return C @ forward / np.maximum(np.sqrt(np.einsum("ij,ij->i", C, C)), 1e-9)

    def children_of(self, nodes):
        """The children of nodes (an array of them), in order. Their counts are read before their first children, as
        add_records() writes them the other way round."""
        counts = self.child_count[nodes]
        total = int(counts.sum())
        if not total:
            return np.zeros(0, np.int64)
        starts = self.first_child[nodes].astype(np.int64)
        return np.arange(total) + np.repeat(starts - (np.cumsum(counts) - counts), counts)

    # ----- loading -----

    def take_results(self, budget=LOAD_BUDGET):
        """The loader's results, as many as budget seconds allow: tilesets added to the tree, models made into Tiles"""
        if self.loader.dead and self.loader.results.empty():
            # The loader process ended (it shouldn't): loading here from now on, what it was asked for again
            self.loader = Loader(self.fetcher)
            self.error_message = "the tile loader's process ended: loading in Blender instead"
            self.states[self.states == QUEUED] = NEW
            self.in_flight = 0
        start = time.perf_counter()
        while time.perf_counter() - start < budget:
            try:
                job, result = self.loader.results.get_nowait()
            except queue.Empty:
                break
            generation, index, epoch = job
            if generation != self.generation:
                continue                    # asked for before a reset
            self.in_flight -= 1
            if epoch != self.epochs[index]:
                continue                    # its tileset was freed meanwhile (prune())
            if isinstance(result, Exception):
                failures = self.retries.get(index, (0, 0.0))[0] + 1
                retry_at = time.time() + min(RETRY_MAX, RETRY * 2.0 ** (failures - 1))
                self.error_message = f"{type(result).__name__}: {result}"
                status = getattr(result, "status", None)
                if index == 0 and status in (400, 401, 403, 404):
                    # The root itself refused (no key, no API enabled): ask much less often, it's the billed request
                    retry_at = time.time() + RETRY_MAX
                if status is not None and 400 <= status < 500 and index != 0:
                    self.failures_in_a_row += 1
                self.retries[index] = (failures, retry_at)
                self.states[index] = FAILED
                continue
            self.failures_in_a_row = 0
            # Tiles load again: an earlier failure (Google sometimes refuses one, then serves it) is over
            self.error_message = None
            self.retries.pop(index, None)
            if self.kinds[index] == JSON:
                try:
                    # With its root's tile, if the loader fetched it too (Loader.decode())
                    self.add_records(index, *(result if isinstance(result, tuple) else (result,)))
                    self.states[index] = READY
                except (KeyError, ValueError, TypeError, IndexError, AttributeError) as error:
                    self.retries[index] = (1, time.time() + RETRY_MAX)
                    self.states[index] = FAILED
                    self.error_message = f"tileset {self.urls[index]}: {error}"
            elif self.drawn[index] < self.selections - STALE:
                # No selection has wanted it since it was asked for (the view has moved on): not made, which would
                # cost the draw its upload, and its memory until it's trimmed
                self.states[index] = NEW
                self.stale += 1
                continue
            else:
                self.take_tile(index, result)
            self.version += 1
        if (self.google and self.failures_in_a_row >= SESSION_FAILURES
                and time.time() - self.session_started > SESSION_MIN):
            # Google's session has expired: a new one, from the root
            self.drop_all()
            self.reset()

    def take_tile(self, index, content):
        """A node's content, loaded, made its Tile"""
        if self.gpu and self.shader is None:
            self.shader = make_shader()
        tile = Tile(content, self.shader if self.gpu else None)
        self.tiles[index] = tile
        # Needed as of now: with what it was last needed at, a tile that came in between a selection's walk and its
        # trim() was dropped as it came
        self.drawn[index] = self.selections
        self.bytes += tile.bytes
        self.loaded += 1
        self.has_tile[index] = True
        self.states[index] = READY

    def request(self, wanted, start=0):
        """Jobs for the wanted nodes (best first) not loaded or asked for, up to IN_FLIGHT at once, from the start-th
        of them on: returns how far down the list it got, to go on from when there's room again"""
        now = time.time()
        states = self.states
        # No more a second than the server allows (RATE): what the time since the last ones has made room for
        self.allowance = min(RATE_BURST, self.allowance + (now - self.allowed_at) * RATE)
        self.allowed_at = now
        for at in range(start, len(wanted)):
            if self.in_flight >= IN_FLIGHT or self.allowance < 1.0:
                return at
            index = wanted[at]
            state = states[index]
            if state == NEW or (state == FAILED and now >= self.retries.get(index, (0, 0.0))[1]):
                states[index] = QUEUED
                self.in_flight += 1
                # A tileset is two requests: its own and its root's tile's (ROOT_TILES)
                self.allowance -= 2.0 if ROOT_TILES and self.kinds[index] == JSON else 1.0
                self.loader.submit((self.generation, index, int(self.epochs[index])), self.urls[index],
                                   int(self.kinds[index]), self.transforms.get(index), bool(self.adds[index]),
                                   float(self.errors[index]))
        return len(wanted)

    def drop(self, index):
        tile = self.tiles.pop(index, None)
        if tile is not None:
            self.bytes -= tile.bytes
            self.drops += 1
            self.has_tile[index] = False
            self.states[index] = NEW
            self.version += 1

    def drop_all(self):
        for index in list(self.tiles):
            self.drop(index)

    def trim(self):
        """Past MEMORY, the tiles not needed in the last KEEP_SELECTIONS selections dropped until it's back under it:
        the ones a view from here would miss least first, whichever way it looked (the least error their parent would
        show in their place). Least recently needed first, a look around dropped what was behind the drone, the
        finest tiles there are, and asked for it again as it turned back: a sixth of a city flight's loads."""
        if self.bytes <= MEMORY:
            return
        loaded = np.nonzero(self.has_tile[:self.size])[0]
        old = loaded[self.drawn[loaded] < self.selections - KEEP_SELECTIONS]
        if not len(old):
            return
        if self.eye is not None:
            above = np.maximum(self.parents[old], 0)
            away = np.linalg.norm(self.centers[above] - self.eye, axis=1) - self.radii[above]
            missed = self.errors[above] / np.maximum(away, 1e-3)
            old = old[np.argsort(missed, kind="stable")]
        for index in old.tolist():
            if self.bytes <= MEMORY:
                break
            self.drop(index)

    # ----- what a view draws -----

    def select(self, camera, forward, planes, K, target, around=0.0, motion=None):
        """The tiles a view draws, and what it needs loaded, from the camera (meters), its view's planes (in meters
        from the camera: n.x + d >= 0 inside, n unit), K (pixels a radian, about: the screen-space error's scale) and
        the target error (pixels): (draw list, fallbacks (tiles standing in for holes: (index, the holes' nodes)),
        ground list, ground fallbacks, complete, wanted list best first). The ground is what's drawn and, in view or
        not, as finely as if in view, the drone's ground (sphere.ground_height(), surface_distance()) as it moves until
        the next selection: within AROUND times around (its distance to it, meters) of the camera, and straight down
        from anywhere within AROUND - 1 times around of it.

        The tree is walked a depth at a time, each depth's nodes at once in arrays (a view's walk visits a few
        thousand of them, 25 deep in Google's tree; a node at a time in Python, it took 3 ms, 7 to 9 with the flight
        wanting Python's lock too, and held it a tenth of the time): down, which nodes are in view and which of them
        refine into their children; up, how many holes each one's part of the view has (tiles not in yet); then what's
        drawn, what stands in for holes, and what's wanted."""
        with self.walk:
            return self.walked(camera, forward, planes, K, target, around, motion)

    def walked(self, camera, forward, planes, K, target, around, motion):
        """select(), with the tree's lock held"""
        # The view's motion: the camera's velocity (meters a second) and how fast the view turns (radians a second)
        velocity, turning = motion if motion is not None else (None, 0.0)
        moving = motion is not None and (float(velocity @ velocity) > 0.0 or turning > 0.0)
        speed2 = float(velocity @ velocity) if moving else 0.0
        self.eye = camera
        now = self.selections
        kinds, parents, child_count = self.kinds, self.parents, self.child_count
        # The selection's own arrays, each one's entries zero again when it's done
        hole_sums, shown_sums, marks = self.hole_sums, self.shown_sums, self.marks
        occluder = RADIUS + LOWEST
        d_camera = float(np.linalg.norm(camera))
        down = -camera / max(d_camera, 1e-9)
        camera_reach = math.acos(occluder / d_camera) if d_camera > occluder else None
        normals = planes[:, :3].T
        ball, margin = AROUND * around, (AROUND - 1.0) * around
        levels, touched, prefetched = [], [], []
        idx = np.zeros(1, np.int64)
        while len(idx):
            # On the selector thread, between the main thread's turns at Python
            give_way()
            touched.append(idx)
            C = self.centers[idx] - camera
            r = self.radii[idx]
            U, L = self.axes[idx], self.halves[idx]
            distance = np.sqrt(np.einsum("ij,ij->i", C, C))
            # The nearest it can be: outside both its bounding sphere and its box. The sphere alone (a flat tile's is
            # as wide as the tile) put the camera inside the sphere of every tile under it within its width, and so
            # refined them a level or two finer than their screen-space error asked for, near the camera most.
            outside = np.maximum(np.abs(np.einsum("ijk,ik->ij", U, C)) - L, 0.0)
            near = np.maximum(np.maximum(distance - r, np.sqrt(np.einsum("ij,ij->i", outside, outside))), 1e-3)
            # Within each side of the view by the sphere's radius, or the box's reach across the side if less
            spread = np.minimum(r[:, None], np.einsum("ikp,ik->ip", np.abs(U @ normals), L))
            seen = np.all(C @ normals + planes[:, 3] >= -spread, axis=1)
            if camera_reach is not None:
                # The horizon: a sphere is hidden if it's further round the planet than both its horizon and the
                # camera's
                Q = self.centers[idx]
                q = np.maximum(np.linalg.norm(Q, axis=1), 1.0)
                angle = np.arccos(np.clip(Q @ camera / (q * d_camera), -1.0, 1.0))
                spread = np.arcsin(np.minimum(1.0, r / q))
                reach = camera_reach + np.arccos(np.minimum(1.0, occluder / np.maximum(q + r, occluder)))
                seen &= angle - spread <= reach
            viewed = seen.copy()
            give_way()
            # The ground straight below the camera and around it, in view or not: auto scale and tidal lock measure
            # to it (sphere.py). Only down to the planet's center: beyond it is the far side, whose ground isn't below.
            along = C @ down
            across = np.linalg.norm(C - along[:, None] * down, axis=1)
            under = ((across <= r + margin) & (along >= -r) & (along <= d_camera + r)) | (near <= ball)
            seen |= under
            # Refined last time, a tile stays refined until its error is HYSTERESIS times under the target: at the
            # target, every move across it swapped it with its children and back, and the ground under the drone,
            # which auto scale and the lift follow, with them
            error = self.errors[idx] * K / near
            coarse = error <= np.where(self.refined[idx], target / HYSTERESIS, target)
            if moving and levels:
                # A tile whose place on screen the view's motion smears (the camera's speed across the way to it, over
                # its distance, and the view's turning) is refined only into children that are all in already: their
                # detail is drawn while it's there, but not loaded for a blur. Not the drone's own ground (under),
                # which is what it is at any speed.
                ahead = C @ velocity / np.maximum(distance, 1e-9)
                smear = (np.sqrt(np.maximum(speed2 - ahead * ahead, 0.0)) / near + turning) * (K * FRAME)
                blurred = (seen & ~coarse & ~under
                           & (error <= target * np.clip(smear / MOTION_SHARP, 1.0, MOTION_MOST))
                           & (child_count[idx] > 0))
                if blurred.any():
                    which = np.nonzero(blurred)[0]
                    counts = child_count[idx[which]]
                    children = self.children_of(idx[which])
                    child_kind = kinds[children]
                    # In: a tile's content, or nothing to load, or a tileset's root's. Only a linked tileset's first
                    # child is looked up: the last leaves' first_child is one past their block (flatten()), and past
                    # the arrays for the tree's last rows (skimming low and fast over the newest tiles, the
                    # selections failed on it).
                    there = self.has_tile[children] | (child_kind == NONE)
                    linked = (child_kind == JSON) & (child_count[children] > 0)
                    there[linked] |= self.has_tile[self.first_child[children[linked]]]
                    coarse[which[~np.logical_and.reduceat(there, np.cumsum(counts) - counts)]] = True
            if not levels:
                seen[0] = viewed[0] = True
            self.refined[idx] = seen & ~coarse
            if levels and PREFETCH:
                # Out of view, under a tile that's replaced by its children: loaded too (prefetch, below)
                out = ~seen & (marks[parents[idx]] == REPLACED)
                if out.any():
                    prefetched.append((idx[out], near[out]))
            idx, viewed, under, coarse, near = idx[seen], viewed[seen], under[seen], coarse[seen], near[seen]
            kind = kinds[idx]
            count = child_count[idx]
            # A tile refines into its children if it's too coarse, or has nothing of its own to draw; a tileset's
            # link is its tileset's root, once that's in
            link = kind == JSON
            refine = ~link & (count > 0) & ~(coarse & (kind == GLB))
            replaced = refine & ~((kind == GLB) & self.adds[idx])
            marks[idx] = np.where(replaced, REPLACED, 0)
            levels.append((idx, viewed, under, near, kind, count, refine, replaced))
            idx = self.children_of(idx[(count > 0) & (link | refine)])
        give_way()
        # Every depth's nodes in one, the depths one after another
        bounds = np.cumsum([0] + [len(level[0]) for level in levels])
        idx, viewed, under, near, kind, count, refine, replaced = (
            np.concatenate([level[k] for level in levels]) for k in range(8))
        loaded = self.has_tile[idx]
        # Needed, drawn or not (children waiting for a sibling to replace their parent): kept by trim()
        self.drawn[idx[loaded]] = now
        content = kind == GLB
        link = kind == JSON
        added = refine & content & ~replaced
        # A node's own hole: a tile drawn as it is, or drawn with its children (ADD), that isn't loaded; a link whose
        # tileset isn't in
        missing = ((content & ~replaced) & ~loaded) | (link & (count == 0))
        # Loaded, a tile replaced by its children stands in for their holes; else it passes them up, or if they're
        # many, its whole part of the view as one
        stands = replaced & content & loaded
        passes = replaced & ~stands
        below = np.zeros(len(idx), np.int32)
        own = np.zeros(len(idx), np.int32)
        many = np.zeros(len(idx), bool)
        shown_holes = 0
        # Up: each node's holes, its children's added up (and how many of them are in view, for complete)
        for k in range(len(levels) - 1, -1, -1):
            give_way()
            part = slice(bounds[k], bounds[k + 1])
            nodes = idx[part]
            under_it = hole_sums[nodes]
            below[part] = under_it
            too_many = passes[part] & (under_it > MAX_HOLES)
            many[part] = too_many
            holes = np.where(stands[part], 0, np.where(too_many, 1, under_it + missing[part]))
            own[part] = holes
            if holes.any():
                shown = np.where(stands[part], 0, np.where(too_many, viewed[part],
                                                           shown_sums[nodes] + (missing[part] & viewed[part])))
                if k:
                    has = holes > 0
                    np.add.at(hole_sums, parents[nodes[has]], holes[has])
                    np.add.at(shown_sums, parents[nodes[has]], shown[has])
                else:
                    shown_holes = int(shown[0])
        # Too many holes to clip a stand-in to (MAX_HOLES): the tile itself instead of what's drawn under it
        cover = stands & (below > MAX_HOLES)
        stand_in = stands & (below > 0) & ~cover
        drawn = ((content & ~replaced) & loaded) | cover
        if cover.any():
            # Down: what's under a cover isn't drawn
            marks[idx] = 0
            hidden = np.zeros(len(idx), bool)
            for k in range(len(levels)):
                part = slice(bounds[k], bounds[k + 1])
                if k:
                    hidden[part] = marks[parents[idx[part]]] == COVERED
                marks[idx[part]] = np.where(cover[part] | hidden[part], COVERED, 0)
            drawn &= ~hidden
            stand_in &= ~hidden
        # Each stand-in's holes: the nodes whose parts of the view nothing's drawn in, as they're passed up to it
        fallbacks = []
        if stand_in.any():
            holed = np.nonzero(own > 0)[0]
            place = dict(zip(idx[holed].tolist(), holed.tolist()))
            first_child = self.first_child

            def holes_of(index):
                """What a node passes up: itself, if it's a hole or has too many; else its children's, and itself
                with them if it's drawn with them (ADD) and not in"""
                k = place[index]
                if many[k] or (missing[k] and not added[k]):
                    return [index]
                return holes_under(index) + ([index] if missing[k] else [])

            def holes_under(index):
                first = int(first_child[index])
                out = []
                for child in range(first, first + int(child_count[index])):
                    if child in place:
                        out += holes_of(child)
                return out

            for k in np.nonzero(stand_in)[0].tolist():
                index = int(idx[k])
                holes = tuple(holes_under(index))
                first = int(first_child[index])
                if len(holes) == child_count[index] and all(first <= h < first + len(holes) for h in holes):
                    # Every one of its children is a hole: drawn whole, as any tile (their footprints are all of
                    # it). While a city streams in, most stand-ins are these, waiting for a next level none of
                    # which is in yet, and clipped they cost a uniform buffer each and a shader that tests every
                    # pixel against every hole.
                    drawn[k] = True
                else:
                    fallbacks.append((index, holes, bool(viewed[k])))
                if not k % 16:
                    give_way()
        give_way()
        # What's wanted, by how much it's needed: the ground under the drone first (a finer tile's comes in higher
        # or lower than the coarser one's, by up to its geometric error, tens of meters in Google's mountains, and the
        # sooner the drone flies over it, the less often it's caught under it: sphere.recenter_move() lifts it out),
        # then the view's, the coarsest on screen first (coarseness()), and of those as coarse as each other, the ones
        # nearest the middle of the view, where the flight is headed. Coarsest on screen, not coarsest in the tree: a
        # city seen from among its towers needs about as many tiles of each of ten depths, the deep ones the near
        # ground, and by depth the far rings came before the street ahead, each new ring the flight brought into view
        # before it again, so what the drone flew at stayed three levels too coarse for seconds.
        want = (missing & (self.states[idx] != READY)) | (passes & content & (below > 0))
        wanted = idx[want]
        # A tile's children as near the middle as each other (as their parent is), so they're asked for together and
        # come together: a tile with some of its children in is drawn clipped to the rest
        order = (np.where(under[want], 0, 1), -self.coarseness(wanted, near[want], K, target),
                 -self.toward(np.maximum(parents[wanted], 0), camera, forward))
        if prefetched:
            # Children out of view of a tile that's replaced by its children, loaded too (other 3D Tiles clients'
            # loadSiblings): once the view turns to them, they're there. Without it, the tile stood in for them, and
            # before stand-ins were clipped to what's missing, for all its children at once: a tile of the first
            # levels, 260 km of error, did under the drone at 150 m over the Matterhorn, and auto scale and the lift
            # followed its ground 25 km down. Through tilesets' links and tiles without content, three levels on, to
            # the tiles that can stand in. They come as if PREFETCH_LEVELS finer than they are: the coarse ones (a
            # few, whose stand-in covers the most) soon, the fine ones once the view's own are in.
            nodes = np.concatenate([p[0] for p in prefetched])
            edges = np.concatenate([p[1] for p in prefetched])
            for depth in range(4):
                give_way()
                node_kind = kinds[nodes]
                tile = node_kind == GLB
                there = self.has_tile[nodes]
                self.drawn[nodes[tile & there]] = now
                ready = self.states[nodes] == READY
                ask = (tile & ~there) | ((node_kind == JSON) & ~ready)
                if ask.any():
                    asked = nodes[ask]
                    wanted = np.concatenate([wanted, asked])
                    order = (np.concatenate([order[0], np.ones(len(asked), np.int64)]),
                             np.concatenate([order[1], PREFETCH_LEVELS - self.coarseness(asked, edges[ask], K, target)]),
                             np.concatenate([order[2], -self.toward(np.maximum(parents[asked], 0), camera, forward)]))
                if depth == 3:
                    break
                through = nodes[~tile & ready]
                # As near as the node they're under
                edges = np.repeat(edges[~tile & ready], self.child_count[through])
                nodes = self.children_of(through)
                if not len(nodes):
                    break
                touched.append(nodes)
        wanted = wanted[np.lexsort((wanted, order[2], order[1], order[0]))]
        # Wanted is needed (trim()): a tile that came in between a selection's walk and its trim() was only as needed
        # as when it was last loaded, and past MEMORY it was dropped as it came and asked for again, some of them ten
        # times over, nearly half of all a city flight's loads
        self.drawn[wanted] = now
        # The tilesets walked: kept by prune()
        every = np.concatenate(touched)
        self.block_seen[self.block_of[every]] = now
        hole_sums[idx] = 0
        shown_sums[idx] = 0
        marks[idx] = 0
        # Drawn: what's in view. The ground out of view only measured; its holes don't keep the view from complete.
        ground = idx[drawn].tolist()
        return (idx[drawn & viewed].tolist(), [f[:2] for f in fallbacks if f[2]], ground,
                [f[:2] for f in fallbacks], shown_holes == 0, wanted.tolist())

    def selection(self, name, camera, forward, planes, K, target, nearest, around=0.0, moving=True):
        """The view name's latest selection (select()). A new one is made once the camera has moved SELECT_MOVE of the
        nearest distance drawn or turned SELECT_TURN since the last, or tiles came or went (then at most every
        SELECT_INTERVAL), on the selector thread: the draw never waits for it, it draws the latest one made (the first
        is made at once). A selection walks a thousand or two of the tree's nodes in a few milliseconds, but in the
        draw, with the loader's threads busy, waiting for Python's lock took that to 20 to 70 ms. Each new selection's
        loads are requested here, and what's not needed any more trimmed. With ASYNC off (tests), made at once. around:
        select()'s (the drone's view's). moving: whether the view's motion counts (the "tiles_motion" setting): off,
        it's selected for as if it were still."""
        key = (round(math.log2(max(K, 1e-30)) * 16), target, planes.shape)
        last = self.selection_cache.get(name)
        if last is not None and last.generation != self.generation:
            last = None
        now = time.perf_counter()
        velocity, turning = self.motion(name, now, camera, forward) if moving else (np.zeros(3), 0.0)
        # How the nearest ground moves on screen, pixels a frame, within what it changes of the selection
        # (MOTION_SHARP to MOTION_MOST times it)
        smear = min(max((float(np.linalg.norm(velocity)) / max(nearest, 1e-9) + turning) * K * FRAME, MOTION_SHARP),
                    MOTION_SHARP * MOTION_MOST)
        if (last is None or last.key != key
                or (last.version != self.version and now - last.made >= SELECT_INTERVAL)
                or (now - last.made >= SELECT_MIN_INTERVAL
                    and (np.linalg.norm(camera - last.camera) >= SELECT_MOVE * nearest
                         or np.linalg.norm(forward - last.forward) >= SELECT_TURN
                         # Slowed down (or sped up) since: the detail its motion left out is wanted now
                         or not 0.75 <= smear / last.smear <= 1.0 / 0.75))):
            view = (np.array(camera, dtype=np.float64), np.array(forward, dtype=np.float64), np.array(planes), K,
                    target, key, around, (velocity, turning), smear)
            if ASYNC and last is not None:
                self.asked[name] = view
            else:
                self.make_selection(name, view)
        latest = self.selection_cache.get(name)
        if latest is not None:
            # Every draw, as results come in and make room: asked for only once a selection, a fifth of the loader's
            # connections waited for the next one
            if latest.asked < len(latest.wanted) and self.in_flight < IN_FLIGHT:
                latest.asked = self.request(latest.wanted, latest.asked)
            if not latest.applied:
                latest.applied = True
                self.trim()
                self.prune()
        return latest

    def motion(self, name, now, camera, forward):
        """The view name's motion as of this draw: the camera's velocity (meters a second) and how fast the view turns
        (radians a second), smoothed over MOTION_SMOOTH seconds (a frame's own are its hitches too)"""
        last = self.motions.get(name)
        velocity, turning = np.zeros(3), 0.0
        if last is not None and now - last[0] < 0.5:
            then, where, facing, velocity, turning = last
            dt = now - then
            if dt < 1e-4:
                return velocity, turning
            share = 1.0 - math.exp(-dt / MOTION_SMOOTH)
            velocity = velocity + share * ((camera - where) / dt - velocity)
            turning = turning + share * (float(np.linalg.norm(forward - facing)) / dt - turning)
        self.motions[name] = (now, np.array(camera, dtype=np.float64), np.array(forward, dtype=np.float64), velocity,
                              turning)
        return velocity, turning

    def start_selections(self):
        """The views selection() was asked new selections for, given to the selector thread. From the end of the draw
        (sphere.Floor.draw_tiles()): started in the middle of it, the selection took Python's lock from the draw's own
        Python in turns, each waiting for the other, when after the draw Blender itself has a few milliseconds'
        work that needs no Python at all."""
        if self.asked:
            with self.views_changed:
                self.views.update(self.asked)
                self.asked.clear()
                self.views_changed.notify()

    def make_selection(self, name, view):
        """select() for a view (selection()'s), kept as the view name's latest Selection"""
        camera, forward, planes, K, target, key, around, motion, smear = view
        generation, version = self.generation, self.version
        self.selections += 1
        draw, fallbacks, ground, ground_fallbacks, complete, wanted = self.select(camera, forward, planes, K, target,
                                                                                 around, motion)
        if generation != self.generation:
            return              # the tree was made again meanwhile (reset())
        selection = Selection(draw, fallbacks, ground, ground_fallbacks, complete, camera, forward, key)
        selection.generation, selection.version, selection.made = generation, version, time.perf_counter()
        selection.number = self.selections
        selection.wanted, selection.applied, selection.asked = wanted, False, 0
        selection.smear = smear
        self.selection_cache[name] = selection

    def select_views(self):
        """The selector thread: selections for the views selection() gives it, the latest of each"""
        while True:
            with self.views_changed:
                while not self.views and not self.closed:
                    self.views_changed.wait()
                if self.closed:
                    return
                name, view = self.views.popitem()
            try:
                self.make_selection(name, view)
            except Exception as error:      # kept for the view's corner, the thread kept going
                self.error_message = f"selection: {type(error).__name__}: {error}"

    def nearest(self, selection, camera):
        """The distance from the camera to the nearest tile selected (drawn or the ground), outside its bounding sphere
        and box, meters (for selection()): near the ground, the camera is inside the spheres of the tiles under it, and
        with the sphere alone, every frame made a new selection"""
        if not selection.ground:
            return float(np.linalg.norm(camera) - RADIUS) if np.linalg.norm(camera) > RADIUS else 1.0
        idx = np.asarray(selection.ground)
        C = self.centers[idx] - camera
        outside = np.maximum(np.abs(np.einsum("ijk,ik->ij", self.axes[idx], C)) - self.halves[idx], 0.0)
        d = np.maximum(np.linalg.norm(C, axis=1) - self.radii[idx], np.linalg.norm(outside, axis=1))
        return max(float(np.min(d)), 1e-3)

    def batch(self, items):
        """A draw list's loaded Tiles, their centers (meters) and, for fallbacks, their holes' uniform buffers
        (hole_values()): made once for each list, until a tile's dropped (a selection's lists are drawn every frame
        until the next)"""
        kept = self.batches.get(id(items))
        if kept is None or kept[0] is not items or kept[1] != self.drops:
            holes = None
            if items and isinstance(items[0], tuple):
                entries = [entry for entry in items if entry[0] in self.tiles]
                tiles = [self.tiles[entry[0]] for entry in entries]
                holes = [self.hole_values(*entry) for entry in entries]
            else:
                tiles = [self.tiles[i] for i in items if i in self.tiles]
            kept = (items, self.drops, tiles, np.array([tile.center for tile in tiles]).reshape(-1, 3), holes)
            if len(self.batches) > 8:
                self.batches.clear()
            self.batches[id(items)] = kept
        return kept[2:]

    def draw(self, draw_list, camera, s, values, fallbacks=()):
        """The tiles of a draw list, and fallbacks clipped to their holes (Selection), into the bound framebuffer
        (depth tested and written by the caller's state): camera in meters, s map units a meter, values the frame's
        uniform buffer (frame_values()). All of the selection's: its view is a little wider than this one
        (SELECT_MARGIN), but finding the tenth of its tiles that are out of this one cost more than drawing them."""
        import gpu
        if self.shader is None:
            return
        if self.white is None:
            self.white = gpu.types.GPUTexture((1, 1), format='SRGB8_A8',
                                              data=gpu.types.Buffer('FLOAT', 4, [1.0, 1.0, 1.0, 1.0]))
        white = self.white
        for clip, items in ((False, draw_list), (True, fallbacks)):
            tiles, centers, holes = self.batch(items)
            if not tiles:
                continue
            shader = self.shader
            if clip:
                if self.clip_shader is None:
                    self.clip_shader = make_shader(clip=True)
                shader = self.clip_shader
            # Every tile's offset at once, as two float32s each whose sum it is (numpy's calls cost more than their
            # arithmetic, a tile at a time): a row a tile, as the shader's uTile takes it
            offset = (centers - camera) * s
            rows = np.empty((len(tiles), 8), np.float32)
            rows[:, :3] = offset
            rows[:, 3] = s
            rows[:, 4:7] = offset - rows[:, :3]
            rows[:, 7] = 1.0
            shader.bind()
            shader.uniform_block("tileValues", values)
            place = shader.uniform_from_name("uTile")
            for k, (tile, row) in enumerate(zip(tiles, rows)):
                if clip:
                    shader.uniform_block("holeValues", holes[k])
                shader.uniform_vector_float(place, row, 4, 2)
                for batch, texture in tile.parts:
                    shader.uniform_sampler("tileColor", texture or white)
                    batch.draw(shader)

    def hole_values(self, index, holes):
        """A fallback's uniform buffer (its shader's HoleValues): each hole's footprint, its box (box_spheres()) seen
        from its up, in meters from the tile's center (float32 is ample there): its center and its two axes across its
        up, with their half lengths; the rest never met. Kept for the same tile and holes."""
        import gpu
        tile = self.tiles[index]
        key = (index, id(tile), holes)
        buffer = self.hole_buffers.get(key)
        if buffer is not None:
            return buffer
        rows = np.zeros((3 * MAX_HOLES, 4), np.float32)
        rows[0::3, 3] = -1.0
        for k, h in enumerate(holes[:MAX_HOLES]):
            center = self.centers[h]
            U, L = self.axes[h], self.halves[h]
            up = center / max(np.linalg.norm(center), 1e-9)
            across = sorted(range(3), key=lambda a: abs(float(U[a] @ up)))[:2]
            rows[3 * k] = (*(center - tile.center), L[across[0]])
            rows[3 * k + 1] = (*U[across[0]], L[across[1]])
            rows[3 * k + 2] = (*U[across[1]], 0.0)
        if len(self.hole_buffers) > 512:
            self.hole_buffers.clear()
        buffer = self.hole_buffers[key] = gpu.types.GPUUniformBuf(rows.ravel())
        return buffer

    def ground_boxes(self, draw_list):
        """A draw list's loaded tiles and their bounding boxes (centers, axes, half lengths: box_spheres()), for
        ground() and nearest_ground(), made once for each list (the last two: the draw list's and the fallbacks')
        until a tile's dropped. Their boxes, not their spheres: a flat tile's sphere is as wide as the tile, so the
        line straight down met a dozen tiles' spheres, and a drone low over the ground was inside as many, each one
        then searched triangle by triangle."""
        kept = self.ground_kept.get(id(draw_list))
        if kept is None or kept[0] is not draw_list or kept[1] != self.drops:
            idx = [i for i in draw_list if i in self.tiles]
            rows = np.array(idx, dtype=np.int64)
            kept = (draw_list, self.drops, idx, self.centers[rows], self.axes[rows], self.halves[rows])
            if len(self.ground_kept) > 2:
                self.ground_kept.clear()
            self.ground_kept[id(draw_list)] = kept
        return kept[2:]

    def ground(self, draw_list, point, direction):
        """Along the line from point (meters) in direction (unit, down), the nearest of the drawn tiles' surfaces ahead
        (t > 0), else behind (t < 0, so point is under the ground): t, or None if the line meets none of them. Ahead
        only as far as the planet's center: from under the ground, the line's next surface ahead was the far side of
        the planet, 12,742 km down."""
        reach = float(np.linalg.norm(point))
        idx, centers, axes, halves = self.ground_boxes(draw_list)
        if not idx:
            return None
        # The tile the ground was found in last, alone first: drawn tiles don't overlap, so it's usually still the one
        last = self.tiles.get(self.ground_tile) if self.ground_tile in draw_list else None
        if last is not None and last.last_hit is not None:
            t = triangle_hit(last.mesh[0], last.mesh[1][last.last_hit], point - last.center, direction)
            if t is not None and t <= reach:
                return t
        met = np.nonzero(line_meets_boxes(point, direction, centers, axes, halves))[0]
        t, found = self.first_surface([idx[k] for k in met], point, direction, reach)
        if found is not None:
            self.ground_tile = found
        return t

    def first_surface(self, indexes, point, direction, reach):
        """Of these tiles, the nearest surface ahead along the line from point, else the nearest behind (ground()),
        and the tile it's in: (t, index), or (None, None)"""
        ahead, behind, found = math.inf, -math.inf, None
        for index in indexes:
            tile = self.tiles[index]
            t = tile.ground(point - tile.center, direction)
            t = t[t <= reach]
            if len(t):
                if (t >= 0.0).any() and float(t[t >= 0.0].min()) < ahead:
                    ahead = float(t[t >= 0.0].min())
                    found = index
                if (t < 0.0).any() and float(t[t < 0.0].max()) > behind:
                    behind = float(t[t < 0.0].max())
                    if not math.isfinite(ahead):
                        found = index
        if math.isfinite(ahead):
            return ahead, found
        if math.isfinite(behind):
            return behind, found
        return None, None

    def hole_ground(self, fallbacks, point, direction):
        """ground() where the drawn tiles have a hole: of the stand-in drawn there, the one with a hole whose footprint
        the point is over (as its shader clips it to them: hole_values()). A selection's stand-ins are coarser tiles
        of every size, one over another, each drawn only in its own holes: the nearest surface of them all along the
        line was often one standing in for a hole somewhere else, a hundred meters over what's drawn here, and auto
        scale shrank the drone to less than half in a tick."""
        kept = self.holes_kept
        if kept is None or kept[0] is not fallbacks or kept[1] != self.drops:
            owners, holes = [], []
            for index, tile_holes in fallbacks:
                if index in self.tiles:
                    owners += [index] * len(tile_holes)
                    holes += tile_holes
            rows = np.array(holes, dtype=np.int64)
            centers, U, L = self.centers[rows], self.axes[rows], self.halves[rows]
            # Each hole's box's two axes across its up, and their half lengths
            up = centers / np.maximum(np.linalg.norm(centers, axis=1), 1e-9)[:, None]
            across = np.argsort(np.abs(np.einsum("ijk,ik->ij", U, up)), axis=1, kind="stable")[:, :2]
            each = np.arange(len(rows))[:, None]
            kept = self.holes_kept = (fallbacks, self.drops, owners, centers, U[each, across], L[each, across])
        owners, centers, axes, halves = kept[2:]
        if not owners:
            return None
        over = np.all(np.abs(np.einsum("ijk,ik->ij", axes, point - centers)) <= halves, axis=1)
        stand_ins = list(dict.fromkeys(owners[k] for k in np.nonzero(over)[0]))
        return self.first_surface(stand_ins, point, direction, float(np.linalg.norm(point)))[0]

    def loaded_around(self, point, reach):
        """The loaded tiles a line from within reach meters of point (meters, across the way up) to the planet's center
        can meet, for loaded_ground(): those whose boxes come within reach of the line through point and the center,
        the finest first, as (nodes, centers, axes, half lengths, half diagonals). Kept for AROUND_KEPT seconds, while
        point's moves and the reach's growth add up to less than AROUND_MOVE of the reach: follow_base() asks every
        tick, and made from every loaded tile (thousands over a city), each probe took a third of a millisecond or
        more. A tile loaded since isn't in it yet (the coarser one it refines is), one dropped since is passed over."""
        kept = self.around_kept
        now = time.perf_counter()
        # Kept while every line it was made for still covers those from round point (lines to the planet's center
        # only come nearer each other further down)
        if (kept is not None and now - kept[0] < AROUND_KEPT
                and reach + float(np.linalg.norm(point - kept[1])) <= kept[2]):
            return kept[3]
        rows = np.fromiter(self.tiles, np.int64, len(self.tiles))
        centers = self.centers[rows]
        up = point / np.linalg.norm(point)
        across = centers - np.outer(centers @ up, up)
        # Room for the point's moves and the reach's growth while it's kept
        wide = reach * (1.0 + AROUND_MOVE)
        near = np.linalg.norm(across, axis=1) - np.linalg.norm(self.halves[rows], axis=1) <= wide
        rows = rows[near]
        rows = rows[np.argsort(-self.depths[rows], kind="stable")]
        halves = self.halves[rows]
        boxes = (rows, self.centers[rows], self.axes[rows], halves, np.linalg.norm(halves, axis=1))
        self.around_kept = (now, point.copy(), wide, boxes)
        return boxes

    def loaded_ground(self, point, direction, among=None, lazy=False):
        """ground() in the finest loaded tile the line meets, selected or not: where the selection's ground has a gap
        (the drone flown off it since), coarser or finer than it would be, but the ground nearby, not none. Only among
        those given (loaded_around()) if any: follow_base() looks down from a few points a tick, each of which went
        through every loaded tile. lazy: None, not known yet, where a tile whose triangles' boxes aren't made
        (Tile.triangle_boxes()) would be searched, its boxes made on the worker thread (background.py) for a later
        look: follow_base() looks down beside the drone, where nothing else has looked, and a tile's first search
        makes its boxes, a millisecond a city tile (41 of a round's 49 ms over Tokyo, 2026-11-17)."""
        if not self.tiles:
            return None
        if among is None:
            rows = np.fromiter(self.tiles, np.int64, len(self.tiles))
            rows = rows[np.argsort(-self.depths[rows], kind="stable")]
            met = rows[line_meets_boxes(point, direction, self.centers[rows], self.axes[rows], self.halves[rows])]
        else:
            rows, centers, axes, halves, diagonals = among
            # The boxes whose spheres the line passes through, first: a few of the hundreds near the drone
            v = centers - point
            close = np.flatnonzero(np.linalg.norm(v - np.outer(v @ direction, direction), axis=1) <= diagonals)
            met = rows[close[line_meets_boxes(point, direction, centers[close], axes[close], halves[close])]]
        reach = float(np.linalg.norm(point))
        lazy = lazy and background is not None and background.enabled
        if lazy:
            background.finished(BOXES_OWNER)
        for index in met.tolist():
            tile = self.tiles.get(index)
            if tile is None:
                continue
            if lazy and tile.boxes is None:
                # Not known here yet: a coarser tile's ground (hundreds of meters off, over Tokyo) isn't the answer
                background.submit((BOXES_OWNER, index), tile.triangle_boxes)
                return None
            t = tile.ground(point - tile.center, direction)
            t = t[t <= reach]
            if len(t):
                return float(t[t >= 0.0].min()) if (t >= 0.0).any() else float(t.max())
        return None

    def nearest_ground(self, draw_list, point, limit):
        """The distance from point (meters) to the nearest of the drawn tiles' surfaces, if under limit (meters), else
        limit, and that nearest point (meters; None if none was under limit)"""
        idx, centers, axes, halves = self.ground_boxes(draw_list)
        if not idx:
            return limit, None
        # How far each tile's box is, at least: nearest first, as far as the nearest found so far
        outside = np.maximum(np.abs(np.einsum("ijk,ik->ij", axes, point - centers)) - halves, 0.0)
        gap = np.sqrt(np.einsum("ij,ij->i", outside, outside))
        nearest = None
        for k in np.argsort(gap):
            if gap[k] >= limit:
                break
            tile = self.tiles.get(idx[k])
            if tile is not None:
                d, q = tile.nearest(point - tile.center, limit)
                if q is not None and d < limit:
                    limit, nearest = d, tile.center + q
        return limit, nearest

    def attribution(self, draw_list):
        """Google's terms' line: the data providers of the tiles drawn (draw_list: with the fallbacks' tiles), most
        frequent first"""
        kept = self.attributed
        if kept is None or kept[0] is not draw_list:
            # A selection's list is drawn every frame until the next: its tiles' providers counted once
            counts = Counter()
            for index in list(draw_list):
                tile = self.tiles.get(index)
                if tile is not None and tile.copyright:
                    counts.update(part.strip() for part in tile.copyright.split(";") if part.strip())
            kept = self.attributed = (draw_list, "; ".join(name for name, _count in counts.most_common()))
        return kept[1]

    def info(self):
        return (f"tiles {len(self.tiles)} loaded ({self.bytes / 1e6:.0f} MB), {self.in_flight} loading, {self.live} nodes, "
                f"{self.loader.requests} requests ({self.loader.received / 1e6:.0f} MB)")

    def close(self):
        self.loader.stop()
        with self.views_changed:
            self.closed = True
            self.views_changed.notify()
        self.drop_all()


def fog_rows(color, distance, air=None):
    """The fog's rows of the shaders' uniforms (FOG_SOURCE): uFog, uAir, uUp, uHaze and uClearAir. color: the fog's
    (linear); distance: how far into plain fog the light fades (world units, EXTINCTION times as far for each channel);
    air: a planet's, (h0, c, up, error, haze, clear): the camera's height and its distance from the center (world
    units), the way up (world space), the far end's height error per unit of distance, and each layer's (extinction per
    world unit at the ground (rgb), scale height); or None for plain fog."""
    def fit(x):
        return max(-1e30, min(1e30, float(x)))
    if air is None:
        return [(*color, fit(distance)), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0, 1.0)]
    height, center, up, error, haze, clear = air
    # Past 10^18 the planet's curve is nothing to the view (and the shader squares the view's reach over it)
    return [(*color, fit(distance)), (fit(height), min(float(center), 1e18), fit(error), 1.0), (*up, 0.0),
            (*[fit(x) for x in haze[0]], fit(haze[1])), (*[fit(x) for x in clear[0]], fit(clear[1]))]


def frame_values(view_projection, world_from_map, fog):
    """The tiles' uniform buffer for a frame (TILE_MATRICES, TILE_ROWS); fog: fog_rows()"""
    import gpu
    data = np.zeros(16 * len(TILE_MATRICES) + 4 * len(TILE_ROWS), np.float32)
    for i, matrix in enumerate((view_projection, world_from_map)):
        m4 = np.eye(4)
        m = np.asarray(matrix, dtype=float)
        m4[:m.shape[0], :m.shape[1]] = m
        data[16 * i:16 * i + 16] = m4.T.ravel()          # GLSL keeps matrices by columns
    data[32:32 + 4 * len(fog)] = np.ravel(fog)
    return gpu.types.GPUUniformBuf(data)


def view_planes(view_projection, world_from_map, s, far=True, margin=SELECT_MARGIN):
    """The view's frustum planes in meters from the camera (n unit, n.x + d >= 0 inside): left, right, bottom, top and
    the far plane, from the clip matrix of map units from the camera (view_projection after world_from_map), its sides
    widened by margin (of their tangents)"""
    clip = np.asarray(view_projection, dtype=np.float64)
    M = np.empty((4, 4))
    M[:, :3] = clip[:, :3] @ np.asarray(world_from_map, dtype=np.float64)
    M[:, 3] = clip[:, 3]
    M[:2] /= margin
    planes = np.stack([M[3] + M[0], M[3] - M[0], M[3] + M[1], M[3] - M[1]] + ([M[3] - M[2]] if far else []))
    # In map units n.x + d; x = s meters: (s n).x_m + d, made unit
    planes[:, :3] *= s
    return planes / np.sqrt(np.einsum("ij,ij->i", planes[:, :3], planes[:, :3]))[:, None]


def draw_attribution(tileset, draw_list, ui_scale=1.0):
    """The attribution Google's terms require, in the view's bottom left corner, clear of the add-on's HUD (its scale
    readout is in the bottom right): Google's name (Google Maps' logo's text), then the data providers, or the
    tileset's trouble if it has any. Sized by Blender's UI scale, so it reads on a Retina screen too."""
    import blf
    providers = tileset.attribution(draw_list)
    text = ("Google" if tileset.google else "3D Tiles") + (f"  ·  {providers}" if providers else "")
    if tileset.error_message and not draw_list:
        text += f"  ·  {tileset.error_message}"
    blf.size(0, 12.0 * ui_scale)
    blf.color(0, 1.0, 1.0, 1.0, 0.9)
    blf.enable(0, blf.SHADOW)
    blf.shadow(0, 3, 0.0, 0.0, 0.0, 0.8)
    blf.position(0, 10.0 * ui_scale, 8.0 * ui_scale, 0)
    blf.draw(0, text)
    blf.disable(0, blf.SHADOW)


_tilesets = {}


def tileset(source):
    """The Tileset of a source ("google", or a tileset's URL or path), made the first time"""
    key = source if source == GOOGLE or is_web(source) else os.path.abspath(os.path.expanduser(source))
    t = _tilesets.get(key)
    if t is None:
        if key != GOOGLE and not is_web(key):
            key_url = urllib.parse.urljoin("file:", urllib.request.pathname2url(key))
        else:
            key_url = key
        if not _tilesets:
            _switch_interval[0] = sys.getswitchinterval()
            sys.setswitchinterval(min(_switch_interval[0], SWITCH_INTERVAL))
        t = _tilesets[key] = Tileset(key_url)
    return t


_switch_interval = [None]       # Python's, before tilesets set theirs (tileset())


def forget():
    for t in _tilesets.values():
        t.close()
    _tilesets.clear()
    if _switch_interval[0] is not None:
        sys.setswitchinterval(_switch_interval[0])
        _switch_interval[0] = None


# Worker threads stopped when Python ends, not left mid-request while it shuts down
atexit.register(forget)
