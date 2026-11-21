"""Make a planet for sphere maps' "planet" setting (eternal_return/planet.py draws it): a tile pyramid of an
equirectangular image of a whole planet, from a single image or a grid of equal images covering it (NASA's Blue Marble
Next Generation comes as 4 x 2 of 21600 x 21600, A1..D2).

    Blender -b --factory-startup --python tools/make_planet.py -- --out ~/blender_tests/sphere/earth/bmng_07 \
        --grid 4x2 A1.jpg B1.jpg C1.jpg D1.jpg A2.jpg B2.jpg C2.jpg D2.jpg [--name ...] [--credit ...] [--quality 90]
    ... -- --out <dir> <equirect.jpg>                       (one image of the whole planet, 2:1)

The images are listed west to east, then north to south. Longitude -180 is the left edge, latitude 90 the top.

The pyramid (read by planet.Pyramid):
- planet.json: the tile size T, the level-0 grid (columns x rows, 2:1), the number of levels, name and credit
- level<k>.pack: every tile of level k, JPEG (or PNG with --format png, lossless), one after another;
  level<k>.index.npy: int64 (rows, columns, 2), each tile's offset and length in the pack. Rows count from the south
  pole (row 0) and columns east from longitude -180, as ImBuf's pixels are bottom-up.
- Level k is columns x rows tiles times 2^k each way, its pixels 360 / (columns T 2^k) degrees; the last level is the
  images' own resolution.
- Each tile is T + 2 pixels square: its T x T and a pixel of its neighbors' all round (across longitude 180 the other
  side's, past a pole its own edge again), so the GPU can blend across tile edges without fetching two tiles.

Levels finer than the coarsest whose whole image fits MOSAIC_BYTES are made image by image (each with the 2^k-pixel
edges of its neighbors, read first); the rest from one image of the whole planet at that level, in memory. Each level
is the one finer averaged 2 x 2 (in sRGB, as map tiles usually are).
"""

import json
import math
import os
import sys
import tempfile
import time

import numpy as np

MOSAIC_BYTES = float(os.environ.get("PLANET_MOSAIC_BYTES", 1.0e9))   # the most a whole level may take in memory (RGB)
QUALITY = 90
TILE_RANGE = (200, 512)     # a tile's size T is picked in this range


def parse_args(argv):
    args = {"out": None, "grid": (1, 1), "images": [], "name": None, "credit": "", "quality": QUALITY,
            "format": "jpeg"}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--out", "--name", "--credit", "--quality", "--grid", "--format"):
            value = argv[i + 1]
            i += 2
            if a == "--grid":
                columns, rows = value.lower().split("x")
                args["grid"] = (int(columns), int(rows))
            elif a == "--quality":
                args["quality"] = int(value)
            elif a == "--format":
                assert value in ("jpeg", "png"), "--format jpeg or png"
                args["format"] = value
            else:
                args[a[2:]] = os.path.abspath(os.path.expanduser(value)) if a == "--out" else value
        else:
            args["images"].append(os.path.abspath(os.path.expanduser(a)))
            i += 1
    assert args["out"] and args["images"], "make_planet.py -- --out <dir> [--grid CxR] <images...>"
    assert len(args["images"]) == args["grid"][0] * args["grid"][1], "one image per cell of the grid"
    return args


def geometry(width, height):
    """The pyramid for a planet width x height pixels at its finest (2:1): (T, level-0 columns, levels), the level-0
    image as small as it can be with T in TILE_RANGE and an even number of columns"""
    assert width == 2 * height, f"an equirectangular planet is twice as wide as it's high, not {width} x {height}"
    best = None
    for finest in range(0, 16):
        if width % (1 << finest):
            break
        base = width >> finest
        for T in range(TILE_RANGE[1], TILE_RANGE[0] - 1, -1):
            if base % (2 * T) == 0:
                candidate = (base, T, base // T, finest + 1)
                if best is None or candidate[0] < best[0]:
                    best = candidate
                break
    assert best is not None, f"no tile size in {TILE_RANGE} divides {width} x {height} into levels"
    _base, T, columns, levels = best
    return T, columns, levels


def with_pixels(path, use):
    """use(pixels) for an image's pixels, RGB, bottom-up (row 0 the south edge), a view into its ImBuf: use must keep
    no view of them (the buffer can't be let go while one is left), only what it makes of them"""
    import imbuf
    ib = imbuf.load(path)
    assert ib is not None, path
    if ib.buffer_type != 'BYTE':
        ib.convert_buffer_type('BYTE')
    with ib.with_buffer() as buffer:
        pixels = np.asarray(buffer)[..., :3]
        result = use(pixels)
        del pixels
    ib.free()
    return result


def half(a):
    """a (h, w, 3) uint8, h and w even, averaged 2 x 2, in bands of rows so a big image needs no big temporaries"""
    h, w = a.shape[0] // 2, a.shape[1] // 2
    out = np.empty((h, w, 3), np.uint8)
    band = max(1, (1 << 24) // max(1, w))
    for r in range(0, h, band):
        s = slice(2 * r, 2 * min(h, r + band))
        q = (a[s][0::2, 0::2].astype(np.uint16) + a[s][1::2, 0::2] + a[s][0::2, 1::2] + a[s][1::2, 1::2] + 2) >> 2
        out[r:r + q.shape[0]] = q
    return out


class Writer:
    """Writes a level's tiles, JPEG, into its pack and index"""

    def __init__(self, out, level, rows, columns, T, quality, file_format="jpeg"):
        import imbuf
        self.pack_path = os.path.join(out, f"level{level}.pack")
        self.index_path = os.path.join(out, f"level{level}.index.npy")
        self.pack = open(self.pack_path, "wb")
        self.index = np.full((rows, columns, 2), -1, np.int64)
        self.size = T + 2
        self.ib = imbuf.new((self.size, self.size), planes=24)
        self.ib.file_type = 'JPEG' if file_format == "jpeg" else 'PNG'
        if file_format == "jpeg":
            self.ib.quality = quality
        self.tmp = os.path.join(tempfile.gettempdir(), f"make_planet_{os.getpid()}.{file_format}")
        self.offset = 0

    def write(self, row, column, tile):
        import imbuf
        assert tile.shape == (self.size, self.size, 3), tile.shape
        with self.ib.with_buffer(write=True) as buffer:
            b = np.asarray(buffer)
            b[..., :3] = tile
            b[..., 3] = 255
        imbuf.write(self.ib, filepath=self.tmp)
        with open(self.tmp, "rb") as f:
            data = f.read()
        self.pack.write(data)
        self.index[row, column] = (self.offset, len(data))
        self.offset += len(data)

    def close(self):
        self.pack.close()
        assert (self.index[..., 1] > 0).all(), f"{self.pack_path}: tiles missing"
        np.save(self.index_path, self.index)
        if os.path.exists(self.tmp):
            os.remove(self.tmp)
        return self.offset


def edges(a, width):
    """An image's edges, width pixels deep: (south rows, north rows, west columns, east columns)"""
    return (a[:width].copy(), a[-width:].copy(), a[:, :width].copy(), a[:, -width:].copy())


def padded(a, i, j, grid, strips, width):
    """Image (i, j) of the grid (i east from longitude -180, j north from the south pole) with width pixels of its
    neighbors' all round: west and east across longitude 180 too, the corners from the diagonal neighbors, and past a
    pole its own edge row again (clamp_poles())"""
    columns, rows = grid
    h, w = a.shape[:2]
    P = np.empty((h + 2 * width, w + 2 * width, 3), np.uint8)
    P[width:width + h, width:width + w] = a

    def strip(ii, jj, which):
        return strips[(ii % columns, jj)][which]
    west, east = (i - 1) % columns, (i + 1) % columns
    P[width:width + h, :width] = strip(west, j, 3)                # the west neighbor's east columns
    P[width:width + h, width + w:] = strip(east, j, 2)
    for side, jj, rows_of in ((0, j - 1, 1), (1, j + 1, 0)):      # south: the one below's north rows; north: above's south
        if 0 <= jj < rows:
            target = slice(0, width) if side == 0 else slice(width + h, None)
            P[target, width:width + w] = strip(i, jj, rows_of)
            P[target, :width] = strip(west, jj, rows_of)[:, -width:]
            P[target, width + w:] = strip(east, jj, rows_of)[:, :width]
    clamp_poles(P, width, j == 0, j == rows - 1)
    return P


def clamp_poles(P, border, south, north):
    """Past a pole, a padded image's border rows are its edge row again, borders and all (as a whole level's tiles
    clamp there, write_level_from_mosaic()): after every halving too, since halving the padded image would average
    the edge row with itself instead of with the row next to it"""
    if south:
        P[:border] = P[border]
    if north:
        P[-border:] = P[-border - 1]


def write_tiles(writer, P, T, row0, column0, tiles_down, tiles_across, border):
    """Tiles from a padded image P (border pixels round it): tile (r, c) of it is tile (row0 + r, column0 + c) of the
    level, its T x T at P[border + r T, border + c T] and a pixel round that"""
    for r in range(tiles_down):
        for c in range(tiles_across):
            y, x = border + r * T - 1, border + c * T - 1
            writer.write(row0 + r, column0 + c, P[y:y + T + 2, x:x + T + 2])


def write_level_from_mosaic(writer, M, T):
    """Every tile of a level from its whole image M: across longitude 180 wrapped, past the poles the edge again"""
    h, w = M.shape[:2]
    for r in range(h // T):
        ys = np.clip(np.arange(r * T - 1, r * T + T + 1), 0, h - 1)
        for c in range(w // T):
            xs = np.arange(c * T - 1, c * T + T + 1) % w
            writer.write(r, c, M[np.ix_(ys, xs)])


def main():
    args = parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    out, (grid_columns, grid_rows), images = args["out"], args["grid"], args["images"]
    os.makedirs(out, exist_ok=True)
    start = time.time()

    # The grid's images by (i east, j north): listed west to east, then north to south
    cells = {}
    for k, path in enumerate(images):
        row_from_north, i = divmod(k, grid_columns)
        cells[(i, grid_rows - 1 - row_from_north)] = path
    import imbuf
    first = imbuf.load(images[0])
    image_w, image_h = first.size
    first.free()
    width, height = image_w * grid_columns, image_h * grid_rows
    T, columns, levels = geometry(width, height)
    rows = columns // 2
    finest = levels - 1
    # The mosaic level: the finest whose whole image fits in memory, and no finer than any image's own tiles allow
    mosaic = 0
    for k in range(finest, -1, -1):
        if (columns * T << k) * (rows * T << k) * 3 <= MOSAIC_BYTES:
            mosaic = k
            break
    per_image = list(range(finest, mosaic, -1))          # made image by image
    for k in per_image:
        assert (image_w >> (finest - k)) % T == 0 and (image_h >> (finest - k)) % T == 0, (k, image_w, T)
    strip_width = 1 << max(0, len(per_image) - 1)
    print(f"planet {width} x {height}: T {T}, level 0 {columns} x {rows} tiles, {levels} levels, "
          f"levels {per_image} image by image, {mosaic}..0 from the whole level {mosaic}", flush=True)

    writers = {k: Writer(out, k, rows << k, columns << k, T, args["quality"], args["format"]) for k in range(levels)}
    M = np.empty(((rows * T) << mosaic, (columns * T) << mosaic, 3), np.uint8)

    if per_image:
        # Every image's edges first, for its neighbors' tile borders
        strips = {}
        for cell, path in cells.items():
            strips[cell] = with_pixels(path, lambda a: edges(a, strip_width))
        for (i, j), path in sorted(cells.items()):
            t0 = time.time()
            P = with_pixels(path, lambda a: padded(a, i, j, (grid_columns, grid_rows), strips, strip_width))
            border = strip_width
            for k in per_image:
                h, w = P.shape[0] - 2 * border, P.shape[1] - 2 * border
                write_tiles(writers[k], P, T, j * (h // T), i * (w // T), h // T, w // T, border)
                if k > mosaic + 1:
                    P = half(P)
                    border //= 2
                    clamp_poles(P, border, j == 0, j == grid_rows - 1)
                else:
                    # The next level is the mosaic's: this image's part of it, without the border
                    part = half(P[border:-border, border:-border])
                    M[j * part.shape[0]:(j + 1) * part.shape[0], i * part.shape[1]:(i + 1) * part.shape[1]] = part
            del P
            print(f"  image {os.path.basename(path)} ({i}, {j}): {time.time() - t0:.1f} s", flush=True)
    else:
        def into_mosaic(a, i, j):
            part = a
            for _ in range(finest - mosaic):
                part = half(part)
            M[j * part.shape[0]:(j + 1) * part.shape[0], i * part.shape[1]:(i + 1) * part.shape[1]] = part
        for (i, j), path in sorted(cells.items()):
            with_pixels(path, lambda a: into_mosaic(a, i, j))

    for k in range(mosaic, -1, -1):
        t0 = time.time()
        write_level_from_mosaic(writers[k], M, T)
        print(f"  level {k} from the whole level: {time.time() - t0:.1f} s", flush=True)
        if k > 0:
            M = half(M)

    sizes = {k: writers[k].close() for k in range(levels)}
    meta = {
        "format": args["format"], "tile": T, "border": 1, "columns": columns, "rows": rows, "levels": levels,
        "name": args["name"] or os.path.basename(out.rstrip("/")),
        "credit": args["credit"],
        "sources": [os.path.basename(p) for p in images], "grid": [grid_columns, grid_rows],
        "bytes": sizes,
    }
    with open(os.path.join(out, "planet.json"), "w") as f:
        json.dump(meta, f, indent=2)
    total = sum(sizes.values())
    print(f"MADE {out}: {levels} levels, {total / 1e6:.0f} MB, {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()
