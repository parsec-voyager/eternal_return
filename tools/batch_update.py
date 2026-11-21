"""Apply one edit to every map: a dry run first, then the saves, then a read-back. The edit is a small Python file; everything around it (finding the maps, the windowed saves, the camera
check, the backups, the reports) is here.

Usage (plain python3, which launches Blender for each map):
    python3 tools/batch_update.py <edit.py>                  dry run: what each map would change, nothing saved
    python3 tools/batch_update.py <edit.py> --save           the real pass, then the read-back
        [--maps LIST] [--root DIR ...] [--profile DIR] [--out DIR] [--jobs N] [--timeout S]

The edit file defines
    def edit(addon):
        ...change bpy.data...
        return ["what it changed", ...]          # empty: nothing to change, the map isn't saved
An entry starting with "note:" is reported but isn't a change (something for you to look at, like a script that
mentions a renamed setting).
and may set MOVES_CAMERA = True when moving a drone camera is the point (otherwise a map whose drone camera moved
is reported and not saved). addon(name) imports the installed add-on's module of that name (addon("hud")). An edit
should be idempotent: the read-back runs it again on the saved maps and expects no changes. tools/batch_edits/ has
examples.

The maps: the .blend files under each --root (default: the Proton Drive blender folder and ~/blender_tests) that
contain a drone (their bytes hold the drone mesh's name, _ER_Drone; a map with a drone mesh no scene flies is opened
too, and the edit finds nothing to do), skipping
Blender profiles (a test profile's extensions and startup files) and .blend1 backups. --maps names them instead, one
path per line; every map listed is opened, drone or not.

Blender runs with the add-on as installed in --profile (a BLENDER_USER_RESOURCES folder, see install_test.sh), or
by default as installed in your own Blender. The dry run and the read-back run headless, --jobs at a time. The saves
run in a windowed Blender, one map at a time, because a headless save turns Rendered viewports to Solid; the windows
open without taking focus, and each map is saved and closed a second after its window is up.

Before each save, the map's .blend1 (the backup Blender is about to replace) is cloned to <out>/backups at its path
under the home folder; on APFS a clone costs no space until one of the two changes. Clones of maps that weren't
saved are removed afterwards. Reports: one JSON per map in <out>/dry, <out>/save and <out>/verify, and a summary
printed at the end. An interrupted --save picks up where it stopped (maps with a save report are skipped); dry runs
and read-backs start over.
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
HOME = os.path.expanduser("~")
DEFAULT_ROOTS = sorted(glob.glob(os.path.join(HOME, "Library/CloudStorage/ProtonDrive-*-folder/blender"))) + [
    os.path.join(HOME, "blender_tests")]
# The drone mesh's name (rig.MESH_NAME)
DRONE_MARKS = (b"_ER_Drone",)
CHUNK = 64 << 20


# ---------------------------------------------------------------------------------------------------------------
# Inside Blender: open, edit, check, save
# ---------------------------------------------------------------------------------------------------------------

def run_in_blender():
    """Run the edit on the open map, write the report and quit (os._exit: a map's timers and threads mustn't keep
    Blender open, or slow its exit, once the report is written)"""
    import importlib
    import importlib.util
    import tomllib
    import traceback
    import bpy

    args = sys.argv[sys.argv.index("--") + 1:]
    edit_path, out, save = args[0], args[1], "--save" in args
    report = {"file": bpy.data.filepath, "changes": [], "errors": [], "saved": False, "cameras_moved": {}}
    try:
        base = next((name for name in sys.modules if name.startswith("bl_ext.") and name.endswith(".eternal_return")),
                    None)
        if base is None:
            raise RuntimeError("the Eternal Return add-on isn't enabled in this Blender profile")
        manifest = os.path.join(os.path.dirname(importlib.import_module(base).__file__), "blender_manifest.toml")
        with open(manifest, "rb") as f:
            report["addon_version"] = tomllib.load(f).get("version")

        def addon(name):
            return importlib.import_module(f"{base}.{name}")

        spec = importlib.util.spec_from_file_location("batch_edit", edit_path)
        edit_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(edit_module)

        before = camera_matrices(bpy)
        returned = list(edit_module.edit(addon) or [])
        report["notes"] = [entry for entry in returned if entry.lower().startswith("note:")]
        report["changes"] = [entry for entry in returned if not entry.lower().startswith("note:")]
        after = camera_matrices(bpy)
        report["cameras_moved"] = {key: max(abs(x - y) for ra, rb in zip(before[key], after[key])
                                            for x, y in zip(ra, rb))
                                   for key in before if key in after}
        moved = [key for key, difference in report["cameras_moved"].items() if difference > 1e-6]
        if moved and not getattr(edit_module, "MOVES_CAMERA", False):
            report["errors"].append(f"drone camera moved: {moved}")
        if save and report["changes"] and not report["errors"]:
            bpy.ops.wm.save_mainfile()
            report["saved"] = True
    except Exception:
        report["errors"].append(traceback.format_exc())
    with open(out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    sys.stdout.flush()
    os._exit(0)


def camera_matrices(bpy):
    """Every scene's drone camera's evaluated world matrix, by "scene/camera" """
    matrices = {}
    for scene in bpy.data.scenes:
        drone_rig = getattr(scene, "drone_rig", None)
        camera = drone_rig.camera if drone_rig else None
        if camera is None:
            continue
        view_layer = scene.view_layers[0]
        view_layer.update()
        matrices[f"{scene.name}/{camera.name}"] = [list(row) for row in
                                                   camera.evaluated_get(view_layer.depsgraph).matrix_world]
    return matrices


# ---------------------------------------------------------------------------------------------------------------
# Outside Blender: the maps, the passes, the summary
# ---------------------------------------------------------------------------------------------------------------

def in_profile(path):
    """Whether path is inside a Blender profile (an installed add-on's files, or a profile's startup/userpref files)"""
    parts = path.split(os.sep)
    return "extensions" in parts or "config" in parts


def has_drone(path):
    """Whether the file's bytes hold a drone mesh's name, read through the compression of a compressed .blend (zstd
    since Blender 3.0, gzip before). One this Python can't read counts as having one."""
    with open(path, "rb") as raw:
        magic = raw.read(4)
    try:
        if magic == b"\x28\xb5\x2f\xfd":
            from compression import zstd  # Python 3.14+
            f = zstd.open(path, "rb")
        elif magic[:2] == b"\x1f\x8b":
            import gzip
            f = gzip.open(path, "rb")
        else:
            f = open(path, "rb")
    except ImportError:
        return True
    with f:
        tail = b""
        while chunk := f.read(CHUNK):
            data = tail + chunk
            if any(mark in data for mark in DRONE_MARKS):
                return True
            tail = data[-32:]
    return False


def find_maps(roots):
    maps = []
    for root in roots:
        for folder, folders, files in os.walk(root):
            folders[:] = sorted(name for name in folders if not in_profile(os.path.join(folder, name)))
            maps += [os.path.join(folder, name) for name in sorted(files) if name.endswith(".blend")]
    return [path for path in maps if has_drone(path)]


def report_path(folder, path):
    return os.path.join(folder, hashlib.md5(path.encode()).hexdigest()[:10] + ".json")


def blender_command(path, edit, out, save, windowed):
    command = [BLENDER] + (["--no-window-focus"] if windowed else ["-b"]) + [
        path, "--python", os.path.abspath(__file__), "--", os.path.abspath(edit), out] + (["--save"] if save else [])
    return command


def run_pass(maps, edit, folder, env, timeout, jobs, save=False, windowed=False, before_each=None):
    """Run the edit on every map, jobs at a time; returns {map: report}. A map whose Blender wrote no report (crash,
    timeout) gets one saying so, with the log's path."""
    os.makedirs(folder, exist_ok=True)
    pending = [path for path in maps if not os.path.isfile(report_path(folder, path))]
    running = []
    while pending or running:
        while pending and len(running) < jobs:
            path = pending.pop(0)
            if before_each:
                before_each(path)
            out = report_path(folder, path)
            log = open(out[:-5] + ".log", "w")
            process = subprocess.Popen(blender_command(path, edit, out, save, windowed), stdout=log,
                                       stderr=subprocess.STDOUT, env=env)
            running.append((path, process, time.monotonic(), log))
        for item in list(running):
            path, process, started, log = item
            if process.poll() is None and time.monotonic() - started < timeout:
                continue
            if process.poll() is None:
                process.kill()
                process.wait()
            log.close()
            running.remove(item)
            out = report_path(folder, path)
            if not os.path.isfile(out):
                with open(out, "w") as f:
                    json.dump({"file": path, "changes": [], "saved": False, "errors": [
                        f"no report (Blender {'timed out' if process.returncode == -9 else 'exited'}), see "
                        f"{out[:-5]}.log"]}, f, indent=1)
            report = json.load(open(out))
            print(f"{time.strftime('%H:%M:%S')} {summary_word(report, save)} {short(path)}", flush=True)
        time.sleep(0.2)
    return {path: json.load(open(report_path(folder, path))) for path in maps}


def summary_word(report, save):
    if report["errors"]:
        return "ERROR  "
    if save:
        return "saved  " if report["saved"] else "same   "
    return "change " if report["changes"] else "same   "


def short(path):
    return "~" + path[len(HOME):] if path.startswith(HOME) else path


def print_summary(title, reports):
    errors = {path: report["errors"] for path, report in reports.items() if report["errors"]}
    changed = {path: report["changes"] for path, report in reports.items() if report["changes"]}
    print(f"\n{title}: {len(reports)} maps, {len(changed)} with changes, {len(errors)} with errors")
    counts = {}
    for changes in changed.values():
        for change in changes:
            counts[change] = counts.get(change, 0) + 1
    for change, count in sorted(counts.items(), key=lambda item: -item[1])[:20]:
        print(f"  {count:4} x {change}")
    for path, report in reports.items():
        for note in dict.fromkeys(report.get("notes", [])):
            print(f"  {short(path)}: {note}")
    for path, messages in errors.items():
        print(f"  ERROR {short(path)}: {messages[0].strip().splitlines()[-1]}")
    return errors


def main():
    parser = argparse.ArgumentParser(prog="batch_update.py", description="Apply one edit to every map")
    parser.add_argument("edit", help="the edit file (defines edit(addon))")
    parser.add_argument("--save", action="store_true", help="save the maps the edit changes, then read them back")
    parser.add_argument("--maps", help="a file listing the maps, one path per line (default: search --root)")
    parser.add_argument("--root", action="append", help="a folder to search for maps (repeatable)")
    parser.add_argument("--profile", help="a BLENDER_USER_RESOURCES folder with the add-on installed")
    parser.add_argument("--out", help="the reports' folder (default ~/blender_batch/<edit name>)")
    parser.add_argument("--jobs", type=int, default=3, help="headless Blenders at a time (dry run, read-back)")
    parser.add_argument("--timeout", type=float, default=900, help="seconds a map may take")
    args = parser.parse_args()

    edit = os.path.abspath(args.edit)
    out = os.path.abspath(args.out or os.path.join(HOME, "blender_batch", os.path.splitext(os.path.basename(edit))[0]))
    env = dict(os.environ)
    if args.profile:
        env["BLENDER_USER_RESOURCES"] = os.path.abspath(args.profile)
    if args.maps:
        maps = [line.strip() for line in open(args.maps) if line.strip()]
        missing = [path for path in maps if not os.path.isfile(path)]
        if missing:
            sys.exit(f"missing maps: {missing}")
    else:
        maps = find_maps(args.root or DEFAULT_ROOTS)
    print(f"{len(maps)} maps, reports in {short(out)}", flush=True)

    if not args.save:
        shutil.rmtree(os.path.join(out, "dry"), ignore_errors=True)
        print_summary("Dry run", run_pass(maps, edit, os.path.join(out, "dry"), env, args.timeout, args.jobs))
        print("\nNothing was saved. Run again with --save to save the maps with changes.")
        return

    backups = os.path.join(out, "backups")

    def back_up(path):
        """Clone the .blend1 the save is about to replace (cp -c: an APFS clone, no space until one changes)"""
        backup = path + "1"
        if os.path.isfile(backup):
            target = os.path.join(backups, os.path.relpath(backup, HOME) if backup.startswith(HOME)
                                  else backup.lstrip(os.sep))
            if not os.path.isfile(target):
                os.makedirs(os.path.dirname(target), exist_ok=True)
                subprocess.run(["cp", "-c", "-p", backup, target], check=True)

    saved = run_pass(maps, edit, os.path.join(out, "save"), env, args.timeout, 1, save=True, windowed=True,
                     before_each=back_up)
    errors = print_summary("Saves", saved)
    # The backups of maps that weren't saved are still their maps' .blend1
    for path, report in saved.items():
        if not report["saved"]:
            target = os.path.join(backups, os.path.relpath(path + "1", HOME) if path.startswith(HOME)
                                  else (path + "1").lstrip(os.sep))
            if os.path.isfile(target):
                os.remove(target)
    print(f"{sum(report['saved'] for report in saved.values())} saved; replaced .blend1 backups in {short(backups)}")

    shutil.rmtree(os.path.join(out, "verify"), ignore_errors=True)
    verify = run_pass([path for path, report in saved.items() if report["saved"]], edit, os.path.join(out, "verify"),
                      env, args.timeout, args.jobs)
    left = print_summary("Read-back (the edit again, on the saved maps: expects no changes)", verify)
    left.update({path: report["changes"] for path, report in verify.items() if report["changes"]})
    print("\nALL CLEAN" if not left and not errors else "\nSEE ABOVE")


if __name__ == "__main__":
    if "--" in sys.argv:
        run_in_blender()
    else:
        main()
