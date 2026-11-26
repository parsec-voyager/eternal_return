"""A build process (background.py): a Python of its own, Blender's (sys.executable), that works out the builds a worker
thread hands it, so builds run on cores of their own instead of taking turns with the flight for Python's one lock.

Started by background.py's worker threads, one each, as

    python -I build_process.py <host:port> <key> <the add-on's folder> <its package's name>

it connects back, makes the add-on's modules importable by the same names as in Blender (its package without its
__init__, which registers with Blender: no bpy here, and no build needs it), and then serves messages until the
connection closes (Blender quit, or the worker let it go):

- ("define", key, module, class, data): an object every build shares (a surface's Topology), made here once from data
  (its class's from_shared()) and kept by key: builds refer to it by key, not by a copy (background.dumps()).
- ("call", pickled (function, args)): function(*args), answered ("ok", pickled result) or ("error", its traceback).
"""

import importlib
import os
import sys
import traceback
import types
from multiprocessing.connection import Client

# How much lower than Blender's a build process runs (os.nice())
NICE = 10


def main():
    address, key, folder, package = sys.argv[1:5]
    # Below Blender's own threads: where the cores are short, the flight's frame goes first, and the builds take what's
    # left (the efficiency cores too)
    try:
        os.nice(NICE)
    except OSError:
        pass
    host, port = address.rsplit(":", 1)
    # The package's parents (bl_ext.user_default for an installed extension) as empty packages, and the package itself
    # over the add-on's folder, so its modules and what's pickled from them import by their names in Blender
    parts = package.split(".")
    for i in range(1, len(parts)):
        name = ".".join(parts[:i])
        if name not in sys.modules:
            parent = types.ModuleType(name)
            parent.__path__ = []
            sys.modules[name] = parent
    module = types.ModuleType(package)
    module.__path__ = [folder]
    sys.modules[package] = module
    background = importlib.import_module(package + ".background")
    connection = background.sturdy(Client((host, int(port)), authkey=bytes.fromhex(key)))
    while True:
        try:
            message = connection.recv()
        except (EOFError, OSError):
            return
        if message[0] == "define":
            _kind, shared_key, module_name, class_name, data = message
            cls = getattr(importlib.import_module(module_name), class_name)
            background.share(cls.from_shared(data), shared_key)
        elif message[0] == "call":
            try:
                function, args = background.loads(message[1])
                answer = ("ok", background.dumps(function(*args))[0])
            except Exception:
                answer = ("error", traceback.format_exc())
            connection.send(answer)


if __name__ == "__main__":
    main()
