"""flight_sim probe: the substitution maps' giant frames windows built on the main thread, not a worker
(parts.exact_async() off for their lattices), to compare the two. It only changes the flight: fly it with a
measuring probe too, e.g. --probe tools/probes/frames_on_main.py --probe tools/probes/missing_view.py"""
import functools
import importlib


def setup(m):
    package = m.__name__.rpartition(".")[0]
    parts = importlib.import_module(package + ".parts")
    parts_substitution = importlib.import_module(package + ".parts_substitution")
    original = parts.exact_async

    @functools.wraps(original)
    def exact_async(lattice, state):
        return not isinstance(lattice, parts_substitution.SubstitutionLattice) and original(lattice, state)

    parts.exact_async = exact_async
